//! High-level analysis driver: parse tree in, per-output-element symbolic
//! expressions (or a race/deadlock/structured-CTA error) out.

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use volta_frontend::ast::{Function, Module, TopLevelItem, VarDecl};

use crate::canon::parent_counts;
use crate::equiv::{DEFAULT_RECYCLE_TERMS, EquivError, EquivSession};
use crate::eval::{AnalysisConfig, AnalysisOutput, EvalError, Interpreter, Stats};
use crate::logging::info;
use crate::lower_error::LowerError;
use crate::lowering::lower_function;
use crate::numeric;
use crate::symbolic::{ExprArena, ExprId};

/// Errors from the end-to-end analysis of one kernel.
#[derive(Debug)]
pub enum AnalysisError {
    KernelNotFound { name: Option<String> },
    Lower(LowerError),
    Eval(EvalError),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelNotFound { name: Some(name) } => {
                write!(f, "no kernel named '{}' in module", name)
            }
            Self::KernelNotFound { name: None } => write!(f, "no kernel entry in module"),
            Self::Lower(e) => write!(f, "lowering failed: {}", e),
            Self::Eval(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for AnalysisError {}

impl From<LowerError> for AnalysisError {
    fn from(e: LowerError) -> Self {
        Self::Lower(e)
    }
}

impl From<EvalError> for AnalysisError {
    fn from(e: EvalError) -> Self {
        Self::Eval(e)
    }
}

/// Find a kernel entry point by name, or the unique entry if `name` is None.
pub fn find_kernel<'m>(
    module: &'m Module,
    name: Option<&str>,
) -> Result<&'m Function, AnalysisError> {
    let mut entries = module.items.iter().filter_map(|item| match item {
        TopLevelItem::Entry(f) => Some(f),
        _ => None,
    });
    match name {
        Some(name) => {
            entries
                .find(|f| f.name.to_string() == name)
                .ok_or(AnalysisError::KernelNotFound {
                    name: Some(name.to_string()),
                })
        }
        None => entries
            .next()
            .ok_or(AnalysisError::KernelNotFound { name: None }),
    }
}

/// Module-level variable declarations (extern shared memory, module globals).
pub fn module_vars(module: &Module) -> Vec<VarDecl> {
    module
        .items
        .iter()
        .filter_map(|item| match item {
            TopLevelItem::Variable(v) => Some(v.clone()),
            _ => None,
        })
        .collect()
}

/// Analyze one kernel: lower it and symbolically execute all threads of
/// CTA (0,0,0) under the given configuration.
pub fn analyze_kernel(
    module: &Module,
    kernel: Option<&str>,
    config: AnalysisConfig,
) -> Result<AnalysisOutput, AnalysisError> {
    let func = find_kernel(module, kernel)?;
    let vars = module_vars(module);
    let program = lower_function(func, &vars)?;
    info!(
        "analyzing kernel {:?}: block={:?} grid={:?}",
        kernel, config.block_dim, config.grid_dim
    );
    let mut interp = Interpreter::new(&program, config)?;
    interp.run()?;
    Ok(interp.into_output()?)
}

/// A single output element where the two kernels disagree.
#[derive(Debug, Clone)]
pub struct Mismatch {
    pub array: String,
    pub index: u64,
}

/// Result of comparing two analysis outputs.
#[derive(Debug)]
pub enum EquivOutcome {
    Equivalent,
    NotEquivalent { mismatches: Vec<Mismatch> },
}

/// Errors from output comparison.
#[derive(Debug)]
pub enum EquivCheckError {
    /// The two outputs have different arrays or element counts.
    ShapeMismatch { message: String },
    /// The underlying symbolic check failed.
    Equiv(EquivError),
    /// The f64 oracle contradicted (or could not confirm) a verdict.
    Numeric { message: String },
    /// A later solve iteration disagreed with iteration 1 on an element's
    /// verdict. The decision procedure is deterministic, so this can only
    /// mean a bug (or memory corruption) - a hard error, never papered over.
    IterationDisagreement { message: String },
}

impl fmt::Display for EquivCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { message } => write!(f, "output shape mismatch: {}", message),
            Self::Equiv(e) => write!(f, "equivalence check failed: {}", e),
            Self::Numeric { message } => write!(f, "numeric oracle: {}", message),
            Self::IterationDisagreement { message } => {
                write!(f, "solve iterations disagree: {}", message)
            }
        }
    }
}

impl std::error::Error for EquivCheckError {}

impl From<EquivError> for EquivCheckError {
    fn from(e: EquivError) -> Self {
        Self::Equiv(e)
    }
}

/// Options for [`check_output_equivalence_with`].
#[derive(Debug, Clone)]
pub struct EquivCheckOptions {
    /// Check at most this many common elements per array (0 = all).
    pub sample: u64,
    /// Confirm every verdict with the f64 numeric oracle.
    pub verify_numeric: bool,
    /// Recycle the VC intern tables past this many interned terms
    /// (0 = never); see `EquivSession::with_recycle_terms`.
    pub recycle_terms: usize,
    /// Run the solve phase this many times (default 1). Every iteration
    /// re-solves the same sampled elements with a *fresh* `EquivSession`,
    /// so each iteration is a cold start (comparable timings) and memory
    /// stays bounded (each session drops before the next). The verdict
    /// comes from iteration 1; later iterations must agree with it, and a
    /// disagreement is [`EquivCheckError::IterationDisagreement`] - a
    /// determinism check for free. `verify_numeric` runs on iteration 1
    /// only. Per-iteration timings land in
    /// [`EquivCheckReport::check_iters`].
    pub iterations: NonZeroUsize,
    /// Solve the element list on this many worker threads (default 1:
    /// one session checks every element, the historical serial loop).
    /// The sampled elements are split into contiguous chunks, one
    /// worker with its own private `EquivSession` per chunk. Per-element
    /// verdicts are independent of session state (the same property the
    /// `iterations` agreement check rests on), so the partition cannot
    /// change them; contiguity keeps row-local shared structure (an
    /// attention row's softmax denominator) inside one session, so only
    /// structure straddling a chunk boundary is re-canonicalized.
    /// `recycle_terms` remains the *aggregate* memory cap: each worker
    /// recycles at `recycle_terms / workers` (at least 1). Each worker
    /// session also gets its own canon term-op budget. Under
    /// parallelism > 1 the summed [`EquivCheckReport::check_iters`]
    /// spans run concurrently - they include cross-worker contention
    /// and exceed wall clock ([`EquivCheckReport::wall_iters`]); keep 1
    /// for timings comparable across backends and to the paper's.
    pub parallelism: NonZeroUsize,
}

impl Default for EquivCheckOptions {
    fn default() -> Self {
        Self {
            sample: 0,
            verify_numeric: false,
            recycle_terms: DEFAULT_RECYCLE_TERMS,
            iterations: NonZeroUsize::MIN,
            parallelism: NonZeroUsize::MIN,
        }
    }
}

/// One checked element's iteration-1 decision-procedure check duration.
#[derive(Debug, Clone)]
pub struct ElementCheckTime {
    pub array: String,
    pub index: u64,
    /// The element's `EquivSession::check` duration in solve iteration 1.
    pub check: Duration,
}

/// The outcome of a comparison plus how much of the footprint it covered.
#[derive(Debug)]
pub struct EquivCheckReport {
    pub outcome: EquivOutcome,
    /// Elements actually compared (less than total when sampling).
    pub elements_checked: u64,
    /// Comparable elements in the reference footprints.
    pub elements_total: u64,
    /// Per-iteration time spent in the decision procedure itself: each
    /// entry is one solve iteration's summed `EquivSession::check` calls,
    /// and nothing else. VC pairing and the optional numeric-oracle
    /// verification are excluded, so these are the numbers to put beside
    /// another backend's solver time (the paper's tables) - they do not
    /// move when `verify_numeric` is toggled. One entry per
    /// `EquivCheckOptions::iterations`, in order; never empty.
    pub check_iters: Vec<Duration>,
    /// Iteration 1's per-element check durations, one entry per checked
    /// element in `sampled_elements` order (so `check_iters[0]` is their
    /// sum). Recorded outside the timed spans - the entries are the same
    /// measurements `check_iters[0]` accumulates, not extra timing work -
    /// so carrying them does not move the iteration totals.
    pub element_checks: Vec<ElementCheckTime>,
    /// Per-iteration wall-clock time of the whole element pass (worker
    /// spawn through join; iteration 1's span includes any oracle
    /// confirmations, which run inside the workers). At
    /// [`EquivCheckOptions::parallelism`] 1 this tracks `check_iters`;
    /// above 1 it is the honest elapsed number, while `check_iters`
    /// stays the summed backend-comparable measure. One entry per
    /// iteration, aligned with `check_iters`.
    pub wall_iters: Vec<Duration>,
    /// Time spent pairing the two footprints (`paired_elements`). Callers
    /// that account "VC generation" as symbolic execution plus pairing
    /// (the bench harness) add this to their execution time.
    pub pair_time: Duration,
    /// Total time in the f64-oracle confirmations (`verify_verdict`,
    /// iteration 1 only - the only iteration the oracle runs on). `Some`
    /// exactly when `EquivCheckOptions::verify_numeric` was set, so
    /// `None` means the oracle was off, not that it was free. Kept out
    /// of `check_iters` so the solve timings do not move when the oracle
    /// is toggled.
    pub verify_time: Option<Duration>,
}

impl EquivCheckReport {
    /// Iteration 1's decision-procedure time - the verdict-producing run,
    /// and the historical meaning of this report's single timing field.
    pub fn check_time(&self) -> Duration {
        self.check_iters[0]
    }
}

/// Pair up the two runs' written elements for each array the caller
/// names: both runs must have written every named array with an
/// identical index set, element for element (arrays the caller does not
/// name are not compared - e.g. auxiliary exports like FlashAttention's
/// softmax `l`/`m` statistics that only the optimized kernel computes).
/// The list must be nonempty: checking nothing is an error, not a
/// vacuous pass. Shared by `check_output_equivalence_with` (the decision
/// procedure) and any other backend (e.g. `volta_z3`) that needs the
/// exact same element correspondence to be a fair comparison.
pub fn paired_elements(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    arrays: &[String],
) -> Result<Vec<(String, Vec<(u64, ExprId, ExprId)>)>, EquivCheckError> {
    if arrays.is_empty() {
        return Err(EquivCheckError::ShapeMismatch {
            message: "no arrays specified to check".to_string(),
        });
    }
    let mut result = Vec::with_capacity(arrays.len());
    for name in arrays {
        let Some((_, ref_elems)) = reference.outputs.iter().find(|(n, _)| n == name) else {
            return Err(EquivCheckError::ShapeMismatch {
                message: format!("reference run has no output array '{}'", name),
            });
        };
        let Some((_, opt_elems)) = optimized.outputs.iter().find(|(n, _)| n == name) else {
            return Err(EquivCheckError::ShapeMismatch {
                message: format!("optimized run has no output array '{}'", name),
            });
        };

        if ref_elems.len() != opt_elems.len() {
            return Err(EquivCheckError::ShapeMismatch {
                message: format!(
                    "array '{}': {} elements written vs {}",
                    name,
                    ref_elems.len(),
                    opt_elems.len()
                ),
            });
        }
        let mut common = Vec::with_capacity(ref_elems.len());
        for (&(ri, r), &(oi, o)) in ref_elems.iter().zip(opt_elems.iter()) {
            if ri != oi {
                return Err(EquivCheckError::ShapeMismatch {
                    message: format!(
                        "array '{}': written footprints differ (element {} vs {})",
                        name, ri, oi
                    ),
                });
            }
            common.push((ri, r, o));
        }
        result.push((name.clone(), common));
    }
    Ok(result)
}

/// Flatten paired footprints into the per-element check list under a
/// sample limit: each array's element prefix, capped at `sample` elements
/// per array (0 = all), arrays in `paired` order. This is the one
/// definition of *which* elements a run checks - the decision procedure
/// (`check_output_equivalence_with`), the Z3 backend
/// (`volta_z3::check_output_equivalence`), and the bench harness's Z3
/// re-solve loop all iterate exactly this list, so their per-element
/// results correspond positionally.
pub fn sampled_elements(
    paired: &[(String, Vec<(u64, ExprId, ExprId)>)],
    sample: u64,
) -> Vec<(&str, u64, ExprId, ExprId)> {
    let mut sampled = Vec::new();
    for (name, common) in paired {
        let limit = match sample {
            0 => common.len(),
            n => common.len().min(n as usize),
        };
        for &(index, r, o) in common.iter().take(limit) {
            sampled.push((name.as_str(), index, r, o));
        }
    }
    sampled
}

/// Verify that a later solve iteration reproduced iteration 1's verdict
/// for one element. `iteration` is 1-based (so always >= 2 here). The
/// decision procedure is deterministic; a disagreement is a hard error.
fn check_iteration_agreement(
    first: bool,
    later: bool,
    array: &str,
    index: u64,
    iteration: usize,
) -> Result<(), EquivCheckError> {
    if first == later {
        return Ok(());
    }
    Err(EquivCheckError::IterationDisagreement {
        message: format!(
            "array '{}' element {}: iteration {} returned {} but iteration 1 returned {}",
            array,
            index,
            iteration,
            if later {
                "equivalent"
            } else {
                "not equivalent"
            },
            if first {
                "equivalent"
            } else {
                "not equivalent"
            },
        ),
    })
}

/// What a solve iteration does per element beyond the decision check
/// itself - iteration 1 produces the verdicts (and confirms them with the
/// oracle when enabled); later iterations must reproduce them.
#[derive(Clone, Copy)]
enum IterationRole<'a> {
    First { verify_numeric: bool },
    Later {
        iteration: usize,
        first_verdicts: &'a [bool],
    },
}

/// One element's product inside a solve iteration.
struct SlotOutcome {
    equivalent: bool,
    /// The `EquivSession::check` duration (what `check_iters` sums).
    check: Duration,
    /// The oracle confirmation's duration (`IterationRole::First` with
    /// `verify_numeric` only).
    verify: Option<Duration>,
}

/// One element's work inside a solve iteration: the timed decision check,
/// then either the oracle confirmation (iteration 1) or the agreement
/// check against iteration 1's verdict (later iterations).
fn check_element(
    session: &mut EquivSession<'_>,
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    role: IterationRole<'_>,
    slot: usize,
    element: (&str, u64, ExprId, ExprId),
) -> Result<SlotOutcome, EquivCheckError> {
    let (array, index, r, o) = element;
    let check_start = Instant::now();
    let equivalent = session.check(r, o)?;
    let check = check_start.elapsed();
    let mut verify = None;
    match role {
        IterationRole::First { verify_numeric } => {
            if verify_numeric {
                let verify_start = Instant::now();
                numeric::verify_verdict(&reference.arena, r, &optimized.arena, o, equivalent)
                    .map_err(|message| EquivCheckError::Numeric {
                        message: format!("array '{}' element {}: {}", array, index, message),
                    })?;
                verify = Some(verify_start.elapsed());
            }
        }
        IterationRole::Later {
            iteration,
            first_verdicts,
        } => {
            check_iteration_agreement(first_verdicts[slot], equivalent, array, index, iteration)?;
        }
    }
    Ok(SlotOutcome {
        equivalent,
        check,
        verify,
    })
}

/// One solve iteration over the sampled elements: contiguous chunks of
/// `chunk_size` elements, one worker thread with its own `EquivSession`
/// per chunk (see `EquivCheckOptions::parallelism`). Returns one slot per
/// element, in element order; a worker stops its chunk at its first
/// error (matching the serial loop, which never checks past an error),
/// leaving that chunk's later slots empty, so scanning the slots in
/// order reaches the lowest-slot error before any empty slot.
fn run_solve_iteration(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    checked: &[(&str, u64, ExprId, ExprId)],
    counts: (&Arc<Vec<u32>>, &Arc<Vec<u32>>),
    worker_recycle: usize,
    chunk_size: usize,
    role: IterationRole<'_>,
) -> Vec<Option<Result<SlotOutcome, EquivCheckError>>> {
    let mut slots: Vec<Option<Result<SlotOutcome, EquivCheckError>>> = Vec::new();
    slots.resize_with(checked.len(), || None);
    std::thread::scope(|scope| {
        for (chunk_index, (elements, outcomes)) in checked
            .chunks(chunk_size)
            .zip(slots.chunks_mut(chunk_size))
            .enumerate()
        {
            let (counts1, counts2) = counts;
            std::thread::Builder::new()
                .name(format!("vc-solve-{}", chunk_index))
                .spawn_scoped(scope, move || {
                    let mut session = EquivSession::with_shared_counts(
                        &reference.arena,
                        &optimized.arena,
                        worker_recycle,
                        Arc::clone(counts1),
                        Arc::clone(counts2),
                    );
                    let base = chunk_index * chunk_size;
                    for (offset, (&element, outcome)) in
                        elements.iter().zip(outcomes.iter_mut()).enumerate()
                    {
                        let result = check_element(
                            &mut session,
                            reference,
                            optimized,
                            role,
                            base + offset,
                            element,
                        );
                        let stop = result.is_err();
                        *outcome = Some(result);
                        if stop {
                            return;
                        }
                    }
                })
                .expect("spawning a VC solver thread");
        }
    });
    slots
}

/// Check two analysis outputs element by element under `options`. Within
/// one solve iteration each worker's `EquivSession` is shared across all
/// its elements (one worker checks everything at the default
/// `parallelism` of 1), so structure shared between elements (and between
/// the two kernels) canonicalizes once per worker; each further iteration
/// (see `EquivCheckOptions::iterations`) re-solves the same sampled
/// elements from fresh sessions.
pub fn check_output_equivalence_with(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    arrays: &[String],
    options: &EquivCheckOptions,
) -> Result<EquivCheckReport, EquivCheckError> {
    let pair_start = Instant::now();
    let paired = paired_elements(reference, optimized, arrays)?;
    let pair_time = pair_start.elapsed();

    // Flatten the sampled elements once: every iteration solves exactly
    // this list, in this order (`sampled_elements` takes each array's
    // prefix, so "the same sampled elements" holds by construction).
    let elements_total: u64 = paired.iter().map(|(_, common)| common.len() as u64).sum();
    let checked = sampled_elements(&paired, options.sample);

    // Parent counts once per call: every worker session in every
    // iteration (and every recycle) shares these two computations.
    let counts1 = Arc::new(parent_counts(&reference.arena));
    let counts2 = Arc::new(parent_counts(&optimized.arena));

    // The partition, fixed across iterations so each one redoes identical
    // work: `workers` contiguous chunks, and the aggregate recycle cap
    // split evenly so total warm memory stays what `recycle_terms` says.
    let workers = options.parallelism.get().min(checked.len().max(1));
    let chunk_size = checked.len().div_ceil(workers).max(1);
    let worker_recycle = match options.recycle_terms {
        0 => 0,
        cap => (cap / workers).max(1),
    };

    let mut verify_time = options.verify_numeric.then_some(Duration::ZERO);
    let mut check_iters = Vec::with_capacity(options.iterations.get());
    let mut wall_iters = Vec::with_capacity(options.iterations.get());
    let mut element_checks = Vec::with_capacity(checked.len());
    let mut first_verdicts: Vec<bool> = Vec::with_capacity(checked.len());
    for iteration in 1..=options.iterations.get() {
        let role = if iteration == 1 {
            IterationRole::First {
                verify_numeric: options.verify_numeric,
            }
        } else {
            IterationRole::Later {
                iteration,
                first_verdicts: &first_verdicts,
            }
        };
        let wall_start = Instant::now();
        let slots = run_solve_iteration(
            reference,
            optimized,
            &checked,
            (&counts1, &counts2),
            worker_recycle,
            chunk_size,
            role,
        );
        wall_iters.push(wall_start.elapsed());

        let mut iter_time = Duration::ZERO;
        for (slot, outcome) in slots.into_iter().enumerate() {
            // A worker stops its chunk at its first error, so in slot
            // order every error precedes its chunk's unfilled slots -
            // reaching an empty slot here is impossible, and `?` on the
            // first error yields the lowest-slot error deterministically.
            let outcome = outcome.expect("worker fills every slot up to its first error");
            let SlotOutcome {
                equivalent,
                check,
                verify,
            } = outcome?;
            iter_time += check;
            if iteration == 1 {
                let (name, index, ..) = checked[slot];
                element_checks.push(ElementCheckTime {
                    array: name.to_string(),
                    index,
                    check,
                });
                if let (Some(total), Some(verify)) = (verify_time.as_mut(), verify) {
                    *total += verify;
                }
                first_verdicts.push(equivalent);
            }
        }
        check_iters.push(iter_time);
    }

    let mismatches: Vec<Mismatch> = checked
        .iter()
        .zip(&first_verdicts)
        .filter(|&(_, &equivalent)| !equivalent)
        .map(|(&(name, index, _, _), _)| Mismatch {
            array: name.to_string(),
            index,
        })
        .collect();
    let outcome = if mismatches.is_empty() {
        EquivOutcome::Equivalent
    } else {
        EquivOutcome::NotEquivalent { mismatches }
    };
    Ok(EquivCheckReport {
        outcome,
        elements_checked: checked.len() as u64,
        elements_total,
        check_iters,
        element_checks,
        wall_iters,
        pair_time,
        verify_time,
    })
}

/// A snapshot of one kernel's verification conditions: the expression arena
/// plus the output footprint (index -> root `ExprId`). This is exactly what
/// `check_output_equivalence_with` needs and nothing else - `Stats`/op-counts
/// describe the symbolic-execution run that produced it, not the VCs
/// themselves, so they aren't part of the snapshot.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct VcSnapshot {
    pub arena: ExprArena,
    pub outputs: Vec<(String, Vec<(u64, ExprId)>)>,
}

impl VcSnapshot {
    pub fn from_output(output: AnalysisOutput) -> Self {
        Self {
            arena: output.arena,
            outputs: output.outputs,
        }
    }

    /// Rehydrate into the shape `check_output_equivalence_with` accepts.
    /// `stats`/`op_counts` are empty: they belong to a symbolic-execution
    /// run, and a dump has none to report.
    pub fn into_analysis_output(self) -> AnalysisOutput {
        AnalysisOutput {
            arena: self.arena,
            outputs: self.outputs,
            stats: Stats::default(),
            op_counts: std::collections::BTreeMap::new(),
        }
    }

    /// Check that every id in the snapshot points inside its arena. Run
    /// this on snapshots rebuilt from external data (a dump file): a
    /// corrupt or version-skewed file that still decodes would otherwise
    /// panic with an index-out-of-bounds deep inside the equivalence check.
    pub fn validate(&self) -> Result<(), String> {
        self.arena.validate()?;
        let n_nodes = self.arena.node_count();
        for (name, elems) in &self.outputs {
            for &(index, root) in elems {
                if id_collections::Id::to_index(root) as usize >= n_nodes {
                    return Err(format!(
                        "output '{}' element {} references expression {} but the arena has {} nodes",
                        name,
                        index,
                        id_collections::Id::to_index(root),
                        n_nodes
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The reference and optimized kernels' verification conditions, as
/// persisted to `.vcdump` files by `volta compare --dump-vcs` and by the
/// bench harness (one per equivalence benchmark), and reloaded by
/// `volta compare --from-dump`. The on-disk format lives in [`vc_dump`] -
/// the one implementation both binaries share.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct VcDump {
    pub reference: VcSnapshot,
    pub optimized: VcSnapshot,
}

impl VcDump {
    /// Validate both snapshots (see `VcSnapshot::validate`).
    pub fn validate(&self) -> Result<(), String> {
        self.reference
            .validate()
            .map_err(|e| format!("reference: {}", e))?;
        self.optimized
            .validate()
            .map_err(|e| format!("optimized: {}", e))
    }
}

pub mod vc_dump {
    //! The `.vcdump` on-disk format: how a [`VcDump`] is persisted and
    //! reloaded. The single implementation behind `volta compare
    //! --dump-vcs`/`--from-dump` and the bench harness's per-benchmark VC
    //! files, so a dump written by either tool loads in both.
    //!
    //! Layout: a 12-byte header (magic + format version) followed by the
    //! bincode (fixint little-endian) payload. Reloading checks the header
    //! before decoding, so a truncated/foreign file fails with a clear
    //! message instead of a bincode parse deep in the stream, and
    //! validates every id in the decoded dump so a corrupt file errors
    //! instead of panicking later.

    use std::io::{self, Write};
    use std::path::Path;

    use bincode::Options;

    use super::VcDump;

    const DUMP_MAGIC: &[u8; 8] = b"VOLTAVCD";
    /// Dump format version, semver-style: bump `DUMP_MINOR` for
    /// backwards-compatible additions (older files stay loadable), bump
    /// `DUMP_MAJOR` and reset minor for breaking changes (e.g. any change
    /// to `ExprNode`'s bincode layout). A reader accepts files with its
    /// own major and a minor no newer than its own.
    // 2.0: removed the never-produced SignExtend/ZeroExtend/Truncate ExprNode
    // variants, which shifts the bincode variant tags of every later variant.
    // The equally never-produced ToInt variant was then removed within the same
    // unreleased format revision; its tag shift rides the same 2.0 (no build
    // that wrote the intermediate layout ever shipped). FloatConst(f64) then
    // became RealConst(Real) - exact rationals on the wire - within the same
    // unreleased revision, riding the same 2.0 for the same reason.
    const DUMP_MAJOR: u16 = 2;
    const DUMP_MINOR: u16 = 0;

    /// Write `dump` to `path`, creating or truncating the file.
    pub fn write_vc_dump(path: &Path, dump: &VcDump) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = io::BufWriter::new(file);
        write_vc_dump_to(&mut writer, dump)?;
        writer.flush()
    }

    /// Serialize `dump` (header + payload) into any writer - the exact
    /// byte stream [`write_vc_dump`] puts on disk. Split out so a caller
    /// that needs the bytes as they are produced (volta-bench's manifest
    /// fingerprint hashes them through a tee writer) can observe them
    /// without buffering a second, possibly GiB-scale, copy in memory.
    /// Does not flush.
    pub fn write_vc_dump_to(writer: &mut impl Write, dump: &VcDump) -> io::Result<()> {
        writer.write_all(DUMP_MAGIC)?;
        writer.write_all(&DUMP_MAJOR.to_le_bytes())?;
        writer.write_all(&DUMP_MINOR.to_le_bytes())?;
        // `serialize_into` uses fixint little-endian; `read_vc_dump` must
        // decode with a matching `with_fixint_encoding()` (bincode's
        // `options()` default is varint, which would not round-trip).
        bincode::serialize_into(writer, dump).map_err(io::Error::other)
    }

    /// Read a dump written by [`write_vc_dump`], checking the header and
    /// validating the decoded contents.
    pub fn read_vc_dump(path: &Path) -> io::Result<VcDump> {
        read_vc_dump_bytes(&std::fs::read(path)?)
    }

    /// Decode a dump from its full file contents (see [`read_vc_dump`]).
    /// Split out so a caller that already holds the raw bytes -
    /// volta-bench's `solve` fingerprints them against its manifest
    /// before decoding anything - can decode the same buffer instead of
    /// reading the file twice.
    pub fn read_vc_dump_bytes(bytes: &[u8]) -> io::Result<VcDump> {
        if bytes.len() < 12 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a volta VC dump (file is shorter than the header)",
            ));
        }
        let (header, payload) = bytes.split_at(12);
        if &header[..8] != DUMP_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a volta VC dump (bad magic)",
            ));
        }
        let major = u16::from_le_bytes([header[8], header[9]]);
        let minor = u16::from_le_bytes([header[10], header[11]]);
        if major != DUMP_MAJOR || minor > DUMP_MINOR {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported dump version {}.{} (this build reads {}.0 through {}.{})",
                    major, minor, DUMP_MAJOR, DUMP_MAJOR, DUMP_MINOR
                ),
            ));
        }

        // Cap decode allocations at the payload size, so a crafted length
        // prefix can't make bincode try to allocate (e.g.) a terabyte
        // before it ever reads that many bytes.
        let dump: VcDump = bincode::options()
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .with_limit(payload.len() as u64)
            .deserialize(payload)
            .map_err(io::Error::other)?;
        dump.validate().map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("dump is corrupt or from an incompatible version: {}", e),
            )
        })?;
        Ok(dump)
    }

    #[cfg(test)]
    mod tests {
        use super::super::VcSnapshot;
        use super::*;
        use crate::symbolic::{ExprArena, ExprId};

        /// A dump written by `write_vc_dump` reloads to an identical
        /// structure via `read_vc_dump` - the fixint round-trip and the
        /// magic header together. The arena deliberately contains the
        /// `Real` constants most at risk on the wire: a fold-produced
        /// non-dyadic rational (1/3 - no f64 denotes it), both infinities,
        /// the smallest positive subnormal, and a -0.0 ingestion (which
        /// normalizes to the rational zero). The round-trip must be
        /// byte-exact - re-serializing the loaded dump reproduces the
        /// original file bit for bit, so any verdict computed from the
        /// reloaded VCs equals the original's.
        #[test]
        fn dump_round_trips() {
            let mut arena = ExprArena::new();
            let x = arena.param_symbol("x");
            let one = arena.int(1);
            let sum = arena.add(x, one);
            let one_f = arena.float_from_f64(1.0).unwrap();
            let three_f = arena.float_from_f64(3.0).unwrap();
            let third = arena.div(one_f, three_f);
            let pos_inf = arena.float_from_f64(f64::INFINITY).unwrap();
            let neg_inf = arena.float_from_f64(f64::NEG_INFINITY).unwrap();
            let subnormal = arena.float_from_f64(f64::from_bits(1)).unwrap();
            let neg_zero = arena.float_from_f64(-0.0).unwrap();
            let specials = [third, pos_inf, neg_inf, subnormal, neg_zero];
            let outputs: Vec<(String, Vec<(u64, ExprId)>)> = vec![(
                "out".to_string(),
                [(0, sum), (1, x)]
                    .into_iter()
                    .chain(specials.iter().enumerate().map(|(i, &e)| (2 + i as u64, e)))
                    .collect(),
            )];
            let dump = VcDump {
                reference: VcSnapshot {
                    arena: arena.clone(),
                    outputs: outputs.clone(),
                },
                optimized: VcSnapshot { arena, outputs },
            };

            let dir = std::env::temp_dir();
            let path = dir.join(format!("volta_vc_dump_test_{}.vcdump", std::process::id()));
            let path2 = dir.join(format!(
                "volta_vc_dump_test_rt_{}.vcdump",
                std::process::id()
            ));
            write_vc_dump(&path, &dump).expect("write_vc_dump");
            let loaded = read_vc_dump(&path).expect("read_vc_dump");
            write_vc_dump(&path2, &loaded).expect("write_vc_dump (reloaded)");
            let original_bytes = std::fs::read(&path).expect("read original dump");
            let rewritten_bytes = std::fs::read(&path2).expect("read rewritten dump");
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(&path2);

            assert_eq!(
                original_bytes, rewritten_bytes,
                "round-trip must be byte-exact"
            );
            assert_eq!(
                loaded.reference.outputs, dump.reference.outputs,
                "reference footprint survived the round-trip"
            );
            assert_eq!(loaded.optimized.outputs, dump.optimized.outputs);
            for &e in &specials {
                assert_eq!(
                    loaded.reference.arena.node(e),
                    dump.reference.arena.node(e),
                    "special constant survived the round-trip"
                );
            }
        }

        /// `read_vc_dump` on a corrupt file yields the given `InvalidData`
        /// error containing `needle` - never `Ok`, never a panic. (`VcDump`
        /// isn't `Debug`, so we can't use `expect_err`.)
        fn assert_load_rejects(bytes: &[u8], tag: &str, needle: &str) {
            let path = std::env::temp_dir().join(format!(
                "volta_vc_dump_{}_{}.bin",
                tag,
                std::process::id()
            ));
            std::fs::write(&path, bytes).unwrap();
            let result = read_vc_dump(&path);
            let _ = std::fs::remove_file(&path);
            match result {
                Ok(_) => panic!("{}: corrupt dump was accepted", tag),
                Err(e) => {
                    assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{}: error kind", tag);
                    assert!(
                        e.to_string().contains(needle),
                        "{}: message {:?} lacks {:?}",
                        tag,
                        e.to_string(),
                        needle
                    );
                }
            }
        }

        /// A file without the magic header is rejected cleanly (not decoded).
        #[test]
        fn read_vc_dump_rejects_bad_magic() {
            assert_load_rejects(
                b"not a volta dump at all, definitely",
                "badmagic",
                "not a volta VC dump",
            );
        }

        /// A file shorter than the 12-byte header is rejected with the header
        /// message rather than a raw EOF.
        #[test]
        fn read_vc_dump_rejects_short_file() {
            assert_load_rejects(b"VOLTA", "short", "shorter than the header");
        }

        /// Semver acceptance: same-major newer-minor files (written by a
        /// future backwards-compatible writer) and other-major files are both
        /// refused by this reader with a version message.
        #[test]
        fn read_vc_dump_rejects_unsupported_versions() {
            let mut newer_minor = Vec::from(*DUMP_MAGIC);
            newer_minor.extend_from_slice(&DUMP_MAJOR.to_le_bytes());
            newer_minor.extend_from_slice(&(DUMP_MINOR + 1).to_le_bytes());
            assert_load_rejects(&newer_minor, "newerminor", "unsupported dump version");

            let mut other_major = Vec::from(*DUMP_MAGIC);
            other_major.extend_from_slice(&(DUMP_MAJOR + 1).to_le_bytes());
            other_major.extend_from_slice(&0u16.to_le_bytes());
            assert_load_rejects(&other_major, "othermajor", "unsupported dump version");
        }
    }
}

/// Write a per-instruction-kind execution profile, most-executed first.
/// The one formatter for `AnalysisOutput::op_counts`, shared by `volta`
/// and `volta-bench` so their profile tables cannot drift.
pub fn write_op_counts(
    out: &mut dyn std::io::Write,
    label: &str,
    counts: &std::collections::BTreeMap<&'static str, u64>,
) -> std::io::Result<()> {
    if counts.is_empty() {
        return Ok(());
    }
    let total: u64 = counts.values().sum();
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    writeln!(out, "{} profile:", label)?;
    for (kind, count) in entries {
        let pct = 100.0 * *count as f64 / total as f64;
        writeln!(out, "  {:<16} {:>10}  ({:>5.1}%)", kind, count, pct)?;
    }
    Ok(())
}

/// Check that two analysis outputs agree on every element of every named
/// array under the default options: all elements checked, no numeric
/// oracle.
pub fn check_output_equivalence(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    arrays: &[String],
) -> Result<EquivOutcome, EquivCheckError> {
    check_output_equivalence_with(reference, optimized, arrays, &EquivCheckOptions::default())
        .map(|report| report.outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Stats;

    fn output_with(arrays: &[(&str, &[u64])]) -> AnalysisOutput {
        let mut arena = ExprArena::new();
        let outputs = arrays
            .iter()
            .map(|(name, indices)| {
                let elems = indices
                    .iter()
                    .map(|&i| {
                        let sid = arena.intern_string(*name);
                        (i, arena.input_element(sid, i))
                    })
                    .collect();
                (name.to_string(), elems)
            })
            .collect();
        AnalysisOutput {
            arena,
            outputs,
            stats: Stats::default(),
            op_counts: std::collections::BTreeMap::new(),
        }
    }

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    /// The caller's array list is the spec: named arrays must exist on
    /// BOTH sides, unnamed arrays on either side are ignored, and naming
    /// nothing is an error rather than a vacuous pass.
    #[test]
    fn paired_elements_follows_the_callers_list() {
        let reference = output_with(&[("out", &[0, 1])]);
        let optimized = output_with(&[("out", &[0, 1]), ("aux", &[0])]);

        // Unnamed optimized-only "aux" is ignored.
        let paired = paired_elements(&reference, &optimized, &names(&["out"])).unwrap();
        assert_eq!(paired.len(), 1);
        assert_eq!(paired[0].1.len(), 2);

        // Naming an array absent from either side is an error.
        assert!(paired_elements(&reference, &optimized, &names(&["aux"])).is_err());
        assert!(paired_elements(&optimized, &reference, &names(&["out", "aux"])).is_err());
        assert!(paired_elements(&reference, &optimized, &names(&["missing"])).is_err());

        // An empty list is an error, not a vacuous pass.
        assert!(paired_elements(&reference, &optimized, &[]).is_err());

        // Differing footprints for a named array are an error.
        let narrower = output_with(&[("out", &[0])]);
        assert!(paired_elements(&reference, &narrower, &names(&["out"])).is_err());
    }

    /// `iterations = N` re-solves the same elements N times: the report
    /// carries one timing per iteration, the verdict is iteration 1's, and
    /// element counts don't multiply.
    #[test]
    fn iterations_time_every_solve_and_keep_one_verdict() {
        let reference = output_with(&[("out", &[0, 1, 2])]);
        let optimized = output_with(&[("out", &[0, 1, 2])]);
        let options = EquivCheckOptions {
            iterations: NonZeroUsize::new(3).unwrap(),
            ..EquivCheckOptions::default()
        };
        let report =
            check_output_equivalence_with(&reference, &optimized, &names(&["out"]), &options)
                .unwrap();
        assert_eq!(report.check_iters.len(), 3);
        assert_eq!(report.wall_iters.len(), 3);
        assert_eq!(report.check_time(), report.check_iters[0]);
        assert_eq!(report.elements_checked, 3);
        assert_eq!(report.elements_total, 3);
        assert!(matches!(report.outcome, EquivOutcome::Equivalent));
        // Oracle off: no oracle-time bucket, rather than a zero one.
        assert_eq!(report.verify_time, None);
        // Per-element times: iteration 1 only (no multiplication across
        // iterations), in `sampled_elements` order, summing to iteration
        // 1's total.
        assert_eq!(
            report
                .element_checks
                .iter()
                .map(|e| (e.array.as_str(), e.index))
                .collect::<Vec<_>>(),
            vec![("out", 0), ("out", 1), ("out", 2)]
        );
        assert_eq!(
            report
                .element_checks
                .iter()
                .map(|e| e.check)
                .sum::<Duration>(),
            report.check_iters[0]
        );

        // Sampling caps the per-array element count once, not per iteration,
        // and a NotEquivalent verdict survives the later iterations'
        // agreement check (same mismatch every time is agreement). The
        // mismatched side writes "out" but computes it from a different
        // input array.
        let mut arena = ExprArena::new();
        let sid = arena.intern_string("other");
        let elems: Vec<(u64, ExprId)> = (0..3).map(|i| (i, arena.input_element(sid, i))).collect();
        let mismatched = AnalysisOutput {
            arena,
            outputs: vec![("out".to_string(), elems)],
            stats: Stats::default(),
            op_counts: std::collections::BTreeMap::new(),
        };
        let options = EquivCheckOptions {
            sample: 2,
            iterations: NonZeroUsize::new(2).unwrap(),
            ..EquivCheckOptions::default()
        };
        let report =
            check_output_equivalence_with(&reference, &mismatched, &names(&["out"]), &options)
                .unwrap();
        assert_eq!(report.elements_checked, 2);
        assert_eq!(report.elements_total, 3);
        assert_eq!(report.check_iters.len(), 2);
        let EquivOutcome::NotEquivalent { mismatches } = report.outcome else {
            panic!("input-element reads of different arrays must differ");
        };
        assert_eq!(mismatches.len(), 2);
    }

    /// With `verify_numeric` on, the oracle's time is reported in its own
    /// bucket (`Some` exactly when the flag is set), never folded into the
    /// solve iterations - including when the oracle runs inside parallel
    /// workers.
    #[test]
    fn verify_numeric_time_is_reported_separately() {
        let reference = output_with(&[("out", &[0, 1])]);
        let optimized = output_with(&[("out", &[0, 1])]);
        for parallelism in [1, 2] {
            let options = EquivCheckOptions {
                verify_numeric: true,
                parallelism: NonZeroUsize::new(parallelism).unwrap(),
                ..EquivCheckOptions::default()
            };
            let report =
                check_output_equivalence_with(&reference, &optimized, &names(&["out"]), &options)
                    .unwrap();
            assert!(matches!(report.outcome, EquivOutcome::Equivalent));
            assert!(report.verify_time.is_some());
        }
    }

    /// Parallel solving is a pure partition of the element list: any
    /// worker count (more chunks than elements included) and even
    /// per-element session recycling produce the serial run's verdicts,
    /// mismatch list, element order, and timing-shape invariants.
    #[test]
    fn parallel_solve_matches_serial() {
        let arrays: &[(&str, &[u64])] = &[("out", &[0, 1, 2, 3, 4, 5]), ("aux", &[0, 1, 2])];
        let bad: &[(&str, u64)] = &[("out", 2), ("out", 5), ("aux", 1)];
        let reference = output_with(arrays);
        let mut arena = ExprArena::new();
        let wrong = arena.intern_string("wrong");
        let outputs = arrays
            .iter()
            .map(|(name, indices)| {
                let good = arena.intern_string(*name);
                let elems = indices
                    .iter()
                    .map(|&i| {
                        let sid = if bad.contains(&(*name, i)) { wrong } else { good };
                        (i, arena.input_element(sid, i))
                    })
                    .collect();
                (name.to_string(), elems)
            })
            .collect();
        let optimized = AnalysisOutput {
            arena,
            outputs,
            stats: Stats::default(),
            op_counts: std::collections::BTreeMap::new(),
        };

        let order: Vec<(&str, u64)> = arrays
            .iter()
            .flat_map(|(name, indices)| indices.iter().map(|&i| (*name, i)))
            .collect();
        for (parallelism, recycle_terms) in [
            (1, DEFAULT_RECYCLE_TERMS),
            (3, DEFAULT_RECYCLE_TERMS),
            // More workers than elements: capped at one element per chunk.
            (64, DEFAULT_RECYCLE_TERMS),
            // Tiny aggregate cap: per-worker recycling on nearly every
            // element must not change verdicts.
            (2, 1),
        ] {
            let options = EquivCheckOptions {
                iterations: NonZeroUsize::new(2).unwrap(),
                parallelism: NonZeroUsize::new(parallelism).unwrap(),
                recycle_terms,
                ..EquivCheckOptions::default()
            };
            let report = check_output_equivalence_with(
                &reference,
                &optimized,
                &names(&["out", "aux"]),
                &options,
            )
            .unwrap();
            assert_eq!(report.elements_checked, 9);
            assert_eq!(report.check_iters.len(), 2);
            assert_eq!(report.wall_iters.len(), 2);
            // Element results stay in `sampled_elements` order no matter
            // how the list was partitioned, and still sum to iteration 1.
            assert_eq!(
                report
                    .element_checks
                    .iter()
                    .map(|e| (e.array.as_str(), e.index))
                    .collect::<Vec<_>>(),
                order,
                "parallelism {}",
                parallelism
            );
            assert_eq!(
                report
                    .element_checks
                    .iter()
                    .map(|e| e.check)
                    .sum::<Duration>(),
                report.check_iters[0]
            );
            let EquivOutcome::NotEquivalent { mismatches } = report.outcome else {
                panic!("parallelism {}: expected the planted mismatches", parallelism);
            };
            assert_eq!(
                mismatches
                    .iter()
                    .map(|m| (m.array.as_str(), m.index))
                    .collect::<Vec<_>>(),
                bad,
                "parallelism {}",
                parallelism
            );
        }
    }

    /// An element error (here an `Undefined` reaching the decision
    /// procedure) surfaces from a parallel run exactly as from the serial
    /// loop, whichever chunk it lands in.
    #[test]
    fn parallel_solve_surfaces_element_errors() {
        let reference = output_with(&[("out", &[0, 1, 2, 3, 4])]);
        let mut arena = ExprArena::new();
        let sid = arena.intern_string("out");
        let elems = (0..5)
            .map(|i| {
                let e = if i == 3 {
                    arena.undefined()
                } else {
                    arena.input_element(sid, i)
                };
                (i, e)
            })
            .collect();
        let broken = AnalysisOutput {
            arena,
            outputs: vec![("out".to_string(), elems)],
            stats: Stats::default(),
            op_counts: std::collections::BTreeMap::new(),
        };
        for parallelism in [1, 4] {
            let options = EquivCheckOptions {
                parallelism: NonZeroUsize::new(parallelism).unwrap(),
                ..EquivCheckOptions::default()
            };
            let err =
                check_output_equivalence_with(&reference, &broken, &names(&["out"]), &options)
                    .unwrap_err();
            assert!(
                matches!(err, EquivCheckError::Equiv(_)),
                "parallelism {}: {}",
                parallelism,
                err
            );
        }
    }

    /// `sampled_elements` is the one definition of which elements get
    /// checked: each array's prefix, capped per array, in `paired` order.
    #[test]
    fn sampled_elements_takes_each_arrays_prefix() {
        let reference = output_with(&[("a", &[0, 1, 2]), ("b", &[7])]);
        let optimized = output_with(&[("a", &[0, 1, 2]), ("b", &[7])]);
        let paired = paired_elements(&reference, &optimized, &names(&["a", "b"])).unwrap();

        let all = sampled_elements(&paired, 0);
        assert_eq!(
            all.iter().map(|&(n, i, _, _)| (n, i)).collect::<Vec<_>>(),
            vec![("a", 0), ("a", 1), ("a", 2), ("b", 7)]
        );

        let capped = sampled_elements(&paired, 2);
        assert_eq!(
            capped
                .iter()
                .map(|&(n, i, _, _)| (n, i))
                .collect::<Vec<_>>(),
            vec![("a", 0), ("a", 1), ("b", 7)]
        );
    }

    /// The determinism guard: agreeing iterations pass, a flipped verdict
    /// is a hard error naming the element and iteration. (The full check
    /// loop can't produce a disagreement - canon is deterministic - so the
    /// guard is exercised directly.)
    #[test]
    fn iteration_agreement_guard() {
        assert!(check_iteration_agreement(true, true, "out", 7, 2).is_ok());
        assert!(check_iteration_agreement(false, false, "out", 7, 3).is_ok());

        let err = check_iteration_agreement(true, false, "out", 7, 2).unwrap_err();
        let EquivCheckError::IterationDisagreement { message } = &err else {
            panic!("disagreement must be IterationDisagreement, got {}", err);
        };
        assert!(message.contains("'out'"), "names the array: {}", message);
        assert!(
            message.contains("element 7"),
            "names the element: {}",
            message
        );
        assert!(
            message.contains("iteration 2"),
            "names the iteration: {}",
            message
        );
    }
}
