//! The one benchmark pipeline: generate VCs (`--iterations` timed runs),
//! persist the last generation's dump, solve with the decision procedure
//! (`--iterations` timed runs), optionally solve with Z3 (`--z3`, same
//! iteration scheme), and record everything.
//!
//! The pipeline's two halves are also runnable separately: the
//! `generate` subcommand ([`BenchmarkRunner::run_generate`]) stops after
//! the dump write (and records the dump in the vcs manifest -
//! `crate::manifest`), and the `solve` subcommand (`crate::solve`)
//! replays the solve phases from the dumps. Both halves are the *same
//! functions* the one-shot pipeline calls (`generate_inner`,
//! `check_equivalence`, `run_z3_phase`), so phase-decoupled runs measure
//! and decide exactly what one-shot runs do.
//!
//! Phase timing:
//!
//! - **VC generation** re-runs everything it takes to produce the
//!   verification conditions from the parsed modules - lowering, both
//!   symbolic executions, and footprint pairing - once per iteration.
//!   Each kernel file is read and parsed once per benchmark, *outside*
//!   the timed loop: file I/O and parsing are not VC generation. The
//!   last iteration's outputs feed the dump file and both solve phases
//!   (earlier ones are dropped before the next starts, so peak memory is
//!   one generation); every later iteration is fingerprint-checked
//!   against iteration 1 (same outcome kind, same per-array footprints,
//!   and same expression identities: arena node count plus per-element
//!   `ExprId`s), so a nondeterministic interpreter regression fails
//!   loudly instead of silently timing different work.
//! - **Decision solve** and the optional **Z3 solve** re-solve the same
//!   sampled elements per iteration (see
//!   `EquivCheckOptions::iterations` and `crate::z3_phase`).
//!
//! Race-check benchmarks have only the generation phase (their whole
//! analysis is the symbolic execution); both solve phases and the dump
//! are skipped, and the Z3 section stays empty even under `--z3`.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use volta_analysis::driver::{
    AnalysisError, ElementCheckTime, EquivCheckOptions, EquivOutcome, VcDump, VcSnapshot,
    analyze_kernel, check_output_equivalence_with, paired_elements, sampled_elements,
    vc_dump::write_vc_dump_to,
};
use volta_analysis::eval::{AnalysisOutput, EvalError};
use volta_analysis::symbolic::ExprId;
use volta_frontend::ascii::AsAscii;
use volta_frontend::ast::Module;
use volta_frontend::parse::Parser;

use crate::config::{BenchmarkCategory, BenchmarkDef, ExpectedOutcome, KernelRun};
use crate::results::{cv, median, vc_dump_path};
use crate::z3_phase::{Z3Options, Z3PhaseOutcome, run_z3_phase};

/// A phase's per-iteration coefficient of variation above this prints a
/// noisy-timing warning (see [`warn_noisy_phases`]).
pub const NOISY_CV_THRESHOLD: f64 = 0.10;

/// Statistics collected from a benchmark run
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStats {
    /// VC-generation wall time per iteration, seconds: each entry is one
    /// full generation from the parsed modules - lowering plus symbolic
    /// execution for both kernels (just the reference for race-check
    /// benchmarks) plus footprint pairing (nothing to pair for race-check
    /// and rejected benchmarks). File reading and parsing happen once per
    /// benchmark, outside the timed loop; writing the VC dump file is
    /// excluded too (tracked in `dump_write_secs`). Empty only for
    /// infrastructure failures.
    pub vc_gen_iters_secs: Vec<f64>,
    /// Time writing the `.vcdump` file (once, from the last generation);
    /// `None` when no dump was written.
    pub dump_write_secs: Option<f64>,
    /// Decision-procedure solve time per iteration, seconds: each entry
    /// is one solve iteration's summed canon equivalence checks only
    /// (`EquivCheckReport::check_iters`) - excludes VC pairing and the
    /// optional `--verify-numeric` oracle, so the solve columns report
    /// the same quantity whether or not verification aids are switched
    /// on. Under `--parallel` N > 1 the summed spans run concurrently
    /// (they include cross-worker contention and exceed wall clock -
    /// see `solve_wall_iters_secs`). Empty for race-check benchmarks
    /// and failures.
    pub solve_iters_secs: Vec<f64>,
    /// Wall-clock time of each decision-solve iteration's element pass
    /// (`EquivCheckReport::wall_iters`). Tracks `solve_iters_secs` at
    /// `--parallel` 1; under parallelism it is the honest elapsed
    /// number while `solve_iters_secs` stays the summed
    /// backend-comparable measure. Empty when no solve ran.
    pub solve_wall_iters_secs: Vec<f64>,
    /// Time in the `--verify-numeric` f64-oracle confirmations (they run
    /// on solve iteration 1 only); `Some` exactly when the flag was on.
    /// Excluded from `solve_iters_secs` - see
    /// `EquivCheckReport::verify_time`.
    pub verify_numeric_secs: Option<f64>,
    /// Iteration 1's per-element decision-procedure check durations, in
    /// `driver::sampled_elements` order (summing to
    /// `solve_iters_secs[0]`); empty when no solve ran.
    pub decision_elements: Vec<ElementCheckTime>,
    /// Time reading the `.vcdump` file (`solve` subcommand only; `None`
    /// on runs that generated their VCs). Excluded from the solve spans:
    /// loading is transport, not solving.
    pub dump_load_secs: Option<f64>,
    /// bar.sync executions across all threads (optimized kernel if present)
    pub block_syncs: u64,
    /// Warp-level sync executions across all threads
    pub warp_syncs: u64,
    /// Instructions executed (both kernels)
    pub instructions: u64,
    /// Output elements compared
    pub elements_checked: u64,
    /// Output elements in the footprint (>= elements_checked when sampling)
    pub elements_total: u64,
    /// Reference kernel's instructions executed, broken down by kind.
    pub reference_op_counts: std::collections::BTreeMap<&'static str, u64>,
    /// Optimized kernel's instructions executed, broken down by kind.
    /// Empty for race-only benchmarks (no optimized kernel).
    pub optimized_op_counts: std::collections::BTreeMap<&'static str, u64>,
}

impl BenchmarkStats {
    /// Median VC-generation time (the table's "Gen (s)" column); 0 when
    /// nothing ran.
    pub fn vc_gen_median_secs(&self) -> f64 {
        median(&self.vc_gen_iters_secs).unwrap_or(0.0)
    }

    /// Median decision-solve time (the table's "Solve (s)" column); 0
    /// when no solve ran.
    pub fn solve_median_secs(&self) -> f64 {
        median(&self.solve_iters_secs).unwrap_or(0.0)
    }
}

/// Actual outcome of running a benchmark
#[derive(Debug, Clone)]
pub enum ActualOutcome {
    Equivalent,
    NotEquivalent {
        mismatches: usize,
        first: String,
    },
    /// The analysis rejected the kernel (data race, deadlock, or another
    /// soundness error); `is_race` distinguishes true races.
    Rejected {
        description: String,
        is_race: bool,
    },
    RaceFree,
    /// The `generate` subcommand's outcome for an equivalence benchmark:
    /// its VCs were generated and dumped without being solved, so no
    /// equivalence claim exists yet (that is `solve`'s job).
    VcsGenerated,
    /// The `solve` subcommand ran with `--backend z3`: only the Z3 phase
    /// ran, so there is no decision-procedure verdict and the Z3
    /// per-element outcomes are the run's data. `refutations` counts the
    /// `not_equivalent` verdicts across the plain run *and* the
    /// `+exp-axiom` sub-run (folded in by [`BenchmarkRunner::
    /// assemble_result`] from the phase counts): with no decision verdict
    /// to rule the row, an affirmative Z3 refutation is the one verdict
    /// that contradicts the expectation and must fail the benchmark
    /// (status `Z3 DIFF`). `unknown`/`timeout`/`unsupported` remain
    /// non-failing data, as in every mode.
    Z3Only {
        refutations: usize,
    },
    Error {
        message: String,
    },
}

impl ActualOutcome {
    pub fn matches(&self, expected: ExpectedOutcome) -> bool {
        match (self, expected) {
            (Self::Equivalent, ExpectedOutcome::Equivalent) => true,
            (Self::RaceFree, ExpectedOutcome::RaceFree) => true,
            (Self::Rejected { is_race, .. }, ExpectedOutcome::DataRace) => *is_race,
            // Phase-limited runs make no equivalence claim, so an
            // equivalence expectation has nothing to contradict: the
            // phase that ran completed, which is all it promised.
            // (Race-check benchmarks never produce these outcomes:
            // `generate` runs them to their real verdicts and `solve`
            // skips them.)
            (Self::VcsGenerated, ExpectedOutcome::Equivalent) => true,
            // ... with one exception: in a z3-only solve nothing else
            // rules the row, so a Z3 `not_equivalent` verdict (plain or
            // +exp-axiom) is an affirmative refutation of the expected
            // equivalence and fails the benchmark. It could in principle
            // be spurious - volta_z3's translation documents one known
            // semantic divergence, SMT's underspecified division at zero
            // (`x/x = 1` is falsifiable there but not in canon's field
            // model) - which is precisely why it must be surfaced for
            // inspection rather than swallowed. Non-verdicts
            // (unknown/timeout/unsupported/error elements) stay
            // non-failing data, faithful to the paper's Table 8 rows.
            (Self::Z3Only { refutations }, ExpectedOutcome::Equivalent) => *refutations == 0,
            _ => false,
        }
    }

    pub fn status(&self) -> &'static str {
        match self {
            Self::Equivalent => "EQUIV",
            Self::NotEquivalent { .. } => "DIFF",
            Self::Rejected { is_race: true, .. } => "RACE",
            Self::Rejected { is_race: false, .. } => "REJECT",
            Self::RaceFree => "OK",
            Self::VcsGenerated => "GEN",
            Self::Z3Only { refutations: 0 } => "Z3",
            Self::Z3Only { .. } => "Z3 DIFF",
            Self::Error { .. } => "ERR",
        }
    }
}

/// Result of running a single benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub category: BenchmarkCategory,
    pub elapsed_secs: f64,
    pub outcome: ActualOutcome,
    pub stats: BenchmarkStats,
    /// The outcome matched the benchmark's expectation. When the
    /// decision procedure ran, the Z3 phase plays no part in this; in a
    /// z3-only solve the Z3 `not_equivalent` counts *are* part of the
    /// outcome ([`ActualOutcome::Z3Only`]'s refutations), because
    /// nothing else rules the row there.
    pub outcome_matched: bool,
    /// `outcome_matched` and, when a Z3 phase was requested, it ran to
    /// completion. Z3 *non-verdicts* (unknown/timeout/unsupported) never
    /// affect this - they are the comparison's data, not failures; an
    /// affirmative Z3 refutation fails a z3-only row via
    /// `outcome_matched` (see `ActualOutcome::matches`).
    pub passed: bool,
    /// Where this benchmark's VC dump was written (equivalence benchmarks
    /// under a configured `vcs_dir` only).
    pub dump_path: Option<PathBuf>,
    /// The Z3 phase's results: `None` when `--z3` was off or the
    /// benchmark has no solve phase (race checks, rejections, failures).
    pub z3: Option<Z3PhaseOutcome>,
}

/// Benchmark runner configuration
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Base directory for kernel files
    pub kernels_dir: PathBuf,
    pub verbose: bool,
    /// Check at most this many output elements per array (0 = all).
    pub sample: u64,
    /// Confirm every verdict with the f64 numeric oracle.
    pub verify_numeric: bool,
    /// Recycle the VC intern tables past this many terms (0 = never).
    pub recycle_terms: usize,
    /// How many times each timed phase runs (VC generation, decision
    /// solve, and the Z3 solve when enabled); tables report medians.
    pub iterations: NonZeroUsize,
    /// Worker threads for the decision solve's element loop
    /// (`EquivCheckOptions::parallelism`; `--recycle-terms` stays the
    /// aggregate cap across workers). Keep 1 for paper-comparable
    /// timings.
    pub parallelism: NonZeroUsize,
    /// Write each equivalence benchmark's VC dump under this directory
    /// (`None` = don't persist VCs).
    pub vcs_dir: Option<PathBuf>,
    /// Run the Z3 solve phase (`--z3`); `None` = decision procedure only.
    pub z3: Option<Z3Options>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            kernels_dir: PathBuf::from(crate::KERNELS_DIR),
            verbose: false,
            sample: 0,
            verify_numeric: false,
            recycle_terms: volta_analysis::equiv::DEFAULT_RECYCLE_TERMS,
            iterations: NonZeroUsize::MIN,
            parallelism: NonZeroUsize::MIN,
            vcs_dir: None,
            z3: None,
        }
    }
}

/// An infrastructure failure inside a per-benchmark run
/// ([`BenchmarkRunner::run_inner`], the `generate` and `solve` inners),
/// carrying the path of any VC dump that exists on disk despite the
/// failure - written before it (a one-shot solve error, a `generate`
/// manifest-write failure) or found and then rejected by it (`solve`'s
/// manifest disagreement or fingerprint mismatch). The benchmark's
/// record should keep pointing at the file in all these cases.
pub(crate) struct RunFailure {
    pub(crate) error: anyhow::Error,
    pub(crate) dump_path: Option<PathBuf>,
}

impl From<anyhow::Error> for RunFailure {
    /// For `?` on failures that occur before any dump is written: no
    /// path to preserve. Post-dump failure sites attach the path
    /// explicitly instead of using this.
    fn from(error: anyhow::Error) -> Self {
        Self {
            error,
            dump_path: None,
        }
    }
}

impl RunFailure {
    /// The failure as a finished [`RunOutput`]: an `Error` outcome with
    /// empty stats. A failure after the dump was written (e.g. the
    /// equivalence check erroring) still leaves the dump on disk; the
    /// record keeps pointing at it.
    pub(crate) fn into_output(self) -> RunOutput {
        RunOutput {
            outcome: ActualOutcome::Error {
                message: format!("{:#}", self.error),
            },
            stats: BenchmarkStats::default(),
            dump_path: self.dump_path,
            z3: None,
        }
    }
}

/// Everything a successful (in the infrastructure sense) run produces.
/// `pub(crate)` so the `solve` module (crate::solve) can hand its phase
/// products to the same [`BenchmarkRunner::assemble_result`] tail the
/// one-shot pipeline and `generate` use.
pub(crate) struct RunOutput {
    pub(crate) outcome: ActualOutcome,
    pub(crate) stats: BenchmarkStats,
    pub(crate) dump_path: Option<PathBuf>,
    pub(crate) z3: Option<Z3PhaseOutcome>,
}

/// One equivalence benchmark's paired footprints: per output array, the
/// common `(index, reference expr, optimized expr)` element list
/// (`driver::paired_elements`' shape).
type PairedFootprints = Vec<(String, Vec<(u64, ExprId, ExprId)>)>;

/// The generation half of the pipeline, as produced by
/// [`BenchmarkRunner::generate_inner`].
enum GeneratedRun {
    /// No VCs to solve: a rejection or a race-check benchmark's race-free
    /// completion. The outcome is already final.
    Done(RunOutput),
    /// An equivalence benchmark's VCs, generated (and dumped when a
    /// `vcs_dir` is configured) and ready for the solve phases.
    Vcs(GeneratedVcs),
}

/// An equivalence benchmark's generation product: both kernels' outputs,
/// the paired footprints, and what (if anything) got persisted. `stats`
/// carries the generation-phase numbers (gen iterations, dump write time,
/// execution counters, footprint size); the solve phases add theirs.
struct GeneratedVcs {
    stats: BenchmarkStats,
    reference: AnalysisOutput,
    optimized: AnalysisOutput,
    paired: PairedFootprints,
    /// What became of the VC dump. The one-shot pipeline treats a failed
    /// write as a warning (the verdict is still computable); the
    /// `generate` subcommand treats it as a failure (the dump is its
    /// product).
    dump: DumpPersistence,
}

/// What persisting one benchmark's VCs produced (see [`persist_vcs`]).
enum DumpPersistence {
    /// No `vcs_dir` configured: nothing was (or should have been)
    /// written.
    NotConfigured,
    /// The dump is on disk.
    Written {
        path: PathBuf,
        /// Wall time of the write (the stats' `dump_write_secs`).
        write_secs: f64,
        /// FNV-1a hash of the exact bytes written - what `generate`
        /// records as the manifest's `vc_fingerprint` (`crate::manifest`).
        fingerprint: u64,
    },
    /// The write was attempted and failed (already warned on stderr);
    /// nothing usable is on disk.
    Failed(String),
}

impl DumpPersistence {
    fn path(&self) -> Option<&Path> {
        match self {
            Self::Written { path, .. } => Some(path),
            Self::NotConfigured | Self::Failed(_) => None,
        }
    }

    fn write_secs(&self) -> Option<f64> {
        match self {
            Self::Written { write_secs, .. } => Some(*write_secs),
            Self::NotConfigured | Self::Failed(_) => None,
        }
    }
}

/// One VC-generation iteration's product.
enum Generated {
    /// The analysis rejected a kernel (data race, deadlock, another
    /// soundness error) - the expected outcome for racy benchmarks.
    Rejected { outcome: ActualOutcome },
    /// A race-check benchmark ran to completion: race-free.
    RaceFree { reference: AnalysisOutput },
    /// An equivalence benchmark's full VCs: both outputs plus the paired
    /// footprints along the reference config's declared output arrays.
    Equivalence {
        reference: AnalysisOutput,
        optimized: AnalysisOutput,
        paired: PairedFootprints,
    },
}

/// One benchmark's kernel file(s), read and parsed once per run (see
/// [`BenchmarkRunner::load_benchmark`]): every generation iteration
/// re-analyzes the same parsed modules. Each module is paired with its
/// `KernelRun` so [`generate`] cannot mix a module up with the wrong
/// launch config.
struct LoadedBenchmark<'d> {
    reference: (&'d KernelRun, Module),
    optimized: Option<(&'d KernelRun, Module)>,
}

/// One VC-generation iteration over the already-parsed modules: lower
/// and run the kernel(s), then pair the footprints. `Err` is an
/// infrastructure failure (lowering, footprint pairing); an analysis
/// *rejection* (race, deadlock, ...) is a `Generated::Rejected` outcome,
/// expected for racy benchmarks.
fn generate(loaded: &LoadedBenchmark) -> Result<Generated> {
    let (reference_run, reference_module) = &loaded.reference;
    let reference = match analyze(reference_module, reference_run)? {
        Ok(output) => output,
        Err(e) => {
            return Ok(Generated::Rejected {
                outcome: rejected_outcome(e),
            });
        }
    };
    let Some((optimized_run, optimized_module)) = &loaded.optimized else {
        // Race-check benchmark: reaching the end means no race.
        return Ok(Generated::RaceFree { reference });
    };
    let optimized = match analyze(optimized_module, optimized_run)? {
        Ok(output) => output,
        Err(e) => {
            return Ok(Generated::Rejected {
                outcome: rejected_outcome(e),
            });
        }
    };
    // Pair along the reference config's declared output arrays - the
    // tail of VC generation, shared by both solve backends.
    let arrays = reference_run.config.output_array_names();
    let paired = paired_elements(&reference, &optimized, &arrays)
        .map_err(|e| anyhow!("pairing footprints: {}", e))?;
    Ok(Generated::Equivalence {
        reference,
        optimized,
        paired,
    })
}

/// Lower and run one kernel from its parsed module, splitting the two
/// failure modes the runner cares about: the outer error is an
/// infrastructure failure (lowering); the inner error is an analysis
/// rejection (race, deadlock, structured-CTA violation, ...), which for
/// a race-check benchmark is itself the expected outcome.
fn analyze(module: &Module, run: &KernelRun) -> Result<Result<AnalysisOutput, EvalError>> {
    match analyze_kernel(module, Some(&run.kernel), run.config.clone()) {
        Ok(output) => Ok(Ok(output)),
        Err(AnalysisError::Eval(e)) => Ok(Err(e)),
        Err(e) => Err(anyhow!("{}: {}", run.path, e)),
    }
}

/// The cheap fingerprint of one generation, kept across iterations
/// (without the arenas) to check that every iteration generated the same
/// thing - footprints *and* expression identities.
///
/// For rejections only the verdict-bearing part is compared - the status
/// (RACE vs REJECT), not the diagnostic text: diagnostic text may embed
/// schedule-dependent details, so verdict kinds are the contract; a
/// rejection *kind* flip would change the benchmark verdict and must
/// fail loudly.
#[derive(PartialEq)]
enum GenShape {
    Rejected {
        status: &'static str,
    },
    RaceFree {
        reference: KernelFingerprint,
    },
    Equivalence {
        reference: KernelFingerprint,
        optimized: KernelFingerprint,
    },
}

/// One kernel's generation fingerprint: the arena's node count plus the
/// full per-array `(index, ExprId)` output lists. Each generation builds
/// a fresh arena deterministically, so identical construction order is
/// equivalent to identical ids - `ExprId` equality across independent
/// arenas is a strong expression-identity check that costs nothing (no
/// rendering, no arena retained).
#[derive(PartialEq)]
struct KernelFingerprint {
    node_count: usize,
    outputs: Vec<(String, Vec<(u64, ExprId)>)>,
}

fn kernel_fingerprint(output: &AnalysisOutput) -> KernelFingerprint {
    KernelFingerprint {
        node_count: output.arena.node_count(),
        outputs: output.outputs.clone(),
    }
}

impl Generated {
    fn shape(&self) -> GenShape {
        match self {
            Self::Rejected { outcome } => GenShape::Rejected {
                status: outcome.status(),
            },
            Self::RaceFree { reference } => GenShape::RaceFree {
                reference: kernel_fingerprint(reference),
            },
            Self::Equivalence {
                reference,
                optimized,
                ..
            } => GenShape::Equivalence {
                reference: kernel_fingerprint(reference),
                optimized: kernel_fingerprint(optimized),
            },
        }
    }
}

impl GenShape {
    fn kind(&self) -> &'static str {
        match self {
            Self::Rejected { .. } => "a rejection",
            Self::RaceFree { .. } => "a race-free completion",
            Self::Equivalence { .. } => "equivalence footprints",
        }
    }
}

/// First difference between one kernel's fingerprints across two
/// generation iterations, as a message fragment; `None` when identical.
fn fingerprint_mismatch(
    kernel: &str,
    a: &KernelFingerprint,
    b: &KernelFingerprint,
) -> Option<String> {
    if a == b {
        return None;
    }
    if a.node_count != b.node_count {
        return Some(format!(
            "{} kernel: built {} vs {} arena nodes",
            kernel, a.node_count, b.node_count
        ));
    }
    for ((an, ae), (bn, be)) in a.outputs.iter().zip(&b.outputs) {
        if an != bn {
            return Some(format!(
                "{} kernel: output array '{}' vs '{}'",
                kernel, an, bn
            ));
        }
        if ae.len() != be.len() {
            return Some(format!(
                "{} kernel: array '{}' wrote {} vs {} elements",
                kernel,
                an,
                ae.len(),
                be.len()
            ));
        }
        for (&(ai, a_expr), &(bi, b_expr)) in ae.iter().zip(be) {
            if ai != bi {
                return Some(format!(
                    "{} kernel: array '{}' wrote element {} vs {}",
                    kernel, an, ai, bi
                ));
            }
            if a_expr != b_expr {
                return Some(format!(
                    "{} kernel: array '{}' element {} built expression {:?} vs {:?}",
                    kernel, an, ai, a_expr, b_expr
                ));
            }
        }
    }
    Some(format!(
        "{} kernel: {} vs {} output arrays",
        kernel,
        a.outputs.len(),
        b.outputs.len()
    ))
}

/// How a later generation iteration's fingerprint disagrees with
/// iteration 1's; `None` when they agree. The interpreter is
/// deterministic, so any disagreement is a bug to fail loudly on, not to
/// time quietly.
fn gen_shape_mismatch(first: &GenShape, later: &GenShape) -> Option<String> {
    match (first, later) {
        (GenShape::Rejected { status: a }, GenShape::Rejected { status: b }) => {
            (a != b).then(|| format!("iteration 1 rejected as {}, this one as {}", a, b))
        }
        (GenShape::RaceFree { reference: a }, GenShape::RaceFree { reference: b }) => {
            fingerprint_mismatch("the", a, b)
        }
        (
            GenShape::Equivalence {
                reference: r1,
                optimized: o1,
            },
            GenShape::Equivalence {
                reference: r2,
                optimized: o2,
            },
        ) => fingerprint_mismatch("reference", r1, r2)
            .or_else(|| fingerprint_mismatch("optimized", o1, o2)),
        _ => Some(format!(
            "iteration 1 produced {}, this one {}",
            first.kind(),
            later.kind()
        )),
    }
}

pub struct BenchmarkRunner {
    /// `pub(crate)` for the `solve` half of the runner (crate::solve).
    pub(crate) config: RunnerConfig,
}

impl BenchmarkRunner {
    pub fn new(config: RunnerConfig) -> Self {
        Self { config }
    }

    pub fn run(&self, def: &BenchmarkDef) -> BenchmarkResult {
        self.note_raceless_dump(def);
        let start = Instant::now();
        let output = self
            .run_inner(def)
            .unwrap_or_else(|failure| failure.into_output());
        self.assemble_result(def, start, output)
    }

    /// The `generate` subcommand's per-benchmark run: the generation half
    /// of the pipeline only. Race-check benchmarks run to their real
    /// verdicts (their whole analysis is the symbolic execution);
    /// equivalence benchmarks stop at [`ActualOutcome::VcsGenerated`]
    /// once their dump and manifest entry are written - here (unlike the
    /// one-shot pipeline, where the verdict is still computable) a failed
    /// dump or manifest write fails the benchmark, because the dump *is*
    /// the product `solve` consumes.
    pub fn run_generate(&self, def: &BenchmarkDef) -> BenchmarkResult {
        self.note_raceless_dump(def);
        let start = Instant::now();
        let output = self
            .generate_only_inner(def)
            .unwrap_or_else(|failure| failure.into_output());
        self.assemble_result(def, start, output)
    }

    /// `generate` for one benchmark, up to the manifest update; `Err` is
    /// an infrastructure failure. A failure that leaves no valid fresh
    /// dump also removes any previous run's dump and manifest entry
    /// ([`remove_stale_dump`](Self::remove_stale_dump)); a failure *with*
    /// a fresh dump on disk (the manifest update failing after a
    /// successful write) keeps the dump and points the record at it.
    fn generate_only_inner(&self, def: &BenchmarkDef) -> Result<RunOutput, RunFailure> {
        let Some(vcs_dir) = self.config.vcs_dir.as_deref() else {
            // `generate` without a dump directory would produce nothing;
            // the CLI always configures one.
            return Err(anyhow!("generate requires a VC dump directory").into());
        };
        // Pre-flight: fail before the (possibly hours-long) generation
        // loop if the manifest is unreadable. The read feeding the
        // read-modify-write happens again just before the write below,
        // so the update is against a manifest microseconds old, not
        // hours. Race-check benchmarks write no dump and skip the
        // manifest entirely.
        if def.optimized.is_some()
            && let Err(e) = crate::manifest::read_or_new(vcs_dir)
        {
            return Err(self.fail_generate(def, e));
        }
        let vcs = match self
            .generate_inner(def)
            .map_err(|f| self.fail_generate(def, f.error))?
        {
            GeneratedRun::Done(output) => return Ok(output),
            GeneratedRun::Vcs(vcs) => vcs,
        };
        let (dump_path, fingerprint) = match vcs.dump {
            DumpPersistence::Written {
                ref path,
                fingerprint,
                ..
            } => (path.clone(), fingerprint),
            DumpPersistence::Failed(ref e) => {
                return Err(self.fail_generate(def, anyhow!("writing the VC dump failed: {}", e)));
            }
            DumpPersistence::NotConfigured => {
                unreachable!("vcs_dir is checked at the top of generate_only_inner")
            }
        };
        // The read-modify-write. From here on the fresh dump is on disk
        // and valid, so failures carry its path (and nothing is removed):
        // a later `solve` against a stale manifest entry fails loudly on
        // the fingerprint check.
        let with_dump = |error: anyhow::Error| RunFailure {
            error,
            dump_path: Some(dump_path.clone()),
        };
        let mut manifest = crate::manifest::read_or_new(vcs_dir)
            .context("VC dump written, but re-reading the manifest failed")
            .map_err(with_dump)?;
        crate::manifest::record_dump(
            &mut manifest,
            &def.name,
            fingerprint,
            &vcs.reference.outputs,
            &vcs.optimized.outputs,
        );
        crate::manifest::write_manifest(vcs_dir, &manifest)
            .context("VC dump written, but updating the manifest failed")
            .map_err(with_dump)?;
        Ok(RunOutput {
            outcome: ActualOutcome::VcsGenerated,
            stats: vcs.stats,
            dump_path: Some(dump_path),
            z3: None,
        })
    }

    /// Wrap a `generate` failure that leaves no valid fresh dump on
    /// disk, first removing whatever a *previous* run left there
    /// (`remove_stale_dump`): the user asked to regenerate and the
    /// regeneration failed, so an old dump surviving would let a later
    /// `solve` silently solve pre-failure VCs.
    fn fail_generate(&self, def: &BenchmarkDef, error: anyhow::Error) -> RunFailure {
        self.remove_stale_dump(def);
        RunFailure {
            error,
            dump_path: None,
        }
    }

    /// Best-effort removal of a benchmark's dump file and manifest entry
    /// after a failed regeneration, so a later `solve` hits the loud
    /// missing-dump error naming `generate` instead of silently solving
    /// an older run's VCs. Removal failures only warn: the benchmark is
    /// already failing, and `solve` still fails loudly on whichever
    /// piece could not be cleaned up (an unreadable manifest is itself a
    /// hard `solve` error).
    fn remove_stale_dump(&self, def: &BenchmarkDef) {
        let Some(vcs_dir) = self.config.vcs_dir.as_deref() else {
            return;
        };
        if def.optimized.is_none() {
            return; // race-check benchmarks have no dump
        }
        let path = vc_dump_path(vcs_dir, &def.name);
        match std::fs::remove_file(&path) {
            Ok(()) => eprintln!(
                "note: {}: removed outdated VC dump {} (this generation failed; \
                 re-run `volta-bench generate` before solving)",
                def.name,
                path.display()
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => eprintln!(
                "warning: {}: could not remove outdated VC dump {}: {}",
                def.name,
                path.display(),
                e
            ),
        }
        match crate::manifest::read_manifest(vcs_dir) {
            Ok(Some(mut manifest)) => {
                if crate::manifest::remove_entry(&mut manifest, &def.name)
                    && let Err(e) = crate::manifest::write_manifest(vcs_dir, &manifest)
                {
                    eprintln!(
                        "warning: {}: could not drop the vcs manifest entry: {:#}",
                        def.name, e
                    );
                }
            }
            Ok(None) => {}
            Err(e) => eprintln!(
                "warning: {}: could not read the vcs manifest to drop its entry: {:#}",
                def.name, e
            ),
        }
    }

    /// The console note for race-check benchmarks under a configured
    /// `vcs_dir`: they produce no VC dump. stderr, like the runner's
    /// other progress chatter: it must not panic mid-run when stdout is
    /// a closed pipe (`| head`) - the results files still have to be
    /// written afterwards.
    fn note_raceless_dump(&self, def: &BenchmarkDef) {
        if self.config.vcs_dir.is_some() && def.optimized.is_none() {
            eprintln!(
                "note: {}: race-check benchmark (no optimized kernel) - no VC dump",
                def.name
            );
        }
    }

    /// The one result-assembly tail, shared by every per-benchmark entry
    /// point (`run`, `run_generate`, and `solve`'s `run_solve`): elapsed
    /// time from `start`, the expected-outcome judgment, the pass rule,
    /// and the noisy-timing warning.
    ///
    /// The pass rule: the outcome matched the expectation, and the Z3
    /// phase - when one ran - completed. Z3 *non-verdicts*
    /// (unknown/timeout/unsupported/error elements) are the comparison's
    /// data, never failures. A Z3 `not_equivalent` verdict is data too
    /// when the decision procedure ruled the row, but in a z3-only solve
    /// nothing else rules it, so the refutation counts (plain run plus
    /// the `+exp-axiom` sub-run) are folded into the `Z3Only` outcome
    /// here and fail the row through the expectation match - see
    /// [`ActualOutcome::matches`] for why a possibly-spurious refutation
    /// must still be surfaced.
    pub(crate) fn assemble_result(
        &self,
        def: &BenchmarkDef,
        start: Instant,
        mut output: RunOutput,
    ) -> BenchmarkResult {
        let elapsed_secs = start.elapsed().as_secs_f64();
        if let ActualOutcome::Z3Only { refutations } = &mut output.outcome {
            *refutations = z3_refutations(output.z3.as_ref());
        }
        let outcome_matched = output.outcome.matches(def.expected);
        let passed = outcome_matched && !matches!(output.z3, Some(Z3PhaseOutcome::Failed(_)));
        let result = BenchmarkResult {
            name: def.name.clone(),
            category: def.category,
            elapsed_secs,
            outcome: output.outcome,
            stats: output.stats,
            outcome_matched,
            passed,
            dump_path: output.dump_path,
            z3: output.z3,
        };
        warn_noisy_phases(&result);
        result
    }

    pub fn run_all(&self, defs: &[BenchmarkDef]) -> Vec<BenchmarkResult> {
        self.run_each(defs, |def| self.run(def))
    }

    /// [`run_generate`] over a benchmark list, with the same verbose
    /// chatter as [`run_all`].
    pub fn generate_all(&self, defs: &[BenchmarkDef]) -> Vec<BenchmarkResult> {
        self.run_each(defs, |def| self.run_generate(def))
    }

    fn run_each(
        &self,
        defs: &[BenchmarkDef],
        run_one: impl Fn(&BenchmarkDef) -> BenchmarkResult,
    ) -> Vec<BenchmarkResult> {
        defs.iter()
            .map(|def| {
                if self.config.verbose {
                    eprintln!("running {} ...", def.name);
                }
                let result = run_one(def);
                if self.config.verbose {
                    eprintln!(
                        "  -> {} in {:.1}s",
                        result.outcome.status(),
                        result.elapsed_secs
                    );
                }
                result
            })
            .collect()
    }

    /// Read and parse the benchmark's kernel file(s) - once per run,
    /// before the timed generation loop: file I/O and parsing are not
    /// part of VC generation (lowering is; it happens inside
    /// `analyze_kernel`, per iteration).
    fn load_benchmark<'d>(&self, def: &'d BenchmarkDef) -> Result<LoadedBenchmark<'d>> {
        let load = |run: &'d KernelRun| {
            load_module(&self.config.kernels_dir.join(&run.path)).map(|module| (run, module))
        };
        Ok(LoadedBenchmark {
            reference: load(&def.reference)?,
            optimized: def.optimized.as_ref().map(load).transpose()?,
        })
    }

    /// The one-shot pipeline: the generation phase, then the solve
    /// phase(s) over its VCs in the same process.
    fn run_inner(&self, def: &BenchmarkDef) -> Result<RunOutput, RunFailure> {
        let vcs = match self.generate_inner(def)? {
            GeneratedRun::Done(output) => return Ok(output),
            GeneratedRun::Vcs(vcs) => vcs,
        };
        let GeneratedVcs {
            mut stats,
            reference,
            optimized,
            paired,
            // A failed write was already warned about inside
            // `persist_vcs`; the one-shot verdict does not depend on the
            // dump.
            dump,
        } = vcs;
        let dump_path = dump.path().map(Path::to_path_buf);

        // --- Decision solve: `iterations` runs over the same sampled
        // elements. A failure past this point happens *after* the dump
        // was written, so it carries the dump path.
        let arrays = def.reference.config.output_array_names();
        let outcome = self
            .check_equivalence(&reference, &optimized, &arrays, &mut stats)
            .map_err(|error| RunFailure {
                error,
                dump_path: dump_path.clone(),
            })?;

        // --- Z3 solve (optional): the exact same sampled elements.
        let z3 = self.config.z3.as_ref().map(|options| {
            let sampled = sampled_elements(&paired, self.config.sample);
            run_z3_phase(
                &reference,
                &optimized,
                &arrays,
                &sampled,
                self.config.sample,
                self.config.iterations,
                options,
            )
        });

        Ok(RunOutput {
            outcome,
            stats,
            dump_path,
            z3,
        })
    }

    /// The generation phase, shared verbatim by the one-shot pipeline
    /// and the `generate` subcommand: parse-once load, `iterations`
    /// fingerprint-checked generation runs, execution counters, and the
    /// VC dump write (when a `vcs_dir` is configured).
    fn generate_inner(&self, def: &BenchmarkDef) -> Result<GeneratedRun, RunFailure> {
        let mut stats = BenchmarkStats::default();

        // Kernel files are read and parsed once, outside the timed
        // generation loop below - file I/O and parsing are not part of
        // the VC-generation phase.
        let loaded = self.load_benchmark(def)?;

        // --- VC generation: `iterations` timed runs. Only the last
        // one's outputs are kept (dropping the previous before the next
        // starts, so peak memory is a single generation); every later
        // iteration's fingerprint (outcome kind, footprints, expression
        // identities) must match iteration 1's.
        let mut first_shape: Option<GenShape> = None;
        let mut last: Option<Generated> = None;
        for iteration in 1..=self.config.iterations.get() {
            drop(last.take());
            let gen_start = Instant::now();
            let generated = generate(&loaded)?;
            stats
                .vc_gen_iters_secs
                .push(gen_start.elapsed().as_secs_f64());
            let shape = generated.shape();
            match &first_shape {
                None => first_shape = Some(shape),
                Some(first) => {
                    if let Some(diff) = gen_shape_mismatch(first, &shape) {
                        return Err(anyhow!(
                            "VC generation is nondeterministic: iteration {} disagrees \
                             with iteration 1: {}",
                            iteration,
                            diff
                        )
                        .into());
                    }
                }
            }
            last = Some(generated);
        }

        let (reference, optimized, paired) = match last.expect("iterations >= 1") {
            Generated::Rejected { outcome } => {
                return Ok(GeneratedRun::Done(RunOutput {
                    outcome,
                    stats,
                    dump_path: None,
                    z3: None,
                }));
            }
            Generated::RaceFree { reference } => {
                stats.instructions = reference.stats.instructions;
                stats.block_syncs = reference.stats.block_syncs;
                stats.warp_syncs = reference.stats.warp_syncs;
                stats.reference_op_counts = reference.op_counts.clone();
                return Ok(GeneratedRun::Done(RunOutput {
                    outcome: ActualOutcome::RaceFree,
                    stats,
                    dump_path: None,
                    z3: None,
                }));
            }
            Generated::Equivalence {
                reference,
                optimized,
                paired,
            } => (reference, optimized, paired),
        };

        // Execution counters from the last generation (every iteration
        // executes identically; the fingerprint check above guards the
        // footprint-and-expression part of that). The paper's tables
        // report the optimized kernel's sync counts.
        stats.instructions = reference.stats.instructions + optimized.stats.instructions;
        stats.block_syncs = optimized.stats.block_syncs;
        stats.warp_syncs = optimized.stats.warp_syncs;
        stats.reference_op_counts = reference.op_counts.clone();
        stats.optimized_op_counts = optimized.op_counts.clone();
        // The footprint size is generation data (the solve phases
        // re-derive the same total from the same pairing).
        stats.elements_total = paired.iter().map(|(_, common)| common.len() as u64).sum();

        // Persist the last generation's verification conditions (the
        // write itself is timed into `dump_write_secs`, not the
        // generation iterations).
        let persisted = persist_vcs(
            self.config.vcs_dir.as_deref(),
            &def.name,
            reference,
            optimized,
        );
        stats.dump_write_secs = persisted.dump.write_secs();

        Ok(GeneratedRun::Vcs(GeneratedVcs {
            stats,
            reference: persisted.reference,
            optimized: persisted.optimized,
            paired,
            dump: persisted.dump,
        }))
    }

    /// The decision-solve phase: compare the two outputs element for
    /// element along the named arrays, `iterations` times, filling the
    /// solve fields of `stats`. The actual element loop lives in
    /// `volta_analysis::driver`. Shared by the one-shot pipeline and the
    /// `solve` subcommand (crate::solve), whose VCs come from dumps.
    pub(crate) fn check_equivalence(
        &self,
        reference: &AnalysisOutput,
        optimized: &AnalysisOutput,
        arrays: &[String],
        stats: &mut BenchmarkStats,
    ) -> Result<ActualOutcome> {
        let options = EquivCheckOptions {
            sample: self.config.sample,
            verify_numeric: self.config.verify_numeric,
            recycle_terms: self.config.recycle_terms,
            iterations: self.config.iterations,
            parallelism: self.config.parallelism,
        };
        let report = check_output_equivalence_with(reference, optimized, arrays, &options)
            .context("checking output equivalence")?;
        stats.elements_checked = report.elements_checked;
        stats.elements_total = report.elements_total;
        stats.solve_iters_secs = report.check_iters.iter().map(|d| d.as_secs_f64()).collect();
        stats.solve_wall_iters_secs = report.wall_iters.iter().map(|d| d.as_secs_f64()).collect();
        stats.verify_numeric_secs = report.verify_time.map(|d| d.as_secs_f64());
        stats.decision_elements = report.element_checks;
        Ok(match report.outcome {
            EquivOutcome::Equivalent => ActualOutcome::Equivalent,
            EquivOutcome::NotEquivalent { mismatches } => {
                let first = mismatches
                    .first()
                    .map(|m| format!("{}[{}]", m.array, m.index))
                    .unwrap_or_default();
                ActualOutcome::NotEquivalent {
                    mismatches: mismatches.len(),
                    first,
                }
            }
        })
    }
}

/// Total `not_equivalent` verdicts across a completed Z3 phase - the
/// plain encoding plus, when present, the `+exp-axiom` sub-run. Zero for
/// an absent or failed phase (a failed phase already fails the pass rule
/// on its own).
fn z3_refutations(z3: Option<&Z3PhaseOutcome>) -> usize {
    match z3 {
        Some(Z3PhaseOutcome::Ran(phase)) => {
            phase.plain.counts.not_equivalent
                + phase
                    .axiom
                    .as_ref()
                    .map_or(0, |axiom| axiom.counts.not_equivalent)
        }
        Some(Z3PhaseOutcome::Failed(_)) | None => 0,
    }
}

/// Print a stderr warning for every timed phase whose per-iteration
/// coefficient of variation exceeds [`NOISY_CV_THRESHOLD`]: the median
/// is still the headline number, but the reader should know it came from
/// noisy samples.
fn warn_noisy_phases(result: &BenchmarkResult) {
    // (Called from `assemble_result`, so every per-benchmark entry point
    // - one-shot, generate, solve - gets the warning.)
    let mut phases: Vec<(&str, &[f64])> = vec![
        ("VC generation", &result.stats.vc_gen_iters_secs),
        ("decision solve", &result.stats.solve_iters_secs),
    ];
    if let Some(Z3PhaseOutcome::Ran(phase)) = &result.z3 {
        phases.push(("z3 solve", &phase.plain.iters_secs));
        if let Some(axiom) = &phase.axiom {
            phases.push(("z3 +exp-axiom solve", &axiom.iters_secs));
        }
    }
    for (phase, iters) in phases {
        if let Some(cv) = cv(iters)
            && cv > NOISY_CV_THRESHOLD
        {
            eprintln!(
                "warning: {}: {} timing noisy (CV {:.2} > {:.2}); \
                 consider more iterations or a quieter machine",
                result.name, phase, cv, NOISY_CV_THRESHOLD
            );
        }
    }
}

/// The result of [`persist_vcs`]: the analysis outputs (moved through the
/// dump, never cloned - the arenas can be GiB-scale) plus what got written.
struct PersistedVcs {
    reference: AnalysisOutput,
    optimized: AnalysisOutput,
    dump: DumpPersistence,
}

/// Write one dump and return the FNV-1a fingerprint of the exact bytes
/// put on disk (the manifest's `vc_fingerprint`; see `crate::manifest`).
/// The dump serializes straight through the hashing tee to the file, so
/// no second in-memory copy of the (possibly GiB-scale) payload exists;
/// `solve` recomputes the same digest from `fs::read` of the file.
fn write_dump_hashed(path: &Path, dump: &VcDump) -> std::io::Result<u64> {
    let file = std::fs::File::create(path)?;
    let mut writer = crate::manifest::HashingWriter::new(std::io::BufWriter::new(file));
    write_vc_dump_to(&mut writer, dump)?;
    std::io::Write::flush(&mut writer)?;
    Ok(writer.fingerprint())
}

/// Persist one equivalence benchmark's verification conditions to
/// `<vcs_dir>/<sanitized-name>.vcdump` via the shared
/// `volta_analysis::driver::vc_dump` format (the same file `volta compare
/// --dump-vcs` writes and `--from-dump` reads), overwriting any previous
/// run's dump - VCs are deterministic (and the generation phase's
/// fingerprint check enforces the footprint-and-expression-identity part
/// of that per run). A write failure
/// warns and carries on: a full disk should not change a benchmark
/// verdict.
///
/// The outputs are moved into the dump and moved back out
/// (`into_analysis_output`), which clears their `stats`/`op_counts` -
/// callers must record those before calling. Byte-identity across runs
/// rests on one premise: no production code path creates machine symbols
/// (`ExprArena::symbol`, the only id drawn from a process-global
/// counter), so every id in a dump is deterministic; a future
/// machine-symbol caller would void byte-identity across runs but not
/// the dumps' validity - `--from-dump` never depends on the numeric id
/// values, and the manifest fingerprint pins whatever bytes were
/// actually written.
fn persist_vcs(
    vcs_dir: Option<&Path>,
    benchmark_name: &str,
    reference: AnalysisOutput,
    optimized: AnalysisOutput,
) -> PersistedVcs {
    let Some(vcs_dir) = vcs_dir else {
        return PersistedVcs {
            reference,
            optimized,
            dump: DumpPersistence::NotConfigured,
        };
    };
    let path = vc_dump_path(vcs_dir, benchmark_name);
    let dump = VcDump {
        reference: VcSnapshot::from_output(reference),
        optimized: VcSnapshot::from_output(optimized),
    };
    // The directory is a one-time setup cost, not part of any dump's
    // write time - create it before starting the write timer. (Hashing
    // is inside the timed write: it rides along with the byte stream.)
    let created = std::fs::create_dir_all(vcs_dir);
    let write0 = Instant::now();
    let written = created.and_then(|_| write_dump_hashed(&path, &dump));
    let persisted = match written {
        Ok(fingerprint) => DumpPersistence::Written {
            path,
            write_secs: write0.elapsed().as_secs_f64(),
            fingerprint,
        },
        Err(e) => {
            eprintln!("warning: could not write VC dump {}: {}", path.display(), e);
            DumpPersistence::Failed(e.to_string())
        }
    };
    PersistedVcs {
        reference: dump.reference.into_analysis_output(),
        optimized: dump.optimized.into_analysis_output(),
        dump: persisted,
    }
}

fn rejected_outcome(e: EvalError) -> ActualOutcome {
    let is_race = matches!(e, EvalError::DataRace { .. });
    ActualOutcome::Rejected {
        description: e.to_string(),
        is_race,
    }
}

#[cfg(test)]
mod tests {
    use id_collections::Id;

    use super::*;

    fn fp(node_count: usize, elems: &[(u64, u32)]) -> KernelFingerprint {
        KernelFingerprint {
            node_count,
            outputs: vec![(
                "out".to_string(),
                elems
                    .iter()
                    .map(|&(i, id)| (i, ExprId::from_index(id)))
                    .collect(),
            )],
        }
    }

    #[test]
    fn identical_fingerprints_agree() {
        let a = fp(7, &[(0, 3), (1, 5)]);
        let b = fp(7, &[(0, 3), (1, 5)]);
        assert_eq!(fingerprint_mismatch("reference", &a, &b), None);
    }

    #[test]
    fn node_count_divergence_is_named() {
        let a = fp(7, &[(0, 3)]);
        let b = fp(8, &[(0, 3)]);
        let msg = fingerprint_mismatch("reference", &a, &b).unwrap();
        assert!(msg.contains("7 vs 8 arena nodes"), "{}", msg);
    }

    #[test]
    fn expression_identity_divergence_is_named() {
        // Same footprint indices, different ExprIds: exactly the case
        // the pre-fingerprint shape check let slip through.
        let a = fp(7, &[(0, 3), (1, 5)]);
        let b = fp(7, &[(0, 3), (1, 6)]);
        let msg = fingerprint_mismatch("optimized", &a, &b).unwrap();
        assert!(
            msg.contains("array 'out' element 1 built expression"),
            "{}",
            msg
        );
    }

    #[test]
    fn footprint_index_divergence_is_named() {
        let a = fp(7, &[(0, 3)]);
        let b = fp(7, &[(2, 3)]);
        let msg = fingerprint_mismatch("the", &a, &b).unwrap();
        assert!(msg.contains("wrote element 0 vs 2"), "{}", msg);
    }

    #[test]
    fn rejection_kind_flip_is_named_but_text_is_not_compared() {
        // Rejections compare by verdict kind only: diagnostic text may
        // embed schedule-dependent details.
        let race = GenShape::Rejected { status: "RACE" };
        let reject = GenShape::Rejected { status: "REJECT" };
        assert_eq!(
            gen_shape_mismatch(&race, &GenShape::Rejected { status: "RACE" }),
            None
        );
        let msg = gen_shape_mismatch(&race, &reject).unwrap();
        assert!(msg.contains("RACE") && msg.contains("REJECT"), "{}", msg);
    }

    /// An equivalence benchmark (expected Equivalent) whose kernel paths
    /// are never read - only `assemble_result` is exercised.
    fn equivalence_def() -> BenchmarkDef {
        let mut config = volta_analysis::eval::AnalysisConfig::new((1, 1, 1));
        config.arrays = vec![crate::config::f32_output("out", 0x1000, 1)];
        let run = || crate::config::KernelRun::new("unused.ptx", "k", config.clone());
        BenchmarkDef::equivalence(
            "Synthetic Pair",
            crate::config::BenchmarkCategory::Reduction,
            run(),
            run(),
        )
    }

    fn z3_mode_run(counts: volta_z3::Z3Counts) -> crate::z3_phase::Z3ModeRun {
        crate::z3_phase::Z3ModeRun {
            iters_secs: vec![0.0],
            counts,
            elements: Vec::new(),
        }
    }

    fn assemble_z3_only(z3: Z3PhaseOutcome) -> BenchmarkResult {
        let runner = BenchmarkRunner::new(RunnerConfig::default());
        runner.assemble_result(
            &equivalence_def(),
            Instant::now(),
            RunOutput {
                // `refutations` starts at 0; `assemble_result` folds the
                // real counts in - exactly what `solve --backend z3`
                // hands it.
                outcome: ActualOutcome::Z3Only { refutations: 0 },
                stats: BenchmarkStats::default(),
                dump_path: None,
                z3: Some(z3),
            },
        )
    }

    /// The Z3-refutation pass rule: in a z3-only solve, a
    /// `not_equivalent` count > 0 fails the row (status `Z3 DIFF`) - in
    /// that mode nothing else rules it, and an affirmative refutation
    /// (even a potentially spurious one) must be surfaced, not swallowed.
    #[test]
    fn z3_only_refutation_fails_the_row() {
        use crate::z3_phase::Z3Phase;

        let refuted = |not_equivalent| volta_z3::Z3Counts {
            not_equivalent,
            ..Default::default()
        };

        // A refutation in the plain run.
        let result = assemble_z3_only(Z3PhaseOutcome::Ran(Z3Phase {
            plain: z3_mode_run(refuted(1)),
            axiom: None,
        }));
        assert!(!result.passed);
        assert!(!result.outcome_matched);
        assert_eq!(result.outcome.status(), "Z3 DIFF");
        assert!(matches!(
            result.outcome,
            ActualOutcome::Z3Only { refutations: 1 }
        ));

        // A refutation only in the +exp-axiom sub-run counts the same.
        let result = assemble_z3_only(Z3PhaseOutcome::Ran(Z3Phase {
            plain: z3_mode_run(volta_z3::Z3Counts::default()),
            axiom: Some(z3_mode_run(refuted(2))),
        }));
        assert!(!result.passed);
        assert_eq!(result.outcome.status(), "Z3 DIFF");
        assert!(matches!(
            result.outcome,
            ActualOutcome::Z3Only { refutations: 2 }
        ));
    }

    /// Z3 non-verdicts (unknown/timeout/unsupported) stay non-failures
    /// in a z3-only solve - the paper-honest reading of Table 8's rows.
    #[test]
    fn z3_only_non_verdicts_still_pass() {
        use crate::z3_phase::Z3Phase;

        let result = assemble_z3_only(Z3PhaseOutcome::Ran(Z3Phase {
            plain: z3_mode_run(volta_z3::Z3Counts {
                unknown: 1,
                timeout: 2,
                unsupported: 3,
                ..Default::default()
            }),
            axiom: None,
        }));
        assert!(result.passed);
        assert!(result.outcome_matched);
        assert_eq!(result.outcome.status(), "Z3");
    }

    /// When the decision procedure ruled the row (one-shot `--z3` or
    /// `solve --backend both`), z3 counts stay comparison data: a
    /// `not_equivalent` there does not override the decision verdict.
    #[test]
    fn z3_refutation_is_data_when_the_decision_procedure_ruled() {
        use crate::z3_phase::Z3Phase;

        let runner = BenchmarkRunner::new(RunnerConfig::default());
        let result = runner.assemble_result(
            &equivalence_def(),
            Instant::now(),
            RunOutput {
                outcome: ActualOutcome::Equivalent,
                stats: BenchmarkStats::default(),
                dump_path: None,
                z3: Some(Z3PhaseOutcome::Ran(Z3Phase {
                    plain: z3_mode_run(volta_z3::Z3Counts {
                        not_equivalent: 1,
                        ..Default::default()
                    }),
                    axiom: None,
                })),
            },
        );
        assert!(result.passed);
        assert_eq!(result.outcome.status(), "EQUIV");
    }
}

/// Load and parse a PTX module.
pub fn load_module(path: &Path) -> Result<Module> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let ascii_src = contents
        .as_bytes()
        .as_ascii_slice()
        .context("file contains non-ASCII characters")?;
    let mut parser = Parser::new(ascii_src);
    parser
        .parse_module()
        .map_err(|e| anyhow!("parse error: {}", e.error.title()))
}
