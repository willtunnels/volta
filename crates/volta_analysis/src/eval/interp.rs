//! The interpreter: round-robin symbolic execution with χ-context race
//! detection (paper Sections 3 and 5).
//!
//! Each thread runs until it blocks (barrier or warp-cooperative op) or
//! exits; then the next ready thread runs. When no thread is ready, complete
//! barrier/warp groups fire; if none can, the program is deadlocked. By the
//! confluence theorem, this particular schedule is as good as any other.

use id_collections::IdVec;

use volta_frontend::ast::ScalarType;

use crate::eval::config::{AnalysisConfig, ParamValue};
use crate::eval::error::{EvalError, EvalResult};
use crate::eval::memory::{MemAccessError, Memory};
use crate::eval::race::RaceTracker;
use crate::eval::value::{RegFile, Value};
use crate::eval::{ThreadId, WARP_SIZE};
use crate::logging::{info, trace, warn};
use crate::lowered::{
    BinOp, Clamp, CmpOp, InstrId, LoweredInstr, LoweredProgram, MemSpace, Operand, UnaryOp,
};
use crate::symbolic::{ExprArena, ExprId, Real, StringId};
use crate::symbols::{MODULE_GLOBAL_BASE, ParamId, RegId, SpecialRegKind};
use crate::types::ScalarTypeExt;

/// Per-array output footprint: `(array name, [(element index, value)])`.
pub type OutputFootprints = Vec<(String, Vec<(u64, ExprId)>)>;

/// Execution statistics matching the paper's table columns.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Total instructions executed across all threads
    pub instructions: u64,
    /// `bar.sync` executions across all threads ("#Block Sync")
    pub block_syncs: u64,
    /// Warp-level sync operations, counted once per fired group
    /// (`shfl.sync`, `ldmatrix`, `mma.sync`, `wmma.*`, ...; "#Warp Sync" -
    /// the paper's tables count these per warp, not per thread)
    pub warp_syncs: u64,
}

/// Scheduling status of one thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::eval) enum Status {
    Ready,
    /// Blocked at `bar.sync id` (at the current pc)
    AtBarrier {
        id: u32,
    },
    /// Blocked at a warp-cooperative instruction at the current pc.
    /// `mask` is the participating-lane mask within the thread's warp.
    AtWarpOp {
        mask: u32,
    },
    Exited,
}

#[derive(Debug)]
pub(in crate::eval) struct ThreadState {
    pub pc: InstrId,
    pub regs: RegFile,
    pub status: Status,
}

/// A contiguous validity region within one memory space.
#[derive(Debug, Clone)]
struct Region {
    base: u64,
    size: u64,
}

impl Region {
    /// Whole-access containment, in subtraction form: no sum here can
    /// overflow in any build profile. The additive form
    /// `addr + width <= base + size` wraps for addresses near `u64::MAX`
    /// (e.g. a negative index reaching `effective_addr`) and in release
    /// mode silently *accepts* the wrapped access; verification-relevant
    /// checks must not rely on debug overflow panics.
    fn contains(&self, addr: u64, width: u64) -> bool {
        width <= self.size && addr >= self.base && addr - self.base <= self.size - width
    }

    /// Whether `addr` lies inside `[base, base + size)`, i.e. this region
    /// owns the byte. Subtraction form; a zero-size region owns nothing.
    fn owns(&self, addr: u64) -> bool {
        addr >= self.base && addr - self.base < self.size
    }
}

/// Declared regions per space; every access must fall entirely inside one
/// region (this is what catches per-array out-of-bounds accesses).
#[derive(Debug, Default)]
struct MemRegions {
    global: Vec<Region>,
    shared: Vec<Region>,
    local: Vec<Region>,
}

/// Result of a completed analysis.
#[derive(Debug)]
pub struct AnalysisOutput {
    pub arena: ExprArena,
    /// Output arrays: (name, written elements as (index, expression)),
    /// sorted by index. Only elements the kernel wrote appear.
    pub outputs: Vec<(String, Vec<(u64, ExprId)>)>,
    pub stats: Stats,
    /// Instructions executed, broken down by kind (`LoweredInstr::kind_name`)
    /// summed across all threads. A `BTreeMap` keeps printed/exported tables
    /// in deterministic order.
    pub op_counts: std::collections::BTreeMap<&'static str, u64>,
}

pub struct Interpreter<'p> {
    pub(in crate::eval) program: &'p LoweredProgram,
    pub(in crate::eval) arena: ExprArena,
    config: AnalysisConfig,
    n_threads: u32,
    /// Shared `Undefined` node returned for reads of never-written
    /// registers (see `read_reg`).
    undefined: ExprId,
    params: IdVec<ParamId, Value>,
    pub(in crate::eval) threads: IdVec<ThreadId, ThreadState>,
    pub(in crate::eval) global: Memory,
    pub(in crate::eval) shared: Memory,
    locals: IdVec<ThreadId, Memory>,
    regions: MemRegions,
    pub(in crate::eval) race: RaceTracker,
    pub(in crate::eval) stats: Stats,
    /// Per-kind instruction counts, indexed by `LoweredInstr::kind_index`.
    /// A fixed array (not a map) because this is bumped once per executed
    /// instruction in `step`, the interpreter's innermost loop; `finish`
    /// folds it into the `BTreeMap` shape `AnalysisOutput` exposes.
    pub(in crate::eval) op_counts: [u64; crate::lowered::KIND_COUNT],
}

impl<'p> Interpreter<'p> {
    pub fn new(program: &'p LoweredProgram, config: AnalysisConfig) -> EvalResult<Self> {
        let n_threads = config.num_threads();
        if n_threads == 0 {
            return Err(EvalError::Config {
                message: "block has zero threads".to_string(),
            });
        }

        config
            .validate()
            .map_err(|message| EvalError::Config { message })?;

        let mut arena = ExprArena::new();
        let undefined = arena.undefined();

        // Bind parameters positionally.
        let declared = program.symbols.params();
        if declared.len() != config.params.len() {
            return Err(EvalError::Config {
                message: format!(
                    "kernel declares {} parameters but {} were provided",
                    declared.len(),
                    config.params.len()
                ),
            });
        }
        let mut params: IdVec<ParamId, Value> = IdVec::new();
        for value in &config.params {
            let v = match value {
                ParamValue::Int(v) => Value::Scalar(arena.int(*v)),
                // Exact ingestion; NaN was rejected by `config.validate()`
                // above, but the conversion stays fallible so a bypassing
                // caller still fails loudly.
                ParamValue::Float(v) => {
                    Value::Scalar(arena.float_from_f64(*v).map_err(|e| EvalError::Config {
                        message: format!("float parameter: {}", e),
                    })?)
                }
                ParamValue::SymFloat(name) => Value::Scalar(arena.param_symbol(name.clone())),
                ParamValue::ArrayPtr(name) => {
                    let array = config.array(name).ok_or_else(|| EvalError::Config {
                        message: format!("parameter references unknown array '{}'", name),
                    })?;
                    Value::Scalar(arena.int(array.base as i64))
                }
            };
            let _ = params.push(v);
        }

        // Build validity regions. Every region must satisfy
        // `base + size <= u64::MAX` (checked here, release-active): with
        // that invariant and the subtraction-form `Region::contains`, any
        // access that passes `check_bounds` has `addr + width` within u64
        // range, so the byte-range loops downstream (race recording, input
        // materialization, memory granules, vector element addressing)
        // can never wrap. The argument needs nothing from the address
        // itself - `effective_addr` produces arbitrary, possibly wrapped
        // u64s - only that ownership is checked before any of those loops
        // run, which `mem_read`/`mem_write` and the whole-footprint checks
        // at the vector/tensor-core sites guarantee.
        fn push_region(list: &mut Vec<Region>, base: u64, size: u64, what: &str) -> EvalResult<()> {
            if size > u64::MAX - base {
                return Err(EvalError::Config {
                    message: format!(
                        "{} region [{:#x}, {:#x} + {}) overflows the address space",
                        what, base, base, size
                    ),
                });
            }
            list.push(Region { base, size });
            Ok(())
        }
        let mut regions = MemRegions::default();
        // Config arrays and module-scope globals share the one global
        // region list built below. `config.validate()` keeps the arrays
        // pairwise disjoint and the symbol-table packer keeps the module
        // globals pairwise disjoint, so the only possible cross-family
        // overlap is an array intersecting the reserved module-global
        // window - reject it here to uphold the region-disjointness
        // premise of `check_bounds` (an overlapping array would silently
        // shadow the module global it covers). The window end cannot
        // overflow: `declare_global_var` checks `MODULE_GLOBAL_BASE +
        // offset + size` for every variable it places. Shared and local
        // variables cannot collide with config arrays by construction:
        // they are packed in their own address spaces, and `check_bounds`
        // consults only the accessed `MemSpace`'s region list, so no
        // check is needed for them.
        let module_global_size = program.symbols.module_global_size();
        if module_global_size > 0 {
            let window_base = MODULE_GLOBAL_BASE;
            let window_end = MODULE_GLOBAL_BASE + module_global_size;
            for array in &config.arrays {
                // `validate()` established `base + size_bytes()` fits.
                let array_end = array.base + array.size_bytes();
                if array.base < window_end && window_base < array_end {
                    return Err(EvalError::Config {
                        message: format!(
                            "array '{}' ([{:#x}, {:#x})) overlaps the reserved \
                             module-global region [{:#x}, {:#x})",
                            array.name, array.base, array_end, window_base, window_end
                        ),
                    });
                }
            }
        }
        for array in &config.arrays {
            push_region(&mut regions.global, array.base, array.size_bytes(), "array")?;
        }
        for var in program.symbols.global_vars() {
            push_region(
                &mut regions.global,
                var.addr,
                var.size_bytes,
                "global variable",
            )?;
        }
        if program.symbols.has_extern_shared() && config.dynamic_shared_bytes == 0 {
            return Err(EvalError::Config {
                message: "kernel uses extern shared memory; set dynamic_shared_bytes".to_string(),
            });
        }
        for info in program.symbols.shared_vars() {
            if info.is_extern {
                // Every extern name aliases the one dynamic window; a single
                // region for it is added below.
                continue;
            }
            push_region(
                &mut regions.shared,
                info.offset,
                info.size_bytes,
                "shared variable",
            )?;
        }
        // The dynamic (`.extern .shared`) window: based after all static
        // allocations, sized by the launch configuration.
        if let Some(base) = program.symbols.extern_shared_base() {
            push_region(
                &mut regions.shared,
                base,
                config.dynamic_shared_bytes,
                "dynamic shared",
            )?;
        }
        for var in program.symbols.local_vars() {
            push_region(
                &mut regions.local,
                var.offset,
                var.size_bytes,
                "local variable",
            )?;
        }

        // Input-array symbols are materialized lazily on first read (arrays
        // can be huge - e.g. 4096x4096 matmul operands - while a single CTA
        // touches only a sliver). Module-scope globals are placed eagerly.
        let mut global = Memory::new();
        for (name, value) in &config.global_values {
            let var = program
                .symbols
                .get_global_var(name)
                .ok_or_else(|| EvalError::Config {
                    message: format!("no module-scope .global variable named '{}'", name),
                })?;
            let v = Value::Scalar(arena.int(*value));
            // Analysis-setup placement, not a PTX memory instruction, so
            // the natural-alignment rule does not apply; the address is
            // naturally aligned anyway (`declare_global_var` packs with
            // `align_up` from the naturally-aligned `MODULE_GLOBAL_BASE`).
            global
                .init(var.addr, var.size_bytes, v)
                .expect("module-global initialization cannot fail");
        }

        let counts = program.register_counts();
        let threads = IdVec::from_vec(
            (0..n_threads)
                .map(|_| ThreadState {
                    pc: program.entry_pc,
                    regs: RegFile::new(&counts),
                    status: Status::Ready,
                })
                .collect(),
        );
        let locals = IdVec::from_vec((0..n_threads).map(|_| Memory::new()).collect());

        Ok(Self {
            program,
            arena,
            config,
            n_threads,
            undefined,
            params,
            threads,
            global,
            shared: Memory::new(),
            locals,
            regions,
            race: RaceTracker::new(n_threads as usize),
            stats: Stats::default(),
            op_counts: [0; crate::lowered::KIND_COUNT],
        })
    }

    /// Run to completion (all threads exited) or an analysis error.
    pub fn run(&mut self) -> EvalResult<()> {
        loop {
            match self.next_ready() {
                Some(t) => self.run_thread(t)?,
                None => {
                    if self.threads.values().all(|t| t.status == Status::Exited) {
                        info!(
                            "execution complete: {} instructions, {} block syncs, {} warp syncs",
                            self.stats.instructions, self.stats.block_syncs, self.stats.warp_syncs
                        );
                        return Ok(());
                    }
                    if !self.try_fire()? {
                        return Err(self.deadlock_error());
                    }
                }
            }
        }
    }

    /// Extract the kernel's output footprint: for each output array, every
    /// element the program actually wrote (a single CTA typically writes
    /// only its tile of a large output tensor). Elements are keyed by index
    /// so two kernels' footprints can be compared exactly.
    pub fn extract_outputs(&self) -> EvalResult<OutputFootprints> {
        let mut outputs = Vec::new();
        for array in &self.config.arrays {
            if !array.kind.is_output() {
                continue;
            }
            let size = array.size_bytes();
            let mut elems: Vec<(u64, ExprId)> = Vec::new();
            for (addr, width, value) in self.global.dirty_cells() {
                // Whole-cell containment in subtraction form (as in
                // `Region::contains`): no overflowing sums for any cell.
                if addr < array.base || width > size || addr - array.base > size - width {
                    continue;
                }
                let offset = addr - array.base;
                if offset % array.elem_width != 0 {
                    return Err(EvalError::Config {
                        message: format!(
                            "output array '{}' was written at misaligned offset {:#x}",
                            array.name, offset
                        ),
                    });
                }
                let index = offset / array.elem_width;
                match (value, width == array.elem_width) {
                    (Value::Scalar(e), true) => {
                        if self.arena.is_undefined(e) {
                            return Err(EvalError::UndefinedOutput {
                                array: array.name.clone(),
                                index,
                            });
                        }
                        elems.push((index, e));
                    }
                    // A packed pair granule over two adjacent narrow elements.
                    (Value::Pair(lo, hi), false) if width == 2 * array.elem_width => {
                        for (k, e) in [(0, lo), (1, hi)] {
                            if self.arena.is_undefined(e) {
                                return Err(EvalError::UndefinedOutput {
                                    array: array.name.clone(),
                                    index: index + k,
                                });
                            }
                            elems.push((index + k, e));
                        }
                    }
                    _ => {
                        return Err(EvalError::Config {
                            message: format!(
                                "output array '{}' element {} was written at width {} \
                                 (element width {})",
                                array.name, index, width, array.elem_width
                            ),
                        });
                    }
                }
            }
            elems.sort_by_key(|(i, _)| *i);
            outputs.push((array.name.clone(), elems));
        }
        Ok(outputs)
    }

    /// Consume the interpreter, producing the analysis output.
    pub fn into_output(self) -> EvalResult<AnalysisOutput> {
        let outputs = self.extract_outputs()?;
        let op_counts = self
            .op_counts
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(i, &count)| (crate::lowered::KIND_NAMES[i], count))
            .collect();
        Ok(AnalysisOutput {
            arena: self.arena,
            outputs,
            stats: self.stats,
            op_counts,
        })
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    // =====================================================================
    // Scheduling
    // =====================================================================

    fn next_ready(&self) -> Option<ThreadId> {
        self.threads
            .iter()
            .find(|(_, t)| t.status == Status::Ready)
            .map(|(id, _)| id)
    }

    /// Run one thread until it blocks or exits.
    fn run_thread(&mut self, t: ThreadId) -> EvalResult<()> {
        while self.threads[t].status == Status::Ready {
            self.step(t)?;
        }
        Ok(())
    }

    /// Try to fire complete warp groups and barriers. Returns whether any
    /// group made progress.
    fn try_fire(&mut self) -> EvalResult<bool> {
        let mut any = false;
        loop {
            if let Some((pc, mask, members)) = self.find_ready_warp_group()? {
                trace!(
                    "warp op at pc {} fired (mask {:#010x}, {} lanes)",
                    pc.0,
                    mask,
                    members.len()
                );
                self.execute_warp_op(pc, mask, &members)?;
                any = true;
                continue;
            }
            if self.try_fire_barrier() {
                any = true;
                continue;
            }
            return Ok(any);
        }
    }

    /// Find a warp group whose live members have all arrived at the same pc
    /// with the same mask. Returns (pc, mask, live member threads).
    fn find_ready_warp_group(&self) -> EvalResult<Option<(InstrId, u32, Vec<ThreadId>)>> {
        'candidates: for (leader, state) in self.threads.iter() {
            let Status::AtWarpOp { mask } = state.status else {
                continue;
            };
            let pc = state.pc;
            let warp_base = (leader.0 / WARP_SIZE) * WARP_SIZE;
            let mut members = Vec::new();
            for lane in 0..WARP_SIZE {
                if mask & (1 << lane) == 0 {
                    continue;
                }
                let tid = warp_base + lane;
                if tid >= self.n_threads {
                    return Err(EvalError::WarpMismatch {
                        pc,
                        reason: format!(
                            "mask {:#010x} includes lane {} but the CTA has only {} threads",
                            mask, lane, self.n_threads
                        ),
                    });
                }
                let member = &self.threads[ThreadId(tid)];
                match member.status {
                    Status::AtWarpOp { mask: m } if m == mask && member.pc == pc => {
                        members.push(ThreadId(tid));
                    }
                    // An exited lane counts as arrived at *every* warp op,
                    // not just pure syncs: the paper's Sync rule fires when
                    // each i in I is at the sync *or at return*, and the ISA
                    // says the same for shfl.sync ("wait until all
                    // non-exited threads corresponding to membermask have
                    // executed shfl.sync"). Exited lanes execute nothing
                    // (they are excluded from `members`) but rejoin the
                    // group for the chi-clear in `execute_warp_op`; data
                    // sourced from them is handled per-op.
                    Status::Exited => {}
                    // A live lane elsewhere (different pc or mask): the
                    // group is not ready. Requiring one shared pc is a
                    // deliberate conservative deviation from both
                    // authorities: the paper's syncs are unnamed ("whichever
                    // sync instances happen to align in the dynamics are
                    // matched", section 4.1) and the sm_70+ ISA matches
                    // bar.warp.sync/shfl.sync instances by mask and
                    // qualifiers, not by program point - under either,
                    // differently-located syncs could pair. Volta matches
                    // only at a single pc, for implementation simplicity; a
                    // group whose live lanes never converge at one pc stays
                    // stuck and surfaces as a loud Deadlock rather than
                    // being cross-matched.
                    _ => continue 'candidates,
                }
            }
            return Ok(Some((pc, mask, members)));
        }
        Ok(None)
    }

    /// Fire the CTA barrier if every live thread waits on the same id.
    fn try_fire_barrier(&mut self) -> bool {
        let mut id: Option<u32> = None;
        for state in self.threads.values() {
            match state.status {
                Status::Exited => {}
                Status::AtBarrier { id: this_id } => match id {
                    None => id = Some(this_id),
                    Some(prev) if prev == this_id => {}
                    Some(_) => return false, // waiting on different barriers
                },
                _ => return false, // someone is ready or at a warp op
            }
        }
        if id.is_none() {
            return false; // everyone exited (or nobody is at a barrier)
        }
        // Deliberately the paper's Sync'/syncMem semantics with I = the full
        // CTA: exited threads count as arrived (the loop above) and are
        // *included* in the chi-clear - `sync_all` empties every pending
        // set, theirs too. This is stronger than the ISA's barrier{.cta}
        // ordering, which only orders accesses "relative to all threads
        // participating in the barrier" (an exited thread participates in
        // nothing), so a spec-level race pairing a thread's pre-exit access
        // with another thread's post-barrier access is intentionally not
        // reported.
        self.race.sync_all();
        trace!("fired bar.sync {}", id.unwrap_or(0));
        for state in self.threads.values_mut() {
            if let Status::AtBarrier { .. } = state.status {
                state.status = Status::Ready;
                state.pc = InstrId(state.pc.0 + 1);
            }
        }
        true
    }

    fn deadlock_error(&self) -> EvalError {
        let blocked: Vec<_> = self
            .threads
            .iter()
            .filter(|(_, t)| !matches!(t.status, Status::Exited))
            .map(|(id, t)| (id, t.pc))
            .collect();
        warn!("deadlock: {} threads blocked", blocked.len());
        EvalError::Deadlock { blocked }
    }

    /// Apply the χ synchronization of a fired warp group. Called *before*
    /// the group's cooperative memory accesses so they cannot race with the
    /// group's own pre-sync accesses.
    pub(in crate::eval) fn sync_warp_group(&mut self, members: &[ThreadId]) {
        let mut group = fixedbitset::FixedBitSet::with_capacity(self.n_threads as usize);
        for &m in members {
            group.insert(m.0 as usize);
        }
        self.race.sync_group(&group);
    }

    /// Unblock the members of a fired warp group and advance their pcs.
    pub(in crate::eval) fn advance_warp_group(&mut self, members: &[ThreadId]) {
        for &m in members {
            let state = &mut self.threads[m];
            state.status = Status::Ready;
            state.pc = InstrId(state.pc.0 + 1);
        }
    }

    // =====================================================================
    // Single-instruction execution
    // =====================================================================

    fn step(&mut self, t: ThreadId) -> EvalResult<()> {
        let pc = self.threads[t].pc;
        let Some(instr) = self.program.instruction(pc) else {
            return Err(EvalError::Unsupported {
                pc,
                what: "execution fell off the end of the program".to_string(),
            });
        };
        let instr = instr.clone();

        self.stats.instructions += 1;
        self.op_counts[instr.kind_index()] += 1;
        if self.stats.instructions > self.config.max_instructions {
            return Err(EvalError::InstructionLimit {
                limit: self.config.max_instructions,
            });
        }

        // Predicate guard: must be concrete (structured-CTA).
        if let Some(pred) = self.program.predicate(pc) {
            let value = self.read_reg(t, pc, pred.reg)?;
            let cond = self.as_concrete_bool(t, pc, value, "guard predicate")?;
            if cond == pred.negated {
                self.threads[t].pc = InstrId(pc.0 + 1);
                return Ok(());
            }
        }

        let mut next_pc = InstrId(pc.0 + 1);
        match &instr {
            // Parameter reads are interpreter-internal value bindings (an
            // `IdVec` lookup), not byte-addressed memory accesses, so the
            // natural-alignment rule for memory instructions has nothing to
            // apply to. Byte-addressed `.param`/`.const` accesses that do
            // reach `Load`/`Store` are rejected as unsupported in
            // `check_bounds`.
            LoweredInstr::LoadParam { dst, param_id } => {
                let v = self.params[*param_id];
                self.threads[t].regs.write(*dst, v);
            }

            LoweredInstr::Load {
                dst,
                space,
                base,
                offset,
                ty,
            } => {
                let addr = self.effective_addr(t, pc, base, *offset)?;
                let v = self.mem_read(t, pc, *space, addr, ty.size_bytes() as u64)?;
                let v = self.canon_loaded(t, pc, *ty, *dst, v)?;
                self.threads[t].regs.write(*dst, v);
            }

            LoweredInstr::LoadVec {
                dst,
                space,
                base,
                offset,
                ty,
            } => {
                let addr = self.effective_addr(t, pc, base, *offset)?;
                let width = ty.size_bytes() as u64;
                // The access size of a vector load is the *total* number of
                // bytes accessed (`ld.v4.b32` is one 16-byte access, PTX
                // ISA 6.4.1), so the whole vector's bounds and alignment
                // are checked once here. The whole-footprint bounds check
                // is load-bearing: the per-element checks in `mem_read`
                // below would each pass inside a *different* region and let
                // a v4 straddle two adjacent arrays silently.
                self.check_bounds(t, pc, *space, addr, dst.len() as u64 * width)?;
                self.check_alignment(t, pc, *space, addr, dst.len() as u64 * width)?;
                for (k, reg) in dst.iter().enumerate() {
                    let v = self.mem_read(t, pc, *space, addr + k as u64 * width, width)?;
                    let v = self.canon_loaded(t, pc, *ty, *reg, v)?;
                    self.threads[t].regs.write(*reg, v);
                }
            }

            LoweredInstr::Store {
                space,
                base,
                offset,
                src,
                ty,
            } => {
                let addr = self.effective_addr(t, pc, base, *offset)?;
                let v = self.operand_value(t, pc, src)?;
                let v = self.canon_stored(t, pc, *ty, operand_reg_bits(src), v)?;
                self.mem_write(t, pc, *space, addr, ty.size_bytes() as u64, v)?;
            }

            LoweredInstr::StoreVec {
                space,
                base,
                offset,
                src,
                ty,
            } => {
                let addr = self.effective_addr(t, pc, base, *offset)?;
                let width = ty.size_bytes() as u64;
                // As for `LoadVec`: a vector store is one access of the
                // total size, so its whole footprint must fit in the one
                // region owning its first byte, and its alignment is the
                // total size's.
                self.check_bounds(t, pc, *space, addr, src.len() as u64 * width)?;
                self.check_alignment(t, pc, *space, addr, src.len() as u64 * width)?;
                for (k, reg) in src.iter().enumerate() {
                    let v = self.read_reg(t, pc, *reg)?;
                    let v = self.canon_stored(t, pc, *ty, Some(reg_bits(*reg)), v)?;
                    self.mem_write(t, pc, *space, addr + k as u64 * width, width, v)?;
                }
            }

            LoweredInstr::Mov { dst, src, ty } => {
                let v = self.operand_value(t, pc, src)?;
                // Rebind the value at the mov's own type: `mov.u32 %r, -1`
                // must leave the same canonical constant in `%r` as
                // `not.b32 %r, 0` (consumers see the type-canonical value,
                // not the source operand's producer-typed rendering).
                let v = match v {
                    Value::Scalar(e) => Value::Scalar(self.canon_operand(*ty, e)),
                    pair @ Value::Pair(_, _) => pair,
                };
                self.threads[t].regs.write(*dst, v);
            }

            // Only `cvta.to.global` reaches evaluation (lowering rejects
            // every other cvta form): global addresses are absolute u64s
            // and the generic window over global is identity-mapped, so
            // the conversion is the identity.
            LoweredInstr::Cvta { dst, src, .. } => {
                let v = self.operand_value(t, pc, src)?;
                self.threads[t].regs.write(*dst, v);
            }

            LoweredInstr::BinOp {
                op,
                dst,
                src_a,
                src_b,
                ty,
                clamp,
            } => {
                let a = self.scalar_operand(t, pc, src_a)?;
                let b = self.scalar_operand(t, pc, src_b)?;
                let r = self.eval_binop(t, pc, *op, *ty, a, b)?;
                let r = self.apply_clamp(*clamp, r);
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::UnaryOp { op, dst, src, ty } => {
                let a = self.scalar_operand(t, pc, src)?;
                let r = self.eval_unop(pc, *op, *ty, a)?;
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::Fma {
                dst,
                src_a,
                src_b,
                src_c,
                clamp,
                ..
            } => {
                let a = self.scalar_operand(t, pc, src_a)?;
                let b = self.scalar_operand(t, pc, src_b)?;
                let c = self.scalar_operand(t, pc, src_c)?;
                let r = self.arena.fma(a, b, c);
                let r = self.apply_clamp(*clamp, r);
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::Mad {
                dst,
                src_a,
                src_b,
                src_c,
                ty,
                mode,
            } => {
                let a = self.scalar_operand(t, pc, src_a)?;
                let b = self.scalar_operand(t, pc, src_b)?;
                let c = self.scalar_operand(t, pc, src_c)?;
                let product = match mode {
                    crate::lowered::MulMode::Lo => self.eval_binop(t, pc, BinOp::Mul, *ty, a, b)?,
                    crate::lowered::MulMode::Wide => self.mul_wide(*ty, a, b),
                    crate::lowered::MulMode::Hi => {
                        // Same guard as MulHi: mul_hi composes `(a*b) >> bits`,
                        // which cannot represent the high half above 32 bits.
                        if ty.bits() > 32 {
                            return Err(EvalError::Unsupported {
                                pc,
                                what: format!("mad.hi at width {}", ty.bits()),
                            });
                        }
                        self.mul_hi(*ty, a, b)
                    }
                };
                let r = match mode {
                    crate::lowered::MulMode::Lo => {
                        self.eval_binop(t, pc, BinOp::Add, *ty, product, c)?
                    }
                    _ => self.arena.add(product, c),
                };
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::MulWide {
                dst,
                src_a,
                src_b,
                src_ty,
            } => {
                let a = self.scalar_operand(t, pc, src_a)?;
                let b = self.scalar_operand(t, pc, src_b)?;
                let r = self.mul_wide(*src_ty, a, b);
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::MulHi {
                dst,
                src_a,
                src_b,
                ty,
            } => {
                if ty.bits() > 32 {
                    return Err(EvalError::Unsupported {
                        pc,
                        what: format!("mul.hi at width {}", ty.bits()),
                    });
                }
                let a = self.scalar_operand(t, pc, src_a)?;
                let b = self.scalar_operand(t, pc, src_b)?;
                let r = self.mul_hi(*ty, a, b);
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::Bfi {
                dst,
                src_a,
                src_b,
                start,
                len,
                ..
            } => {
                let a = self.concrete_operand(t, pc, src_a, "bfi operand")?;
                let b = self.concrete_operand(t, pc, src_b, "bfi operand")?;
                let start = self.concrete_operand(t, pc, start, "bfi start")? as u64 & 0xff;
                let len = self.concrete_operand(t, pc, len, "bfi len")? as u64 & 0xff;
                let mask = if len >= 64 {
                    u64::MAX
                } else {
                    ((1u64 << len) - 1) << start.min(63)
                };
                let r = ((b as u64) & !mask) | (((a as u64) << start.min(63)) & mask);
                let r = self.arena.int(r as i64);
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::Setp {
                cmp,
                dst,
                src_a,
                src_b,
                ty,
            } => {
                let a = self.scalar_operand(t, pc, src_a)?;
                let b = self.scalar_operand(t, pc, src_b)?;
                let r = self.eval_cmp(pc, *cmp, *ty, a, b)?;
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::Selp {
                dst,
                src_a,
                src_b,
                pred,
                ty,
            } => {
                let a = self.scalar_operand(t, pc, src_a)?;
                let b = self.scalar_operand(t, pc, src_b)?;
                // Reinterpret concrete arms at the instruction type
                // before building the select: `selp.b32 %r, -1, 0, %p`
                // must export the same canonical 4294967295 a computed
                // operand would (see [`Self::canon_operand`]).
                let a = self.canon_operand(*ty, a);
                let b = self.canon_operand(*ty, b);
                let cond = self.scalar_operand(t, pc, pred)?;
                let r = self.arena.select(cond, a, b);
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::Set { .. } => {
                return Err(EvalError::Unsupported {
                    pc,
                    what: "set (value-producing comparison)".to_string(),
                });
            }

            LoweredInstr::Cvt {
                dst,
                src,
                dst_ty,
                src_ty,
                clamp,
            } => {
                let a = self.scalar_operand(t, pc, src)?;
                let r = self.eval_cvt(pc, *dst_ty, *src_ty, a)?;
                let r = self.apply_clamp(*clamp, r);
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::Bra { target } => {
                next_pc = *target;
            }

            LoweredInstr::Ret | LoweredInstr::Exit => {
                self.threads[t].status = Status::Exited;
                return Ok(());
            }

            LoweredInstr::BarSync { barrier_id } => {
                self.stats.block_syncs += 1;
                self.threads[t].status = Status::AtBarrier { id: *barrier_id };
                return Ok(()); // pc advances when the barrier fires
            }

            LoweredInstr::BarWarpSync { mask } => {
                let mask = self.concrete_operand(t, pc, mask, "warp sync mask")? as u32;
                self.block_at_warp_op(t, pc, mask)?;
                return Ok(());
            }

            LoweredInstr::Membar { .. } | LoweredInstr::Nop => {}

            LoweredInstr::Shfl { .. } => {
                return Err(EvalError::Unsupported {
                    pc,
                    what: "shfl without .sync (deprecated warp-unsynchronized shuffle)".to_string(),
                });
            }

            LoweredInstr::ShflSync { membermask, .. } => {
                let mask = self.concrete_operand(t, pc, membermask, "shfl.sync membermask")? as u32;
                self.block_at_warp_op(t, pc, mask)?;
                return Ok(());
            }

            // Tensor-core operations synchronize the full warp.
            LoweredInstr::Ldmatrix { .. }
            | LoweredInstr::Mma { .. }
            | LoweredInstr::WmmaLoad { .. }
            | LoweredInstr::WmmaStore { .. }
            | LoweredInstr::WmmaMma { .. } => {
                self.block_at_warp_op(t, pc, u32::MAX)?;
                return Ok(());
            }

            LoweredInstr::Activemask { dst } => {
                // The OR of `1 << lane` over the executing thread's warp
                // lanes that exist in the CTA and have not exited (ISA
                // 9.7.13.11: an "exited or inactive or predicated-off
                // thread will contribute 0"). Predication and divergence
                // are deliberately unmodeled - the per-thread interpreter
                // runs every thread's full straight-line program - so this
                // is exact for the converged case. Which lanes have exited
                // when a given thread executes activemask depends on the
                // round-robin schedule, just as it depends on timing on
                // real hardware; no cross-thread agreement is implied.
                let warp_base = (t.0 / WARP_SIZE) * WARP_SIZE;
                let mut mask: u32 = 0;
                for lane in 0..WARP_SIZE {
                    let tid = warp_base + lane;
                    if tid < self.n_threads && self.threads[ThreadId(tid)].status != Status::Exited
                    {
                        mask |= 1 << lane;
                    }
                }
                let r = self.arena.int(mask as i64);
                self.threads[t].regs.write(*dst, Value::Scalar(r));
            }

            LoweredInstr::Trap => {
                return Err(EvalError::TrapReached { thread: t, pc });
            }
        }

        self.threads[t].pc = next_pc;
        Ok(())
    }

    /// Block `t` at a warp-cooperative instruction with the given lane mask.
    fn block_at_warp_op(&mut self, t: ThreadId, pc: InstrId, mask: u32) -> EvalResult<()> {
        if mask == 0 {
            return Err(EvalError::WarpMismatch {
                pc,
                reason: "empty lane mask".to_string(),
            });
        }
        let lane = t.0 % WARP_SIZE;
        if mask & (1 << lane) == 0 {
            return Err(EvalError::WarpMismatch {
                pc,
                reason: format!("executing lane {} is not in mask {:#010x}", lane, mask),
            });
        }
        self.threads[t].status = Status::AtWarpOp { mask };
        Ok(())
    }

    // =====================================================================
    // Operand and register access
    // =====================================================================

    /// Read a register. A never-written register reads as `Undefined`
    /// rather than erroring: nvcc emits reads of dead uninitialized values
    /// (e.g. the accumulator-init idiom `selp.f32 %f, 0.0, %f, %p` on the
    /// first loop iteration). The undefined value is an error only if it
    /// reaches an output or a point that requires a concrete value.
    pub(in crate::eval) fn read_reg(
        &self,
        t: ThreadId,
        _pc: InstrId,
        reg: RegId,
    ) -> EvalResult<Value> {
        Ok(self.threads[t]
            .regs
            .read(reg)
            .unwrap_or(Value::Scalar(self.undefined)))
    }

    /// Resolve an operand to a runtime value.
    pub(in crate::eval) fn operand_value(
        &mut self,
        t: ThreadId,
        pc: InstrId,
        op: &Operand,
    ) -> EvalResult<Value> {
        match op {
            Operand::Reg(reg) => self.read_reg(t, pc, *reg),
            Operand::SpecialReg(kind) => {
                let v = self.special_reg(t, pc, *kind)?;
                Ok(Value::Scalar(self.arena.int(v)))
            }
            Operand::ImmI64(v) => Ok(Value::Scalar(self.arena.int(*v))),
            Operand::ImmU64(v) => Ok(Value::Scalar(self.arena.int(*v as i64))),
            // Exact ingestion. Lowering rejects NaN literals, so this only
            // fails if an unvetted immediate slips past it - loudly.
            Operand::ImmF64(v) => match self.arena.float_from_f64(*v) {
                Ok(e) => Ok(Value::Scalar(e)),
                Err(err) => Err(EvalError::Unsupported {
                    pc,
                    what: format!("float immediate: {}", err),
                }),
            },
        }
    }

    /// Resolve an operand that must be a scalar.
    pub(in crate::eval) fn scalar_operand(
        &mut self,
        t: ThreadId,
        pc: InstrId,
        op: &Operand,
    ) -> EvalResult<ExprId> {
        match self.operand_value(t, pc, op)? {
            Value::Scalar(e) => Ok(e),
            Value::Pair(_, _) => Err(EvalError::ValueKindMismatch {
                thread: t,
                pc,
                what: "packed pair used as a scalar",
            }),
        }
    }

    /// Resolve an operand that must be a concrete integer.
    pub(in crate::eval) fn concrete_operand(
        &mut self,
        t: ThreadId,
        pc: InstrId,
        op: &Operand,
        what: &'static str,
    ) -> EvalResult<i64> {
        let e = self.scalar_operand(t, pc, op)?;
        self.arena.as_i64(e).ok_or(EvalError::NotConcrete {
            thread: t,
            pc,
            what,
        })
    }

    fn as_concrete_bool(
        &self,
        t: ThreadId,
        pc: InstrId,
        value: Value,
        what: &'static str,
    ) -> EvalResult<bool> {
        let Value::Scalar(e) = value else {
            return Err(EvalError::ValueKindMismatch {
                thread: t,
                pc,
                what: "packed pair used as a predicate",
            });
        };
        self.arena.as_bool(e).ok_or(EvalError::NotConcrete {
            thread: t,
            pc,
            what,
        })
    }

    /// The (x, y, z) thread indices of a linear thread id.
    fn thread_coords(&self, t: ThreadId) -> (u32, u32, u32) {
        let (bx, by, _) = self.config.block_dim;
        (t.0 % bx, (t.0 / bx) % by, t.0 / (bx * by))
    }

    fn special_reg(&self, t: ThreadId, pc: InstrId, kind: SpecialRegKind) -> EvalResult<i64> {
        let (x, y, z) = self.thread_coords(t);
        let v = match kind {
            SpecialRegKind::TidX => x as i64,
            SpecialRegKind::TidY => y as i64,
            SpecialRegKind::TidZ => z as i64,
            SpecialRegKind::NtidX => self.config.block_dim.0 as i64,
            SpecialRegKind::NtidY => self.config.block_dim.1 as i64,
            SpecialRegKind::NtidZ => self.config.block_dim.2 as i64,
            // The CTA under analysis is always block (0,0,0) (paper: CTAs
            // are checked pairwise at block 0).
            SpecialRegKind::CtaidX | SpecialRegKind::CtaidY | SpecialRegKind::CtaidZ => 0,
            SpecialRegKind::NctaidX => self.config.grid_dim.0 as i64,
            SpecialRegKind::NctaidY => self.config.grid_dim.1 as i64,
            SpecialRegKind::NctaidZ => self.config.grid_dim.2 as i64,
            SpecialRegKind::LaneId => (t.0 % WARP_SIZE) as i64,
            SpecialRegKind::WarpId => (t.0 / WARP_SIZE) as i64,
            SpecialRegKind::NWarpId => self.n_threads.div_ceil(WARP_SIZE) as i64,
            SpecialRegKind::DynamicSmemSize => self.config.dynamic_shared_bytes as i64,
            other => {
                return Err(EvalError::Unsupported {
                    pc,
                    what: format!("special register {}", other.as_str()),
                });
            }
        };
        Ok(v)
    }

    // =====================================================================
    // Memory access
    // =====================================================================

    /// Compute the concrete effective address `base + offset`.
    ///
    /// Hardware semantics: the register holds a 64-bit address and the
    /// immediate is a two's-complement byte offset, so the `[reg + imm]`
    /// sum is u64 arithmetic mod 2^64. A wrapped sum is not itself an
    /// error - it is simply an address, and unless a declared region owns
    /// it the ownership bounds check (`check_bounds`, subtraction-form,
    /// wrap-proof) rejects it loudly as `OutOfBounds` in every build
    /// profile (see `test_negative_index_wrap_is_out_of_bounds`). A
    /// checked i64 sum here would guard the wrong boundary: it rejects
    /// valid accesses that merely cross 2^63 (an array based just below
    /// the sign bit) while letting genuine u64 wraps through untouched.
    pub(in crate::eval) fn effective_addr(
        &mut self,
        t: ThreadId,
        pc: InstrId,
        base: &Operand,
        offset: i64,
    ) -> EvalResult<u64> {
        let base = self.concrete_operand(t, pc, base, "memory address")?;
        Ok(Self::offset_addr(base, offset))
    }

    /// `base + offset` for an already-concrete base: `effective_addr`'s
    /// arithmetic (see its doc for the wrap semantics), shared with the
    /// warp-cooperative loads/stores, which obtain their base per lane or
    /// warp-uniformly before applying the instruction's offset.
    pub(in crate::eval) fn offset_addr(base: i64, offset: i64) -> u64 {
        (base as u64).wrapping_add(offset as u64)
    }

    /// Ownership containment: the region owning the access's *first byte*
    /// must contain the whole access; an access whose first byte no region
    /// owns is out of bounds. Anchoring at the first byte's owner makes an
    /// access that starts inside one array and runs past its end a loud
    /// `OutOfBounds` even when the trailing bytes land inside an adjacent
    /// array (the paper's §6.2 point: hardware happens to tolerate
    /// out-of-bounds shared reads, the model must not). Regions never
    /// overlap - config validation keeps arrays pairwise disjoint, the
    /// symbol-table packer keeps each space's variables disjoint, and
    /// `Interpreter::new` rejects arrays overlapping the module-global
    /// window - so the owner is unique; `find` keeps the answer
    /// deterministic regardless.
    fn check_bounds(
        &self,
        t: ThreadId,
        pc: InstrId,
        space: MemSpace,
        addr: u64,
        width: u64,
    ) -> EvalResult<()> {
        let regions = match space {
            MemSpace::Global => &self.regions.global,
            MemSpace::Shared => &self.regions.shared,
            MemSpace::Local => &self.regions.local,
            MemSpace::Param | MemSpace::Const => {
                return Err(EvalError::Unsupported {
                    pc,
                    what: format!("{:?}-space memory access", space),
                });
            }
        };
        match regions.iter().find(|r| r.owns(addr)) {
            Some(owner) if owner.contains(addr, width) => Ok(()),
            _ => Err(EvalError::OutOfBounds {
                thread: t,
                pc,
                space,
                addr,
                width,
            }),
        }
    }

    /// Natural-alignment check. PTX ISA 6.4.1: "The address must be
    /// naturally aligned to a multiple of the access size. If an address is
    /// not properly aligned, the resulting behavior is undefined". A
    /// misaligned kernel has no defined hardware semantics to model, so the
    /// access is rejected loudly instead. Addresses are always concrete
    /// here, so the check is a single modulo.
    ///
    /// `required` is the access size for scalar loads/stores (`mem_read`/
    /// `mem_write` check every access at its granule width), the *total*
    /// size for vector accesses (checked once at the `LoadVec`/`StoreVec`
    /// sites; the per-element checks below them are implied), and the
    /// row/fragment alignment for the tensor-core cooperative ops (checked
    /// at the `ldmatrix`/`wmma` sites in `eval::warp`).
    pub(in crate::eval) fn check_alignment(
        &self,
        t: ThreadId,
        pc: InstrId,
        space: MemSpace,
        addr: u64,
        required: u64,
    ) -> EvalResult<()> {
        if addr.is_multiple_of(required) {
            Ok(())
        } else {
            Err(EvalError::Misaligned {
                thread: t,
                pc,
                space,
                addr,
                required,
            })
        }
    }

    fn mem_error(&self, t: ThreadId, pc: InstrId, space: MemSpace, e: MemAccessError) -> EvalError {
        match e {
            MemAccessError::Uninitialized { addr } => EvalError::UninitializedMemory {
                thread: t,
                pc,
                space,
                addr,
            },
            MemAccessError::Reinterpret { addr, width } => EvalError::Reinterpretation {
                thread: t,
                pc,
                space,
                addr,
                width,
            },
        }
    }

    /// Bounds-check, race-check, and read memory.
    ///
    /// Reading in-bounds shared/global bytes that were never written yields
    /// `Undefined` rather than an error: the read is still recorded in χ, so
    /// a later conflicting write is reported as a race (this is exactly the
    /// paper's motivating example, where thread 0 reads `buf[1]` before
    /// thread 1 has written it). The undefined value is an error only if it
    /// reaches an output or a point that requires a concrete value.
    pub(in crate::eval) fn mem_read(
        &mut self,
        t: ThreadId,
        pc: InstrId,
        space: MemSpace,
        addr: u64,
        width: u64,
    ) -> EvalResult<Value> {
        self.check_bounds(t, pc, space, addr, width)?;
        // Every program access flows through here (scalar ld/st directly;
        // vector and tensor-core ops per element, after their own
        // larger-granule checks), so this is the one natural-alignment
        // chokepoint. Param space never reaches it: `check_bounds` rejects
        // Param/Const accesses, and `LoadParam` reads interpreter-internal
        // parameter bindings, not byte-addressed memory.
        self.check_alignment(t, pc, space, addr, width)?;
        let memory = match space {
            MemSpace::Global | MemSpace::Shared => {
                self.race
                    .read(space, addr, width, t, pc)
                    .map_err(|race| EvalError::DataRace {
                        space: race.space,
                        addr: race.addr,
                        prior: race.prior,
                        current: race.current,
                    })?;
                if space == MemSpace::Global {
                    &self.global
                } else {
                    &self.shared
                }
            }
            MemSpace::Local => &self.locals[t],
            _ => unreachable!("bounds check rejects other spaces"),
        };
        match memory.read(addr, width) {
            Ok(v) => Ok(v),
            Err(MemAccessError::Uninitialized { .. }) if space == MemSpace::Global => {
                // Reading an input array materializes its symbols on demand.
                if self.materialize_input(addr, width) {
                    self.global
                        .read(addr, width)
                        .map_err(|e| self.mem_error(t, pc, space, e))
                } else {
                    Ok(Value::Scalar(self.arena.undefined()))
                }
            }
            Err(MemAccessError::Uninitialized { .. }) if space == MemSpace::Shared => {
                Ok(Value::Scalar(self.arena.undefined()))
            }
            Err(e) => Err(self.mem_error(t, pc, space, e)),
        }
    }

    /// Create the input-element symbols for every input-array element
    /// overlapping `[addr, addr + width)` that is not yet present in
    /// global memory. Returns whether any element was materialized.
    fn materialize_input(&mut self, addr: u64, width: u64) -> bool {
        // Collect missing elements first (the array list borrows the config).
        // (addr, width, index, interned array name or None for identity
        // indices); the array's name is interned once and shared by all of
        // its elements.
        let mut missing: Vec<(u64, u64, u64, Option<StringId>)> = Vec::new();
        for array in &self.config.arrays {
            if !array.kind.is_input() {
                continue;
            }
            // Overlap of [addr, addr+width) with [base, base+size), in
            // subtraction form so neither sum is formed: the intervals
            // overlap iff each start lies short of the other end.
            let size = array.size_bytes();
            let overlaps = if addr >= array.base {
                addr - array.base < size
            } else {
                array.base - addr < width
            };
            if !overlaps {
                continue;
            }
            // Both sums below are exact: the access was bounds-checked
            // (`addr + width` fits inside its owning region) and the
            // array's end fits in u64 (`AnalysisConfig::validate`), so
            // with width >= 1 neither `addr + width - 1` nor
            // `base + size - 1` can wrap.
            let end = array.base + size;
            let first = (addr.max(array.base) - array.base) / array.elem_width;
            let last = ((addr + width - 1).min(end - 1) - array.base) / array.elem_width;
            let mut array_sid: Option<StringId> = None;
            for i in first..=last {
                let elem_addr = array.base + i * array.elem_width;
                if !self.global.has_cell_at(elem_addr) {
                    let value = match array.kind {
                        crate::eval::config::ArrayKind::IndexInput => None,
                        _ => Some(
                            *array_sid.get_or_insert_with(|| self.arena.intern_string(&array.name)),
                        ),
                    };
                    missing.push((elem_addr, array.elem_width, i, value));
                }
            }
        }

        // Analysis-setup placement (not a PTX access, so no alignment check
        // applies): each granule lands at `base + i*elem_width`, naturally
        // aligned because `AnalysisConfig::validate` requires every array
        // base to be a multiple of its element width.
        let mut any = false;
        for (elem_addr, elem_width, index, array_sid) in missing {
            let value = match array_sid {
                Some(sid) => self.arena.input_element(sid, index),
                // Identity index array: element i holds the value i.
                None => self.arena.int(index as i64),
            };
            if self
                .global
                .init(elem_addr, elem_width, Value::Scalar(value))
                .is_ok()
            {
                any = true;
            }
        }
        any
    }

    /// Bounds-check, race-check, and write memory.
    pub(in crate::eval) fn mem_write(
        &mut self,
        t: ThreadId,
        pc: InstrId,
        space: MemSpace,
        addr: u64,
        width: u64,
        value: Value,
    ) -> EvalResult<()> {
        self.check_bounds(t, pc, space, addr, width)?;
        // See `mem_read`: the write-side natural-alignment chokepoint.
        self.check_alignment(t, pc, space, addr, width)?;
        let memory = match space {
            MemSpace::Global | MemSpace::Shared => {
                self.race
                    .write(space, addr, width, t, pc)
                    .map_err(|race| EvalError::DataRace {
                        space: race.space,
                        addr: race.addr,
                        prior: race.prior,
                        current: race.current,
                    })?;
                if space == MemSpace::Global {
                    &mut self.global
                } else {
                    &mut self.shared
                }
            }
            MemSpace::Local => &mut self.locals[t],
            _ => unreachable!("bounds check rejects other spaces"),
        };
        memory
            .write(addr, width, value)
            .map_err(|e| self.mem_error(t, pc, space, e))
    }

    // =====================================================================
    // Arithmetic
    // =====================================================================

    /// Evaluate a binary op. Integer ops on concrete values use exact
    /// width/signedness semantics; symbolic values get real-valued nodes.
    fn eval_binop(
        &mut self,
        t: ThreadId,
        pc: InstrId,
        op: BinOp,
        ty: ScalarType,
        a: ExprId,
        b: ExprId,
    ) -> EvalResult<ExprId> {
        if ty.is_predicate() {
            return self.eval_pred_binop(pc, op, a, b);
        }

        if !ty.is_float()
            && let (Some(ca), Some(cb)) = (self.arena.as_i64(a), self.arena.as_i64(b))
        {
            let r = self.concrete_int_binop(t, pc, op, ty, ca, cb)?;
            return Ok(self.arena.int(r));
        }

        // One operand concrete, the other symbolic: reinterpret the
        // concrete side at the instruction type before building the node,
        // so `add.u32 %r, %sym, -1` and a chain that produced 4294967295
        // build identical expressions (the register/immediate rendering
        // is the producer's, not this instruction's).
        let a = self.canon_operand(ty, a);
        let b = self.canon_operand(ty, b);

        Ok(match op {
            BinOp::Add => self.arena.add(a, b),
            BinOp::Sub => self.arena.sub(a, b),
            BinOp::Mul => self.arena.mul(a, b),
            BinOp::Div => self.arena.div(a, b),
            BinOp::Rem => self.arena.rem(a, b),
            BinOp::And => self.arena.bit_and(a, b),
            BinOp::Or => self.arena.bit_or(a, b),
            BinOp::Xor => self.arena.bit_xor(a, b),
            BinOp::Shl => self.arena.shl(a, b),
            BinOp::Shr => {
                if ty.is_signed_int() {
                    self.arena.shr(a, b)
                } else {
                    self.arena.lshr(a, b)
                }
            }
            BinOp::Min => self.arena.min(a, b),
            BinOp::Max => self.arena.max(a, b),
        })
    }

    /// Exact concrete integer semantics for `ty`.
    fn concrete_int_binop(
        &self,
        t: ThreadId,
        pc: InstrId,
        op: BinOp,
        ty: ScalarType,
        a: i64,
        b: i64,
    ) -> EvalResult<i64> {
        let bits = ty.bits().min(64);
        let signed = ty.is_signed_int();
        let ua = mask_to(a, bits);
        let ub = mask_to(b, bits);
        let sa = canon_int(a, bits, true);
        let sb = canon_int(b, bits, true);

        let raw: u64 = match op {
            BinOp::Add => ua.wrapping_add(ub),
            BinOp::Sub => ua.wrapping_sub(ub),
            BinOp::Mul => ua.wrapping_mul(ub),
            BinOp::Div => {
                if ub == 0 {
                    return Err(EvalError::Unsupported {
                        pc,
                        what: format!("division by zero (thread {})", t),
                    });
                }
                if signed {
                    sa.wrapping_div(sb) as u64
                } else {
                    ua / ub
                }
            }
            BinOp::Rem => {
                if ub == 0 {
                    return Err(EvalError::Unsupported {
                        pc,
                        what: format!("remainder by zero (thread {})", t),
                    });
                }
                if signed {
                    sa.wrapping_rem(sb) as u64
                } else {
                    ua % ub
                }
            }
            BinOp::And => ua & ub,
            BinOp::Or => ua | ub,
            BinOp::Xor => ua ^ ub,
            // PTX shifts clamp: shifting by >= width produces 0 (or the sign
            // fill for arithmetic right shift).
            BinOp::Shl => {
                if ub >= bits as u64 {
                    0
                } else {
                    ua << ub
                }
            }
            BinOp::Shr => {
                if signed {
                    let sh = ub.min(bits as u64 - 1);
                    (sa >> sh) as u64
                } else if ub >= bits as u64 {
                    0
                } else {
                    ua >> ub
                }
            }
            BinOp::Min => {
                if signed {
                    sa.min(sb) as u64
                } else {
                    ua.min(ub)
                }
            }
            BinOp::Max => {
                if signed {
                    sa.max(sb) as u64
                } else {
                    ua.max(ub)
                }
            }
        };
        Ok(canon_int(raw as i64, bits, signed))
    }

    /// Boolean (predicate) binary ops.
    fn eval_pred_binop(
        &mut self,
        pc: InstrId,
        op: BinOp,
        a: ExprId,
        b: ExprId,
    ) -> EvalResult<ExprId> {
        if let (Some(ca), Some(cb)) = (self.arena.as_bool(a), self.arena.as_bool(b)) {
            let r = match op {
                BinOp::And => ca && cb,
                BinOp::Or => ca || cb,
                BinOp::Xor => ca != cb,
                _ => {
                    return Err(EvalError::Unsupported {
                        pc,
                        what: format!("{} on predicates", op.as_str()),
                    });
                }
            };
            return Ok(self.arena.bool_val(r));
        }
        Ok(match op {
            BinOp::And => self.arena.and(a, b),
            BinOp::Or => self.arena.or(a, b),
            // Boolean xor is inequality.
            BinOp::Xor => self.arena.ne(a, b),
            _ => {
                return Err(EvalError::Unsupported {
                    pc,
                    what: format!("{} on predicates", op.as_str()),
                });
            }
        })
    }

    fn eval_unop(
        &mut self,
        pc: InstrId,
        op: UnaryOp,
        ty: ScalarType,
        a: ExprId,
    ) -> EvalResult<ExprId> {
        Ok(match op {
            UnaryOp::Neg => self.arena.neg(a),
            UnaryOp::Abs => self.arena.abs(a),
            UnaryOp::Not => {
                if ty.is_predicate() {
                    if let Some(c) = self.arena.as_bool(a) {
                        self.arena.bool_val(!c)
                    } else {
                        self.arena.not(a)
                    }
                } else {
                    // Bitwise not; folds when concrete (via canonical i64).
                    let bits = ty.bits().min(64);
                    if let Some(c) = self.arena.as_i64(a) {
                        let r = canon_int(!c, bits, ty.is_signed_int());
                        self.arena.int(r)
                    } else {
                        self.arena.bit_not(a)
                    }
                }
            }
            UnaryOp::Rcp => self.arena.rcp(a),
            UnaryOp::Sqrt => self.arena.sqrt(a),
            UnaryOp::Rsqrt => {
                let s = self.arena.sqrt(a);
                self.arena.rcp(s)
            }
            UnaryOp::Exp => self.arena.exp(a),
            UnaryOp::Ex2 | UnaryOp::Lg2 | UnaryOp::Sin | UnaryOp::Cos => {
                return Err(EvalError::Unsupported {
                    pc,
                    what: format!("transcendental {}", op.as_str()),
                });
            }
        })
    }

    /// Reinterpret a concrete operand as canonical for `ty`. Registers and
    /// memory granules hold values canonicalized by their *producing*
    /// instruction, so a value written as signed may be consumed as
    /// unsigned (or vice versa): nvcc emits `mul.wide.u16 %r, %rs, -17873`
    /// where the immediate is really the u16 magic constant 47663. Every
    /// consumer that gives a concrete integer a type of its own re-reads
    /// the value through this (mov, cvt sources, wide/hi multiplies,
    /// binops with a symbolic side; loads and stores go through
    /// [`Self::canon_loaded`]/[`Self::canon_stored`]). Symbolic operands
    /// pass through unchanged, as do float and bool constants (a
    /// `mov.b32 %r, %f` bit-move must not coerce the float to an int).
    fn canon_operand(&mut self, ty: ScalarType, e: ExprId) -> ExprId {
        if ty.is_float() || ty.is_predicate() {
            return e;
        }
        if let Some(c) = self.arena.as_int_const(e) {
            let canon = canon_int(c, ty.bits().min(64), ty.is_signed_int());
            if canon != c {
                return self.arena.int(canon);
            }
        }
        e
    }

    /// Canonicalize a value crossing a store boundary. Memory holds bit
    /// patterns: a concrete integer is reduced to the unsigned low bits of
    /// the store type (`st.u8` of 300 stores 44); sign/zero extension is
    /// the *load*'s job (see [`Self::canon_loaded`]). Floats, `Undefined`,
    /// and packed pairs at their full 4-byte width pass through unchanged.
    ///
    /// A *symbolic* integer stored below its source register's width would
    /// need a truncation node we deliberately do not model, so that store
    /// is a loud error rather than a silently unsound pass-through.
    /// Equal-width symbolic stores are exact and pass through (an f16 half
    /// stored from a 16-bit register via `st.u16` - the corpus's only
    /// symbolic sub-word stores). Likewise a packed f16x2 pair stored
    /// below 4 bytes would stuff the whole two-half value into a narrower
    /// granule - a shape the memory model never anticipates - so it too
    /// is a loud error (its truncation would be a half extraction we do
    /// not model at store boundaries).
    fn canon_stored(
        &mut self,
        t: ThreadId,
        pc: InstrId,
        ty: ScalarType,
        src_reg_bits: Option<u32>,
        v: Value,
    ) -> EvalResult<Value> {
        if matches!(v, Value::Pair(..)) && ty.size_bytes() < 4 {
            return Err(EvalError::Unsupported {
                pc,
                what: format!(
                    "packed f16x2 pair stored at {}-bit width (thread {})",
                    ty.bits(),
                    t
                ),
            });
        }
        if ty.is_float() || ty.is_predicate() {
            return Ok(v);
        }
        let Value::Scalar(e) = v else {
            return Ok(v); // packed f16 pairs at full width
        };
        if let Some(c) = self.arena.as_int_const(e) {
            let canon = canon_int(c, ty.bits().min(64), false);
            return Ok(if canon == c {
                v
            } else {
                Value::Scalar(self.arena.int(canon))
            });
        }
        if let Some(reg_bits) = src_reg_bits
            && ty.bits() < reg_bits
            && !self.arena.is_undefined(e)
            && !self.arena.is_concrete(e)
        {
            return Err(EvalError::Unsupported {
                pc,
                what: format!(
                    "symbolic value stored at sub-register width \
                     ({}-bit store of a {}-bit register, thread {})",
                    ty.bits(),
                    reg_bits,
                    t
                ),
            });
        }
        Ok(v)
    }

    /// Canonicalize a value crossing a load boundary: `ld` extends the
    /// memory pattern to the destination register per the *load type* -
    /// sign-extension for `.s8`/`.s16`/..., zero-extension for unsigned
    /// and bits types (the ISA's ld extension rules) - so `ld.s8` of the
    /// byte 0xFF yields -1 while `ld.u8` yields 255. Floats, packed
    /// pairs, and `Undefined` pass through unchanged.
    ///
    /// A *symbolic* scalar loaded at a type narrower than the destination
    /// register would need an extension node we deliberately do not model:
    /// loud error. Equal-width symbolic loads are exact and pass through
    /// (f16 halves loaded into 16-bit registers).
    fn canon_loaded(
        &mut self,
        t: ThreadId,
        pc: InstrId,
        ty: ScalarType,
        dst: RegId,
        v: Value,
    ) -> EvalResult<Value> {
        if ty.is_float() || ty.is_predicate() {
            return Ok(v);
        }
        let Value::Scalar(e) = v else {
            return Ok(v); // packed f16 pairs
        };
        if let Some(c) = self.arena.as_int_const(e) {
            let canon = canon_int(c, ty.bits().min(64), ty.is_signed_int());
            return Ok(if canon == c {
                v
            } else {
                Value::Scalar(self.arena.int(canon))
            });
        }
        let dst_bits = reg_bits(dst);
        if ty.bits() < dst_bits && !self.arena.is_undefined(e) && !self.arena.is_concrete(e) {
            return Err(EvalError::Unsupported {
                pc,
                what: format!(
                    "symbolic value loaded at sub-register width \
                     ({}-bit load into a {}-bit register, thread {})",
                    ty.bits(),
                    dst_bits,
                    t
                ),
            });
        }
        Ok(v)
    }

    /// Widening product: operands are reinterpreted at the source type, and
    /// the product is exact in the 2x-wide destination type.
    fn mul_wide(&mut self, src_ty: ScalarType, a: ExprId, b: ExprId) -> ExprId {
        let a = self.canon_operand(src_ty, a);
        let b = self.canon_operand(src_ty, b);
        self.arena.mul(a, b)
    }

    /// High half of the widening product (nvcc's divide-by-constant idiom).
    /// Composed from existing nodes so it works symbolically and folds when
    /// concrete: `(a * b) >> bits`.
    fn mul_hi(&mut self, ty: ScalarType, a: ExprId, b: ExprId) -> ExprId {
        let a = self.canon_operand(ty, a);
        let b = self.canon_operand(ty, b);
        let bits = self.arena.int(ty.bits().min(64) as i64);
        let product = self.arena.mul(a, b);
        if ty.is_signed_int() {
            self.arena.shr(product, bits)
        } else {
            self.arena.lshr(product, bits)
        }
    }

    fn eval_cmp(
        &mut self,
        _pc: InstrId,
        cmp: CmpOp,
        ty: ScalarType,
        a: ExprId,
        b: ExprId,
    ) -> EvalResult<ExprId> {
        // Concrete integer comparisons need width/signedness care
        // (`setp.lt.u32` on canonical values would misorder negatives).
        if !ty.is_float()
            && let (Some(ca), Some(cb)) = (self.arena.as_i64(a), self.arena.as_i64(b))
        {
            let bits = ty.bits().min(64);
            let unsigned_cmp = matches!(cmp, CmpOp::Lo | CmpOp::Ls | CmpOp::Hi | CmpOp::Hs)
                || ty.is_unsigned_int()
                || ty.is_bits_type();
            let r = if unsigned_cmp {
                let (ua, ub) = (mask_to(ca, bits), mask_to(cb, bits));
                match cmp {
                    CmpOp::Eq | CmpOp::Equ => ua == ub,
                    CmpOp::Ne | CmpOp::Neu => ua != ub,
                    CmpOp::Lt | CmpOp::Lo | CmpOp::Ltu => ua < ub,
                    CmpOp::Le | CmpOp::Ls | CmpOp::Leu => ua <= ub,
                    CmpOp::Gt | CmpOp::Hi | CmpOp::Gtu => ua > ub,
                    CmpOp::Ge | CmpOp::Hs | CmpOp::Geu => ua >= ub,
                    CmpOp::Num => true,
                    CmpOp::Nan => false,
                }
            } else {
                let (sa, sb) = (canon_int(ca, bits, true), canon_int(cb, bits, true));
                match cmp {
                    CmpOp::Eq | CmpOp::Equ => sa == sb,
                    CmpOp::Ne | CmpOp::Neu => sa != sb,
                    CmpOp::Lt | CmpOp::Ltu => sa < sb,
                    CmpOp::Le | CmpOp::Leu => sa <= sb,
                    CmpOp::Gt | CmpOp::Gtu => sa > sb,
                    CmpOp::Ge | CmpOp::Geu => sa >= sb,
                    CmpOp::Lo | CmpOp::Ls | CmpOp::Hi | CmpOp::Hs => {
                        unreachable!("unsigned comparisons handled above")
                    }
                    CmpOp::Num => true,
                    CmpOp::Nan => false,
                }
            };
            return Ok(self.arena.bool_val(r));
        }

        // Symbolic: reinterpret a concrete side at the instruction type
        // first, exactly like eval_binop's fallback, so
        // `setp.eq.s32 %p, %sym, -1` and a compare against a
        // chain-computed 4294967295 build identical nodes. Over the
        // reals there are no NaNs, so unordered comparisons coincide
        // with their ordered counterparts.
        let a = self.canon_operand(ty, a);
        let b = self.canon_operand(ty, b);
        Ok(match cmp {
            CmpOp::Eq | CmpOp::Equ => self.arena.eq(a, b),
            CmpOp::Ne | CmpOp::Neu => self.arena.ne(a, b),
            CmpOp::Lt | CmpOp::Lo | CmpOp::Ltu => self.arena.lt(a, b),
            CmpOp::Le | CmpOp::Ls | CmpOp::Leu => self.arena.le(a, b),
            CmpOp::Gt | CmpOp::Hi | CmpOp::Gtu => self.arena.gt(a, b),
            CmpOp::Ge | CmpOp::Hs | CmpOp::Geu => self.arena.ge(a, b),
            CmpOp::Num => self.arena.bool_val(true),
            CmpOp::Nan => self.arena.bool_val(false),
        })
    }

    /// Apply a float value clamp (`.sat`/`.relu`) to a result expression.
    ///
    /// Over the floats-as-reals model these are exact: `.relu` is
    /// `max(r, 0)` and `.sat` is `min(max(r, 0), 1)`. Concrete operands
    /// fold through the arena's min/max constant folding. The spec's
    /// `.sat` additionally flushes a NaN result to +0.0 (and cvt's
    /// `.relu` canonicalizes NaN); NaN is out of model over the reals,
    /// as everywhere else in the interpreter.
    fn apply_clamp(&mut self, clamp: Option<Clamp>, r: ExprId) -> ExprId {
        match clamp {
            None => r,
            Some(Clamp::Relu) => {
                let zero = self.arena.real(Real::zero());
                self.arena.max(r, zero)
            }
            Some(Clamp::Sat) => {
                let zero = self.arena.real(Real::zero());
                let one = self.arena.real(Real::one());
                let low_clamped = self.arena.max(r, zero);
                self.arena.min(low_clamped, one)
            }
        }
    }

    fn eval_cvt(
        &mut self,
        pc: InstrId,
        dst_ty: ScalarType,
        src_ty: ScalarType,
        a: ExprId,
    ) -> EvalResult<ExprId> {
        // Float-to-float conversions (f16 <-> f32 <-> f64) are the identity
        // over the reals; rounding is deliberately not modeled (paper).
        if dst_ty.is_float() && src_ty.is_float() {
            return Ok(a);
        }
        if src_ty.is_float() {
            return Err(EvalError::Unsupported {
                pc,
                what: format!("cvt float->int ({:?} -> {:?})", src_ty, dst_ty),
            });
        }
        // Integer source: cvt reads its source at the *source* format
        // first (ISA Table 15: "extension ... follows the source format"),
        // so a register canonicalized unsigned by its producer (`and.b32`
        // leaving 4294967288) reads as -8 under `cvt.s64.s32`. Symbolic
        // integers pass through (they are data, not addresses, so width
        // games cannot occur in a structured-CTA).
        let a = self.canon_operand(src_ty, a);
        if dst_ty.is_float() {
            return Ok(self.arena.to_float(a));
        }
        // ... then the result is renormalized at the destination width.
        if let Some(c) = self.arena.as_int_const(a) {
            let bits = dst_ty.bits().min(64);
            let r = canon_int(c, bits, dst_ty.is_signed_int());
            return Ok(self.arena.int(r));
        }
        Ok(a)
    }
}

/// Bit width of a register's storage class.
fn reg_bits(reg: RegId) -> u32 {
    (reg.class.size_bytes() * 8) as u32
}

/// Bit width of the register behind `op`, if it is a register operand.
/// Immediates and special registers resolve to concrete values, which
/// never trip the symbolic sub-register-width store policy.
fn operand_reg_bits(op: &Operand) -> Option<u32> {
    match op {
        Operand::Reg(r) => Some(reg_bits(*r)),
        _ => None,
    }
}

/// Zero-extend the low `bits` of `v` into a u64.
fn mask_to(v: i64, bits: u32) -> u64 {
    if bits >= 64 {
        v as u64
    } else {
        (v as u64) & ((1u64 << bits) - 1)
    }
}

/// Canonicalize the low `bits` of `v`: sign-extended if `signed`, else
/// zero-extended.
fn canon_int(v: i64, bits: u32, signed: bool) -> i64 {
    if bits >= 64 {
        return v;
    }
    let masked = mask_to(v, bits);
    if signed {
        let shift = 64 - bits;
        ((masked << shift) as i64) >> shift
    } else {
        masked as i64
    }
}
