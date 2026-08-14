# Volta Architecture Documentation

**Purpose**: Codebase context for AI-assisted development

## Overview

Volta is an abstract interpreter for NVIDIA PTX kernels, implementing the approach from "Equivalence Checking of ML GPU Kernels" (arXiv:2511.12638). It detects data races and verifies kernel equivalence.

## Coding practices

- Take advantage of the type system to ensure correctness. E.g., rather than using a `u32`, to avoid mixing up different kinds of indices, consider using a new type that wraps a `u32`. Likewise, consider where it is better to use a custom, two variant enum in place of a `bool`.
- Shared state makes reasoning about code complex. Err on the side of slightly less efficient but pure implementations.

## Crate Structure

```
crates/
├── volta_common/     # Base utilities (spans, file caching, error reporting, run logs)
├── volta_frontend/   # PTX lexer and parser
├── volta_analysis/   # Abstract interpreter
├── volta_z3/         # Z3 comparison backend (SMT-LIB2 via linked libz3)
├── volta_bench/      # Paper-evaluation benchmark harness
└── volta_cli/        # Command-line interface
```

### Dependency Graph

Direct dependencies of each crate:

```
volta_cli      → volta_z3, volta_analysis, volta_frontend, volta_common
volta_bench    → volta_z3, volta_analysis, volta_frontend, volta_common
volta_z3       → volta_analysis
volta_analysis → volta_frontend, volta_common
volta_frontend → volta_common
volta_common   → (nothing)
```

## Crate: volta_common

**Path**: `crates/volta_common/`

- `Span` - Source location (low + high byte offset)
- `FileCache` - Caches file content to make sure we always use a consistent version of each file
- `Locate<E>` - Error wrapper with optional location info (span + file path)
- `report_error` - Produces an error message from a title, message, span, and file content. Extracts out and includes the code snippet at the given span in the given file content
- `run_log::RunLog` - Per-invocation log file (`<unix-seconds>-<pid>-<command>.log` under `--log-dir`, default `volta-logs/`), shared by the `volta` and `volta-bench` binaries; `tee` mirrors an `env_logger` target into it under the binaries' `logging` features

The pattern is to create an error kind type, and then an alias for `Locate` of that error kind. `locate_span` can be used to tag a `Locate` with a span if it does not already have one. `locate_path` can be used to tag a `Locate` with a path if it does not already have one.

## Crate: volta_frontend

**Path**: `crates/volta_frontend/`

### Lexer (`lex.rs`)

Tokenizes PTX source. Key methods: `next()`, `peek()`, `expect(kind)`.

### Parser (`parse.rs`)

Pratt parser producing AST. Entry point: `parse_module()`.

### AST (`ast.rs`)

- `Module` - Top-level: version, target, address_size, directives
- `Function` - Kernel/device function with params and body
- `Instruction` - Generic instruction: the trie already resolves the mnemonic to a typed `InstrKind` at parse time (`InstructionOp::Unparsed { kind, modifiers, operands }`)
- `ScalarType` - Pred, Signed/Unsigned/Float/Bits with width

### Instruction Parsing (`instr.rs`, `instr_parse.rs`)

- `InstrTrie` - O(n) lookup for PTX mnemonics → `InstrKind`
- `ParsedInstruction` - Strongly-typed enum (~105 instruction variants, defined in `ast.rs`)
- Converts generic `Instruction` to typed variants with validated modifiers

## Crate: volta_analysis

**Path**: `crates/volta_analysis/`

### ID Types

Strongly-typed IDs (`#[id_type]` from `id_collections`), each declared next
to its subsystem:

- `InstrId` (`lowered.rs`), `ThreadId` (`eval/mod.rs`), `ParamId` and
  `RegId` = `RegClass` + `RegIndex` (`symbols.rs`; class: Pred,
  Bits8/16/32/64/128), `ExprId`/`StringId`/`SymbolId` (`symbolic.rs`;
  `SymbolId::fresh()` draws from a process-global counter)
- `types.rs` holds `ScalarTypeExt` (width/kind helpers over the AST's
  `ScalarType`), not IDs

### Symbolic Expressions (`symbolic.rs`)

Arena-allocated: nodes live in an `ExprArena`, referenced by copyable `ExprId`
handles. Constructors constant-fold eagerly - **exactly**: float constants
are `Real`s (arbitrary-precision rationals via `rug`, boxed, plus
`NegInf`/`PosInf`; NaN is rejected at every f64 ingestion point -
`Real::from_f64`/`arena.float_from_f64` are fallible, NaN literals are a
lowering error, NaN params a config validation error), so the fold algebra
and canon's rational algebra coincide by construction. Folds are exact on
ℚ (div/rcp fold only fully-concrete quotients with a nonzero divisor -
`x/0` and `0/symbolic` both stay unfolded, so a formally-zero
denominator always reaches canon's loud division error); ±inf folds
only the unambiguous extended-real forms (max/min absorption, neg,
inf±finite, inf·nonzero), undefined forms (inf−inf, 0·inf - integer or
real zero alike - anything/0) build unfolded nodes.

- **Atoms**: `IntConst`, `RealConst(Real)`, `BoolConst`, `Symbol(SymbolId)`, `ParamSymbol(StringId)`, `InputElement { array, index }`, `Undefined`
- Symbol identity is typed (`SymbolRef`: `Param`/`Element`/`Machine`,
  disjoint namespaces; one mapping in `ExprNode::symbol_ref`). Identity
  comes only from launch-config names - PTX-source names are scoped and
  must not carry identity; values without a config binding are fresh
  machine `Symbol`s. `AnalysisConfig::validate` rejects ambiguous configs.
- **Arithmetic**: `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Neg`, `Fma`
- **Transcendental**: `Exp`, `Log`, `Sqrt`, `Rcp`
- **Bitwise**: `BitAnd`, `BitOr`, `BitXor`, `BitNot`, `Shl`, `Shr`, `LShr`
- **Comparison**: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge` (return boolean)
- **Boolean**: `And`, `Or`, `Not`
- **Other**: `Select` (ternary), `Min`, `Max`, `Abs`, type conversions

### Lowering (`lowering.rs`, `lowered.rs`)

Converts AST to linear instruction format:

- `LoweredProgram` - `IdVec<InstrId, LoweredInstr>` + `SymbolTable` + `SourceMap`
- `LoweredInstr` variants: `LoadParam`, `Load`, `Store`, `Mov`, `BinOp`, `UnaryOp`, `Fma`, `Mad`, `MulWide`, `MulHi`, `Setp`, `Selp`, `Cvt`, `Bra`, `Ret`, `Exit`, `BarSync`, `BarWarpSync`, `ShflSync`, `Ldmatrix`, `Mma`, `WmmaLoad/Store/Mma`, `Activemask`, `Trap`, etc.
- `SymbolTable` - Register/param/label name → ID resolution; assigns addresses to shared/local/module-global variables
- `SourceMap` - Maps lowered elements back to source spans
- The nvcc callseq idiom for `call __symexpf` (the paper's symbolic-exp hook) collapses to `UnaryOp::Exp` at lowering time
- `define_instr_kinds!` generates the profiling table from one variant
  list: `KIND_COUNT`, `KIND_NAMES`, `kind_index()` (dense index for the
  interpreter's fixed-size counters), `kind_name()`

### Special Registers (`symbols.rs`)

`SpecialRegKind`: `TidX/Y/Z`, `NtidX/Y/Z`, `CtaidX/Y/Z`, `LaneId`, `WarpId`, etc.

### Evaluator (`eval/`)

The interpreter from the paper (per-thread round-robin symbolic execution):

- `eval/interp.rs` - `Interpreter`: scheduler (run a thread until it blocks or exits), instruction evaluation into the arena, barrier firing per the paper's Sync rule, deadlock detection, structured-CTA concreteness checks. Every sync (CTA barrier and warp group alike) deliberately uses the paper's Sync'/syncMem semantics: exited threads count as arrived *and* are included in the χ-clear over the whole sync set I - stronger than the ISA's participating-threads-only memory ordering, so a spec-level race pairing a pre-sync exit with a post-sync access is intentionally not reported; tensor-core ops are the one deliberate exception to the paper's rule (it would fire with returned threads, but the ISA makes exited-lane tensor ops whole-op UB, so Volta rejects them loudly instead)
- `eval/value.rs` - `Value::{Scalar, Pair}` (`Pair` = packed f16 halves in a 32-bit register) and per-thread `RegFile`
- `eval/memory.rs` - byte-addressed granule memory; 4-byte reads combine two 2-byte granules into a `Pair`, 2-byte accesses split `Pair` granules; program writes are `dirty` (the output footprint)
- `eval/race.rs` - χ-context race detection per byte (paper Section 3.2); full-CTA barrier sync is a wholesale clear
- `eval/warp.rs` - warp-cooperative ops (`bar.warp.sync`, `shfl.sync`, `ldmatrix`, `mma.sync`, `wmma.*`): block until all *live* mask lanes converge at the pc (exited lanes count as arrived), sync χ over the full mask (exited lanes included), execute via the `tensor_core.rs` fragment tables with exact per-lane access attribution; a shfl source lane that exited yields `Undefined` data, while the tensor-core ops reject exited lanes loudly (the ISA defines them as UB "if any thread in the warp has exited")
- `eval/config.rs` - `AnalysisConfig`: launch dims, positional `ParamValue`s (int/float/symbolic-float/array-pointer), `ArrayDef`s (`Input`/`Output`/`InputOutput`/`IndexInput`), module-global values, dynamic shared size
- `eval/error.rs` - `EvalError`: `DataRace`, `Deadlock`, `NotConcrete`, `OutOfBounds`, `UndefinedOutput`, `TrapReached`, etc.

Key semantics: input-array symbols materialize lazily on first read; reads of
never-written registers/shared bytes yield `Undefined` (an error only if it
reaches an output or a concreteness point) - the paper's race example and
nvcc's `selp` accumulator-init idiom both rely on this.

### Driver (`driver.rs`)

- `analyze_kernel(module, kernel_name, config) -> Result<AnalysisOutput, AnalysisError>`
- `AnalysisOutput`: per-output-array written elements as `(index, ExprId)` + `Stats` (instructions, block syncs; warp syncs counted per fired group, not per thread) + `op_counts` (per-instruction-kind execution counts; the interpreter tallies into a fixed `[u64; KIND_COUNT]` and folds to this `BTreeMap` at the end)
- `paired_elements(ref, opt, arrays)` - pairs the two outputs' written
  elements for each array the caller names (both sides must have each
  named array with identical index sets; unnamed arrays are not compared
  - FlashAttention's optimized-only `l`/`m` exports rely on this; an
  empty list is an error). Callers supply the list explicitly: the bench
  harness derives it from the reference config's declared output arrays;
  the CLI requires it on the command line (`--check-array`).
  Shared by the decision procedure and `volta_z3` so both backends check
  exactly the same elements. `sampled_elements(paired, sample)` flattens
  the paired footprints to each array's sampled prefix - the one
  definition of *which* elements get checked, used by both backends and
  the bench harness's Z3 re-solve loop
- `check_output_equivalence_with(ref, opt, arrays, options)` - the per-element
  check, one `EquivSession` per worker per solve iteration (one shared
  session at the default `parallelism` 1).
  `EquivCheckOptions`: `sample`, `verify_numeric` (f64 oracle per
  element; iteration 1 only), `recycle_terms`, `iterations`
  (`NonZeroUsize`, default 1: re-solve the same sampled elements N
  times, fresh session each - cold-start timings, memory bounded since
  each session drops before the next; the verdict comes from iteration 1
  and later iterations must agree, else `IterationDisagreement` - a
  free determinism check), `parallelism` (`NonZeroUsize`, default 1:
  split the element list into contiguous chunks, one worker thread with
  a private session per chunk - verdicts are session-independent (the
  property the iterations check rests on) so the partition can't change
  them, contiguity keeps row-local shared structure in one session, and
  `recycle_terms` stays the *aggregate* cap (each worker recycles at
  cap/workers); the summed timings then include cross-worker contention,
  so keep 1 for paper-comparable numbers). Returns a report with the
  outcome, checked/total element counts, `check_iters: Vec<Duration>`
  (each iteration's summed `EquivSession::check` durations only -
  pairing and the oracle excluded; `check_time()` = iteration 1's),
  `element_checks:
  Vec<ElementCheckTime>` (iteration 1's per-element check durations in
  `sampled_elements` order - the same measurements `check_iters[0]`
  sums, recorded outside the timed spans, so carrying them is free; the
  bench harness's `decision_elements`), `wall_iters` (each iteration's
  elapsed element pass - tracks `check_iters` at parallelism 1, the
  honest elapsed number above it), `pair_time`
  (the `paired_elements` call), and `verify_time` (the oracle's total
  time, `Some` iff `verify_numeric`).
- `check_output_equivalence(ref, opt, arrays)` - the Default-options wrapper
  (all elements, no oracle, one iteration)
- `VcSnapshot`/`VcDump` - serde-serializable arena + output footprint, the
  payload of `volta compare --dump-vcs`/`--from-dump` and of volta-bench's
  per-benchmark VC files; `validate()` checks
  every id (in bounds and children-before-parents) so a corrupt dump errors
  instead of panicking. The arena's serde impls are plain derives over
  `IdVec` via `id_collections`'s `serde` feature (wire-identical to `Vec`,
  serialized in place - no clone of GiB-scale arenas at dump time).
- `vc_dump` submodule - the one `.vcdump` on-disk format implementation:
  `write_vc_dump(path, &VcDump)` / `read_vc_dump(path)`, layered over
  `write_vc_dump_to(&mut impl Write, ..)` / `read_vc_dump_bytes(&[u8])`
  so volta-bench can hash the exact byte stream on write (through a tee
  writer) and fingerprint-check raw file bytes before decoding on solve.
  `VOLTAVCD` magic
  + u16 major/minor header (a reader accepts its own major with minor <=
  its own), bincode fixint payload with decode allocations capped at
  payload size, `validate()` on load. Shared by `volta_cli` and
  `volta_bench`, so a dump written by either tool loads in both.
- `write_op_counts(out, label, counts)` - the one profile-table formatter,
  used by both `volta` and `volta-bench`

### Decision procedure (`canon/`, `equiv.rs`, `numeric.rs`)

The paper's canonicalizer, in Rust:

- `canon/` - expressions canonicalize to interned `Σ c·monomial·e^{poly}`
  rationals in one memoized bottom-up pass per `Session` (both kernels, all
  VC elements share intern tables). Exact i128 rational coefficients;
  `e^a·e^b` fuses at term multiplication; max/min flatten into sorted atoms;
  ops outside the fragment (sqrt/log/bitwise/comparisons/select/symbolic
  array reads) become opaque `Atom::Uninterp` atoms over an `UninterpOp`
  enum - sound, incomplete. Fraction equality goes id-compare →
  monomial-quotient (softmax rescaling) → cross-multiplication under a term
  budget. Two load-bearing invariants: single-use chain intermediates stay
  *transient* owned vectors (interning everything retains O(K²) per
  accumulator), and polys sort by *descending* TermId so chain unwinding
  appends in O(1).
- `equiv.rs` - thin wrapper: `EquivSession` (reuse across elements;
  recycles its intern tables past a configurable term bound -
  `with_recycle_terms`, default `DEFAULT_RECYCLE_TERMS` = 4M) and one-shot
  `check_equivalent`. The interning policy's sharing test (per-side
  parent counts, `canon::parent_counts`) is plain arena-derived data:
  sessions take it precomputed (`with_shared_counts` /
  `Session::provide_ref_counts`), so recycles and the driver's parallel
  workers share one scan instead of each re-walking GiB-scale arenas.
  Memory scale: exp-heavy attention terms run 2-4 KB
  each, so one warm FlashAttention output row retains several GiB; small
  bounds trade re-canonicalization time for bounded memory.
- `numeric.rs` - the f64 oracle: seeded random inputs, memoized DAG eval;
  `verify_verdict` confirms EQUIV/DIFF claims (volta-bench
  `--verify-numeric`). Agreement at random points ⇒ equality almost surely
  for this fragment (the paper's own Schwartz-Zippel argument).

### Logging (`logging.rs`)

Gated by the `logging` feature (`volta_analysis`, passed through by
`volta_cli`); without it the macros are no-op stubs. Wired at the decision
points: barrier/warp-group fires (trace), deadlock (warn), launch config,
completion stats, and VC session recycles (info), fraction-equality
escalation (debug). `cargo run -p volta_cli --features logging --
--log-level info analyze ...` narrates a run.

## Crate: volta_z3

**Path**: `crates/volta_z3/`

Z3 comparison backend for the same verification conditions: generates
SMT-LIB2 text and evaluates it through libz3's C API (`ffi.rs`, a
hand-written eight-function binding - no `z3-sys`/bindgen; building
requires `libz3-dev`). Each query runs in a **worker subprocess** (the
binary re-invokes itself via `std::process::Command`; thread-safe, no
separate executable) killed on timeout expiry: z3 4.8.12 does not
reliably honor its soft timeout or `Z3_interrupt` in the quantifier
loop the exp-axiom mode provokes (measured), so a hard kill is the only
real bound - which also gives per-element crash containment. Contract:
any binary that evaluates queries through this crate calls
`volta_z3::init_worker()` as the first statement of `main` (loudly
checked via a handshake). A capability/timing comparison point against
`canon`, not a replacement.

Two `ExpMode`s reproduce the paper's section 6.5 baselines: the default
`PowerBounded` (`(^ e a)`, bounded free `e`; attention VCs come back
`unknown`) and `AdditionAxiom` (uninterpreted `uexp` plus
`forall x y. uexp(x) uexp(y) = uexp(x+y)`; attention VCs run until the
budget kills them, reported `Timeout` - Table 8's "with axiom" column,
10-minute budget in the paper).

- `translate.rs` - a *direct semantic image* of the fragment
  (arithmetic + `Exp` + `Max`/`Min`; everything else `Unsupported`):
  every node maps to its defining SMT term and ALL algebraic reasoning
  is left to the solver, so timings measure Z3, not the translator
  (`max`/`min` are `ite` case splits, not opaque atoms; no
  canonicalization, no structural short-circuit). The translation owns
  fidelity/transport only: exact rational literals straight from
  `RealConst` (same reading as `canon`/`numeric`; the infinities are
  loud `Unsupported`), user symbols as an injection of the
  typed `SymbolRef` namespaces (`|p!name|` params, `|e!array[i]|`
  elements - a param named `t0`/`e` cannot capture generated names),
  memoized `let`-bound DAG sharing (query text linear in the arena;
  deeply nested `let`s are fine for z3, `define-fun` chains are not -
  measured), `stacker`-guarded recursion for deep accumulator spines,
  linear `let`-chain assembly, and the exp base as a strictly-bounded
  free constant (a definite rational base proved false equivalences).
  Inherited SMT semantic: division is underspecified at zero, so
  `x/x = 1` is falsifiable - unlike canon's field model (moot on the
  corpus: division only occurs inside exp-laden VCs).
- `ffi.rs` - `init_worker` (the host-binary hook that turns the
  re-invoked process into a solver worker), `eval_smtlib2` (spawns the
  worker, writes the query to its stdin, enforces the hard deadline via
  kill; inside the worker: fresh context per query, no-op error handler
  so z3 API errors surface as `(error ...)` text instead of aborting,
  soft timeout via the process-global `timeout` param) and `z3_version`.
  The worker times the libz3 evaluation itself (empty-script warmup
  first, so z3's lazy per-context frontend setup stays outside the
  span) and reports it in-band (a `t:<nanoseconds>` line after the
  handshake); `eval_smtlib2` returns it as
  `EvalOutcome::Output { text, solve }`. Solver time is measured there
  because the worker's fixed scaffolding - process
  spawn/exec/link/pipes ~1.6ms plus z3 context create/destroy and
  frontend setup ~9ms (all measured) - is several times an entire
  polynomial-fragment solve; an outer timer measures scaffolding, not
  z3.
- `lib.rs` - per-element querying (`check_equivalent`; every element is
  a genuine solver query - no structural short-circuit, so identical
  sides cost a full spawn+solve, which is the point), verdict
  parsing, `Z3Counts`, `check_output_equivalence` over
  `driver::paired_elements` (the same element pairing as the decision
  procedure), and the regression tests for every invariant above.
  Reported solve time (`Z3CheckResult::solve`) is the in-worker
  measurement - process spawn and translation excluded; Timeout
  verdicts report the budget itself rather than a measurement (the
  paper's convention for timeout rows), under either delivery
  mechanism (hard kill or z3's in-band soft cancel).

## Crate: volta_bench

**Path**: `crates/volta_bench/`

Reproduces the paper's evaluation over `kernels/` (the `.cu` + `.ptx` for
every benchmark in the paper, organized by table/section).
Benchmark definitions with full launch/param configs live in
`src/benchmarks/*.rs`. Run with `cargo run --release -p volta_bench --
category <reduction|matmul|attention|causal|conv|agent|tilelang|race>
[--sample N] [--verify-numeric] [--recycle-terms N] [--iterations N]
[--parallel N] [--z3] [--z3-timeout N] [--out-dir DIR]` (also `all`, `single <name>`,
`list`, and the phase-decoupled `generate`/`solve` below; release mode
matters: ~20x, and the binary prints loud stderr
warnings at startup for a debug build or, under the `logging` feature,
a `--log-level` of info+ - both corrupt timings).

One pipeline per benchmark (`src/runner.rs`): **VC generation**
(`--iterations` timed runs of lowering + both symbolic executions +
`paired_elements`; kernel files are read and parsed once, outside the
timed loop; only the last generation's outputs are kept - each
iteration drops its predecessor, peak memory = one generation - and
every iteration is fingerprint-checked against iteration 1: same
outcome kind, per-array footprints, and expression identities (arena
node count + per-element `ExprId`s - fresh deterministic arenas make id
equality a strong check at zero cost); rejections compare by rejection
kind only, since diagnostic text may embed schedule-dependent details -
verdict kinds are the contract; a mismatch fails the benchmark loudly), then
the dump written once from the last generation (timed as
`dump_write_secs`), then **decision solve** (`--iterations` runs via
`driver::check_output_equivalence_with` - fresh session(s) per iteration,
verdict from iteration 1, later iterations must agree; `--parallel N`
solves each iteration's element list on N worker threads - verdicts
unchanged, summed solve timings then include contention, so keep 1 for
paper-comparable numbers), then the
optional **Z3 solve** (`--z3`; `src/z3_phase.rs`) over the exact same
`driver::sampled_elements` list. Race-check benchmarks stop after
generation (no dump/solve/Z3). Every timed phase defaults to
`--iterations` 10; tables print medians (the convention: median is the
headline number - iteration 1 includes process/allocator warmup, the
median absorbs it), and the harness warns per benchmark when a phase's
CV (sample sd/mean) exceeds 0.10 (`runner::NOISY_CV_THRESHOLD`).

The Z3 phase (columns appear in the tables only under `--z3`) solves
with `ExpMode::PowerBounded`, plus a `+exp-axiom` sub-run
(`ExpMode::AdditionAxiom`) when the VCs contain `Exp` nodes. Solve
columns cover the deciding work only, so they are comparable:
decision = summed `EquivSession::check` (pairing and the optional
numeric oracle excluded), Z3 = in-worker libz3 solve time (worker
spawn/exec and translation excluded; timeout elements count the full
budget). One Z3 carve-out: elements whose iteration-1 outcome is
timeout/unsupported/error are solved once, their iteration-1 time
charged to every iteration's total (re-solving a timeout would multiply
its budget); Z3 verdict counts and per-element results always come from
iteration 1. A Z3 phase *failure* (not an unknown/timeout verdict -
those are data) fails the benchmark (`Z3PhaseOutcome::Failed`). Paper
Table 8 reproduction: `--sample 1 --z3 --z3-timeout 600 all`.

The pipeline's halves also run separately, calling the same phase
functions (no duplicated pipeline): `generate <all|category <c>|single
<name>>` runs the generation half only (`generate_inner`: fingerprint
check, dump write) and records each dump in `vcs/manifest.json`
(`src/manifest.rs`: benchmark name, timestamp, `vc_fingerprint` = FNV-1a
of the exact dump bytes as written - hashed through a tee writer, no
second in-memory copy - plus informational per-array element counts;
read-modify-write per dump with the read just before the atomic
temp+rename write, so partial regenerations keep other entries; single
writer per out-dir assumed) - race-check benchmarks reach their real
verdicts here, so the race table comes from a `generate` run;
equivalence benchmarks report `GEN` (`ActualOutcome::VcsGenerated`), a
failed dump/manifest write fails them (the dump is the product), and any
failure leaving no fresh dump also deletes the benchmark's stale dump +
manifest entry so a later `solve` errs on the missing dump instead of
silently solving pre-failure VCs. `solve <target> [--backend
decision|z3|both]` (`src/solve.rs`) replays the solve phase(s) from the
dumps via `check_equivalence`/`run_z3_phase`: each dump's raw bytes are
fingerprint-checked against the manifest *before* decoding (catches any
content difference from what the last successful `generate` recorded,
including same-shape expression drift; deliberately does not check
currency with the source tree - regenerated-together stale sets pass by
design; missing manifest = warning only), then decoded through the
shared validated reader (`dump_load_secs` recorded, excluded from solve
timings); a missing dump or a fingerprint/name disagreement is a
per-benchmark failure naming `generate` (with `dump_path` still set
whenever the file exists), race benchmarks are skipped with a note, and
`--backend z3` yields no decision verdict (`ActualOutcome::Z3Only`:
passing = the phase completed *and* z3 refuted nothing - a
`not_equivalent` count > 0, plain or `+exp-axiom`, fails the row as `Z3
DIFF`, since nothing else rules it and even a spurious refutation -
volta_z3's division-at-zero divergence - must surface;
unknown/timeout/unsupported stay non-failing data). `--sample` applies
to `solve`; `--verify-numeric`/`--recycle-terms`/`--parallel` act on its
decision phase and `--z3-timeout` on its z3 phase (each
noted-and-ignored under a
backend without that phase, and by `generate`); `--z3` is one-shot-only
and both new subcommands reject it. Records split accordingly:
`generate` records carry gen fields only, `solve` records carry solve
fields plus `dump_load_secs`, and the header adds `backend` +
`vcs_from_dumps`. Paper workflow: `generate all`, then `--recycle-terms
0 solve all`, `solve all --sample 1`, `solve all --sample 1 --backend z3
--z3-timeout 600` (Table 8; Table 7 is static, from `generate`).

Output files under `--out-dir` (default `bench-out/`, gitignored):
`vcs/<sanitized-name>.vcdump` per equivalence benchmark (written via the
shared `driver::vc_dump` module, so `volta compare --from-dump` replays
them; overwritten on rerun; a run whose benchmark names collide under
sanitization is rejected up front naming both offenders -
`results::check_slug_collisions`, corpus-guarded by a test; race-check
benchmarks are skipped with a console note) and
`results/<unix-seconds>-<pid>-<command>.json` for every run command,
one schema (built in `src/results.rs`): header
(argv/timestamp/iterations/sample/recycle-terms/parallelism/`z3` flag +
timeout +
the carve-out convention) and per-benchmark records - status/detail/
passed, element counts, per-phase stats (`vc_gen_*`, `solve_*`, and
`solve_wall_*` - the solve iterations' wall clock, which tracks
`solve_*` at `--parallel` 1 and is the honest elapsed number above it:
full
`*_iters_secs` array + median/min/mean/cv), `dump_write_secs`,
`verify_numeric_secs` (oracle time, `Some` iff the flag),
`decision_elements` (iteration 1's per-element decision times, from
`EquivCheckReport::element_checks`), a `z3` section (null without
`--z3`; else `solve_*` stats + verdict `counts` + iteration-1
`elements` as `{array, index, outcome, detail, solve_secs}`, an `axiom`
sub-section of the same shape or null, and `error`), instruction/sync
counters, and `dump_path` - kept on failures that occur after the dump
was written. `--json <path>` additionally writes the same document to
an explicit path. Results files are written before any console report
prints (`all`/`category` tables and `single`'s result block alike), and
report printing tolerates a broken stdout pipe (`| head`), so
the files always land. Every run writes a `volta_common::run_log` file
(`--log-dir`/`--no-log-file`).
Memory: full-element attention wants tens of GiB warm - on small machines
run one category at a time under `ulimit -v` with `--recycle-terms 250000`
(bounded at ~5 GiB, slower VCs).

## Crate: volta_cli

**Path**: `crates/volta_cli/`

Commands:

- `volta parse <file>` - Check syntax
- `volta analyze <file> -k <kernel> -b 32,4 -g 1,2 --array name:base:width:len:kind --param ptr:name ...` - Run symbolic execution, report races/deadlocks, print output expressions (+ a per-instruction-kind profile; `--no-profile` to skip)
- `volta compare <ref.ptx> <opt.ptx> --kernel1 .. --kernel2 .. --check-array out` - Two-kernel
  equivalence check (launch flags shared with `analyze` via a flattened
  `LaunchArgs`; `--block2`/`--grid2` override the optimized kernel's dims).
  `--check-array NAME` (repeatable, required) names the output arrays to
  check - the explicit `paired_elements` list, checked against the
  declared config before execution and by `paired_elements` after.
  `--backend decision|z3`, `--iterations N` and `--parallel N` (both
  decision backend only, default 1; `--parallel` solves the element list
  on N worker threads - same verdicts, and the console then reports
  summed-across-workers decision time beside wall clock), `--dump-vcs`/
  `--from-dump` (the shared `driver::vc_dump` format - volta-bench's
  `bench-out/vcs/*.vcdump` files replay here too). Exits 0 only when every
  checked element is proved equivalent.

Every run writes a `volta_common::run_log` file (`--log-dir`/
`--no-log-file`).

## Data Flow

```
PTX Source
    │
    ▼
Lexer (lex.rs) ──► Tokenizes source
    │
    ▼
Parser (parse.rs) ──► Builds AST
    │
    ▼
Instruction Parser (instr_parse.rs) ──► Strongly-typed ParsedInstruction
    │
    ▼
Lowering (lowering.rs) ──► LoweredProgram (resolved RegIds, InstrIds)
    │
    ▼
Evaluator (eval/interp.rs) ──► Symbolic execution with N threads
    │                           (χ race detection, warp/tensor-core ops)
    ▼
AnalysisOutput ──► per-element output expressions + statistics
    │
    ▼
Decision procedure (canon/ via equiv.rs) ──► per-element VC checking
                                             (+ numeric.rs oracle)
```

## Key Design Decisions

### 1. Symbolic Execution over Concrete Values

All values are `Expr` (symbolic expressions). This allows analyzing behavior for arbitrary thread indices and detecting races without enumerating all inputs.

### 2. Concrete Addresses for Race Detection

Memory addresses must be concrete (`u64`) for race detection. Thread indices are concrete (specific block configuration). Symbolic address accesses produce a `NotConcrete` error.

### 3. χ-Context for Race Detection

From the paper (Section 4.2): Track which threads haven't synchronized since each memory access. After barrier, threads in sync set remove each other from "needs sync" sets. Race detected when accessing thread is in the "needs sync" set.

### 4. Round-Robin Scheduling

Simple, deterministic interleaving. Sufficient for race detection (any interleaving that produces a race proves the race exists).

### 5. Strongly-Typed IDs

Newtype pattern for IDs (`RegId`, `InstrId`, `ThreadId`) prevents mixing at compile time. Uses `IdVec<K, V>` for type-safe indexed collections.

### 6. Two-Phase Instruction Parsing

1. Lexer/Parser → generic `Instruction { op: String, operands }`
2. Instruction Parser → strongly-typed `ParsedInstruction` variants

Enables robust modifier validation and better error messages.

### 7. Separate Memory Spaces

Global, shared, local, param memories are separate `Memory` instances, matching PTX memory model.

## Adding a New Instruction

1. **Frontend**: Add `InstrKind` variant in `instr.rs`, add to trie
2. **Instruction Parsing**: Add `ParsedInstruction` variant, parser function
   in `instr_parse.rs` (use `expect_operands` for exact-arity operand lists)
3. **Lowering**: Add `LoweredInstr` variant, lowering case
4. **Evaluation**: Add evaluation case in `eval/interp.rs`
