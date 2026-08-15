# Volta

Volta is a data race and equivalence checker for NVIDIA GPU kernels, implementing the approach from ["Equivalence Checking of ML GPU Kernels"](https://arxiv.org/pdf/2511.12638). Given a reference kernel implementation and an optimized counterpart, Volta proves their semantic equivalence over the reals, i.e., that they produce identical outputs for all valid inputs modulo floating point error, thereby verifying the correctness of the optimized kernel.

## Features

- **Deadlock Detection**: Identify deadlocks arising from over synchronization
- **Data Race Detection**: Identify races arising from under synchronization
- **Equivalence Checking**: Verify that optimized GPU kernels are semantically equivalent to their reference implementations
- **Two-kernel equivalence from the CLI**: `volta compare` checks a reference/optimized pair directly, without going through `volta_bench`
- **VC dump/replay**: persist the verification conditions from one run and rerun just the equivalence check later, skipping parse/lower/symbolic-execution entirely
- **Per-run logging and execution profiling**: every run gets a log file, and a per-instruction-kind execution profile is shown by default
- **Z3 backend**: check the same verification conditions with Z3 instead of the built-in decision procedure, for a "decides vs. cannot decide" timing/capability comparison

## How It Works

Volta has two phases:

1. **Symbolic Execution**: Executes both kernels symbolically (round-robin over all threads of CTA 0), tracking memory accesses and synchronization to detect data races and deadlocks and producing symbolic expressions representing output values as functions of input tensors.

2. **Equivalence Checking**: Verifies that the symbolic expressions from both kernels are mathematically equal over the reals. Each output element canonicalizes to a rational function whose polynomials are sums `c * monomial * e^{poly}` terms with exact rational coefficients. An optional `f64` oracle (`--verify-numeric`) re-checks every verdict at seeded random inputs.

## Soundness and Completeness

Equivalence checking treats floating-point values as reals. Within that model:

- Race and deadlock detection is sound and complete for structured-CTAs (see
  [requirements](#requirements)) using `+`, `-`, `*`, `/`, `exp`, and `max`/`min`
  with symmetric CTAs (only CTA 0 is checked, but note that the grid size still
  matters for index computations).

- `sqrt`, `log`, `abs`, `rem`, bitwise ops, shifts, comparisons, boolean
  ops, `select`, and data-dependent array reads are carried as uninterpreted
  atoms, equal only when syntactically identical after canonicalizing their
  arguments. We lose completeness but not soundness.

## Requirements

The input to Volta is PTX code (the lowest level of the public-facing language stack for NVIDIA GPUs). PTX files can be generated from CUDA or CUTLASS code using `nvcc`.

We require that kernels are _structured-CTAs_. That is:

- Tensor/array sizes are statically known
- Branch targets and memory addresses can be resolved statically given the grid dimensions and input arrays
- There is no recursion

The only synchronization primitives we currently support are barriers, such as `syncwarp`, `syncthreads`, and the implicit warp-level barriers of tensor core operations (`mma.sync`, `wmma.*`, `ldmatrix`, `shfl.sync`). We do not support asynchronous primitives such as `arrive` and `wgmma`.

## Building

```bash
cargo build --release   # release mode matters: ~20x faster analysis
cargo test --workspace  # run the test suite
```

## Usage

### Parse a PTX file (syntax check)

```bash
cargo run --release -- parse <file.ptx>
```

### Analyze one kernel

Symbolically executes a kernel: reports data races and deadlocks, and prints
the symbolic expressions for each output array element.

```bash
cargo run --release -- analyze <file.ptx> -k <kernel> -b 32,4 -g 1 \
    --array "vals:0x100000000:4:2048:in" \
    --array "out:0x200000000:4:2048:out" \
    --param ptr:out --param ptr:vals --param int:2048 \
    --dyn-shared 1024
```

- `-k, --kernel`: kernel entry name (defaults to the first kernel in the module)
- `-b, --block` / `-g, --grid`: launch dimensions, e.g. `128` or `32,4,1`
- `--array "name:base:elem_width:len:kind"` (repeatable): declares a global
  array at address `base` with `len` elements of `elem_width` bytes; `kind`
  is `in` (symbolic input), `out`, `inout`, or `index` (concrete
  `arr[i] = i`, for index/permutation inputs)
- `--param` (repeatable, in declaration order): `int:N`, `float:X`,
  `sym:name` (a named symbolic float), or `ptr:array_name`
- `--global NAME=value` (repeatable): module-scope `.global` variable values
- `--dyn-shared N`: dynamic (extern) shared memory bytes
- `--print-outputs N`: print up to N elements per output array (default 8)
- `--no-profile`: skip the per-instruction-kind execution profile (shown by default)

### Compare two kernels

`volta compare` checks a reference/optimized pair for equivalence directly
from the CLI (races/deadlocks are still checked for each kernel individually).
Arrays/params/globals are shared by both kernels by default; give `--block2`/
`--grid2` if the optimized kernel's launch config differs (e.g. a
single-thread reference vs. a 128-thread optimized kernel computing the
same tile). Comparison follows the paper's CTA-to-CTA model: it runs
along the declared output arrays (`out`/`inout` kinds), and both CTA-0
runs must write each of them with identical per-array footprints.
Arrays not declared as outputs are not compared - which is how
auxiliary exports like FlashAttention's softmax statistics stay out of
a comparison against a reference that never computes them.

```bash
cargo run --release -- compare <ref.ptx> <opt.ptx> \
    --kernel1 <ref_kernel> --kernel2 <opt_kernel> -b 128 \
    --array "in:0x10000:4:128:in" --array "out:0x20000:4:1:out" \
    --param ptr:in --param ptr:out
```

- `--sample N`, `--verify-numeric`, `--recycle-terms N`, `--iterations N`:
  same meaning as the `volta_bench` flags below (`--iterations` defaults
  to 1 here and applies to the decision backend only)
- `--no-profile`: skip the per-instruction-kind execution profile (shown by default)
- `--backend decision|z3` (default `decision`): which decision procedure to
  check equivalence with - see [Z3 backend](#z3-backend)
- `--exp-axiom` (with `--backend z3`): the paper's "with axiom" exp
  encoding - see [Z3 backend](#z3-backend)

Exit code: `compare` exits 0 only when every checked element was proved
equivalent (with `--backend z3` that excludes `unknown`/`timeout`/
`unsupported`/error elements, not just mismatches - a run that verified
nothing does not exit 0).

**VC dump/replay**: after symbolic execution, persist both kernels'
verification conditions (the expression arena + output footprint) to disk,
then rerun just the equivalence check from that dump later - no PTX parsing,
lowering, or symbolic execution involved on replay.

```bash
cargo run --release -- compare <ref.ptx> <opt.ptx> ... --dump-vcs pair.vcdump
cargo run --release -- compare --from-dump pair.vcdump   # rerun later, instantly
```

Dump files carry a magic/version header and are validated on load, so a
truncated, corrupted, or version-skewed dump fails with a clean error
rather than a crash.

### Logging

Every `volta`/`volta-bench` run writes a log file under `volta-logs/`
(`<unix-seconds>-<pid>-<command>.log`; the pid keeps two runs in the same
second from clobbering each other), recording the exact command line and a
one-line outcome summary - independent of the `logging` feature, so it
works in a plain build. Pass `--log-dir <path>` to change the directory or
`--no-log-file` to disable it. Building with `--features logging` also
mirrors the `log` crate's trace/debug/info/warn output into the same file
(`--log-level`), in addition to stderr:

```bash
cargo run --release --features logging -- --log-level info analyze ...
```

### Z3 backend

Checks the same verification conditions with Z3 instead of Volta's own
decision procedure, for a timing/capability comparison. Queries are
generated as SMT-LIB2 text (auditable: any query can be replayed against
a standalone z3) and evaluated through libz3's C API - a hand-written
eight-function binding, no `z3-sys`/bindgen/libclang, no temp files, and
no `z3` binary needed at runtime (each query runs in a worker subprocess,
but that worker is this same binary re-invoked - see the timeout note
below). The one prerequisite is the Z3 shared library at build time:

```bash
sudo apt-get install -y libz3-dev
cargo run --release -- compare <ref.ptx> <opt.ptx> ... --backend z3
```

The expected shape of the results reproduces the paper's section 6.5 and
Table 8. Z3 decides the polynomial fragment (reduction/matmul/conv) in
milliseconds. On the exponential fragment (softmax/attention) there are
two documented baselines, and the backend implements both:

- **Default encoding**: the exponential is a nonlinear power term with a
  strictly-bounded base. Z3 returns `unknown` - no decision procedure
  covers symbolic real exponents. This is the paper's no-intervention
  baseline.
- **`--exp-axiom`**: the exponential becomes an uninterpreted function
  plus the addition-law axiom `forall x y. e^x e^y = e^(x+y)` (the
  paper's "Z3 with axiom" setup). The axiom sends Z3 into an unbounded
  quantifier-instantiation loop on softmax-shaped VCs, so instead of a
  fast `unknown` the query runs until the time budget kills it -
  reported as `timeout`. The paper used a 10-minute budget.

`--z3-timeout` is a *hard* per-query bound: z3 4.8.12 does not reliably
honor its own soft timeout in the axiom-induced loop (measured: a
3-second soft timeout still running after 90 seconds), so each query
evaluates in a worker subprocess (the binary re-invoking itself; no
separate executable) that is killed on expiry. `timeout` in the
element counts means the budget expired; `unknown` means z3 itself gave
up with budget to spare.

Because the translation deliberately does no reasoning, *every* element
costs a genuine spawn+solve - trivially identical sides included, at
tens of milliseconds of wall clock each. Reported Z3 time is the
*solver* time only: it is measured inside the worker, spanning exactly
libz3's evaluation of the query text, so the worker's fixed scaffolding
- process spawn/exec/pipes plus z3's context creation and lazy frontend
setup, ~10.5ms together (measured), which would otherwise swamp a
polynomial-fragment query's actual solve - is excluded, as is
translation/query construction. Elements that exhaust the budget report
the budget itself as their time - the paper's convention for timeout
rows - whether the budget was enforced by the parent's hard kill or by
z3's own soft cancellation. A full-footprint run over a large output
(tens of thousands of elements) therefore takes hours where the decision
procedure takes seconds; that gap is a result, not an inefficiency. Use
`--sample` to bound the element count (Table 8 uses `--sample 1`).

Covers the arithmetic + `Exp` + `Max`/`Min` fragment as a **direct
semantic image**: every expression node maps to its defining SMT term
and all algebraic reasoning (commutativity, cancellation, distribution,
max/min case analysis - `max`/`min` render as `ite` over real
comparisons) is left to the solver, so the timings measure Z3, not the
translator. What the translation owns is fidelity and transport: float
constants as exact rationals (the same reading as the decision
procedure and the numeric oracle), user symbol names in reserved
namespaces so they cannot collide with generated solver names, and
`let`-bound DAG sharing so query text stays linear in the expression
arena. One inherited SMT semantic: real division is total but
underspecified at zero, so field identities like `x/x = 1` are
falsifiable (countermodel `x = 0`) - corpus VCs only divide inside
exp-laden softmax terms, where the verdict is `unknown` regardless.
Anything outside the fragment (`Select`, comparisons, bitwise ops,
data-dependent array reads, ...) is reported `unsupported` for that
element rather than guessed at unsoundly. See
`crates/volta_z3/src/translate.rs` for the exact boundary.

### Reproduce the paper's evaluation

`volta_bench` runs every benchmark from the paper (39 in total) over the PTX
collected in `crates/volta_bench/kernels/`.

```bash
cargo run --release -p volta_bench -- list
cargo run --release -p volta_bench -- all
cargo run --release -p volta_bench -- category <reduction|matmul|attention|causal|conv|agent|tilelang|race>
cargo run --release -p volta_bench -- single "(Attention, FA1)"
cargo run --release -p volta_bench -- --z3 category reduction
```

Every benchmark runs through one pipeline: generate the verification
conditions (`--iterations` timed runs), write the VC dump (once, from
the last generation), solve with the decision procedure (`--iterations`
timed runs), optionally solve with Z3 (`--z3`, same iteration scheme),
and record everything in one results document. Race-check benchmarks
stop after generation (their whole analysis is the symbolic execution) -
no dump, no solve phases, and no Z3 even under `--z3`.

The pipeline's two halves also run separately, over the same `all`/
`category <c>`/`single <name>` selectors: `generate` runs just the
generation phase and writes the dumps (plus `vcs/manifest.json`, its
record of what each dump contains), and `solve` replays just the solve
phase(s) from those dumps - no parsing, lowering, or symbolic execution
- with `--backend decision|z3|both` choosing the solver(s). Both halves
call the same phase functions as the one-shot pipeline, so they measure
and decide exactly the same things; see
[Reproducing the paper](#reproducing-the-paper) for the intended
workflow.

```bash
cargo run --release -p volta_bench -- generate category attention   # VCs + dumps only
cargo run --release -p volta_bench -- solve category attention --sample 1
```

Useful flags (global):

- `--sample N`: check at most N output elements per array (0 = all)
- `--verify-numeric`: confirm every verdict with the f64 oracle
  (iteration 1 only)
- `--recycle-terms N`: recycle the VC intern tables past N interned terms
  (0 = never). Lower values bound memory at the cost of re-canonicalizing
  shared structure
- `--iterations N` (default 10): run every timed phase N times per
  benchmark for statistically meaningful numbers - see
  [Timing, iterations, and output files](#timing-iterations-and-output-files)
- `--z3` / `--z3-timeout N`: also solve every equivalence benchmark's
  VCs with Z3, side by side with the decision procedure - see
  [Comparing against Z3](#comparing-against-z3)
- `--out-dir <path>` (default `bench-out/`): where VC dumps and results
  JSON files land - same section
- `--json <path>` (on every run command): *also* write the
  results JSON document to this explicit path (the timestamped file under
  `<out-dir>/results/` is always written; both contain the same document)

`--sample`, `--verify-numeric`, and `--recycle-terms` are solve-phase
options: they apply to the one-shot commands and to `solve`, and
`generate` notes and ignores them (as it does a non-default
`--z3-timeout`). Within `solve`, `--verify-numeric` and
`--recycle-terms` act on the decision procedure, so under `--backend
z3` the run prints a note that they have no effect and proceeds;
symmetrically, `--z3-timeout` acts on the z3 phase and is
noted-and-ignored under `--backend decision`. `--z3` belongs to the
one-shot commands only; `solve` picks its solver(s) with `--backend`.
`--iterations` applies to whichever phases a command runs.

`single` also prints a per-instruction-kind execution profile for both
kernels automatically (matching `volta compare`'s default); `all`/`category`
stay compact and don't, to avoid flooding the table with one profile per
benchmark row.

At startup the binary prints loud stderr warnings when the environment
would corrupt the timings: a build without optimizations (timings ~20x
off - use `--release`) or, under the `logging` feature, a `--log-level`
of `info` or above (log output is emitted from inside the timed phases).

### Reproducing the paper

The full evaluation is four commands: generate every benchmark's VCs
once, then solve the same dumps three ways. Each command runs its own
timed phase the default 10 `--iterations`.

```bash
# 1. Generate and dump every benchmark's VCs - the "Gen" timings; no solving.
cargo run --release -p volta_bench -- generate all

# 2. Decision-solve ALL elements from the dumps - the full-footprint "Solve" timings.
cargo run --release -p volta_bench -- --recycle-terms 0 solve all

# 3. Decision-solve one element per output array - the paper's sampled setting.
cargo run --release -p volta_bench -- solve all --sample 1

# 4. Z3 on the same sampled elements under a 10-minute budget - Table 8.
#    unknown/timeout/unsupported are Table 8's data, never failures; a z3
#    `not_equivalent` on any element FAILS that row (`Z3 DIFF`, nonzero exit) -
#    with no decision verdict in --backend z3, an affirmative refutation is
#    exactly what this step must surface (even one that turns out spurious:
#    volta_z3's SMT division-at-zero semantics diverge from canon's field model).
cargo run --release -p volta_bench -- solve all --sample 1 --backend z3 --z3-timeout 600
```

Step 1 also settles the race table: Table 7 is static - verdicts, not
solve timings - and those verdicts come from generation (race-check
benchmarks' whole analysis is the symbolic execution), so they land in
step 1's results file and `solve` skips those benchmarks with a note.
Steps 2-4 never re-execute a kernel: they load `bench-out/vcs/*.vcdump`
(each file's bytes are hashed against `bench-out/vcs/manifest.json`
*before* decoding, then validated on load, so a stale, mixed, or
modified dump directory fails loudly instead of quietly solving the
wrong VCs; the load time is reported as `dump_load_secs`, excluded from
the solve timings). On a memory-limited machine, replace step 2 with
per-category `solve category <c>` runs under a positive
`--recycle-terms` (see the memory note below) - step 2 as written wants
the full-footprint attention working set.

### Timing, iterations, and output files

Every benchmark's work splits into separately-timed phases:

- **VC generation** (`vc_gen_*`): lowering, both kernels' symbolic
  executions, and footprint pairing - everything it takes to *produce*
  the verification conditions from the parsed modules. Each kernel file
  is read and parsed once per benchmark, outside the timed loop; writing
  the dump file is excluded too (reported separately as
  `dump_write_secs`; the dump is written once, from the last generation
  iteration, whose outputs also feed the solve phases).
- **VC solving** (`solve_*`): the per-element equivalence checking only -
  the summed decision-procedure checks, excluding pairing and the
  optional `--verify-numeric` oracle (the oracle's own time is reported
  separately as `verify_numeric_secs`, present exactly when the flag is
  on).
- **Z3 solving** (the `z3` record section, under `--z3`): the in-worker
  libz3 solve time - see [Comparing against Z3](#comparing-against-z3).

Each phase runs `--iterations` times (default 10) so the numbers are
statistically meaningful. The convention: **the median is the headline
number** - iteration 1 includes process/allocator warmup (and, for the
solve phase, first-touch of the VC arenas), and the median absorbs it.
Console tables print medians (noted on each table's header line); the
results JSON keeps every phase's full per-iteration array plus
median/min/mean and the coefficient of variation (sample standard
deviation over the mean). When a phase's CV exceeds 0.10 the harness
prints a per-benchmark warning ("timing noisy; consider more iterations
or a quieter machine") - the numbers still land, but treat them with
suspicion.

Re-running phases is also a correctness check, not just a timing one:

- Every solve iteration re-solves the same sampled elements from a fresh
  VC session (cold-start comparability; memory stays bounded because
  each session drops before the next starts). The verdict comes from
  iteration 1, and later iterations must reproduce it - a disagreement
  is a hard error.
- Every generation iteration must reproduce iteration 1's
  *fingerprint* - the same outcome kind and, element for element, the
  same written footprints and expression identities (arena node count plus
  per-element `ExprId`s: each generation builds a fresh arena
  deterministically, so identical construction order is equivalent to
  identical ids). Rejections compare by rejection kind only - diagnostic
  text may embed schedule-dependent details; verdict kinds are the
  contract. Either way a nondeterministic interpreter regression fails
  loudly instead of silently timing different work. Only the last
  generation's outputs are kept (each iteration drops its predecessor
  first), so peak memory stays at one generation.

Each run writes under `--out-dir` (default `bench-out/`, gitignored):

- `bench-out/vcs/<name>.vcdump` - every equivalence benchmark's
  verification conditions (both kernels' expression arenas + output
  footprints), named by a sanitized benchmark name ("(Attention, FA1)"
  becomes `attention-fa1.vcdump`; a benchmark set whose sanitized names
  collide is rejected up front, naming both offenders) and overwritten
  on rerun (VCs are deterministic). Dumps are byte-identical across runs
  because no production code path creates machine symbols - the one id
  drawn from a process-global counter (`ExprArena::symbol`); a future
  caller of it would void byte-identity but not replayability, since
  `--from-dump` never depends on the numeric id values. These are
  exactly `volta compare --dump-vcs` files -
  one shared format implementation
  (`volta_analysis::driver::vc_dump`) - so they replay directly:

  ```bash
  cargo run --release -p volta_cli -- compare --from-dump bench-out/vcs/red-1-red-2.vcdump
  ```

  Race-check benchmarks have no VCs (nothing to compare); they are
  skipped with a console note.
- `bench-out/vcs/manifest.json` - written by `generate` (read-modify-
  write per dump, atomically via temp file + rename, so partial
  regenerations keep the other entries): each dump's benchmark name,
  generation timestamp, `vc_fingerprint` (a stable FNV-1a hash of the
  exact `.vcdump` bytes written), and per-array footprint element
  counts (informational). `solve` hashes every dump file's bytes
  against the manifest *before* decoding them and hard-errors on
  disagreement. The guard catches any difference between the dump
  being solved and the one the last successful `generate` recorded -
  footprint drift, same-shape expression drift (a one-constant change
  in the PTX), truncation or corruption; what it deliberately does not
  check is currency with the current source tree (a dump set
  consistently regenerated together stays valid however old - solving
  from dumps is decoupled by design). A generate run that *fails* for
  a benchmark also deletes that benchmark's leftover dump and entry,
  so a later `solve` errors on the missing dump instead of silently
  solving pre-failure VCs. A missing manifest or entry is only a
  warning, so hand-copied dumps stay usable.
- `bench-out/results/<unix-seconds>-<pid>-<command>.json` - the results
  of every run command (timestamped like
  the run logs, so runs never clobber each other). One schema for every
  run: a header (argv, timestamp, iterations, sample, recycle-terms,
  whether `--z3` was on plus its timeout and the iteration carve-out
  convention; `solve` headers add the `backend` and `vcs_from_dumps:
  true`), and one record per benchmark:

  - identity and verdict: `name`, `category`, `status`, `detail`,
    `passed`, `elements_checked`/`elements_total`
  - per-phase timing stats, one set per timed phase
    (`vc_gen_*`, `solve_*`): the full `*_iters_secs` array plus
    `*_median_secs`/`*_min_secs`/`*_mean_secs`/`*_cv`
  - `dump_write_secs` and the `dump_path` of its vcdump (kept even when
    a benchmark fails after its dump was written), and
    `verify_numeric_secs` (oracle time; null unless `--verify-numeric`)
  - `decision_elements`: iteration 1's per-element decision-procedure
    check times as `{array, index, solve_secs}` (empty when no solve
    ran; on a full-footprint run this is one entry per element)
  - `z3`: null without `--z3` (and for benchmarks with no solve phase);
    otherwise the Z3 phase's results - see
    [Comparing against Z3](#comparing-against-z3)
  - instruction and sync counters

  `generate` records carry only the identity/verdict and generation
  fields (no solve fields - nothing was solved); `solve` records carry
  only the identity/verdict and solve fields plus `dump_load_secs` (no
  generation fields - the VCs came from dumps), with skipped race-check
  benchmarks as `status: "SKIP"` records.

### Comparing against Z3

`--z3` adds a Z3 solve phase to the same pipeline (builds against
`libz3` - see [Z3 backend](#z3-backend)):

```bash
cargo run --release -p volta_bench -- --z3 all --json results.json
cargo run --release -p volta_bench -- --z3 category reduction
cargo run --release -p volta_bench -- --z3 single "(Attention, FA1)"
```

Every equivalence benchmark's VCs are then solved by *both* backends -
the exact same sampled elements - and the tables gain two columns: the
median Z3 solve time and Z3's per-element
equivalent/not-equivalent/unknown/timeout/unsupported/error breakdown.
The decision and Z3 columns measure only the deciding work, so they are
comparable: `Solve (s)` is the summed canon equivalence checks (VC
pairing and the optional `--verify-numeric` oracle excluded), and
`Z3 (s)` is in-worker solver time as described in
[Z3 backend](#z3-backend) - worker spawn/exec and translation excluded,
timeout elements counted at their full budget. Both are medians over
`--iterations`, with one Z3 carve-out: an element whose iteration-1
outcome is timeout/unsupported/error is *not* re-solved in later
iterations (re-solving a timeout would multiply its full budget into
every iteration; unsupported/error elements never reach the solver) -
its iteration-1 time is charged to every iteration's total, and the
verdict counts always come from iteration 1. Benchmarks whose VCs
contain exponentials additionally get a `+exp-axiom` sub-row: the same
elements rerun under the paper's addition-law-axiom encoding (expected
outcome: `timeout`, versus `unknown` on the default row - see
[Z3 backend](#z3-backend)). Race-check benchmarks have no VCs and no Z3
phase. `--z3-timeout N` hard-bounds each Z3 query in seconds (default
30, `0` = no limit); the global `--sample` flag applies to both
backends, and `--recycle-terms`/`--verify-numeric` to the decision
procedure only, exactly as without `--z3`. A benchmark whose Z3 phase
fails outright (as opposed to returning unknown/timeout verdicts, which
are data) fails the run. Without `--z3`, Z3 is never invoked.

In the results JSON, each benchmark's `z3` section carries the phase's
full data: `solve_iters_secs` (every iteration, carve-out included) with
median/min/mean/CV, the verdict `counts`, and `elements` - iteration 1's
per-element results as `{array, index, outcome, detail, solve_secs}`
(per-element times for *every* iteration would bloat the document;
iteration-1 elements plus per-iteration totals is the shape). The
`axiom` sub-section repeats all of that for the `+exp-axiom` rerun
(null for exponential-free benchmarks), and `error` is non-null when
the phase failed.

To reproduce the paper's Table 8 exactly - one element per output tensor,
a 10-minute budget per query:

```bash
cargo run --release -p volta_bench -- --sample 1 --z3 --z3-timeout 600 all
```

(Expect the attention/causal rows to spend the full budget per element on
their `+exp-axiom` sub-rows.) One caveat on small machines: the
axiom-induced grind also eats memory, and if z3 exhausts memory before
the deadline it gives up with `unknown` instead of surviving to the kill
(measured on a 15 GiB box: the memory wall arrives after roughly half a
minute of grinding; the paper's 10-minute timeouts assume its 220 GB
machine). Pick a budget below the memory wall - e.g. `--z3-timeout 20` -
to see the `timeout` outcome on constrained hardware, and run one
category at a time.

**Memory note**: symbolic execution plus a warm VC session can use tens of
GiB on the attention benchmarks (each output row retains a large shared
softmax denominator). On machines with limited RAM, run one category at a
time and bound the VC tables, e.g.:

```bash
bash -c 'ulimit -v 12582912; exec cargo run --release -p volta_bench -- \
    --recycle-terms 250000 category attention'
```

which holds peak memory near the symbolic-execution floor (~5 GiB) in
exchange for slower VC checking. The other categories are far lighter
(full matmul: ~2 GiB).

## Citation

```bibtex
@misc{dubey2025equivalencecheckingmlgpu,
      title={Equivalence Checking of ML GPU Kernels},
      author={Kshitij Dubey and Benjamin Driscoll and Anjiang Wei and Neeraj Kayal and Rahul Sharma and Alex Aiken},
      year={2025},
      eprint={2511.12638},
      archivePrefix={arXiv},
      primaryClass={cs.PL},
      url={https://arxiv.org/abs/2511.12638},
}
```

## License

This repository is licensed under [LICENSE](LICENSE).
