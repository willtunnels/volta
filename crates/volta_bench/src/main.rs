//! Volta benchmark runner CLI.
//!
//! Note: run in release mode; symbolic execution of the larger kernels is
//! ~20x slower unoptimized (the binary warns at startup when it isn't).
//!
//! ```bash
//! cargo run --release -p volta_bench -- all --sample 16
//! cargo run --release -p volta_bench -- category reduction
//! cargo run --release -p volta_bench -- --z3 category reduction
//! cargo run --release -p volta_bench -- single "(Red-1, Red-2)"
//! cargo run --release -p volta_bench -- generate all
//! cargo run --release -p volta_bench -- solve all --sample 1 --backend z3
//! cargo run --release -p volta_bench -- list
//! ```

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use volta_analysis::driver::write_op_counts;
use volta_bench::{
    BenchmarkCategory, BenchmarkDef, BenchmarkResult, BenchmarkRunner, KERNELS_DIR, RunnerConfig,
    SolveBackend, SolveItem, TableMode, Z3Options, all_benchmarks, print_all_results,
    print_results_table, print_single_result, print_summary, results,
};
use volta_common::run_log::RunLog;

/// The `--z3-timeout` default, named so the "flag has no effect here"
/// notes below can detect a non-default value.
const DEFAULT_Z3_TIMEOUT_SECS: u64 = 30;

/// Log level for controlling `log`-crate output verbosity (mirrors
/// `volta_cli`'s so the two tools take the same `--log-level` values).
#[cfg(feature = "logging")]
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum LogLevel {
    /// Only show errors
    Error,
    /// Show warnings and errors
    #[default]
    Warn,
    /// Show info, warnings, and errors
    Info,
    /// Show debug output and above
    Debug,
    /// Show all log output including trace
    Trace,
}

#[cfg(feature = "logging")]
impl From<LogLevel> for log::LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
        }
    }
}

#[derive(Parser)]
#[command(name = "volta-bench")]
#[command(about = "Volta benchmark runner - reproduces the paper evaluation")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output (prints progress per benchmark)
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Custom kernels directory
    #[arg(long, global = true)]
    kernels_dir: Option<PathBuf>,

    /// Check at most this many output elements per array (0 = all).
    #[arg(long, global = true, default_value_t = 0)]
    sample: u64,

    /// Confirm every equivalence verdict with the f64 numeric oracle
    #[arg(long, global = true)]
    verify_numeric: bool,

    /// Recycle the VC intern tables past this many interned terms. Lower
    /// values bound VC memory at the cost of re-canonicalizing shared
    /// structure (0 = never recycle).
    #[arg(long, global = true, default_value_t = volta_analysis::equiv::DEFAULT_RECYCLE_TERMS)]
    recycle_terms: usize,

    /// Run every timed phase (VC generation, decision solve, and the Z3
    /// solve under --z3) this many times per benchmark; tables report
    /// medians, the results JSON keeps every iteration
    #[arg(long, global = true, default_value_t = NonZeroUsize::new(10).unwrap())]
    iterations: NonZeroUsize,

    /// Solve each benchmark's VC elements on this many worker threads
    /// (decision solve only; contiguous element chunks, one private
    /// session per worker; --recycle-terms stays the aggregate memory
    /// cap). Verdicts are unaffected; above 1 the summed solve timings
    /// include cross-worker contention (the records add solve_wall_*),
    /// so keep 1 for paper-comparable measurements
    #[arg(long, global = true, default_value_t = NonZeroUsize::MIN)]
    parallel: NonZeroUsize,

    /// Also solve every equivalence benchmark's VCs with Z3 (SMT-LIB2
    /// evaluated via libz3 in a killable worker subprocess) for a
    /// side-by-side comparison; exp-containing benchmarks get a second
    /// +exp-axiom sub-run
    #[arg(long, global = true)]
    z3: bool,

    /// Soft per-query Z3 timeout in seconds under --z3 or `solve
    /// --backend z3|both` (0 = no limit; expiry reports `timeout`)
    #[arg(long, global = true, default_value_t = DEFAULT_Z3_TIMEOUT_SECS)]
    z3_timeout: u64,

    /// Output directory: VC dumps under <out-dir>/vcs/, results JSON
    /// under <out-dir>/results/
    #[arg(long, global = true, default_value = "bench-out")]
    out_dir: PathBuf,

    /// Log level for `log`-crate output verbosity
    #[cfg(feature = "logging")]
    #[arg(long, value_enum, default_value = "warn", global = true)]
    log_level: LogLevel,

    /// Directory for per-run log files
    #[arg(long, global = true, default_value = "volta-logs")]
    log_dir: PathBuf,

    /// Don't write a per-run log file
    #[arg(long, global = true)]
    no_log_file: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all benchmarks
    All {
        /// Also write the results document to this explicit path (the
        /// timestamped file under <out-dir>/results/ is always written)
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Run benchmarks for one category
    Category {
        /// reduction | matmul | attention | causal | conv | agent | tilelang | race
        category: String,
        /// Also write the results document to this explicit path (the
        /// timestamped file under <out-dir>/results/ is always written)
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Run a single benchmark by name
    Single {
        name: String,
        /// Also write the results document to this explicit path (the
        /// timestamped file under <out-dir>/results/ is always written)
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Generate and dump VCs without solving them (the pipeline's
    /// generation phase, x --iterations); race-check benchmarks run to
    /// their real verdicts here. Writes <out-dir>/vcs/<slug>.vcdump and
    /// updates <out-dir>/vcs/manifest.json; `solve` runs from these
    Generate {
        #[command(subcommand)]
        target: Target,
        /// Also write the results document to this explicit path (the
        /// timestamped file under <out-dir>/results/ is always written)
        #[arg(long, global = true)]
        json: Option<PathBuf>,
    },
    /// Solve previously generated VC dumps (see `generate`), x
    /// --iterations - no parsing, lowering, or symbolic execution.
    /// Race-check benchmarks are skipped (their verdicts come from
    /// `generate`)
    Solve {
        #[command(subcommand)]
        target: Target,
        /// Which backend(s) to solve with: the decision procedure, z3
        /// (no decision verdict; the z3 outcomes are the data, except
        /// that a not_equivalent verdict fails the row), or both side by
        /// side. --verify-numeric confirms decision-procedure verdicts,
        /// so it needs decision|both
        #[arg(long, global = true, value_enum, default_value_t = BackendArg::Decision)]
        backend: BackendArg,
        /// Also write the results document to this explicit path (the
        /// timestamped file under <out-dir>/results/ is always written)
        #[arg(long, global = true)]
        json: Option<PathBuf>,
    },
    /// List all benchmarks
    List,
}

/// Which benchmarks a `generate`/`solve` run covers - the same selectors
/// as the one-shot commands.
#[derive(Subcommand)]
enum Target {
    /// All benchmarks
    All,
    /// One category
    Category {
        /// reduction | matmul | attention | causal | conv | agent | tilelang | race
        category: String,
    },
    /// A single benchmark by name
    Single { name: String },
}

/// `--backend` for `solve` (clap surface of [`SolveBackend`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendArg {
    Decision,
    Z3,
    Both,
}

impl From<BackendArg> for SolveBackend {
    fn from(arg: BackendArg) -> Self {
        match arg {
            BackendArg::Decision => Self::Decision,
            BackendArg::Z3 => Self::Z3,
            BackendArg::Both => Self::Both,
        }
    }
}

/// Loud stderr warnings for environments that make the timed phases
/// untrustworthy. Warnings, not errors: the run proceeds either way.
fn print_environment_warnings(#[cfg(feature = "logging")] log_level: LogLevel) {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: built without optimizations (debug_assertions on); \
             timings will be ~20x off - use `cargo run --release`"
        );
    }
    // The `logging` feature compiles volta_analysis's log statements in;
    // at info and above they emit during the timed phases themselves
    // (barrier/warp fires at trace, launch/completion/session recycles at
    // info), so the timings include logging overhead. At error/warn the
    // compiled-in statements only fire on exceptional paths.
    #[cfg(feature = "logging")]
    if log::LevelFilter::from(log_level) >= log::LevelFilter::Info {
        eprintln!(
            "WARNING: the `logging` feature is compiled in and --log-level {:?} \
             emits log output during timed phases; timings will include logging \
             overhead - drop to --log-level warn (or build without the feature) \
             for clean measurements",
            log_level
        );
    }
}

/// Refuse to run a benchmark set whose names collide under the VC-dump
/// slug sanitization: `sanitize_name` is many-to-one and dumps overwrite
/// by slug, so a colliding set would silently clobber one benchmark's
/// dump with another's. Prints both offending names and fails the run
/// before anything executes.
fn ensure_unique_slugs<'a>(
    defs: impl IntoIterator<Item = &'a BenchmarkDef>,
    log: &mut RunLog,
) -> Result<(), ExitCode> {
    match results::check_slug_collisions(defs.into_iter().map(|d| d.name.as_str())) {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("{:#}", e);
            log.record(&format!("{:#}", e));
            Err(ExitCode::FAILURE)
        }
    }
}

/// Handle a stdout print result: ignore a broken pipe (`volta-bench ...
/// | head` closes stdout early, and the run must carry on so the results
/// files still get written), panic on any other write failure.
fn print_stdout(result: anyhow::Result<()>) {
    match result {
        Ok(()) => {}
        Err(e)
            if e.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::BrokenPipe) => {}
        Err(e) => panic!("failed to write to stdout: {:#}", e),
    }
}

/// One stdout line under the [`print_stdout`] broken-pipe policy - for
/// progress and results-path lines, which must not panic before (or
/// between) the results-file writes.
fn println_tolerant(line: String) {
    use std::io::Write;
    print_stdout(writeln!(std::io::stdout(), "{line}").map_err(anyhow::Error::from));
}

/// Write the results document: always to the timestamped file under
/// `<out_dir>/results/`, and additionally to the explicit `--json` path
/// when given. Failures warn rather than abort - a full disk should not
/// repaint a completed benchmark run as failed.
fn write_results(
    out_dir: &Path,
    meta: &results::RunMeta,
    records: Vec<serde_json::Value>,
    json: Option<&Path>,
) {
    let doc = results::results_doc(meta, records);
    match results::write_results_file(out_dir, meta.command, &doc) {
        Ok(path) => println_tolerant(format!("Results written to {}", path.display())),
        Err(e) => eprintln!("warning: {:#}", e),
    }
    if let Some(path) = json {
        match results::write_results_to(path, &doc) {
            Ok(()) => println_tolerant(format!("Results exported to {}", path.display())),
            Err(e) => eprintln!("warning: {:#}", e),
        }
    }
}

/// Announce the Z3 phase once per run: solver version (provenance for
/// the results) and the iteration carve-out convention.
fn announce_z3(iterations: NonZeroUsize) {
    println_tolerant(format!(
        "z3 {} (libz3, worker subprocess)",
        volta_z3::z3_version()
    ));
    println_tolerant(format!(
        "z3 solve: {} iteration(s); elements whose iteration-1 outcome is \
         timeout/unsupported/error are solved once, with that time charged to \
         every iteration",
        iterations
    ));
}

fn main() -> ExitCode {
    // Must precede everything: if this process was spawned as a z3
    // solver worker, this runs the query and exits (see volta_z3::ffi).
    volta_z3::init_worker();

    let cli = Cli::parse();

    let command_name = match &cli.command {
        Commands::All { .. } => "run-all",
        Commands::Category { .. } => "category",
        Commands::Single { .. } => "single",
        Commands::Generate { .. } => "generate",
        Commands::Solve { .. } => "solve",
        Commands::List => "list",
    };
    let mut log = RunLog::open(&cli.log_dir, command_name, cli.no_log_file);

    // env_logger's target borrows the log file (via `tee`) before the
    // command match borrows `log` mutably for `record` - initialize it here.
    #[cfg(feature = "logging")]
    env_logger::Builder::new()
        .filter_level(cli.log_level.into())
        .format_timestamp(None)
        .format_target(false)
        .target(env_logger::Target::Pipe(log.tee(std::io::stderr())))
        .init();

    print_environment_warnings(
        #[cfg(feature = "logging")]
        cli.log_level,
    );

    let out_dir = cli.out_dir.clone();
    let iterations = cli.iterations;
    let z3_timeout_secs = cli.z3.then_some(cli.z3_timeout);
    // `mut`: the `solve` arm records its backend (and derives the Z3
    // phase from it rather than from `--z3`).
    let mut meta = results::RunMeta {
        command: command_name,
        iterations: iterations.get(),
        sample: cli.sample,
        verify_numeric: cli.verify_numeric,
        recycle_terms: cli.recycle_terms,
        parallelism: cli.parallel.get(),
        z3_timeout_secs,
        solve_backend: None,
    };

    let mut runner_config = RunnerConfig {
        kernels_dir: cli
            .kernels_dir
            .unwrap_or_else(|| PathBuf::from(KERNELS_DIR)),
        verbose: cli.verbose,
        sample: cli.sample,
        verify_numeric: cli.verify_numeric,
        recycle_terms: cli.recycle_terms,
        iterations: cli.iterations,
        parallelism: cli.parallel,
        vcs_dir: Some(cli.out_dir.join("vcs")),
        z3: cli.z3.then(|| Z3Options {
            timeout: volta_z3::timeout_from_secs(cli.z3_timeout),
        }),
    };

    let code = match cli.command {
        Commands::All { json } => {
            let suite = all_benchmarks();
            if let Err(code) = ensure_unique_slugs(&suite.benchmarks, &mut log) {
                return finish(log, code);
            }
            println_tolerant(format!("Running {} benchmarks...", suite.benchmarks.len()));
            if cli.z3 {
                announce_z3(iterations);
            }
            let runner = BenchmarkRunner::new(runner_config);
            let run_results = runner.run_all(&suite.benchmarks);
            // Results files first, console tables second: a broken pipe
            // must not lose the files.
            let records = run_results.iter().map(results::benchmark_record).collect();
            write_results(&out_dir, &meta, records, json.as_deref());
            let mut stdout = std::io::stdout();
            print_stdout(print_all_results(
                &mut stdout,
                &run_results,
                iterations.get(),
                cli.z3,
                TableMode::Combined,
            ));
            let passed = run_results.iter().filter(|r| r.passed).count();
            log.record(&format!(
                "run-all: {}/{} benchmarks passed",
                passed,
                run_results.len()
            ));
            exit_by_pass(passed == run_results.len())
        }
        Commands::Category { category, json } => {
            let Some(category) = parse_category(&category) else {
                eprintln!("Unknown category: {}", category);
                eprintln!(
                    "Available: reduction, matmul, attention, causal, conv, agent, tilelang, race"
                );
                log.record(&format!("category: unknown category '{}'", category));
                return finish(log, ExitCode::FAILURE);
            };
            let suite = all_benchmarks();
            let filtered: Vec<_> = suite
                .filter_category(category)
                .into_iter()
                .cloned()
                .collect();
            if let Err(code) = ensure_unique_slugs(&filtered, &mut log) {
                return finish(log, code);
            }
            println_tolerant(format!(
                "Running {} benchmarks for {}...",
                filtered.len(),
                category.name()
            ));
            if cli.z3 {
                announce_z3(iterations);
            }
            let runner = BenchmarkRunner::new(runner_config);
            let run_results = runner.run_all(&filtered);
            // Results files first, console tables second: a broken pipe
            // must not lose the files.
            let records = run_results.iter().map(results::benchmark_record).collect();
            write_results(&out_dir, &meta, records, json.as_deref());
            let mut stdout = std::io::stdout();
            print_stdout(print_results_table(
                &mut stdout,
                &run_results,
                category,
                iterations.get(),
                cli.z3,
                TableMode::Combined,
            ));
            print_stdout(print_summary(&mut stdout, &run_results));
            let passed = run_results.iter().filter(|r| r.passed).count();
            log.record(&format!(
                "category {}: {}/{} benchmarks passed",
                category.name(),
                passed,
                run_results.len()
            ));
            exit_by_pass(passed == run_results.len())
        }
        Commands::Single { name, json } => {
            let suite = all_benchmarks();
            let Some(def) = suite.benchmarks.iter().find(|b| b.name == name) else {
                eprintln!("Benchmark not found: {}", name);
                eprintln!("Use 'volta-bench list' to see available benchmarks.");
                log.record(&format!("single: benchmark not found '{}'", name));
                return finish(log, ExitCode::FAILURE);
            };
            println_tolerant(format!("Running {} ...", name));
            if cli.z3 {
                announce_z3(iterations);
            }
            let runner = BenchmarkRunner::new(runner_config);
            let result = runner.run(def);
            // Results files first, console report second: a broken pipe
            // must not lose the files.
            let records = vec![results::benchmark_record(&result)];
            write_results(&out_dir, &meta, records, json.as_deref());
            let mut stdout = std::io::stdout().lock();
            print_stdout(print_single_result(
                &mut stdout,
                &result,
                TableMode::Combined,
            ));
            print_stdout(
                write_op_counts(&mut stdout, "reference", &result.stats.reference_op_counts)
                    .map_err(anyhow::Error::from),
            );
            print_stdout(
                write_op_counts(&mut stdout, "optimized", &result.stats.optimized_op_counts)
                    .map_err(anyhow::Error::from),
            );
            if !result.passed {
                print_stdout(print_summary(&mut stdout, std::slice::from_ref(&result)));
            }
            drop(stdout);
            log.record(&format!(
                "single {}: {} ({})",
                name,
                result.outcome.status(),
                if result.passed { "pass" } else { "FAIL" }
            ));
            exit_by_pass(result.passed)
        }
        Commands::Generate { target, json } => {
            if cli.z3 {
                eprintln!(
                    "--z3 does not apply to `generate` (nothing is solved); use the \
                     one-shot commands, or `solve --backend z3|both` on the dumps"
                );
                log.record("generate: rejected --z3");
                return finish(log, ExitCode::FAILURE);
            }
            // Solve-phase flags don't change what `generate` produces;
            // say so instead of silently ignoring them.
            let mut ignored = Vec::new();
            if cli.sample != 0 {
                ignored.push("--sample");
            }
            if cli.verify_numeric {
                ignored.push("--verify-numeric");
            }
            if cli.recycle_terms != volta_analysis::equiv::DEFAULT_RECYCLE_TERMS {
                ignored.push("--recycle-terms");
            }
            if cli.parallel.get() != 1 {
                ignored.push("--parallel");
            }
            if cli.z3_timeout != DEFAULT_Z3_TIMEOUT_SECS {
                ignored.push("--z3-timeout");
            }
            if !ignored.is_empty() {
                eprintln!(
                    "note: {} ignored by `generate` (solve-phase options; pass them \
                     to `solve`)",
                    ignored.join(", ")
                );
            }
            let defs = match resolve_target(&target, "generate", &mut log) {
                Ok(defs) => defs,
                Err(code) => return finish(log, code),
            };
            if let Err(code) = ensure_unique_slugs(&defs, &mut log) {
                return finish(log, code);
            }
            println_tolerant(format!(
                "Generating VCs for {} benchmark(s) into {} ...",
                defs.len(),
                out_dir.join("vcs").display()
            ));
            let runner = BenchmarkRunner::new(runner_config);
            let run_results = runner.generate_all(&defs);
            // Results files first, console tables second: a broken pipe
            // must not lose the files.
            let records = run_results.iter().map(results::generate_record).collect();
            write_results(&out_dir, &meta, records, json.as_deref());
            print_phase_results(
                &run_results,
                &target,
                iterations.get(),
                false,
                TableMode::GenerateOnly,
            );
            let passed = run_results.iter().filter(|r| r.passed).count();
            log.record(&format!(
                "generate: {}/{} benchmarks passed",
                passed,
                run_results.len()
            ));
            exit_by_pass(passed == run_results.len())
        }
        Commands::Solve {
            target,
            backend,
            json,
        } => {
            if cli.z3 {
                eprintln!(
                    "--z3 does not apply to `solve`; pick the backend with \
                     `--backend decision|z3|both`"
                );
                log.record("solve: rejected --z3");
                return finish(log, ExitCode::FAILURE);
            }
            let backend = SolveBackend::from(backend);
            // Flags for a phase this backend doesn't run are noted and
            // ignored (the run proceeds), like `generate`'s note.
            if cli.verify_numeric && !backend.runs_decision() {
                eprintln!(
                    "note: --verify-numeric confirms decision-procedure verdicts; \
                     it has no effect with --backend z3"
                );
            }
            if cli.recycle_terms != volta_analysis::equiv::DEFAULT_RECYCLE_TERMS
                && !backend.runs_decision()
            {
                eprintln!(
                    "note: --recycle-terms tunes the decision procedure's intern \
                     tables; it has no effect with --backend z3"
                );
            }
            if cli.parallel.get() != 1 && !backend.runs_decision() {
                eprintln!(
                    "note: --parallel parallelizes the decision solve's element \
                     loop; it has no effect with --backend z3"
                );
            }
            if cli.z3_timeout != DEFAULT_Z3_TIMEOUT_SECS && !backend.runs_z3() {
                eprintln!(
                    "note: --z3-timeout bounds z3 queries; it has no effect with \
                     --backend decision"
                );
            }
            let defs = match resolve_target(&target, "solve", &mut log) {
                Ok(defs) => defs,
                Err(code) => return finish(log, code),
            };
            if let Err(code) = ensure_unique_slugs(&defs, &mut log) {
                return finish(log, code);
            }
            // The Z3 phase follows the backend, not `--z3`; the header
            // records the backend and that VCs come from dumps.
            meta.solve_backend = Some(backend.name());
            meta.z3_timeout_secs = backend.runs_z3().then_some(cli.z3_timeout);
            runner_config.z3 = backend.runs_z3().then(|| Z3Options {
                timeout: volta_z3::timeout_from_secs(cli.z3_timeout),
            });
            if backend.runs_z3() {
                announce_z3(iterations);
            }
            println_tolerant(format!(
                "Solving {} benchmark(s) from dumps in {} ...",
                defs.len(),
                out_dir.join("vcs").display()
            ));
            let runner = BenchmarkRunner::new(runner_config);
            let items = match runner.solve_suite(&defs, backend) {
                Ok(items) => items,
                Err(e) => {
                    eprintln!("{:#}", e);
                    log.record(&format!("solve: {:#}", e));
                    return finish(log, ExitCode::FAILURE);
                }
            };
            // Results files first, console tables second: a broken pipe
            // must not lose the files.
            let records = items
                .iter()
                .map(|item| match item {
                    SolveItem::Solved(r) => results::solve_record(r),
                    SolveItem::Skipped { name, category } => {
                        results::skip_record(name, *category, volta_bench::RACE_SKIP_NOTE)
                    }
                })
                .collect();
            write_results(&out_dir, &meta, records, json.as_deref());
            let solved: Vec<BenchmarkResult> = items
                .into_iter()
                .filter_map(|item| match item {
                    SolveItem::Solved(r) => Some(*r),
                    SolveItem::Skipped { .. } => None,
                })
                .collect();
            let skipped = defs.len() - solved.len();
            print_phase_results(
                &solved,
                &target,
                iterations.get(),
                backend.runs_z3(),
                TableMode::SolveOnly,
            );
            if solved.is_empty() {
                println_tolerant(format!(
                    "Nothing to solve: all {} selected benchmark(s) are race checks \
                     (their verdicts come from `volta-bench generate`)",
                    skipped
                ));
            }
            let passed = solved.iter().filter(|r| r.passed).count();
            log.record(&format!(
                "solve[{}]: {}/{} solved benchmarks passed, {} skipped",
                backend.name(),
                passed,
                solved.len(),
                skipped
            ));
            exit_by_pass(passed == solved.len())
        }
        Commands::List => {
            let suite = all_benchmarks();
            for category in suite.categories() {
                println!("{}:", category.name());
                for b in suite.filter_category(category) {
                    println!("  - {}", b.name);
                }
            }
            println!("Total: {} benchmarks", suite.benchmarks.len());
            log.record("list");
            ExitCode::SUCCESS
        }
    };

    finish(log, code)
}

/// Print the run-log path (if any) and return the exit code - the last
/// thing every command path does, mirroring `volta_cli`.
fn finish(log: RunLog, code: ExitCode) -> ExitCode {
    if let Some(path) = log.path() {
        eprintln!("log: {}", path.display());
    }
    code
}

fn exit_by_pass(passed: bool) -> ExitCode {
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Resolve a `generate`/`solve` target to its benchmark list, with the
/// same console errors as the one-shot commands on an unknown category
/// or benchmark name.
fn resolve_target(
    target: &Target,
    command: &str,
    log: &mut RunLog,
) -> Result<Vec<BenchmarkDef>, ExitCode> {
    let suite = all_benchmarks();
    match target {
        Target::All => Ok(suite.benchmarks),
        Target::Category { category } => {
            let Some(parsed) = parse_category(category) else {
                eprintln!("Unknown category: {}", category);
                eprintln!(
                    "Available: reduction, matmul, attention, causal, conv, agent, tilelang, race"
                );
                log.record(&format!("{}: unknown category '{}'", command, category));
                return Err(ExitCode::FAILURE);
            };
            Ok(suite.filter_category(parsed).into_iter().cloned().collect())
        }
        Target::Single { name } => match suite.benchmarks.iter().find(|b| b.name == *name) {
            Some(def) => Ok(vec![def.clone()]),
            None => {
                eprintln!("Benchmark not found: {}", name);
                eprintln!("Use 'volta-bench list' to see available benchmarks.");
                log.record(&format!("{}: benchmark not found '{}'", command, name));
                Err(ExitCode::FAILURE)
            }
        },
    }
}

/// Console presentation for a `generate`/`solve` run: category tables
/// (plus summary) for `all`/`category` targets, the detailed
/// single-benchmark report for `single` - mirroring the one-shot
/// commands' presentation.
fn print_phase_results(
    results: &[BenchmarkResult],
    target: &Target,
    iterations: usize,
    z3: bool,
    mode: TableMode,
) {
    let mut stdout = std::io::stdout().lock();
    match target {
        Target::All | Target::Category { .. } => {
            print_stdout(print_all_results(
                &mut stdout,
                results,
                iterations,
                z3,
                mode,
            ));
        }
        Target::Single { .. } => {
            // Empty exactly when the one selected benchmark was skipped
            // (a race check under `solve`); the skip note already printed.
            let Some(result) = results.first() else {
                return;
            };
            print_stdout(print_single_result(&mut stdout, result, mode));
            // Execution profiles exist only when this run executed the
            // kernels (`generate`); `write_op_counts` skips empty maps.
            print_stdout(
                write_op_counts(&mut stdout, "reference", &result.stats.reference_op_counts)
                    .map_err(anyhow::Error::from),
            );
            print_stdout(
                write_op_counts(&mut stdout, "optimized", &result.stats.optimized_op_counts)
                    .map_err(anyhow::Error::from),
            );
            if !result.passed {
                print_stdout(print_summary(&mut stdout, std::slice::from_ref(result)));
            }
        }
    }
}

fn parse_category(name: &str) -> Option<BenchmarkCategory> {
    match name.to_lowercase().as_str() {
        "reduction" | "red" => Some(BenchmarkCategory::Reduction),
        "matmul" | "mm" => Some(BenchmarkCategory::MatMul),
        "attention" | "attn" => Some(BenchmarkCategory::Attention),
        "causal" | "causal-attention" | "causal-attn" => Some(BenchmarkCategory::CausalAttention),
        "convolution" | "conv" => Some(BenchmarkCategory::Convolution),
        "agent" | "agent-generated" => Some(BenchmarkCategory::AgentGenerated),
        "compiler" | "compiler-generated" | "tilelang" | "tl" => {
            Some(BenchmarkCategory::CompilerGenerated)
        }
        "datarace" | "race" | "races" => Some(BenchmarkCategory::DataRace),
        _ => None,
    }
}
