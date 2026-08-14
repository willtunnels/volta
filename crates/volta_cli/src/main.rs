//! Volta CLI - PTX analysis tool
//!
//! Commands:
//! - `volta parse <file>` - Parse a PTX file and report any errors
//! - `volta analyze <file>` - Symbolically execute one kernel
//! - `volta compare <file1> <file2>` - Check two kernels for equivalence

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand, ValueEnum};
use volta_analysis::driver::{
    EquivCheckOptions, EquivOutcome, VcDump, VcSnapshot, analyze_kernel,
    check_output_equivalence_with,
    vc_dump::{read_vc_dump, write_vc_dump},
    write_op_counts,
};
use volta_analysis::equiv::DEFAULT_RECYCLE_TERMS;
use volta_analysis::eval::{AnalysisConfig, AnalysisOutput, ArrayDef, ArrayKind, ParamValue};
use volta_common::run_log;
use volta_frontend::ascii::{AsAscii, AsciiChar};
use volta_frontend::ast::{Module, TopLevelItem};
use volta_frontend::file_cache::FileCache;
use volta_frontend::parse;
use volta_frontend::report::{Report, locate_path, report_error};

/// Log level for controlling output verbosity
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

/// Which decision procedure to check equivalence with.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
    /// `volta_analysis::canon` - Volta's own decision procedure (default)
    Decision,
    /// Z3, via SMT-LIB2 evaluated worker subprocess through libz3. Covers a
    /// narrower fragment (see `volta_z3::translate`'s docs) and reports
    /// per-element unsat/sat/unknown rather than a single verdict.
    Z3,
}

#[derive(Parser)]
#[command(name = "volta")]
#[command(about = "The Volta PTX analysis engine.")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Log level for output verbosity
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
    /// Parse a PTX file and check for syntax errors
    Parse {
        /// PTX file to parse
        file: PathBuf,
    },

    /// Symbolically execute a kernel: detect races/deadlocks and print the
    /// symbolic output tensors
    Analyze(AnalyzeArgs),

    /// Check two kernels for semantic equivalence (each is also checked for
    /// data races/deadlocks). Arrays/params/globals are shared by both
    /// kernels unless a `--block2`/`--grid2` override is given.
    Compare(CompareArgs),
}

/// Launch configuration shared by `analyze` and `compare`: the flags that
/// feed `build_config`. Flattened into both subcommands so their
/// block/grid/array/param/global handling cannot drift apart. (For
/// `compare` these describe the reference kernel; `--block2`/`--grid2` on
/// `CompareArgs` override the dims for the optimized kernel only.)
#[derive(Args)]
struct LaunchArgs {
    /// Block dimensions, e.g. "128" or "32,4,1"
    #[arg(short, long, default_value = "1")]
    block: String,

    /// Grid dimensions, e.g. "64,64"
    #[arg(short, long, default_value = "1")]
    grid: String,

    /// Global array: "name:base:elem_width:len:kind" where kind is
    /// in|out|inout|index (e.g. "in:0x10000:4:128:in"). Repeatable.
    #[arg(long = "array")]
    arrays: Vec<String>,

    /// Kernel parameter (in declaration order): "int:N", "float:X",
    /// "sym:name", or "ptr:array_name". Repeatable.
    #[arg(long = "param")]
    params: Vec<String>,

    /// Module-scope .global variable value: "NAME=value". Repeatable.
    #[arg(long = "global")]
    globals: Vec<String>,

    /// Dynamic (extern) shared memory size in bytes
    #[arg(long, default_value_t = 0)]
    dyn_shared: u64,
}

#[derive(Args)]
struct AnalyzeArgs {
    /// PTX file to analyze
    file: PathBuf,

    /// Kernel entry name (defaults to the first kernel in the module)
    #[arg(short, long)]
    kernel: Option<String>,

    #[command(flatten)]
    launch: LaunchArgs,

    /// Print up to N elements of each output array
    #[arg(long, default_value_t = 8)]
    print_outputs: u64,

    /// Skip the per-instruction-kind execution profile (shown by default)
    #[arg(long = "no-profile", action = clap::ArgAction::SetFalse, default_value_t = true)]
    profile: bool,
}

#[derive(Args)]
struct CompareArgs {
    /// Reference PTX file (omit when using --from-dump)
    file1: Option<PathBuf>,

    /// Optimized PTX file (omit when using --from-dump)
    file2: Option<PathBuf>,

    /// Reference kernel entry name (defaults to the first in the module)
    #[arg(long)]
    kernel1: Option<String>,

    /// Optimized kernel entry name (defaults to the first in the module)
    #[arg(long)]
    kernel2: Option<String>,

    #[command(flatten)]
    launch: LaunchArgs,

    /// Block dimensions for the optimized kernel only, if it differs
    #[arg(long)]
    block2: Option<String>,

    /// Grid dimensions for the optimized kernel only, if it differs
    #[arg(long)]
    grid2: Option<String>,

    /// Output array to check, by name (repeatable, at least one
    /// required; order = check order). Each name must be an output
    /// array declared in the launch config (with --from-dump: recorded
    /// in the dump).
    #[arg(long = "check-array", value_name = "NAME", required = true)]
    check_arrays: Vec<String>,

    /// Check at most this many common elements per array (0 = all)
    #[arg(long, default_value_t = 0)]
    sample: u64,

    /// Confirm every verdict with the f64 numeric oracle
    #[arg(long)]
    verify_numeric: bool,

    /// Recycle the VC intern tables past this many interned terms (0 = never)
    #[arg(long, default_value_t = DEFAULT_RECYCLE_TERMS)]
    recycle_terms: usize,

    /// Run the solve phase N times (fresh session each; verdict from
    /// iteration 1, later iterations must agree). Decision backend only.
    #[arg(long, default_value_t = NonZeroUsize::MIN)]
    iterations: NonZeroUsize,

    /// Solve the VC elements on N worker threads (contiguous element
    /// chunks, one private session per worker; --recycle-terms stays the
    /// aggregate memory cap). Verdicts are unaffected; above 1 the
    /// reported decision-procedure time is summed across workers and
    /// exceeds wall clock. Decision backend only.
    #[arg(long, default_value_t = NonZeroUsize::MIN)]
    parallel: NonZeroUsize,

    /// Skip the per-instruction-kind execution profile (shown by default)
    #[arg(long = "no-profile", action = clap::ArgAction::SetFalse, default_value_t = true)]
    profile: bool,

    /// Which decision procedure to use
    #[arg(long, value_enum, default_value = "decision")]
    backend: BackendArg,

    /// Per-query Z3 timeout in seconds, only used with --backend z3 (0 =
    /// no limit). A hard bound: the solver worker is killed on expiry.
    #[arg(long, default_value_t = 30)]
    z3_timeout: u64,

    /// With --backend z3: encode the exponential as an uninterpreted
    /// function with the addition-law axiom (forall x y. e^x e^y =
    /// e^(x+y)) instead of the default bounded-base power encoding - the
    /// paper's "Z3 with axiom" baseline, which drives Z3 into a timeout
    /// on softmax-shaped VCs rather than a fast `unknown`.
    #[arg(long)]
    exp_axiom: bool,

    /// After symbolic execution, dump both kernels' verification
    /// conditions (the expression arena + output footprint) to this
    /// file. Reload them later with --from-dump to rerun the
    /// equivalence check without parsing/symbolic execution.
    #[arg(long)]
    dump_vcs: Option<PathBuf>,

    /// Skip parsing and symbolic execution entirely and check
    /// equivalence directly from a --dump-vcs file. FILE1/FILE2 and the
    /// launch-config flags are ignored when this is set.
    #[arg(long)]
    from_dump: Option<PathBuf>,
}

fn main() -> ExitCode {
    // Must precede everything: if this process was spawned as a z3
    // solver worker, this runs the query and exits (see volta_z3::ffi).
    volta_z3::init_worker();

    let cli = Cli::parse();

    let command_name = match &cli.command {
        Commands::Parse { .. } => "parse",
        Commands::Analyze(_) => "analyze",
        Commands::Compare(_) => "compare",
    };
    let mut log = run_log::RunLog::open(&cli.log_dir, command_name, cli.no_log_file);

    #[cfg(feature = "logging")]
    env_logger::Builder::new()
        .filter_level(cli.log_level.into())
        .format_timestamp(None)
        .format_target(false)
        .target(env_logger::Target::Pipe(log.tee(std::io::stderr())))
        .init();

    let code = match cli.command {
        Commands::Parse { file } => cmd_parse(&file),
        Commands::Analyze(args) => cmd_analyze(args, &mut log),
        Commands::Compare(args) => cmd_compare(args, &mut log),
    };

    if let Some(path) = log.path() {
        eprintln!("log: {}", path.display());
    }
    code
}

/// Parse "x[,y[,z]]" dimensions.
fn parse_dims(s: &str) -> Result<(u32, u32, u32), String> {
    let mut parts = s.split(',').map(|p| p.trim().parse::<u32>());
    let x = parts
        .next()
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(1);
    let y = parts
        .next()
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(1);
    let z = parts
        .next()
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(1);
    Ok((x, y, z))
}

fn parse_u64_value(s: &str) -> Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        s.parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())
    }
}

/// Parse "name:base:elem_width:len:kind".
fn parse_array(s: &str) -> Result<ArrayDef, String> {
    let parts: Vec<&str> = s.split(':').collect();
    let [name, base, elem_width, len, kind] = parts.as_slice() else {
        return Err(format!(
            "expected name:base:elem_width:len:kind, got '{}'",
            s
        ));
    };
    let kind = match *kind {
        "in" => ArrayKind::Input,
        "out" => ArrayKind::Output,
        "inout" => ArrayKind::InputOutput,
        "index" => ArrayKind::IndexInput,
        other => return Err(format!("unknown array kind '{}'", other)),
    };
    Ok(ArrayDef {
        name: name.to_string(),
        base: parse_u64_value(base)?,
        elem_width: parse_u64_value(elem_width)?,
        len: parse_u64_value(len)?,
        kind,
    })
}

/// Parse "int:N" | "float:X" | "sym:name" | "ptr:array".
fn parse_param(s: &str) -> Result<ParamValue, String> {
    let Some((kind, value)) = s.split_once(':') else {
        return Err(format!("expected kind:value, got '{}'", s));
    };
    match kind {
        "int" => Ok(ParamValue::Int(
            value.parse().map_err(|e| format!("{}", e))?,
        )),
        "float" => Ok(ParamValue::Float(
            value.parse().map_err(|e| format!("{}", e))?,
        )),
        "sym" => Ok(ParamValue::SymFloat(value.to_string())),
        "ptr" => Ok(ParamValue::ArrayPtr(value.to_string())),
        other => Err(format!("unknown param kind '{}'", other)),
    }
}

/// Shared launch-config inputs, parsed into an `AnalysisConfig` by
/// `build_config`. Used by both `analyze` and `compare` so the two
/// commands' array/param/global flags behave identically.
struct ConfigInput<'a> {
    block: &'a str,
    grid: &'a str,
    arrays: &'a [String],
    params: &'a [String],
    globals: &'a [String],
    dyn_shared: u64,
}

impl<'a> ConfigInput<'a> {
    /// The reference-kernel launch config straight from the shared flags.
    /// `compare` overrides `block`/`grid` afterwards for the optimized
    /// kernel via `--block2`/`--grid2`.
    fn from_launch(launch: &'a LaunchArgs) -> Self {
        Self {
            block: &launch.block,
            grid: &launch.grid,
            arrays: &launch.arrays,
            params: &launch.params,
            globals: &launch.globals,
            dyn_shared: launch.dyn_shared,
        }
    }
}

fn build_config(input: ConfigInput) -> Result<AnalysisConfig, String> {
    let block_dim = parse_dims(input.block).map_err(|e| format!("invalid --block: {}", e))?;
    let mut config = AnalysisConfig::new(block_dim);
    config.grid_dim = parse_dims(input.grid).map_err(|e| format!("invalid --grid: {}", e))?;
    config.dynamic_shared_bytes = input.dyn_shared;
    for a in input.arrays {
        config
            .arrays
            .push(parse_array(a).map_err(|e| format!("invalid --array: {}", e))?);
    }
    for p in input.params {
        config
            .params
            .push(parse_param(p).map_err(|e| format!("invalid --param: {}", e))?);
    }
    for g in input.globals {
        let (name, value) = g
            .split_once('=')
            .ok_or_else(|| format!("invalid --global (expected NAME=value): {}", g))?;
        let v: i64 = value
            .parse()
            .map_err(|e: std::num::ParseIntError| format!("invalid --global value: {}", e))?;
        config.global_values.push((name.to_string(), v));
    }
    Ok(config)
}

fn cmd_analyze(args: AnalyzeArgs, log: &mut run_log::RunLog) -> ExitCode {
    let file = &args.file;
    let module = match load_module(file) {
        Ok(m) => m,
        Err(code) => return code,
    };

    let config = match build_config(ConfigInput::from_launch(&args.launch)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let start = Instant::now();
    let result = analyze_kernel(&module, args.kernel.as_deref(), config);
    let elapsed = start.elapsed().as_secs_f64();

    match result {
        Ok(output) => {
            println!("Analysis complete: no data races or deadlocks detected.");
            println!(
                "  time: {:.3}s  instructions: {}  block syncs: {}  warp syncs: {}",
                elapsed,
                output.stats.instructions,
                output.stats.block_syncs,
                output.stats.warp_syncs
            );
            for (name, elems) in &output.outputs {
                println!("  output '{}': {} element(s) written", name, elems.len());
                for (index, expr) in elems.iter().take(args.print_outputs as usize) {
                    println!(
                        "    {}[{}] = {}",
                        name,
                        index,
                        output.arena.display_expr(*expr)
                    );
                }
                if elems.len() as u64 > args.print_outputs {
                    println!("    ... ({} more)", elems.len() as u64 - args.print_outputs);
                }
            }
            if args.profile {
                let _ = write_op_counts(
                    &mut std::io::stdout().lock(),
                    "instruction",
                    &output.op_counts,
                );
            }
            log.record(&format!(
                "analyze {}: OK in {:.3}s, {} instructions",
                file.display(),
                elapsed,
                output.stats.instructions
            ));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("analysis failed: {}", e);
            log.record(&format!("analyze {}: FAILED: {}", file.display(), e));
            ExitCode::FAILURE
        }
    }
}

/// How a whole Z3 comparison run maps to a process exit status. `compare`
/// is a verification command, so only a run that *proved* every element
/// equivalent may exit 0; an undecided-only run is distinguished from a
/// real difference purely so we can explain the nonzero exit.
#[derive(Debug, PartialEq, Eq)]
enum Z3Verdict {
    /// Every checked element proved equivalent (vacuously true for an empty
    /// footprint) - exit 0.
    AllEquivalent,
    /// At least one element is definitively not equivalent - exit nonzero.
    HasDifference,
    /// No differences, but some elements could not be decided
    /// (unknown/unsupported/solver error) - exit nonzero, since undecided
    /// is not a proof.
    OnlyUndecided { undecided: usize },
}

fn z3_verdict(counts: &volta_z3::Z3Counts) -> Z3Verdict {
    if counts.all_equivalent() {
        Z3Verdict::AllEquivalent
    } else if counts.not_equivalent > 0 {
        Z3Verdict::HasDifference
    } else {
        Z3Verdict::OnlyUndecided {
            undecided: counts.unknown + counts.timeout + counts.unsupported + counts.error,
        }
    }
}

/// Reject duplicate `--check-array` names - checking an array twice only
/// double-counts its elements. Clap enforces that at least one name was
/// given (the list is always explicit, matching `paired_elements`'
/// interface - there is no derived default); membership is checked
/// against the declared launch config before symbolic execution, and by
/// `paired_elements` (the authority) after.
fn validate_check_arrays(arrays: &[String]) -> Result<(), String> {
    for (i, name) in arrays.iter().enumerate() {
        if arrays[..i].contains(name) {
            return Err(format!("--check-array '{}' given more than once", name));
        }
    }
    Ok(())
}

fn cmd_compare(args: CompareArgs, log: &mut run_log::RunLog) -> ExitCode {
    if let Err(e) = validate_check_arrays(&args.check_arrays) {
        eprintln!("error: {}", e);
        return ExitCode::FAILURE;
    }
    if args.from_dump.is_some() && args.dump_vcs.is_some() {
        eprintln!("note: --dump-vcs is a no-op with --from-dump (nothing new to dump)");
    }
    if args.exp_axiom && !matches!(args.backend, BackendArg::Z3) {
        eprintln!("note: --exp-axiom only affects --backend z3");
    }
    if args.iterations.get() > 1 && matches!(args.backend, BackendArg::Z3) {
        eprintln!("note: --iterations only affects --backend decision");
    }
    if args.parallel.get() > 1 && matches!(args.backend, BackendArg::Z3) {
        eprintln!("note: --parallel only affects --backend decision");
    }

    let (reference, optimized, exec_secs): (AnalysisOutput, AnalysisOutput, Option<f64>) =
        if let Some(dump_path) = &args.from_dump {
            if args.file1.is_some() || args.file2.is_some() {
                eprintln!("note: --from-dump ignores FILE1/FILE2 and the launch-config flags");
            }
            let dump = match read_vc_dump(dump_path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("error: failed to read dump {}: {}", dump_path.display(), e);
                    return ExitCode::FAILURE;
                }
            };
            (
                dump.reference.into_analysis_output(),
                dump.optimized.into_analysis_output(),
                None,
            )
        } else {
            let (Some(file1), Some(file2)) = (args.file1.as_ref(), args.file2.as_ref()) else {
                eprintln!("error: compare needs FILE1 and FILE2 (or --from-dump)");
                return ExitCode::FAILURE;
            };

            let module1 = match load_module(file1) {
                Ok(m) => m,
                Err(code) => return code,
            };
            let module2 = match load_module(file2) {
                Ok(m) => m,
                Err(code) => return code,
            };

            let config1 = match build_config(ConfigInput::from_launch(&args.launch)) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            let config2 = match build_config(ConfigInput {
                block: args.block2.as_deref().unwrap_or(&args.launch.block),
                grid: args.grid2.as_deref().unwrap_or(&args.launch.grid),
                ..ConfigInput::from_launch(&args.launch)
            }) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };

            // Fail fast on a name no declared output array matches: a
            // run's recorded outputs are exactly the config's out/inout
            // arrays (and the config is shared by both kernels), so this
            // catches a typo before symbolic execution, which can run
            // minutes. `paired_elements` stays the authority afterwards.
            for name in &args.check_arrays {
                if !config1
                    .arrays
                    .iter()
                    .any(|a| a.kind.is_output() && &a.name == name)
                {
                    eprintln!(
                        "error: --check-array '{}' is not a declared output array",
                        name
                    );
                    return ExitCode::FAILURE;
                }
            }

            let start = Instant::now();
            let reference = match analyze_kernel(&module1, args.kernel1.as_deref(), config1) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("error: reference kernel: {}", e);
                    log.record(&format!("compare: reference kernel FAILED: {}", e));
                    return ExitCode::FAILURE;
                }
            };
            let optimized = match analyze_kernel(&module2, args.kernel2.as_deref(), config2) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("error: optimized kernel: {}", e);
                    log.record(&format!("compare: optimized kernel FAILED: {}", e));
                    return ExitCode::FAILURE;
                }
            };
            let exec_secs = start.elapsed().as_secs_f64();

            if args.profile {
                let mut out = std::io::stdout().lock();
                let _ = write_op_counts(&mut out, "reference instruction", &reference.op_counts);
                let _ = write_op_counts(&mut out, "optimized instruction", &optimized.op_counts);
            }
            println!(
                "Exec: {:.3}s  instructions: {}  block syncs: {}  warp syncs: {}",
                exec_secs,
                reference.stats.instructions + optimized.stats.instructions,
                optimized.stats.block_syncs,
                optimized.stats.warp_syncs
            );

            if let Some(dump_path) = &args.dump_vcs {
                let dump = VcDump {
                    reference: VcSnapshot::from_output(reference),
                    optimized: VcSnapshot::from_output(optimized),
                };
                if let Err(e) = write_vc_dump(dump_path, &dump) {
                    eprintln!("error: failed to write dump {}: {}", dump_path.display(), e);
                    return ExitCode::FAILURE;
                }
                println!("Dumped verification conditions to {}", dump_path.display());
                (
                    dump.reference.into_analysis_output(),
                    dump.optimized.into_analysis_output(),
                    Some(exec_secs),
                )
            } else {
                (reference, optimized, Some(exec_secs))
            }
        };

    if exec_secs.is_none() {
        println!("Loaded verification conditions from dump (no fresh symbolic execution).");
    }

    // The arrays to check: exactly the --check-array names, in their given
    // order. `paired_elements` rejects any that both runs don't have.
    let check_arrays: &[String] = &args.check_arrays;

    match args.backend {
        BackendArg::Decision => {
            let options = EquivCheckOptions {
                sample: args.sample,
                verify_numeric: args.verify_numeric,
                recycle_terms: args.recycle_terms,
                iterations: args.iterations,
                parallelism: args.parallel,
            };
            let vc_start = Instant::now();
            let report =
                check_output_equivalence_with(&reference, &optimized, check_arrays, &options);
            let vc_secs = vc_start.elapsed().as_secs_f64();

            match report {
                Ok(report) => {
                    let elems = if report.elements_checked == report.elements_total {
                        format!("{}", report.elements_total)
                    } else {
                        format!("{}/{}", report.elements_checked, report.elements_total)
                    };
                    // Wall clock for the whole phase; the parenthetical is
                    // the decision procedure alone (summed canon checks -
                    // pairing and the optional numeric oracle excluded),
                    // the figure comparable to the z3 backend's solver
                    // time.
                    println!(
                        "VC check: {:.3}s (decision procedure {:.3}s)  elements: {}",
                        vc_secs,
                        report.check_time().as_secs_f64(),
                        elems
                    );
                    if args.parallel.get() > 1 {
                        println!(
                            "  ({} workers; decision-procedure time is summed across \
                             workers, wall clock {:.3}s)",
                            args.parallel,
                            report.wall_iters[0].as_secs_f64()
                        );
                    }
                    if report.check_iters.len() > 1 {
                        let per_iter: Vec<String> = report
                            .check_iters
                            .iter()
                            .map(|d| format!("{:.3}", d.as_secs_f64()))
                            .collect();
                        println!(
                            "  decision procedure iterations (s): [{}]",
                            per_iter.join(", ")
                        );
                    }
                    match report.outcome {
                        EquivOutcome::Equivalent => {
                            println!("EQUIVALENT");
                            log.record(&format!(
                                "compare: EQUIVALENT ({} elements, exec {:?}, vc {:.3}s)",
                                elems, exec_secs, vc_secs
                            ));
                            ExitCode::SUCCESS
                        }
                        EquivOutcome::NotEquivalent { mismatches } => {
                            println!("NOT EQUIVALENT: {} mismatched element(s)", mismatches.len());
                            for m in mismatches.iter().take(10) {
                                println!("  {}[{}]", m.array, m.index);
                            }
                            if mismatches.len() > 10 {
                                println!("  ... ({} more)", mismatches.len() - 10);
                            }
                            log.record(&format!(
                                "compare: NOT EQUIVALENT, {} mismatches",
                                mismatches.len()
                            ));
                            ExitCode::FAILURE
                        }
                    }
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    log.record(&format!("compare: FAILED: {}", e));
                    ExitCode::FAILURE
                }
            }
        }
        BackendArg::Z3 => {
            let timeout = volta_z3::timeout_from_secs(args.z3_timeout);
            let mode = if args.exp_axiom {
                volta_z3::ExpMode::AdditionAxiom
            } else {
                volta_z3::ExpMode::PowerBounded
            };
            let vc_start = Instant::now();
            let report = volta_z3::check_output_equivalence(
                &reference,
                &optimized,
                check_arrays,
                args.sample,
                timeout,
                mode,
            );
            let vc_secs = vc_start.elapsed().as_secs_f64();

            match report {
                Ok(report) => {
                    let counts = report.counts();
                    // Wall clock for the whole phase; the parenthetical is
                    // in-worker solver time only (worker spawn/exec and
                    // translation excluded; timeouts count their budget).
                    println!(
                        "VC check: {:.3}s (z3 solver time {:.3}s)  elements: {}",
                        vc_secs,
                        report.total_solve_secs(),
                        report.elements.len()
                    );
                    println!("  {}", counts);
                    for e in report
                        .elements
                        .iter()
                        .filter(|e| !matches!(e.outcome, volta_z3::ElementOutcome::Equivalent))
                        .take(10)
                    {
                        println!("  {}[{}]: {:?}", e.array, e.index, e.outcome);
                    }
                    log.record(&format!(
                        "compare (z3): {} (vc {:.3}s)",
                        counts.compact(),
                        vc_secs
                    ));
                    // `compare` is a verification command: exit 0 must mean
                    // "every element proved equivalent". A definitive
                    // NotEquivalent aside, a run that only failed to *decide*
                    // some elements (unknown/unsupported/solver error) still
                    // exits nonzero, so no undecided result is mistaken for a
                    // proof.
                    match z3_verdict(&counts) {
                        Z3Verdict::AllEquivalent => ExitCode::SUCCESS,
                        Z3Verdict::HasDifference => ExitCode::FAILURE,
                        Z3Verdict::OnlyUndecided { undecided } => {
                            println!(
                                "z3 could not decide {} element(s); exiting nonzero \
                                 (only fully-proved runs exit 0)",
                                undecided
                            );
                            ExitCode::FAILURE
                        }
                    }
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    log.record(&format!("compare (z3): FAILED: {}", e));
                    ExitCode::FAILURE
                }
            }
        }
    }
}

/// Load and parse a module, reporting errors nicely.
fn load_module(file: &Path) -> Result<Module, ExitCode> {
    let mut files = FileCache::new();
    let contents = match files.read(file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to read {}: {}", file.display(), e);
            return Err(ExitCode::FAILURE);
        }
    };
    let ascii_src: &[AsciiChar] = match contents.as_bytes().as_ascii_slice() {
        Some(src) => src,
        None => {
            eprintln!("error: file contains non-ASCII character");
            return Err(ExitCode::FAILURE);
        }
    };
    let mut parser = parse::Parser::new(ascii_src);
    match parser.parse_module().map_err(locate_path(file)) {
        Ok(module) => Ok(module),
        Err(e) => {
            let _ = report_error(
                &mut std::io::stderr(),
                &files,
                Report {
                    path: e.path.as_deref(),
                    span: e.span,
                    title: e.error.title(),
                    message: e.error.message().as_deref(),
                },
            );
            Err(ExitCode::FAILURE)
        }
    }
}

/// Parse a PTX file and report any errors
fn cmd_parse(file: &Path) -> ExitCode {
    let mut files = FileCache::new();

    let contents = match files.read(file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to read {}: {}", file.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let ascii_src: &[AsciiChar] = match contents.as_bytes().as_ascii_slice() {
        Some(src) => src,
        None => {
            eprintln!("error: file contains non-ASCII character");
            return ExitCode::FAILURE;
        }
    };

    let mut parser = parse::Parser::new(ascii_src);
    match parser.parse_module().map_err(locate_path(file)) {
        Ok(module) => {
            println!("Parsed successfully: {}", file.display());
            print_module_summary(&module);
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = report_error(
                &mut std::io::stderr(),
                &files,
                Report {
                    path: e.path.as_deref(),
                    span: e.span,
                    title: e.error.title(),
                    message: e.error.message().as_deref(),
                },
            );
            ExitCode::FAILURE
        }
    }
}

/// Print a summary of the parsed module
fn print_module_summary(module: &Module) {
    let mut entries = 0;
    let mut functions = 0;

    for item in &module.items {
        match item {
            TopLevelItem::Entry(_) => entries += 1,
            TopLevelItem::Function(_) => functions += 1,
            _ => {}
        }
    }

    println!("  Entries (kernels): {}", entries);
    if functions > 0 {
        println!("  Functions: {}", functions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volta_z3::Z3Counts;

    fn counts(equivalent: usize, not_equivalent: usize, unknown: usize) -> Z3Counts {
        Z3Counts {
            equivalent,
            not_equivalent,
            unknown,
            ..Z3Counts::default()
        }
    }

    #[test]
    fn unique_check_arrays_validate() {
        // Membership is deliberately not checked here: the declared-config
        // check owns it before execution, `paired_elements` after.
        assert!(validate_check_arrays(&["out".to_string(), "aux".to_string()]).is_ok());
    }

    #[test]
    fn duplicate_check_arrays_are_rejected() {
        let arrays = vec!["out".to_string(), "aux".to_string(), "out".to_string()];
        let err = validate_check_arrays(&arrays).unwrap_err();
        assert!(err.contains("'out'"), "{}", err);
    }

    #[test]
    fn compare_requires_at_least_one_check_array() {
        // The list is required, never derived - parity with
        // `paired_elements`' explicit-list interface.
        let without = ["volta", "compare", "a.ptx", "b.ptx"];
        assert!(Cli::try_parse_from(without).is_err());
        let with = ["volta", "compare", "a.ptx", "b.ptx", "--check-array", "out"];
        assert!(Cli::try_parse_from(with).is_ok());
    }

    #[test]
    fn z3_verdict_all_equivalent_exits_success() {
        assert_eq!(z3_verdict(&counts(3, 0, 0)), Z3Verdict::AllEquivalent);
        // An empty footprint is vacuously all-equivalent (matches the
        // decision backend), so it too exits 0.
        assert_eq!(z3_verdict(&counts(0, 0, 0)), Z3Verdict::AllEquivalent);
    }

    #[test]
    fn z3_verdict_any_difference_exits_failure() {
        assert_eq!(z3_verdict(&counts(2, 1, 0)), Z3Verdict::HasDifference);
        // A difference dominates even when other elements are undecided.
        assert_eq!(z3_verdict(&counts(0, 1, 5)), Z3Verdict::HasDifference);
    }

    #[test]
    fn z3_verdict_undecided_only_exits_failure() {
        // The regression this guards: an all-unknown run used to exit 0.
        assert_eq!(
            z3_verdict(&counts(0, 0, 3)),
            Z3Verdict::OnlyUndecided { undecided: 3 }
        );
        // Partial proof with leftover undecided is still not a full proof.
        assert_eq!(
            z3_verdict(&counts(4, 0, 2)),
            Z3Verdict::OnlyUndecided { undecided: 2 }
        );
        assert_eq!(
            z3_verdict(&Z3Counts {
                equivalent: 1,
                not_equivalent: 0,
                unknown: 1,
                timeout: 1,
                unsupported: 2,
                error: 3,
            }),
            Z3Verdict::OnlyUndecided { undecided: 7 }
        );
    }
}
