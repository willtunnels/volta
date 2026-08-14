//! Persistent run artifacts: the `--out-dir` layout, benchmark-name
//! sanitization for VC-dump filenames, solve-timing summary statistics,
//! and the per-run results JSON document.
//!
//! Layout under `--out-dir` (default `bench-out/`):
//!
//! - `vcs/<sanitized-benchmark-name>.vcdump` - each equivalence
//!   benchmark's verification conditions (both kernels' arenas + output
//!   footprints), written once per run through
//!   `volta_analysis::driver::vc_dump` - the same format as `volta
//!   compare --dump-vcs`, so `volta compare --from-dump` replays them.
//!   Overwritten on rerun (VCs are deterministic). Race-check benchmarks
//!   have no VC and are skipped with a console note.
//! - `vcs/manifest.json` - `generate`'s record of what each dump
//!   contains; `solve`'s staleness guard (see `crate::manifest`).
//! - `results/<unix-seconds>-<pid>-<command>.json` - the machine-readable
//!   results for every run command, timestamped like
//!   `volta_common::run_log` files so runs never clobber each other. The
//!   `--json <path>` flag writes the same document to an explicit path in
//!   addition. One-shot runs use the full record shape
//!   ([`benchmark_record`]); `generate` runs drop the solve fields
//!   ([`generate_record`]); `solve` runs drop the generation fields and
//!   add `dump_load_secs` ([`solve_record`]), with skipped race checks as
//!   [`skip_record`]s.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::runner::BenchmarkResult;
use crate::z3_phase::{Z3ModeRun, Z3PhaseOutcome};

/// Reduce a display name like "(Attention, FA1)" to a filesystem-safe
/// slug like "attention-fa1": lowercase alphanumerics, every other run of
/// characters collapsed to a single '-', no leading/trailing '-'.
pub fn sanitize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() {
        "benchmark".to_string()
    } else {
        out
    }
}

/// Where a benchmark's VC dump lives under the vcs directory.
pub fn vc_dump_path(vcs_dir: &Path, benchmark_name: &str) -> PathBuf {
    vcs_dir.join(format!("{}.vcdump", sanitize_name(benchmark_name)))
}

/// Reject a benchmark set whose names collide under [`sanitize_name`]:
/// sanitization is many-to-one ("(Red-1, Red-2)" and "Red 1 Red 2" both
/// slug to "red-1-red-2"), and VC dumps are keyed by slug and overwrite
/// silently, so a colliding set would clobber one benchmark's dump with
/// another's. The error names both offenders. Run this wherever a run's
/// benchmark set is known, before anything is written.
pub fn check_slug_collisions<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut by_slug: std::collections::HashMap<String, &str> = std::collections::HashMap::new();
    for name in names {
        match by_slug.entry(sanitize_name(name)) {
            std::collections::hash_map::Entry::Occupied(entry) => anyhow::bail!(
                "benchmark names '{}' and '{}' both sanitize to VC-dump slug '{}'; \
                 their dumps would overwrite each other - rename one of them",
                entry.get(),
                name,
                entry.key(),
            ),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(name);
            }
        }
    }
    Ok(())
}

/// Median of the samples (mean of the two middle values for even counts);
/// `None` for an empty slice.
pub fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

/// Arithmetic mean; `None` for an empty slice.
pub fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().sum::<f64>() / xs.len() as f64)
}

/// Minimum; `None` for an empty slice.
pub fn min(xs: &[f64]) -> Option<f64> {
    xs.iter().copied().min_by(f64::total_cmp)
}

/// Coefficient of variation: sample standard deviation (n-1 denominator)
/// over the mean. `None` for fewer than two samples or a nonpositive
/// mean (all-zero timings have no meaningful relative spread).
pub fn cv(xs: &[f64]) -> Option<f64> {
    if xs.len() < 2 {
        return None;
    }
    let mean = mean(xs)?;
    if mean <= 0.0 {
        return None;
    }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    Some(var.sqrt() / mean)
}

/// The run-wide header of a results document: what was asked for, so a
/// results file is interpretable without the shell history.
pub struct RunMeta {
    pub command: &'static str,
    pub iterations: usize,
    pub sample: u64,
    pub verify_numeric: bool,
    pub recycle_terms: usize,
    /// Decision-solve worker threads (`--parallel`). Above 1 the
    /// benchmarks' `solve_iters_secs` sums include cross-worker
    /// contention (see `solve_wall_iters_secs`), so a results reader
    /// needs this to interpret the solve columns.
    pub parallelism: usize,
    /// `Some(per-query timeout)` exactly when the Z3 phase is enabled.
    pub z3_timeout_secs: Option<u64>,
    /// `Some(backend name)` exactly for `solve` runs, whose VCs come
    /// from dumps rather than fresh generation (the header records both
    /// facts).
    pub solve_backend: Option<&'static str>,
}

/// Insert one timed phase's stats: `<prefix>_iters_secs` (the full
/// per-iteration array) plus `<prefix>_median_secs` / `<prefix>_min_secs`
/// / `<prefix>_mean_secs` / `<prefix>_cv` (nulls when the phase didn't
/// run; the CV is also null for a single iteration). The median is the
/// headline number - iteration 1 includes process/allocator warmup, and
/// the median absorbs it.
fn insert_phase_stats(obj: &mut serde_json::Map<String, Value>, prefix: &str, iters: &[f64]) {
    obj.insert(format!("{prefix}_iters_secs"), json!(iters));
    obj.insert(format!("{prefix}_median_secs"), json!(median(iters)));
    obj.insert(format!("{prefix}_min_secs"), json!(min(iters)));
    obj.insert(format!("{prefix}_mean_secs"), json!(mean(iters)));
    obj.insert(format!("{prefix}_cv"), json!(cv(iters)));
}

fn z3_counts_json(c: &volta_z3::Z3Counts) -> Value {
    json!({
        "equivalent": c.equivalent,
        "not_equivalent": c.not_equivalent,
        "unknown": c.unknown,
        "timeout": c.timeout,
        "unsupported": c.unsupported,
        "error": c.error,
    })
}

/// One Z3 element outcome as its JSON form: the outcome name, plus the
/// detail message for the carrying variants (unsupported/error).
fn z3_element_outcome(outcome: &volta_z3::ElementOutcome) -> (&'static str, Option<&str>) {
    use volta_z3::ElementOutcome::*;
    match outcome {
        Equivalent => ("equivalent", None),
        NotEquivalent => ("not_equivalent", None),
        Unknown => ("unknown", None),
        Timeout => ("timeout", None),
        Unsupported(detail) => ("unsupported", Some(detail)),
        Error(detail) => ("error", Some(detail)),
    }
}

/// One Z3 mode's results (the default encoding or the `+exp-axiom`
/// rerun): per-iteration solve stats, iteration-1 verdict counts, and
/// iteration-1 per-element results.
fn z3_mode_json(run: &Z3ModeRun) -> Value {
    let mut obj = serde_json::Map::new();
    insert_phase_stats(&mut obj, "solve", &run.iters_secs);
    obj.insert("counts".into(), z3_counts_json(&run.counts));
    let elements: Vec<Value> = run
        .elements
        .iter()
        .map(|e| {
            let (outcome, detail) = z3_element_outcome(&e.outcome);
            json!({
                "array": e.array,
                "index": e.index,
                "outcome": outcome,
                "detail": detail,
                "solve_secs": e.solve.as_secs_f64(),
            })
        })
        .collect();
    obj.insert("elements".into(), Value::Array(elements));
    Value::Object(obj)
}

/// A benchmark's `z3` section: `null` when the phase didn't apply (no
/// `--z3`, or a benchmark with no solve phase), `{"error": ...}` when it
/// failed, and the full results object when it ran (`error` null there,
/// so consumers can always check the one field).
fn z3_section(z3: Option<&Z3PhaseOutcome>) -> Value {
    match z3 {
        None => Value::Null,
        Some(Z3PhaseOutcome::Failed(message)) => json!({ "error": message }),
        Some(Z3PhaseOutcome::Ran(phase)) => {
            let mut obj = serde_json::Map::new();
            obj.insert("error".into(), Value::Null);
            let Value::Object(plain) = z3_mode_json(&phase.plain) else {
                unreachable!("z3_mode_json builds an object");
            };
            obj.extend(plain);
            obj.insert(
                "axiom".into(),
                match &phase.axiom {
                    Some(rerun) => z3_mode_json(rerun),
                    None => Value::Null,
                },
            );
            Value::Object(obj)
        }
    }
}

/// The identity fields every record shape starts with.
fn identity_fields(r: &BenchmarkResult) -> serde_json::Map<String, Value> {
    let mut obj = serde_json::Map::new();
    obj.insert("name".into(), json!(r.name));
    obj.insert("category".into(), json!(r.category.name()));
    obj.insert("status".into(), json!(r.outcome.status()));
    obj.insert(
        "detail".into(),
        json!(crate::reporter::describe(&r.outcome)),
    );
    obj.insert("passed".into(), json!(r.passed));
    obj
}

/// The generation-phase fields (the one-shot and `generate` records).
fn insert_gen_fields(obj: &mut serde_json::Map<String, Value>, r: &BenchmarkResult) {
    insert_phase_stats(obj, "vc_gen", &r.stats.vc_gen_iters_secs);
    obj.insert("dump_write_secs".into(), json!(r.stats.dump_write_secs));
    obj.insert("instructions".into(), json!(r.stats.instructions));
    obj.insert("block_syncs".into(), json!(r.stats.block_syncs));
    obj.insert("warp_syncs".into(), json!(r.stats.warp_syncs));
}

/// The solve-phase fields (the one-shot and `solve` records).
fn insert_solve_fields(obj: &mut serde_json::Map<String, Value>, r: &BenchmarkResult) {
    obj.insert("elements_checked".into(), json!(r.stats.elements_checked));
    insert_phase_stats(obj, "solve", &r.stats.solve_iters_secs);
    // Wall clock of the same iterations: tracks the summed stats above at
    // --parallel 1, diverges under parallelism (the header records it).
    insert_phase_stats(obj, "solve_wall", &r.stats.solve_wall_iters_secs);
    // The `--verify-numeric` oracle's time (iteration 1 only); null when
    // the flag was off. Kept out of the solve stats so those don't move
    // when the oracle is toggled.
    obj.insert(
        "verify_numeric_secs".into(),
        json!(r.stats.verify_numeric_secs),
    );
    // Iteration 1's per-element decision-procedure check times (empty
    // when no solve ran); positionally aligned with the z3 section's
    // `elements`.
    let decision_elements: Vec<Value> = r
        .stats
        .decision_elements
        .iter()
        .map(|e| {
            json!({
                "array": e.array,
                "index": e.index,
                "solve_secs": e.check.as_secs_f64(),
            })
        })
        .collect();
    obj.insert("decision_elements".into(), Value::Array(decision_elements));
    obj.insert("z3".into(), z3_section(r.z3.as_ref()));
}

/// One benchmark as a results-JSON record - the single record shape for
/// every one-shot run command (`all`/`category`/`single`), with the Z3
/// phase as a nested `z3` section (null without `--z3`).
pub fn benchmark_record(r: &BenchmarkResult) -> Value {
    let mut obj = identity_fields(r);
    obj.insert("elements_total".into(), json!(r.stats.elements_total));
    insert_gen_fields(&mut obj, r);
    insert_solve_fields(&mut obj, r);
    obj.insert("dump_path".into(), json!(r.dump_path));
    Value::Object(obj)
}

/// A `generate` run's record: the one-shot shape minus every solve field
/// (nothing was solved). `elements_total` is the generated footprint
/// size; race-check benchmarks carry their real verdicts here - this is
/// the record the race table comes from.
pub fn generate_record(r: &BenchmarkResult) -> Value {
    let mut obj = identity_fields(r);
    obj.insert("elements_total".into(), json!(r.stats.elements_total));
    insert_gen_fields(&mut obj, r);
    obj.insert("dump_path".into(), json!(r.dump_path));
    Value::Object(obj)
}

/// A `solve` run's record: the one-shot shape minus every generation
/// field (VCs came from a dump; there are no gen timings or execution
/// counters to report), plus `dump_load_secs` (the load is excluded from
/// the solve spans). `dump_path` is the dump the VCs were read from.
pub fn solve_record(r: &BenchmarkResult) -> Value {
    let mut obj = identity_fields(r);
    obj.insert("elements_total".into(), json!(r.stats.elements_total));
    obj.insert("dump_load_secs".into(), json!(r.stats.dump_load_secs));
    insert_solve_fields(&mut obj, r);
    obj.insert("dump_path".into(), json!(r.dump_path));
    Value::Object(obj)
}

/// A `solve` run's record for a benchmark it skipped (race checks:
/// nothing to solve).
pub fn skip_record(name: &str, category: crate::config::BenchmarkCategory, note: &str) -> Value {
    json!({
        "name": name,
        "category": category.name(),
        "status": "SKIP",
        "detail": note,
        "passed": true,
    })
}

/// Assemble the full results document: run header + per-benchmark records.
pub fn results_doc(meta: &RunMeta, benchmarks: Vec<Value>) -> Value {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut doc = serde_json::Map::new();
    doc.insert("command".into(), json!(meta.command));
    doc.insert(
        "argv".into(),
        json!(std::env::args().collect::<Vec<String>>()),
    );
    doc.insert("timestamp_unix".into(), json!(timestamp));
    doc.insert("iterations".into(), json!(meta.iterations));
    doc.insert("sample".into(), json!(meta.sample));
    doc.insert("verify_numeric".into(), json!(meta.verify_numeric));
    doc.insert("recycle_terms".into(), json!(meta.recycle_terms));
    doc.insert("parallelism".into(), json!(meta.parallelism));
    if let Some(backend) = meta.solve_backend {
        // A `solve` run: which backend(s) solved, and that the VCs were
        // loaded from `generate`'s dumps rather than freshly generated.
        doc.insert("backend".into(), json!(backend));
        doc.insert("vcs_from_dumps".into(), json!(true));
    }
    doc.insert("z3".into(), json!(meta.z3_timeout_secs.is_some()));
    if let Some(timeout) = meta.z3_timeout_secs {
        // Record the linked libz3's version: solver behavior (notably how
        // the exp-axiom quantifier loop exits - budget kill vs its own
        // resource-bound "unknown") is version-dependent, so a results
        // file must say which solver produced it.
        doc.insert("z3_version".into(), json!(volta_z3::z3_version()));
        doc.insert("z3_timeout_secs".into(), json!(timeout));
        // The carve-out convention (see `z3_phase`): re-solving a
        // timeout would multiply its full budget into every iteration,
        // and unsupported/error elements never reach the solver.
        doc.insert(
            "z3_iteration_convention".into(),
            json!(
                "elements whose iteration-1 z3 outcome is timeout/unsupported/error \
                 are solved once; their iteration-1 solve time is charged to every \
                 iteration's total"
            ),
        );
    }
    doc.insert("benchmarks".into(), Value::Array(benchmarks));
    Value::Object(doc)
}

/// Write `doc` to `<out_dir>/results/<unix-seconds>-<pid>-<command>.json`
/// (the `run_log` naming convention, so concurrent or scripted runs never
/// clobber each other) and return the path.
pub fn write_results_file(out_dir: &Path, command: &str, doc: &Value) -> Result<PathBuf> {
    let dir = out_dir.join("results");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating results directory {}", dir.display()))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("{stamp}-{}-{command}.json", std::process::id()));
    write_results_to(&path, doc)?;
    Ok(path)
}

/// Write `doc` to an explicit path (the `--json` flag).
pub fn write_results_to(path: &Path, doc: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(doc)?)
        .with_context(|| format!("writing results to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_name_makes_filesystem_safe_slugs() {
        assert_eq!(sanitize_name("(Attention, FA1)"), "attention-fa1");
        assert_eq!(sanitize_name("(Red-1, Red-2)"), "red-1-red-2");
        assert_eq!(sanitize_name("plain"), "plain");
        assert_eq!(sanitize_name("  spaced   out  "), "spaced-out");
        assert_eq!(sanitize_name("A/B\\C:D"), "a-b-c-d");
        // Nothing usable left: still a nonempty filename.
        assert_eq!(sanitize_name("()"), "benchmark");
        assert_eq!(sanitize_name(""), "benchmark");
    }

    #[test]
    fn solve_stats_handle_odd_even_and_empty() {
        assert_eq!(median(&[]), None);
        assert_eq!(mean(&[]), None);
        assert_eq!(min(&[]), None);

        // Odd count: the middle element; order must not matter.
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        // Even count: mean of the two middle elements.
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[5.0]), Some(5.0));

        assert_eq!(mean(&[1.0, 2.0, 6.0]), Some(3.0));
        assert_eq!(min(&[3.0, 1.0, 2.0]), Some(1.0));
    }

    /// CV = sample standard deviation / mean; undefined (None) for fewer
    /// than two samples or an all-zero phase - a JSON null, never a NaN.
    #[test]
    fn cv_is_sample_sd_over_mean() {
        assert_eq!(cv(&[]), None);
        assert_eq!(cv(&[1.0]), None);
        assert_eq!(cv(&[0.0, 0.0]), None);

        // Identical samples: zero spread.
        assert_eq!(cv(&[2.0, 2.0, 2.0]), Some(0.0));
        // {1, 3}: mean 2, sample variance ((1)^2 + (1)^2) / 1 = 2.
        let got = cv(&[1.0, 3.0]).unwrap();
        assert!((got - 2.0f64.sqrt() / 2.0).abs() < 1e-12, "{got}");
    }

    /// The vcdump path is deterministic per benchmark name, so reruns
    /// overwrite instead of accumulating.
    #[test]
    fn vc_dump_path_is_stable() {
        let dir = Path::new("bench-out/vcs");
        assert_eq!(
            vc_dump_path(dir, "(Attention, FA1)"),
            Path::new("bench-out/vcs/attention-fa1.vcdump")
        );
        assert_eq!(
            vc_dump_path(dir, "(Attention, FA1)"),
            vc_dump_path(dir, "(Attention, FA1)"),
        );
    }

    /// `sanitize_name` is many-to-one; a set containing two names with
    /// the same slug must be rejected up front, naming both offenders.
    #[test]
    fn slug_collisions_fail_with_both_names() {
        check_slug_collisions(["(Red-1, Red-2)", "(Red-1, Red-3)"]).unwrap();
        check_slug_collisions(std::iter::empty::<&str>()).unwrap();

        let err = check_slug_collisions(["(Red-1, Red-2)", "Red 1 Red: 2"]).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("(Red-1, Red-2)"), "{message}");
        assert!(message.contains("Red 1 Red: 2"), "{message}");
        assert!(message.contains("'red-1-red-2'"), "{message}");
    }

    /// Guardrail over the real corpus: every name in `all_benchmarks()`
    /// must slug uniquely, so a future benchmark whose name collides with
    /// an existing one fails here (in CI) instead of silently overwriting
    /// that benchmark's VC dump.
    #[test]
    fn full_corpus_has_no_slug_collisions() {
        let suite = crate::all_benchmarks();
        check_slug_collisions(suite.benchmarks.iter().map(|b| b.name.as_str())).unwrap();
    }

    fn result_with(outcome: crate::runner::ActualOutcome, passed: bool) -> BenchmarkResult {
        BenchmarkResult {
            name: "x".to_string(),
            category: crate::config::BenchmarkCategory::Reduction,
            elapsed_secs: 0.0,
            outcome,
            stats: Default::default(),
            outcome_matched: passed,
            passed,
            dump_path: None,
            z3: None,
        }
    }

    /// A benchmark that fails *after* its VC dump is written still has
    /// the dump on disk: the record builder must keep a `Some(dump_path)`
    /// alongside an error outcome rather than nulling it.
    #[test]
    fn error_records_keep_the_dump_path() {
        let dump = PathBuf::from("bench-out/vcs/x.vcdump");
        let mut result = result_with(
            crate::runner::ActualOutcome::Error {
                message: "post-dump failure".to_string(),
            },
            false,
        );
        result.dump_path = Some(dump.clone());
        let record = benchmark_record(&result);
        assert_eq!(record["status"], json!("ERR"));
        assert_eq!(record["dump_path"], json!(dump));
    }

    /// The three record shapes partition the fields: `generate` records
    /// carry the gen fields and no solve fields, `solve` records carry
    /// the solve fields plus `dump_load_secs` and no gen fields, and the
    /// one-shot record carries both (and no `dump_load_secs`).
    #[test]
    fn generate_and_solve_records_split_the_one_shot_shape() {
        let mut result = result_with(crate::runner::ActualOutcome::VcsGenerated, true);
        result.stats.vc_gen_iters_secs = vec![1.0, 2.0];
        result.stats.dump_write_secs = Some(0.5);
        result.stats.elements_total = 4;

        let generated = generate_record(&result);
        assert_eq!(generated["status"], json!("GEN"));
        assert_eq!(generated["vc_gen_iters_secs"], json!([1.0, 2.0]));
        assert_eq!(generated["dump_write_secs"], json!(0.5));
        assert_eq!(generated["elements_total"], json!(4));
        for solve_field in [
            "solve_iters_secs",
            "z3",
            "elements_checked",
            "dump_load_secs",
        ] {
            assert!(generated.get(solve_field).is_none(), "{solve_field}");
        }

        let mut result = result_with(crate::runner::ActualOutcome::Equivalent, true);
        result.stats.solve_iters_secs = vec![3.0];
        result.stats.dump_load_secs = Some(0.25);
        result.stats.elements_checked = 4;
        result.stats.elements_total = 4;

        let solve = solve_record(&result);
        assert_eq!(solve["status"], json!("EQUIV"));
        assert_eq!(solve["solve_iters_secs"], json!([3.0]));
        assert_eq!(solve["dump_load_secs"], json!(0.25));
        assert_eq!(solve["elements_checked"], json!(4));
        assert!(solve["z3"].is_null());
        for gen_field in ["vc_gen_iters_secs", "dump_write_secs", "instructions"] {
            assert!(solve.get(gen_field).is_none(), "{gen_field}");
        }

        // The one-shot record keeps its historical shape: both field
        // sets, no dump_load_secs.
        let one_shot = benchmark_record(&result);
        assert!(one_shot.get("vc_gen_iters_secs").is_some());
        assert!(one_shot.get("solve_iters_secs").is_some());
        assert!(one_shot.get("dump_load_secs").is_none());

        let skip = skip_record(
            "Red-5 (racy)",
            crate::config::BenchmarkCategory::Reduction,
            "race-check benchmark",
        );
        assert_eq!(skip["status"], json!("SKIP"));
        assert_eq!(skip["passed"], json!(true));
    }

    /// A solve run's header names the backend and that the VCs came from
    /// dumps; other runs' headers carry neither key.
    #[test]
    fn results_doc_header_records_solve_backend() {
        let mut meta = RunMeta {
            command: "solve",
            iterations: 10,
            sample: 1,
            verify_numeric: false,
            recycle_terms: 0,
            parallelism: 1,
            z3_timeout_secs: None,
            solve_backend: Some("decision"),
        };
        let doc = results_doc(&meta, vec![]);
        assert_eq!(doc["backend"], json!("decision"));
        assert_eq!(doc["vcs_from_dumps"], json!(true));
        assert_eq!(doc["z3"], json!(false));

        meta.solve_backend = None;
        let doc = results_doc(&meta, vec![]);
        assert!(doc.get("backend").is_none());
        assert!(doc.get("vcs_from_dumps").is_none());
    }

    /// The oracle-time bucket is `Some` exactly when `--verify-numeric`
    /// ran (JSON: a number vs null), so its absence is distinguishable
    /// from a fast oracle.
    #[test]
    fn verify_numeric_secs_round_trips() {
        let mut result = result_with(crate::runner::ActualOutcome::Equivalent, true);
        assert!(benchmark_record(&result)["verify_numeric_secs"].is_null());

        result.stats.verify_numeric_secs = Some(0.25);
        assert_eq!(
            benchmark_record(&result)["verify_numeric_secs"],
            json!(0.25)
        );
    }

    /// The one record shape: phase stats carry the full iteration arrays
    /// plus median/min/mean/cv, per-element decision times serialize as
    /// `{array, index, solve_secs}`, and the `z3` section is null without
    /// `--z3`, `{"error": ...}` on a failed phase, and the full
    /// plain+axiom results when it ran.
    #[test]
    fn benchmark_record_carries_phase_stats_and_z3_section() {
        use crate::z3_phase::{Z3ModeRun, Z3Phase, Z3PhaseOutcome};
        use std::time::Duration;

        let mut result = result_with(crate::runner::ActualOutcome::Equivalent, true);
        result.stats.vc_gen_iters_secs = vec![2.0, 1.0, 1.0];
        result.stats.solve_iters_secs = vec![4.0, 2.0];
        result.stats.decision_elements = vec![volta_analysis::driver::ElementCheckTime {
            array: "out".to_string(),
            index: 7,
            check: Duration::from_millis(250),
        }];

        let record = benchmark_record(&result);
        assert_eq!(record["vc_gen_iters_secs"], json!([2.0, 1.0, 1.0]));
        assert_eq!(record["vc_gen_median_secs"], json!(1.0));
        assert_eq!(record["vc_gen_min_secs"], json!(1.0));
        assert!(record["vc_gen_cv"].is_number());
        assert_eq!(record["solve_iters_secs"], json!([4.0, 2.0]));
        assert_eq!(record["solve_median_secs"], json!(3.0));
        assert_eq!(
            record["decision_elements"],
            json!([{"array": "out", "index": 7, "solve_secs": 0.25}])
        );
        // No --z3: the section is null, not an empty object.
        assert!(record["z3"].is_null());

        // A failed phase records the error (and fails the benchmark
        // elsewhere; the record just carries the message).
        result.z3 = Some(Z3PhaseOutcome::Failed("boom".to_string()));
        let record = benchmark_record(&result);
        assert_eq!(record["z3"], json!({"error": "boom"}));

        // A completed phase: per-iteration stats, counts, iteration-1
        // elements, and the axiom sub-run in the same shape.
        let mode = |secs: Vec<f64>, outcome: volta_z3::ElementOutcome| Z3ModeRun {
            iters_secs: secs,
            counts: volta_z3::Z3EquivReport {
                elements: vec![volta_z3::ElementResult {
                    array: "out".to_string(),
                    index: 7,
                    outcome: outcome.clone(),
                    solve: Duration::from_millis(500),
                }],
            }
            .counts(),
            elements: vec![volta_z3::ElementResult {
                array: "out".to_string(),
                index: 7,
                outcome,
                solve: Duration::from_millis(500),
            }],
        };
        result.z3 = Some(Z3PhaseOutcome::Ran(Z3Phase {
            plain: mode(vec![0.5, 0.25], volta_z3::ElementOutcome::Unknown),
            axiom: Some(mode(vec![30.0, 30.0], volta_z3::ElementOutcome::Timeout)),
        }));
        let record = benchmark_record(&result);
        let z3 = &record["z3"];
        assert!(z3["error"].is_null());
        assert_eq!(z3["solve_iters_secs"], json!([0.5, 0.25]));
        assert_eq!(z3["counts"]["unknown"], json!(1));
        assert_eq!(
            z3["elements"],
            json!([{"array": "out", "index": 7, "outcome": "unknown", "detail": null,
                    "solve_secs": 0.5}])
        );
        assert_eq!(z3["axiom"]["solve_iters_secs"], json!([30.0, 30.0]));
        assert_eq!(z3["axiom"]["counts"]["timeout"], json!(1));
        assert_eq!(z3["axiom"]["elements"][0]["outcome"], json!("timeout"));
    }
}
