//! `ferrox bench --suite` / `--render`: the llama-bench-shaped ledger.
//!
//! Numbers are `pp<N>` / `tg<N>` from `ferrox bench` vs `llama-bench` —
//! no HTTP, template, tokenizer, or sampler. That is what a kernel
//! change moves, and what [`benchmarks/RESULTS.md`](../../../benchmarks/RESULTS.md)
//! quotes.
//!
//! Each suite entry runs in a **fresh child process**. Backend selection
//! reads process-global environment and the rayon pool is built once, so
//! benchmarking several backends inside one process would silently
//! measure the first one's configuration for all of them.

use anyhow::Context;
use std::path::{Path, PathBuf};

/// One `models[]` entry of `benchmarks/suite.json`.
struct SuiteEntry {
    id: String,
    name: String,
    gguf: String,
    backends: Vec<String>,
    estimated_ram_gb: f64,
}

pub struct SuiteArgs {
    pub bench_dir: PathBuf,
    pub n_prompt: usize,
    pub n_gen: usize,
    pub reps: usize,
    pub only_id: Option<String>,
    pub only_backend: Option<String>,
    pub fit_host: bool,
    pub skip_missing: bool,
    /// Forwarded to every child `bench` run: the 1-minute load average
    /// above which a timed run refuses to start.
    pub max_load: f64,
}

fn suite_path(bench_dir: &Path) -> PathBuf {
    bench_dir.join("suite.json")
}

fn engine_receipt_dir(bench_dir: &Path) -> PathBuf {
    bench_dir.join("receipts").join("engine")
}

fn load_suite(bench_dir: &Path) -> anyhow::Result<Vec<SuiteEntry>> {
    let path = suite_path(bench_dir);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading suite at {}", path.display()))?;
    let root: serde_json::Value = serde_json::from_str(&text)?;
    let models = root
        .get("models")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("suite.json has no `models` array"))?;
    Ok(models
        .iter()
        .filter_map(|m| {
            Some(SuiteEntry {
                id: m.get("id")?.as_str()?.to_string(),
                name: m.get("name")?.as_str()?.to_string(),
                gguf: m.get("gguf")?.as_str()?.to_string(),
                backends: m
                    .get("backends")?
                    .as_array()?
                    .iter()
                    .filter_map(|b| Some(b.as_str()?.to_string()))
                    .collect(),
                estimated_ram_gb: m
                    .get("estimated_ram_gb")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
            })
        })
        .collect())
}

/// Physical RAM in GiB, for `--fit-host`.
fn host_ram_gb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn sysctlbyname(
                name: *const std::os::raw::c_char,
                oldp: *mut std::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut std::ffi::c_void,
                newlen: usize,
            ) -> std::os::raw::c_int;
        }
        let key = std::ffi::CString::new("hw.memsize").unwrap();
        let mut out: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: `hw.memsize` returns a u64 and `out`/`len` describe one.
        let rc = unsafe {
            sysctlbyname(
                key.as_ptr(),
                &mut out as *mut u64 as *mut std::ffi::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && out > 0 {
            return out as f64 / (1024.0 * 1024.0 * 1024.0);
        }
    }
    0.0
}

pub fn run_suite(args: SuiteArgs) -> anyhow::Result<()> {
    // The suite is the unit of truth for RESULTS.md, so check the host
    // once up front rather than discovering at model 9 of 13 that the
    // first eight rows were measured on a busy box. Children re-check
    // individually, because load can rise mid-suite, and the loop below
    // waits for the previous entry's own load to decay before starting
    // the next one so the suite does not lock itself out.
    crate::host_state::ensure_quiet_enough(args.max_load)?;
    let entries = load_suite(&args.bench_dir)?;
    let exe = std::env::current_exe()?;
    let ram = host_ram_gb();
    let out_dir = engine_receipt_dir(&args.bench_dir);
    std::fs::create_dir_all(&out_dir)?;

    for entry in &entries {
        if let Some(only) = &args.only_id {
            if &entry.id != only {
                continue;
            }
        }
        for backend in &entry.backends {
            if let Some(only) = &args.only_backend {
                if backend != only {
                    continue;
                }
            }
            if backend == "cuda" && cfg!(target_os = "macos") {
                eprintln!("skip {} {backend}: no CUDA on this host", entry.id);
                continue;
            }
            if backend == "metal" && !cfg!(feature = "metal") {
                eprintln!(
                    "skip {} metal: this binary was built without --features metal",
                    entry.id
                );
                continue;
            }
            // 75% of physical RAM headroom for OS + weights + KV.
            if args.fit_host && ram > 0.0 && entry.estimated_ram_gb > 0.75 * ram {
                eprintln!(
                    "skip {} {backend}: needs ~{:.0} GiB, host has {ram:.0} GiB",
                    entry.id, entry.estimated_ram_gb
                );
                continue;
            }
            // Total RAM says the model COULD fit this machine. Free RAM
            // says whether it fits right now. A 32 GiB box with 3.5 GiB
            // free accepts a 10 GiB model on the check above, then runs
            // it out of swap and reports a real-looking number for work
            // the disk did. Skipping keeps the previous receipt, which
            // is stale and says so, rather than replacing it with a
            // paged one that does not.
            if args.fit_host && args.max_load > 0.0 {
                if let Some(free) = crate::host_state::free_ram_gb() {
                    if entry.estimated_ram_gb + 2.0 > free {
                        eprintln!(
                            "skip {} {backend}: needs ~{:.0} GiB, only {free:.1} GiB free \
                             (it would run from swap)",
                            entry.id, entry.estimated_ram_gb
                        );
                        continue;
                    }
                }
            }
            let model_path = args.bench_dir.join("..").join(&entry.gguf);
            if !model_path.exists() {
                if args.skip_missing {
                    eprintln!("skip {} {backend}: {} not present", entry.id, entry.gguf);
                    continue;
                }
                anyhow::bail!("missing GGUF for {}: {}", entry.id, entry.gguf);
            }

            let receipt = out_dir.join(format!("{}_{backend}.json", entry.id));
            eprintln!("\n=== {} [{}] {backend} ===", entry.id, entry.name);
            // The previous entry's own benchmark is still in the
            // 1-minute average, and the child re-checks the bar. Let it
            // decay instead of letting the suite lock itself out.
            // Skip this entry rather than abandoning the suite. `?` here
            // meant one busy stretch killed the whole run and every
            // model after it never went, which is how a 12-model suite
            // stopped at 8 and left the table half old and half new.
            // A missing GGUF already skips; an unclearable host is the
            // same kind of "not now", and the previous receipt stands.
            if let Err(why) = crate::host_state::wait_until_quiet_enough(
                args.max_load,
                std::time::Duration::from_secs(180),
            ) {
                eprintln!("skip {} {backend}: {why}", entry.id);
                continue;
            }
            let status = std::process::Command::new(&exe)
                .arg("bench")
                .args(["-m", &entry.gguf])
                .args(["-p", &args.n_prompt.to_string()])
                .args(["-n", &args.n_gen.to_string()])
                .args(["-r", &args.reps.to_string()])
                .args(["--n-gpu-layers", if backend == "cpu" { "0" } else { "99" }])
                .arg("--compare")
                .args(["--suite-id", &entry.id])
                .args(["--backend-label", backend])
                .args(["--receipt", receipt.to_str().unwrap()])
                .args(["--max-load", &args.max_load.to_string()])
                .status()?;
            if !status.success() {
                eprintln!(
                    "!! {} {backend} failed ({status}); leaving previous receipt alone",
                    entry.id
                );
            }
        }
    }
    render(&args.bench_dir)
}

/// Rewrites the engine table in `RESULTS.md` from the receipts on disk,
/// leaving the header and Open notes outside the markers untouched.
const BEGIN: &str = "<!-- BEGIN ENGINE TABLE (generated by `ferrox bench --render`) -->";
const END: &str = "<!-- END ENGINE TABLE -->";

pub fn render(bench_dir: &Path) -> anyhow::Result<()> {
    let dir = engine_receipt_dir(bench_dir);
    let mut receipts: Vec<serde_json::Value> = Vec::new();
    if dir.is_dir() {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        paths.sort();
        for p in paths {
            if let Ok(text) = std::fs::read_to_string(&p) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                    receipts.push(v);
                }
            }
        }
    }

    let suite = load_suite(bench_dir).unwrap_or_default();
    let name_of = |id: &str| {
        suite
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.name.clone())
            .unwrap_or_else(|| id.to_string())
    };

    #[derive(Clone)]
    struct Row {
        model: String,
        backend: String,
        test: String,
        ferrox: Option<f64>,
        llama: Option<f64>,
        gap: Option<f64>,
    }

    let mut rows: Vec<Row> = Vec::new();
    for r in &receipts {
        let id = r.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let backend = r
            .get("backend")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let Some(tests) = r.get("tests").and_then(|v| v.as_array()) else {
            continue;
        };
        for t in tests {
            rows.push(Row {
                model: name_of(id),
                backend: backend.clone(),
                test: t
                    .get("test")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string(),
                ferrox: t.get("ferrox_tps").and_then(|v| v.as_f64()),
                llama: t.get("llama_tps").and_then(|v| v.as_f64()),
                gap: t.get("gap").and_then(|v| v.as_f64()),
            });
        }
    }

    // Worst-first within (backend, test): high gap = ferrox farther behind.
    rows.sort_by(|a, b| {
        let backend_ord = |s: &str| match s {
            "metal" => 0,
            "cuda" => 1,
            "cpu" => 2,
            _ => 3,
        };
        let test_ord = |s: &str| {
            if s.starts_with("pp") {
                0
            } else if s.starts_with("tg") {
                1
            } else {
                2
            }
        };
        backend_ord(&a.backend)
            .cmp(&backend_ord(&b.backend))
            .then_with(|| test_ord(&a.test).cmp(&test_ord(&b.test)))
            .then_with(|| {
                b.gap
                    .partial_cmp(&a.gap)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.model.cmp(&b.model))
    });

    let mut table = String::new();
    table.push_str(BEGIN);
    table.push_str("\n\n## Engine (`ferrox bench` vs `llama-bench`)\n\n");
    table.push_str(
        "No HTTP, no chat template, no tokenizer, no sampler — this is the engine\n\
         alone. `pp512` is batched prefill, `tg128` is decode. **Neither engine's\n\
         thread count is forced**: each picks its own default, because llama.cpp\n\
         defaults to performance cores and loses 2–4× when pushed above them, so\n\
         pinning both to the same count does not make the comparison fairer.\n\n\
         **Gap** = `llama / ferrox` (<1 ferrox faster). Rows are grouped by\n\
         backend (Metal → CUDA → CPU), then test (`pp` then `tg`), then **worst\n\
         gap first**. Regenerate with `ferrox bench --suite` / `--render`.\n\n",
    );

    // Compact "at a glance" for the largest prefill losses.
    let mut worst_pp: Vec<&Row> = rows
        .iter()
        .filter(|r| r.test.starts_with("pp") && r.gap.is_some_and(|g| g > 1.05))
        .collect();
    worst_pp.sort_by(|a, b| {
        b.gap
            .partial_cmp(&a.gap)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if !worst_pp.is_empty() {
        table.push_str("**Largest engine prefill gaps (pp\\*, gap > 1.05×):**\n\n");
        for r in worst_pp.iter().take(8) {
            table.push_str(&format!(
                "- `{}` / {} / {}: {}\n",
                r.model,
                r.backend,
                r.test,
                r.gap.map(gap_cell).unwrap_or_else(|| "—".into()),
            ));
        }
        table.push('\n');
    }

    fn push_section(table: &mut String, title: &str, rows: &[&Row]) {
        if rows.is_empty() {
            return;
        }
        table.push_str(&format!("### {title}\n\n"));
        table.push_str("| Model | Test | ferrox tok/s | llama.cpp tok/s | Gap |\n");
        table.push_str("|---|---|---|---|---|\n");
        for r in rows {
            table.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                r.model,
                r.test,
                r.ferrox
                    .map(|v| format!("**{v:.2}**"))
                    .unwrap_or_else(|| "—".into()),
                r.llama
                    .map(|v| format!("**{v:.2}**"))
                    .unwrap_or_else(|| "—".into()),
                r.gap.map(gap_cell).unwrap_or_else(|| "—".into()),
            ));
        }
        table.push('\n');
    }

    let metal: Vec<&Row> = rows.iter().filter(|r| r.backend == "metal").collect();
    let cuda: Vec<&Row> = rows.iter().filter(|r| r.backend == "cuda").collect();
    let cpu: Vec<&Row> = rows.iter().filter(|r| r.backend == "cpu").collect();
    let other: Vec<&Row> = rows
        .iter()
        .filter(|r| !matches!(r.backend.as_str(), "metal" | "cuda" | "cpu"))
        .collect();

    if rows.is_empty() {
        table.push_str("| _no engine receipts yet_ | | | | | |\n\n");
    } else {
        push_section(&mut table, "Metal", &metal);
        push_section(&mut table, "CUDA", &cuda);
        push_section(&mut table, "CPU", &cpu);
        push_section(&mut table, "Other backends", &other);
    }

    table.push_str(END);

    let results = bench_dir.join("RESULTS.md");
    let existing = std::fs::read_to_string(&results).unwrap_or_default();
    let updated = splice(&existing, &table);
    std::fs::write(&results, updated)?;
    eprintln!(
        "ferrox bench: engine table written to {}",
        results.display()
    );
    Ok(())
}

fn gap_cell(g: f64) -> String {
    let marker = if g < 0.95 {
        "🟢"
    } else if g <= 1.05 {
        "⚪"
    } else {
        "🔴"
    };
    format!("{marker} **{g:.2}×**")
}

/// Replaces the marked block, or appends it if the markers are absent.
fn splice(existing: &str, block: &str) -> String {
    if let (Some(start), Some(end)) = (existing.find(BEGIN), existing.find(END)) {
        let mut out = String::with_capacity(existing.len() + block.len());
        out.push_str(&existing[..start]);
        out.push_str(block);
        out.push_str(&existing[end + END.len()..]);
        return out;
    }
    let mut out = existing.to_string();
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push('\n');
    out.push_str(block);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_cell_colours_match_the_ledger_convention() {
        assert!(gap_cell(0.80).starts_with("🟢"));
        assert!(gap_cell(1.00).starts_with("⚪"));
        assert!(gap_cell(0.96).starts_with("⚪"));
        assert!(gap_cell(1.40).starts_with("🔴"));
    }

    #[test]
    fn splice_replaces_an_existing_block_and_keeps_the_surrounding_text() {
        let doc = format!("before\n{BEGIN}\nold\n{END}\nafter\n");
        let out = splice(&doc, &format!("{BEGIN}\nnew\n{END}"));
        assert!(out.contains("before"), "text before the block must survive");
        assert!(out.contains("after"), "text after the block must survive");
        assert!(out.contains("new"));
        assert!(!out.contains("old"), "the old block must be gone");
    }

    #[test]
    fn splice_appends_when_the_markers_are_missing() {
        let out = splice("just some prose\n", &format!("{BEGIN}\nfresh\n{END}"));
        assert!(out.starts_with("just some prose"));
        assert!(out.contains("fresh"));
    }

    #[test]
    fn splice_does_not_duplicate_the_block_on_a_second_render() {
        let block = format!("{BEGIN}\nv1\n{END}");
        let once = splice("doc\n", &block);
        let twice = splice(&once, &format!("{BEGIN}\nv2\n{END}"));
        assert_eq!(twice.matches(BEGIN).count(), 1, "exactly one engine block");
        assert!(twice.contains("v2") && !twice.contains("v1"));
    }
}
