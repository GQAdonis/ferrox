//! `ferrox bench -m model.gguf` — a `llama-bench` work-alike.
//!
//! Why this exists: the fair-chat suite measures two engines *through
//! their HTTP servers*, with chat templating, tokenization, sampling and
//! SSE in the loop. That is a legitimate number — it is what a user of
//! `ferrox-server` actually gets — but it is not an engine number, and
//! mixing the two makes a regression impossible to localize. `llama.cpp`
//! draws the same line, which is why `llama-bench` exists next to
//! `llama-server`.
//!
//! So: same workload definition as `llama-bench` (`pp<N>` = batched
//! prefill of N synthetic tokens, `tg<N>` = N single-token decode steps
//! after a one-token prime), same reporting (median ± population
//! stddev over `-r` repetitions, after one discarded warmup), same flag
//! names. Output is directly comparable to
//! `llama-bench -m <same gguf> -p <N> -n <N> -t <T> -ngl <L>`.
//!
//! Deliberately *not* here: chat template, real tokenizer, sampling.
//! Synthetic token ids exercise the same weights and the same KV growth
//! without making the number depend on a tokenizer's behavior.

use anyhow::Context;
use ferrox_core::cache::KvCache;
use ferrox_models::{Decoder, ModelConfig};
use std::path::Path;
use std::time::Instant;

/// One `llama-bench` row: a named workload and its per-rep tok/s.
struct Row {
    test: String,
    samples: Vec<f64>,
}

impl Row {
    fn median(&self) -> f64 {
        let mut s = self.samples.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        if n == 0 {
            return 0.0;
        }
        if n % 2 == 1 {
            s[n / 2]
        } else {
            0.5 * (s[n / 2 - 1] + s[n / 2])
        }
    }

    /// Population stddev, matching what `llama-bench` prints after `±`.
    fn stddev(&self) -> f64 {
        let n = self.samples.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.samples.iter().sum::<f64>() / n as f64;
        (self.samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
    }
}

/// Applies the same backend/threading env `ferrox run` does, before any
/// worker thread exists. Kept separate from [`run`] so the caller can
/// order it ahead of everything else in `main`.
///
/// # Safety
/// Must be called while the process is still single-threaded.
pub fn apply_env(threads: usize, n_gpu_layers: usize) {
    // SAFETY: called from `main` before rayon/Metal workers spawn.
    unsafe {
        if threads > 0 {
            std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
            std::env::set_var("FERROX_CPU_THREADS", threads.to_string());
        }
        if n_gpu_layers == 0 {
            std::env::set_var("FERROX_METAL", "0");
            std::env::set_var("FERROX_METAL_ATTN", "0");
            std::env::set_var("FERROX_CUDA", "0");
        } else {
            std::env::set_var("FERROX_METAL", "auto");
            std::env::set_var("FERROX_METAL_ATTN", "1");
            std::env::set_var("FERROX_CUDA", "auto");
        }
        ferrox_core::weight_matrix::default_cpu_int_dot_on();
    }
    ferrox_core::threads::init_cpu_pool();
}

pub struct BenchArgs {
    pub model: String,
    pub n_prompt: usize,
    pub n_gen: usize,
    pub reps: usize,
    pub ctx_size: usize,
    /// Also run `llama-bench` on the same GGUF with matching flags and
    /// report the gap, instead of leaving the comparison to the reader.
    pub compare: bool,
    /// Backend label recorded in the receipt (`cpu` / `metal` / `cuda`).
    pub backend: String,
    /// Where to write the JSON receipt, if anywhere.
    pub receipt: Option<std::path::PathBuf>,
    /// Suite id this run belongs to, for the receipt.
    pub id: Option<String>,
}

pub fn run(args: BenchArgs) -> anyhow::Result<()> {
    let model = crate::pull::resolve_model_path(&args.model)?;
    let path = Path::new(&model);
    if !path.exists() {
        anyhow::bail!("model not found: {model}");
    }

    let file = ferrox_gguf::ShardedGguf::open(path)?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    let config = ModelConfig::from_gguf(&file)
        .with_context(|| format!("reading model config for arch {arch}"))?;
    ferrox_models::ensure_generic_decoder(&arch)
        .map_err(|e| anyhow::anyhow!("{e}: `ferrox bench` covers the generic Decoder path only"))?;

    let params_b = estimate_params(&config);
    let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let load_t = Instant::now();
    let decoder = Decoder::from_gguf(path, config)?;
    let load_s = load_t.elapsed().as_secs_f64();

    let ctx_needed = args.n_prompt.max(1) + args.n_gen + 2;
    if args.ctx_size > 0 && ctx_needed > args.ctx_size {
        anyhow::bail!(
            "workload needs {ctx_needed} positions but -c is {}",
            args.ctx_size
        );
    }

    let mut rows: Vec<Row> = Vec::new();
    if args.n_prompt > 0 {
        rows.push(Row {
            test: format!("pp{}", args.n_prompt),
            samples: bench_prefill(&decoder, args.n_prompt, args.reps)?,
        });
    }
    if args.n_gen > 0 {
        rows.push(Row {
            test: format!("tg{}", args.n_gen),
            samples: bench_decode(&decoder, args.n_gen, args.reps)?,
        });
    }

    let backend = active_backend();
    let threads = ferrox_core::threads::resolve_cpu_threads();
    println!(
        "| {:<30} | {:>10} | {:>10} | {:<10} | {:>7} | {:>15} | {:>20} |",
        "model", "size", "params", "backend", "threads", "test", "t/s"
    );
    println!(
        "| {:-<30} | {:->10} | {:->10} | {:-<10} | {:->7} | {:->15} | {:->20} |",
        "", "", "", "", "", "", ""
    );
    for row in &rows {
        println!(
            "| {:<30} | {:>10} | {:>10} | {:<10} | {:>7} | {:>15} | {:>13.2} ± {:>4.2} |",
            truncate(&format!("{arch} {}", quant_label(&file)), 30),
            human_bytes(size_bytes),
            human_params(params_b),
            backend,
            threads,
            row.test,
            row.median(),
            row.stddev(),
        );
    }
    eprintln!(
        "\nferrox bench: load {load_s:.2}s, {} reps + 1 discarded warmup each",
        args.reps
    );

    let ngl = if backend == "CPU" { 0 } else { 99 };
    let llama = if args.compare {
        match run_llama_bench(&model, args.n_prompt, args.n_gen, args.reps, ngl) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("ferrox bench: llama-bench comparison unavailable: {e}");
                None
            }
        }
    } else {
        eprintln!(
            "compare: llama-bench -m {model} -p {} -n {} -ngl {ngl}   (or pass --compare)",
            args.n_prompt, args.n_gen,
        );
        None
    };

    if let Some(llama) = &llama {
        println!();
        println!(
            "| {:<15} | {:>12} | {:>12} | {:>8} |",
            "test", "ferrox", "llama.cpp", "gap"
        );
        println!("| {:-<15} | {:->12} | {:->12} | {:->8} |", "", "", "", "");
        for row in &rows {
            let l = llama.get(&row.test).copied();
            let gap = l.map(|l| l / row.median());
            println!(
                "| {:<15} | {:>12.2} | {:>12} | {:>8} |",
                row.test,
                row.median(),
                l.map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into()),
                gap.map(|g| format!("{g:.2}×"))
                    .unwrap_or_else(|| "—".into()),
            );
        }
        eprintln!("gap = llama / ferrox; <1 means ferrox is faster");
    }

    if let Some(dest) = &args.receipt {
        write_receipt(
            dest,
            &args,
            &model,
            &arch,
            &file,
            size_bytes,
            params_b,
            threads,
            backend,
            &rows,
            llama.as_ref(),
            load_s,
        )?;
        eprintln!("ferrox bench: receipt written to {}", dest.display());
    }
    Ok(())
}

/// `pp<N>`: one batched forward over N tokens into fresh KV caches.
/// Reported as prompt tokens per second, as `llama-bench` does.
fn bench_prefill(decoder: &Decoder, n_prompt: usize, reps: usize) -> anyhow::Result<Vec<f64>> {
    let tokens = synthetic_tokens(decoder, n_prompt);
    let mut out = Vec::with_capacity(reps);
    for rep in 0..reps + 1 {
        let mut caches = fresh_caches(decoder);
        let t = Instant::now();
        let rows = decoder.forward_batch(&tokens, 0, &mut caches);
        let dt = t.elapsed().as_secs_f64();
        anyhow::ensure!(
            !rows.is_empty(),
            "forward_batch returned no logits rows for a {n_prompt}-token prompt"
        );
        // Rep 0 is the warmup: first touch of every weight page, and on
        // Metal the first pipeline compile. Timing it would measure the
        // OS and the shader cache, not the engine.
        if rep > 0 {
            out.push(n_prompt as f64 / dt);
        }
    }
    Ok(out)
}

/// `tg<N>`: N single-token decode steps after a one-token prime, so KV
/// length grows exactly as it does in real generation.
fn bench_decode(decoder: &Decoder, n_gen: usize, reps: usize) -> anyhow::Result<Vec<f64>> {
    let vocab = decoder.config.vocab_size;
    let mut out = Vec::with_capacity(reps);
    for rep in 0..reps + 1 {
        let mut caches = fresh_caches(decoder);
        let _ = decoder.forward_token(0, 0, &mut caches);
        let t = Instant::now();
        for i in 0..n_gen {
            let tok = (i + 1) % vocab;
            let _ = decoder.forward_token(tok, i + 1, &mut caches);
        }
        let dt = t.elapsed().as_secs_f64();
        if rep > 0 {
            out.push(n_gen as f64 / dt);
        }
    }
    Ok(out)
}

fn fresh_caches(decoder: &Decoder) -> Vec<KvCache> {
    decoder
        .layers
        .iter()
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect()
}

/// Token ids that exist in the vocabulary but carry no linguistic
/// meaning: the point is to move the same bytes through the same
/// kernels, not to generate text.
fn synthetic_tokens(decoder: &Decoder, n: usize) -> Vec<usize> {
    let vocab = decoder.config.vocab_size.max(1);
    (0..n).map(|i| (i * 7 + 1) % vocab).collect()
}

fn active_backend() -> &'static str {
    #[cfg(feature = "metal")]
    {
        if ferrox_core::weight_matrix::metal_dense_enabled() {
            return "Metal";
        }
    }
    #[cfg(feature = "cuda")]
    {
        if ferrox_core::weight_matrix::cuda_dense_enabled() {
            return "CUDA";
        }
    }
    "CPU"
}

/// Rough total for the report column only — embeddings plus the
/// per-layer estimate `ModelConfig` already computes. Not a substitute
/// for the GGUF's own tensor accounting, and not used in any tok/s math.
fn estimate_params(config: &ModelConfig) -> u64 {
    let embeddings = 2 * config.vocab_size as u64 * config.hidden_dim as u64;
    embeddings + config.approx_active_params_per_token() as u64
}

fn quant_label(file: &ferrox_gguf::ShardedGguf) -> String {
    file.metadata_str("general.file_type")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "quantized".to_string())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

fn human_bytes(b: u64) -> String {
    let mib = b as f64 / (1024.0 * 1024.0);
    if mib >= 1024.0 {
        format!("{:.2} GiB", mib / 1024.0)
    } else {
        format!("{mib:.2} MiB")
    }
}

fn human_params(p: u64) -> String {
    let b = p as f64;
    if b >= 1e9 {
        format!("{:.2} B", b / 1e9)
    } else {
        format!("{:.2} M", b / 1e6)
    }
}

/// Runs `llama-bench` on the same GGUF with matching workload flags and
/// parses its markdown table into `{test -> tok/s}`.
///
/// Thread count is deliberately *not* forced. Each engine picking its
/// own default is the comparison that means something: llama.cpp
/// defaults to performance cores and degrades sharply above them, so
/// pinning both engines to the same oversubscribed count (as the old
/// suite did with `-t 10`) handicaps llama by 2-4x on Apple Silicon and
/// flatters ferrox.
fn run_llama_bench(
    model: &str,
    n_prompt: usize,
    n_gen: usize,
    reps: usize,
    ngl: usize,
) -> anyhow::Result<std::collections::BTreeMap<String, f64>> {
    let out = std::process::Command::new("llama-bench")
        .args([
            "-m",
            model,
            "-p",
            &n_prompt.to_string(),
            "-n",
            &n_gen.to_string(),
            "-r",
            &reps.to_string(),
            "-ngl",
            &ngl.to_string(),
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("could not run llama-bench (is it on PATH?): {e}"))?;
    if !out.status.success() {
        anyhow::bail!("llama-bench exited with {}", out.status);
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(parse_llama_bench_table(&text))
}

/// Extracts `{test -> tok/s}` from `llama-bench`'s markdown table.
///
/// Its rows look like `| model | size | params | backend | threads |
/// test | t/s |`, where the `t/s` cell is `123.45 ± 6.78`. Anything that
/// does not have that shape (header, separator, the trailing `build:`
/// line) is skipped rather than guessed at.
fn parse_llama_bench_table(text: &str) -> std::collections::BTreeMap<String, f64> {
    let mut out = std::collections::BTreeMap::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 8 {
            continue;
        }
        let test = cells[6];
        if test.is_empty() || test == "test" || test.starts_with('-') {
            continue;
        }
        let Some(value) = cells[7].split('±').next() else {
            continue;
        };
        if let Ok(v) = value.trim().parse::<f64>() {
            out.insert(test.to_string(), v);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)] // one call site; every field is reported
fn write_receipt(
    dest: &Path,
    args: &BenchArgs,
    model: &str,
    arch: &str,
    file: &ferrox_gguf::ShardedGguf,
    size_bytes: u64,
    params: u64,
    threads: usize,
    backend: &str,
    rows: &[Row],
    llama: Option<&std::collections::BTreeMap<String, f64>>,
    load_s: f64,
) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tests = Vec::new();
    for row in rows {
        let l = llama.and_then(|m| m.get(&row.test)).copied();
        tests.push(serde_json::json!({
            "test": row.test,
            "ferrox_tps": row.median(),
            "ferrox_stddev": row.stddev(),
            "ferrox_samples": row.samples,
            "llama_tps": l,
            "gap": l.map(|l| l / row.median()),
        }));
    }
    let receipt = serde_json::json!({
        "schema": 1,
        "kind": "engine",
        "id": args.id,
        "model_path": model,
        "arch": arch,
        "quant": quant_label(file),
        "size_bytes": size_bytes,
        "approx_params": params,
        "backend": args.backend,
        "backend_active": backend,
        "threads": threads,
        "reps": args.reps,
        "load_s": load_s,
        "ferrox_version": env!("CARGO_PKG_VERSION"),
        "tests": tests,
    });
    std::fs::write(dest, serde_json::to_string_pretty(&receipt)? + "\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(samples: &[f64]) -> Row {
        Row {
            test: "tg128".to_string(),
            samples: samples.to_vec(),
        }
    }

    #[test]
    fn median_of_an_odd_sample_count_is_the_middle_value_regardless_of_order() {
        assert_eq!(row(&[30.0, 10.0, 20.0]).median(), 20.0);
    }

    #[test]
    fn median_of_an_even_sample_count_averages_the_two_middle_values() {
        assert_eq!(row(&[10.0, 20.0, 30.0, 40.0]).median(), 25.0);
    }

    #[test]
    fn stddev_is_zero_for_identical_samples_and_undefined_counts_report_zero() {
        assert_eq!(row(&[5.0, 5.0, 5.0]).stddev(), 0.0);
        assert_eq!(row(&[5.0]).stddev(), 0.0);
        assert_eq!(row(&[]).stddev(), 0.0);
    }

    #[test]
    fn stddev_is_the_population_form_that_llama_bench_prints() {
        // mean 20, deviations -10/0/+10 -> sqrt(200/3), not the sample
        // form sqrt(200/2). Matching llama-bench matters because these
        // numbers are printed side by side.
        let got = row(&[10.0, 20.0, 30.0]).stddev();
        assert!(
            (got - (200.0f64 / 3.0).sqrt()).abs() < 1e-9,
            "population stddev expected, got {got}"
        );
    }

    #[test]
    fn human_readable_sizes_switch_units_at_a_gibibyte() {
        assert_eq!(human_bytes(512 * 1024 * 1024), "512.00 MiB");
        assert_eq!(human_bytes(2 * 1024 * 1024 * 1024), "2.00 GiB");
    }

    #[test]
    fn human_readable_params_switch_units_at_a_billion() {
        assert_eq!(human_params(135_000_000), "135.00 M");
        assert_eq!(human_params(8_000_000_000), "8.00 B");
    }

    #[test]
    fn truncate_leaves_short_strings_alone_and_clips_long_ones() {
        assert_eq!(truncate("llama Q8_0", 30), "llama Q8_0");
        assert_eq!(truncate(&"x".repeat(40), 30), "x".repeat(30));
    }
}
