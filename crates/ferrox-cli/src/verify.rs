//! Cross-backend output agreement.
//!
//! The benchmark harness measures throughput and never looks at what the
//! engine produced. That gap has hidden real bugs more than once: a d=64
//! prefill kernel applied softcap twice through a cross-simdgroup race
//! while reporting healthy tok/s; SmolLM2 once generated word salad on
//! Metal at full speed; Gemma-2 still emits a bare `:` on Metal. Every
//! one of those rows looked green.
//!
//! So: generate the same greedy continuation on two backends and compare
//! the token sequences. Greedy decoding is deterministic, and a kernel
//! that computes the wrong thing diverges within a few tokens. The CPU
//! path is the reference because it is the one cross-validated against
//! NumPy in `gguf_roundtrip.rs`.
//!
//! Backend choice is a process-lifetime `OnceLock` read from the
//! environment, so this cannot switch backends in-process — it re-invokes
//! itself once per backend, exactly as `bench --suite` does.

use anyhow::Context;
use std::path::Path;
use std::process::Command;

/// How many tokens to compare. Divergence from a real kernel bug shows up
/// almost immediately; the cost of going further is wall time on the
/// slowest backend.
const N_TOKENS: usize = 24;

/// Prompt is fixed so runs are comparable across invocations and models.
const PROMPT: &str = "The capital of France is";

pub struct VerifyArgs {
    pub model: String,
    /// Backend to check against the CPU reference: `metal` or `cuda`.
    pub backend: String,
    /// Emit the token ids rather than a verdict (used by the child).
    pub emit: bool,
}

/// Marker the child prints so the parent can find the payload even if the
/// engine writes other things to stdout.
const TAG: &str = "FERROX_VERIFY_TOKENS ";

pub fn run(args: VerifyArgs) -> anyhow::Result<()> {
    if args.emit {
        return emit_tokens(&args.model);
    }

    let reference = child_tokens(&args.model, "cpu")?;
    let candidate = child_tokens(&args.model, &args.backend)?;

    if reference.is_empty() {
        anyhow::bail!("CPU reference produced no tokens; cannot verify");
    }

    let diverge = reference
        .iter()
        .zip(&candidate)
        .position(|(a, b)| a != b)
        .or_else(|| {
            (reference.len() != candidate.len()).then_some(reference.len().min(candidate.len()))
        });

    match diverge {
        None => {
            println!(
                "verify {}: OK — {} tokens identical on cpu and {}",
                short(&args.model),
                reference.len(),
                args.backend
            );
            Ok(())
        }
        Some(i) => {
            // Report where and with what, not just that it failed: the
            // index says whether the kernel is wrong from the first token
            // (dense path) or drifts (accumulating / attention path),
            // which is the first thing anyone debugging this needs.
            println!(
                "verify {}: DIVERGED at token {i} — cpu={:?} {}={:?}",
                short(&args.model),
                &reference[i.saturating_sub(2)..reference.len().min(i + 3)],
                args.backend,
                &candidate[i.saturating_sub(2)..candidate.len().min(i + 3)],
            );
            anyhow::bail!(
                "{} disagrees with the CPU reference from token {i}",
                args.backend
            )
        }
    }
}

fn short(model: &str) -> String {
    Path::new(model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.to_string())
}

/// Runs one backend in a child and parses back the token ids.
fn child_tokens(model: &str, backend: &str) -> anyhow::Result<Vec<u32>> {
    let exe = std::env::current_exe()?;
    let out = Command::new(&exe)
        .arg("verify")
        .args(["-m", model])
        .args(["--backend", backend])
        .arg("--emit")
        // `-ngl` is what actually selects the backend for `run`; the env
        // var alone is overridden by the CLI default.
        .env("FERROX_METAL", if backend == "cpu" { "0" } else { "1" })
        .env("FERROX_CUDA", if backend == "cuda" { "1" } else { "0" })
        .output()
        .with_context(|| format!("spawning verify child for {backend}"))?;
    if !out.status.success() {
        anyhow::bail!(
            "verify child for {backend} failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("(no stderr)")
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix(TAG))
        .with_context(|| format!("verify child for {backend} printed no token line"))?;
    Ok(line
        .split_whitespace()
        .filter_map(|t| t.parse::<u32>().ok())
        .collect())
}

/// Child side: greedy-decode `N_TOKENS` and print the ids.
fn emit_tokens(model: &str) -> anyhow::Result<()> {
    let path = crate::pull::resolve_model_path(model)?;
    let ids = crate::verify_engine::greedy_token_ids(Path::new(&path), PROMPT, N_TOKENS)?;
    let joined: Vec<String> = ids.iter().map(|t| t.to_string()).collect();
    println!("{TAG}{}", joined.join(" "));
    Ok(())
}
