//! Answer parity against llama.cpp, measured on the distribution rather
//! than on sampled text.
//!
//! Why this exists: `ferrox verify` compares ferrox-CPU against
//! ferrox-Metal, so it can prove the two ferrox backends agree with each
//! other while both disagree with the reference implementation. The
//! quality gate in `docs/plans/llama-cpp-parity-push.md` says a row is
//! only closed when "answers match llama", and nothing measured that.
//!
//! Why not greedy text: it was tried. With tokenization matched by hand
//! (6 tokens on both sides) ferrox and llama.cpp still diverged after
//! about three tokens on TinyLlama Q8_0, a model with no exotic graph
//! features. Greedy text is a chain of argmaxes, so ordinary
//! last-bit numeric drift flips one token and every token after it is
//! then conditioned on different input. A text diff cannot separate "the
//! graph is wrong" from "two near-tied logits swapped", and those are
//! the only two hypotheses worth telling apart.
//!
//! So this compares the FIRST-TOKEN distribution at the last prompt
//! position: one forward pass per engine, no sampling, no feedback. The
//! same token ids are fed to both sides, which removes the tokenizer
//! from the experiment — ferrox tokenizes, and llama.cpp is handed the
//! resulting ids.
//!
//! The reference side is a small C program (`tools/llama_logits.c`)
//! linked against the installed `libllama`, in the same spirit as the
//! IQ-tier goldens, which link ggml's own `ggml-quants.c` rather than
//! re-reading the format spec. A reference you re-implemented is not a
//! reference. Build it with `tools/build_llama_logits.sh`.

use anyhow::Context;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Where the reference dumper is looked for, in order. `target/` is
/// where `tools/build_llama_logits.sh` puts it; the second entry keeps
/// working for anyone who built it by hand before it moved into the
/// tracked tree.
const DUMPER_CANDIDATES: &[&str] = &["target/llama_logits", ".local-scripts/llama_logits"];

/// Environment override, for a dumper built somewhere else entirely.
const DUMPER_ENV: &str = "FERROX_LLAMA_LOGITS";

/// Prompt held fixed so runs are comparable across models and sessions.
const PROMPT: &str = "The capital of France is";

/// Below this KL the two engines are doing the same arithmetic to within
/// f32 accumulation-order noise. Chosen as the scale at which a
/// reordered reduction over a few thousand terms lands: it is a noise
/// floor, not a quality target.
const KL_NOISE: f64 = 1e-3;

/// Above this KL the distributions differ by more than accumulation
/// order — a different rope base, a missing bias, a skipped norm. This
/// is the "wrong graph" line.
const KL_WRONG: f64 = 1e-2;

pub struct ParityArgs {
    pub model: String,
    pub prompt: Option<String>,
    pub prompt_tokens: Option<usize>,
    pub top_k: usize,
    /// Path to the compiled reference dumper.
    pub dumper: Option<String>,
}

pub fn run(args: ParityArgs) -> anyhow::Result<()> {
    let path = crate::pull::resolve_model_path(&args.model)?;
    let prompt = args.prompt.clone().unwrap_or_else(|| PROMPT.to_string());

    let (tokens, ferrox_logits) =
        crate::verify_engine::prefill_logits(Path::new(&path), &prompt, args.prompt_tokens)
            .context("ferrox prefill")?;

    // The reference is pinned to llama.cpp's CPU path (n_gpu_layers = 0),
    // because ferrox's CPU path is the one cross-validated against NumPy.
    // Say which side ferrox ran on rather than leaving it to the
    // environment: a Metal-vs-llama-CPU verdict has two possible causes
    // and must not be read as one.
    let backend = ferrox_core::weight_matrix::active_backend();
    if backend != ferrox_core::kernel_registry::Backend::Cpu {
        eprintln!(
            "parity: ferrox ran on {}, the reference on llama.cpp CPU — a non-MATCH verdict \
             here could be either engine or the backend difference. Run with FERROX_METAL=0 \
             FERROX_CUDA=0 to isolate the engines.",
            backend.as_str()
        );
    }

    let dumper = dumper_path(args.dumper.as_deref())?;
    let ref_logits = reference_logits(&dumper, &path, &tokens)?;

    if ref_logits.len() != ferrox_logits.len() {
        // Not a tolerance question: the two engines disagree about how
        // many tokens the model can emit, which is a loader bug on one
        // side and makes every other number here meaningless.
        anyhow::bail!(
            "vocab size disagrees: llama.cpp {} vs ferrox {} — the logit vectors are not \
             comparable, fix the loader before reading any metric",
            ref_logits.len(),
            ferrox_logits.len()
        );
    }

    let report = compare(&ref_logits, &ferrox_logits, args.top_k);
    print_report(
        &args.model,
        tokens.len(),
        args.top_k,
        backend.as_str(),
        &report,
    );

    if report.verdict == Verdict::Wrong {
        anyhow::bail!(
            "ferrox disagrees with llama.cpp beyond numeric noise (KL {:.3e} nats)",
            report.kl_ref_ferrox
        );
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// Same distribution to within accumulation-order noise.
    Match,
    /// Same top-1, but the distributions drift more than noise explains.
    Drift,
    /// Top-1 differs, and the reference's own top-2 margin is so small
    /// that noise of the observed size is enough to swap them. This is
    /// NOT evidence of a wrong graph and must not be reported as one.
    TieFlip,
    /// Different distributions.
    Wrong,
}

struct Report {
    verdict: Verdict,
    kl_ref_ferrox: f64,
    kl_ferrox_ref: f64,
    total_variation: f64,
    max_prob_delta: f64,
    top1_ref: usize,
    top1_ferrox: usize,
    /// Where the reference's top-1 sits in ferrox's ordering (0 = top).
    ref_top1_rank_in_ferrox: usize,
    /// p1 - p2 in the reference distribution: how close the call was.
    ref_top2_margin: f64,
    topk_overlap: usize,
}

fn compare(ref_logits: &[f32], ferrox_logits: &[f32], k: usize) -> Report {
    let p = softmax(ref_logits);
    let q = softmax(ferrox_logits);

    let mut kl_pq = 0.0f64;
    let mut kl_qp = 0.0f64;
    let mut tv = 0.0f64;
    let mut max_delta = 0.0f64;
    for i in 0..p.len() {
        let (pi, qi) = (p[i], q[i]);
        // Terms where p is zero contribute nothing to KL(p||q) by the
        // 0 ln 0 = 0 convention; guarding q avoids an infinity from a
        // single underflowed float rather than from a real disagreement.
        if pi > 0.0 {
            kl_pq += pi * (pi.max(f64::MIN_POSITIVE) / qi.max(f64::MIN_POSITIVE)).ln();
        }
        if qi > 0.0 {
            kl_qp += qi * (qi.max(f64::MIN_POSITIVE) / pi.max(f64::MIN_POSITIVE)).ln();
        }
        let d = (pi - qi).abs();
        tv += d;
        if d > max_delta {
            max_delta = d;
        }
    }
    tv *= 0.5;

    let ref_order = order_desc(&p);
    let fx_order = order_desc(&q);
    let top1_ref = ref_order[0];
    let top1_ferrox = fx_order[0];
    let ref_top1_rank_in_ferrox = fx_order.iter().position(|&i| i == top1_ref).unwrap_or(0);
    let ref_top2_margin = if ref_order.len() > 1 {
        p[ref_order[0]] - p[ref_order[1]]
    } else {
        1.0
    };

    let k = k.min(ref_order.len());
    let ref_top: std::collections::HashSet<usize> = ref_order[..k].iter().copied().collect();
    let topk_overlap = fx_order[..k].iter().filter(|i| ref_top.contains(i)).count();

    let verdict = if top1_ref == top1_ferrox {
        if kl_pq < KL_NOISE {
            Verdict::Match
        } else if kl_pq < KL_WRONG {
            Verdict::Drift
        } else {
            Verdict::Wrong
        }
    } else if kl_pq < KL_WRONG && ref_top2_margin <= max_delta {
        // The two engines put nearly the same mass on both candidates and
        // the reference itself could barely separate them. Calling this a
        // failure would make the instrument cry wolf on exactly the case
        // that motivated building it.
        Verdict::TieFlip
    } else {
        Verdict::Wrong
    };

    Report {
        verdict,
        kl_ref_ferrox: kl_pq,
        kl_ferrox_ref: kl_qp,
        total_variation: tv,
        max_prob_delta: max_delta,
        top1_ref,
        top1_ferrox,
        ref_top1_rank_in_ferrox,
        ref_top2_margin,
        topk_overlap,
    }
}

fn print_report(model: &str, n_tokens: usize, k: usize, backend: &str, r: &Report) {
    let name = Path::new(model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.to_string());
    let verdict = match r.verdict {
        Verdict::Match => "MATCH",
        Verdict::Drift => "DRIFT",
        Verdict::TieFlip => "TIE-FLIP",
        Verdict::Wrong => "WRONG",
    };
    println!(
        "parity {name}: {verdict} ({n_tokens}-token prompt, first-token distribution, \
         ferrox {backend} vs llama.cpp cpu)"
    );
    println!(
        "  KL(llama||ferrox) {:.3e} nats   KL(ferrox||llama) {:.3e} nats",
        r.kl_ref_ferrox, r.kl_ferrox_ref
    );
    println!(
        "  total variation   {:.3e}        max |delta p|     {:.3e}",
        r.total_variation, r.max_prob_delta
    );
    println!(
        "  top-1  llama {} / ferrox {}   (llama's top-1 is rank {} for ferrox)",
        r.top1_ref, r.top1_ferrox, r.ref_top1_rank_in_ferrox
    );
    println!(
        "  top-{k} overlap    {}/{k}          llama top-2 margin {:.3e}",
        r.topk_overlap, r.ref_top2_margin
    );
    match r.verdict {
        Verdict::Match => println!("  same distribution to within f32 accumulation-order noise."),
        Verdict::Drift => println!(
            "  same token, but the distributions differ by more than accumulation order \
             explains — worth a per-layer divergence run before trusting this row."
        ),
        Verdict::TieFlip => println!(
            "  top-1 differs, but llama's own top-2 margin ({:.3e}) is under the observed \
             per-token noise ({:.3e}): a tie swapped, not a wrong graph.",
            r.ref_top2_margin, r.max_prob_delta
        ),
        Verdict::Wrong => println!(
            "  the graphs disagree. This is not sampling noise and not a tie: something in \
             the forward pass differs."
        ),
    }
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut out: Vec<f64> = logits.iter().map(|&v| (v as f64 - max).exp()).collect();
    let sum: f64 = out.iter().sum();
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

fn order_desc(p: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    // Ties break by index on both sides, so a tie can never by itself
    // make the two engines' orderings look different.
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
    idx
}

fn dumper_path(explicit: Option<&str>) -> anyhow::Result<PathBuf> {
    if let Some(e) = explicit {
        let p = PathBuf::from(e);
        if p.exists() {
            return Ok(p);
        }
        // An explicit path that does not exist is a typo, not a reason to
        // silently fall back to some other binary and report its answer
        // as the reference.
        anyhow::bail!("--dumper {} does not exist", p.display());
    }
    if let Some(e) = std::env::var_os(DUMPER_ENV) {
        let p = PathBuf::from(e);
        if p.exists() {
            return Ok(p);
        }
        anyhow::bail!(
            "{DUMPER_ENV} points at {}, which does not exist",
            p.display()
        );
    }
    for c in DUMPER_CANDIDATES {
        let p = PathBuf::from(c);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!(
        "reference dumper not built. It links llama.cpp's own library, so it is built \
         separately from the cargo workspace:\n\n  ./tools/build_llama_logits.sh\n\n\
         (set LLAMA_CPP_PREFIX if llama.cpp is not a Homebrew install, or --dumper / \
         {DUMPER_ENV} to point at a binary elsewhere)"
    )
}

/// Runs the reference dumper on the SAME token ids and reads back its
/// logit vector.
fn reference_logits(dumper: &Path, model: &str, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
    let out_path = std::env::temp_dir().join(format!("ferrox-parity-{}.bin", std::process::id()));
    let mut cmd = Command::new(dumper);
    cmd.arg(model).arg(&out_path);
    for t in tokens {
        cmd.arg(t.to_string());
    }
    let out = cmd.output().context("running the reference dumper")?;
    if !out.status.success() {
        anyhow::bail!(
            "reference dumper failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("(no stderr)")
        );
    }
    let bytes = std::fs::read(&out_path)
        .with_context(|| format!("reading reference logits from {}", out_path.display()))?;
    let _ = std::fs::remove_file(&out_path);
    if bytes.len() % 4 != 0 {
        anyhow::bail!("reference logits file is not a whole number of f32");
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_logits_are_a_match() {
        let l = vec![0.1f32, 5.0, -2.0, 3.3];
        let r = compare(&l, &l, 3);
        assert_eq!(r.verdict, Verdict::Match);
        assert!(r.kl_ref_ferrox < 1e-12, "KL was {}", r.kl_ref_ferrox);
        assert_eq!(r.top1_ref, r.top1_ferrox);
        assert_eq!(r.topk_overlap, 3);
        assert_eq!(r.ref_top1_rank_in_ferrox, 0);
    }

    #[test]
    fn a_different_graph_is_reported_wrong() {
        // Mass moved to a different token entirely.
        let a = vec![0.0f32, 8.0, 0.0, 0.0];
        let b = vec![0.0f32, 0.0, 8.0, 0.0];
        let r = compare(&a, &b, 2);
        assert_eq!(r.verdict, Verdict::Wrong);
        assert_ne!(r.top1_ref, r.top1_ferrox);
    }

    #[test]
    fn a_near_tie_that_swaps_is_a_tie_flip_not_a_failure() {
        // Two candidates the reference itself can barely separate: the
        // whole point of measuring the distribution instead of the draw.
        let a = vec![-10.0f32, 2.000_01, 2.0];
        let b = vec![-10.0f32, 2.0, 2.000_01];
        let r = compare(&a, &b, 2);
        assert_eq!(r.verdict, Verdict::TieFlip);
        assert_ne!(r.top1_ref, r.top1_ferrox);
        // Both engines still agree on WHICH two tokens are in play.
        assert_eq!(r.topk_overlap, 2);
    }

    #[test]
    fn same_top1_with_a_shifted_tail_is_drift_not_a_match() {
        // Top-1 survives, but the rest of the distribution has moved
        // further than accumulation order explains.
        let a = vec![6.0f32, 1.0, 1.0, 1.0];
        let b = vec![6.0f32, 1.0, 1.0, 2.0];
        let r = compare(&a, &b, 4);
        assert_eq!(r.verdict, Verdict::Drift);
        assert_eq!(r.top1_ref, r.top1_ferrox);
        assert!(r.kl_ref_ferrox >= KL_NOISE && r.kl_ref_ferrox < KL_WRONG);
    }

    #[test]
    fn a_uniform_logit_shift_is_invisible_because_softmax_is_shift_invariant() {
        // Guards against ever "fixing" a false alarm by comparing raw
        // logits: an additive constant is not a disagreement.
        let a = vec![0.5f32, 1.5, -3.0];
        let b: Vec<f32> = a.iter().map(|v| v + 7.25).collect();
        let r = compare(&a, &b, 3);
        assert_eq!(r.verdict, Verdict::Match);
    }
}
