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
//! Removing the tokenizer from THIS experiment is right, and for years
//! it also meant nothing tested the tokenizer at all: a single hardcoded
//! pre-tokenizer regex mistokenized every digit run of four or more and
//! every whitespace run of two or more, on Llama-3.x, Qwen, DeepSeek and
//! SmolLM, and the repo's only cross-engine oracle was structurally
//! unable to see it. So `parity` now runs a SECOND comparison first, in
//! [`tokenize`]: same file, same library, ferrox's token ids against
//! llama.cpp's for a corpus built out of the inputs the pre-tokenizers
//! actually disagree on. It runs before the logit comparison because a
//! tokenizer divergence means the distribution comparison below is
//! measuring two different prompts.
//!
//! The reference side is a small C program (`tools/llama_logits.c`)
//! linked against the installed `libllama`, in the same spirit as the
//! IQ-tier goldens, which link ggml's own `ggml-quants.c` rather than
//! re-reading the format spec. A reference you re-implemented is not a
//! reference. Build it with `tools/build_llama_logits.sh`.

mod dump;
mod tokenize;

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

/// How far two BUILDS OF THE REFERENCE disagree with each other on a
/// K-quant checkpoint. Not a ferrox number: measured with ferrox out of
/// the experiment entirely.
///
/// Measured 2026-09-04 on the fixed parity prompt, comparing the logits
/// of `llama_logits` linked against Homebrew libllama b7650 with the
/// same program linked against `.scratch/llama.cpp` (ggml 0.18.0):
///
/// | checkpoint | KL(b7650 ‖ newer) |
/// |---|---|
/// | Llama-3.2-1B **Q8_0** | **0.0 — bit-identical** |
/// | TinyLlama-1.1B **Q8_0** | **0.0 — bit-identical** |
/// | Llama-3.2-1B IQ4_XS | 2.9e-4 |
/// | Llama-3.2-1B Q6_K | 1.3e-3 |
/// | Llama-3.2-1B Q4_K_M | 2.1e-3 |
/// | Llama-3.2-1B Q5_K_M | 2.1e-3 |
/// | Phi-4-mini Q4_K_M | 5.3e-3 |
/// | DeepSeek-R1-Distill Q4_K_M | 1.6e-2 |
/// | **Qwen2.5-1.5B Q4_K_M** | **2.735e-2** |
///
/// The Q8_0 rows are the control and they are exact zeros: the two
/// builds are the same program for weights llama.cpp dots against
/// Q8_0 activations. Every K-quant row moves, and the newer build's
/// `libggml-cpu` carries interleaved-repack kernels for `block_q5_K`
/// and `block_q6_K` that the bottle does not, which is a change in
/// exactly the arithmetic §10 of the gap inventory is about.
///
/// This constant is here so [`KL_WRONG_Q8K_DOTTED`] is derived from
/// a measurement instead of restated beside one. See
/// [#102](https://github.com/antonellof/ferrox/issues/102).
const KL_REFERENCE_BUILD_SPREAD_KQUANT: f64 = 2.735e-2;

/// The WRONG line for a checkpoint whose arithmetic llama.cpp dots
/// against Q8_K-quantized activations; logits KL is noisier than a
/// single matvec but still same top-1 (DeepSeek-R1 #99).
///
/// It was called `KL_WRONG_LM_HEAD_KQUANT` and applied on the strength
/// of the OUTPUT HEAD's quantization. That was the wrong tensor, and
/// [`DominantQuant`] carries the measurement that says so.
///
/// It is DERIVED from the measured spread above rather than restated
/// beside it, so the two cannot drift apart: the only way to put the
/// line back under the reference's own spread is to edit
/// [`KL_WRONG_Q8K_DOTTED_MARGIN`] below 1, which reads as the
/// mistake it is. It was 2.5e-2, a bare constant BELOW that spread, and
/// the consequence was not hypothetical: Qwen2.5-1.5B
/// Q4_K_M read DRIFT (KL 7.7e-3) against the bottle and WRONG (KL
/// 2.7e-2) against the newer build, from the same ferrox. A line drawn
/// under the reference's own build-to-build spread cannot mean "the
/// graphs disagree", because two binaries with an IDENTICAL graph cross
/// it. Whatever else 2.5e-2 was measuring, it was not that.
///
/// Raising it costs the sensitivity to a real K-quant graph bug in
/// the band 2.5e-2..3.0e-2, which is a real cost. It buys a verdict
/// that does not change with the reference's vintage, and a run whose
/// WRONG can be believed. Note what is NOT raised: [`KL_WRONG`], for
/// everything else, stays at 1e-2, because the same experiment puts the
/// reference's build-to-build spread on a Q8_0 checkpoint at exactly
/// zero.
const KL_WRONG_Q8K_DOTTED: f64 = KL_REFERENCE_BUILD_SPREAD_KQUANT * KL_WRONG_Q8K_DOTTED_MARGIN;

/// How far above the measured reference spread the WRONG line sits.
///
/// A judgement, not a measurement, and deliberately thin: the spread is
/// the largest of nine checkpoints, so a tenth could exceed it, but
/// every point of headroom is sensitivity to a real graph bug thrown
/// away. Widen it only with a checkpoint that measured wider.
const KL_WRONG_Q8K_DOTTED_MARGIN: f64 = 1.1;

/// The verdict ladder, checked at COMPILE time.
///
/// These four constants have to agree about an ordering and until
/// 2026-09-04 nothing made them: `KL_WRONG_Q8K_DOTTED` was a bare
/// 2.5e-2 sitting BELOW the 2.735e-2 that two builds of llama.cpp
/// produce from an identical graph, so the "the graphs disagree" line
/// fired on a difference llama.cpp reproduces against itself, and one
/// real row (#102) read DRIFT or WRONG depending on which bottle was
/// installed. Tightening it again now fails the build rather than
/// quietly re-arming that.
const _: () = {
    assert!(KL_NOISE < KL_WRONG, "MATCH must be tighter than DRIFT");
    assert!(
        KL_WRONG < KL_WRONG_Q8K_DOTTED,
        "the Q8_K-dotted allowance must be a relaxation, not a tightening"
    );
    assert!(
        KL_WRONG_Q8K_DOTTED > KL_REFERENCE_BUILD_SPREAD_KQUANT,
        "a WRONG line under the reference's own build-to-build spread cannot mean \
         `the graphs disagree`: two binaries with an identical graph cross it"
    );
};

pub struct ParityArgs {
    pub model: String,
    pub prompt: Option<String>,
    pub prompt_tokens: Option<usize>,
    pub top_k: usize,
    /// Path to the compiled reference dumper.
    pub dumper: Option<String>,
    /// Prefix to write both compared logit vectors under, so the same
    /// comparison can be redone against a DIFFERENT reference build
    /// without ferrox in the middle. See [`dump`].
    pub dump_logits: Option<String>,
}

pub fn run(args: ParityArgs) -> anyhow::Result<()> {
    let path = crate::pull::resolve_model_path(&args.model)?;

    // Resolved before anything expensive runs: a missing dumper is the
    // most common way this command fails, and finding that out after a
    // multi-second model load helps nobody.
    let dumper = dumper_path(args.dumper.as_deref())?;

    // The tokenizer comparison goes first. It is vocab-only on both
    // sides, so it costs a fraction of the prefill below, and a
    // divergence here means the logit numbers underneath it were
    // computed from a prompt the two engines do not even agree on.
    let tokens_report = tokenize::run(&dumper, Path::new(&path))?;
    match &tokens_report {
        Some(r) => tokenize::print_report(r),
        None => println!(
            "tokenizer: SKIPPED — the installed libllama cannot load this checkpoint, so it has \
             no tokenization to compare against. The logit comparison below will fail for the \
             same reason."
        ),
    }
    println!();

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

    let reference = reference_logits(&dumper, &path, &tokens)?;
    let ref_logits = reference.logits;

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

    // Dumped BEFORE the verdict is computed, so a run that ends in the
    // `bail!` below still leaves the evidence behind. The whole reason
    // to dump is to investigate a bad verdict; losing the vectors
    // exactly when the verdict is bad would defeat it.
    if let Some(prefix) = args.dump_logits.as_deref() {
        let paths = dump::write(prefix, &tokens, &ref_logits, &ferrox_logits)?;
        println!("wrote {}", paths[0].display());
        println!("wrote {}", paths[1].display());
        println!("wrote {}", paths[2].display());
        println!();
    }

    // ONE value, read by the threshold below and by the DRIFT message
    // in `print_report`. Deriving it twice is #109.
    let gguf = ferrox_gguf::GgufFile::open(&path).ok();
    let quant = DominantQuant::of(gguf.as_ref().map_or(&[][..], |f| &f.tensors));

    let report = compare(
        &ref_logits,
        &ferrox_logits,
        args.top_k,
        quant.wrong_kl_threshold(),
    );
    print_report(
        &args.model,
        tokens.len(),
        args.top_k,
        backend.as_str(),
        &quant,
        reference.libllama.as_deref(),
        &report,
    );

    // Both halves are reported before either is allowed to abort the
    // command: a run that printed the tokenizer divergence and then
    // exited would hide the logit numbers that say whether the graph is
    // also wrong, and those are two separate pieces of work.
    let mut failures: Vec<String> = Vec::new();
    if tokens_report.as_ref().is_some_and(|r| r.diverged()) {
        failures.push("ferrox and llama.cpp tokenize the same text differently".to_string());
    }
    if report.verdict == Verdict::Wrong {
        failures.push(format!(
            "ferrox disagrees with llama.cpp beyond numeric noise (KL {:.3e} nats)",
            report.kl_ref_ferrox
        ));
    }
    if !failures.is_empty() {
        anyhow::bail!("{}", failures.join("; "));
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
    /// The same call in LOGIT space, on both sides, and how many
    /// representable f32 values separate the reference's two
    /// candidates.
    ///
    /// A TIE-FLIP verdict is a claim about magnitude — "the two
    /// candidates are closer than the arithmetic can resolve" — and
    /// probability margin does not settle it, because softmax over a
    /// 128k vocabulary turns a wide logit gap into a small probability
    /// gap whenever the distribution is flat. The gap in ulps is the
    /// claim's own units: single digits is a tie, millions is a
    /// difference that happens to sit near the argmax
    /// ([#103](https://github.com/antonellof/ferrox/issues/103)).
    ref_top2_logit_gap: f32,
    ref_top2_logit_gap_ulps: i64,
    /// The same gap as ferrox sees it, signed by the reference's
    /// ordering: negative means ferrox ranks them the other way round.
    ferrox_gap_on_ref_pair: f32,
    topk_overlap: usize,
}

/// Distance between two f32 in representable values.
///
/// `(a - b).abs()` cannot answer "is this a tie": near 3.0 an absolute
/// gap of 1e-6 is about four ulps, near 3e-8 the same gap is the whole
/// number. The count of representable values between two floats is the
/// only scale-free way to say how close a float comparison was, and a
/// tie-flip verdict is exactly that claim.
///
/// Works by mapping f32 to a monotone integer key: for non-negative
/// values the bit pattern already increases with the value, and for
/// negative values it increases as the value decreases, so the
/// magnitude bits are negated.
fn ulps_between(a: f32, b: f32) -> i64 {
    fn key(x: f32) -> i64 {
        let bits = i64::from(x.to_bits() & 0x7fff_ffff);
        if x.is_sign_negative() {
            -bits
        } else {
            bits
        }
    }
    (key(a) - key(b)).abs()
}

fn compare(ref_logits: &[f32], ferrox_logits: &[f32], k: usize, kl_wrong: f64) -> Report {
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
    let (ref_top2_margin, ref_top2_logit_gap, ref_top2_logit_gap_ulps, ferrox_gap_on_ref_pair) =
        if ref_order.len() > 1 {
            let (i1, i2) = (ref_order[0], ref_order[1]);
            (
                p[i1] - p[i2],
                ref_logits[i1] - ref_logits[i2],
                ulps_between(ref_logits[i1], ref_logits[i2]),
                // Same pair, ferrox's numbers, keeping the reference's
                // order: the sign is the flip itself.
                ferrox_logits[i1] - ferrox_logits[i2],
            )
        } else {
            (1.0, 0.0, 0, 0.0)
        };

    let k = k.min(ref_order.len());
    let ref_top: std::collections::HashSet<usize> = ref_order[..k].iter().copied().collect();
    let topk_overlap = fx_order[..k].iter().filter(|i| ref_top.contains(i)).count();

    let verdict = if top1_ref == top1_ferrox {
        if kl_pq < KL_NOISE {
            Verdict::Match
        } else if kl_pq < kl_wrong {
            Verdict::Drift
        } else {
            Verdict::Wrong
        }
    } else if kl_pq < kl_wrong && ref_top2_margin <= max_delta {
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
        ref_top2_logit_gap,
        ref_top2_logit_gap_ulps,
        ferrox_gap_on_ref_pair,
        topk_overlap,
    }
}

/// The quantization of the per-layer weights, and whether llama.cpp
/// dots it against 8-bit-quantized ACTIVATIONS.
///
/// This exists because a `DRIFT` verdict on a K-quant is expected rather
/// than suspicious, and the old message sent the reader off to do a
/// per-layer divergence run on a difference that has a known cause.
///
/// ggml declares a `vec_dot_type` per quantization
/// (`ggml/src/ggml-cpu/ggml-cpu.c`). For the K-quants it is
/// `GGML_TYPE_Q8_K`: llama.cpp quantizes the activation to 8 bits and
/// accumulates in integers. ferrox keeps activations in f32. Both are
/// defensible and ferrox is the more precise of the two, but they are
/// not the same arithmetic, so the distributions differ by more than
/// summation order.
///
/// Measured on five quantizations of one checkpoint: the verdict tracks
/// this predicate on the 96 per-layer tensors exactly. See
/// `docs/plans/llama-cpp-gap-inventory.md` §10.
fn llama_dots_this_against_q8k(kind: &str) -> bool {
    // Spelled as `GgmlType`'s own variant names (`Q4K`, not `Q4_K`) --
    // these come from `format!("{:?}", dtype)`, and writing the ggml
    // spelling here would have matched nothing while looking correct.
    matches!(
        kind,
        "Q2K"
            | "Q3K"
            | "Q4K"
            | "Q5K"
            | "Q6K"
            | "IQ2XXS"
            | "IQ2XS"
            | "IQ2S"
            | "IQ3XXS"
            | "IQ3S"
            | "IQ1S"
            | "IQ1M"
            | "IQ4XS"
    )
}

/// The most common quantization among a checkpoint's PER-LAYER tensors.
///
/// The FILENAME is not this. `Llama-3.2-1B-Instruct-IQ4_XS.gguf`
/// contains no IQ4_XS tensors at all -- 96 of its per-layer weights are
/// `IQ4_NL` -- because the name is the quantization RECIPE and the
/// recipe falls back. Reading the tensor table is the only way to know,
/// and mistaking the two is what made that file look like a
/// counterexample.
///
/// The output head and the embedding table are EXCLUDED, so "body" here
/// means the body and cannot be outvoted into meaning the head on a
/// one-layer model. [`lm_head_quant`] is the other half, and
/// [`DominantQuant`] is the single place that weighs the two.
fn body_quant(tensors: &[ferrox_gguf::TensorInfo]) -> Option<String> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for t in tensors {
        if t.name == "output.weight" || t.name == "token_embd.weight" {
            continue;
        }
        let kind = format!("{:?}", t.dtype);
        if kind != "F32" && kind != "F16" && kind != "BF16" {
            *counts.entry(kind).or_default() += 1;
        }
    }
    counts.into_iter().max_by_key(|(_, n)| *n).map(|(k, _)| k)
}

/// Dtype of the logits projection: untied `output.weight`, else tied embed.
fn lm_head_quant(tensors: &[ferrox_gguf::TensorInfo]) -> Option<String> {
    tensors
        .iter()
        .find(|t| t.name == "output.weight")
        .or_else(|| tensors.iter().find(|t| t.name == "token_embd.weight"))
        .map(|t| format!("{:?}", t.dtype))
}

/// WHICH QUANTIZATION'S ARITHMETIC DOMINATES THIS COMPARISON — the one
/// value the WRONG threshold and the DRIFT message both read.
///
/// Until [#109](https://github.com/antonellof/ferrox/issues/109) they
/// were two rules. The threshold keyed on the OUTPUT HEAD alone and the
/// message keyed on `lm_head.or(body)`, so a `Q8_0` head over a K-quant
/// body was judged against the line for a model containing no K-quant
/// arithmetic at all, and told to go run a per-layer divergence for a
/// difference §10 fully explains. They agreed on every checkpoint in
/// `models/`, which is why it never fired there — so a checkpoint of
/// that shape was BUILT to see what happens (`llama-quantize --pure
/// --output-tensor-type q8_0 --token-embedding-type q8_0 … Q4_K_S`),
/// against Homebrew libllama b7650:
///
/// | checkpoint | head | body | KL(llama‖ferrox) | verdict then |
/// |---|---|---|---|---|
/// | Qwen3-0.6B q8-head | Q8_0 | Q4_K | **1.297e-2** | **WRONG** |
/// | Qwen3-0.6B `--pure` | Q4_K | Q4_K | 1.975e-2 | DRIFT |
/// | Llama-3.2-1B q8-head | Q8_0 | Q4_K | 1.417e-3 | DRIFT |
/// | Llama-3.2-1B `--pure` | Q4_K | Q4_K | 1.126e-3 | DRIFT |
/// | Llama-3.2-1B q6-head | Q6_K | Q8_0 | **2.313e-4** | MATCH |
///
/// The first two rows are the same body arithmetic on the same
/// architecture, and the row with the MORE precise head read WRONG at a
/// SMALLER KL than the row with the K-quant head read DRIFT. `ferrox
/// parity` exited non-zero on the more accurate of the two.
///
/// The last row is what settles which tensor to key on: a K-quant head
/// over a Q8_0 body lands at 2.313e-4, three orders of magnitude below
/// its K-quant-bodied siblings and inside the MATCH floor. **The body
/// carries the divergence and the head barely contributes** — the body
/// is every layer, the head is one matvec — so the body is what this
/// answers with when it is Q8_K-dotted.
///
/// The head still gets to TRIGGER the relaxation when the body is not
/// Q8_K-dotted, even though the row above says the effect is tiny. The
/// alternative is a rule that can invent a WRONG for a checkpoint whose
/// only Q8_K arithmetic is in the head, which is the defect being
/// fixed, in the other direction; and the sensitivity given up sits at
/// 2.3e-4, four orders below either line, so nothing that could be
/// measured is being traded away. That makes this a pure relaxation of
/// what shipped: no row can move DRIFT → WRONG because of it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DominantQuant(Option<String>);

impl DominantQuant {
    /// Reads the tensor table. An empty slice — the file could not be
    /// opened — and a table with no quantized tensor both give `None`,
    /// which is the unrelaxed [`KL_WRONG`] line.
    fn of(tensors: &[ferrox_gguf::TensorInfo]) -> Self {
        Self::weigh(
            lm_head_quant(tensors).as_deref(),
            body_quant(tensors).as_deref(),
        )
    }

    /// The rule itself, taking the two halves directly so it can be
    /// exercised on shapes no checkpoint on this disk has.
    fn weigh(lm_head: Option<&str>, body: Option<&str>) -> Self {
        // A Q8_K-dotted BODY wins outright: it is every layer, and the
        // measurement above says it is what carries the divergence.
        // Otherwise the head, which still relaxes the line when it is
        // itself a K-quant — that fallback is the `or`, not a third
        // branch, because "the head when it is Q8_K-dotted" and "the
        // head as the plain label" are the same expression and writing
        // them separately gives one arm that can never be reached.
        let picked = if body.is_some_and(llama_dots_this_against_q8k) {
            body
        } else {
            lm_head.or(body)
        };
        Self(picked.map(str::to_owned))
    }

    /// What to call it in the report.
    fn label(&self) -> Option<&str> {
        self.0.as_deref()
    }

    /// Whether llama.cpp dots this checkpoint's dominant arithmetic
    /// against Q8_K-quantized activations. The threshold and the DRIFT
    /// message both go through here, so they cannot disagree.
    fn q8k_dotted(&self) -> bool {
        self.0.as_deref().is_some_and(llama_dots_this_against_q8k)
    }

    /// The "the graphs disagree" line for this checkpoint.
    fn wrong_kl_threshold(&self) -> f64 {
        if self.q8k_dotted() {
            KL_WRONG_Q8K_DOTTED
        } else {
            KL_WRONG
        }
    }
}

fn print_report(
    model: &str,
    n_tokens: usize,
    k: usize,
    backend: &str,
    quant: &DominantQuant,
    libllama: Option<&str>,
    r: &Report,
) {
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
    // Named on every row, not only when it surprises. Two libllama
    // builds gave the same checkpoint DRIFT and WRONG (#102), and the
    // difference between the runs was invisible in this report.
    match libllama {
        Some(p) => println!("  reference         {p}"),
        None => println!(
            "  reference         (this dumper predates the `libllama` line — rebuild with \
             tools/build_llama_logits.sh to have the verdict name its reference)"
        ),
    }
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
    // Printed for every verdict, not only TIE-FLIP: how close the
    // argmax call was is what says whether the NEXT prompt would flip
    // too, and a row that did not flip is the evidence that a row which
    // did was one prompt's luck.
    println!(
        "  top-1/top-2 gap   {:+.3e} logits ({} ulps)   ferrox on the same pair {:+.3e}",
        r.ref_top2_logit_gap, r.ref_top2_logit_gap_ulps, r.ferrox_gap_on_ref_pair
    );
    match r.verdict {
        Verdict::Match => println!("  same distribution to within f32 accumulation-order noise."),
        // `q8k_dotted` is the SAME predicate on the SAME value that
        // chose the WRONG line, so the explanation and the threshold
        // are never about different tensors (#109).
        Verdict::Drift => match quant.label().filter(|_| quant.q8k_dotted()) {
            // Expected, with a named cause. Saying "go run a per-layer
            // divergence" here would send the reader after a bug that
            // is not there.
            Some(q) => println!(
                "  same token. The distributions differ by more than summation order, and for \
                 {q} that is EXPECTED: llama.cpp declares `vec_dot_type = Q8_K` for it, so it \
                 quantizes the ACTIVATION to 8 bits and accumulates in integers, while ferrox \
                 keeps activations in f32. Different arithmetic, with ferrox on the more \
                 precise side. See docs/plans/llama-cpp-gap-inventory.md §10."
            ),
            None => println!(
                "  same token, but the distributions differ by more than accumulation order \
                 explains — worth a per-layer divergence run before trusting this row."
            ),
        },
        Verdict::TieFlip => {
            println!(
                "  top-1 differs, but llama's own top-2 margin ({:.3e}) is under the observed \
                 per-token noise ({:.3e}): a tie swapped, not a wrong graph.",
                r.ref_top2_margin, r.max_prob_delta
            );
            // The magnitude, in the units the claim is made in. Without
            // it "tie" is an assertion; with it a reader can check
            // whether the two candidates really are inseparable or
            // whether a real difference merely landed near the argmax.
            println!(
                "  the two candidates are {} ulps apart in llama's logits ({:+.3e} absolute); \
                 ferrox puts the same pair {:+.3e} apart, so the flip is a sign change of a gap \
                 this size — not a redistribution.",
                r.ref_top2_logit_gap_ulps, r.ref_top2_logit_gap, r.ferrox_gap_on_ref_pair
            );
        }
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

/// What the reference said, and which library said it.
struct Reference {
    logits: Vec<f32>,
    /// The libllama the dumper actually loaded, as it reported itself.
    ///
    /// `None` when the dumper predates the line — an older binary is
    /// still usable, it just cannot be attributed, and that is worth
    /// saying rather than guessing a path from the dumper's own.
    libllama: Option<String>,
}

/// Prefix the dumper uses to report the library it loaded. Spelled once
/// here and once in `tools/llama_logits.c`; the parse is tolerant of
/// its absence precisely because those two are the only things that
/// have to agree and nothing links them.
const LIBLLAMA_LINE: &str = "libllama ";

/// Runs the reference dumper on the SAME token ids and reads back its
/// logit vector.
fn reference_logits(dumper: &Path, model: &str, tokens: &[u32]) -> anyhow::Result<Reference> {
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
    Ok(Reference {
        logits: bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        libllama: parse_libllama(&String::from_utf8_lossy(&out.stdout)),
    })
}

/// Picks the `libllama <path>` line out of the dumper's stdout.
fn parse_libllama(stdout: &str) -> Option<String> {
    stdout
        .lines()
        .find_map(|l| l.trim().strip_prefix(LIBLLAMA_LINE))
        .map(str::trim)
        .filter(|p| !p.is_empty() && *p != "unknown")
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_logits_are_a_match() {
        let l = vec![0.1f32, 5.0, -2.0, 3.3];
        let r = compare(&l, &l, 3, KL_WRONG);
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
        let r = compare(&a, &b, 2, KL_WRONG);
        assert_eq!(r.verdict, Verdict::Wrong);
        assert_ne!(r.top1_ref, r.top1_ferrox);
    }

    #[test]
    fn a_near_tie_that_swaps_is_a_tie_flip_not_a_failure() {
        // Two candidates the reference itself can barely separate: the
        // whole point of measuring the distribution instead of the draw.
        let a = vec![-10.0f32, 2.000_01, 2.0];
        let b = vec![-10.0f32, 2.0, 2.000_01];
        let r = compare(&a, &b, 2, KL_WRONG);
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
        let r = compare(&a, &b, 4, KL_WRONG);
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
        let r = compare(&a, &b, 3, KL_WRONG);
        assert_eq!(r.verdict, Verdict::Match);
    }

    /// Adjacent floats are one ulp apart wherever they sit on the line,
    /// and the count is scale-free.
    ///
    /// This is the measurement a TIE-FLIP verdict rests on. An absolute
    /// difference cannot make the claim: `3.0` and its neighbour differ
    /// by 2.4e-7 while `3e-8` and its neighbour differ by 1e-15, and
    /// both are one ulp. A tie-flip reported in absolute units is
    /// therefore unfalsifiable, which is what #103 pointed at.
    #[test]
    fn ulps_between_counts_representable_values_not_absolute_distance() {
        for anchor in [3.0f32, 3e-8, -3.0, 1.0, 65_536.0] {
            let next = f32::from_bits(if anchor.is_sign_negative() {
                anchor.to_bits() - 1
            } else {
                anchor.to_bits() + 1
            });
            assert_eq!(
                ulps_between(anchor, next),
                1,
                "{anchor} and its neighbour {next} must be one ulp apart"
            );
        }
        // Symmetric, and zero for equal values.
        assert_eq!(ulps_between(1.5, 1.5), 0);
        assert_eq!(ulps_between(2.0, 1.0), ulps_between(1.0, 2.0));
        // Crossing zero must not wrap: -0.0 and +0.0 are the same
        // value, and a naive bit subtraction would call them 2^31
        // apart, which would turn every sign change into "not a tie".
        assert_eq!(ulps_between(-0.0, 0.0), 0);
        // The two representable values either side of zero are two
        // steps apart, not 2^32: a naive bit subtraction makes any sign
        // change look maximally far, which would turn every flipped
        // near-tie into "not a tie".
        let tiny = f32::from_bits(1);
        assert_eq!(ulps_between(-tiny, tiny), 2);
        // A gap that is genuinely large stays large: 1.0 and 2.0 are a
        // whole binade apart, which is 2^23 representable values.
        assert_eq!(ulps_between(1.0, 2.0), 1 << 23);
    }

    /// The reported top-1/top-2 gap describes the pair the REFERENCE
    /// picked, on both sides, so the flip shows up as a sign change.
    ///
    /// Reporting ferrox's own top-2 pair instead would compare two
    /// different questions and could print two positive gaps for a run
    /// whose whole finding is that the order reversed.
    #[test]
    fn the_reported_gap_is_the_same_pair_on_both_sides_and_flips_sign() {
        let a = vec![-10.0f32, 2.000_01, 2.0];
        let b = vec![-10.0f32, 2.0, 2.000_01];
        let r = compare(&a, &b, 2, KL_WRONG);
        assert_eq!(r.verdict, Verdict::TieFlip);
        assert!(
            r.ref_top2_logit_gap > 0.0,
            "the reference ranks its own pair the right way round"
        );
        assert!(
            r.ferrox_gap_on_ref_pair < 0.0,
            "ferrox ranks the SAME pair the other way: gap {}",
            r.ferrox_gap_on_ref_pair
        );
        assert!(
            r.ref_top2_logit_gap_ulps > 0 && r.ref_top2_logit_gap_ulps < 1_000,
            "a tie must be a handful of ulps, got {}",
            r.ref_top2_logit_gap_ulps
        );
    }

    /// Every K-quant / IQ spelling that reaches the allowance.
    const Q8K_DOTTED: &[&str] = &[
        "Q2K", "Q3K", "Q4K", "Q5K", "Q6K", "IQ2XXS", "IQ2XS", "IQ2S", "IQ3XXS", "IQ3S", "IQ1S",
        "IQ1M", "IQ4XS",
    ];

    #[test]
    fn a_kquant_checkpoint_uses_a_higher_wrong_threshold() {
        let kq = DominantQuant::weigh(Some("Q6K"), Some("Q6K"));
        assert_eq!(kq.wrong_kl_threshold(), KL_WRONG_Q8K_DOTTED);
        let q8 = DominantQuant::weigh(Some("Q8_0"), Some("Q8_0"));
        assert_eq!(q8.wrong_kl_threshold(), KL_WRONG);
    }

    /// EVERY quantization that routes to the K-quant allowance gets a
    /// WRONG line above the reference's own build-to-build spread.
    ///
    /// The ordering of the constants is checked at compile time; this
    /// walks the PREDICATE, which is the half that can silently stop
    /// agreeing with them. A spelling dropped from
    /// `llama_dots_this_against_q8k` sends that quantization back to
    /// `KL_WRONG` = 1e-2, which is a quarter of the spread two llama.cpp
    /// builds show on exactly these types — the row would read WRONG and
    /// the constants would still look right.
    #[test]
    fn every_q8k_dotted_body_clears_the_references_build_to_build_spread() {
        for kind in Q8K_DOTTED {
            let line = DominantQuant::weigh(Some("Q8_0"), Some(kind)).wrong_kl_threshold();
            assert!(
                line > KL_REFERENCE_BUILD_SPREAD_KQUANT,
                "{kind} gets a WRONG line of {line:e}, under the {KL_REFERENCE_BUILD_SPREAD_KQUANT:e} \
                 two builds of llama.cpp produce from an identical graph"
            );
        }
        // And the non-K-quant line is deliberately NOT raised: the same
        // experiment measured the reference's Q8_0 spread at exactly
        // zero, so there is nothing there to make room for.
        assert_eq!(
            DominantQuant::weigh(Some("Q8_0"), Some("Q8_0")).wrong_kl_threshold(),
            1e-2
        );
    }

    /// A `Q8_0` OUTPUT HEAD OVER A K-QUANT BODY IS A K-QUANT
    /// COMPARISON — the shape #109 is about.
    ///
    /// It shipped judged as a Q8_0 one, because the threshold read the
    /// head and never asked what the layers were. No checkpoint in
    /// `models/` has the shape, so one was built:
    /// `llama-quantize --pure --output-tensor-type q8_0
    /// --token-embedding-type q8_0 Qwen3-0.6B-BF16.gguf out.gguf
    /// Q4_K_S`. Against Homebrew libllama b7650 it measured KL
    /// **1.297e-2** and read **WRONG** — while the same model quantized
    /// `--pure` (K-quant head as well) measured a LARGER 1.975e-2 and
    /// read DRIFT. Same body arithmetic, and the more precise of the two
    /// was the one `ferrox parity` exited non-zero on.
    #[test]
    fn a_q8_0_head_over_a_kquant_body_is_judged_as_the_kquant_it_is() {
        for body in Q8K_DOTTED {
            let q = DominantQuant::weigh(Some("Q8_0"), Some(body));
            assert!(
                q.q8k_dotted(),
                "a {body} body dots against Q8_K activations whatever the output head is"
            );
            assert_eq!(q.label(), Some(*body), "the body is what the report names");
            assert_eq!(q.wrong_kl_threshold(), KL_WRONG_Q8K_DOTTED);
        }
        // The measured row, so the constant is tied to the observation
        // rather than merely ordered against another constant.
        let measured_kl = 1.297e-2;
        assert!(
            measured_kl < DominantQuant::weigh(Some("Q8_0"), Some("Q4K")).wrong_kl_threshold(),
            "the constructed Q8_0-head/Q4_K-body checkpoint must read DRIFT, not WRONG"
        );
        // When BOTH halves are Q8_K-dotted the threshold is the same
        // either way, so only the label distinguishes the rules — and
        // the body is the honest label, because it is the layers that
        // carry the drift (2.313e-4 for a K-quant head alone). Keying
        // the label on the head, as the message did before #109, must
        // be visible here rather than only in the shape that changes
        // the verdict.
        assert_eq!(
            DominantQuant::weigh(Some("Q6K"), Some("Q4K")).label(),
            Some("Q4K"),
            "with a K-quant on both ends the report names the body"
        );
    }

    /// The head can still TRIGGER the allowance, so no row can move
    /// DRIFT → WRONG because of #109's fix.
    ///
    /// Measured cost of keeping it: a Q6_K head over a Q8_0 body
    /// (`--pure Q8_0 --output-tensor-type q6_K`) lands at KL 2.313e-4,
    /// four orders below either line, so the sensitivity being conceded
    /// is not sensitivity anything could use.
    #[test]
    fn a_kquant_head_over_a_q8_0_body_still_relaxes_rather_than_inventing_a_wrong() {
        let q = DominantQuant::weigh(Some("Q6K"), Some("Q8_0"));
        assert!(q.q8k_dotted());
        assert_eq!(q.label(), Some("Q6K"));
        assert_eq!(q.wrong_kl_threshold(), KL_WRONG_Q8K_DOTTED);
    }

    fn tensor(name: &str, dtype: ferrox_gguf::GgmlType) -> ferrox_gguf::TensorInfo {
        ferrox_gguf::TensorInfo {
            name: name.to_string(),
            shape: vec![1],
            dtype,
            offset: 0,
        }
    }

    /// The body is read off the LAYERS, and the output head cannot vote
    /// in it.
    ///
    /// `body_quant` counts tensors, so on a checkpoint with few layers
    /// and a head at a different precision the head could otherwise
    /// join the tally it is being weighed against — two roles for one
    /// tensor, which is how the head came to decide the threshold in
    /// the first place. This also pins the shape #109 is about end to
    /// end, from a tensor table rather than from two strings.
    #[test]
    fn the_body_is_read_off_the_layers_and_the_output_head_does_not_vote_in_it() {
        use ferrox_gguf::GgmlType;
        // One layer, so the head would outvote it if it were counted.
        let table = vec![
            tensor("token_embd.weight", GgmlType::Q8_0),
            tensor("output.weight", GgmlType::Q8_0),
            tensor("blk.0.attn_q.weight", GgmlType::Q4K),
            tensor("blk.0.attn_norm.weight", GgmlType::F32),
        ];
        assert_eq!(body_quant(&table).as_deref(), Some("Q4K"));
        assert_eq!(lm_head_quant(&table).as_deref(), Some("Q8_0"));
        let q = DominantQuant::of(&table);
        assert_eq!(
            q.label(),
            Some("Q4K"),
            "a Q8_0 head does not hide a Q4_K body"
        );
        assert_eq!(q.wrong_kl_threshold(), KL_WRONG_Q8K_DOTTED);

        // A tied-embedding model: no `output.weight`, and the embedding
        // is the head. It still must not count as the body.
        let tied = vec![
            tensor("token_embd.weight", GgmlType::Q8_0),
            tensor("blk.0.ffn_up.weight", GgmlType::Q4K),
        ];
        assert_eq!(body_quant(&tied).as_deref(), Some("Q4K"));
        assert_eq!(
            DominantQuant::of(&tied).wrong_kl_threshold(),
            KL_WRONG_Q8K_DOTTED
        );

        // An unreadable file relaxes nothing.
        assert_eq!(DominantQuant::of(&[]).label(), None);
        assert_eq!(DominantQuant::of(&[]).wrong_kl_threshold(), KL_WRONG);
    }

    /// The DRIFT message and the WRONG threshold read ONE value.
    ///
    /// This is #109's actual defect, in the repo's dominant shape: two
    /// rules that had to agree about which quantization a verdict is
    /// about, with nothing enforcing it. Both now go through
    /// `q8k_dotted` on the same `DominantQuant`, and this walks every
    /// head/body combination asserting they never part.
    #[test]
    fn the_drift_message_and_the_wrong_line_never_disagree_about_the_quant() {
        let kinds: Vec<Option<&str>> = std::iter::once(None)
            .chain(
                ["Q8_0", "Q4_0", "Q5_0", "IQ4NL"]
                    .into_iter()
                    .chain(Q8K_DOTTED.iter().copied())
                    .map(Some),
            )
            .collect();
        for head in &kinds {
            for body in &kinds {
                let q = DominantQuant::weigh(*head, *body);
                // The message explains the divergence as the Q8_K
                // activation path exactly when the threshold made room
                // for it. One is the other's condition.
                let message_blames_q8k = q.label().filter(|_| q.q8k_dotted()).is_some();
                let line_made_room = q.wrong_kl_threshold() == KL_WRONG_Q8K_DOTTED;
                assert_eq!(
                    message_blames_q8k, line_made_room,
                    "head {head:?} body {body:?}: message says Q8_K={message_blames_q8k} but \
                     the threshold says {line_made_room}"
                );
                // And the relaxation happens whenever ANY of the
                // checkpoint's arithmetic is Q8_K-dotted, so the fix
                // cannot move a row DRIFT -> WRONG.
                let any_q8k = [*head, *body]
                    .into_iter()
                    .flatten()
                    .any(llama_dots_this_against_q8k);
                assert_eq!(line_made_room, any_q8k);
            }
        }
    }

    /// The dumper's self-report is read, and its absence is not
    /// mistaken for a library called "unknown".
    #[test]
    fn the_reference_identity_is_parsed_and_its_absence_is_not_invented() {
        assert_eq!(
            parse_libllama("libllama /opt/homebrew/lib/libllama.dylib\nn_vocab 32000\n").as_deref(),
            Some("/opt/homebrew/lib/libllama.dylib")
        );
        // An older dumper prints no such line: report that, do not guess.
        assert_eq!(parse_libllama("n_vocab 32000\n"), None);
        // The dumper says "unknown" when dladdr fails; that is an
        // absence too, and passing it through would print a reference
        // path of "unknown" as though it were one.
        assert_eq!(parse_libllama("libllama unknown\nn_vocab 32000\n"), None);
        assert_eq!(parse_libllama("libllama   \n"), None);
    }

    /// The Q8_K predicate is spelled in `GgmlType`'s variant names, not
    /// ggml's.
    ///
    /// `format!("{:?}", dtype)` yields `Q4K`, and ggml calls the same
    /// thing `Q4_K`. Writing the ggml spelling here matches NOTHING
    /// while looking exactly right, and the symptom is the generic
    /// "go run a per-layer divergence" message on every K-quant — which
    /// is the message this predicate exists to suppress. That mistake
    /// was made once already; this is what catches it.
    #[test]
    fn the_q8k_predicate_uses_the_dtype_debug_spelling() {
        for kind in ["Q2K", "Q3K", "Q4K", "Q5K", "Q6K", "IQ4XS", "IQ2S"] {
            assert!(
                llama_dots_this_against_q8k(kind),
                "{kind} declares vec_dot_type = Q8_K in ggml-cpu.c"
            );
        }
        // The ggml spelling must NOT match, or the underscore bug is
        // back and invisible.
        for wrong in ["Q4_K", "Q6_K", "IQ4_XS"] {
            assert!(
                !llama_dots_this_against_q8k(wrong),
                "{wrong} is ggml's spelling, not GgmlType's -- if this now matches, the \
                 predicate is accepting both and the next reader cannot tell which is real"
            );
        }
        // Q8_0-dotted quants must stay out: these are the ones that
        // MATCH, and claiming the divergence is expected for them would
        // excuse a real bug.
        for q8_0_dotted in ["Q8_0", "Q4_0", "Q5_0", "IQ4NL"] {
            assert!(!llama_dots_this_against_q8k(q8_0_dotted));
        }
    }
}
