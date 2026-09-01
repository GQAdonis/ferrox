//! Triage of the `LoadError::UnauditedArchitecture` refusals.
//!
//! ferrox's generic GQA path is opt-in: an architecture that is not in
//! `capability::AUDITED_GENERIC_GQA` refuses rather than running on the
//! guess that it is plain GQA. That closed the "loads and computes
//! something else" class — `gpt2`, `mpt`, `refact`, `bloom` and `jais`
//! all did exactly that — but it left **47** architectures refusing with
//! one identical paragraph whose only content is "nobody has checked
//! this".
//!
//! That paragraph is useless for the decision a user actually has, which
//! is whether their model is one fixture away or needs an attention
//! implementation. `docs/plans/llama-cpp-gap-inventory.md` §1.3 shows
//! the 47 split at least three ways, and this suite pins the split for
//! the architectures that have been read on **both** sides:
//!
//! - **fixture-away** — everything is implemented; evidence is missing.
//! - **one match arm** — one small, nameable piece: an activation, a
//!   norm slot, a routing flag, an ordering.
//! - **new code** — a different attention or residual structure.
//! - **UNKNOWN** — reading did not settle it; the verdict says what
//!   would.
//!
//! **Why this suite exists rather than only the unit tests in
//! `capability.rs`:** the failure mode here is not a compile error, it
//! is a *confident wrong verdict*. This repo has now found four
//! architectures whose refusal named something that was not the real
//! blocker — `glm4moe` was told it lacked an MLA hyper-parameter it must
//! not have, and `minimax-m2` was blamed on MTP weights no converter can
//! emit. Each assertion below therefore pins a specific claim about a
//! specific llama.cpp line, so that changing the verdict without
//! changing the reading fails.
//!
//! Nothing here loads a checkpoint. Every claim it pins was read in
//! `.scratch/llama.cpp/src/models/*.cpp` against ferrox's generic
//! decoder, and the citation is in the verdict string itself.

use ferrox_models::capability::{
    architecture_catalog, is_audited_generic, unaudited_refusal_detail, unaudited_triage, ArchPath,
    TriageClass, TRIAGE_PENDING,
};

/// Every architecture that reaches the unaudited refusal gets a
/// non-empty detail line — triaged or not.
///
/// A blank detail would be the old refusal wearing a new field.
#[test]
fn every_unaudited_architecture_renders_a_detail_line() {
    let mut n = 0;
    for p in architecture_catalog() {
        if !matches!(p.path, ArchPath::GenericGqa { .. }) || is_audited_generic(p.gguf_name) {
            continue;
        }
        n += 1;
        let detail = unaudited_refusal_detail(p.gguf_name);
        assert!(
            detail.starts_with("TRIAGE"),
            "`{}` renders {detail:?}",
            p.gguf_name
        );
        assert!(detail.len() > 100, "`{}` renders {detail:?}", p.gguf_name);
    }
    assert_eq!(
        n, 47,
        "the unaudited count moved; docs/plans/llama-cpp-gap-inventory.md §1.2 says 47. \
         Either an architecture was audited (good — update the count and the docs) or one \
         was added (check it was triaged)"
    );
}

/// Batch 1: the architectures people actually download, with the class
/// each was placed in and the llama.cpp fact that decides it.
///
/// The class is asserted together with a substring of the blocker on
/// purpose. Asserting the class alone would let somebody flip a verdict
/// and leave the (now-contradictory) reasoning in place, which is
/// exactly how `glm4moe` came to refuse for a reason it did not have.
#[test]
fn batch_one_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        // --- fixture-away: implemented, unevidenced ------------------
        //
        // gemma.cpp:16-33 creates only the tensors the generic decoder
        // loads; the three Gemma-specific pieces (sqrt(n_embd)
        // embedding scale, GeGLU, 1/sqrt(head_dim) attention scale) are
        // all implemented for GemmaFamily.
        ("gemma", TriageClass::FixtureAway, "gemma.cpp:16-33"),
        // internlm2.cpp:25-33 is the plain Llama tensor set.
        ("internlm2", TriageClass::FixtureAway, "internlm2.cpp:25-33"),
        // exaone.cpp:29-38 likewise, plus the global rope_freqs.weight
        // loader.rs already loads. NOT exaone4, which is a different
        // graph and a different class below.
        ("exaone", TriageClass::FixtureAway, "exaone.cpp:29-38"),
        // ernie4-5.cpp's dense branch; the only unslotted tensor is an
        // OPTIONAL attn_output.bias, which fails closed by name.
        (
            "ernie4_5",
            TriageClass::FixtureAway,
            "attn_output.bias at :45",
        ),
        // bailingmoe2.cpp: fused QKV that ferrox splits, per-head QK
        // norm BEFORE RoPE, sigmoid/softmax read from metadata,
        // exp_probs_b, shared experts, leading dense — all implemented.
        (
            "bailingmoe2",
            TriageClass::FixtureAway,
            "bailingmoe2.cpp is plain GQA",
        ),
        // --- one match arm: small and nameable -----------------------
        //
        // seed-oss.cpp:36-37,113-115 — post_attention_norm IS the
        // pre-FFN norm and there is no ffn_norm. gpt-oss's slot, behind
        // an `is_gpt_oss` flag that would be widened.
        ("seed_oss", TriageClass::OneMatchArm, "is_gpt_oss"),
        // deepseek.cpp:145-155 passes norm_w=false, and the converter
        // never writes expert_weights_norm, so ferrox renormalises
        // where llama.cpp does not.
        (
            "deepseek",
            TriageClass::OneMatchArm,
            "NO_TOPK_RENORMALIZE_ARCHITECTURES",
        ),
        // ernie4-5-moe.cpp:64 — MoE layers are interleaved by
        // n_moe_layer_step, which `layer_is_dense` does not implement.
        (
            "ernie4_5-moe",
            TriageClass::OneMatchArm,
            "interleave_moe_layer_step",
        ),
        // hunyuan-moe.cpp:93-118 — QK norm AFTER RoPE, not before.
        (
            "hunyuan-moe",
            TriageClass::OneMatchArm,
            "AFTER ggml_rope_ext",
        ),
        // --- new code: a different graph -----------------------------
        //
        // olmo2.cpp:47,52,92,169 — no attn_norm and no ffn_norm at all;
        // Q/K/V come off the raw residual.
        ("olmo2", TriageClass::NewCode, "olmo2.cpp:47,52"),
        // exaone4.cpp:60-67,118,159 — the same post-norm-only topology.
        ("exaone4", TriageClass::NewCode, "exaone4.cpp:60-67"),
        // granite.cpp:7-10,188,241-242,301-302 — four multipliers the
        // generic decoder does not apply, plus a rope_finetuned gate.
        ("granite", TriageClass::NewCode, "f_residual_scale"),
        ("granitemoe", TriageClass::NewCode, "f_residual_scale"),
        ("granite-moe", TriageClass::NewCode, "f_residual_scale"),
        // --- unknown: say so, and say what would settle it -----------
        //
        // `phi4` is not a llama.cpp architecture at all, so there is no
        // graph to diff against.
        ("phi4", TriageClass::Unknown, "WHAT WOULD SETTLE IT"),
    ];

    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// The two olmo2-shaped architectures are NOT fixture-away, and this
/// test exists because the inventory said they might be.
///
/// `docs/plans/llama-cpp-gap-inventory.md` §1.3 groups `olmo2`,
/// `seed_oss` and `exaone4` together as "likely a fixture away, if the
/// loader wires the post-norm slots for non-Gemma families". The wiring
/// question has a yes answer — `loader.rs` reads
/// `blk.N.post_attention_norm.weight` and `blk.N.post_ffw_norm.weight`
/// for every non-gpt-oss architecture, and the decoder applies them in
/// llama.cpp's own places — but it is the wrong question. `olmo2` and
/// `exaone4` create **no** `attn_norm` and **no** `ffn_norm`
/// (`olmo2.cpp:42-52`, `exaone4.cpp:53-67`) and project Q/K/V straight
/// off the residual (`olmo2.cpp:92`, `exaone4.cpp:118`), while the
/// generic decoder requires both pre-norms and applies them on every
/// layer. And `seed_oss` is a third thing again: it *has* `attn_norm`
/// and uses `attn_post_norm` as its pre-FFN norm.
///
/// So the three do not share a class. Pinning that stops the grouping
/// from being restored from the prose.
#[test]
fn the_post_norm_group_does_not_share_one_class() {
    let olmo2 = unaudited_triage("olmo2").expect("olmo2 verdict");
    let exaone4 = unaudited_triage("exaone4").expect("exaone4 verdict");
    let seed_oss = unaudited_triage("seed_oss").expect("seed_oss verdict");

    assert_eq!(olmo2.class, TriageClass::NewCode);
    assert_eq!(exaone4.class, TriageClass::NewCode);
    assert_ne!(
        seed_oss.class, olmo2.class,
        "seed_oss has attn_norm and olmo2 does not; they cannot share a class"
    );
    for (name, t) in [("olmo2", olmo2), ("exaone4", exaone4)] {
        assert!(
            t.blocker.contains("NO attn_norm") || t.blocker.contains("no attn_norm"),
            "`{name}`'s verdict must name the missing pre-norm, not the post-norms: {}",
            t.blocker
        );
        assert!(
            !t.blocker.contains("fixture away"),
            "`{name}` is not a fixture away"
        );
    }
}

/// `ernie4_5-moe` is softmax-routed, not sigmoid-routed.
///
/// The inventory (§1.3) lists it beside `bailingmoe2` as "sigmoid-routed
/// MoE with `ffn_exp_probs_b` router bias". `bailingmoe2` reads its
/// gating function from metadata (`bailingmoe2.cpp:11`) so it can be
/// either; `ernie4-5-moe.cpp:90` **hardcodes**
/// `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`. Carrying that error into a
/// refusal would send whoever picks the work up looking at the wrong
/// routing path.
#[test]
fn ernie_moe_is_not_described_as_sigmoid_routed() {
    let t = unaudited_triage("ernie4_5-moe").expect("verdict");
    assert!(
        t.blocker.contains("NOT sigmoid-routed"),
        "ernie4_5-moe's verdict must correct the sigmoid claim: {}",
        t.blocker
    );
    assert!(
        t.blocker.contains("SOFTMAX"),
        "and name what it really is: {}",
        t.blocker
    );
}

/// Every triaged architecture really does reach the unaudited refusal.
///
/// A verdict on a row that refuses for a *different*, named reason would
/// be dead text the user never sees — the same shape as the `glm4moe`
/// bug, where the reason shown and the reason true were two different
/// strings.
#[test]
fn every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited() {
    for p in architecture_catalog() {
        let Some(t) = p.triage else { continue };
        assert!(
            matches!(p.path, ArchPath::GenericGqa { .. }),
            "`{}` carries a {:?} verdict but resolves to {:?}, which refuses elsewhere",
            p.gguf_name,
            t.class,
            p.path
        );
        assert!(
            !is_audited_generic(p.gguf_name),
            "`{}` is audited and runs; a triage verdict there is never rendered",
            p.gguf_name
        );
    }
}

/// The pending list is a to-do with a shrinking count, not a parking
/// bay.
///
/// If this number goes UP without the total moving, an architecture lost
/// its verdict.
#[test]
fn the_remaining_work_is_counted() {
    assert_eq!(
        TRIAGE_PENDING.len(),
        32,
        "batch 1 triaged 15 of 47; update this number as later batches land"
    );
    let triaged = architecture_catalog()
        .iter()
        .filter(|p| p.triage.is_some())
        .count();
    assert_eq!(triaged + TRIAGE_PENDING.len(), 47);
}

/// A pending architecture's message says it is untriaged and does NOT
/// imply a class.
///
/// "Unaudited" and "untriaged" are different claims and the message has
/// to keep them apart: reading a class into the pending message is how a
/// guess becomes a citation.
#[test]
fn a_pending_architecture_claims_no_class() {
    for arch in ["grok", "dbrx", "apertus", "smallthinker"] {
        let d = unaudited_refusal_detail(arch);
        assert!(d.contains("not done for"), "{arch}: {d}");
        for label in [
            TriageClass::FixtureAway.label(),
            TriageClass::OneMatchArm.label(),
            TriageClass::NewCode.label(),
        ] {
            assert!(
                !d.contains(label),
                "{arch}'s untriaged message must not imply {label}: {d}"
            );
        }
    }
}
