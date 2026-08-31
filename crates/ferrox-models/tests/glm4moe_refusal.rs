//! GLM-4.5-MoE (`glm4moe`) coverage: what ferrox refuses, and why the
//! reason it used to give was the wrong one.
//!
//! GLM-4.5, GLM-4.5-Air and GLM-4.6 all tag `glm4moe`, so this is a
//! model people actually try to run. ferrox refuses it — correctly, it
//! has no graph for it — but the refusal said "use
//! `ferrox_models::glm52_decoder` / `glm52_gguf_loader`", and
//! `ferrox-cli`'s `run.rs` acts on that: `glm4moe` is dispatched
//! straight into `run_glm52_infer`. That loader's
//! `read_glm52_hparams` requires four MLA hyper-parameters —
//! `{arch}.attention.q_lora_rank`, `.kv_lora_rank`, `.qk_nope_head_dim`
//! and `.qk_rope_head_dim` — and **GLM-4.5 is not an MLA model**:
//! `.scratch/llama.cpp/src/models/glm4-moe.cpp`'s `load_arch_hparams`
//! reads none of the four, and its `load_arch_tensors` calls
//! `create_tensor_qkv` (plain Q/K/V) with no `attn_kv_a_mqa` /
//! `attn_kv_b` / `attn_q_a` / `attn_q_b` anywhere in the file. So a real
//! GLM-4.5-Air download failed with "missing hparam
//! glm4moe.attention.q_lora_rank", which is a true statement about a key
//! the architecture is not supposed to have.
//!
//! `tests/fixtures/glm4moe_tiny.gguf` is the evidence. It is a 2-layer
//! (one leading dense, one MoE), 32-wide, 6-expert `glm4moe` checkpoint
//! from `scripts/make_glm4moe_fixture.py` (fixed seed, byte-stable), and
//! **llama.cpp itself loads and decodes it** — checked with
//! `scripts/gptoss_reference_logits.cpp` against a real `libllama`,
//! which prints `n_expert = 6`, `rope type = 2` and produces logits. A
//! valid glm4moe checkpoint therefore carries none of the four MLA keys,
//! which is what makes the old refusal unreachable-by-construction
//! rather than merely unlucky.
//!
//! The real gap is one norm slot. `glm4-moe.cpp:75` creates
//! `blk.N.post_attention_norm.weight` and **no** `blk.N.ffn_norm.weight`,
//! and applies it to `ffn_inp` at :215 — after the attention residual,
//! i.e. as the pre-FFN norm. ferrox's generic decoder requires
//! `ffn_norm` and applies `post_attention_norm` in Gemma's other slot
//! (on the attention branch, before the residual add). That is gpt-oss's
//! slot exactly, and `loader.rs` already implements it behind an
//! `is_gpt_oss` flag.
//!
//! Regenerating the fixture:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_glm4moe_fixture.py \
//!     crates/ferrox-models/tests/fixtures/glm4moe_tiny.gguf
//! ```

use ferrox_gguf::TensorSource;
use ferrox_models::loader::LoadError;
use ferrox_models::ModelConfig;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/glm4moe_tiny.gguf"
);

fn open() -> ferrox_gguf::GgufFile {
    ferrox_gguf::GgufFile::open(FIXTURE).expect("fixture opens")
}

/// The fixture really is a `glm4moe` file, and really does lack every
/// MLA key — otherwise the rest of this suite would be testing a
/// strawman rather than the architecture.
#[test]
fn a_valid_glm4moe_checkpoint_carries_no_mla_hparams() {
    let file = open();
    assert_eq!(file.metadata_str("general.architecture"), Some("glm4moe"));
    for key in [
        "glm4moe.attention.q_lora_rank",
        "glm4moe.attention.kv_lora_rank",
        "glm4moe.attention.qk_nope_head_dim",
        "glm4moe.attention.qk_rope_head_dim",
    ] {
        assert!(
            file.metadata(key).is_none(),
            "{key} must not exist: glm4-moe.cpp's load_arch_hparams never reads it"
        );
    }
    // Plain GQA projections, not MLA's compressed pair.
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
    ] {
        assert!(file.find_tensor(name).is_some(), "{name} must exist");
    }
    for name in [
        "blk.0.attn_kv_a_mqa.weight",
        "blk.0.attn_kv_b.weight",
        "blk.0.attn_q_a.weight",
        "blk.0.attn_q_b.weight",
    ] {
        assert!(file.find_tensor(name).is_none(), "{name} must not exist");
    }
}

/// The divergence that actually keeps glm4moe off the generic path: the
/// pre-FFN norm is spelled `post_attention_norm`, and there is no
/// `ffn_norm` at all.
#[test]
fn glm4moe_stores_its_pre_ffn_norm_under_post_attention_norm() {
    let file = open();
    for l in 0..2 {
        assert!(
            file.find_tensor(&format!("blk.{l}.post_attention_norm.weight"))
                .is_some(),
            "glm4-moe.cpp:75 makes this REQUIRED"
        );
        assert!(
            file.find_tensor(&format!("blk.{l}.ffn_norm.weight"))
                .is_none(),
            "glm4-moe.cpp creates no ffn_norm; the generic decoder demands one"
        );
    }
}

/// The loader the old refusal pointed at cannot read this file, and
/// fails on a key the architecture is not supposed to carry.
///
/// This is the whole finding in one assertion: a refusal that names the
/// wrong destination is worse than no refusal, because `ferrox-cli`
/// dispatches on it.
#[test]
fn the_glm52_mla_loader_cannot_read_a_glm4moe_checkpoint() {
    let file = open();
    match ferrox_models::glm52_gguf_loader::read_glm52_hparams(&file) {
        Err(LoadError::MissingHparam(key)) => {
            assert!(
                key.contains("lora_rank") || key.contains("head_dim"),
                "expected a missing MLA hparam, got `{key}`"
            );
        }
        Err(other) => panic!("expected a missing MLA hparam, got {other:?}"),
        Ok(_) => panic!(
            "read_glm52_hparams accepted a checkpoint with no MLA keys; \
             if that is now legitimate, this test and the capability \
             reason for `glm4moe` both need rewriting"
        ),
    }
}

/// The generic path refuses, and the reason names the norm slot rather
/// than sending the reader to the MLA loader.
#[test]
fn glm4moe_refuses_with_the_reason_that_is_actually_true() {
    let file = open();
    match ModelConfig::from_gguf(&file) {
        Err(LoadError::DedicatedArchitectureRequired(arch, reason)) => {
            assert_eq!(arch, "glm4moe");
            assert!(
                reason.contains("post_attention_norm") && reason.contains("ffn_norm"),
                "the reason must name the norm slot, got: {reason}"
            );
            assert!(
                reason.contains("NOT MLA"),
                "the reason must say it is not MLA, so nobody sends it back to \
                 glm52_gguf_loader: {reason}"
            );
        }
        other => panic!("glm4moe must fail closed with a named reason, got {other:?}"),
    }
}

/// The refusal is not an accident of an unaudited-architecture gate: it
/// stands even with `FERROX_ALLOW_UNAUDITED_ARCH` set, because
/// `DedicatedOnly` is checked first and is not overridable.
#[test]
fn glm4moe_is_not_on_the_generic_path_at_all() {
    use ferrox_models::capability::{resolve_architecture, ArchPath};
    assert!(matches!(
        resolve_architecture("glm4moe"),
        Some(ArchPath::DedicatedOnly { .. })
    ));
    assert!(!ferrox_models::capability::is_audited_generic("glm4moe"));
}
