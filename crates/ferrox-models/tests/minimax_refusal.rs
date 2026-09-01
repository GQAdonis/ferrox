//! MiniMax (`minimax-m2` / `minimax-m3`) coverage: ferrox is right to
//! refuse, and the reason it used to give was the wrong one.
//!
//! Both architectures refused with a single shared string, "MiniMax
//! 256-expert sigmoid MoE + MTP — see minimax_engine.rs". Checked
//! against llama.cpp, two of those three clauses are false and the
//! remaining one is true of only one of the two models:
//!
//! * **MTP does not exist in a MiniMax GGUF.** Neither
//!   `.scratch/llama.cpp/src/models/minimax-m2.cpp` nor `minimax-m3.cpp`
//!   creates a `nextn.*` tensor. `gguf-py/gguf/constants.py`'s
//!   `MODEL_ARCH.MINIMAXM2` and `.MINIMAXM3` tensor lists carry no
//!   `NEXTN_*` entry, so the writer physically cannot emit one, and
//!   `conversion/minimax.py:102` says the HF-side `mtp` tensors are
//!   *dropped*. `minimax-m3.cpp:9` states it: "MTP is not in released
//!   model weights." This is exactly glm4moe's `q_lora_rank` defect —
//!   a refusal naming something no valid checkpoint can contain, so it
//!   is unreachable by construction rather than merely unlucky.
//! * **Sigmoid MoE routing is not missing.** `loader.rs` reads
//!   `{arch}.expert_gating_func` into `GatingFunction::Sigmoid`, loads
//!   `blk.N.exp_probs_b.bias`, and honours `expert_weights_scale` /
//!   `expert_weights_norm` — the DeepSeek-V3-shaped routing already
//!   validated on `dots1`. `minimax-m2.cpp:131-141` asks for nothing
//!   else. "256 experts" is an hparam, not a ceiling.
//! * **Block-sparse attention is M3's, not MiniMax's.** M2 builds
//!   ordinary dense GQA (`minimax-m2.cpp:112`).
//!
//! So the two split. `minimax-m2` is **unaudited, not unimplemented**:
//! plain GQA, whole-vector QK-norm, partial NEOX RoPE, sigmoid MoE with
//! a router bias — every piece of which the generic path has. What is
//! missing is evidence. `minimax-m3` is genuinely unimplemented, and the
//! blocker is MiniMax Sparse Attention: a per-layer indexer
//! (`minimax-m3.cpp:76-82`) driving its own KV cache
//! (`llama-kv-cache-msa.h`), of which `ferrox_core::block_sparse` is
//! only the selection rule.
//!
//! `tests/fixtures/minimax_m2_tiny.gguf` is the evidence: a 2-layer,
//! 32-wide, 6-expert `minimax-m2` checkpoint from
//! `scripts/make_minimax_fixture.py` (fixed seed, byte-stable), shaped
//! from `minimax-m2.cpp` and `conversion/minimax.py`.
//!
//! Regenerating the fixture:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_minimax_fixture.py \
//!     crates/ferrox-models/tests/fixtures/minimax_m2_tiny.gguf
//! ```
//!
//! NOTE ON SCOPE: nothing here claims minimax-m2 *runs*. Admitting it to
//! `AUDITED_GENERIC_GQA` needs a first-token logit comparison against
//! llama.cpp on this file (`ferrox parity` / `tools/llama_logits.c`), and
//! until that exists the honest state is a refusal that says "unaudited".

use ferrox_gguf::TensorSource;
use ferrox_models::capability::{resolve_architecture, resolve_profile, ArchPath, QkNormStyle};
use ferrox_models::loader::LoadError;
use ferrox_models::ModelConfig;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/minimax_m2_tiny.gguf"
);

fn open() -> ferrox_gguf::GgufFile {
    ferrox_gguf::GgufFile::open(FIXTURE).expect("fixture opens")
}

/// The fixture is a real `minimax-m2` file and carries no MTP anything —
/// otherwise the rest of this suite would be arguing with a strawman.
#[test]
fn a_valid_minimax_checkpoint_carries_no_mtp_tensors() {
    let file = open();
    assert_eq!(
        file.metadata_str("general.architecture"),
        Some("minimax-m2")
    );
    assert!(
        file.metadata("minimax-m2.nextn_predict_layers").is_none(),
        "gguf-py's MINIMAXM2 has no NEXTN entry; the key cannot be written"
    );
    for l in 0..2 {
        for suffix in [
            "nextn.eh_proj",
            "nextn.embed_tokens",
            "nextn.enorm",
            "nextn.hnorm",
            "nextn.shared_head_head",
            "nextn.shared_head_norm",
        ] {
            let name = format!("blk.{l}.{suffix}.weight");
            assert!(
                file.find_tensor(&name).is_none(),
                "{name} must not exist: minimax-m2.cpp creates no MTP tensor"
            );
        }
    }
}

/// Plain GQA with a whole-vector Q/K norm and partial RoPE — the three
/// facts that decide whether the generic path is even the right shape.
///
/// `attn_q_norm` at `n_head * head_dim` is the discriminator: Qwen3 and
/// GLM-4.5 store a `head_dim`-wide per-head vector here, and M3 does too.
/// M2 does not.
#[test]
fn minimax_m2_is_plain_gqa_with_whole_vector_qk_norm_and_partial_rope() {
    let file = open();
    let head_dim = file
        .metadata_u64("minimax-m2.attention.key_length")
        .unwrap() as usize;
    let n_head = file
        .metadata_u64("minimax-m2.attention.head_count")
        .unwrap() as usize;
    let n_head_kv = file
        .metadata_u64("minimax-m2.attention.head_count_kv")
        .unwrap() as usize;
    let rope_dim = file
        .metadata_u64("minimax-m2.rope.dimension_count")
        .unwrap() as usize;

    assert!(
        rope_dim < head_dim,
        "minimax-m2.cpp:51 -- head_dim 128 but n_rot 64; RoPE is partial"
    );

    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
    ] {
        assert!(file.find_tensor(name).is_some(), "{name} must exist");
    }
    // No MLA pair, and no attention bias: create_tensor_qkv marks q/k/v
    // bias NOT_REQUIRED and MiniMax-M2 ships none.
    for name in [
        "blk.0.attn_kv_a_mqa.weight",
        "blk.0.attn_kv_b.weight",
        "blk.0.attn_q.bias",
        "blk.0.attn_k.bias",
        "blk.0.attn_v.bias",
    ] {
        assert!(file.find_tensor(name).is_none(), "{name} must not exist");
    }

    let q_norm = file
        .find_tensor("blk.0.attn_q_norm.weight")
        .expect("attn_q_norm is required (minimax-m2.cpp:30)");
    let k_norm = file
        .find_tensor("blk.0.attn_k_norm.weight")
        .expect("attn_k_norm is required (minimax-m2.cpp:31)");
    assert_eq!(
        q_norm.shape[0] as usize,
        n_head * head_dim,
        "whole-vector Q norm, not per-head: minimax-m2.cpp:30"
    );
    assert_eq!(
        k_norm.shape[0] as usize,
        n_head_kv * head_dim,
        "whole-vector K norm, not per-head: minimax-m2.cpp:31"
    );
    assert_ne!(
        q_norm.shape[0] as usize, head_dim,
        "a head_dim-wide q_norm would make this Qwen3's per-head style, which M2 is not"
    );
}

/// Every layer is MoE, the router bias is present, and the two hparams
/// `minimax-m2.cpp` deliberately does NOT read are absent.
#[test]
fn minimax_m2_moe_is_the_routing_ferrox_already_has() {
    let file = open();
    for key in [
        // conversion/minimax.py writes both only for M3 (:70-71), and
        // minimax-m2.cpp reads neither.
        "minimax-m2.expert_weights_scale",
        "minimax-m2.expert_weights_norm",
        // No leading dense block, no shared expert.
        "minimax-m2.leading_dense_block_count",
        "minimax-m2.expert_shared_count",
    ] {
        assert!(file.metadata(key).is_none(), "{key} must not exist");
    }
    // It DOES carry the gating function, and must: llama.cpp's default
    // is GATING_FUNC_TYPE_NONE, which hits GGML_ABORT in build_moe_ffn
    // (llama-graph.cpp:2019). A runnable M2 file always has this key.
    assert_eq!(
        file.metadata_u64("minimax-m2.expert_gating_func"),
        Some(2),
        "2 == LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID"
    );
    for l in 0..2 {
        for name in [
            format!("blk.{l}.ffn_gate_inp.weight"),
            format!("blk.{l}.exp_probs_b.bias"),
            format!("blk.{l}.ffn_gate_exps.weight"),
            format!("blk.{l}.ffn_up_exps.weight"),
            format!("blk.{l}.ffn_down_exps.weight"),
            // M2 keeps a real ffn_norm, unlike glm4moe.
            format!("blk.{l}.ffn_norm.weight"),
        ] {
            assert!(file.find_tensor(&name).is_some(), "{name} must exist");
        }
        assert!(
            file.find_tensor(&format!("blk.{l}.ffn_gate_shexp.weight"))
                .is_none(),
            "M2 has no shared expert (minimax-m2.cpp:23-40)"
        );
    }
}

/// The refusal a user actually hits, and the whole point of this file:
/// it must not blame MTP, and it must say which of the two problems it
/// is — missing evidence, or missing code.
#[test]
fn minimax_m2_refuses_with_the_reason_that_is_actually_true() {
    let file = open();
    match ModelConfig::from_gguf(&file) {
        Err(LoadError::DedicatedArchitectureRequired(arch, reason)) => {
            assert_eq!(arch, "minimax-m2");
            let lower = reason.to_ascii_lowercase();
            assert!(
                !lower.contains("mtp") && !lower.contains("nextn"),
                "no MiniMax GGUF can carry MTP tensors; the reason must not blame them: {reason}"
            );
            assert!(
                lower.contains("unaudited"),
                "m2 is unaudited, not unimplemented -- the reason must say which: {reason}"
            );
            assert!(
                reason.contains("minimax-m2.cpp"),
                "the reason must cite the llama.cpp graph it was checked against: {reason}"
            );
        }
        other => panic!("minimax-m2 must fail closed with a named reason, got {other:?}"),
    }
}

/// M3's reason names MiniMax Sparse Attention, which is the real
/// blocker, rather than MTP or the MoE.
#[test]
fn minimax_m3_refuses_for_sparse_attention_not_for_mtp() {
    let Some(ArchPath::DedicatedOnly { reason }) = resolve_architecture("minimax-m3") else {
        panic!("minimax-m3 must be DedicatedOnly");
    };
    let lower = reason.to_ascii_lowercase();
    assert!(
        lower.contains("indexer"),
        "the M3 blocker is the MSA indexer (minimax-m3.cpp:76-82): {reason}"
    );
    assert!(
        lower.contains("msa") || lower.contains("sparse attention"),
        "name the attention scheme, not the MoE: {reason}"
    );
    assert!(
        !lower.contains("mtp") && !lower.contains("nextn"),
        "minimax-m3.cpp:9 -- MTP is absent from the released weights, so it cannot be \
         the reason for anything: {reason}"
    );
}

/// The two are different architectures and may not share one reason
/// again. They also differ in QK-norm style, which the shared row got
/// wrong for M3.
#[test]
fn m2_and_m3_are_not_the_same_architecture() {
    let m2 = resolve_architecture("minimax-m2").expect("m2 in catalog");
    let m3 = resolve_architecture("minimax-m3").expect("m3 in catalog");
    let (ArchPath::DedicatedOnly { reason: r2 }, ArchPath::DedicatedOnly { reason: r3 }) = (m2, m3)
    else {
        panic!("both MiniMax rows must be DedicatedOnly");
    };
    assert_ne!(
        r2, r3,
        "M2 is plain GQA and M3 is sparse-attention; one string cannot be true of both"
    );

    // minimax-m2.cpp:30 is `n_embd_head_k * n_head` wide; minimax-m3.cpp:54
    // is `n_embd_head_k`, with llama.cpp's own comment "per-head QK-norm".
    assert_eq!(
        resolve_profile("minimax-m2").map(|p| p.qk_norm),
        Some(QkNormStyle::WholeVector)
    );
    assert_eq!(
        resolve_profile("minimax-m3").map(|p| p.qk_norm),
        Some(QkNormStyle::PerHead)
    );
}

/// Neither may be silently admitted to the generic path. Adding either to
/// `AUDITED_GENERIC_GQA` without a logit comparison converts an honest
/// refusal into a silent wrong answer.
#[test]
fn neither_minimax_is_on_the_generic_path() {
    for arch in ["minimax-m2", "minimax-m3"] {
        assert!(
            matches!(
                resolve_architecture(arch),
                Some(ArchPath::DedicatedOnly { .. })
            ),
            "{arch} must fail closed"
        );
        assert!(
            !ferrox_models::capability::is_audited_generic(arch),
            "{arch} has no parity evidence; it may not be listed as audited"
        );
    }
}
