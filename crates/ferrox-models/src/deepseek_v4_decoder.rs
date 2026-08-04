//! A dedicated decoder skeleton for DeepSeek V4's real architecture,
//! separate from `ferrox-models::decoder::Decoder` (the generic GQA path
//! every other preset uses) — analogous to `glm52_decoder.rs` /
//! `kimi_decoder.rs`: composes the already-independently-tested mHC,
//! CSA/HCA compression + attention, derope, grouped output projection,
//! and sqrtsoftplus MoE primitives into one synthetic forward pass.
//!
//! **Synthetic weights only — not a real checkpoint path.** No GGUF
//! loader, no `Engine` wiring, no claim of oracle-correct output against
//! a production DeepSeek V4 file. This module exists to prove the real
//! primitives compose into a finite forward pass end-to-end on tiny dims,
//! the same rigor `glm52_decoder.rs` applies before a real loader lands.
//!
//! Real per-layer flow (simplified to one layer here, transcribed from
//! llama.cpp PR #24162 `deepseek4.cpp`'s layer loop):
//!
//! ```text
//! attn_in  = mHC_pre(hc_streams, attn_hc_pre)
//! attn_out = rms_norm(attn_in) |> HCA/CSA attention |> derope |> grouped wo_a/wo_b
//! hc_streams = mHC_post(attn_out, ...)
//! ffn_in   = mHC_pre(hc_streams, ffn_hc_pre)
//! ffn_out  = rms_norm(ffn_in) |> MoE (sqrtsoftplus routing)
//! hc_streams = mHC_post(ffn_out, ...)
//! hidden   = mHC_head(hc_streams)
//! logits   = output_head(rms_norm(hidden))
//! ```
//!
//! Deliberately **not** implemented in this skeleton (real, cited scope
//! for later slices): incremental DSV4 KV/compressor state
//! (`llama-kv-cache-dsv4.cpp`), CSA's `coff=2` dual-role projection,
//! hash-based first-layer MoE selection, and multi-layer stacking.

use ferrox_core::attention::apply_rope_back;
use ferrox_core::csa_hca_compress::compress_block;
use ferrox_core::deepseek_v4_attention::hca_attention;
use ferrox_core::matmul::rms_norm;
use ferrox_core::tensor::Tensor;
use ferrox_core::weight_matrix::WeightMatrix;
use ferrox_moe::{combine_expert_outputs, route_top_k, run_expert, ExpertWeights, GatingFunction};

use crate::hyper_connections::{
    head as hc_head, post as hc_post, pre as hc_pre, HyperConnectionHeadWeights,
    HyperConnectionPreWeights, HC_MULT,
};
use crate::output_projection::grouped_output_projection;

/// One layer's attention-side weights (synthetic, tiny-dim fixtures only).
pub struct DeepseekV4AttnWeights {
    pub q_proj: WeightMatrix,
    pub k_proj: WeightMatrix,
    pub v_proj: WeightMatrix,
    /// Per-group down-projections (`wo_a`), one per contiguous head block.
    pub group_down: Vec<WeightMatrix>,
    pub wo_b: WeightMatrix,
    /// Block-compression gate (`attn_comp_wgate`) and norm for HCA pooling.
    pub comp_gate: WeightMatrix,
    pub comp_norm: Vec<f32>,
}

/// MoE FFN weights for one layer (`sqrtsoftplus` gating, no hash routing).
pub struct DeepseekV4MoeFfnWeights {
    pub router_weight: WeightMatrix,
    pub experts: Vec<ExpertWeights>,
    pub shared_expert: ExpertWeights,
}

pub struct DeepseekV4DecoderLayerWeights {
    pub attn_hc_pre: HyperConnectionPreWeights,
    pub attn_norm_weight: Vec<f32>,
    pub attn: DeepseekV4AttnWeights,
    pub ffn_hc_pre: HyperConnectionPreWeights,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn: DeepseekV4MoeFfnWeights,
}

pub struct DeepseekV4DecoderWeights {
    pub embedding: Tensor,
    pub layer: DeepseekV4DecoderLayerWeights,
    pub hc_head: HyperConnectionHeadWeights,
    pub final_norm_weight: Vec<f32>,
    pub output_head: WeightMatrix,
}

pub struct DeepseekV4DecoderConfig {
    pub rms_norm_eps: f32,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,
    pub n_heads: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    pub qk_rope: usize,
    pub compress_rope_theta: f32,
    pub n_experts_active: usize,
    pub moe_renormalize: bool,
    /// HCA block length for synthetic compression (real V4 uses 128).
    pub hca_compress_ratio: usize,
}

/// Minimal per-layer state: raw K/V for the SWA window plus optional
/// compressed entries. No incremental DSV4 cache yet — callers append
/// one token at a time and optionally pool when `hca_compress_ratio`
/// raw positions are available.
pub struct DeepseekV4LayerState {
    hc_streams: [Vec<f32>; HC_MULT],
    raw_k: Vec<f32>,
    raw_v: Vec<f32>,
    compressed_k: Vec<f32>,
    compressed_v: Vec<f32>,
    token_count: usize,
}

impl DeepseekV4LayerState {
    pub fn new(hidden_dim: usize) -> Self {
        let zero = vec![0.0; hidden_dim];
        Self {
            hc_streams: std::array::from_fn(|_| zero.clone()),
            raw_k: Vec::new(),
            raw_v: Vec::new(),
            compressed_k: Vec::new(),
            compressed_v: Vec::new(),
            token_count: 0,
        }
    }

    fn reset_hc_from_hidden(&mut self, hidden: &[f32]) {
        for stream in self.hc_streams.iter_mut() {
            stream.copy_from_slice(hidden);
        }
    }
}

pub struct DeepseekV4DecodeState {
    layer: DeepseekV4LayerState,
}

impl DeepseekV4DecodeState {
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            layer: DeepseekV4LayerState::new(hidden_dim),
        }
    }
}

fn derope_attn_out(
    attn_out: &mut [f32],
    n_heads: usize,
    v_head_dim: usize,
    qk_rope: usize,
    pos: usize,
    theta: f32,
) {
    assert!(qk_rope <= v_head_dim);
    for h in 0..n_heads {
        let head_start = h * v_head_dim;
        let rope_start = head_start + v_head_dim - qk_rope;
        apply_rope_back(
            &mut attn_out[rope_start..head_start + v_head_dim],
            pos,
            theta,
        );
    }
}

fn attn_forward_token(
    weights: &DeepseekV4AttnWeights,
    cfg: &DeepseekV4DecoderConfig,
    attn_in: &[f32],
    state: &mut DeepseekV4LayerState,
) -> Vec<f32> {
    let q = weights.q_proj.apply(attn_in);
    let k = weights.k_proj.apply(attn_in);
    let v = weights.v_proj.apply(attn_in);
    debug_assert_eq!(q.len(), cfg.n_heads * cfg.qk_head_dim);
    debug_assert_eq!(k.len(), cfg.n_heads * cfg.qk_head_dim);
    debug_assert_eq!(v.len(), cfg.n_heads * cfg.v_head_dim);

    state.raw_k.extend_from_slice(&k);
    state.raw_v.extend_from_slice(&v);
    state.token_count += 1;
    let n_raw = state.token_count;

    let ratio = cfg.hca_compress_ratio;
    if n_raw >= ratio && n_raw.is_multiple_of(ratio) {
        let per_token_k = cfg.n_heads * cfg.qk_head_dim;
        let per_token_v = cfg.n_heads * cfg.v_head_dim;
        let block_start = (n_raw - ratio) * per_token_k;
        let block_end = n_raw * per_token_k;
        let kv_block: Vec<Vec<f32>> = state.raw_k[block_start..block_end]
            .chunks(cfg.qk_head_dim)
            .map(|row| row.to_vec())
            .collect();
        let score_block: Vec<Vec<f32>> = kv_block
            .iter()
            .map(|row| weights.comp_gate.apply(row))
            .collect();
        let compressed_k = compress_block(
            &kv_block,
            &score_block,
            &weights.comp_norm,
            cfg.rms_norm_eps,
            cfg.qk_rope,
            n_raw / ratio,
            cfg.compress_rope_theta,
        );
        let v_block_start = (n_raw - ratio) * per_token_v;
        let v_block_end = n_raw * per_token_v;
        let v_block: Vec<Vec<f32>> = state.raw_v[v_block_start..v_block_end]
            .chunks(cfg.v_head_dim)
            .map(|row| row.to_vec())
            .collect();
        let compressed_v = compress_block(
            &v_block,
            &score_block,
            &weights.comp_norm,
            cfg.rms_norm_eps,
            cfg.qk_rope,
            n_raw / ratio,
            cfg.compress_rope_theta,
        );
        state.compressed_k.extend_from_slice(&compressed_k);
        state.compressed_v.extend_from_slice(&compressed_v);
    }

    let n_compressed = if cfg.qk_head_dim > 0 {
        state.compressed_k.len() / (cfg.n_heads * cfg.qk_head_dim)
    } else {
        0
    };
    let mut attn_out = hca_attention(
        &q,
        &state.raw_k,
        &state.raw_v,
        n_raw,
        &state.compressed_k,
        &state.compressed_v,
        n_compressed,
        cfg.n_heads,
        cfg.qk_head_dim,
        cfg.v_head_dim,
    );

    derope_attn_out(
        &mut attn_out,
        cfg.n_heads,
        cfg.v_head_dim,
        cfg.qk_rope,
        state.token_count.saturating_sub(1),
        cfg.compress_rope_theta,
    );

    grouped_output_projection(&attn_out, &weights.group_down, &weights.wo_b)
}

fn moe_ffn_forward(
    weights: &DeepseekV4MoeFfnWeights,
    cfg: &DeepseekV4DecoderConfig,
    x: &[f32],
) -> Vec<f32> {
    let router_logits = weights.router_weight.apply(x);
    let decision = route_top_k(
        &router_logits,
        cfg.n_experts_active,
        GatingFunction::SqrtSoftplus,
        cfg.moe_renormalize,
    );
    let routed_outputs: Vec<(Vec<f32>, f32)> = decision
        .expert_ids
        .iter()
        .zip(decision.weights.iter())
        .map(|(&e, &w)| (run_expert(x, &weights.experts[e]), w))
        .collect();
    let shared_out = run_expert(x, &weights.shared_expert);
    combine_expert_outputs(&routed_outputs, &[shared_out], x.len())
}

/// One decode step through the single synthetic layer, then final norm +
/// output projection. `token_id` indexes the embedding table.
pub fn deepseek_v4_forward_token(
    weights: &DeepseekV4DecoderWeights,
    cfg: &DeepseekV4DecoderConfig,
    token_id: usize,
    state: &mut DeepseekV4DecodeState,
) -> Vec<f32> {
    let hidden_dim = weights.embedding.cols();
    let hidden = weights.embedding.row(token_id).to_vec();
    state.layer.reset_hc_from_hidden(&hidden);
    let layer = &weights.layer;

    let hc_residual = state.layer.hc_streams.clone();
    let (attn_in, attn_post, attn_comb) = hc_pre(
        &layer.attn_hc_pre,
        &hc_residual,
        cfg.rms_norm_eps,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
    );
    let attn_normed = rms_norm(&attn_in, &layer.attn_norm_weight, cfg.rms_norm_eps);
    let attn_out = attn_forward_token(&layer.attn, cfg, &attn_normed, &mut state.layer);
    state.layer.hc_streams = hc_post(&attn_out, &hc_residual, &attn_post, &attn_comb);

    let hc_residual = state.layer.hc_streams.clone();
    let (ffn_in, ffn_post, ffn_comb) = hc_pre(
        &layer.ffn_hc_pre,
        &hc_residual,
        cfg.rms_norm_eps,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
    );
    let ffn_normed = rms_norm(&ffn_in, &layer.ffn_norm_weight, cfg.rms_norm_eps);
    let ffn_out = moe_ffn_forward(&layer.ffn, cfg, &ffn_normed);
    state.layer.hc_streams = hc_post(&ffn_out, &hc_residual, &ffn_post, &ffn_comb);

    let collapsed = hc_head(
        &weights.hc_head,
        &state.layer.hc_streams,
        cfg.rms_norm_eps,
        cfg.hc_eps,
    );
    let final_normed = rms_norm(&collapsed, &weights.final_norm_weight, cfg.rms_norm_eps);
    assert_eq!(final_normed.len(), hidden_dim);
    weights.output_head.apply(&final_normed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyper_connections::HyperConnectionPreWeights;

    const HIDDEN_DIM: usize = 8;
    const EPS: f32 = 1e-5;
    const NUM_HEADS: usize = 1;
    const QK_HEAD_DIM: usize = 4;
    const V_HEAD_DIM: usize = 4;
    const QK_ROPE: usize = 2;
    const N_GROUPS: usize = 1;
    const O_LORA_RANK: usize = 2;
    const O_GROUP_DIM: usize = (NUM_HEADS * V_HEAD_DIM) / N_GROUPS;
    const N_EXPERTS: usize = 4;
    const N_EXPERTS_ACTIVE: usize = 2;
    const MOE_FFN_DIM: usize = 3;
    const OUTPUT_VOCAB: usize = 5;
    const HC_FLAT: usize = HC_MULT * HIDDEN_DIM;

    fn wm(data: Vec<f32>, rows: usize, cols: usize) -> WeightMatrix {
        assert_eq!(data.len(), rows * cols);
        WeightMatrix::F32(Tensor::new(data, vec![rows, cols]))
    }

    fn synth(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((seed * 131 + i * 17 + 7) % 23) as f32 * 0.05) - 0.55)
            .collect()
    }

    fn make_hc_pre(seed: usize) -> HyperConnectionPreWeights {
        HyperConnectionPreWeights {
            fn_proj: wm(
                synth(seed, (2 + HC_MULT) * HC_MULT * HC_FLAT),
                (2 + HC_MULT) * HC_MULT,
                HC_FLAT,
            ),
            scale: [0.5, 0.5, 0.5],
            base_pre: [0.1; HC_MULT],
            base_post: [0.2; HC_MULT],
            base_comb: [0.01; HC_MULT * HC_MULT],
        }
    }

    fn make_hc_head(seed: usize) -> HyperConnectionHeadWeights {
        HyperConnectionHeadWeights {
            fn_proj: wm(synth(seed, HC_MULT * HC_FLAT), HC_MULT, HC_FLAT),
            scale: 0.5,
            base: [0.1; HC_MULT],
        }
    }

    fn make_weights() -> DeepseekV4DecoderWeights {
        let expert = |seed: usize| ExpertWeights {
            gate: wm(
                synth(seed, MOE_FFN_DIM * HIDDEN_DIM),
                MOE_FFN_DIM,
                HIDDEN_DIM,
            ),
            up: wm(
                synth(seed + 1, MOE_FFN_DIM * HIDDEN_DIM),
                MOE_FFN_DIM,
                HIDDEN_DIM,
            ),
            down: wm(
                synth(seed + 2, HIDDEN_DIM * MOE_FFN_DIM),
                HIDDEN_DIM,
                MOE_FFN_DIM,
            ),
        };

        DeepseekV4DecoderWeights {
            embedding: Tensor::new(
                synth(1000, OUTPUT_VOCAB * HIDDEN_DIM),
                vec![OUTPUT_VOCAB, HIDDEN_DIM],
            ),
            layer: DeepseekV4DecoderLayerWeights {
                attn_hc_pre: make_hc_pre(100),
                attn_norm_weight: vec![1.0; HIDDEN_DIM],
                attn: DeepseekV4AttnWeights {
                    q_proj: wm(
                        synth(110, NUM_HEADS * QK_HEAD_DIM * HIDDEN_DIM),
                        NUM_HEADS * QK_HEAD_DIM,
                        HIDDEN_DIM,
                    ),
                    k_proj: wm(
                        synth(111, NUM_HEADS * QK_HEAD_DIM * HIDDEN_DIM),
                        NUM_HEADS * QK_HEAD_DIM,
                        HIDDEN_DIM,
                    ),
                    v_proj: wm(
                        synth(112, NUM_HEADS * V_HEAD_DIM * HIDDEN_DIM),
                        NUM_HEADS * V_HEAD_DIM,
                        HIDDEN_DIM,
                    ),
                    group_down: (0..N_GROUPS)
                        .map(|g| {
                            wm(
                                synth(120 + g, O_LORA_RANK * O_GROUP_DIM),
                                O_LORA_RANK,
                                O_GROUP_DIM,
                            )
                        })
                        .collect(),
                    wo_b: wm(
                        synth(130, HIDDEN_DIM * O_LORA_RANK * N_GROUPS),
                        HIDDEN_DIM,
                        O_LORA_RANK * N_GROUPS,
                    ),
                    comp_gate: wm(
                        synth(140, QK_HEAD_DIM * QK_HEAD_DIM),
                        QK_HEAD_DIM,
                        QK_HEAD_DIM,
                    ),
                    comp_norm: vec![1.0; QK_HEAD_DIM],
                },
                ffn_hc_pre: make_hc_pre(200),
                ffn_norm_weight: vec![1.0; HIDDEN_DIM],
                ffn: DeepseekV4MoeFfnWeights {
                    router_weight: wm(synth(300, N_EXPERTS * HIDDEN_DIM), N_EXPERTS, HIDDEN_DIM),
                    experts: (0..N_EXPERTS).map(|e| expert(400 + e * 10)).collect(),
                    shared_expert: expert(900),
                },
            },
            hc_head: make_hc_head(500),
            final_norm_weight: vec![1.0; HIDDEN_DIM],
            output_head: wm(
                synth(1100, OUTPUT_VOCAB * HIDDEN_DIM),
                OUTPUT_VOCAB,
                HIDDEN_DIM,
            ),
        }
    }

    fn decoder_cfg() -> DeepseekV4DecoderConfig {
        DeepseekV4DecoderConfig {
            rms_norm_eps: EPS,
            hc_sinkhorn_iters: 4,
            hc_eps: 1e-6,
            n_heads: NUM_HEADS,
            qk_head_dim: QK_HEAD_DIM,
            v_head_dim: V_HEAD_DIM,
            qk_rope: QK_ROPE,
            compress_rope_theta: 1_000_000.0,
            n_experts_active: N_EXPERTS_ACTIVE,
            moe_renormalize: true,
            hca_compress_ratio: 2,
        }
    }

    #[test]
    fn one_layer_synthetic_forward_produces_finite_logits() {
        let weights = make_weights();
        let cfg = decoder_cfg();
        let mut state = DeepseekV4DecodeState::new(HIDDEN_DIM);

        for token_id in 0..3 {
            let logits =
                deepseek_v4_forward_token(&weights, &cfg, token_id % OUTPUT_VOCAB, &mut state);
            assert_eq!(logits.len(), OUTPUT_VOCAB);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "token {token_id}: logits must be finite, got {logits:?}"
            );
            assert!(
                !logits.iter().any(|v| v.is_nan()),
                "token {token_id}: logits must not contain NaN, got {logits:?}"
            );
        }
    }
}
