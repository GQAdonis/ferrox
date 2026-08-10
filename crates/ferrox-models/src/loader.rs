//! Loads a real `Decoder` from an on-disk GGUF file, using the
//! llama.cpp-style tensor naming convention
//! (`token_embd.weight`, `blk.N.attn_q.weight`, `blk.N.ffn_gate.weight`
//! or, for MoE, `blk.N.ffn_gate_exps.weight`, `output_norm.weight`,
//! `output.weight`). Until this module existed, ferrox could only run
//! correctly-shaped *random* weights.
//!
//! Quantized tensors (Q8_0 / Q4_0) are loaded as `WeightMatrix::Quantized`
//! backed by `WeightBytes::Mapped` -- a zero-copy view into the same
//! mmap `GgufFile` already holds, with no intermediate heap copy of the
//! tensor's bytes at all. So a checkpoint's resident memory is the
//! mmap page cache, not the mmap plus a second in-process copy of every
//! weight. `WeightMatrix::apply` dispatches to ferrox-quant's fused
//! dequant+dot kernels directly against those mapped bytes at inference
//! time. F32 tensors (norms, embeddings, and any weight not natively
//! quantized) still copy into an owned `Tensor`, since they're small
//! relative to the quantized weight matrices and need per-element
//! access patterns a raw byte view doesn't support as cleanly.
//!
//! Verified end to end (see `crates/ferrox-models/tests/gguf_roundtrip.rs`)
//! against a genuinely Q8_0-quantized, generated on-disk GGUF fixture
//! for the dense (single-expert) case, and against real OLMoE / Qwen2-MoE
//! checkpoints for the multi-expert 3D-packed-tensor path.

use ferrox_core::expert_store::{ExpertKey, ExpertSource, ExpertStore};
use ferrox_core::tensor::Tensor;
use ferrox_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
use ferrox_gguf::{GgmlType, GgufError, GgufValue, ShardedGguf, TensorInfo, TensorSource};
use ferrox_moe::{ExpertWeights, GatingFunction, MoeLayerConfig};
use std::sync::Arc;
use thiserror::Error;

use crate::config::ModelConfig;
use crate::decoder::{AttnWeights, Decoder, ExpertBacking, LayerWeights, MoeWeights};
#[cfg(feature = "metal")]
use crate::decoder::MoePackedQ4Planes;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Shard(#[from] ferrox_gguf::ShardError),
    #[error("tensor '{0}' has unsupported dtype {1:?}")]
    UnsupportedDtype(String, GgmlType),
    #[error(
        "MoE tensor '{0}' is not 3D or its expert count {1} does not match config n_experts {2}"
    )]
    ExpertCountMismatch(String, usize, usize),
    #[error("GGUF file is missing required hparam metadata key '{0}'")]
    MissingHparam(String),
    /// `general.architecture` is not in the capability registry — refuse
    /// to guess RoPE/gating rather than emit fluent-but-wrong logits.
    #[error(
        "unsupported GGUF architecture '{0}': not in ferrox's capability registry \
         (unknown required features fail closed; see ferrox_models::capability)"
    )]
    UnsupportedArchitecture(String),
    /// Architecture exists but must not use the generic GQA decoder.
    #[error("architecture '{0}' cannot use the generic Decoder: {1}")]
    DedicatedArchitectureRequired(String, &'static str),
    /// Metadata advertises a feature the generic decoder does not implement.
    #[error("architecture '{0}' requires unimplemented feature: {1}")]
    UnsupportedFeature(String, String),
}

/// Architecture-family name strings (GGUF's `general.architecture` value)
/// known, from reading ik_llama.cpp's `llama-hparams.cpp`
/// (`LLM_ARCH_DEEPSEEK2`, `LLM_ARCH_GLM4_MOE` cases), to default to
/// sigmoid MoE gating with post-selection renormalization rather than
/// softmax. See docs/MODELS.md for the citations behind this list.
const SIGMOID_GATING_ARCHITECTURES: &[&str] = &["deepseek2", "glm4moe"];

/// Architecture-family names whose real reference implementation skips
/// renormalizing top-k softmax routing weights after selection (GGUF
/// carries no metadata key for this -- it's hardcoded per-architecture in
/// both the real HF `transformers` model code and llama.cpp's
/// `build_moe_ffn` call sites, not read from the file). Confirmed for
/// `olmoe` against `OlmoeTopKRouter.forward` in
/// `transformers/models/olmoe/modeling_olmoe.py` (`config.norm_topk_prob`
/// is `false` in the real published config.json) and llama.cpp's
/// `src/models/olmoe.cpp` (`build_moe_ffn(..., false, ...,
/// LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, ...)`). See
/// `MoeLayerConfig::norm_topk_prob`'s doc comment for why this matters:
/// getting it wrong silently produces wrong generation output even
/// though the file loads and shape-validates fine.
// Architectures whose reference graphs pass `norm_w=false` to
// `build_moe_ffn` (llama.cpp) / `norm_topk_prob=false` in HF config.
// Qwen2-MoE: `.scratch/llama.cpp/src/models/qwen2moe.cpp` — Softmax +
// `false` for the norm_topk slot. Renormalizing top-k weights made
// Qwen1.5-MoE greedy decode emit garbage despite shared-expert load.
const NO_TOPK_RENORMALIZE_ARCHITECTURES: &[&str] = &["olmoe", "qwen2moe"];

fn metadata_u64_any(file: &impl TensorSource, keys: &[String]) -> Option<u64> {
    keys.iter().find_map(|k| file.metadata_u64(k))
}

fn metadata_f32_any(file: &impl TensorSource, keys: &[String]) -> Option<f32> {
    keys.iter()
        .find_map(|k| file.metadata(k).and_then(GgufValue::as_f32))
}

impl ModelConfig {
    /// Derives a `ModelConfig` from a real GGUF file's own hyperparameter
    /// metadata, following llama.cpp's `general.architecture`-prefixed key
    /// convention (`{arch}.block_count`, `{arch}.embedding_length`,
    /// `{arch}.attention.head_count`, `{arch}.expert_count`, ...) rather
    /// than requiring a hand-written preset to already match the file's
    /// shape exactly. This is what lets `ferrox-server` (and `ferrox
    /// run-real`) load an arbitrary checkpoint, not just the three
    /// hand-tuned presets in `config.rs`.
    ///
    /// Fields with no corresponding metadata key fall back to widely-used
    /// llama.cpp defaults (documented inline) and are listed in the
    /// returned config's `best_effort_fields`, following the same
    /// confirmed-vs-estimated discipline as the hand-written presets.
    pub fn from_gguf(file: &impl TensorSource) -> Result<Self, LoadError> {
        let arch = file
            .metadata_str("general.architecture")
            .ok_or_else(|| LoadError::MissingHparam("general.architecture".to_string()))?
            .to_string();
        let arch_profile = crate::capability::resolve_profile(&arch)
            .ok_or_else(|| LoadError::UnsupportedArchitecture(arch.clone()))?;
        let rope_layout = match arch_profile.path {
            crate::capability::ArchPath::GenericGqa { rope }
            | crate::capability::ArchPath::TestFixture { rope } => rope,
            crate::capability::ArchPath::DedicatedOnly { reason } => {
                return Err(LoadError::DedicatedArchitectureRequired(
                    arch.clone(),
                    reason,
                ));
            }
            crate::capability::ArchPath::Deferred { reason } => {
                return Err(LoadError::UnsupportedFeature(
                    arch.clone(),
                    format!("architecture deferred from Ferrox text-generation scope: {reason}"),
                ));
            }
        };
        let qk_norm_style = arch_profile.qk_norm;
        for (meta_key, feature) in crate::capability::unsupported_feature_keys(&arch) {
            if let Some(v) = metadata_f32_any(file, std::slice::from_ref(&meta_key)) {
                if v > 0.0 {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!("{feature} (metadata {meta_key}={v})"),
                    ));
                }
            }
            if let Some(v) = metadata_u64_any(file, std::slice::from_ref(&meta_key)) {
                if v > 0 {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        feature.to_string(),
                    ));
                }
            }
        }
        let key = |suffix: &str| format!("{arch}.{suffix}");

        let name: &'static str = Box::leak(
            file.metadata_str("general.name")
                .unwrap_or(&arch)
                .to_string()
                .into_boxed_str(),
        );

        let n_layers =
            file.metadata_u64(&key("block_count"))
                .ok_or_else(|| LoadError::MissingHparam(key("block_count")))? as usize;
        let hidden_dim = file
            .metadata_u64(&key("embedding_length"))
            .ok_or_else(|| LoadError::MissingHparam(key("embedding_length")))?
            as usize;
        let n_heads = file
            .metadata_u64(&key("attention.head_count"))
            .ok_or_else(|| LoadError::MissingHparam(key("attention.head_count")))?
            as usize;

        let mut best_effort_fields: Vec<&'static str> = Vec::new();

        let n_kv_heads = file
            .metadata_u64(&key("attention.head_count_kv"))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                best_effort_fields.push("n_kv_heads (no attention.head_count_kv key; assumed equal to n_heads, i.e. plain MHA)");
                n_heads
            });
        let head_dim = file
            .metadata_u64(&key("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                best_effort_fields.push(
                    "head_dim (no attention.key_length key; derived as hidden_dim / n_heads)",
                );
                hidden_dim / n_heads
            });
        let v_head_dim = file
            .metadata_u64(&key("attention.value_length"))
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        if v_head_dim != head_dim {
            return Err(LoadError::UnsupportedFeature(
                arch.clone(),
                format!(
                    "split K/V head dims (key_length={head_dim}, value_length={v_head_dim}); \
                     generic decoder requires equal head dims"
                ),
            ));
        }
        let vocab_size = file
            .metadata("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GgufValue::Array(items) => Some(items.len()),
                _ => None,
            })
            .or_else(|| file.metadata_u64(&key("vocab_size")).map(|v| v as usize))
            .unwrap_or_else(|| {
                best_effort_fields.push("vocab_size (no tokenizer.ggml.tokens array or {arch}.vocab_size key; fell back to output.weight's own row count)");
                // `output.weight`'s real raw shape is `[hidden_dim,
                // vocab_size]` (ggml's fastest-first `ne[]` order --
                // see `load_weight_matrix`'s doc comment), so vocab_size
                // is the *last* element, not the first.
                file.find_tensor("output.weight")
                    .and_then(|t| t.shape.last().copied())
                    .unwrap_or(0) as usize
            });
        let rope_theta = metadata_f32_any(file, &[key("rope.freq_base")]).unwrap_or_else(|| {
            best_effort_fields.push("rope_theta (no rope.freq_base key; defaulted to 10000.0)");
            10000.0
        });
        let rms_norm_eps = metadata_f32_any(
            file,
            &[
                key("attention.layer_norm_rms_epsilon"),
                key("attention.layer_norm_epsilon"),
            ],
        )
        .unwrap_or_else(|| {
            best_effort_fields
                .push("rms_norm_eps (no layer_norm_rms_epsilon key; defaulted to 1e-5)");
            1e-5
        });

        let n_experts = metadata_u64_any(file, &[key("expert_count")]).unwrap_or(0) as usize;
        let is_moe = n_experts > 1;

        let n_experts_active = if is_moe {
            metadata_u64_any(file, &[key("expert_used_count")]).unwrap_or_else(|| {
                best_effort_fields
                    .push("moe.n_experts_active (no expert_used_count key; defaulted to 2)");
                2
            }) as usize
        } else {
            1
        };
        // Prefer the GGUF hparam when present. Qwen2MoE (and some other
        // HF→GGUF exports) omit `expert_shared_count` but still ship
        // `blk.N.ffn_{gate,up,down}_shexp.weight` — without a tensor-
        // presence fallback those weights are silently dropped and the
        // model runs with a large chunk of active FFN missing.
        let n_shared_experts = match metadata_u64_any(file, &[key("expert_shared_count")]) {
            Some(n) => n as usize,
            None if is_moe && file.find_tensor("blk.0.ffn_gate_shexp.weight").is_some() => {
                best_effort_fields.push(
                    "moe.n_shared_experts (no expert_shared_count; inferred 1 from blk.0.ffn_gate_shexp.weight)",
                );
                1
            }
            None => 0,
        };
        // MoE GGUFs often only set `feed_forward_length` (OLMoE=1024,
        // Qwen2-MoE=5632 for the shared expert). `expert_feed_forward_length`
        // is optional. llama.cpp `qwen2moe.cpp` uses
        // `n_ff_exp = n_ff_exp ? n_ff_exp : n_ff / n_expert_used` (1408 for
        // Qwen1.5-MoE); the shared expert keeps the full `n_ff` (5632).
        let feed_forward_length =
            metadata_u64_any(file, &[key("feed_forward_length")]);
        let expert_ffn_dim = metadata_u64_any(file, &[key("expert_feed_forward_length")])
            .or_else(|| {
                feed_forward_length.and_then(|ff| {
                    if is_moe && n_experts_active > 0 {
                        Some(ff / n_experts_active as u64)
                    } else {
                        Some(ff)
                    }
                })
            })
            .unwrap_or_else(|| {
                best_effort_fields.push(
                    "moe.expert_ffn_dim (no expert_feed_forward_length/feed_forward_length; defaulted to 4x hidden_dim)",
                );
                (hidden_dim * 4) as u64
            }) as usize;
        let n_dense_leading_layers =
            metadata_u64_any(file, &[key("leading_dense_block_count")]).unwrap_or(0) as usize;

        // ik_llama.cpp's real gating-function hparam
        // (LLM_KV_EXPERT_GATING_FUNC: 1=softmax, 2=sigmoid) if the file
        // carries it; otherwise fall back to the same architecture-name
        // convention the hand-written presets in config.rs use (see
        // docs/MODELS.md for the citations behind that list).
        let gating = match metadata_u64_any(file, &[key("expert_gating_func")]) {
            Some(2) => GatingFunction::Sigmoid,
            Some(1) => GatingFunction::Softmax,
            _ => {
                if SIGMOID_GATING_ARCHITECTURES.contains(&arch.as_str()) {
                    GatingFunction::Sigmoid
                } else {
                    if is_moe {
                        best_effort_fields.push(
                            "moe.gating (no expert_gating_func key and architecture not in the known-sigmoid list; defaulted to softmax)",
                        );
                    }
                    GatingFunction::Softmax
                }
            }
        };

        // See `NO_TOPK_RENORMALIZE_ARCHITECTURES`'s doc comment: no GGUF
        // metadata key exists for this, so it's an architecture-name
        // lookup, the same convention `gating`'s fallback above uses.
        let norm_topk_prob = !NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&arch.as_str());
        if is_moe && matches!(gating, GatingFunction::Softmax) {
            best_effort_fields.push(
                "moe.norm_topk_prob (no GGUF metadata key exists for this; defaulted by architecture-name lookup against NO_TOPK_RENORMALIZE_ARCHITECTURES)",
            );
        }

        // Real GGUF key (`{arch}.attention.sliding_window`, confirmed
        // against `gguf-py/gguf/constants.py`'s real
        // `LLM_KV_ATTENTION_SLIDING_WINDOW`). Some checkpoints
        // (confirmed for real published Qwen1.5-MoE/Qwen2-MoE GGUFs)
        // carry a nonzero window value even when the model's own
        // config disables sliding-window attention entirely
        // (`use_sliding_window: false`) -- llama.cpp's own convention
        // is that a window of 0 means "unused," so only a real nonzero
        // value here is treated as active.
        let sliding_window = metadata_u64_any(file, &[key("attention.sliding_window")])
            .map(|v| v as usize)
            .filter(|&w| w > 0);

        // Gemma alternating SWA period (`attention.sliding_window_pattern`).
        // llama.cpp: gemma2 defaults period=2, gemma3 defaults period=6 when
        // the pattern key is absent. A missing key must NOT mean "all SWA".
        let swa_pattern = metadata_u64_any(file, &[key("attention.sliding_window_pattern")])
            .map(|v| v as usize)
            .filter(|&p| p > 1)
            .or_else(|| {
                if sliding_window.is_none() {
                    return None;
                }
                match arch_profile.family {
                    crate::capability::DecoderFamily::GemmaFamily => {
                        // Prefer architecture string: gemma2 → 2, else gemma3+ → 6.
                        let arch = file
                            .metadata_str("general.architecture")
                            .unwrap_or("")
                            .to_ascii_lowercase();
                        if arch == "gemma2" || arch.starts_with("gemma2") {
                            Some(2)
                        } else {
                            Some(6)
                        }
                    }
                    _ => None,
                }
            });

        let attn_logit_softcap = metadata_f32_any(
            file,
            &[
                key("attention.logit_softcapping"),
                key("attn_logit_softcapping"),
            ],
        )
        .filter(|&v| v > 0.0);
        let final_logit_softcap =
            metadata_f32_any(file, &[key("final_logit_softcapping")]).filter(|&v| v > 0.0);

        // Gemma: embeddings are scaled by sqrt(hidden_dim) at input.
        let embedding_scale = if matches!(
            arch_profile.family,
            crate::capability::DecoderFamily::GemmaFamily
        ) {
            Some((hidden_dim as f32).sqrt())
        } else {
            None
        };

        // Gemma's f_attention_scale equals 1/sqrt(n_embd_head_k) for non-27B,
        // which is already what `causal_gqa_attention` applies. Do not also
        // pre-scale Q (that double-scales scores vs llama.cpp's
        // `build_attn(..., 1.0f)` after an explicit Q scale).
        let attention_scale = None;

        // SWA-layer RoPE base (llama.cpp default 10000 when key absent).
        let rope_theta_swa = if sliding_window.is_some() {
            Some(
                metadata_f32_any(
                    file,
                    &[key("rope.freq_base_swa"), key("rope_freq_base_swa")],
                )
                .unwrap_or(10_000.0),
            )
        } else {
            None
        };

        let ffn_activation = match arch_profile.family {
            crate::capability::DecoderFamily::GemmaFamily => crate::config::FfnActivation::Gelu,
            crate::capability::DecoderFamily::PhiFamily => {
                crate::config::FfnActivation::SwigluFused
            }
            _ => crate::config::FfnActivation::Swiglu,
        };

        // Llama 3/3.1/3.2's real per-band RoPE frequency correction: one
        // model-level tensor (`TENSOR_NOT_REQUIRED`, `TENSOR_DUPLICATED`
        // for every layer but the first in the real llama.cpp source --
        // i.e. every layer shares this same array), not per-layer. See
        // `ferrox_core::attention::apply_rope_with_freq_factors`'s doc
        // comment for why this matters.
        let rope_freqs = load_f32_vec_optional(file, "rope_freqs.weight")?;

        // RoPE layout comes from the capability registry above (fail-
        // closed). Getting this wrong for `llama` (needs Norm) was the
        // real root cause of the Llama-3.1-8B early-stop/wrong-logits bug.

        if best_effort_fields.is_empty() {
            best_effort_fields.push(
                "none -- every field above was read directly from this file's own GGUF metadata",
            );
        }

        Ok(ModelConfig {
            name,
            n_layers,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            rope_theta,
            rms_norm_eps,
            // No GGUF file encodes a hybrid KDA/Gated-MLA attention
            // topology today; every real checkpoint loaded this way
            // runs the standard Gqa path.
            attention: crate::config::AttentionKind::Gqa,
            sliding_window,
            swa_pattern,
            moe: MoeLayerConfig {
                n_experts: n_experts.max(1),
                n_experts_active,
                n_shared_experts,
                hidden_dim,
                expert_ffn_dim,
                gating,
                norm_topk_prob,
                expert_group_count: metadata_u64_any(file, &[key("expert_group_count")])
                    .map(|v| v as usize)
                    .filter(|&c| c > 1),
                expert_group_used_count: metadata_u64_any(file, &[key("expert_group_used_count")])
                    .map(|v| v as usize)
                    .filter(|&c| c > 0),
            },
            n_dense_leading_layers,
            rope_freqs,
            rope_layout,
            qk_norm_style,
            attn_logit_softcap,
            final_logit_softcap,
            embedding_scale,
            attention_scale,
            rope_theta_swa,
            ffn_activation,
            best_effort_fields: Box::leak(best_effort_fields.into_boxed_slice()),
        })
    }
}

fn find_info<'a>(file: &'a impl TensorSource, name: &str) -> Result<&'a TensorInfo, LoadError> {
    file.find_tensor(name)
        .ok_or_else(|| LoadError::Gguf(GgufError::TensorNotFound(name.to_string())))
}

/// Maps a GGUF tensor's on-disk dtype to the `QuantKind` `WeightMatrix`
/// uses to pick a fused dequant+dot kernel, or `None` for dtypes with
/// no quantized kernel (F32, or a dtype not yet implemented at all).
fn quant_kind_for(dtype: GgmlType) -> Option<QuantKind> {
    match dtype {
        GgmlType::Q8_0 => Some(QuantKind::Q8_0),
        GgmlType::Q4_0 => Some(QuantKind::Q4_0),
        GgmlType::Q4K => Some(QuantKind::Q4K),
        GgmlType::Q5K => Some(QuantKind::Q5K),
        GgmlType::Q6K => Some(QuantKind::Q6K),
        GgmlType::Q2K => Some(QuantKind::Q2K),
        GgmlType::Q3K => Some(QuantKind::Q3K),
        GgmlType::Q4_1 => Some(QuantKind::Q4_1),
        GgmlType::Q5_0 => Some(QuantKind::Q5_0),
        GgmlType::Q5_1 => Some(QuantKind::Q5_1),
        GgmlType::Q8_1 => Some(QuantKind::Q8_1),
        GgmlType::IQ4NL => Some(QuantKind::IQ4NL),
        GgmlType::IQ4XS => Some(QuantKind::IQ4XS),
        GgmlType::IQ1S => Some(QuantKind::IQ1S),
        GgmlType::IQ2XXS => Some(QuantKind::IQ2XXS),
        GgmlType::IQ3XXS => Some(QuantKind::IQ3XXS),
        GgmlType::MXFP4 => Some(QuantKind::Mxfp4Gguf),
        _ => None,
    }
}

/// Like `load_f32_vec`, but for tensors that only exist on some
/// checkpoints (e.g. `attn_q_norm`/`attn_k_norm` -- OLMoE-style
/// per-projection QK-RMSNorm applied to the full q_proj/k_proj output
/// before RoPE, confirmed against `OlmoeAttention.forward` in
/// `transformers/models/olmoe/modeling_olmoe.py`: `q_norm(q_proj(x))`,
/// `k_norm(k_proj(x))`, both plain RMSNorm over the whole projected
/// width, not per-head). Absent for every other preset/fixture this
/// loader already handles -- `None` there is correct, not a missing
/// feature.
fn load_f32_vec_optional(
    file: &impl TensorSource,
    name: &str,
) -> Result<Option<Vec<f32>>, LoadError> {
    if file.find_tensor(name).is_none() {
        return Ok(None);
    }
    Ok(Some(load_f32_vec(file, name)?))
}

/// Slice `n` rows starting at `start` out of a quantized matrix without
/// dequantizing: every `Quantized` kind stores one interleaved block
/// buffer per row (fixed `row_bytes`), so a row range is a contiguous
/// byte range. Mapped sources stay zero-copy (sub-range of the same
/// mmap); other backings get an owned copy. Returns `None` for non-
/// quantized matrices (F32 / MXFP4) — callers fall back to dequant.
fn slice_quantized_rows(m: &WeightMatrix, start: usize, n: usize) -> Option<WeightMatrix> {
    let WeightMatrix::Quantized {
        data,
        rows,
        cols,
        kind,
    } = m
    else {
        return None;
    };
    let total = data.len();
    if *rows == 0 || total % *rows != 0 || start + n > *rows {
        return None;
    }
    let row_bytes = total / *rows;
    let (b0, b1) = (start * row_bytes, (start + n) * row_bytes);
    let bytes = match data {
        WeightBytes::Mapped { mmap, range } => WeightBytes::Mapped {
            mmap: mmap.clone(),
            range: range.start + b0..range.start + b1,
        },
        other => WeightBytes::Owned(other.as_slice()[b0..b1].to_vec()),
    };
    Some(WeightMatrix::Quantized {
        data: bytes,
        rows: n,
        cols: *cols,
        kind: *kind,
    })
}

/// Loads Q/K/V projections: prefers split `attn_{q,k,v}.weight`, falls
/// back to fused `attn_qkv.weight` (Phi-3 / some Qwen GGUFs) by
/// slicing quantized rows (zero-copy for mmapped GGUFs; dequant only
/// for non-quantized storage). Mirrors llama.cpp `create_tensor_qkv`.
fn load_qkv_projections(
    file: &impl TensorSource,
    layer: usize,
    config: &ModelConfig,
) -> Result<(WeightMatrix, WeightMatrix, WeightMatrix), LoadError> {
    let q_name = format!("blk.{layer}.attn_q.weight");
    let k_name = format!("blk.{layer}.attn_k.weight");
    let v_name = format!("blk.{layer}.attn_v.weight");
    let fused_name = format!("blk.{layer}.attn_qkv.weight");

    if file.find_tensor(&q_name).is_some() {
        return Ok((
            load_weight_matrix(file, &q_name)?,
            load_weight_matrix(file, &k_name)?,
            load_weight_matrix(file, &v_name)?,
        ));
    }
    if file.find_tensor(&fused_name).is_none() {
        return Err(LoadError::Gguf(GgufError::TensorNotFound(q_name)));
    }

    let fused = load_weight_matrix(file, &fused_name)?;
    let q_rows = config.n_heads * config.head_dim;
    let kv_rows = config.n_kv_heads * config.head_dim;
    let expected = q_rows + 2 * kv_rows;
    if fused.rows() != expected {
        // Phi-3 sometimes stores Q as full n_embd (== q_rows when MHA).
        return Err(LoadError::UnsupportedFeature(
            config.name.to_string(),
            format!(
                "{fused_name} has {} rows; expected q+k+v = {} \
                 (n_heads*head_dim + 2*n_kv_heads*head_dim)",
                fused.rows(),
                expected
            ),
        ));
    }
    let cols = fused.cols();
    // Quantized fused tensor: split by row ranges without dequantizing,
    // keeping Q/K/V on the quantized (Metal-capable) matvec path.
    if let (Some(q), Some(k), Some(v)) = (
        slice_quantized_rows(&fused, 0, q_rows),
        slice_quantized_rows(&fused, q_rows, kv_rows),
        slice_quantized_rows(&fused, q_rows + kv_rows, kv_rows),
    ) {
        return Ok((q, k, v));
    }
    // Non-quantized storage: dequant once and split.
    let mut full = Vec::with_capacity(fused.rows() * cols);
    for r in 0..fused.rows() {
        full.extend_from_slice(&fused.dequant_row(r));
    }
    let q = WeightMatrix::F32(Tensor::new(
        full[..q_rows * cols].to_vec(),
        vec![q_rows, cols],
    ));
    let k = WeightMatrix::F32(Tensor::new(
        full[q_rows * cols..(q_rows + kv_rows) * cols].to_vec(),
        vec![kv_rows, cols],
    ));
    let v = WeightMatrix::F32(Tensor::new(
        full[(q_rows + kv_rows) * cols..].to_vec(),
        vec![kv_rows, cols],
    ));
    Ok((q, k, v))
}

/// Dense-layer FFN tensors: standard gate/up/down, or Phi-3 fused
/// `ffn_up` with `2 * expert_ffn_dim` rows and no separate gate.
fn load_dense_expert(
    file: &impl TensorSource,
    layer: usize,
    config: &ModelConfig,
) -> Result<ExpertWeights, LoadError> {
    let gate_name = format!("blk.{layer}.ffn_gate.weight");
    let up_name = format!("blk.{layer}.ffn_up.weight");
    let down_name = format!("blk.{layer}.ffn_down.weight");
    if file.find_tensor(&gate_name).is_some() {
        return Ok(ExpertWeights {
            gate: load_weight_matrix(file, &gate_name)?,
            up: load_weight_matrix(file, &up_name)?,
            down: load_weight_matrix(file, &down_name)?,
        });
    }
    // Phi-3 fused SwiGLU: up is [hidden, 2*ff], first half gate, second up.
    let fused = load_weight_matrix(file, &up_name)?;
    let ff = config.moe.expert_ffn_dim;
    if fused.rows() != 2 * ff {
        return Err(LoadError::UnsupportedFeature(
            config.name.to_string(),
            format!(
                "{up_name} has {} rows without a companion ffn_gate; \
                 expected fused SwiGLU with 2*ffn_dim = {} rows",
                fused.rows(),
                2 * ff
            ),
        ));
    }
    let cols = fused.cols();
    // Quantized fused gate+up: split by rows, no dequant (Metal-capable).
    if let (Some(gate), Some(up)) = (
        slice_quantized_rows(&fused, 0, ff),
        slice_quantized_rows(&fused, ff, ff),
    ) {
        return Ok(ExpertWeights {
            gate,
            up,
            down: load_weight_matrix(file, &down_name)?,
        });
    }
    let mut full = Vec::with_capacity(fused.rows() * cols);
    for r in 0..fused.rows() {
        full.extend_from_slice(&fused.dequant_row(r));
    }
    let gate = WeightMatrix::F32(Tensor::new(full[..ff * cols].to_vec(), vec![ff, cols]));
    let up = WeightMatrix::F32(Tensor::new(full[ff * cols..].to_vec(), vec![ff, cols]));
    Ok(ExpertWeights {
        gate,
        up,
        down: load_weight_matrix(file, &down_name)?,
    })
}

fn load_f32_vec(file: &impl TensorSource, name: &str) -> Result<Vec<f32>, LoadError> {
    let info = find_info(file, name)?;
    let raw = file.tensor_bytes(name)?;
    match info.dtype {
        GgmlType::F32 => {
            let mut out = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(out)
        }
        GgmlType::BF16 => ferrox_quant::dequant_bf16(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::BF16)),
        GgmlType::Q8_0 => ferrox_quant::dequant_q8_0(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q8_0)),
        GgmlType::Q4_0 => ferrox_quant::dequant_q4_0(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q4_0)),
        GgmlType::Q4K => ferrox_quant::dequant_q4_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q4K)),
        GgmlType::Q5K => ferrox_quant::dequant_q5_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q5K)),
        GgmlType::Q6K => ferrox_quant::dequant_q6_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q6K)),
        GgmlType::Q2K => ferrox_quant::dequant_q2_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q2K)),
        GgmlType::Q3K => ferrox_quant::dequant_q3_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q3K)),
        GgmlType::Q4_1 => ferrox_quant::dequant_q4_1(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q4_1)),
        GgmlType::Q5_0 => ferrox_quant::dequant_q5_0(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q5_0)),
        GgmlType::Q5_1 => ferrox_quant::dequant_q5_1(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q5_1)),
        GgmlType::Q8_1 => ferrox_quant::dequant_q8_1(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q8_1)),
        GgmlType::IQ4NL => ferrox_quant::dequant_iq4_nl(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ4NL)),
        GgmlType::IQ4XS => ferrox_quant::dequant_iq4_xs(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ4XS)),
        other => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
    }
}

/// Loads a 2D weight matrix, keeping Q8_0/Q4_0 tensors quantized (raw
/// bytes copied out, never dequantized) and only expanding truly F32
/// tensors. This is the memory- and bandwidth-saving path: for a
/// multi-billion-parameter checkpoint the difference between this and
/// "dequant everything on load" is the difference between fitting in
/// RAM and not.
fn load_weight_matrix(file: &impl TensorSource, name: &str) -> Result<WeightMatrix, LoadError> {
    let info = find_info(file, name)?;
    // GGUF's on-disk `ne[]` shape array is fastest-varying-dimension-first
    // (ggml convention), i.e. `[in_features, out_features]` for a 2D
    // weight matrix -- the *reverse* of the row-major `[rows, cols]` =
    // `[out_features, in_features]` order `WeightMatrix`/`matmul_f32`
    // need. Reversed here once so every consumer below gets the correct
    // orientation. Before this reversal existed, every 2D tensor in an
    // externally-produced GGUF file was silently loaded transposed -- a
    // real bug found by running a real downloaded checkpoint
    // (TinyLlama-1.1B-Chat, e.g. `attn_k.weight`'s real raw shape is
    // `[2048, 256]` = `[hidden_dim, kv_dim]` = `[in, out]`) -- found
    // as a real transposition bug affecting every externally-produced
    // GGUF file, caught by serving a real downloaded checkpoint.
    let shape: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
    let (rows, cols) = match shape.as_slice() {
        [r, c] => (*r, *c),
        other => {
            return Err(LoadError::UnsupportedDtype(
                format!("{name} (expected 2D, got shape {other:?})"),
                info.dtype,
            ))
        }
    };

    match info.dtype {
        // BF16 has no block/scale structure to keep quantized-in-place
        // the way Q4_0/Q8_0/K-quants do -- there's no fused dot kernel
        // that would make sense for a plain narrowed float, so it's
        // eagerly widened to an owned f32 Tensor exactly like F32
        // tensors already are.
        GgmlType::F32 | GgmlType::BF16 => {
            let data = load_f32_vec(file, name)?;
            Ok(WeightMatrix::F32(Tensor::new(data, shape)))
        }
        other => match quant_kind_for(other) {
            Some(kind) => {
                let (mmap, range) = file.tensor_mapped_range(name)?;
                #[cfg(feature = "metal")]
                ferrox_metal::gpu::register_weight_mmap(Arc::clone(&mmap));
                Ok(WeightMatrix::Quantized {
                    data: WeightBytes::Mapped { mmap, range },
                    rows,
                    cols,
                    kind,
                })
            }
            None => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
        },
    }
}

/// Splits a packed 3D MoE expert tensor `blk.N.ffn_{gate,up,down}_exps.weight`
/// (shape `[n_experts, out_dim, in_dim]`) into per-expert `WeightMatrix`es,
/// slicing raw bytes directly (quantized tensors stay quantized; block
/// boundaries never cross expert boundaries since `in_dim` is a whole
/// number of quantization blocks). Matches llama.cpp/ik_llama.cpp layout
/// confirmed on real OLMoE and Qwen2-MoE GGUF checkpoints.
fn split_expert_tensor(
    file: &impl TensorSource,
    name: &str,
    n_experts: usize,
) -> Result<Vec<WeightMatrix>, LoadError> {
    let info = find_info(file, name)?;
    // Real raw shape is `[in_dim, out_dim, n_experts]` (ggml's
    // fastest-first `ne[]` order -- see `load_weight_matrix`'s doc
    // comment for the confirmed 2D case this generalizes from). `n_experts`
    // is the slowest-varying (last, i.e. outermost/most-major) dimension,
    // so each expert's `out_dim*in_dim` block is contiguous with experts
    // back-to-back in the mmap.
    if info.shape.len() != 3 || info.shape[2] as usize != n_experts {
        let file_experts = info.shape.last().map(|&d| d as usize).unwrap_or(0);
        return Err(LoadError::ExpertCountMismatch(
            name.to_string(),
            file_experts,
            n_experts,
        ));
    }
    let out_dim = info.shape[1] as usize;
    let in_dim = info.shape[0] as usize;
    let raw = file.tensor_bytes(name)?;

    match info.dtype {
        GgmlType::F32 | GgmlType::BF16 => {
            let all = if info.dtype == GgmlType::BF16 {
                ferrox_quant::dequant_bf16(raw)
                    .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::BF16))?
            } else {
                let mut out = Vec::with_capacity(raw.len() / 4);
                for chunk in raw.chunks_exact(4) {
                    out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
                out
            };
            let per_expert = out_dim * in_dim;
            Ok((0..n_experts)
                .map(|e| {
                    WeightMatrix::F32(Tensor::new(
                        all[e * per_expert..(e + 1) * per_expert].to_vec(),
                        vec![out_dim, in_dim],
                    ))
                })
                .collect())
        }
        other => match quant_kind_for(other) {
            Some(kind) => {
                let (mmap, full_range) = file.tensor_mapped_range(name)?;
                #[cfg(feature = "metal")]
                ferrox_metal::gpu::register_weight_mmap(Arc::clone(&mmap));
                let bytes_per_expert = raw.len() / n_experts;
                Ok((0..n_experts)
                    .map(|e| WeightMatrix::Quantized {
                        data: WeightBytes::Mapped {
                            mmap: Arc::clone(&mmap),
                            range: (full_range.start + e * bytes_per_expert)
                                ..(full_range.start + (e + 1) * bytes_per_expert),
                        },
                        rows: out_dim,
                        cols: in_dim,
                        kind,
                    })
                    .collect())
            }
            None => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
        },
    }
}

/// When every routed expert is mmap-backed with a Metal simdgroup-GEMM
/// kind and back-to-back slices, record the combined gate/up/down planes
/// for Metal packed MoE. Gate/up/down may differ in kind (e.g. Q4_K /
/// Q4_K / Q8_0) but must be uniform across experts per role.
#[cfg(feature = "metal")]
fn try_build_moe_packed_q4_planes(experts: &[ExpertWeights]) -> Option<MoePackedQ4Planes> {
    use ferrox_core::weight_matrix::{QuantKind, WeightBytes};
    use std::sync::Arc;

    if experts.is_empty() {
        return None;
    }

    fn mapped_sg(m: &WeightMatrix) -> Option<(WeightBytes, usize, &'static str)> {
        match m {
            WeightMatrix::Quantized {
                data: WeightBytes::Mapped { mmap, range },
                rows,
                kind,
                ..
            } => {
                let kind_str = match kind {
                    QuantKind::Q4_0 => "Q4_0",
                    QuantKind::Q4K => "Q4_K",
                    QuantKind::Q5K => "Q5_K",
                    QuantKind::Q6K => "Q6_K",
                    QuantKind::Q8_0 => "Q8_0",
                    QuantKind::IQ4XS => "IQ4_XS",
                    _ => return None,
                };
                let _ = ferrox_metal::gpu::mul_mm_sg_meta(kind_str)?;
                Some((
                    WeightBytes::Mapped {
                        mmap: Arc::clone(mmap),
                        range: range.clone(),
                    },
                    *rows,
                    kind_str,
                ))
            }
            _ => None,
        }
    }

    let (gate0, ffn_rows, gate_kind) = mapped_sg(&experts[0].gate)?;
    let (up0, up_rows, up_kind) = mapped_sg(&experts[0].up)?;
    let (down0, hidden_rows, down_kind) = mapped_sg(&experts[0].down)?;
    if up_rows != ffn_rows {
        return None;
    }
    let WeightBytes::Mapped {
        mmap: gate_mmap,
        range: gate0_range,
    } = &gate0
    else {
        return None;
    };
    let WeightBytes::Mapped {
        mmap: up_mmap,
        range: up0_range,
    } = &up0
    else {
        return None;
    };
    let WeightBytes::Mapped {
        mmap: down_mmap,
        range: down0_range,
    } = &down0
    else {
        return None;
    };

    let gate_stride = gate0_range.len();
    let up_stride = up0_range.len();
    let down_stride = down0_range.len();
    if gate_stride == 0 || up_stride == 0 || down_stride == 0 {
        return None;
    }

    let n = experts.len();
    for (i, ex) in experts.iter().enumerate().skip(1) {
        let (g, fr, gk) = mapped_sg(&ex.gate)?;
        let (u, ur, uk) = mapped_sg(&ex.up)?;
        let (d, hr, dk) = mapped_sg(&ex.down)?;
        if gk != gate_kind || uk != up_kind || dk != down_kind {
            return None;
        }
        let WeightBytes::Mapped { mmap, range } = &g else {
            return None;
        };
        if fr != ffn_rows {
            return None;
        }
        if !Arc::ptr_eq(mmap, gate_mmap)
            || range.len() != gate_stride
            || range.start != gate0_range.start + i * gate_stride
        {
            return None;
        }
        let WeightBytes::Mapped { mmap, range } = &u else {
            return None;
        };
        if ur != ffn_rows
            || !Arc::ptr_eq(mmap, up_mmap)
            || range.len() != up_stride
            || range.start != up0_range.start + i * up_stride
        {
            return None;
        }
        let WeightBytes::Mapped { mmap, range } = &d else {
            return None;
        };
        if hr != hidden_rows
            || !Arc::ptr_eq(mmap, down_mmap)
            || range.len() != down_stride
            || range.start != down0_range.start + i * down_stride
        {
            return None;
        }
    }

    Some(MoePackedQ4Planes::new(
        WeightBytes::Mapped {
            mmap: Arc::clone(gate_mmap),
            range: gate0_range.start..gate0_range.start + n * gate_stride,
        },
        WeightBytes::Mapped {
            mmap: Arc::clone(up_mmap),
            range: up0_range.start..up0_range.start + n * up_stride,
        },
        WeightBytes::Mapped {
            mmap: Arc::clone(down_mmap),
            range: down0_range.start..down0_range.start + n * down_stride,
        },
        gate_stride,
        up_stride,
        down_stride,
        n,
        ffn_rows,
        hidden_rows,
        gate_kind,
        up_kind,
        down_kind,
    ))
}

/// One matrix's place inside a store-backed expert's combined byte
/// buffer (gate bytes, then up, then down, concatenated by
/// `GgufExpertSource::read_expert`).
#[derive(Debug, Clone, Copy)]
pub struct StoredMatrixSpec {
    pub offset: usize,
    pub len: usize,
    pub rows: usize,
    pub cols: usize,
    pub kind: QuantKind,
}

/// Byte-range layout of one store-backed routed expert.
#[derive(Debug, Clone, Copy)]
pub struct StoredExpertLayout {
    pub gate: StoredMatrixSpec,
    pub up: StoredMatrixSpec,
    pub down: StoredMatrixSpec,
}

impl StoredExpertLayout {
    pub fn total_bytes(&self) -> usize {
        self.gate.len + self.up.len + self.down.len
    }

    /// Builds temporary zero-copy `WeightMatrix` views over a leased
    /// buffer. Each view's `WeightBytes::Shared` clone of the lease's
    /// `Arc` keeps the cache entry pinned for the view's lifetime.
    pub fn materialize(&self, lease: &ferrox_core::expert_store::ExpertLease) -> ExpertWeights {
        let mk = |spec: &StoredMatrixSpec| WeightMatrix::Quantized {
            data: WeightBytes::Shared {
                buf: lease.shared_buf(),
                range: spec.offset..spec.offset + spec.len,
            },
            rows: spec.rows,
            cols: spec.cols,
            kind: spec.kind,
        };
        ExpertWeights {
            gate: mk(&self.gate),
            up: mk(&self.up),
            down: mk(&self.down),
        }
    }
}

/// [`ExpertSource`] over a (possibly sharded) GGUF checkpoint: each
/// expert's gate/up/down byte ranges are read positionally from the
/// owning shard file and concatenated, so a store miss touches exactly
/// that expert's bytes -- no mmap of the expert region, no shared seek
/// cursor.
pub struct GgufExpertSource {
    files: Vec<std::fs::File>,
    /// (layer, expert) -> the three (file index, offset, len) segments
    /// in gate/up/down order.
    segments: std::collections::HashMap<ExpertKey, [(usize, u64, usize); 3]>,
}

impl ExpertSource for GgufExpertSource {
    fn expert_len(&self, key: ExpertKey) -> Option<usize> {
        self.segments
            .get(&key)
            .map(|segs| segs.iter().map(|&(_, _, len)| len).sum())
    }

    fn read_expert(&self, key: ExpertKey) -> std::io::Result<Vec<u8>> {
        let segs = self
            .segments
            .get(&key)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, format!("{key:?}")))?;
        let total: usize = segs.iter().map(|&(_, _, len)| len).sum();
        let mut buf = vec![0u8; total];
        let mut written = 0;
        for &(fi, offset, len) in segs {
            let dst = &mut buf[written..written + len];
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                self.files[fi].read_exact_at(dst, offset)?;
            }
            #[cfg(not(unix))]
            {
                use std::io::{Read, Seek, SeekFrom};
                let mut f = &self.files[fi];
                f.seek(SeekFrom::Start(offset))?;
                f.read_exact(dst)?;
            }
            written += len;
        }
        Ok(buf)
    }
}

/// Collects the per-expert `(file, offset, len)` segments and layout
/// for one packed 3D expert tensor -- the store-backed counterpart of
/// `split_expert_tensor`, sharing its shape/offset math. Only
/// quantized dtypes are supported (an F32/BF16 expert tensor keeps the
/// resident path; the store exists for the quantized multi-hundred-GB
/// case).
/// One packed 3D expert tensor's store-backed description: the owning
/// shard index, each expert's `(offset, len)` within that shard file,
/// and the matrix spec shared by every expert's slice.
struct StoredTensorSpecs {
    shard: usize,
    per_expert: Vec<(u64, usize)>,
    spec: StoredMatrixSpec,
}

fn stored_expert_specs(
    file: &ShardedGguf,
    name: &str,
    n_experts: usize,
) -> Result<Option<StoredTensorSpecs>, LoadError> {
    let info = find_info(file, name)?;
    if info.shape.len() != 3 || info.shape[2] as usize != n_experts {
        let file_experts = info.shape.last().map(|&d| d as usize).unwrap_or(0);
        return Err(LoadError::ExpertCountMismatch(
            name.to_string(),
            file_experts,
            n_experts,
        ));
    }
    let out_dim = info.shape[1] as usize;
    let in_dim = info.shape[0] as usize;
    let Some(kind) = quant_kind_for(info.dtype) else {
        return Ok(None); // F32/BF16 (or unsupported): resident fallback
    };
    let shard = file
        .tensor_shard_index(name)
        .expect("find_info succeeded, shard index must exist");
    // The mmap range of a tensor within a GgufFile IS its byte offset
    // range within that shard file (the mmap covers the whole file).
    let (_, full_range) = file.tensor_mapped_range(name)?;
    let total_len = full_range.end - full_range.start;
    let bytes_per_expert = total_len / n_experts;
    let per_expert: Vec<(u64, usize)> = (0..n_experts)
        .map(|e| {
            (
                (full_range.start + e * bytes_per_expert) as u64,
                bytes_per_expert,
            )
        })
        .collect();
    let spec = StoredMatrixSpec {
        offset: 0, // caller assigns the position within the combined buffer
        len: bytes_per_expert,
        rows: out_dim,
        cols: in_dim,
        kind,
    };
    Ok(Some(StoredTensorSpecs {
        shard,
        per_expert,
        spec,
    }))
}

impl Decoder {
    /// Loads real weights from `path` for the given `config`. `config`
    /// supplies the architecture shape (layer count, head counts, MoE
    /// topology); tensor names are resolved against it using the
    /// llama.cpp naming convention described in the module docs.
    ///
    /// A `config.moe.n_experts <= 1` model is treated as dense: expert
    /// weights are read from the plain `blk.N.ffn_{gate,up,down}.weight`
    /// tensor names rather than the packed 3D `_exps` variant.
    pub fn from_gguf(
        path: impl AsRef<std::path::Path>,
        config: ModelConfig,
    ) -> Result<Self, LoadError> {
        Self::from_gguf_with_expert_cache(path, config, None)
    }

    /// Like `from_gguf`, but with `expert_cache_bytes: Some(budget)`
    /// routed experts are NOT loaded resident: each layer holds only
    /// byte-range layouts, and expert bytes are read on demand through
    /// one bounded, lease-protected `ExpertStore` shared by every
    /// layer (a single global byte budget; see
    /// `ferrox_core::expert_store`). Dense layers, shared experts,
    /// attention, embeddings, and the output head stay resident/mapped
    /// exactly as before -- only routed experts stream. Layers whose
    /// expert tensors are F32/BF16 fall back to resident loading (the
    /// store exists for the quantized case). Output is bit-identical
    /// to the resident path -- same bytes, same kernels -- pinned by
    /// the roundtrip suite's equivalence test.
    pub fn from_gguf_with_expert_cache(
        path: impl AsRef<std::path::Path>,
        mut config: ModelConfig,
        expert_cache_bytes: Option<u64>,
    ) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let file = ShardedGguf::open(path)?;

        // One store for the whole model (keys are (layer, expert)),
        // built up-front with every stored expert's segments; created
        // only when the cache is enabled AND some layer can use it.
        let mut store_segments: std::collections::HashMap<ExpertKey, [(usize, u64, usize); 3]> =
            std::collections::HashMap::new();
        let mut stored_layouts: Vec<Option<Vec<StoredExpertLayout>>> = Vec::new();

        // Loaded like any other weight matrix: a quantized embedding
        // table stays quantized (zero-copy mmap) and token lookup
        // dequantizes one row via `WeightMatrix::dequant_row`, instead
        // of the whole vocabulary tensor being widened to f32 up front.
        let embedding = load_weight_matrix(&file, "token_embd.weight")?;

        let mut layers = Vec::with_capacity(config.n_layers);
        let mut refined_qk_norm = config.qk_norm_style;
        for l in 0..config.n_layers {
            let (q_proj, k_proj, v_proj) = load_qkv_projections(&file, l, &config)?;
            let q_norm = load_f32_vec_optional(&file, &format!("blk.{l}.attn_q_norm.weight"))?;
            let k_norm = load_f32_vec_optional(&file, &format!("blk.{l}.attn_k_norm.weight"))?;
            // Refine WholeVector vs PerHead from the first observed norm length.
            if let Some(ref w) = q_norm {
                if w.len() == config.head_dim {
                    refined_qk_norm = crate::capability::QkNormStyle::PerHead;
                } else if w.len() == config.n_heads * config.head_dim {
                    refined_qk_norm = crate::capability::QkNormStyle::WholeVector;
                } else {
                    return Err(LoadError::UnsupportedFeature(
                        config.name.to_string(),
                        format!(
                            "blk.{l}.attn_q_norm.weight length {} matches neither head_dim={} \
                             nor n_heads*head_dim={}",
                            w.len(),
                            config.head_dim,
                            config.n_heads * config.head_dim
                        ),
                    ));
                }
            }
            let attn = AttnWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj: load_weight_matrix(&file, &format!("blk.{l}.attn_output.weight"))?,
                norm_weight: load_f32_vec(&file, &format!("blk.{l}.attn_norm.weight"))?,
                q_norm,
                k_norm,
                // Qwen2/Qwen2-MoE-family real QKV bias (`attn_{q,k,v}.bias`,
                // real config `qkv_bias`, `o_proj` has none) -- see
                // `AttnWeights::q_bias`'s doc comment.
                q_bias: load_f32_vec_optional(&file, &format!("blk.{l}.attn_q.bias"))?,
                k_bias: load_f32_vec_optional(&file, &format!("blk.{l}.attn_k.bias"))?,
                v_bias: load_f32_vec_optional(&file, &format!("blk.{l}.attn_v.bias"))?,
                post_attn_norm: load_f32_vec_optional(
                    &file,
                    &format!("blk.{l}.post_attention_norm.weight"),
                )?,
                post_ffn_norm: load_f32_vec_optional(
                    &file,
                    &format!("blk.{l}.post_ffw_norm.weight"),
                )?,
            };

            // Leading dense layers (see ModelConfig::layer_is_dense's
            // doc comment) load from the plain dense tensor names
            // regardless of this model's global MoE topology, matching
            // the DeepSeek-2/3-family convention found in
            // ik_llama.cpp's source. A model with n_experts<=1
            // globally (the dense test fixture) is dense on every
            // layer either way.
            let is_dense_layer = config.layer_is_dense(l) || config.moe.n_experts <= 1;
            let n_experts = if is_dense_layer {
                1
            } else {
                config.moe.n_experts
            };
            let experts: ExpertBacking = if is_dense_layer {
                ExpertBacking::Resident(vec![load_dense_expert(&file, l, &config)?])
            } else {
                // Try store-backed layouts first when the cache is
                // enabled; fall back to resident when any of the three
                // tensors isn't a supported quantized dtype.
                let stored = if expert_cache_bytes.is_some() {
                    let g = stored_expert_specs(
                        &file,
                        &format!("blk.{l}.ffn_gate_exps.weight"),
                        n_experts,
                    )?;
                    let u = stored_expert_specs(
                        &file,
                        &format!("blk.{l}.ffn_up_exps.weight"),
                        n_experts,
                    )?;
                    let d = stored_expert_specs(
                        &file,
                        &format!("blk.{l}.ffn_down_exps.weight"),
                        n_experts,
                    )?;
                    match (g, u, d) {
                        (Some(gt), Some(ut), Some(dt)) => {
                            let mut layouts = Vec::with_capacity(n_experts);
                            for e in 0..n_experts {
                                let key = ExpertKey {
                                    layer: l as u32,
                                    expert: e as u32,
                                };
                                store_segments.insert(
                                    key,
                                    [
                                        (gt.shard, gt.per_expert[e].0, gt.per_expert[e].1),
                                        (ut.shard, ut.per_expert[e].0, ut.per_expert[e].1),
                                        (dt.shard, dt.per_expert[e].0, dt.per_expert[e].1),
                                    ],
                                );
                                let mut gate = gt.spec;
                                let mut up = ut.spec;
                                let mut down = dt.spec;
                                gate.offset = 0;
                                up.offset = gate.len;
                                down.offset = gate.len + up.len;
                                layouts.push(StoredExpertLayout { gate, up, down });
                            }
                            Some(layouts)
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                match stored {
                    Some(layouts) => {
                        // Placeholder; the shared store is attached in a
                        // second pass below once every layer's segments
                        // are collected.
                        stored_layouts.push(Some(layouts));
                        ExpertBacking::Resident(Vec::new())
                    }
                    None => {
                        let gates = split_expert_tensor(
                            &file,
                            &format!("blk.{l}.ffn_gate_exps.weight"),
                            n_experts,
                        )?;
                        let ups = split_expert_tensor(
                            &file,
                            &format!("blk.{l}.ffn_up_exps.weight"),
                            n_experts,
                        )?;
                        let downs = split_expert_tensor(
                            &file,
                            &format!("blk.{l}.ffn_down_exps.weight"),
                            n_experts,
                        )?;
                        ExpertBacking::Resident(
                            gates
                                .into_iter()
                                .zip(ups)
                                .zip(downs)
                                .map(|((gate, up), down)| ExpertWeights { gate, up, down })
                                .collect(),
                        )
                    }
                }
            };
            if stored_layouts.len() < layers.len() + 1 {
                stored_layouts.push(None);
            }

            let shared_experts: Vec<ExpertWeights> =
                if config.moe.n_shared_experts > 0 && !is_dense_layer {
                    vec![ExpertWeights {
                        gate: load_weight_matrix(&file, &format!("blk.{l}.ffn_gate_shexp.weight"))?,
                        up: load_weight_matrix(&file, &format!("blk.{l}.ffn_up_shexp.weight"))?,
                        down: load_weight_matrix(&file, &format!("blk.{l}.ffn_down_shexp.weight"))?,
                    }]
                } else {
                    Vec::new()
                };

            let router = if !is_dense_layer {
                load_weight_matrix(&file, &format!("blk.{l}.ffn_gate_inp.weight"))?
            } else {
                // dense layer: no real router; a zero [1, hidden] matrix
                // always selects the single expert deterministically.
                WeightMatrix::F32(Tensor::zeros(vec![1, config.hidden_dim]))
            };

            let n_for_counts = match &experts {
                ExpertBacking::Resident(v) if v.is_empty() => n_experts,
                other => other.n_experts(),
            };
            let activation_counts = (0..n_for_counts)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect();
            // Qwen2-MoE-specific real tensor (`blk.N.ffn_gate_inp_shexp.weight`,
            // real on-disk shape `[hidden_dim]`, confirmed against
            // llama.cpp's real `qwen2moe.cpp`) -- see
            // `MoeWeights::shared_expert_gate`'s doc comment. Presence
            // of the tensor itself is the real signal (not an
            // architecture-name list): every other supported
            // architecture's checkpoints simply don't carry this
            // tensor, so this naturally stays `None` there.
            let shared_expert_gate = if is_dense_layer {
                None
            } else {
                load_f32_vec_optional(&file, &format!("blk.{l}.ffn_gate_inp_shexp.weight"))?
            };
            #[cfg(feature = "metal")]
            let packed_q4 = match &experts {
                ExpertBacking::Resident(v) if !v.is_empty() => try_build_moe_packed_q4_planes(v),
                _ => None,
            };
            let moe = MoeWeights {
                router,
                experts,
                shared_experts,
                shared_expert_gate,
                norm_weight: load_f32_vec(&file, &format!("blk.{l}.ffn_norm.weight"))?,
                activation_counts,
                #[cfg(feature = "metal")]
                packed_q4,
            };

            layers.push(LayerWeights { attn, moe });
        }

        let final_norm = load_f32_vec(&file, "output_norm.weight")?;
        // Many small Llama/Gemma-family GGUFs tie the lm-head to
        // `token_embd.weight` and omit `output.weight` (llama.cpp
        // `llama_model_loader` falls back the same way). Prefer the
        // explicit head when present.
        let output_head = match load_weight_matrix(&file, "output.weight") {
            Ok(w) => w,
            Err(_) => load_weight_matrix(&file, "token_embd.weight")?,
        };

        // Second pass: attach the one shared store to every
        // store-backed layer. Opening the shard files fresh (plain
        // `File` handles for positional reads, not mmaps) keeps the
        // stored experts' bytes out of the process's mapped footprint
        // entirely.
        if !store_segments.is_empty() {
            let budget = expert_cache_bytes
                .expect("store_segments only populated when a cache budget is set")
                as usize;
            let files: Result<Vec<std::fs::File>, std::io::Error> =
                file.shard_paths().iter().map(std::fs::File::open).collect();
            let files = files.map_err(GgufError::from)?;
            let store = std::sync::Arc::new(ExpertStore::new(
                GgufExpertSource {
                    files,
                    segments: store_segments,
                },
                budget,
            ));
            for (l, layer) in layers.iter_mut().enumerate() {
                if let Some(layouts) = stored_layouts.get_mut(l).and_then(Option::take) {
                    layer.moe.experts = ExpertBacking::Stored {
                        store: std::sync::Arc::clone(&store),
                        layouts,
                        layer: l as u32,
                    };
                }
            }
        }

        config.qk_norm_style = refined_qk_norm;

        let family = crate::capability::resolve_profile(
            file.metadata_str("general.architecture").unwrap_or("llama"),
        )
        .map(|p| p.family)
        .unwrap_or(crate::capability::DecoderFamily::StandardGqa);
        let memory_kind = crate::capability::resolve_profile(
            file.metadata_str("general.architecture").unwrap_or("llama"),
        )
        .map(|p| p.memory)
        .unwrap_or(crate::capability::MemoryKind::KvGqa);
        let execution_plan = crate::execution_plan::ExecutionPlan::from_config(
            &config,
            family,
            memory_kind,
            crate::execution_plan::ExecutionPlan::probe_metal_caps(),
        );

        Ok(Decoder {
            config,
            embedding,
            layers,
            final_norm,
            output_head,
            gpu_vram_budget_bytes: None,
            #[cfg(feature = "metal")]
            metal_attn_kv: std::sync::Mutex::new(None),
            execution_plan,
            plan_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        buf.write_all(s.as_bytes()).unwrap();
    }

    fn write_kv_str(buf: &mut Vec<u8>, key: &str, val: &str) {
        write_string(buf, key);
        buf.write_u32::<LittleEndian>(8).unwrap(); // type = string
        write_string(buf, val);
    }

    /// A minimal, tensor-free GGUF byte buffer declaring only
    /// `general.architecture` (no `{arch}.block_count` or any other
    /// hparam key) -- the shape a stripped-down or malformed file might
    /// take, and the exact case `ModelConfig::from_gguf` must reject
    /// loudly rather than silently default around.
    fn build_arch_only_gguf(arch: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count
        write_kv_str(&mut buf, "general.architecture", arch);
        buf
    }

    #[test]
    fn model_config_from_gguf_fails_loudly_when_required_hparams_are_missing() {
        let tmp = std::env::temp_dir().join("ferrox_test_arch_only.gguf");
        // Use a registered architecture so the failure is MissingHparam,
        // not UnsupportedArchitecture.
        std::fs::write(&tmp, build_arch_only_gguf("llama")).unwrap();
        let file = ferrox_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
        std::fs::remove_file(&tmp).ok();

        match ModelConfig::from_gguf(&file) {
            Err(LoadError::MissingHparam(key)) => {
                assert_eq!(key, "llama.block_count");
            }
            other => panic!(
                "expected LoadError::MissingHparam for a file with no hparam keys, got {other:?}"
            ),
        }
    }

    #[test]
    fn model_config_from_gguf_fails_closed_on_unknown_architecture() {
        let tmp = std::env::temp_dir().join("ferrox_test_unknown_arch.gguf");
        std::fs::write(&tmp, build_arch_only_gguf("bogus-arch-with-no-hparams")).unwrap();
        let file = ferrox_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
        std::fs::remove_file(&tmp).ok();

        match ModelConfig::from_gguf(&file) {
            Err(LoadError::UnsupportedArchitecture(arch)) => {
                assert_eq!(arch, "bogus-arch-with-no-hparams");
            }
            other => panic!(
                "expected LoadError::UnsupportedArchitecture for an unregistered arch, got {other:?}"
            ),
        }
    }

    #[test]
    fn model_config_from_gguf_rejects_dedicated_architectures() {
        let tmp = std::env::temp_dir().join("ferrox_test_dedicated_arch.gguf");
        std::fs::write(&tmp, build_arch_only_gguf("deepseek4")).unwrap();
        let file = ferrox_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
        std::fs::remove_file(&tmp).ok();

        match ModelConfig::from_gguf(&file) {
            Err(LoadError::DedicatedArchitectureRequired(arch, _)) => {
                assert_eq!(arch, "deepseek4");
            }
            other => panic!(
                "expected LoadError::DedicatedArchitectureRequired for deepseek4, got {other:?}"
            ),
        }
    }

    /// The same Q5_K block bytes cross-validated against an independent
    /// Python reference in `ferrox-quant`'s own tests, reused here for
    /// the same full-path proof as the Q6_K test below.
    #[rustfmt::skip]
    const Q5_K_TEST_BLOCK: [u8; 176] = [
        0x66, 0x2a, 0x66, 0x2a, 0x01, 0x01, 0x01, 0x01, 0x4f, 0x4b, 0x10, 0x12, 0x41, 0xe2, 0xc1,
        0xb1, 0x72, 0x2f, 0x20, 0x07, 0x31, 0x0c, 0x38, 0xb3, 0x9c, 0xb8, 0xad, 0x2f, 0x9a, 0xea,
        0x17, 0xd0, 0xee, 0x93, 0x9e, 0x3e, 0x74, 0xbb, 0x28, 0x18, 0x39, 0x25, 0xb6, 0x09, 0x18,
        0x29, 0x1c, 0x1d, 0x29, 0x41, 0x40, 0x0a, 0x74, 0x7d, 0xfd, 0x21, 0xdd, 0x6d, 0x45, 0x73,
        0x0e, 0x1e, 0xc0, 0x4a, 0xfc, 0xf3, 0x8e, 0x24, 0x6b, 0x34, 0x7d, 0xbe, 0x94, 0xde, 0x59,
        0x7a, 0x35, 0x30, 0x36, 0x0a, 0xf9, 0x4a, 0x9b, 0xa2, 0x26, 0x21, 0xa2, 0xfa, 0xdf, 0x4b,
        0x29, 0x64, 0x6f, 0xbb, 0xca, 0x0f, 0x3c, 0xda, 0x20, 0xf4, 0x93, 0x86, 0xab, 0x6e, 0xb9,
        0xe5, 0xd5, 0xa0, 0x82, 0xd6, 0x41, 0xff, 0x12, 0xbc, 0x34, 0xbb, 0xab, 0xb8, 0x20, 0x2f,
        0xbb, 0x5f, 0x0c, 0x10, 0xcf, 0x49, 0xc5, 0x86, 0x5c, 0xdf, 0xff, 0x78, 0x44, 0x26, 0x3b,
        0xc2, 0x23, 0x3d, 0x2b, 0xe9, 0x00, 0x12, 0xf8, 0xea, 0xe2, 0x9e, 0x5e, 0x50, 0x20, 0x9f,
        0x9d, 0x8d, 0x7d, 0x7f, 0xcc, 0x1d, 0x0e, 0x13, 0xf8, 0xc2, 0xf1, 0x3d, 0x08, 0x2f, 0x23,
        0x13, 0xac, 0x0d, 0xa7, 0xe7, 0x20, 0xa3, 0x90, 0xb7, 0xc8, 0x28,
    ];

    fn build_single_q5_k_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "ferrox-q5k-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols,
                                                   // rows] -- reversed from the semantic [rows, cols] this tensor
                                                   // represents (1 row, 256 cols / 1 Q5_K block).
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q5_K block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(13).unwrap(); // dtype tag: Q5_K
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&Q5_K_TEST_BLOCK);
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q5_k_tensor_end_to_end() {
        let tmp = std::env::temp_dir().join("ferrox_test_q5k_tensor.gguf");
        std::fs::write(&tmp, build_single_q5_k_tensor_gguf()).unwrap();
        let file = ferrox_gguf::GgufFile::open(&tmp).expect("real Q5_K GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q5_K tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q5K);
                assert!(
                    data.is_mapped(),
                    "Q5_K tensors should take the zero-copy mmap path, same as Q8_0/Q4_0"
                );
            }
            _ => panic!("expected a Quantized matrix for a Q5_K tensor"),
        }

        let expected = ferrox_quant::dequant_q5_k(&Q5_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < 1e-2,
            "end-to-end loaded+applied Q5_K matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    /// The same Q6_K block bytes cross-validated against an independent
    /// Python reference in `ferrox-quant`'s own tests; reused here to
    /// prove the *full*
    /// path -- real on-disk GGUF bytes, parsed by `ferrox-gguf`, read
    /// through `GgufFile::tensor_mapped_range`, dispatched by
    /// `WeightMatrix::apply` to `ferrox_quant::dot_q6_k_f32` -- produces
    /// the same result as directly dequantizing those bytes, not just
    /// that the isolated kernel is correct in unit-test isolation.
    #[rustfmt::skip]
    const Q6_K_TEST_BLOCK: [u8; 210] = [
        0xe0, 0xa5, 0x40, 0x5c, 0x8d, 0x3a, 0x0a, 0x26, 0xfb, 0x4b, 0x6e, 0x9a, 0xdf, 0x3e, 0xa3,
        0xc4, 0xf8, 0x2b, 0x1d, 0x95, 0x76, 0x7d, 0x3b, 0xcd, 0xfd, 0xef, 0xc2, 0x0b, 0x07, 0x63,
        0x29, 0xfb, 0x81, 0x57, 0xbe, 0xbe, 0x06, 0xf7, 0x3a, 0x92, 0xc4, 0x43, 0xff, 0xad, 0xac,
        0x7e, 0x0f, 0x00, 0x2a, 0x4f, 0xf0, 0xf8, 0xa9, 0xfa, 0x3c, 0x90, 0x6d, 0x73, 0x2d, 0x5a,
        0xe6, 0xc6, 0x46, 0xf2, 0x0d, 0x55, 0x4c, 0x25, 0x38, 0x71, 0x2b, 0x35, 0x38, 0x82, 0x16,
        0x37, 0x5f, 0x32, 0x61, 0x02, 0xdd, 0x2f, 0x6f, 0x7b, 0x1f, 0xb4, 0x1a, 0x1b, 0x3e, 0x4f,
        0x11, 0xa3, 0x17, 0x40, 0x5a, 0x5f, 0x76, 0xcd, 0x19, 0x27, 0x9b, 0xc7, 0xc8, 0xf7, 0xf7,
        0xee, 0xf4, 0x86, 0xd9, 0xfd, 0xa7, 0xfe, 0x9e, 0xac, 0x70, 0x53, 0x5b, 0x76, 0xfb, 0x39,
        0xf8, 0x4b, 0x98, 0xfe, 0xd0, 0x06, 0x21, 0x4c, 0x4d, 0xbe, 0x10, 0x2b, 0x06, 0x65, 0xc9,
        0x5e, 0xf9, 0x95, 0x72, 0xae, 0x99, 0xd9, 0x7e, 0x15, 0xbd, 0x5e, 0x6d, 0xe8, 0x25, 0x8a,
        0xd5, 0x99, 0xc6, 0x6b, 0x69, 0xc7, 0x84, 0xc6, 0xa4, 0xf7, 0xb9, 0x6d, 0x68, 0x45, 0x0e,
        0x65, 0x69, 0xeb, 0xe6, 0xeb, 0xe9, 0x28, 0xa6, 0xb9, 0x96, 0xf2, 0xe8, 0xa7, 0x9b, 0x6e,
        0x79, 0x8a, 0x68, 0x65, 0x59, 0x98, 0x8b, 0x44, 0x41, 0x98, 0x9a, 0x56, 0x01, 0x01, 0x01,
        0x02, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x02, 0x02, 0x01, 0x01, 0x01, 0x02, 0x1f, 0x25,
    ];

    fn build_single_q6_k_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "ferrox-q6k-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q6_K block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(14).unwrap(); // dtype tag: Q6_K
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&Q6_K_TEST_BLOCK);
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q6_k_tensor_end_to_end() {
        let tmp = std::env::temp_dir().join("ferrox_test_q6k_tensor.gguf");
        std::fs::write(&tmp, build_single_q6_k_tensor_gguf()).unwrap();
        let file = ferrox_gguf::GgufFile::open(&tmp).expect("real Q6_K GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q6_K tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q6K);
                assert!(
                    data.is_mapped(),
                    "Q6_K tensors should take the zero-copy mmap path, same as Q8_0/Q4_0"
                );
            }
            _ => panic!("expected a Quantized matrix for a Q6_K tensor"),
        }

        let expected = ferrox_quant::dequant_q6_k(&Q6_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < 1e-2,
            "end-to-end loaded+applied Q6_K matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    fn build_single_bf16_tensor_gguf(rows: u64, cols: u64, values: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "ferrox-bf16-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(cols).unwrap();
        buf.write_u64::<LittleEndian>(rows).unwrap();
        buf.write_u32::<LittleEndian>(30).unwrap(); // dtype tag: BF16
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for &v in values {
            // Real bf16 truncation (round-toward-zero, matching a real
            // writer closely enough for round-trip test purposes): top
            // 16 bits of the f32 bit pattern.
            let bf16_bits = (v.to_bits() >> 16) as u16;
            buf.extend_from_slice(&bf16_bits.to_le_bytes());
        }
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_bf16_tensor_end_to_end() {
        // Values with zero low-mantissa bits, so f32->bf16 truncation
        // is lossless and this is an exact-equality check.
        let values: Vec<f32> = vec![1.0, -2.5, 0.0, 4.0, -8.0, 16.0];
        let tmp = std::env::temp_dir().join("ferrox_test_bf16_tensor.gguf");
        std::fs::write(&tmp, build_single_bf16_tensor_gguf(2, 3, &values)).unwrap();
        let file = ferrox_gguf::GgufFile::open(&tmp).expect("real BF16 GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("BF16 tensor must load");
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        match &matrix {
            WeightMatrix::F32(tensor) => {
                assert_eq!(tensor.data, values, "BF16 must widen to f32 exactly");
            }
            _ => panic!("expected an F32 matrix for a BF16 tensor (no fused dot kernel for it)"),
        }
    }

    fn build_single_q5_1_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "ferrox-q5-1-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(32).unwrap(); // cols (1 Q5_1 block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(7).unwrap(); // dtype tag: Q5_1
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        // d=0.25 (f16 0x3400), m=1.5 (f16 0x3E00) -- both exact in f16,
        // hand-verified bit patterns to avoid pulling in the `half`
        // crate just for two test constants. qh varied, qs a real
        // (non-degenerate) pattern.
        buf.extend_from_slice(&0x3400u16.to_le_bytes());
        buf.extend_from_slice(&0x3E00u16.to_le_bytes());
        buf.extend_from_slice(&[0x9au8, 0x3c, 0xf0, 0x0f]);
        buf.extend_from_slice(&(0..16u8).map(|i| i | ((15 - i) << 4)).collect::<Vec<u8>>());
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q5_1_tensor_end_to_end() {
        let tmp = std::env::temp_dir().join("ferrox_test_q5_1_tensor.gguf");
        std::fs::write(&tmp, build_single_q5_1_tensor_gguf()).unwrap();
        let file = ferrox_gguf::GgufFile::open(&tmp).expect("real Q5_1 GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q5_1 tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 32);
        let raw = file.tensor_bytes("test.weight").unwrap();
        let expected = ferrox_quant::dequant_q5_1(raw).unwrap();
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q5_1);
                assert!(data.is_mapped());
            }
            _ => panic!("expected a Quantized matrix for a Q5_1 tensor"),
        }

        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.017).cos()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < 1e-2,
            "end-to-end loaded+applied Q5_1 matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    // Same bytes as ferrox-quant's own Q3_K_TEST_BLOCK (Python-cross-
    // validated there); duplicated here to build a real on-disk GGUF
    // file, matching this file's existing per-format test convention
    // (see Q6_K_TEST_BLOCK above).
    const Q3_K_TEST_BLOCK: [u8; 110] = [
        0x56, 0xf2, 0xb4, 0x2b, 0xd5, 0x6f, 0x51, 0x71, 0x3c, 0x0a, 0xb9, 0x1d, 0xd0, 0xb9, 0x3b,
        0xb3, 0x0f, 0xff, 0x8c, 0xb2, 0x83, 0x3a, 0x3d, 0x24, 0xb1, 0x12, 0x56, 0xe3, 0x23, 0x54,
        0xf2, 0xfa, 0x7f, 0xdf, 0x31, 0xe1, 0x18, 0x26, 0x6e, 0xcd, 0x5b, 0x38, 0xee, 0xbd, 0x9f,
        0x8c, 0x57, 0x47, 0x0b, 0x11, 0xcb, 0xfb, 0xb4, 0x83, 0xa0, 0x4e, 0x0b, 0xd4, 0xa7, 0x85,
        0xe0, 0x60, 0xf3, 0xb3, 0xe3, 0x95, 0x43, 0xc6, 0x05, 0x05, 0x77, 0x53, 0xed, 0x23, 0xcc,
        0x6a, 0x0e, 0x89, 0xa1, 0x79, 0x85, 0xf6, 0x6e, 0x5a, 0x23, 0x63, 0xbe, 0x53, 0xfa, 0xa2,
        0x2b, 0xe9, 0xcd, 0xce, 0xf8, 0x3d, 0x6f, 0xd0, 0x42, 0x6e, 0x3b, 0x7f, 0x23, 0x26, 0xd3,
        0xb9, 0x18, 0xbf, 0xa4, 0x34,
    ];

    fn build_single_q3_k_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "ferrox-q3k-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q3_K block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(11).unwrap(); // dtype tag: Q3_K
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&Q3_K_TEST_BLOCK);
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q3_k_tensor_end_to_end() {
        let tmp = std::env::temp_dir().join("ferrox_test_q3k_tensor.gguf");
        std::fs::write(&tmp, build_single_q3_k_tensor_gguf()).unwrap();
        let file = ferrox_gguf::GgufFile::open(&tmp).expect("real Q3_K GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q3_K tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q3K);
                assert!(data.is_mapped());
            }
            _ => panic!("expected a Quantized matrix for a Q3_K tensor"),
        }

        let expected = ferrox_quant::dequant_q3_k(&Q3_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < 1e-1,
            "end-to-end loaded+applied Q3_K matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    // Same bytes as ferrox-quant's own IQ4_XS_TEST_BLOCK (Python-cross-
    // validated there); duplicated here to build a real on-disk GGUF
    // file, matching this file's existing per-format test convention.
    const IQ4_XS_TEST_BLOCK: [u8; 136] = [
        0x5c, 0x33, 0xb4, 0x39, 0xd1, 0x64, 0x97, 0x82, 0xcb, 0xbd, 0x88, 0x95, 0xf3, 0x60, 0x2a,
        0xb5, 0xe7, 0x24, 0xd3, 0xee, 0xfe, 0x71, 0x13, 0xbe, 0x70, 0x84, 0x48, 0x79, 0x7b, 0x3e,
        0xf0, 0x55, 0xdc, 0xb2, 0xb2, 0xde, 0x32, 0xa1, 0x5b, 0x02, 0x01, 0xdc, 0x2a, 0xbb, 0xf7,
        0x0b, 0x8a, 0x88, 0xdd, 0x0b, 0x02, 0x7e, 0x5e, 0x76, 0x87, 0x30, 0x1e, 0x1c, 0xcf, 0x48,
        0xd7, 0x61, 0xf3, 0x51, 0x52, 0x17, 0x98, 0x0a, 0x87, 0xcf, 0x02, 0x91, 0xc8, 0xee, 0xc0,
        0x91, 0x69, 0x2a, 0x4f, 0x64, 0x68, 0xa7, 0xb2, 0xe6, 0x98, 0x21, 0x81, 0x75, 0x53, 0x2a,
        0x8d, 0x12, 0xae, 0xe0, 0xea, 0x0c, 0x75, 0xff, 0x22, 0x5e, 0x25, 0x19, 0xda, 0x2e, 0x51,
        0x4e, 0x81, 0xdc, 0x0e, 0x78, 0x86, 0xd7, 0x58, 0xb5, 0xb7, 0xf6, 0x45, 0xa9, 0x0a, 0x83,
        0xfd, 0x2a, 0x12, 0x7d, 0xf0, 0x12, 0x97, 0xe2, 0xfe, 0xf4, 0xd0, 0xa2, 0x11, 0x14, 0x78,
        0xdb,
    ];

    fn build_single_iq4_xs_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "ferrox-iq4xs-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 IQ4_XS block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(23).unwrap(); // dtype tag: IQ4_XS
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&IQ4_XS_TEST_BLOCK);
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_iq4_xs_tensor_end_to_end() {
        let tmp = std::env::temp_dir().join("ferrox_test_iq4xs_tensor.gguf");
        std::fs::write(&tmp, build_single_iq4_xs_tensor_gguf()).unwrap();
        let file = ferrox_gguf::GgufFile::open(&tmp).expect("real IQ4_XS GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("IQ4_XS tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::IQ4XS);
                assert!(data.is_mapped());
            }
            _ => panic!("expected a Quantized matrix for an IQ4_XS tensor"),
        }

        let expected = ferrox_quant::dequant_iq4_xs(&IQ4_XS_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < 1e-1,
            "end-to-end loaded+applied IQ4_XS matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    // Same bytes as ferrox-quant's own IQ low-bit test blocks
    // (Python-cross-validated there against the real compiled ggml
    // implementation), duplicated as literals for the same reason as
    // IQ4_XS_TEST_BLOCK above.
    const IQ1_S_TEST_BLOCK: [u8; 50] = [
        0x0a, 0x2f, 0xfa, 0x06, 0x1e, 0x37, 0x6f, 0xe3, 0x62, 0xd0, 0xb6, 0xa4, 0x25, 0xae, 0x76,
        0x14, 0x72, 0x5b, 0xfa, 0x05, 0xd1, 0xf1, 0x2a, 0x4c, 0xad, 0x29, 0xae, 0xf4, 0xcf, 0x0c,
        0x96, 0x51, 0x58, 0x03, 0x6d, 0xd3, 0x10, 0x92, 0x70, 0xff, 0x61, 0x58, 0xc8, 0x30, 0x25,
        0x64, 0x49, 0x85, 0xc0, 0x24,
    ];
    const IQ2_XXS_TEST_BLOCK: [u8; 66] = [
        0x29, 0x30, 0xd9, 0x33, 0x95, 0x4c, 0x08, 0x1e, 0xad, 0x79, 0x49, 0xf2, 0x8d, 0x5f, 0x93,
        0xea, 0x78, 0x18, 0x98, 0xb9, 0x94, 0x14, 0xad, 0xce, 0xca, 0x1d, 0xab, 0x81, 0x53, 0x4a,
        0x68, 0xd0, 0x59, 0x96, 0x36, 0x5d, 0xbe, 0x20, 0xc4, 0xff, 0xe4, 0x2c, 0xcd, 0x2f, 0x4f,
        0x4f, 0x67, 0x53, 0xc6, 0xd5, 0xa2, 0xfb, 0xc7, 0xf3, 0xe2, 0x6b, 0xf1, 0x99, 0x23, 0x1e,
        0x2d, 0x5e, 0x8c, 0x78, 0xc2, 0x31,
    ];
    const IQ3_XXS_TEST_BLOCK: [u8; 98] = [
        0x71, 0x31, 0x16, 0x0a, 0x79, 0x04, 0x5d, 0x87, 0xae, 0x2a, 0x4a, 0x43, 0xfd, 0x02, 0xba,
        0x6c, 0x10, 0x42, 0x80, 0xe5, 0x1d, 0x08, 0x22, 0xcb, 0x21, 0x54, 0xf9, 0xaa, 0x8e, 0xc2,
        0xf2, 0x34, 0x66, 0x1e, 0x2a, 0xef, 0x19, 0xae, 0x48, 0x47, 0x29, 0xa0, 0x72, 0xd1, 0x31,
        0xc0, 0x65, 0x49, 0xde, 0x79, 0x32, 0xe6, 0x4d, 0xb6, 0x55, 0x3f, 0x4d, 0xf1, 0x18, 0xbb,
        0x18, 0x59, 0x4c, 0x31, 0xa3, 0xb2, 0x34, 0xdd, 0xf6, 0x4a, 0x91, 0x51, 0x3f, 0x3e, 0x40,
        0x69, 0xad, 0xbf, 0x1a, 0xd0, 0x05, 0xfb, 0xbe, 0x8b, 0x0b, 0xdd, 0xdf, 0x7d, 0x94, 0x74,
        0x92, 0x3e, 0xff, 0x04, 0x2a, 0xc4, 0xea, 0xc9,
    ];

    #[rustfmt::skip]
    const MXFP4_GGUF_TEST_BLOCKS: [u8; 68] = [0x79, 0xb4, 0x8d, 0xe2, 0x62, 0x5d, 0xbb, 0x9d, 0x54, 0xe6, 0xdb, 0x94, 0x59, 0x7d, 0x28, 0xf9, 0x79, 0x7a, 0xfc, 0xc1, 0xfa, 0x1e, 0x53, 0x5b, 0x0e, 0xc2, 0x5a, 0x2f, 0x0c, 0x82, 0x4d, 0xcb, 0x11, 0x28, 0x7b, 0x7c, 0xb6, 0x45, 0xe0, 0xb0, 0x52, 0x40, 0x51, 0xec, 0x30, 0x1a, 0xd2, 0x17, 0xf3, 0xbb, 0xfc, 0x7c, 0x8f, 0xf0, 0x67, 0x83, 0x88, 0x9d, 0x79, 0xdb, 0xf4, 0x45, 0x29, 0x78, 0xe6, 0xf4, 0x99, 0xea];

    fn build_single_iq_lowbit_tensor_gguf(
        arch: &str,
        tag: u32,
        cols: u64,
        block: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count
        write_kv_str(&mut buf, "general.architecture", arch);
        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
        buf.write_u64::<LittleEndian>(cols).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(tag).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(block);
        buf
    }

    /// End-to-end load+apply for the three codebook-grid low-bit
    /// formats the published Dynamic GGUFs are built from: a real
    /// on-disk tensor of each type must load zero-copy as the right
    /// `QuantKind` and produce the same matvec result as dequantizing
    /// the block directly. Dtype tags (19/16/18) verified against
    /// ggml.h's enum ggml_type.
    #[test]
    fn load_weight_matrix_handles_real_on_disk_iq_lowbit_tensors_end_to_end() {
        type DequantFn = fn(&[u8]) -> Result<Vec<f32>, ferrox_quant::QuantError>;
        let cases: [(&str, u32, &[u8], QuantKind, DequantFn); 4] = [
            (
                "iq1s",
                19,
                &IQ1_S_TEST_BLOCK,
                QuantKind::IQ1S,
                ferrox_quant::dequant_iq1_s,
            ),
            (
                "iq2xxs",
                16,
                &IQ2_XXS_TEST_BLOCK,
                QuantKind::IQ2XXS,
                ferrox_quant::dequant_iq2_xxs,
            ),
            (
                "iq3xxs",
                18,
                &IQ3_XXS_TEST_BLOCK,
                QuantKind::IQ3XXS,
                ferrox_quant::dequant_iq3_xxs,
            ),
            (
                "mxfp4_gguf",
                39,
                &MXFP4_GGUF_TEST_BLOCKS,
                QuantKind::Mxfp4Gguf,
                ferrox_quant::dequant_mxfp4_gguf,
            ),
        ];
        for (name, tag, block, kind, dequant) in cases {
            let expected = dequant(block).unwrap();
            let cols = expected.len();
            let tmp = std::env::temp_dir().join(format!("ferrox_test_{name}_tensor.gguf"));
            std::fs::write(
                &tmp,
                build_single_iq_lowbit_tensor_gguf(name, tag, cols as u64, block),
            )
            .unwrap();
            let file = ferrox_gguf::GgufFile::open(&tmp).expect("file must parse");
            std::fs::remove_file(&tmp).ok();

            let matrix =
                load_weight_matrix(&file, "test.weight").expect("low-bit tensor must load");
            assert_eq!((matrix.rows(), matrix.cols()), (1, cols), "{name}");
            match &matrix {
                WeightMatrix::Quantized { kind: k, data, .. } => {
                    assert_eq!(*k, kind, "{name}");
                    assert!(data.is_mapped(), "{name} must load zero-copy");
                }
                _ => panic!("expected a Quantized matrix for {name}"),
            }

            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin()).collect();
            let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let got = matrix.apply(&x);
            assert!(
                (got[0] - expected_dot).abs() < 1e-1,
                "{name}: loaded+applied diverged from direct dequant: got={} expected={}",
                got[0],
                expected_dot
            );
        }
    }

    #[test]
    fn qwen2moe_disables_topk_renorm() {
        assert!(
            NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&"qwen2moe"),
            "qwen2moe must have norm_topk_prob=false (llama.cpp build_moe_ffn norm_w=false)"
        );
    }
}
