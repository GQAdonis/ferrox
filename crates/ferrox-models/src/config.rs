//! Architecture configs. Prefer GGUF / config.json over preset defaults.
//! Unconfirmed preset fields must be listed in `best_effort_fields`.
//! What actually runs: `docs/MODELS.md`.

use ferrox_moe::{GatingFunction, MoeLayerConfig};

/// Which attention mechanism a model uses. `Gqa` (grouped-query
/// attention + RoPE, uniform across every layer) is the only variant
/// `ferrox-core`/`ferrox-models::decoder` actually implement today --
/// it's what every preset runs through, including the two whose real
/// published attention differs (DeepSeek V4 Pro's CSA/HCA, Kimi K3's
/// hybrid KDA/Gated-MLA). `KimiHybrid` exists so Kimi K3's real,
/// cited attention hyperparameters are captured accurately rather than
/// silently discarded, even though `Decoder` itself still runs the
/// GQA path for every layer (the dedicated Kimi decoder is the one
/// consumer of the hybrid variant today).
#[derive(Debug, Clone)]
pub enum AttentionKind {
    Gqa,
    KimiHybrid(KimiHybridAttention),
}

/// Which concrete attention mechanism a single 0-indexed layer uses --
/// the resolved answer `Decoder` needs per layer once it dispatches on
/// `AttentionKind` instead of always running GQA (see
/// `ModelConfig::layer_attention_kind`; only the dedicated Kimi
/// decoder actually dispatches on it today).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerAttentionKind {
    Gqa,
    /// KDA (Kimi Delta Attention) -- see `ferrox_models::kda`.
    KimiKda,
    /// Gated MLA -- see `ferrox_models::mla`.
    KimiMla,
}

/// Kimi K3's real attention topology, transcribed from the published
/// `huggingface.co/moonshotai/Kimi-K3/config.json`'s `linear_attn_config`
/// block. `kda_layers`/`full_attn_layers` are kept exactly as published
/// -- **1-indexed** (layer 1 is the model's first transformer layer),
/// not `ferrox`'s usual 0-indexed `layers` slice -- so a caller wiring
/// this into `Decoder` must subtract 1 before indexing.
#[derive(Debug, Clone)]
pub struct KimiHybridAttention {
    /// 1-indexed layers using KDA (Kimi Delta Attention: gated
    /// linear/recurrent attention with a short causal conv). 69 of 93
    /// layers.
    pub kda_layers: Vec<usize>,
    /// 1-indexed layers using Gated MLA (DeepSeek-style multi-head
    /// latent attention with an output gate). 24 of 93 layers.
    pub full_attn_layers: Vec<usize>,
    pub mla: MlaConfig,
    pub kda: KdaConfig,
}

/// Gated MLA (multi-head latent attention) hyperparameters, verified
/// against Kimi K3's real `config.json` `text_config` block and the
/// real `KimiMLAAttention` reference implementation
/// (`modeling_kimi_linear.py`).
#[derive(Debug, Clone)]
pub struct MlaConfig {
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// Kimi K3's addition on top of standard DeepSeek-style MLA:
    /// `attn_output *= sigmoid(g_proj(hidden_states))` before `o_proj`.
    pub use_output_gate: bool,
    /// `None` reproduces Kimi K3's real, confirmed behavior: no rotary
    /// embedding at all (`mla.rs`'s module doc comment; the real
    /// `KimiMLAAttention.forward` asserts `use_nope` and never calls a
    /// rotary function). `Some` is for architectures whose decoupled
    /// `q_rot`/`k_rot` slices genuinely are position-rotated -- e.g.
    /// GLM-5.2, whose real `config.json` (`zai-org/GLM-5.2`) sets
    /// `rope_interleave: true` for its main attention (confirmed
    /// against llama.cpp PR #25407's `LLAMA_ROPE_TYPE_NORM` rope call
    /// on `q_pe`/`k_pe` in `src/models/glm-dsa.cpp`).
    pub rope: Option<MlaRopeConfig>,
}

/// RoPE parameters for the decoupled `q_rot`/`k_rot` slices of an MLA
/// attention layer that does apply rotation (unlike Kimi K3 -- see
/// `MlaConfig::rope`'s doc comment). Always the interleaved convention
/// (`ferrox_core::attention::apply_rope_interleaved`) for every real
/// architecture confirmed so far to use this (GLM-5.2's
/// `rope_interleave: true`); a separate split-half variant isn't wired
/// in here since no confirmed real user of it exists yet.
#[derive(Debug, Clone, Copy)]
pub struct MlaRopeConfig {
    pub theta: f32,
}

/// KDA (Kimi Delta Attention) hyperparameters, verified against Kimi
/// K3's real `config.json` `linear_attn_config` block and the real
/// gated delta-rule reference implementation in
/// `fla-org/flash-linear-attention`'s `fla/ops/kda/naive.py` (the
/// exact recurrence: decay state by `exp(g)`, then add a rank-1
/// `beta * k ⊗ (v - kᵀS)` correction, then read `o = qᵀS`) and
/// `fla/ops/kda/gate.py` (the lower-bounded gate:
/// `g = gate_lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))`,
/// and `beta = sigmoid(raw_beta)`).
#[derive(Debug, Clone)]
pub struct KdaConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub short_conv_kernel_size: usize,
    pub gate_lower_bound: f32,
    pub use_full_rank_gate: bool,
}

/// Which RoPE pairing convention a model uses. Confirmed against
/// llama.cpp's `llama_model_rope_type` (`src/llama-model.cpp`):
/// `Norm` is adjacent-pair / GPT-J (`LLAMA_ROPE_TYPE_NORM`); `Neox` is
/// split-half / GPT-NeoX (`LLAMA_ROPE_TYPE_NEOX`). Getting this wrong
/// silently produces fluent-but-wrong logits (the real Llama-3.1-8B
/// early-stop bug: ferrox applied NeoX to a Norm architecture).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// Adjacent pairs `(2*i, 2*i+1)` -- llama.cpp `LLAMA_ROPE_TYPE_NORM`.
    /// Used by `llama` (including Llama 3/3.1/3.2), `llama4`, `deepseek2`,
    /// `mistral3`, and related families.
    Norm,
    /// Split-half pairs `(i, i+half)` -- llama.cpp `LLAMA_ROPE_TYPE_NEOX`.
    /// Used by `olmoe`, `qwen2`/`qwen2moe`/`qwen3`, `phi3`, `gemma*`, and
    /// related families. Ferrox's historical default before architecture-
    /// aware dispatch existed.
    Neox,
}

impl RopeLayout {
    /// Maps a GGUF `general.architecture` string onto the RoPE pairing
    /// llama.cpp selects for that family. Prefer
    /// [`crate::capability::resolve_architecture`] for load-time
    /// decisions — unknown architectures must fail closed there rather
    /// than guessing. This helper remains for tests and call sites that
    /// already know the arch is registered; unknowns still return `Neox`
    /// only as a last-resort historical default.
    pub fn for_gguf_architecture(arch: &str) -> Self {
        match crate::capability::resolve_profile(arch) {
            Some(p) => p.rope,
            // Unknown: do not invent Norm for a Qwen/Phi/Gemma-shaped
            // string that happened to miss the registry.
            None => RopeLayout::Neox,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: &'static str,
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub moe: MoeLayerConfig,
    /// `Gqa` for every preset except Kimi K3. `Decoder`'s forward pass
    /// does not yet branch on this -- see `AttentionKind`'s doc
    /// comment.
    pub attention: AttentionKind,
    /// Mistral/Mixtral/Qwen2-family sliding-window attention: when
    /// set, every layer attends only to the most recent `N` cached
    /// positions instead of the full causal history (see
    /// `ferrox_core::attention::causal_gqa_attention_windowed`'s doc
    /// comment for the real source citations). `None` for every
    /// architecture that doesn't use this (most models, including
    /// Qwen1.5/Qwen2-MoE's real published config, which sets
    /// `use_sliding_window: false` despite carrying a `sliding_window`
    /// value -- so this field being `None`/`Some` must come from that
    /// enable flag, not just the window-size field's presence).
    pub sliding_window: Option<usize>,
    /// How many of the model's *first* layers use an ordinary dense
    /// FFN (no expert routing at all) rather than the model's MoE
    /// topology. Found by reading ik_llama.cpp's real GGUF
    /// hparams-loading source (`LLM_KV_LEADING_DENSE_BLOCK_COUNT`):
    /// DeepSeek-2/3-family models don't apply MoE uniformly to every
    /// layer -- the first few layers are always dense. Zero means
    /// "every layer uses this model's MoE topology," the default for
    /// architectures that don't do this.
    pub n_dense_leading_layers: usize,
    /// Llama 3/3.1/3.2's real per-band RoPE frequency correction (the
    /// `rope_freqs.weight` GGUF tensor, `head_dim/2` elements,
    /// `TENSOR_NOT_REQUIRED` so most architectures leave this `None`).
    /// See `ferrox_core::attention::apply_rope_with_freq_factors`'s doc
    /// comment for the real source and the real bug this closes: without
    /// it, every RoPE angle for a Llama-3-family checkpoint is computed
    /// slightly wrong, an error that compounds with position and
    /// eventually produces wrong logits (a spurious early EOS was the
    /// observed real symptom).
    pub rope_freqs: Option<Vec<f32>>,
    /// RoPE pairing convention for this architecture -- see
    /// `RopeLayout`. Independently of `rope_freqs`: a Llama checkpoint
    /// needs both `Norm` pairing *and* the per-band frequency factors.
    pub rope_layout: RopeLayout,
    /// How Q/K RMSNorm weights are applied when present (see
    /// [`crate::capability::QkNormStyle`]).
    pub qk_norm_style: crate::capability::QkNormStyle,
    /// Gemma 2+/3 alternating SWA period. When `Some(p)`, layer `il`
    /// uses sliding-window attention iff `(il + 1) % p != 0` (llama.cpp
    /// `set_swa_pattern` convention); every `p`-th layer is full-attn.
    pub swa_pattern: Option<usize>,
    /// Attention logit soft-capping (Gemma 2+). Applied as
    /// `softcap * tanh(score / softcap)` before softmax.
    pub attn_logit_softcap: Option<f32>,
    /// Final logit soft-capping (Gemma 2+). Applied to lm_head output.
    pub final_logit_softcap: Option<f32>,
    /// Input embedding scale (Gemma: `sqrt(hidden_dim)`).
    pub embedding_scale: Option<f32>,
    /// Optional override for the attention score scale baked into Q
    /// *instead of* the kernel's default `1/sqrt(head_dim)`. When set,
    /// callers must pass `score_scale = 1.0` into the attention kernel
    /// (llama.cpp Gemma: scale Q then `build_attn(..., 1.0f)`). Prefer
    /// leaving this `None` when the override equals `1/sqrt(head_dim)`.
    pub attention_scale: Option<f32>,
    /// RoPE base used on SWA layers (Gemma 3: defaults to `10000` when
    /// the GGUF omits `rope.freq_base_swa`; full-attn layers keep
    /// [`Self::rope_theta`]).
    pub rope_theta_swa: Option<f32>,
    /// Dense/MoE FFN activation pairing.
    pub ffn_activation: FfnActivation,
    /// Every field on this config that is a best-effort estimate rather
    /// than a confirmed value from an official config.json / GGUF file.
    pub best_effort_fields: &'static [&'static str],
}

/// Dense / expert FFN non-linearity used by the generic decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FfnActivation {
    /// `silu(gate) * up` with separate gate/up matrices (Llama / Qwen).
    #[default]
    Swiglu,
    /// Phi-3 fused gate+up in one `ffn_up` matrix (`2 * n_ff` rows).
    SwigluFused,
    /// Gemma GeGLU: `gelu(gate) * up`.
    Gelu,
}

impl ModelConfig {
    /// True if layer `layer_idx` (0-indexed) should be built as an
    /// ordinary dense FFN rather than this model's MoE topology.
    pub fn layer_is_dense(&self, layer_idx: usize) -> bool {
        layer_idx < self.n_dense_leading_layers
    }

    /// Sliding-window size for layer `il`, honouring Gemma-style
    /// alternating SWA patterns. `None` means full causal attention.
    pub fn layer_sliding_window(&self, layer_idx: usize) -> Option<usize> {
        let window = self.sliding_window?;
        match self.swa_pattern {
            None => Some(window),
            Some(period) if period > 1 => {
                // llama.cpp `set_swa_pattern` (dense_first=false):
                // `is_swa = (il % period) < (period - 1)` — equivalent to
                // full attention when `(il + 1) % period == 0`.
                if (layer_idx + 1).is_multiple_of(period) {
                    None
                } else {
                    Some(window)
                }
            }
            Some(_) => Some(window),
        }
    }

    /// RoPE frequency base for layer `il` (SWA layers may differ).
    pub fn layer_rope_theta(&self, layer_idx: usize) -> f32 {
        match (self.layer_sliding_window(layer_idx), self.rope_theta_swa) {
            (Some(_), Some(theta)) => theta,
            _ => self.rope_theta,
        }
    }

    /// Which attention mechanism layer `layer_idx` (0-indexed, ferrox's
    /// usual convention) uses. For `AttentionKind::Gqa` every layer is
    /// `LayerAttentionKind::Gqa`; for `AttentionKind::KimiHybrid`, looks
    /// up `layer_idx + 1` (the real `kda_layers`/`full_attn_layers`
    /// lists are 1-indexed -- see `KimiHybridAttention`'s doc comment)
    /// in those real per-layer lists.
    ///
    /// # Panics
    /// If `layer_idx` isn't covered by either list of a `KimiHybrid`
    /// config -- can't happen for `kimi_k3()`, whose lists are tested
    /// (`kimi_k3_hybrid_attention_layers_partition_every_layer_exactly_once`)
    /// to partition every layer with no gaps, but a caller building a
    /// custom `KimiHybridAttention` must uphold the same invariant.
    pub fn layer_attention_kind(&self, layer_idx: usize) -> LayerAttentionKind {
        match &self.attention {
            AttentionKind::Gqa => LayerAttentionKind::Gqa,
            AttentionKind::KimiHybrid(hybrid) => {
                let one_indexed = layer_idx + 1;
                if hybrid.kda_layers.contains(&one_indexed) {
                    LayerAttentionKind::KimiKda
                } else if hybrid.full_attn_layers.contains(&one_indexed) {
                    LayerAttentionKind::KimiMla
                } else {
                    panic!(
                        "layer {layer_idx} (1-indexed {one_indexed}) is in neither \
                         kda_layers nor full_attn_layers"
                    )
                }
            }
        }
    }

    /// Total parameter count implied by the MoE config, as a sanity
    /// check against the publicly reported total (this is an order of
    /// magnitude check, not an exact parameter-count reproduction).
    pub fn approx_active_params_per_token(&self) -> usize {
        let attn_params_per_layer = 4 * self.hidden_dim * self.hidden_dim; // q,k,v,o (rough)
        let active_experts = self.moe.n_experts_active + self.moe.n_shared_experts;
        let expert_params = active_experts * 3 * self.moe.hidden_dim * self.moe.expert_ffn_dim; // gate,up,down
        self.n_layers * (attn_params_per_layer + expert_params)
    }
}

/// GLM-5.2 (Z.ai) **structural sketch only** — not a supported real
/// inference path. Real DSA lives in `glm_dsa` / `glm52_decoder` and is
/// not wired into `Decoder` / `ferrox-server`. This preset drives
/// smoke/bench with synthetic GQA weights only (~744B / ~40B active
/// hparams as published placeholders).
pub fn glm_5_2() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "glm-5.2",
        attention: AttentionKind::Gqa,
        n_layers: 92,
        hidden_dim: 6144,
        n_heads: 48,
        n_kv_heads: 8,
        head_dim: 128,
        vocab_size: 151552,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-5,
        moe: MoeLayerConfig {
            n_experts: 256,
            n_experts_active: 8,
            n_shared_experts: 1,
            hidden_dim: 6144,
            expert_ffn_dim: 2048,
            // Sigmoid, not softmax: reading ik_llama.cpp's real GGUF
            // hparams-loading source (llama-hparams.cpp,
            // LLM_ARCH_GLM4_MOE case) directly showed GLM4-MoE-family
            // models default to sigmoid gating with post-selection
            // score renormalization. GLM-5.2 is presumed to continue
            // this lineage; not confirmed against GLM-5.2's own
            // config.json (unavailable in this environment).
            gating: GatingFunction::Sigmoid,
            norm_topk_prob: true,
         expert_group_count: None, expert_group_used_count: None,},
        // No evidence found (via ik_llama.cpp source or public
        // reporting) that GLM-5.2 skips MoE on any leading layers;
        // defaulting to 0 (every layer uses this model's MoE
        // topology) rather than assuming DeepSeek's convention
        // applies here too.
        n_dense_leading_layers: 0,
        rope_freqs: None,
        // Placeholder GQA path; real GLM-5.2 DSA uses interleaved RoPE
        // via `glm_dsa`/`mla`, not this preset's Decoder path.
        rope_layout: RopeLayout::Neox,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_pattern: None,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        embedding_scale: None,
        attention_scale: None,
        rope_theta_swa: None,
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &[
            "n_layers",
            "hidden_dim",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "rope_theta",
            "moe.expert_ffn_dim",
            "moe.n_shared_experts",
            "moe.gating (sigmoid assumed from GLM4-MoE-family convention found in ik_llama.cpp source, not confirmed for GLM-5.2 specifically)",
        ],
    }
}

/// DeepSeek V4 Pro **structural sketch only** — CSA/HCA is not on this
/// GQA `Decoder` path. Real primitives live under
/// `deepseek_v4_attention` / `hyper_connections` and are not assembled
/// into a served decoder yet. Hparams (~1.6T / ~49B active) are
/// placeholders for smoke/bench.
pub fn deepseek_v4_pro() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "deepseek-v4-pro",
        attention: AttentionKind::Gqa,
        n_layers: 96,
        hidden_dim: 7168,
        n_heads: 56,
        n_kv_heads: 8,
        head_dim: 128,
        vocab_size: 129280,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        moe: MoeLayerConfig {
            n_experts: 385,
            n_experts_active: 6,
            n_shared_experts: 1,
            hidden_dim: 7168,
            expert_ffn_dim: 2048,
            // Sigmoid, not softmax: this is the stronger-confidence of
            // the two sigmoid-gating corrections in this file.
            // DeepSeek-V3's own published technical report explicitly
            // documents computing per-expert affinity via sigmoid and
            // renormalizing only the selected experts' scores to sum
            // to one; reading ik_llama.cpp's real GGUF hparams-loading
            // source (llama-hparams.cpp, LLM_ARCH_DEEPSEEK2 case)
            // confirmed this is exactly what that code path defaults
            // to for the DeepSeek-2/3 lineage. DeepSeek V4 Pro is
            // presumed to continue using sigmoid gating for the same
            // reason; not confirmed against V4 Pro's own config.json.
            gating: GatingFunction::Sigmoid,
            norm_topk_prob: true,
         expert_group_count: None, expert_group_used_count: None,},
        // DeepSeek-V3's own published technical report documents the
        // first 3 transformer layers as dense (ordinary FFN, no
        // expert routing), with MoE starting from layer 4 onward;
        // ik_llama.cpp's real hparams-loading source
        // (LLM_KV_LEADING_DENSE_BLOCK_COUNT) confirms this is a real,
        // loaded GGUF metadata field for the DeepSeek-2/3 lineage.
        // DeepSeek V4 Pro is presumed to continue this convention;
        // not confirmed against V4 Pro's own config.json.
        n_dense_leading_layers: 3,
        rope_freqs: None,
        // llama.cpp maps LLM_ARCH_DEEPSEEK4 -> LLAMA_ROPE_TYPE_NORM.
        rope_layout: RopeLayout::Norm,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_pattern: None,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        embedding_scale: None,
        attention_scale: None,
        rope_theta_swa: None,
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &[
            "n_layers",
            "hidden_dim",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "moe.expert_ffn_dim",
            "attention_variant (CSA/HCA hybrid NOT implemented, GQA fallback in use)",
            "moe.gating (sqrtsoftplus: confirmed for real V4 in llama.cpp PR #24162; this preset still uses Sigmoid on the wrong GQA sketch path)",
            "n_dense_leading_layers (3: same confidence basis as gating above, DeepSeek-V3 technical report + ik_llama.cpp source, not confirmed for V4 Pro)",
        ],
    }
}

/// Kimi K3 **structural sketch only** for the generic GQA `Decoder`.
/// Real checkpoint work uses the dedicated Kimi stack (`kimi_loader` /
/// `KimiEngine`); slice-verified, not a full end-to-end run. Do not
/// treat this preset as a runnable Kimi substitute.
pub fn kimi_k3() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "kimi-k3",
        n_layers: 93,
        hidden_dim: 7168,
        // n_heads/n_kv_heads/head_dim describe the Gqa fallback
        // Decoder actually runs today, not Kimi K3's real attention
        // (see `attention` below) -- kept at reasonable stand-in
        // values (matching MLA's num_heads=96 and combined
        // qk_nope+qk_rope head dim) rather than deleted, so the
        // placeholder path stays runnable.
        n_heads: 96,
        n_kv_heads: 96,
        head_dim: 192,
        vocab_size: 163840,
        // Not present in the published text_config; RoPE only ever
        // applies to Gated MLA's 64-dim qk_rope_head_dim slice in the
        // real architecture, and Decoder doesn't implement that slicing
        // yet, so this remains an unconfirmed placeholder.
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-5,
        moe: MoeLayerConfig {
            n_experts: 896,
            n_experts_active: 16,
            n_shared_experts: 2,
            hidden_dim: 7168,
            expert_ffn_dim: 3072,
            // Confirmed directly from the real config.json:
            // "moe_router_activation_func": "sigmoid".
            gating: GatingFunction::Sigmoid,
            norm_topk_prob: true,
         expert_group_count: None, expert_group_used_count: None,},
        // Confirmed directly from the real config.json:
        // "first_k_dense_replace": 1.
        n_dense_leading_layers: 1,
        // Kimi K3's real, published attention topology (verified
        // against huggingface.co/moonshotai/Kimi-K3/config.json's
        // linear_attn_config block and the real KimiDeltaAttention /
        // KimiMLAAttention reference implementations in
        // modeling_kimi_linear.py) -- not yet wired into Decoder's
        // forward pass, which still runs the Gqa placeholder above for
        // every layer regardless of this field.
        attention: AttentionKind::KimiHybrid(KimiHybridAttention {
            kda_layers: vec![
                1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29,
                30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55,
                57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82,
                83, 85, 86, 87, 89, 90, 91,
            ],
            full_attn_layers: vec![
                4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84,
                88, 92, 93,
            ],
            mla: MlaConfig {
                num_heads: 96,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                use_output_gate: true,
                // Real, confirmed: Kimi K3's `KimiMLAAttention.forward`
                // never rotates -- see `MlaConfig::rope`'s doc comment.
                rope: None,
            },
            kda: KdaConfig {
                num_heads: 96,
                head_dim: 128,
                short_conv_kernel_size: 4,
                gate_lower_bound: -5.0,
                use_full_rank_gate: true,
            },
        }),
        rope_freqs: None,
        // GQA placeholder path only; real Kimi attention is rope-less MLA
        // or KDA and never reaches Decoder::apply_rope_head.
        rope_layout: RopeLayout::Neox,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_pattern: None,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        embedding_scale: None,
        attention_scale: None,
        rope_theta_swa: None,
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &[
            "n_heads/n_kv_heads/head_dim (describe the unimplemented Gqa placeholder, not Kimi K3's real MLA/KDA attention -- see `attention` field)",
            "rope_theta (not present in the published config; real architecture only applies RoPE to Gated MLA's qk_rope_head_dim slice, which Decoder doesn't implement)",
            "entire preset beyond hyperparameters (the real 2.8T-parameter checkpoint has not been run end to end; only real slices have, via the dedicated kimi_decoder/kimi_loader stack -- see docs/MODELS.md)",
        ],
    }
}

/// Matches the generated on-disk fixture exactly (hidden_dim, head
/// counts, ffn_dim, vocab, rope_theta, eps).
/// Used by `ferrox inspect-run` and the cross-validation test in
/// `crates/ferrox-models/tests/gguf_roundtrip.rs` to prove the real
/// GGUF loader + forward pass produce the same numbers as an
/// independent NumPy reference implementation reading the same file.
pub fn test_dense_fixture() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "ferrox-test-dense",
        attention: AttentionKind::Gqa,
        n_layers: 2,
        hidden_dim: 32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        vocab_size: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        moe: MoeLayerConfig {
            n_experts: 1,
            n_experts_active: 1,
            n_shared_experts: 0,
            hidden_dim: 32,
            expert_ffn_dim: 32,
            gating: GatingFunction::Softmax,
            norm_topk_prob: true,
            expert_group_count: None,
            expert_group_used_count: None,
        },
        n_dense_leading_layers: 0,
        rope_freqs: None,
        // Matches the independent reference's split-half apply_rope.
        rope_layout: RopeLayout::Neox,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_pattern: None,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        embedding_scale: None,
        attention_scale: None,
        rope_theta_swa: None,
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &["this is a synthetic test fixture, not a real model"],
    }
}

/// Matches the generated on-disk multi-expert MoE fixture: 4 experts,
/// top-2 routing, 1 shared
/// expert, packed 3D expert tensors. Used to verify the previously-
/// untested multi-expert loading path (`split_expert_tensor` in
/// `ferrox-models::loader`) against a real file, the same way
/// `test_dense_fixture` verifies the single-expert path.
pub fn test_moe_fixture() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "ferrox-test-moe",
        attention: AttentionKind::Gqa,
        n_layers: 2,
        hidden_dim: 32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        vocab_size: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        moe: MoeLayerConfig {
            n_experts: 4,
            n_experts_active: 2,
            n_shared_experts: 1,
            hidden_dim: 32,
            expert_ffn_dim: 32,
            gating: GatingFunction::Softmax,
            norm_topk_prob: true,
            expert_group_count: None,
            expert_group_used_count: None,
        },
        n_dense_leading_layers: 0,
        rope_freqs: None,
        rope_layout: RopeLayout::Neox,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_pattern: None,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        embedding_scale: None,
        attention_scale: None,
        rope_theta_swa: None,
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &["this is a synthetic multi-expert test fixture, not a real model"],
    }
}

/// Matches the generated on-disk mixed-topology fixture: 3 layers, the
/// first of which is
/// an ordinary dense FFN and the remaining two are genuine MoE (3
/// experts, top-1 routing, 1 shared expert each). Used to verify the
/// "leading dense layers" loading path
/// (`ModelConfig::layer_is_dense`) against a real file -- the pattern
/// found in DeepSeek-2/3-family models via ik_llama.cpp's source
/// (`LLM_KV_LEADING_DENSE_BLOCK_COUNT`), which was previously only
/// documented, not implemented or tested.
pub fn test_mixed_fixture() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "ferrox-test-mixed",
        attention: AttentionKind::Gqa,
        n_layers: 3,
        hidden_dim: 32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        vocab_size: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        moe: MoeLayerConfig {
            n_experts: 3,
            n_experts_active: 1,
            n_shared_experts: 1,
            hidden_dim: 32,
            expert_ffn_dim: 32,
            gating: GatingFunction::Softmax,
            norm_topk_prob: true,
            expert_group_count: None,
            expert_group_used_count: None,
        },
        n_dense_leading_layers: 1,
        rope_freqs: None,
        rope_layout: RopeLayout::Neox,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_pattern: None,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        embedding_scale: None,
        attention_scale: None,
        rope_theta_swa: None,
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &["this is a synthetic mixed dense/MoE test fixture, not a real model"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_layout_for_gguf_architecture_matches_llama_cpp() {
        // Confirmed against llama.cpp's llama_model_rope_type
        // (src/llama-model.cpp): llama -> NORM, olmoe/qwen2/phi3/gemma -> NEOX.
        assert_eq!(RopeLayout::for_gguf_architecture("llama"), RopeLayout::Norm);
        assert_eq!(
            RopeLayout::for_gguf_architecture("llama4"),
            RopeLayout::Norm
        );
        assert_eq!(
            RopeLayout::for_gguf_architecture("deepseek2"),
            RopeLayout::Norm
        );
        assert_eq!(RopeLayout::for_gguf_architecture("olmoe"), RopeLayout::Neox);
        assert_eq!(RopeLayout::for_gguf_architecture("qwen2"), RopeLayout::Neox);
        assert_eq!(
            RopeLayout::for_gguf_architecture("qwen2moe"),
            RopeLayout::Neox
        );
        assert_eq!(RopeLayout::for_gguf_architecture("qwen3"), RopeLayout::Neox);
        assert_eq!(RopeLayout::for_gguf_architecture("phi3"), RopeLayout::Neox);
        assert_eq!(
            RopeLayout::for_gguf_architecture("gemma3"),
            RopeLayout::Neox
        );
        // Unknown architectures keep the historical Neox default at this
        // helper only; load-time uses capability::resolve_architecture and
        // fails closed instead of guessing.
        assert_eq!(
            RopeLayout::for_gguf_architecture("totally-unknown-arch"),
            RopeLayout::Neox
        );
    }

    #[test]
    fn all_presets_have_consistent_moe_hidden_dim() {
        for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
            assert_eq!(
                cfg.hidden_dim, cfg.moe.hidden_dim,
                "{}: attention hidden_dim and MoE hidden_dim must match",
                cfg.name
            );
        }
    }

    #[test]
    fn all_presets_route_fewer_experts_than_total() {
        for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
            assert!(
                cfg.moe.n_experts_active < cfg.moe.n_experts,
                "{}: active experts must be a sparse subset of total experts",
                cfg.name
            );
        }
    }

    #[test]
    fn all_presets_have_divisible_heads() {
        for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
            assert_eq!(
                cfg.n_heads % cfg.n_kv_heads,
                0,
                "{}: n_heads must be a multiple of n_kv_heads for GQA grouping",
                cfg.name
            );
        }
    }

    #[test]
    fn every_preset_declares_its_uncertain_fields() {
        // This is a documentation-honesty test: any preset with zero
        // best_effort_fields would be silently overclaiming precision
        // we don't have. Fail loudly if that ever happens.
        for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
            assert!(
                !cfg.best_effort_fields.is_empty(),
                "{}: must disclose which fields are unconfirmed estimates",
                cfg.name
            );
        }
    }

    /// Kimi K3's `kda_layers`/`full_attn_layers` were transcribed by
    /// hand from the real published config.json; this test guards
    /// against a transcription slip (duplicate, out-of-range, or
    /// missing layer index) rather than trusting the transcription.
    #[test]
    fn kimi_k3_hybrid_attention_layers_partition_every_layer_exactly_once() {
        let cfg = kimi_k3();
        let AttentionKind::KimiHybrid(hybrid) = &cfg.attention else {
            panic!("kimi_k3() must use AttentionKind::KimiHybrid");
        };

        let mut seen = std::collections::HashSet::new();
        for &l in hybrid
            .kda_layers
            .iter()
            .chain(hybrid.full_attn_layers.iter())
        {
            assert!(
                (1..=cfg.n_layers).contains(&l),
                "layer {l} is out of the published 1..={} range",
                cfg.n_layers
            );
            assert!(
                seen.insert(l),
                "layer {l} appears in both/either list twice"
            );
        }
        // Dense-vs-MoE (n_dense_leading_layers) and attention-type
        // (KDA vs Gated MLA) are independent per-layer properties in
        // the real config -- e.g. layer 1 is both the sole dense
        // leading layer *and* a KDA layer -- so every one of the 93
        // layers, dense or not, is covered by exactly one of these two
        // lists (confirmed: 69 + 24 == 93, not 93 - 1).
        assert_eq!(
            hybrid.kda_layers.len() + hybrid.full_attn_layers.len(),
            cfg.n_layers,
            "every layer must be assigned exactly one of KDA or Gated MLA"
        );
        assert_eq!(
            hybrid.kda_layers.len(),
            69,
            "expected 69 KDA layers per the published config"
        );
        assert_eq!(
            hybrid.full_attn_layers.len(),
            24,
            "expected 24 Gated MLA layers per the published config"
        );
    }

    #[test]
    fn layer_attention_kind_is_gqa_for_every_layer_of_a_gqa_model() {
        let cfg = glm_5_2();
        for l in 0..cfg.n_layers {
            assert_eq!(cfg.layer_attention_kind(l), LayerAttentionKind::Gqa);
        }
    }

    #[test]
    fn layer_attention_kind_classifies_every_kimi_k3_layer_without_panicking() {
        let cfg = kimi_k3();
        let AttentionKind::KimiHybrid(hybrid) = &cfg.attention else {
            panic!("kimi_k3() must use AttentionKind::KimiHybrid");
        };
        for l in 0..cfg.n_layers {
            let kind = cfg.layer_attention_kind(l);
            let one_indexed = l + 1;
            if hybrid.kda_layers.contains(&one_indexed) {
                assert_eq!(kind, LayerAttentionKind::KimiKda);
            } else {
                assert_eq!(kind, LayerAttentionKind::KimiMla);
            }
        }
    }

    #[test]
    fn layer_attention_kind_matches_the_real_published_layer_1_and_4() {
        // Layer 1 (1-indexed, so index 0 here) is published as KDA;
        // layer 4 (index 3) is published as the first Gated MLA layer.
        let cfg = kimi_k3();
        assert_eq!(cfg.layer_attention_kind(0), LayerAttentionKind::KimiKda);
        assert_eq!(cfg.layer_attention_kind(3), LayerAttentionKind::KimiMla);
    }

    #[test]
    fn kimi_k3_mla_q_head_dim_matches_gqa_placeholder_head_dim() {
        // The Gqa-placeholder head_dim above is deliberately set to
        // Gated MLA's combined q_head_dim (qk_nope + qk_rope) so the
        // placeholder path at least reflects a real dimension from the
        // published config rather than an arbitrary guess.
        let cfg = kimi_k3();
        let AttentionKind::KimiHybrid(hybrid) = &cfg.attention else {
            panic!("kimi_k3() must use AttentionKind::KimiHybrid");
        };
        assert_eq!(
            cfg.head_dim,
            hybrid.mla.qk_nope_head_dim + hybrid.mla.qk_rope_head_dim
        );
    }

    #[test]
    fn approx_active_params_is_nonzero_and_finite_order_of_magnitude() {
        for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
            let approx = cfg.approx_active_params_per_token();
            // Sanity band: active params/token for these models is
            // reported in the tens of billions; this is a loose
            // order-of-magnitude check (1e9 to 1e12), not a precise
            // parameter-count reproduction.
            assert!(
                approx > 1_000_000_000 && approx < 1_000_000_000_000,
                "{}: approx_active_params_per_token={approx} is outside a plausible range",
                cfg.name
            );
        }
    }
}
