//! Pre-load KV budget arithmetic: answer "will this fit" *before*
//! allocating anything, from terms that are all exact in the GGUF
//! header.
//!
//! ```text
//! weights + n_ctx * per_token_kv + activation_headroom  <=  device_budget
//! per_token_kv = n_layers * n_kv_heads * head_dim * bytes_per_elem * 2
//! ```
//!
//! Everything here is a pure function of a shape plus a byte budget --
//! no I/O, no device handles, no allocation -- so the arithmetic can be
//! unit-tested against hand-computed numbers. The device side (how many
//! bytes a backend actually offers) lives in
//! [`crate::device_budget`]; the whole-checkpoint report that consumes
//! both is [`crate::residency_report`].
//!
//! # Where this is approximate, stated up front
//!
//! - **Weights.** ferrox mmaps quantized tensors and reads them in
//!   place, so "weights resident" is not a number ferrox controls: the
//!   kernel can evict those pages under pressure and fault them back in
//!   later. `weights_bytes` is therefore the *checkpoint's* byte count,
//!   an upper bound on resident cost and a lower bound on the I/O the
//!   run will do -- not a measurement of RSS. A model can exceed this
//!   budget and still run (slowly, page-faulting), and it can fit this
//!   budget and still be killed by something else on the machine.
//! - **Activations.** `activation_headroom_bytes` is a caller-supplied
//!   reserve, not a derived quantity. Nothing here models scratch
//!   buffers, the logits vector, tokenizer state or allocator slack.
//! - **KV element width.** [`KvElem`] is the width of the store that
//!   the *selected backend* keeps. With Metal attention on, the device
//!   holds an f16 KV while the host may still hold an f32 mirror
//!   (`FERROX_CPU_KV_OFFLOAD`); budget the tier you are checking
//!   against, and do not assume the two add up to one number.
//!
//! A conservative, explainable number beats a clever one: none of this
//! tries to track real resident bytes over time.
//!
//! # Why a sliding window is not a saving here
//!
//! This module used to cap sliding-window layers at `window + chunk - 1`
//! positions and subtract them out of the divisor, which made a
//! Gemma-3-4B context look 5.8x cheaper than it is and gpt-oss 2x. **No
//! KV store ferrox allocates ever gave that cap back** (#33):
//!
//! - `ferrox_core::cache::KvCache` has no window concept at all. `push`
//!   extends `k`/`v` for every position, so a plain or pool-backed cache
//!   holds the whole sequence in every layer. This is what the CLI
//!   allocates and what the server allocates on its non-paged paths.
//! - The paged store *can* recycle pages behind a window, but only for a
//!   model whose every layer shares one window
//!   (`ModelConfig::uniform_sliding_window`, `None` by design for the
//!   alternating models -- gpt-oss, Gemma-2/3 -- because a page group
//!   holds one block per layer and the full-attention layers still read
//!   position 0). Even there it recycles only the GENERATION tail: its
//!   own admission arithmetic (`ferrox_server::generate::
//!   paged_hold_positions`) holds `prompt + bound + a page`, and a
//!   budget priced in *context length* has to survive a prompt that
//!   fills that context.
//!
//! So the budget prices every layer at every position, for every model.
//! That is exactly what the two `KvCache` stores allocate and an upper
//! bound on what the paged store reserves, which is the direction that
//! matters: an over-estimate costs context, an under-estimate is
//! admitted and then arrives as an OOM instead of the refusal this
//! engine exists to give.
//!
//! There is deliberately no window field left to fill in. Making a
//! store actually evict is real work and is tracked as #61 (per-layer
//! page groups, and eviction inside the prompt region); when one does,
//! the number it keeps belongs to the STORE, and this module should take
//! it from there rather than restate a rule the store does not follow.

use crate::config::ModelConfig;

/// Element width of one cached K/V scalar, per backend store.
///
/// The block-quantized variants are the ggml/TurboQuant wire formats
/// `ferrox-metal` writes for `FERROX_CTK` (see
/// `ferrox_metal::attn::MetalKvDtype`), so their cost is per 32-element
/// block, not per scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvElem {
    /// Host `ferrox_core::cache::KvCache`, which stores `Vec<f32>`.
    F32,
    /// Metal device KV default (`FERROX_CTK=f16`, llama.cpp `-ctk f16`).
    F16,
    /// ggml Q8_0 wire: 32 elems -> 2-byte scale + 32 int8 = 34 bytes.
    /// `FERROX_CTK=q8_0|turbo8|fp8` all land on this width.
    Q8_0,
    /// TurboQuant 4-bit: 32 elems -> 2-byte scale + 16 nibble bytes.
    Turbo4,
}

impl KvElem {
    /// Bytes needed to store `elems` cached scalars, rounding up to a
    /// whole block for the block-quantized wires (a partial block still
    /// costs a full one).
    ///
    /// Saturating rather than wrapping or panicking. This is a
    /// REPORTING number: it exists to put bytes in a refusal message,
    /// and it is reached with position counts that came off an HTTP
    /// body. `max_tokens: u64::MAX / 64` does not overflow the position
    /// sum, so it reaches here and multiplied past `u64::MAX`, panicking
    /// the request thread while computing the text of the very refusal
    /// that was about to reject it (#36).
    ///
    /// Saturating is right HERE and wrong for a bound. A saturated byte
    /// count still reports "astronomically large", which is the only
    /// thing the message needs to convey. A saturated position bound
    /// would silently turn a nonsense request into a plausible one and
    /// serve it.
    pub fn bytes_for(self, elems: u64) -> u64 {
        match self {
            KvElem::F32 => elems.saturating_mul(4),
            KvElem::F16 => elems.saturating_mul(2),
            KvElem::Q8_0 => {
                let blocks = elems.div_ceil(ferrox_quant::Q8_0_BLOCK_ELEMS as u64);
                blocks.saturating_mul(ferrox_quant::Q8_0_BLOCK_BYTES as u64)
            }
            KvElem::Turbo4 => {
                let blocks = elems.div_ceil(ferrox_quant::TURBO4_KV_GROUP as u64);
                blocks.saturating_mul(ferrox_quant::TURBO4_KV_BLOCK_BYTES as u64)
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            KvElem::F32 => "f32",
            KvElem::F16 => "f16",
            KvElem::Q8_0 => "q8_0",
            KvElem::Turbo4 => "turbo4",
        }
    }

    /// Maps a `FERROX_CTK` / `--ctk` value onto the width the Metal KV
    /// store really keeps. Mirrors
    /// `ferrox_metal::attn::effective_metal_kv_dtype`: `turbo8` and
    /// `fp8` share Q8_0's 34-byte wire, and anything unrecognised or
    /// unimplemented (`turbo3`) falls back to f16 rather than being
    /// budgeted at a width no kernel writes.
    ///
    /// Note this does *not* check the block alignment that function
    /// also checks (`n_kv_heads * head_dim` divisible by 32), so a
    /// misaligned shape is budgeted at the requested width while the
    /// runtime silently uses f16 -- an under-estimate, called out here
    /// rather than papered over.
    pub fn from_ctk(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            // llama.cpp's `-ctk f32`, and the width of ferrox's own
            // host `KvCache`.
            "f32" => KvElem::F32,
            "q8_0" | "turbo8" | "fp8" => KvElem::Q8_0,
            "turbo4" => KvElem::Turbo4,
            _ => KvElem::F16,
        }
    }
}

/// How one layer's KV cache is shaped. Which variant applies is a
/// property of the *decoder that will run*, not of the architecture
/// name -- see [`KvLayout::MlaLatent`]'s doc comment for the one place
/// that distinction bites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    /// Multi-head / grouped-query attention: one K vector and one V
    /// vector of `n_kv_heads * head_dim` per token, per layer. MHA is
    /// just the `n_kv_heads == n_heads` case -- there is no separate
    /// variant for it, and the halving GQA buys shows up entirely in
    /// `n_kv_heads`.
    Gqa { n_kv_heads: usize, head_dim: usize },
    /// MLA in its *absorbed* form: the cache holds only the compressed
    /// latent plus the decoupled RoPE slice, `kv_lora_rank + rope_dim`
    /// scalars per token per layer, and K/V are reconstructed from it
    /// on the fly. One vector, not two -- there is no `* 2` here.
    ///
    /// **ferrox does not run this form today.** `mla::mla_forward_token`
    /// (and therefore `kimi_decoder`, `glm_dsa`, `glm52_decoder`)
    /// caches the *expanded* per-head K and V, so a real ferrox MLA run
    /// costs [`KvLayout::MlaExpanded`]. This variant is what the
    /// absorbed form would cost, and is the right number to plan
    /// against only once a decoder actually caches the latent.
    MlaLatent {
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
    },
    /// MLA as ferrox actually caches it: per-head K of
    /// `qk_nope_head_dim + qk_rope_head_dim` and per-head V of
    /// `v_head_dim`, both materialised (`mla::mla_forward_token`'s
    /// `k_cache`/`v_cache`). K and V head dims differ, which is exactly
    /// why this cannot reuse the `Gqa` arm.
    MlaExpanded {
        n_heads: usize,
        k_head_dim: usize,
        v_head_dim: usize,
    },
}

impl KvLayout {
    /// Cached scalars one token contributes to one layer.
    pub fn elems_per_token_per_layer(self) -> u64 {
        match self {
            KvLayout::Gqa {
                n_kv_heads,
                head_dim,
            } => 2 * n_kv_heads as u64 * head_dim as u64,
            KvLayout::MlaLatent {
                kv_lora_rank,
                qk_rope_head_dim,
            } => kv_lora_rank as u64 + qk_rope_head_dim as u64,
            KvLayout::MlaExpanded {
                n_heads,
                k_head_dim,
                v_head_dim,
            } => n_heads as u64 * (k_head_dim as u64 + v_head_dim as u64),
        }
    }

    /// One-line description of the arithmetic, for the report a user
    /// reads when they want to know why they got the context they got.
    pub fn describe(self) -> String {
        match self {
            KvLayout::Gqa {
                n_kv_heads,
                head_dim,
            } => format!("2 (K+V) x {n_kv_heads} kv-heads x {head_dim} head-dim"),
            KvLayout::MlaLatent {
                kv_lora_rank,
                qk_rope_head_dim,
            } => format!(
                "MLA latent: {kv_lora_rank} kv_lora_rank + {qk_rope_head_dim} rope-dim \
                 (one vector, no K/V doubling)"
            ),
            KvLayout::MlaExpanded {
                n_heads,
                k_head_dim,
                v_head_dim,
            } => format!(
                "MLA expanded: {n_heads} heads x ({k_head_dim} K head-dim + \
                 {v_head_dim} V head-dim)"
            ),
        }
    }
}

/// The KV shape of a whole model: enough to price any context length.
///
/// Every layer keeps every position. That is a statement about the
/// STORES this engine allocates, not about the architectures it runs --
/// see the module doc's "Why a sliding window is not a saving here", and
/// the test that measures a real `ferrox_core::cache::KvCache` rather
/// than restating this multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShape {
    pub n_layers: usize,
    pub layout: KvLayout,
    pub elem: KvElem,
}

impl KvShape {
    /// Reads the shape off a config.
    ///
    /// `config.sliding_window` / `config.swa_pattern` are deliberately
    /// NOT read: they describe what attention *reads*, and this module
    /// prices what the store *keeps*. Nothing here evicts (#33), so a
    /// windowed layer costs exactly what a full-attention one does.
    ///
    /// Always produces a [`KvLayout::Gqa`] layout, because
    /// `ModelConfig` describes the generic GQA decoder -- the MLA
    /// stacks carry their own hyperparameters (`Deepseek2Hparams`,
    /// `MlaConfig`) and should build their shape with
    /// [`KvShape::mla_expanded`].
    pub fn from_config(config: &ModelConfig, elem: KvElem) -> Self {
        KvShape {
            n_layers: config.n_layers,
            layout: KvLayout::Gqa {
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
            },
            elem,
        }
    }

    /// The shape a ferrox MLA decoder really allocates -- see
    /// [`KvLayout::MlaExpanded`].
    pub fn mla_expanded(
        n_layers: usize,
        n_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        elem: KvElem,
    ) -> Self {
        KvShape {
            n_layers,
            layout: KvLayout::MlaExpanded {
                n_heads,
                k_head_dim: qk_nope_head_dim + qk_rope_head_dim,
                v_head_dim,
            },
            elem,
        }
    }

    /// The plan's headline number, and the only per-token number there
    /// is: bytes one token costs across every layer. Exact for f32/f16;
    /// for the block-quantized wires it is exact whenever a layer's
    /// per-token element count is a multiple of the 32-element block
    /// (true for every real head-dim/kv-head combination), and rounds
    /// up otherwise.
    ///
    /// This is also the divisor [`KvBudget::max_context`] uses. There is
    /// no separate "marginal" number any more: a marginal cost below the
    /// per-token cost would mean some layer stops growing, and none
    /// does.
    pub fn per_token_kv_bytes(&self) -> u64 {
        (self.n_layers as u64)
            .saturating_mul(self.elem.bytes_for(self.layout.elems_per_token_per_layer()))
    }

    /// Bytes one request's KV costs at `tokens` of context.
    pub fn kv_bytes_for_tokens(&self, tokens: usize) -> u64 {
        // Every multiplication here saturates, for the reason on
        // `KvElem::bytes_for`: `tokens` can arrive from an HTTP body.
        let per_layer = self.layout.elems_per_token_per_layer();
        (self.n_layers as u64)
            .saturating_mul(self.elem.bytes_for(per_layer.saturating_mul(tokens as u64)))
    }

    /// The sentence a user should be able to read and reproduce with a
    /// calculator.
    pub fn describe(&self) -> String {
        format!(
            "{} layers x [{}] x {} = {} bytes/token",
            self.n_layers,
            self.layout.describe(),
            self.elem.as_str(),
            self.per_token_kv_bytes()
        )
    }
}

/// Which ceiling a rejection hit. The point of naming it is that the
/// two send an operator to different knobs: `ContextLength` is the
/// request's fault and shrinking the prompt fixes it, `DeviceMemory`
/// is the machine's and only a smaller model / smaller `n_ctx` /
/// bigger box does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ceiling {
    /// The request asked for more context than this deployment admitted.
    ContextLength,
    /// weights + KV + headroom does not fit the backend's budget.
    DeviceMemory,
}

impl Ceiling {
    /// Stable machine-readable code, safe to match on in a client.
    pub fn code(self) -> &'static str {
        match self {
            Ceiling::ContextLength => "context_length_exceeded",
            Ceiling::DeviceMemory => "device_memory_budget_exceeded",
        }
    }
}

/// A structured refusal: what it would have cost, what the ceiling was,
/// and which ceiling. Deliberately *not* an allocation failure -- the
/// whole point of computing this before the load is that nobody has to
/// read an OOM to find out.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{code}: {detail} (estimated {estimated_bytes} bytes vs limit {limit_bytes} bytes)",
        code = self.binding.code())]
pub struct KvBudgetError {
    pub binding: Ceiling,
    pub estimated_bytes: u64,
    pub limit_bytes: u64,
    pub detail: String,
}

impl KvBudgetError {
    pub fn code(&self) -> &'static str {
        self.binding.code()
    }

    /// Bytes over the ceiling (saturating, so a fit reads as `0`).
    pub fn overage_bytes(&self) -> u64 {
        self.estimated_bytes.saturating_sub(self.limit_bytes)
    }
}

/// A priced plan: every term of the inequality, kept separately so the
/// report can show the arithmetic rather than just the verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBudget {
    /// Checkpoint bytes. See the module doc on why this is an
    /// approximation for mmap'd weights.
    pub weights_bytes: u64,
    /// Caller-supplied reserve for activations/scratch/allocator slack.
    pub activation_headroom_bytes: u64,
    /// What the backend says it can give us (see
    /// [`crate::device_budget::DeviceBudget::usable_bytes`]).
    pub device_budget_bytes: u64,
    pub shape: KvShape,
    /// KV caches are per request; concurrency multiplies them.
    pub concurrent_requests: usize,
}

impl KvBudget {
    /// Bytes left for KV after weights and headroom, or `0` when those
    /// two alone already overflow the budget.
    pub fn kv_bytes_available(&self) -> u64 {
        self.device_budget_bytes
            .saturating_sub(self.weights_bytes)
            .saturating_sub(self.activation_headroom_bytes)
    }

    /// Total estimated resident bytes at `tokens` of context.
    pub fn estimated_bytes(&self, tokens: usize) -> u64 {
        self.weights_bytes
            + self.activation_headroom_bytes
            + self.shape.kv_bytes_for_tokens(tokens) * self.concurrent_requests.max(1) as u64
    }

    /// The one-line check the plan is named for. `Ok` carries the
    /// estimate so a caller can log it on the happy path too.
    pub fn check(&self, tokens: usize) -> Result<u64, KvBudgetError> {
        let estimated = self.estimated_bytes(tokens);
        if estimated <= self.device_budget_bytes {
            return Ok(estimated);
        }
        Err(KvBudgetError {
            binding: Ceiling::DeviceMemory,
            estimated_bytes: estimated,
            limit_bytes: self.device_budget_bytes,
            detail: format!(
                "{} weight bytes + {} KV bytes at {tokens} tokens x{} concurrent + {} \
                 activation headroom exceeds the {} byte device budget",
                self.weights_bytes,
                self.shape.kv_bytes_for_tokens(tokens) * self.concurrent_requests.max(1) as u64,
                self.concurrent_requests.max(1),
                self.activation_headroom_bytes,
                self.device_budget_bytes,
            ),
        })
    }

    /// Largest context that fits, closed form:
    /// `(budget - weights - headroom) / (per_token_kv * concurrency)`,
    /// floored to `granularity` and clamped to `cap` (the model's own
    /// trained context length).
    ///
    /// Every layer is in the divisor. A sliding-window model used to
    /// have its windowed layers subtracted out of it and added back as a
    /// saturated constant, which is the #33 under-estimate: nothing
    /// evicts, so nothing saturates.
    pub fn max_context(&self, cap: usize, granularity: usize) -> ContextFit {
        let granularity = granularity.max(1);
        let concurrency = self.concurrent_requests.max(1) as u64;
        let available = self.kv_bytes_available();
        let per_token = self.shape.per_token_kv_bytes().saturating_mul(concurrency);

        let (tokens, capped_by) = if available == 0 {
            (0, ContextCap::DeviceBudget)
        } else {
            // `checked_div` rather than a `per_token == 0` guard around
            // a bare `/`: a model with no KV at all (no layers, or a
            // zero-width layout) is not an error here, it is just
            // unbounded by memory, and expressing it as `None` keeps
            // that meaning in one place instead of splitting it across
            // a check and a division that clippy then has to
            // re-associate.
            match available.checked_div(per_token) {
                None => (cap, ContextCap::ModelContextLength),
                Some(raw) => {
                    let raw = raw as usize;
                    // Flooring must never turn a real answer into
                    // "nothing fits": under one granularity step,
                    // report the exact number of tokens rather than
                    // rounding it away.
                    let floored = if raw >= granularity {
                        (raw / granularity) * granularity
                    } else {
                        raw
                    };
                    if floored >= cap {
                        (cap, ContextCap::ModelContextLength)
                    } else {
                        (floored, ContextCap::DeviceBudget)
                    }
                }
            }
        };

        ContextFit {
            tokens,
            cap,
            granularity,
            capped_by,
            kv_available_bytes: available,
            per_token_kv_bytes: self.shape.per_token_kv_bytes(),
            concurrent_requests: concurrency as usize,
            kv_bytes: self.shape.kv_bytes_for_tokens(tokens) * concurrency,
            weights_bytes: self.weights_bytes,
            activation_headroom_bytes: self.activation_headroom_bytes,
            device_budget_bytes: self.device_budget_bytes,
        }
    }
}

/// Why `--ctx auto` chose the number it chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextCap {
    /// The model's own trained context length was the smaller ceiling.
    ModelContextLength,
    /// Memory ran out first.
    DeviceBudget,
}

/// The answer `--ctx auto` produces, with every term that went into it
/// so the user can check the division by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextFit {
    pub tokens: usize,
    pub cap: usize,
    pub granularity: usize,
    pub capped_by: ContextCap,
    pub kv_available_bytes: u64,
    /// The divisor: bytes one token of context costs across every layer.
    pub per_token_kv_bytes: u64,
    pub concurrent_requests: usize,
    pub kv_bytes: u64,
    pub weights_bytes: u64,
    pub activation_headroom_bytes: u64,
    pub device_budget_bytes: u64,
}

impl std::fmt::Display for ContextFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ctx auto = {} tokens ({}): ({} device budget - {} weights - {} activation headroom) \
             = {} for KV; / {} bytes/token/request / {} request(s) -> rounded down to a multiple \
             of {} (reported exactly below one step), capped at the model's {} trained context. \
             KV at the chosen context: {} bytes.",
            self.tokens,
            match self.capped_by {
                ContextCap::ModelContextLength => "limited by the model's context length",
                ContextCap::DeviceBudget => "limited by the device memory budget",
            },
            self.device_budget_bytes,
            self.weights_bytes,
            self.activation_headroom_bytes,
            self.kv_available_bytes,
            self.per_token_kv_bytes,
            self.concurrent_requests,
            self.granularity,
            self.cap,
            self.kv_bytes,
        )
    }
}

/// Granularity `--ctx auto` floors to. Small enough that the rounding
/// never costs a meaningful amount of context, round enough that the
/// reported number looks chosen rather than computed.
pub const CTX_AUTO_GRANULARITY: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    /// Llama-3.1-8B's real shape: 32 layers, 8 kv-heads (GQA 4:1),
    /// head_dim 128. llama.cpp reports 1 MiB/token at f32 for exactly
    /// this model, which is the number reproduced here by hand:
    /// 32 * 2 * 8 * 128 * 4 = 262144 bytes.
    fn llama31_8b() -> KvShape {
        KvShape {
            n_layers: 32,
            layout: KvLayout::Gqa {
                n_kv_heads: 8,
                head_dim: 128,
            },
            elem: KvElem::F32,
        }
    }

    #[test]
    fn gqa_per_token_kv_matches_the_hand_computed_byte_count() {
        let shape = llama31_8b();
        assert_eq!(shape.layout.elems_per_token_per_layer(), 2 * 8 * 128);
        assert_eq!(shape.per_token_kv_bytes(), 32 * 2 * 8 * 128 * 4);
        assert_eq!(shape.per_token_kv_bytes(), 262_144);
        // f16 is exactly half; a block-quantized store is 34/32 of the
        // element count, not 1 byte flat.
        assert_eq!(
            KvShape {
                elem: KvElem::F16,
                ..shape
            }
            .per_token_kv_bytes(),
            131_072
        );
        assert_eq!(
            KvShape {
                elem: KvElem::Q8_0,
                ..shape
            }
            .per_token_kv_bytes(),
            32 * (2 * 8 * 128 / 32) * 34
        );
        assert_eq!(
            KvShape {
                elem: KvElem::Turbo4,
                ..shape
            }
            .per_token_kv_bytes(),
            32 * (2 * 8 * 128 / 32) * 18
        );
    }

    #[test]
    fn ctk_names_map_onto_the_widths_metal_really_writes() {
        assert_eq!(KvElem::from_ctk("f16"), KvElem::F16);
        assert_eq!(KvElem::from_ctk("f32"), KvElem::F32);
        assert_eq!(KvElem::from_ctk("Q8_0"), KvElem::Q8_0);
        // turbo8 and fp8 share Q8_0's wire, per MetalKvDtype.
        assert_eq!(KvElem::from_ctk("turbo8"), KvElem::Q8_0);
        assert_eq!(KvElem::from_ctk("fp8"), KvElem::Q8_0);
        assert_eq!(KvElem::from_ctk("turbo4"), KvElem::Turbo4);
        // turbo3 is unimplemented and falls back to f16, as does junk.
        assert_eq!(KvElem::from_ctk("turbo3"), KvElem::F16);
        assert_eq!(KvElem::from_ctk("  nonsense "), KvElem::F16);
    }

    #[test]
    fn mha_costs_exactly_the_gqa_ratio_more_than_gqa() {
        // Same model with n_kv_heads == n_heads (32) instead of 8: MHA
        // is 4x the KV of 4:1 GQA, and nothing else changes.
        let gqa = llama31_8b();
        let mha = KvShape {
            layout: KvLayout::Gqa {
                n_kv_heads: 32,
                head_dim: 128,
            },
            ..gqa
        };
        assert_eq!(mha.per_token_kv_bytes(), 4 * gqa.per_token_kv_bytes());
        assert_eq!(mha.per_token_kv_bytes(), 32 * 2 * 32 * 128 * 4);
    }

    /// A small alternating-SWA config: 6 layers, every 3rd of them full
    /// attention, a 4-position window. Small enough that a test can
    /// allocate the real stores; alternating, which is the case
    /// `ModelConfig::uniform_sliding_window` refuses to let any store
    /// recycle.
    fn alternating_swa_config() -> ModelConfig {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.n_layers = 6;
        cfg.n_kv_heads = 1;
        cfg.head_dim = 8;
        cfg.sliding_window = Some(4);
        cfg.swa_pattern = Some(3);
        cfg
    }

    /// **The property this module got wrong, measured rather than
    /// restated.**
    ///
    /// The old budget capped a sliding layer at `window + chunk - 1`
    /// positions, but `ferrox_core::cache::KvCache` -- the store the CLI
    /// allocates and the store the server allocates on every non-paged
    /// path -- has no window concept: `push` extends `k`/`v` for every
    /// position, in every layer. So the budget under-priced gpt-oss by
    /// 2x and Gemma-3-4B by 5.8x, `-c auto` approved a context that did
    /// not fit, and the failure arrived as an OOM instead of a refusal
    /// (#33).
    ///
    /// This pushes real positions into the real caches and compares the
    /// bytes they hold against the budget's number. Recomputing the
    /// budget's own multiplication here would assert nothing: the code
    /// was not wrong about arithmetic, it was wrong about the world.
    #[test]
    fn the_budget_prices_exactly_what_the_kv_store_allocates_for_an_alternating_swa_model() {
        let cfg = alternating_swa_config();
        // Well past the 4-position window, which is the whole point:
        // under the old cap the sliding layers stopped being charged
        // here.
        let tokens = 64;
        assert!(
            cfg.sliding_window.is_some() && cfg.uniform_sliding_window().is_none(),
            "the fixture must be an alternating-SWA model, or this proves nothing"
        );

        let mut caches: Vec<ferrox_core::cache::KvCache> = (0..cfg.n_layers)
            .map(|_| ferrox_core::cache::KvCache::new(cfg.n_kv_heads, cfg.head_dim))
            .collect();
        let step = vec![0f32; cfg.n_kv_heads * cfg.head_dim];
        for _ in 0..tokens {
            for cache in caches.iter_mut() {
                cache
                    .push(&step, &step)
                    .expect("a cache built with `new` always accepts a push");
            }
        }
        let allocated: u64 = caches
            .iter()
            .map(|c| (c.k.len() + c.v.len()) as u64 * std::mem::size_of::<f32>() as u64)
            .sum();

        let shape = KvShape::from_config(&cfg, KvElem::F32);
        assert_eq!(
            shape.kv_bytes_for_tokens(tokens),
            allocated,
            "the budget must price what the store holds"
        );
        // The store kept every position in every layer, window or not.
        assert_eq!(allocated, shape.per_token_kv_bytes() * tokens as u64);
    }

    /// The pool-backed store is the other thing a server allocates, and
    /// it reserves `max_seq_len` positions for EVERY layer up front
    /// (`KvCache::with_pool`), rounded up to whole blocks. The budget
    /// must never be under that either -- an admitted request whose
    /// reservation exceeds the estimate is exactly the OOM #33 is about.
    #[test]
    fn the_pool_backed_store_never_reserves_more_positions_than_the_budget_priced() {
        use ferrox_core::cache::{KvBlockPool, KvCache};
        use std::sync::{Arc, Mutex};

        let cfg = alternating_swa_config();
        let tokens = 64usize;
        let block_size = 16usize;
        let pool = Arc::new(Mutex::new(KvBlockPool::new(
            block_size,
            tokens.div_ceil(block_size) * cfg.n_layers,
        )));
        let caches: Vec<KvCache> = (0..cfg.n_layers)
            .map(|_| {
                KvCache::with_pool(cfg.n_kv_heads, cfg.head_dim, Arc::clone(&pool), tokens)
                    .expect("the pool was sized for exactly this")
            })
            .collect();
        let reserved: u64 = caches
            .iter()
            .map(|c| c.k.capacity() as u64 + c.v.capacity() as u64)
            .sum::<u64>()
            * std::mem::size_of::<f32>() as u64;

        let priced = KvShape::from_config(&cfg, KvElem::F32).kv_bytes_for_tokens(tokens);
        // Equal here because `tokens` is a whole number of blocks; the
        // assertion that matters is the direction, which holds for any
        // block size.
        assert!(
            priced >= reserved,
            "budget priced {priced} bytes, the pool reserved {reserved}"
        );
        assert_eq!(priced, reserved);
    }

    /// The two checkpoints #33 measured, at their own byte counts.
    ///
    /// These constants are what the stores allocate, taken from the
    /// issue, not from this module's formula. The numbers the old code
    /// produced were 6,448,742,400 for gpt-oss (half) and 1,585,446,912
    /// for Gemma-3-4B (a sixth).
    #[test]
    fn gpt_oss_and_gemma3_cost_what_the_issue_measured() {
        // gpt-oss-20b: 24 layers, 8 kv-heads, head_dim 64, host f32,
        // 131072 context. Alternating 128-position window, priced at 0.
        let mut gpt_oss = crate::config::test_dense_fixture();
        gpt_oss.n_layers = 24;
        gpt_oss.n_kv_heads = 8;
        gpt_oss.head_dim = 64;
        gpt_oss.sliding_window = Some(128);
        gpt_oss.swa_pattern = Some(2);
        assert_eq!(
            KvShape::from_config(&gpt_oss, KvElem::F32).kv_bytes_for_tokens(131_072),
            12_884_901_888
        );

        // Gemma-3-4B: 34 layers, 4 kv-heads, head_dim 256, 32768 tokens.
        let mut gemma3 = crate::config::test_dense_fixture();
        gemma3.n_layers = 34;
        gemma3.n_kv_heads = 4;
        gemma3.head_dim = 256;
        gemma3.sliding_window = Some(1024);
        gemma3.swa_pattern = Some(6);
        assert_eq!(
            KvShape::from_config(&gemma3, KvElem::F32).kv_bytes_for_tokens(32_768),
            9_126_805_504
        );
    }

    /// A window changes what attention READS, not what the store KEEPS,
    /// so it may not change the price. Stated as an equality between two
    /// configs rather than as a comment, so re-introducing a cap fails
    /// here.
    #[test]
    fn a_windowed_config_is_priced_identically_to_the_same_config_without_a_window() {
        let windowed = alternating_swa_config();
        let mut full = windowed.clone();
        full.sliding_window = None;
        full.swa_pattern = None;
        for tokens in [1, 3, 4, 5, 64, 100_000] {
            assert_eq!(
                KvShape::from_config(&windowed, KvElem::F32).kv_bytes_for_tokens(tokens),
                KvShape::from_config(&full, KvElem::F32).kv_bytes_for_tokens(tokens),
                "tokens={tokens}"
            );
        }
    }

    #[test]
    fn mla_latent_is_one_vector_and_far_cheaper_than_the_expanded_form() {
        // DeepSeek-V2's real MLA numbers: kv_lora_rank 512,
        // qk_rope_head_dim 64, qk_nope_head_dim 128, v_head_dim 128,
        // 128 heads, 60 layers.
        let latent = KvShape {
            n_layers: 60,
            layout: KvLayout::MlaLatent {
                kv_lora_rank: 512,
                qk_rope_head_dim: 64,
            },
            elem: KvElem::F32,
        };
        // 512 + 64 = 576 scalars per token per layer -- one vector, no
        // K/V doubling.
        assert_eq!(latent.layout.elems_per_token_per_layer(), 576);
        assert_eq!(latent.per_token_kv_bytes(), 60 * 576 * 4);

        let expanded = KvShape::mla_expanded(60, 128, 128, 64, 128, KvElem::F32);
        // 128 heads x (192 K + 128 V) = 40960 scalars per token/layer.
        assert_eq!(
            expanded.layout.elems_per_token_per_layer(),
            128 * (192 + 128)
        );
        assert_eq!(expanded.per_token_kv_bytes(), 60 * 40_960 * 4);
        // The absorbed form is ~71x cheaper; this is exactly why the
        // distinction is worth carrying rather than assuming.
        assert!(expanded.per_token_kv_bytes() / latent.per_token_kv_bytes() > 70);

        // A same-sized GQA model for scale: 128 kv-heads x 128 head_dim.
        let gqa = KvShape {
            layout: KvLayout::Gqa {
                n_kv_heads: 128,
                head_dim: 128,
            },
            ..latent
        };
        assert_eq!(gqa.per_token_kv_bytes(), 60 * 2 * 128 * 128 * 4);
    }

    #[test]
    fn from_config_reads_layers_heads_and_head_dim() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.n_layers = 12;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 64;
        cfg.sliding_window = None;
        let shape = KvShape::from_config(&cfg, KvElem::F32);
        assert_eq!(shape.n_layers, 12);
        assert_eq!(shape.per_token_kv_bytes(), 12 * 2 * 2 * 64 * 4);

        // A uniform window changes nothing either: the paged store that
        // could recycle for one still holds the whole prompt, and it is
        // a context length this prices.
        cfg.sliding_window = Some(256);
        cfg.swa_pattern = None;
        assert_eq!(KvShape::from_config(&cfg, KvElem::F32), shape);
    }

    fn budget(weights: u64, device: u64, shape: KvShape) -> KvBudget {
        KvBudget {
            weights_bytes: weights,
            activation_headroom_bytes: 0,
            device_budget_bytes: device,
            shape,
            concurrent_requests: 1,
        }
    }

    #[test]
    fn check_accepts_a_fitting_context_and_names_the_binding_ceiling_otherwise() {
        let shape = llama31_8b(); // 262144 bytes/token
        let b = budget(1_000_000, 1_000_000 + 262_144 * 10, shape);
        assert_eq!(b.check(10).unwrap(), 1_000_000 + 262_144 * 10);
        let err = b.check(11).expect_err("one token past the budget");
        assert_eq!(err.binding, Ceiling::DeviceMemory);
        assert_eq!(err.code(), "device_memory_budget_exceeded");
        assert_eq!(err.estimated_bytes, 1_000_000 + 262_144 * 11);
        assert_eq!(err.limit_bytes, 1_000_000 + 262_144 * 10);
        assert_eq!(err.overage_bytes(), 262_144);
    }

    #[test]
    fn concurrency_multiplies_kv_but_not_weights() {
        let shape = llama31_8b();
        let one = budget(1_000, 1 << 40, shape);
        let four = KvBudget {
            concurrent_requests: 4,
            ..one
        };
        assert_eq!(
            four.estimated_bytes(100) - 1_000,
            4 * (one.estimated_bytes(100) - 1_000)
        );
    }

    #[test]
    fn max_context_is_the_closed_form_division_floored_to_granularity() {
        let shape = llama31_8b(); // 262144 bytes/token
                                  // Room for exactly 1000 tokens of KV after weights.
        let b = budget(5_000_000, 5_000_000 + 262_144 * 1000, shape);
        let fit = b.max_context(131_072, 256);
        assert_eq!(fit.capped_by, ContextCap::DeviceBudget);
        // 1000 floored to a 256-token step is 768.
        assert_eq!(fit.tokens, 768);
        assert_eq!(fit.kv_available_bytes, 262_144 * 1000);
        assert_eq!(fit.per_token_kv_bytes, 262_144);
        // The chosen context really does fit.
        assert!(b.check(fit.tokens).is_ok());
        // One granularity step further does not.
        assert!(b.check(fit.tokens + 256).is_err());
    }

    #[test]
    fn max_context_clamps_to_the_models_trained_context_when_memory_is_plentiful() {
        let b = budget(1_000, 1 << 40, llama31_8b());
        let fit = b.max_context(8192, 256);
        assert_eq!(fit.tokens, 8192);
        assert_eq!(fit.capped_by, ContextCap::ModelContextLength);
    }

    /// Flooring must not round a small-but-real answer down to "nothing
    /// fits" -- found by running `--ctx-size auto` under a tight
    /// `FERROX_DEVICE_BUDGET_BYTES`, where 227 tokens genuinely fitted
    /// and the 256-token granularity reported 0.
    #[test]
    fn a_context_under_one_granularity_step_is_reported_exactly_not_floored_away() {
        let shape = llama31_8b(); // 262144 bytes/token
        let b = budget(1_000, 1_000 + 262_144 * 100, shape);
        let fit = b.max_context(131_072, 256);
        assert_eq!(fit.tokens, 100);
        assert_eq!(fit.capped_by, ContextCap::DeviceBudget);
        assert!(b.check(fit.tokens).is_ok());
        assert!(b.check(fit.tokens + 1).is_err());
    }

    #[test]
    fn max_context_is_zero_when_the_weights_alone_do_not_fit() {
        let b = budget(10_000_000, 1_000_000, llama31_8b());
        let fit = b.max_context(8192, 256);
        assert_eq!(fit.tokens, 0);
        assert_eq!(fit.capped_by, ContextCap::DeviceBudget);
        assert_eq!(fit.kv_available_bytes, 0);
        assert!(b.check(0).is_err(), "weights alone already overflow");
    }

    /// `--ctx auto` on a windowed model used to answer "the model's own
    /// context length" however small the budget was, because the
    /// divisor had every sliding layer taken out of it and a model whose
    /// every layer slid divided by zero bytes per token. It is now
    /// bounded by memory like any other model, and the context it picks
    /// has to survive `check` -- which is the assertion that would have
    /// caught the OOM.
    #[test]
    fn a_windowed_model_is_bounded_by_memory_like_any_other() {
        let mut cfg = alternating_swa_config();
        cfg.swa_pattern = Some(1); // every layer slides: the old zero divisor
        let shape = KvShape::from_config(&cfg, KvElem::F32);
        // Room for 1024 tokens, against a model that would like 1e6.
        let b = budget(1_000, 1_000 + shape.per_token_kv_bytes() * 1024, shape);
        let fit = b.max_context(1_000_000, 256);
        assert_eq!(fit.capped_by, ContextCap::DeviceBudget);
        assert_eq!(fit.tokens, 1024);
        assert!(b.check(fit.tokens).is_ok());
        assert!(
            b.check(fit.tokens + 1).is_err(),
            "the chosen context must be the largest that fits"
        );
    }

    #[test]
    fn ctx_auto_explanation_names_every_term_it_divided() {
        let b = budget(5_000_000, 5_000_000 + 262_144 * 1000, llama31_8b());
        let text = b.max_context(131_072, CTX_AUTO_GRANULARITY).to_string();
        assert!(text.contains("ctx auto = 768 tokens"), "{text}");
        assert!(text.contains("262144"), "per-token divisor missing: {text}");
        assert!(text.contains("5000000"), "weights term missing: {text}");
        assert!(text.contains("131072"), "model cap missing: {text}");
    }
}
