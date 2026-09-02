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

/// Sliding-window attention, which caps how many positions a layer ever
/// keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlidingWindow {
    /// Positions a query may attend back over.
    pub window: usize,
    /// Prefill chunk size. A chunk of `chunk` tokens is processed
    /// against one cache state, so the *first* token of the chunk still
    /// needs its whole window live while the *last* one is being
    /// processed: `window + chunk - 1` positions, not `window`.
    pub chunk: usize,
    /// Gemma 2+/3 alternating pattern (`ModelConfig::swa_pattern`):
    /// layer `il` is sliding iff `(il + 1) % period != 0`, so every
    /// `period`-th layer keeps the full context. `None` means every
    /// layer slides.
    pub pattern: Option<usize>,
}

impl SlidingWindow {
    /// Positions a sliding layer keeps once the sequence is long
    /// enough to saturate it.
    pub fn resident_positions(&self, tokens: usize) -> usize {
        tokens.min(self.window + self.chunk.max(1) - 1)
    }
}

/// The KV shape of a whole model: enough to price any context length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShape {
    pub n_layers: usize,
    pub layout: KvLayout,
    pub elem: KvElem,
    /// `None` = every layer keeps the full causal history.
    pub sliding: Option<SlidingWindow>,
}

impl KvShape {
    /// Reads the shape off a config. `chunk` is the prefill chunk size
    /// the run will use (`FERROX_CHUNKED_PREFILL`, or 1 when prefill is
    /// token-at-a-time); it only ever matters for sliding layers.
    ///
    /// Always produces a [`KvLayout::Gqa`] layout, because
    /// `ModelConfig` describes the generic GQA decoder -- the MLA
    /// stacks carry their own hyperparameters (`Deepseek2Hparams`,
    /// `MlaConfig`) and should build their shape with
    /// [`KvShape::mla_expanded`].
    pub fn from_config(config: &ModelConfig, elem: KvElem, chunk: usize) -> Self {
        KvShape {
            n_layers: config.n_layers,
            layout: KvLayout::Gqa {
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
            },
            elem,
            sliding: config.sliding_window.map(|window| SlidingWindow {
                window,
                chunk,
                pattern: config.swa_pattern,
            }),
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
            sliding: None,
        }
    }

    /// How many layers slide, given the alternating pattern.
    pub fn sliding_layers(&self) -> usize {
        match self.sliding {
            None => 0,
            Some(SlidingWindow { pattern: None, .. }) => self.n_layers,
            Some(SlidingWindow {
                pattern: Some(period),
                ..
            }) => {
                if period <= 1 {
                    self.n_layers
                } else {
                    // llama.cpp `set_swa_pattern`: full attention iff
                    // `(il + 1) % period == 0`.
                    self.n_layers - self.n_layers / period
                }
            }
        }
    }

    /// Layers that keep the full causal history.
    pub fn full_attention_layers(&self) -> usize {
        self.n_layers - self.sliding_layers()
    }

    /// The plan's headline number: bytes one token costs across every
    /// layer, ignoring any sliding-window cap. Exact for f32/f16;
    /// for the block-quantized wires it is exact whenever a layer's
    /// per-token element count is a multiple of the 32-element block
    /// (true for every real head-dim/kv-head combination), and rounds
    /// up otherwise.
    pub fn per_token_kv_bytes(&self) -> u64 {
        self.n_layers as u64 * self.elem.bytes_for(self.layout.elems_per_token_per_layer())
    }

    /// Bytes each *additional* context token costs once the sliding
    /// layers have saturated: only the full-attention layers keep
    /// growing. This is the divisor [`KvBudget::max_context`] uses,
    /// and it is `0` for a model whose every layer slides -- such a
    /// model's KV is bounded no matter how long the context is.
    pub fn marginal_per_token_bytes(&self) -> u64 {
        self.full_attention_layers() as u64
            * self.elem.bytes_for(self.layout.elems_per_token_per_layer())
    }

    /// Bytes one request's KV costs at `tokens` of context, applying
    /// the sliding-window cap per layer class.
    pub fn kv_bytes_for_tokens(&self, tokens: usize) -> u64 {
        // Every multiplication here saturates, for the reason on
        // `KvElem::bytes_for`: `tokens` can arrive from an HTTP body.
        let per_layer = self.layout.elems_per_token_per_layer();
        let full = (self.full_attention_layers() as u64)
            .saturating_mul(self.elem.bytes_for(per_layer.saturating_mul(tokens as u64)));
        let sliding = match self.sliding {
            None => 0,
            Some(w) => (self.sliding_layers() as u64).saturating_mul(
                self.elem
                    .bytes_for(per_layer.saturating_mul(w.resident_positions(tokens) as u64)),
            ),
        };
        full.saturating_add(sliding)
    }

    /// The sentence a user should be able to read and reproduce with a
    /// calculator.
    pub fn describe(&self) -> String {
        let base = format!(
            "{} layers x [{}] x {} = {} bytes/token",
            self.n_layers,
            self.layout.describe(),
            self.elem.as_str(),
            self.per_token_kv_bytes()
        );
        match self.sliding {
            None => base,
            Some(w) => format!(
                "{base}; {} of {} layers slide and cap at min(tokens, {} window + {} chunk - 1) \
                 = {} positions, leaving {} bytes/token marginal",
                self.sliding_layers(),
                self.n_layers,
                w.window,
                w.chunk,
                w.window + w.chunk.max(1) - 1,
                self.marginal_per_token_bytes(),
            ),
        }
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
    /// Sliding-window layers are subtracted out of the divisor (they
    /// stop growing once saturated) and added back as a constant, so a
    /// model whose every layer slides is limited only by `cap`.
    pub fn max_context(&self, cap: usize, granularity: usize) -> ContextFit {
        let granularity = granularity.max(1);
        let concurrency = self.concurrent_requests.max(1) as u64;
        let available = self.kv_bytes_available();
        let marginal = self.shape.marginal_per_token_bytes() * concurrency;

        // The sliding layers' saturated cost is a constant that has to
        // come out of the budget before the full-attention layers get
        // to divide what's left. Priced at `cap` (their worst case).
        let saturated_sliding = {
            let mut shape = self.shape;
            shape.n_layers = shape.sliding_layers();
            match shape.sliding {
                None => 0,
                Some(w) => {
                    shape.n_layers as u64
                        * shape.elem.bytes_for(
                            shape.layout.elems_per_token_per_layer()
                                * w.resident_positions(cap) as u64,
                        )
                        * concurrency
                }
            }
        };
        let for_full_layers = available.saturating_sub(saturated_sliding);

        let (tokens, capped_by) = if available == 0 || for_full_layers == 0 && marginal > 0 {
            (0, ContextCap::DeviceBudget)
        } else {
            // `checked_div` rather than a `marginal == 0` guard around
            // a bare `/`: the zero case is not an error here, it is a
            // real configuration -- every layer slides, so KV is
            // bounded and only the model's own context length limits
            // us -- and expressing it as `None` keeps that meaning in
            // one place instead of splitting it across a check and a
            // division that clippy then has to re-associate.
            match for_full_layers.checked_div(marginal) {
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
            marginal_per_token_bytes: self.shape.marginal_per_token_bytes(),
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
    pub marginal_per_token_bytes: u64,
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
            self.marginal_per_token_bytes,
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
            sliding: None,
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

    #[test]
    fn sliding_window_layers_saturate_and_full_layers_do_not() {
        // Mistral-7B shape with a 4096 window, every layer sliding,
        // prefill one token at a time (chunk = 1 -> cap is exactly the
        // window).
        let shape = KvShape {
            n_layers: 32,
            layout: KvLayout::Gqa {
                n_kv_heads: 8,
                head_dim: 128,
            },
            elem: KvElem::F16,
            sliding: Some(SlidingWindow {
                window: 4096,
                chunk: 1,
                pattern: None,
            }),
        };
        assert_eq!(shape.sliding_layers(), 32);
        assert_eq!(shape.full_attention_layers(), 0);
        // Below the window it costs the same as full attention.
        assert_eq!(
            shape.kv_bytes_for_tokens(1024),
            shape.per_token_kv_bytes() * 1024
        );
        // Above it, cost stops growing.
        let at_window = shape.kv_bytes_for_tokens(4096);
        assert_eq!(shape.kv_bytes_for_tokens(32_768), at_window);
        assert_eq!(shape.kv_bytes_for_tokens(1_000_000), at_window);
        // Marginal cost per extra token is zero once every layer slides.
        assert_eq!(shape.marginal_per_token_bytes(), 0);
    }

    #[test]
    fn chunked_prefill_widens_the_sliding_cap_by_chunk_minus_one() {
        let base = SlidingWindow {
            window: 512,
            chunk: 1,
            pattern: None,
        };
        assert_eq!(base.resident_positions(100_000), 512);
        let chunked = SlidingWindow { chunk: 256, ..base };
        // window + chunk - 1, per the plan: the first token of a chunk
        // still needs its full window when the last one runs.
        assert_eq!(chunked.resident_positions(100_000), 512 + 256 - 1);
        assert_eq!(chunked.resident_positions(300), 300);
    }

    #[test]
    fn gemma_alternating_pattern_leaves_every_sixth_layer_full_attention() {
        // Gemma 3's real 5:1 pattern: layer `il` slides unless
        // `(il + 1) % 6 == 0`, so 26 layers slide and 5 do not out of 31.
        let shape = KvShape {
            n_layers: 30,
            layout: KvLayout::Gqa {
                n_kv_heads: 4,
                head_dim: 256,
            },
            elem: KvElem::F16,
            sliding: Some(SlidingWindow {
                window: 1024,
                chunk: 1,
                pattern: Some(6),
            }),
        };
        assert_eq!(shape.full_attention_layers(), 5);
        assert_eq!(shape.sliding_layers(), 25);
        // Cross-check against ModelConfig's own per-layer answer, so
        // the two SWA implementations cannot drift apart.
        let mut cfg = crate::config::test_dense_fixture();
        cfg.n_layers = 30;
        cfg.sliding_window = Some(1024);
        cfg.swa_pattern = Some(6);
        let per_layer_full = (0..30)
            .filter(|&il| cfg.layer_sliding_window(il).is_none())
            .count();
        assert_eq!(per_layer_full, shape.full_attention_layers());

        // Only the 5 full-attention layers keep growing with context.
        let per_layer_token = shape.elem.bytes_for(2 * 4 * 256);
        assert_eq!(shape.marginal_per_token_bytes(), 5 * per_layer_token);
        // At 8192 tokens: 5 full layers at 8192 positions, 25 sliding
        // layers pinned at 1024.
        assert_eq!(
            shape.kv_bytes_for_tokens(8192),
            5 * shape.elem.bytes_for(2 * 4 * 256 * 8192)
                + 25 * shape.elem.bytes_for(2 * 4 * 256 * 1024)
        );
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
            sliding: None,
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
    fn from_config_reads_layers_heads_and_the_sliding_window() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.n_layers = 12;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 64;
        cfg.sliding_window = None;
        let shape = KvShape::from_config(&cfg, KvElem::F32, 1);
        assert_eq!(shape.n_layers, 12);
        assert_eq!(shape.per_token_kv_bytes(), 12 * 2 * 2 * 64 * 4);
        assert!(shape.sliding.is_none());

        cfg.sliding_window = Some(256);
        cfg.swa_pattern = None;
        let swa = KvShape::from_config(&cfg, KvElem::F32, 64);
        assert_eq!(
            swa.sliding,
            Some(SlidingWindow {
                window: 256,
                chunk: 64,
                pattern: None
            })
        );
        assert_eq!(swa.sliding_layers(), 12);
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
        assert_eq!(fit.marginal_per_token_bytes, 262_144);
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

    #[test]
    fn an_all_sliding_model_is_limited_only_by_its_context_length() {
        let shape = KvShape {
            n_layers: 32,
            layout: KvLayout::Gqa {
                n_kv_heads: 8,
                head_dim: 128,
            },
            elem: KvElem::F16,
            sliding: Some(SlidingWindow {
                window: 4096,
                chunk: 1,
                pattern: None,
            }),
        };
        // Budget covers the saturated window with room to spare.
        let saturated = shape.kv_bytes_for_tokens(4096);
        let b = budget(1_000, 1_000 + saturated * 2, shape);
        let fit = b.max_context(1_000_000, 256);
        assert_eq!(fit.capped_by, ContextCap::ModelContextLength);
        assert_eq!(fit.tokens, 1_000_000);
        assert!(b.check(fit.tokens).is_ok());
    }

    #[test]
    fn a_mixed_swa_model_prices_the_saturated_sliding_layers_before_dividing() {
        // 6 layers, every 3rd full-attention (2 full, 4 sliding).
        let shape = KvShape {
            n_layers: 6,
            layout: KvLayout::Gqa {
                n_kv_heads: 1,
                head_dim: 16,
            },
            elem: KvElem::F32,
            sliding: Some(SlidingWindow {
                window: 128,
                chunk: 1,
                pattern: Some(3),
            }),
        };
        assert_eq!(shape.full_attention_layers(), 2);
        // 2 (K+V) x 1 kv-head x 16 head-dim x 4 bytes.
        let per_layer_token = 2 * 16 * 4;
        let sliding_saturated = 4 * per_layer_token * 128;
        let full_marginal = 2 * per_layer_token;
        // Give the budget the saturated sliding cost plus exactly 512
        // tokens of full-attention growth.
        let b = budget(0, (sliding_saturated + full_marginal * 512) as u64, shape);
        let fit = b.max_context(4096, 256);
        assert_eq!(fit.tokens, 512);
        assert_eq!(fit.capped_by, ContextCap::DeviceBudget);
        assert!(b.check(512).is_ok());
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
