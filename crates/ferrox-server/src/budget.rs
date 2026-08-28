//! One answer to "does this request fit", and one place that prices the
//! loaded model against the machine it landed on.
//!
//! Two things live here, and they are separate on purpose.
//!
//! # [`ContextCeiling`] -- the shared per-request ceiling
//!
//! Before this module the context ceiling existed only inside the
//! continuous batcher's `BlockBudget`, so a deployment running the
//! private `generate` path (the default: continuous batching is opt-in,
//! and is switched off entirely whenever a KV pool or prefix cache is
//! configured) had **no** context ceiling at all. An oversized request
//! there did not get a typed 400 naming what bound it -- it drained the
//! KV pool's admission wait and left with a 503 "retry shortly", which
//! is a lie about a request an idle server would refuse identically.
//!
//! `ContextCeiling` is the whole ceiling: the position limit, the
//! [`KvShape`] that prices a refusal in real bytes, and the rejection
//! counter. Both decode paths hold the *same* `Arc<ContextCeiling>`, so
//! the two cannot drift the way two copies of the same arithmetic
//! would -- the same discipline `crate::stop` applies to stop
//! sequences.
//!
//! # [`derive_limits`] -- pricing the model at load
//!
//! `ferrox` (the CLI) already prices a checkpoint before it loads it:
//! `--ctx-size auto`, a pre-load `KvBudget::check`, and a
//! `Ceiling::DeviceMemory` refusal. `ferrox-server` did not. It admitted
//! requests against ceilings an operator had *configured*
//! (`FERROX_CB_MAX_CONTEXT`, `FERROX_CB_KV_BLOCKS`) or, far more often,
//! against no ceiling at all, and discovered the real one as an OOM
//! kill. [`derive_limits`] closes that: weights plus `n_ctx *
//! per_token_kv` plus headroom against the device budget, every term
//! exact from the GGUF header, evaluated once at load.
//!
//! ## Where derivation deliberately stops
//!
//! - **An explicit setting always wins.** A derived number never
//!   overrides `FERROX_CB_MAX_CONTEXT` or `FERROX_CB_KV_BLOCKS`; it only
//!   ever fills an *absent* ceiling, where the status quo is "unbounded
//!   until the kernel intervenes".
//! - **An unknown budget derives nothing.** No probe, no ceiling: the
//!   same rule the CLI's `resolve_ctx_size` follows. Refusing on the
//!   strength of a number we do not have is worse than not refusing.
//! - **A model that does not fit at all derives nothing either**, and
//!   this is the one place where fail-closed is the *wrong* reading. A
//!   fit of zero tokens would mean a ceiling of zero positions, which is
//!   not a ceiling -- it is a refusal to serve the model, decided by an
//!   estimate. `ferrox_models::kv_budget`'s own module doc is explicit
//!   that `weights_bytes` is the checkpoint's byte count over *mmap'd*
//!   pages, an upper bound on residency rather than a measurement, and
//!   that "a model can exceed this budget and still run". Refusing one
//!   oversized request against a working estimate is a different claim
//!   from refusing every request because the estimate says the model
//!   should not have loaded -- and it *did* load. So this logs loudly,
//!   names `FERROX_DEVICE_BUDGET_BYTES`, and leaves the ceiling absent.

use std::sync::atomic::{AtomicU64, Ordering};

use ferrox_models::{Ceiling, ContextFit, KvBudget, KvElem, KvShape};

use crate::generate::DecodeError;

/// The per-request context ceiling, shared by every decode path.
///
/// `positions` throughout is prompt + `max_tokens`: the worst-case
/// sequence length the request will hold KV for, which is the number
/// both the KV pool and the block ledger reserve against.
#[derive(Debug)]
pub struct ContextCeiling {
    /// Positions any one request may ask for. `None` = no ceiling,
    /// which is what an unpriced deployment has.
    limit: Option<usize>,
    /// This model's real KV geometry, so a refusal states bytes rather
    /// than an opaque position count.
    shape: KvShape,
    /// Requests refused for exceeding `limit`. Counted here rather than
    /// at each call site so `/metrics` reports one number no matter
    /// which path did the refusing.
    refused: AtomicU64,
}

impl ContextCeiling {
    pub fn new(limit: Option<usize>, shape: KvShape) -> Self {
        ContextCeiling {
            limit,
            shape,
            refused: AtomicU64::new(0),
        }
    }

    /// KV bytes `positions` of context costs, sliding-window cap
    /// included.
    pub fn bytes_for(&self, positions: usize) -> u64 {
        self.shape.kv_bytes_for_tokens(positions)
    }

    pub fn refused(&self) -> u64 {
        self.refused.load(Ordering::Relaxed)
    }

    /// The per-request position ceiling, when this deployment has one.
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// The refusal for a prompt that does not fit the ceiling *by
    /// itself*, which is a different answer from one whose prompt plus
    /// budget does not.
    ///
    /// A prompt shorter than the ceiling is servable -- with a smaller
    /// output budget -- so it is clamped rather than refused (see
    /// `generate`). A prompt at or past the ceiling has no budget left
    /// to clamp to, so it is the one case that must fail.
    ///
    /// The wording is load-bearing and copied verbatim: Claude Code and
    /// OpenClaw match on this text to recognise a blown context window,
    /// because the Anthropic wire carries no error code for it.
    pub fn prompt_refusal(&self, prompt_tokens: usize) -> Option<DecodeError> {
        let limit = self.limit?;
        if prompt_tokens < limit {
            return None;
        }
        self.refused.fetch_add(1, Ordering::Relaxed);
        Some(DecodeError::KvBudgetExceeded {
            binding: Ceiling::ContextLength.code(),
            estimated_bytes: self.bytes_for(prompt_tokens),
            limit_bytes: self.bytes_for(limit),
            positions: prompt_tokens,
            positions_limit: limit,
            detail: format!("prompt is too long: {prompt_tokens} tokens > {limit} maximum"),
        })
    }

    /// The typed refusal for a request of `positions` positions, or
    /// `None` when it fits.
    ///
    /// Deliberately a 400 (`retry_after_secs() == None`): the ceiling is
    /// a property of the deployment, so an idle server refuses this
    /// request identically and "retry shortly" would be a lie.
    pub fn refusal(&self, positions: usize) -> Option<DecodeError> {
        let limit = self.limit?;
        if positions <= limit {
            return None;
        }
        self.refused.fetch_add(1, Ordering::Relaxed);
        Some(DecodeError::KvBudgetExceeded {
            binding: Ceiling::ContextLength.code(),
            estimated_bytes: self.bytes_for(positions),
            limit_bytes: self.bytes_for(limit),
            positions,
            positions_limit: limit,
            detail: format!(
                "request asks for {positions} token positions (prompt + max_tokens) but this \
                 deployment admits {limit} per request; shorten the prompt or lower max_tokens"
            ),
        })
    }
}

/// What [`derive_limits`] concluded, with the arithmetic that produced
/// it so an operator can check the division by hand.
#[derive(Debug, Clone, Copy)]
pub struct DerivedLimits {
    /// Positions any one request may hold: the largest context that
    /// fits this machine, capped at the model's own trained context.
    pub max_context: usize,
    /// Blocks for the whole-server KV ledger, at `block_size` positions
    /// each. Floored, never rounded up: a block the machine cannot hold
    /// is not a block to promise.
    pub kv_blocks: usize,
    /// The priced fit this came from; `Display`s as the full inequality.
    pub fit: ContextFit,
}

/// Turns a priced [`KvBudget`] into the two ceilings the server admits
/// on, or `None` when no honest ceiling follows.
///
/// `gguf_ctx` is the model's own trained context length -- the cap, so a
/// machine with room to spare derives the same number llama.cpp would
/// default to rather than a larger one the model was never trained for.
///
/// `None` means "derive nothing" and has exactly one cause: the priced
/// fit is zero tokens, i.e. weights plus headroom already fill the
/// budget. See this module's doc comment for why that is logged rather
/// than turned into a ceiling of zero.
pub fn derive_limits(
    budget: &KvBudget,
    gguf_ctx: usize,
    block_size: usize,
) -> Option<DerivedLimits> {
    assert!(block_size > 0, "kv block size must be positive");
    let cap = gguf_ctx.max(1);
    let fit = budget.max_context(cap, ferrox_models::CTX_AUTO_GRANULARITY);
    if fit.tokens == 0 {
        return None;
    }
    Some(DerivedLimits {
        max_context: fit.tokens,
        // Floor: the ledger's job is to hand out capacity that exists.
        // Rounding a partial block up would promise `block_size`
        // positions the fit says are not there.
        kv_blocks: fit.tokens / block_size,
        fit,
    })
}

/// Fills the ceilings an operator did not set from `derived`, and
/// leaves the ones they did set exactly alone.
///
/// Split out as a pure function because the precedence *is* the
/// contract: an operator who names a number has information this
/// arithmetic does not, so a derived value may only ever occupy an
/// empty slot. Returns what changed, so the caller logs the numbers it
/// actually adopted rather than the ones it computed.
pub fn apply_derived(
    config: &mut crate::serving::batch::BatcherConfig,
    derived: &DerivedLimits,
) -> Adopted {
    let mut adopted = Adopted::default();
    if config.max_context.is_none() {
        config.max_context = Some(derived.max_context);
        adopted.max_context = true;
    }
    // A zero-block ledger would refuse every request, which is the
    // "ceiling of zero" this module refuses to invent -- see the module
    // doc. Leave the ledger absent and let the context ceiling, which
    // is a real number here, do the refusing.
    if config.kv_blocks.is_none() && derived.kv_blocks > 0 {
        config.kv_blocks = Some(derived.kv_blocks);
        adopted.kv_blocks = true;
    }
    adopted
}

/// Which ceilings [`apply_derived`] actually filled in.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Adopted {
    pub max_context: bool,
    pub kv_blocks: bool,
}

/// Prices the GGUF at `path` against the machine, best effort.
///
/// Returns `None` -- and logs why -- for every reason the CLI treats as
/// "do not check": no device-budget probe, or a header this planner
/// cannot read (the MLA/Gemma4/GLM stacks carry their own hparams and
/// never build a `ModelConfig`). A model this server cannot price is
/// served exactly as it was before this module existed.
///
/// The `String` is the probe's own description, so the startup log says
/// where the budget number came from rather than asserting it.
pub fn price_gguf(
    path: &str,
    kv_elem: KvElem,
    prefill_chunk: usize,
    concurrent_requests: usize,
) -> Option<(KvBudget, usize, String)> {
    use ferrox_models::residency_report::{ResidencyAssumptions, ResidencyReport};
    use ferrox_models::{BudgetBackend, DeviceBudget};

    let backend = if cfg!(feature = "metal") {
        BudgetBackend::Metal
    } else if cfg!(feature = "cuda") {
        BudgetBackend::Cuda
    } else {
        BudgetBackend::Cpu
    };
    let device = DeviceBudget::detect(backend);
    if device.is_unknown() {
        tracing::info!("{device}; serving with no derived context ceiling");
        return None;
    }
    let gguf_ctx = gguf_context_length(path)?;
    let assumptions = ResidencyAssumptions {
        context_tokens: gguf_ctx,
        concurrent_requests: concurrent_requests.max(1),
        kv_elem,
        prefill_chunk: prefill_chunk.max(1),
        ..ResidencyAssumptions::default()
    };
    match ResidencyReport::from_gguf(path, assumptions, device.usable_bytes) {
        Ok(report) => Some((report.kv_budget(), gguf_ctx, device.to_string())),
        Err(e) => {
            tracing::info!(
                "KV budget not computed for this checkpoint ({e}); serving with no derived \
                 context ceiling"
            );
            None
        }
    }
}

/// The model's trained context length from its own header, or the same
/// 4096 fallback `ferrox run` uses when the key is absent.
fn gguf_context_length(path: &str) -> Option<usize> {
    let file = ferrox_gguf::ShardedGguf::open(path).ok()?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    Some(
        file.metadata_u64(&format!("{arch}.context_length"))
            .map(|v| v as usize)
            .unwrap_or(4096),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_models::{KvLayout, SlidingWindow};

    /// A shape whose arithmetic is easy to do by hand: 2 layers, 1 kv
    /// head, head_dim 4, f32 -> 2 layers * 4 elems * 2 (K and V) * 4
    /// bytes = 64 bytes per token.
    fn shape() -> KvShape {
        KvShape {
            n_layers: 2,
            layout: KvLayout::Gqa {
                n_kv_heads: 1,
                head_dim: 4,
            },
            elem: KvElem::F32,
            sliding: None,
        }
    }

    fn budget(device_bytes: u64, weights_bytes: u64) -> KvBudget {
        KvBudget {
            weights_bytes,
            activation_headroom_bytes: 0,
            device_budget_bytes: device_bytes,
            shape: shape(),
            concurrent_requests: 1,
        }
    }

    #[test]
    fn per_token_bytes_are_what_the_hand_computation_says() {
        assert_eq!(shape().per_token_kv_bytes(), 64);
    }

    /// The whole point of the ceiling: a request past it is refused
    /// with the *context* code, priced in real bytes, and is not a
    /// retryable error.
    ///
    /// Confirmed to FAIL when `refusal` returns `None` unconditionally,
    /// and when the `positions <= limit` comparison is flipped.
    #[test]
    fn a_request_past_the_ceiling_is_refused_in_bytes_and_is_not_retryable() {
        let ceiling = ContextCeiling::new(Some(100), shape());
        assert!(ceiling.refusal(100).is_none(), "exactly at the limit fits");
        let err = ceiling.refusal(101).expect("101 > 100 must be refused");
        match &err {
            DecodeError::KvBudgetExceeded {
                binding,
                estimated_bytes,
                limit_bytes,
                positions,
                positions_limit,
                ..
            } => {
                assert_eq!(*binding, Ceiling::ContextLength.code());
                assert_eq!(*estimated_bytes, 101 * 64);
                assert_eq!(*limit_bytes, 100 * 64);
                assert_eq!(*positions, 101);
                assert_eq!(*positions_limit, 100);
            }
            other => panic!("expected KvBudgetExceeded, got {other:?}"),
        }
        assert_eq!(
            err.retry_after_secs(),
            None,
            "an idle server refuses this identically, so 'retry shortly' would be a lie"
        );
        assert_eq!(ceiling.refused(), 1, "the refusal must be counted");
    }

    /// An absent ceiling is not a ceiling of zero. A server that could
    /// not price its model must serve exactly as it did before.
    #[test]
    fn no_ceiling_refuses_nothing() {
        let ceiling = ContextCeiling::new(None, shape());
        assert!(ceiling.refusal(usize::MAX / 2).is_none());
        assert_eq!(ceiling.refused(), 0);
    }

    /// 4096 bytes of KV room / 64 bytes per token = 64 tokens, which is
    /// under one `CTX_AUTO_GRANULARITY` step, so the fit reports the
    /// exact number rather than rounding it away to nothing.
    #[test]
    fn the_derived_context_is_the_room_left_after_weights_divided_by_the_per_token_cost() {
        let derived = derive_limits(&budget(8192, 4096), 100_000, 16)
            .expect("4096 bytes of KV room fits some context");
        assert_eq!(derived.max_context, 64);
        assert_eq!(derived.kv_blocks, 4, "64 positions / 16 per block");
    }

    /// The model's own trained context is the cap: a machine with room
    /// to spare must not derive a context the model was never trained
    /// for.
    ///
    /// Confirmed to FAIL when `derive_limits` passes `usize::MAX` as the
    /// cap instead of `gguf_ctx`.
    #[test]
    fn a_roomy_machine_is_still_capped_at_the_models_trained_context() {
        let derived = derive_limits(&budget(1 << 40, 0), 4096, 256)
            .expect("a terabyte of room fits the model's whole context");
        assert_eq!(derived.max_context, 4096);
        assert_eq!(derived.kv_blocks, 16);
    }

    /// A partial block is not a block. Rounding up here would promise
    /// `block_size` positions the priced fit says are not there.
    ///
    /// Confirmed to FAIL when `kv_blocks` uses `div_ceil`.
    #[test]
    fn a_partial_block_is_floored_away_rather_than_promised() {
        // 6400 bytes of KV room -> 100 tokens, block size 64 -> one
        // whole block and 36 positions left over.
        let derived = derive_limits(&budget(6400, 0), 100_000, 64).expect("100 tokens fit");
        assert_eq!(derived.max_context, 100);
        assert_eq!(derived.kv_blocks, 1);
    }

    /// Weights alone fill the budget: no context fits. This derives
    /// *nothing* rather than a ceiling of zero -- see the module doc for
    /// why a refusal to serve is not the same claim as a refusal of one
    /// oversized request.
    #[test]
    fn a_model_that_leaves_no_room_derives_no_ceiling_at_all() {
        assert!(derive_limits(&budget(4096, 4096), 100_000, 16).is_none());
        assert!(derive_limits(&budget(4096, 8192), 100_000, 16).is_none());
    }

    fn derived(max_context: usize, kv_blocks: usize) -> DerivedLimits {
        DerivedLimits {
            max_context,
            kv_blocks,
            fit: budget(1 << 30, 0).max_context(max_context.max(1), 1),
        }
    }

    /// The precedence contract: an operator who set a number keeps it.
    ///
    /// Confirmed to FAIL when `apply_derived` assigns unconditionally
    /// instead of only into an empty slot.
    #[test]
    fn a_configured_ceiling_is_never_overridden_by_a_derived_one() {
        let mut config = crate::serving::batch::BatcherConfig {
            max_context: Some(999),
            kv_blocks: Some(7),
            ..Default::default()
        };
        let adopted = apply_derived(&mut config, &derived(4096, 16));
        assert_eq!(config.max_context, Some(999));
        assert_eq!(config.kv_blocks, Some(7));
        assert_eq!(adopted, Adopted::default(), "nothing was adopted");
    }

    /// An absent ceiling is the slot derivation exists to fill, and the
    /// two slots are independent: setting one by hand must not suppress
    /// the other's derivation.
    #[test]
    fn an_absent_ceiling_is_filled_and_the_two_slots_are_independent() {
        let mut both = crate::serving::batch::BatcherConfig {
            max_context: None,
            kv_blocks: None,
            ..Default::default()
        };
        let adopted = apply_derived(&mut both, &derived(4096, 16));
        assert_eq!(both.max_context, Some(4096));
        assert_eq!(both.kv_blocks, Some(16));
        assert_eq!(
            adopted,
            Adopted {
                max_context: true,
                kv_blocks: true
            }
        );

        let mut half = crate::serving::batch::BatcherConfig {
            max_context: Some(512),
            kv_blocks: None,
            ..Default::default()
        };
        let adopted = apply_derived(&mut half, &derived(4096, 16));
        assert_eq!(half.max_context, Some(512), "the set one survives");
        assert_eq!(half.kv_blocks, Some(16), "the unset one is still derived");
        assert_eq!(
            adopted,
            Adopted {
                max_context: false,
                kv_blocks: true
            }
        );
    }

    /// A fit that yields no whole block leaves the ledger absent rather
    /// than installing a zero-block budget that refuses everything --
    /// the context ceiling, which is a real number here, does the
    /// refusing instead.
    ///
    /// Confirmed to FAIL when the `derived.kv_blocks > 0` guard is
    /// dropped: `kv_blocks` becomes `Some(0)`.
    #[test]
    fn a_fit_smaller_than_one_block_leaves_the_ledger_absent() {
        let mut config = crate::serving::batch::BatcherConfig {
            max_context: None,
            kv_blocks: None,
            ..Default::default()
        };
        apply_derived(&mut config, &derived(100, 0));
        assert_eq!(config.max_context, Some(100));
        assert_eq!(config.kv_blocks, None);
    }

    /// A model whose every layer slides has a bounded KV no matter how
    /// long the context is, so only the model's own context length
    /// limits it -- the `marginal == 0` case `max_context` handles with
    /// `checked_div`.
    #[test]
    fn a_fully_sliding_model_is_limited_only_by_its_context_length() {
        let mut b = budget(1 << 30, 0);
        b.shape.sliding = Some(SlidingWindow {
            window: 128,
            chunk: 1,
            pattern: None,
        });
        let derived = derive_limits(&b, 8192, 256).expect("a bounded KV always fits");
        assert_eq!(derived.max_context, 8192);
    }
}
