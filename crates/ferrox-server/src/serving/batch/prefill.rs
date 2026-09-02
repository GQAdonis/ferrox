//! Chunked prefill: a prompt as a resumable state machine rather than
//! one uninterruptible `forward_token` loop.
//!
//! Chunking is a *scheduling* boundary, not a numerical one. A chunk is
//! still the same per-token sequence at the same positions, so chunk
//! size never changes the logits or the sampled tokens (asserted by
//! `prefill_chunking_does_not_change_logits`).

use std::sync::mpsc::Sender;

use ferrox_core::cache::KvCache;
use ferrox_models::tokenizer::StopTokens;
use ferrox_models::Decoder;
use std::sync::Arc;

use crate::generate::{GenerationParams, PagedLease};

use super::config::BatcherEvent;
use super::queue::AbortId;
use super::row::{RowKv, Slot};
use crate::sample_step::SampleState;
use crate::stop::StopMatcher;

/// One request's prefill as a resumable state machine.
///
/// Holds the KV `caches` being built, how many prompt tokens have been
/// processed, and the logits produced by the most recent token.
/// `step_chunk` advances by at most `chunk_size` tokens and reports
/// whether the prompt is finished, which converts an unbounded prefill
/// into a bounded unit of work the scheduler can interleave with
/// decode.
pub struct PrefillState {
    decoder: Arc<Decoder>,
    kv: RowKv,
    /// The tokens this prefill must run. An *empty* prompt is stored as
    /// a single token 0, matching the private `generate` loop: one
    /// forward pass is still required to produce the logits the first
    /// sampled token comes from.
    tokens: Vec<usize>,
    tokens_processed: usize,
    logits: Vec<f32>,
    chunk_size: usize,
}

impl PrefillState {
    pub fn new(decoder: Arc<Decoder>, prompt_tokens: &[usize], chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "prefill chunk size must be positive");
        let kv = RowKv::Contiguous(
            decoder
                .layers
                .iter()
                .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
                .collect(),
        );
        let tokens = if prompt_tokens.is_empty() {
            vec![0]
        } else {
            prompt_tokens.to_vec()
        };
        PrefillState {
            decoder,
            kv,
            tokens,
            tokens_processed: 0,
            logits: Vec::new(),
            chunk_size,
        }
    }

    /// A prefill whose KV lives in a shared paged store, with its
    /// pages already reserved.
    ///
    /// Reserved for `prompt + max_tokens` at admission for the same
    /// reason the private path does it: the decode step has nowhere to
    /// report a store that ran dry mid-answer.
    pub fn new_paged(
        decoder: Arc<Decoder>,
        prompt_tokens: &[usize],
        chunk_size: usize,
        lease: PagedLease,
    ) -> Self {
        assert!(chunk_size > 0, "prefill chunk size must be positive");
        let tokens = if prompt_tokens.is_empty() {
            vec![0]
        } else {
            prompt_tokens.to_vec()
        };
        PrefillState {
            decoder,
            kv: RowKv::Paged(lease),
            tokens,
            tokens_processed: 0,
            logits: Vec::new(),
            chunk_size,
        }
    }

    /// Prompt tokens already through the model.
    pub fn tokens_processed(&self) -> usize {
        self.tokens_processed
    }

    /// Prompt tokens still to run.
    pub fn tokens_remaining(&self) -> usize {
        self.tokens.len() - self.tokens_processed
    }

    pub fn is_done(&self) -> bool {
        self.tokens_remaining() == 0
    }

    /// Runs at most `chunk_size` further prompt tokens. Returns `true`
    /// once the whole prompt has been processed. Calling it again after
    /// that is a no-op that still returns `true`.
    ///
    /// The KV position of each token is its index in the prompt, so
    /// resuming across chunk boundaries is exactly the sequential
    /// `forward_token` loop it replaces, split at different points.
    pub fn step_chunk(&mut self) -> bool {
        let end = (self.tokens_processed + self.chunk_size).min(self.tokens.len());
        for pos in self.tokens_processed..end {
            self.logits = match &mut self.kv {
                RowKv::Contiguous(caches) => {
                    self.decoder.forward_token(self.tokens[pos], pos, caches)
                }
                RowKv::Paged(lease) => {
                    let store = Arc::clone(lease.store());
                    self.decoder
                        .forward_token_paged(self.tokens[pos], pos, lease.caches_mut(), &store)
                        .expect("the row's pages were reserved at admission")
                }
            };
        }
        self.tokens_processed = end;
        self.is_done()
    }

    /// Consumes a finished prefill into the pieces a decode row needs:
    /// KV caches, the logits the first token is sampled from, and the
    /// position the first generated token occupies.
    /// Hands the row over to decode, returning the prompt ids too.
    ///
    /// The ids used to be dropped here, which is why the batched path
    /// could ADOPT a radix prefix and never PUBLISH one: publishing
    /// needs the whole token sequence and the row no longer had it. The
    /// private generate loop kept them and published; the batcher did
    /// not, so prefix sharing under `FERROX_CONTINUOUS_BATCHING=1` was
    /// adopt-only against a tree nothing filled.
    pub(super) fn into_decode_start(self) -> (RowKv, Vec<f32>, usize, Vec<usize>) {
        debug_assert!(self.is_done(), "prefill must finish before decoding");
        (self.kv, self.logits, self.tokens_processed, self.tokens)
    }
}

/// An accepted job whose prompt is still being prefilled: the
/// resumable `PrefillState` plus everything the decode row will need
/// once the prompt is through.
pub(super) struct Prefill {
    pub(super) state: PrefillState,
    /// The *real* prompt length, for `Usage`. Deliberately not
    /// `state.tokens.len()`, which is 1 for an empty prompt.
    pub(super) prompt_tokens: usize,
    pub(super) params: GenerationParams,
    pub(super) stop_tokens: StopTokens,
    pub(super) reply: Sender<BatcherEvent>,
    pub(super) abort: AbortId,
    /// Blocks reserved at admission; carried into the `Slot` so the
    /// reservation survives the prefill-to-decode handover.
    pub(super) blocks: usize,
}

impl Prefill {
    pub(super) fn into_slot(self) -> Slot {
        let Prefill {
            state,
            prompt_tokens,
            params,
            stop_tokens,
            reply,
            abort,
            blocks,
        } = self;
        let (kv, logits, pos, prompt_ids) = state.into_decode_start();
        Slot {
            kv,
            pos,
            logits,
            sample: SampleState::new(params.seed),
            generated_ids: Vec::with_capacity(params.max_tokens),
            prompt_ids,
            visible: String::new(),
            stops: StopMatcher::new(&params.stop, &params.stop_token_ids),
            prompt_tokens,
            max_tokens: params.max_tokens,
            stop_tokens,
            params,
            reply,
            finish: None,
            error: None,
            abort,
            blocks,
        }
    }
}
