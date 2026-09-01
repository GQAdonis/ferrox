//! The seam for encoder-only (embedding) models, which is deliberately
//! **not** [`crate::engine::Engine`].
//!
//! # Why a second trait and not a variant of the first
//!
//! [`crate::engine::Engine`] is `forward_token(token_id, pos, &mut
//! State) -> Vec<f32>`: one token in, one logit vector out, carrying
//! per-layer state forward. Every part of that signature is an
//! autoregression assumption, and a BERT encoder violates all of them:
//!
//! * **There is no state to carry.** Attention is bidirectional, so
//!   token 0's output depends on token 7. Nothing can be computed until
//!   the whole sequence is present, and nothing computed for one
//!   sequence is reusable for the next. A KV cache is not merely
//!   unnecessary here, it is meaningless — there is no "next token" for
//!   a cached key to be attended to by.
//! * **There are no logits.** The result is `n_tokens × n_embd` hidden
//!   states (llama.cpp's `res->t_embd`); this checkpoint has no output
//!   head at all, and `token_embd.weight` is not tied to one.
//! * **`pos` is not a cursor, it is an index into a learned table.**
//!   BERT adds `position_embd.weight[i]` to the token embedding rather
//!   than rotating Q/K, which is why the sequence length is hard-capped
//!   by `n_ctx_train` instead of merely degrading past it.
//!
//! Forcing that through `Engine` would mean a `State` that is a
//! pretend-cache, a `forward_token` that can only be called with the
//! last position after a hidden batch call, and a `vocab_size` that has
//! no meaning. The Kimi comment on `Engine` already records the cost of
//! bending a trait around a model it does not fit; this is the same
//! judgement, made before rather than after.
//!
//! So: [`TextEncoder`] is sequence-in, matrix-out, stateless. What the
//! two seams *do* share is pooling ([`crate::pooling`]), which is why
//! that lives in its own module and not in either of them.

use crate::pooling::{pool, PoolingError, PoolingType};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("an encoder needs at least one token; got an empty sequence")]
    EmptySequence,
    #[error(
        "sequence of {got} tokens exceeds the {max} learned position embeddings this \
         checkpoint carries ({arch}.context_length). A learned position table cannot be \
         extrapolated the way RoPE can, so this is a hard limit, not a quality cliff — \
         truncate the input or use a longer-context embedding model"
    )]
    TooLong {
        got: usize,
        max: usize,
        arch: String,
    },
    #[error("token id {id} is outside this checkpoint's {vocab_size}-entry vocabulary")]
    TokenOutOfRange { id: u32, vocab_size: usize },
    #[error(transparent)]
    Pooling(#[from] PoolingError),
}

/// A model that turns a whole token sequence into hidden states in one
/// pass, with no carried state and no logits.
pub trait TextEncoder {
    /// Width of one hidden-state row, and of the pooled embedding.
    fn n_embd(&self) -> usize;

    /// Longest sequence this checkpoint can represent. For a learned
    /// position table this is the table's height, and exceeding it is
    /// an error rather than a degradation.
    fn n_ctx_train(&self) -> usize;

    /// What the checkpoint's own `{arch}.pooling_type` said.
    fn pooling_type(&self) -> PoolingType;

    /// Wraps a tokenizer's pieces in whatever the *model* requires
    /// around them — for BERT, `[CLS] … [SEP]`.
    ///
    /// This is on the encoder rather than the tokenizer because ferrox's
    /// tokenizers deliberately encode text only (they are checked
    /// token-for-token against `llama_tokenize(..., add_special =
    /// false, ...)`), while llama.cpp keeps `add_special` in the vocab
    /// and applies it here. The default adds nothing, so an encoder that
    /// genuinely needs no wrapper does not have to say so.
    fn wrap_special(&self, pieces: &[u32]) -> Vec<u32> {
        pieces.to_vec()
    }

    /// `n_tokens × n_embd` hidden states, in row order.
    fn encode_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>, EncodeError>;

    /// [`Self::encode_tokens`] followed by the checkpoint's own pooling.
    /// Not L2-normalized — see [`crate::pooling::pool`].
    fn embed_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>, EncodeError> {
        let hidden = self.encode_tokens(tokens)?;
        Ok(pool(&hidden, self.n_embd(), self.pooling_type())?)
    }
}
