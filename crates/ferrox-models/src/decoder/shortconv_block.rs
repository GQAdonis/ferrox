//! The short-convolution half of an LFM2 layer, at the site attention
//! occupies, on the three cache backings.
//!
//! [`crate::shortconv::ShortConv::forward_rows`] is the arithmetic and
//! takes the state as a closure; this file is the closure, spelled once
//! per backing in ONE function so a backing cannot pad, index or push
//! differently from the others -- the same reason
//! [`super::attn_block::KvStep`] exists for attention.

use ferrox_core::cache::KvCache;

use super::attn_block::KvStep;
use super::{Decoder, LayerWeights};
use crate::shortconv::window_from_history;

impl Decoder {
    /// `rows` consecutive positions of ONE sequence through layer
    /// `layer_idx`'s short convolution. `normed` is `[rows][n_embd]`,
    /// the `attn_norm` output; the result is the branch's contribution
    /// to the residual, which the caller adds (the contract
    /// `attn_block` has).
    ///
    /// Each row's `bx` is pushed to the sequence's layer cache as its
    /// one "K" row (`AttnShape::cache_geometry`) before the window is
    /// read, so the window's newest entry is this row and the state
    /// after the call is the history llama.cpp would carry forward.
    pub(crate) fn shortconv_block(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        rows: usize,
        kv: KvStep<'_>,
    ) -> Vec<f32> {
        let conv =
            layer.attn.shortconv.as_ref().unwrap_or_else(|| {
                panic!("layer {layer_idx} is ShortConv-shaped but has no weights")
            });
        let (l_cache, n_embd) = (conv.l_cache, conv.hidden_dim());
        match kv {
            KvStep::Decode(cache) | KvStep::Batched(cache) => {
                conv.forward_rows(normed, rows, |bx| {
                    contiguous_step(cache, bx, l_cache, n_embd)
                })
            }
            KvStep::Paged { cache, stores } => conv.forward_rows(normed, rows, |bx| {
                {
                    let mut store = stores.write(layer_idx);
                    cache
                        .push(&mut store, bx, &[])
                        .expect("every caller reserves this row's pages before the stack runs");
                }
                let store = stores.read(layer_idx);
                let table = cache.block_table();
                let block = store.block_size();
                window_from_history(l_cache, n_embd, cache.seq_len(), |i| {
                    store.k_row(table[i / block], i % block)
                })
            }),
        }
    }
}

/// Push one `bx` row to a contiguous cache and read the window back.
fn contiguous_step(cache: &mut KvCache, bx: &[f32], l_cache: usize, n_embd: usize) -> Vec<f32> {
    cache
        .push(bx, &[])
        .expect("unbounded/planned KvCache growth is infallible");
    let rows = cache.rows();
    window_from_history(l_cache, n_embd, rows, |i| {
        &cache.k[i * n_embd..(i + 1) * n_embd]
    })
}
