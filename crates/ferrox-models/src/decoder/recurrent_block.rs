//! The recurrent half of a hybrid layer, at the site attention occupies,
//! on the three cache backings.
//!
//! Two blocks stand where attention stands on a zero-KV layer
//! (`crate::layer_shapes::AttnShape`): LFM2's short convolution, whose
//! state is a window of its inputs and lives as the layer's KV history
//! (`crate::shortconv`), and the Mamba-2 block, whose state is a
//! reduction and lives as a `RecurrentState` beside the cache
//! (`crate::mamba2`, `ferrox_core::recurrent_state`). Each body takes
//! its state as a closure or a `&mut`; this file is the ONE place the
//! backing is matched for either, so a backing cannot pad, index, push
//! or carry the state differently from the others -- the same reason
//! [`super::attn_block::KvStep`] exists for attention.

use ferrox_core::cache::KvCache;
use ferrox_core::recurrent_state::RecurrentState;

use super::attn_block::KvStep;
use super::{Decoder, LayerWeights};
use crate::layer_shapes::AttnShape;
use crate::shortconv::window_from_history;

impl Decoder {
    /// `rows` consecutive positions of ONE sequence through layer
    /// `layer_idx`'s recurrent block. `normed` is `[rows][n_embd]`, the
    /// `attn_norm` output; the result is the branch's contribution to
    /// the residual, which the caller adds (the contract `attn_block`
    /// has). Dispatches on the layer's SHAPE, so a layer whose weights
    /// and shape disagree panics here rather than running the wrong
    /// block.
    pub(crate) fn recurrent_block(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        rows: usize,
        kv: KvStep<'_>,
    ) -> Vec<f32> {
        match self.config.layer_shape(layer_idx).attention {
            AttnShape::ShortConv => self.shortconv_block(layer_idx, layer, normed, rows, kv),
            AttnShape::Mamba2 => self.mamba2_block(layer_idx, layer, normed, rows, kv),
            other => unreachable!("layer {layer_idx} is {other:?}, not a recurrent block"),
        }
    }

    /// LFM2's short convolution. Each row's `bx` is pushed to the
    /// sequence's layer cache as its one "K" row
    /// (`AttnShape::cache_geometry`) before the window is read, so the
    /// window's newest entry is this row and the state after the call
    /// is the history llama.cpp would carry forward.
    fn shortconv_block(
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

    /// The Mamba-2 block. The state is the cache's `recurrent` slot,
    /// created at this layer's size on the sequence's first token
    /// (zeros, as `build_rs` zeroes a new sequence's); after the rows
    /// run, the cache is advanced by `rows` EMPTY positions so its
    /// `positions()` / `seq_len()` still says how far the sequence has
    /// got, which is what every consumer of a per-layer cache reads.
    fn mamba2_block(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        rows: usize,
        kv: KvStep<'_>,
    ) -> Vec<f32> {
        let block = layer
            .attn
            .mamba2
            .as_ref()
            .unwrap_or_else(|| panic!("layer {layer_idx} is Mamba2-shaped but has no weights"));
        let eps = self.config.rms_norm_eps;
        let run = |state: &mut Option<RecurrentState>| {
            let state = state.get_or_insert_with(|| block.zero_state());
            block.forward_rows(normed, rows, state, eps)
        };
        match kv {
            KvStep::Decode(cache) | KvStep::Batched(cache) => {
                let out = run(&mut cache.recurrent);
                cache
                    .advance_len(rows)
                    .expect("unbounded/planned KvCache growth is infallible");
                out
            }
            KvStep::Paged { cache, stores } => {
                let out = run(&mut cache.recurrent);
                let mut store = stores.write(layer_idx);
                for _ in 0..rows {
                    cache
                        .push(&mut store, &[], &[])
                        .expect("every caller reserves this row's pages before the stack runs");
                }
                out
            }
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
