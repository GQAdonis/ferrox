//! What a layer with no KV history carries between tokens instead.
//!
//! A Mamba layer (and every other recurrent block llama.cpp keeps in
//! `llama_memory_recurrent`) has no per-position rows to attend over;
//! it has a fixed-size state that the next token reads and overwrites.
//! LFM2's short convolution is the exception that proves the rule: its
//! state IS the last `l_cache - 1` inputs, so `ferrox_models::shortconv`
//! keeps it as the layer's KV history and needs nothing here. A Mamba
//! state is a reduction over the whole prefix, not a window of it, and
//! that is the one property every consumer of a per-layer cache has to
//! know about:
//!
//! - it CLONES with the cache (a prefix-cache fork is a fork of the
//!   state), and CLEARS with it;
//! - it cannot be TRUNCATED to a middle position. llama.cpp's
//!   `llama_memory_recurrent::seq_rm` refuses a `p0 > 0` for the same
//!   reason and its server re-prefills. So [`KvCache::truncate`] on a
//!   cache that holds one refuses anything but "to zero" or "to where
//!   it is", and the callers that roll back -- the prefix cache,
//!   speculative verification, the draft model, the whole-response
//!   cache's back-off -- ask [`KvCache::can_truncate_to`] first or are
//!   fenced off the model.
//!
//! The buffers are flat and the LAYER owns their geometry (its weights
//! say what `d_conv`, the conv width and the scan dims are), so this
//! type cannot disagree with the block about a shape: it is created by
//! the block, on first use, at the size the block asks for.
//!
//! [`KvCache::truncate`]: crate::cache::KvCache::truncate
//! [`KvCache::can_truncate_to`]: crate::cache::KvCache::can_truncate_to

/// One sequence's state for one recurrent layer.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentState {
    /// The conv window, `[d_conv - 1][width]`, oldest row first
    /// (`llama_hparams::n_embd_r`).
    pub conv: Vec<f32>,
    /// The SSM state, `[n_head][head_dim][d_state]`
    /// (`llama_hparams::n_embd_s`).
    pub ssm: Vec<f32>,
}

impl RecurrentState {
    /// A fresh sequence's state: zeros, as `build_rs` zeroes a new
    /// sequence's (`llama-graph.cpp`, `llm_graph_input_rs`).
    pub fn zeros(conv_len: usize, ssm_len: usize) -> Self {
        Self {
            conv: vec![0.0; conv_len],
            ssm: vec![0.0; ssm_len],
        }
    }

    /// Bytes this state holds.
    pub fn bytes(&self) -> usize {
        (self.conv.len() + self.ssm.len()) * std::mem::size_of::<f32>()
    }
}
