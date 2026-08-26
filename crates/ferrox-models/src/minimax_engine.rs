//! MiniMax M2/M3 dedicated stack stub — not generic GQA.
//!
//! Real checkpoints tag `minimax-m2` / `minimax-m3` with 256-expert
//! sigmoid MoE and MTP (multi-token prediction) draft heads. Required
//! tensors (llama.cpp `minimax*.cpp` graph), not implemented here:
//!
//! - Standard emb/norm/output head
//! - MoE: `ffn_gate_inp`, `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`,
//!   `ffn_exp_probs_b.bias` (sigmoid / `noaux_tc` routing)
//! - MTP: `num_nextn_predict_layers` draft-head tensors (`nextn.*` in
//!   llama.cpp) — see `docs/CLI.md` `--mtp` (honest fail until loaded)
//!
//! Fail-closed via [`Self::reject`] until loader + engine land.

use thiserror::Error;

#[derive(Debug, Error)]
#[error("MiniMax dedicated engine not implemented: {reason}")]
pub struct MinimaxUnavailable {
    pub reason: &'static str,
}

pub struct MinimaxEngine {
    pub arch: String,
}

impl MinimaxEngine {
    /// Refuses, naming what exists and what does not.
    ///
    /// The block-sparse SELECTION is ported and tested
    /// ([`ferrox_core::block_sparse`]): which 128-token KV blocks a
    /// query may read, per KV head, with the force-included first and
    /// newest blocks that keep a selection from ever being empty. What
    /// is absent is everything around it -- the loader, the 256-expert
    /// sigmoid MoE, and the MTP draft heads -- so there is nothing for
    /// that selection to select over yet.
    ///
    /// Saying which half exists matters: a bare "not implemented"
    /// invites the next reader to re-port the selection rule that is
    /// already here and already covered.
    pub fn reject(_arch: &str) -> Result<(), MinimaxUnavailable> {
        Err(MinimaxUnavailable {
            reason: "MiniMax needs a loader, 256-expert sigmoid MoE routing and MTP draft heads, \
                     none of which exist yet. Its block-sparse attention SELECTION is ported and \
                     tested (ferrox_core::block_sparse); only the engine around it is missing",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_fail_closed() {
        assert!(MinimaxEngine::reject("minimax-m2").is_err());
    }
}
