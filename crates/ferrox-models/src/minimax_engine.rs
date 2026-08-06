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
    pub fn reject(_arch: &str) -> Result<(), MinimaxUnavailable> {
        Err(MinimaxUnavailable {
            reason:
                "MiniMax 256-expert sigmoid MoE + MTP not yet implemented (see minimax_engine.rs)",
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
