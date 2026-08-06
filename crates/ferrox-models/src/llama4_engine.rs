//! Llama 4 dedicated stack stub — not generic GQA.
//!
//! Real checkpoints use MoE + a non-generic attention graph (llama.cpp
//! `LLM_ARCH_LLAMA4`). Required tensors (names from pinned llama.cpp
//! `LLM_TENSOR_NAMES` / `llama4.cpp`), not implemented here:
//!
//! - `token_embd.weight`, `output_norm.weight`, `output.weight`
//! - Per layer: `blk.{i}.attn_norm.weight`, `blk.{i}.ffn_norm.weight`
//! - MoE FFN: `blk.{i}.ffn_gate_inp.weight`, `ffn_gate_exps.weight`,
//!   `ffn_up_exps.weight`, `ffn_down_exps.weight`, `ffn_exp_probs_b.bias`
//! - Llama-4-specific attention projections (not plain GQA `attn_q`/`attn_k`)
//!
//! Fail-closed via [`Self::reject`] until a real loader + engine land.

use thiserror::Error;

#[derive(Debug, Error)]
#[error("llama4 dedicated engine not implemented: {reason}")]
pub struct Llama4Unavailable {
    pub reason: &'static str,
}

pub struct Llama4Engine {
    pub arch: String,
}

impl Llama4Engine {
    pub fn reject(_arch: &str) -> Result<(), Llama4Unavailable> {
        Err(Llama4Unavailable {
            reason: "llama4 MoE / non-generic graph not yet implemented (see llama4_engine.rs)",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_fail_closed() {
        assert!(Llama4Engine::reject("llama4").is_err());
    }
}
