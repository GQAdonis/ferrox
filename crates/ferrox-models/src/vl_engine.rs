//! Multimodal / VL serve stub (Qwen2-VL / CogVLM / …).
//!
//! Vision primitives for Kimi MoonViT live in [`crate::vision`]; GGUF VL
//! architectures remain `DeferredMultimodal` in the capability registry
//! until a projection + chat-template path is wired.

use thiserror::Error;

#[derive(Debug, Error)]
#[error("multimodal/VL engine not implemented for architecture {arch}")]
pub struct VlUnavailable {
    pub arch: String,
}

pub struct VlEngine {
    pub arch: String,
}

impl VlEngine {
    pub fn reject(arch: &str) -> Result<(), VlUnavailable> {
        Err(VlUnavailable {
            arch: arch.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_fail_closed() {
        assert!(VlEngine::reject("qwen2vl").is_err());
    }
}
