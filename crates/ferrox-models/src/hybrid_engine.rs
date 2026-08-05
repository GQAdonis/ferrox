//! Hybrid attn+SSM engine stub (Jamba / LFM2 / Nemotron-H / …).
//!
//! Fail-closed at load. Layer scheduler will live here when implemented.

use thiserror::Error;

#[derive(Debug, Error)]
#[error("hybrid engine not implemented for architecture {arch}")]
pub struct HybridUnavailable {
    pub arch: String,
}

pub struct HybridEngine {
    pub arch: String,
}

impl HybridEngine {
    pub fn reject(arch: &str) -> Result<(), HybridUnavailable> {
        Err(HybridUnavailable {
            arch: arch.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_fail_closed() {
        assert!(HybridEngine::reject("jamba").is_err());
    }
}
