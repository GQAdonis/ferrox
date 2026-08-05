//! Hybrid attn+SSM engine stub (Jamba / LFM2 / Nemotron-H / Qwen3.5 GDN / …).
//!
//! Fail-closed at load via [`HybridEngine::reject`] (used by
//! [`crate::engine_factory`] for `nemotron_h` and other hybrid arches).
//! The Qwen-style GDN primitive lives in [`crate::gdn`] with unit tests;
//! GGUF load into this engine is not wired yet.

use thiserror::Error;

#[derive(Debug, Error)]
#[error("hybrid engine not implemented for architecture {arch}")]
pub struct HybridUnavailable {
    pub arch: String,
}

/// Placeholder hybrid serve handle.
///
/// Holds the GGUF arch string and an optional note. Until a loader exists,
/// construction is via [`Self::stub`]; the factory still calls [`Self::reject`].
pub struct HybridEngine {
    pub arch: String,
    /// e.g. that [`crate::gdn`] exists but GGUF load is not wired.
    pub note: Option<String>,
}

impl HybridEngine {
    /// Fail-closed entry used by the engine factory until GGUF→GDN load lands.
    pub fn reject(arch: &str) -> Result<(), HybridUnavailable> {
        Err(HybridUnavailable {
            arch: arch.to_string(),
        })
    }

    /// Non-serving stub: records arch + note that the GDN primitive is in
    /// `gdn.rs` but not yet loaded from GGUF.
    pub fn stub(arch: &str) -> Self {
        Self {
            arch: arch.to_string(),
            note: Some(
                "GDN primitive exists in gdn.rs; GGUF load not wired — factory still reject()"
                    .into(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_fail_closed() {
        assert!(HybridEngine::reject("jamba").is_err());
        assert!(HybridEngine::reject("nemotron_h").is_err());
    }

    #[test]
    fn stub_holds_arch_and_gdn_note() {
        let eng = HybridEngine::stub("qwen35");
        assert_eq!(eng.arch, "qwen35");
        let note = eng.note.expect("note");
        assert!(note.contains("gdn.rs"));
        assert!(note.contains("GGUF"));
    }
}
