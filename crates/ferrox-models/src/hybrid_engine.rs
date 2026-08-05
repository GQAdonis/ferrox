//! Hybrid attn+SSM engine stub (Jamba / LFM2 / Nemotron-H / Qwen3.5 GDN / …).
//!
//! Fail-closed at load via [`HybridEngine::reject`] (used by
//! [`crate::engine_factory`] for `nemotron_h` and other hybrid arches).
//! Qwen-style GDN math: [`crate::gdn`]. GGUF hparams + GDN weight load
//! skeleton: [`crate::hybrid_gguf_loader`] (`try_load` still
//! `UnsupportedFeature` until this engine assembles layers for serve).

use thiserror::Error;

#[derive(Debug, Error)]
#[error("hybrid engine not implemented for architecture {arch}")]
pub struct HybridUnavailable {
    pub arch: String,
}

/// Placeholder hybrid serve handle.
///
/// Holds the GGUF arch string and an optional note. Construction is via
/// [`Self::stub`]; the factory still calls [`Self::reject`]. GGUF→GDN
/// weight probing lives in [`crate::hybrid_gguf_loader::try_load`].
#[derive(Debug)]
pub struct HybridEngine {
    pub arch: String,
    /// e.g. that GDN + loader skeleton exist but assemble/serve is incomplete.
    pub note: Option<String>,
}

impl HybridEngine {
    /// Fail-closed entry used by the engine factory until HybridEngine assemble lands.
    pub fn reject(arch: &str) -> Result<(), HybridUnavailable> {
        Err(HybridUnavailable {
            arch: arch.to_string(),
        })
    }

    /// Non-serving stub: records arch + note that GDN + hybrid_gguf_loader
    /// exist but serve assemble is not wired.
    pub fn stub(arch: &str) -> Self {
        Self {
            arch: arch.to_string(),
            note: Some(
                "GDN + hybrid_gguf_loader skeleton exist; HybridEngine assemble/serve not wired — factory still reject()"
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
        assert!(note.contains("GDN"));
        assert!(note.contains("hybrid_gguf_loader"));
    }
}
