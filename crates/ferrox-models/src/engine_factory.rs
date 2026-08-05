//! Load-time engine selection (llama.cpp `llama_model_*` factory analogue).
//!
//! GGUF text-generation architectures that the generic [`crate::decoder::Decoder`]
//! can run are routed there. Dedicated stacks (MLA / DSA / recurrent /
//! T5 encoder-decoder) are selected here and must not accumulate
//! `if arch` branches inside matmul / attention kernels.

use crate::capability::{resolve_profile, ArchPath, DecoderFamily};
use crate::decoder::Decoder;
use crate::engine::{Engine, Glm52Engine, KimiEngine, MlaEngine};
use thiserror::Error;

/// Why a GGUF cannot be served by the currently compiled engine set.
#[derive(Debug, Error)]
pub enum EngineSelectError {
    #[error("architecture {0:?} is outside the text-generation scope ({1})")]
    OutOfScope(String, &'static str),
    #[error("architecture {0:?} requires a dedicated engine that is not wired for serve yet: {1}")]
    DedicatedUnavailable(String, &'static str),
    #[error("unknown or unsupported architecture {0:?}")]
    Unknown(String),
}

/// Result of load-time engine selection for a GGUF architecture string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedEngineKind {
    /// Standard / Phi / Gemma / Qwen3 GQA path via [`Decoder`].
    GenericDecoder,
    /// Kimi / GLM / DeepSeek dedicated stacks (separate loaders).
    DedicatedStack,
    /// MLA memory engines (DeepSeek2 / Mistral4) — fail-closed for generic.
    Mla,
    /// Recurrent / hybrid SSM families — fail-closed until wired.
    RecurrentHybrid,
    /// T5 encoder-decoder — fail-closed until wired.
    EncoderDecoder,
}

/// Resolve which engine kind a GGUF `general.architecture` should use.
pub fn select_engine_kind(arch: &str) -> Result<SelectedEngineKind, EngineSelectError> {
    let profile =
        resolve_profile(arch).ok_or_else(|| EngineSelectError::Unknown(arch.to_string()))?;
    match profile.path {
        ArchPath::GenericGqa { .. } | ArchPath::TestFixture { .. } => {
            Ok(SelectedEngineKind::GenericDecoder)
        }
        ArchPath::Deferred { reason } => {
            Err(EngineSelectError::OutOfScope(arch.to_string(), reason))
        }
        ArchPath::DedicatedOnly { reason } => match profile.family {
            DecoderFamily::Dedicated => Ok(SelectedEngineKind::DedicatedStack),
            DecoderFamily::Mla => Ok(SelectedEngineKind::Mla),
            DecoderFamily::Hybrid | DecoderFamily::Recurrent => {
                Ok(SelectedEngineKind::RecurrentHybrid)
            }
            DecoderFamily::EncoderDecoder => Ok(SelectedEngineKind::EncoderDecoder),
            _ => Err(EngineSelectError::DedicatedUnavailable(
                arch.to_string(),
                reason,
            )),
        },
    }
}

/// Fail-closed check used by `ferrox-server` before constructing a
/// [`Decoder`] for architectures that need another engine.
pub fn ensure_generic_decoder(arch: &str) -> Result<(), EngineSelectError> {
    match select_engine_kind(arch)? {
        SelectedEngineKind::GenericDecoder => Ok(()),
        SelectedEngineKind::DedicatedStack => Err(EngineSelectError::DedicatedUnavailable(
            arch.to_string(),
            "use the dedicated Kimi/GLM/DeepSeek loader, not generic Decoder",
        )),
        SelectedEngineKind::Mla => Err(EngineSelectError::DedicatedUnavailable(
            arch.to_string(),
            "MLA engine exists (ferrox_models::MlaEngine) but DeepSeek-2/Mistral-4 GGUF \
             weight loading is not wired yet — fail-closed until a real checkpoint receipt",
        )),
        SelectedEngineKind::RecurrentHybrid => {
            let _ = crate::recurrent_engine::RecurrentEngine::reject(arch);
            let _ = crate::hybrid_engine::HybridEngine::reject(arch);
            Err(EngineSelectError::DedicatedUnavailable(
                arch.to_string(),
                "recurrent/hybrid SSM engine stub present — not yet on the serve path",
            ))
        }
        SelectedEngineKind::EncoderDecoder => {
            let _ = crate::t5_engine::T5Engine::reject(arch);
            Err(EngineSelectError::DedicatedUnavailable(
                arch.to_string(),
                "T5 encoder-decoder engine stub present — not yet on the serve path",
            ))
        }
    }
}

/// Type-erased serve handle: today ordinary GGUFs use [`Decoder`];
/// Kimi/GLM/MLA use dedicated engines once loaders succeed.
pub enum ServedEngine {
    Decoder(Box<Decoder>),
    Kimi(KimiEngine),
    Glm52(Glm52Engine),
    Mla(MlaEngine),
}

impl ServedEngine {
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Decoder(d) => Engine::vocab_size(d.as_ref()),
            Self::Kimi(k) => Engine::vocab_size(k),
            Self::Glm52(g) => Engine::vocab_size(g),
            Self::Mla(m) => Engine::vocab_size(m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_and_qwen3_select_generic_decoder() {
        assert_eq!(
            select_engine_kind("llama").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
        assert_eq!(
            select_engine_kind("qwen3").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
        assert_eq!(
            select_engine_kind("gemma3").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
        assert_eq!(
            select_engine_kind("phi3").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
        assert_eq!(
            select_engine_kind("mixtral").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
    }

    #[test]
    fn mamba_is_recurrent_fail_closed_for_generic() {
        assert!(matches!(
            select_engine_kind("mamba2").unwrap(),
            SelectedEngineKind::RecurrentHybrid
        ));
        assert!(ensure_generic_decoder("mamba2").is_err());
    }

    #[test]
    fn deepseek2_is_mla_not_generic() {
        assert_eq!(
            select_engine_kind("deepseek2").unwrap(),
            SelectedEngineKind::Mla
        );
        assert!(ensure_generic_decoder("deepseek2").is_err());
    }
}
