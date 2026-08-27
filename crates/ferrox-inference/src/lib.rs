//! Ferrox — a pure-Rust GGUF / MoE inference engine.
//!
//! This crate is a **facade**. It contains no logic of its own: it
//! re-exports the workspace under one name so a dependent writes one
//! line in `Cargo.toml` instead of six, and so the project is findable
//! on crates.io (the name `ferrox` belongs to an unrelated crate).
//!
//! The command-line tools are not here. `cargo install ferrox-cli`
//! installs the `ferrox` binary; `ferrox-server` is the
//! OpenAI-compatible HTTP server. Shipping a second binary called
//! `ferrox` from this crate would just fight the first one over
//! `~/.cargo/bin`.
//!
//! # Layout
//!
//! The stack, bottom to top:
//!
//! | Module | Crate | What it is |
//! |---|---|---|
//! | [`gguf`] | `ferrox-gguf` | GGUF mmap reader, sharded checkpoints |
//! | [`quant`] | `ferrox-quant` | Block layouts and fused dequant+dot |
//! | [`safetensors`] | `ferrox-safetensors` | SafeTensors mmap reader |
//! | [`core`] | `ferrox-core` | Tensor ops, RoPE, GQA, KV cache |
//! | [`edge`] | `ferrox-edge` | Serving policy: `q*` split, radix prefix caches, output parsers |
//! | [`moe`] | `ferrox-moe` | Expert routing and dispatch |
//! | [`models`] | `ferrox-models` | Loaders and decoder stacks |
//! | [`api`] | `ferrox-api` | Route constants + wire DTOs (feature `api`) |
//!
//! # Example
//!
//! ```no_run
//! use ferrox_inference::gguf::ShardedGguf;
//!
//! let file = ShardedGguf::open("model.gguf")?;
//! println!("{} tensors", file.tensor_count());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Features
//!
//! - `metal` — Apple Metal kernels. Apple Silicon only.
//! - `cuda` — CUDA/NVRTC kernels. Needs a CUDA toolkit at build time.
//!   Held to "must compile": there is no pinned benchmark host and no
//!   published receipts for it. See `docs/FEATURES.md`.
//! - `api` — pull in `ferrox-api` for client-side route constants.
//!
//! Neither GPU feature is on by default, because both are wrong to
//! assume: `metal` does not build off Apple Silicon and `cuda` needs a
//! toolkit that most machines do not have.

#![forbid(unsafe_code)]

pub use ferrox_core as core;
pub use ferrox_gguf as gguf;
pub use ferrox_models as models;
pub use ferrox_moe as moe;
pub use ferrox_quant as quant;
pub use ferrox_safetensors as safetensors;

#[cfg(feature = "api")]
pub use ferrox_api as api;

/// The workspace version this facade was built from.
///
/// Every crate in the workspace shares one version, so this is the
/// version of the whole engine, not just of the facade.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    /// The facade is only useful if it actually re-exports. These
    /// paths would fail to COMPILE -- not merely assert false -- if a
    /// module were dropped from `lib.rs`, which is the failure mode
    /// worth catching: a facade that silently stops re-exporting one
    /// layer looks fine until a dependent upgrades and cannot build.
    #[allow(dead_code)]
    fn every_layer_is_reachable_through_the_facade() {
        let _: Option<super::gguf::ShardedGguf> = None;
        let _: Option<super::quant::QuantError> = None;
        let _: Option<super::core::cache::KvCache> = None;
        let _: Option<super::safetensors::SafetensorsFile> = None;
        let _ = std::mem::size_of::<super::models::Decoder>();
        let _ = std::mem::size_of::<super::moe::MoeLayerConfig>();
    }

    #[test]
    fn version_is_the_workspace_version() {
        assert_eq!(super::VERSION, env!("CARGO_PKG_VERSION"));
        assert!(!super::VERSION.is_empty());
    }
}
