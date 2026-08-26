//! ferrox-cuda: hardware capability detection (always compiled, always
//! tested) plus an optional, feature-gated CUDA execution path.
//!
//! Build without any GPU support (the default): `cargo build -p ferrox-cuda`.
//! Build with the CUDA scaffolding included: `cargo build -p ferrox-cuda --features cuda`.
//!
//! The `cuda` feature compiles cleanly in this development sandbox
//! (which has neither a CUDA toolkit nor a GPU) because `cudarc` is
//! configured for dynamic loading -- the driver and NVRTC libraries
//! are `dlopen`'d at runtime, not linked at build time. That means
//! "this crate compiles with `--features cuda`" is a true, checked
//! fact. It does **not** mean the CUDA kernels in `gpu.rs` have ever
//! executed successfully; see that module's docs for exactly what has
//! and has not been verified.

pub mod capability;

#[cfg(feature = "cuda")]
pub mod attn;
// Not feature-gated: the slot-index arithmetic inside is the one part
// of the device copy path a compiler cannot check, so it is compiled
// and tested on every host. Only `CudaExpertPool` itself is behind
// `cuda`.
pub mod expert_pool;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod graph;

pub use capability::{HardwareProfile, SimdCaps};
