//! `ferrox-vulkan`: the Vulkan **beachhead**, not a backend.
//!
//! # What this crate is
//!
//! One question, answered end to end: *can ferrox reach a Vulkan
//! device from Rust, upload a quantized weight verbatim, run one
//! compute shader, and read back a correct answer?* That is the
//! `vulkan-beachhead` GO/NO-GO in `docs/plans/roadmap.md`'s
//! `d-hardware-reach`, and it is deliberately scoped below a backend:
//! **one kernel, one quant kind, no integration**.
//!
//! It is NOT wired into `ferrox-core`. `WeightMatrix::apply_gpu` still
//! knows about exactly two backends, because adding a third to the
//! seam as it stands today means a third copy of four hand-kept tables
//! and 214 `#[cfg]` sites -- which is `backend-seam-refactor`, the item
//! the roadmap puts *before* this one. The verdict document records
//! what that seam would have to become.
//!
//! # What has actually run
//!
//! See `docs/plans/vulkan-beachhead-verdict.md`. The rule this repo
//! holds CUDA to applies here unchanged: nothing in this crate may be
//! described as a measured capability in `docs/FEATURES.md` or
//! `docs/MODELS.md` on the strength of compiling.
//!
//! # Layout
//!
//! - [`spirv`] -- a minimal SPIR-V word emitter, ~250 lines, no
//!   external shader compiler and no build script.
//! - [`q8_0_shader`] -- the Q8_0 matvec compute shader, emitted from
//!   Rust.
//! - [`q8_0_reference`] -- its scalar twin, checked against
//!   `ferrox_quant` and the `half` crate.
//! - [`device`] -- the `ash` host layer (`--features vulkan`):
//!   instance, device, buffers, descriptors, dispatch, readback.
//!
//! The first three compile and are tested **unconditionally**, on a
//! machine with no Vulkan driver at all. A broken shader fails
//! `cargo test -p ferrox-vulkan` with no GPU, which is strictly more
//! than `ferrox-cuda` can say for its kernels (NVRTC compiles at
//! runtime).

pub mod q8_0_reference;
pub mod q8_0_shader;
pub mod spirv;

#[cfg(feature = "vulkan")]
pub mod device;

#[cfg(feature = "vulkan")]
pub mod dispatch;
