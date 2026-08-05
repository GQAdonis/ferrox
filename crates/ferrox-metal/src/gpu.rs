//! Real Metal compute dispatch for GGML matvec kernels, using
//! `objc2-metal`'s bindings to the system Metal framework (no separate
//! CUDA-toolkit-style SDK needed -- the framework ships with macOS).
//!
//! Five matvec kernels are implemented: `Q8_0`/`Q4_0`/`Q4_K`/`Q5_K`/
//! `Q6_K`, plus multi-activation matmul kernels
//! (`Q4_K_MATMUL_BATCH_KERNEL_SRC` / `Q6_K_MATMUL_BATCH_KERNEL_SRC`)
//! for prefill (`batch >= 2`) that reuse weight-block loads across an
//! `NB=4` activation tile — a simpler correct first cut vs full
//! ggml-metal `mul_mm` (no simdgroup_matrix tiles).
//!
//! **Verified directly on real hardware** (unlike `ferrox-cuda`, which
//! needed a rented GPU): this crate is developed on an Apple M2 Pro, so
//! all five `launch_q*_matvec_matches_cpu_reference` tests below have
//! actually been run against the real GPU, not just compiled -- see
//! each test's `#[ignore]` note for why they're still ignored by
//! default (CI/other contributors' machines may not have a
//! Metal-capable GPU at all, same reasoning as the CUDA hardware
//! tests). All five passed cleanly on the first real-hardware run, no
//! bug-fixing needed (unlike the CUDA-side K-quant history recorded in
//! `docs/MODELS.md`) -- `launch_q6_k_matvec_matches_cpu_reference`
//! specifically hit the same real degenerate case the CUDA test did
//! (pseudo-random block bytes decoding to a NaN `half` scale on one
//! row) and the GPU/CPU outputs agreed (both NaN), confirming
//! `assert_close_relative`'s NaN-vs-NaN handling is doing real work
//! here too, not dead code copied over unused.
//!
//! **Persistent device/pipeline/weight cache**: `shared_metal`/
//! `ensure_pipeline` below reuse one process-wide `MTLDevice` +
//! `MTLCommandQueue`, and cache one compiled `MTLComputePipelineState`
//! per kernel function name, instead of recreating them on every call
//! -- the same per-call overhead problem
//! `ferrox-cuda::gpu::shared_device`/`ensure_module_loaded` fixed for
//! CUDA. Quantized weight buffers are also cached by host pointer+length
//! (`resident_weight_buffer`) so decode does not re-upload multi-GB
//! matrices every token. f32 norm buffers are also cached by host
//! pointer+length via `resident_f32_buffer`. Q4_K/Q6_K kernels are
//! multi-row (4 rows per threadgroup) and Q5_K is NSG=2 / N_R0=1
//! (2 rows per TG — ggml keeps N_R0_Q5_K at 1 to avoid register spill);
//! Q8_0/Q4_0 stay one-row-per-TG. This needs an explicit
//! `unsafe impl Send + Sync` for
//! `SharedMetal`/`CachedPipeline`/`ResidentWeightBuffer`/`ResidentF32Buffer`
//! because
//! `objc2-metal`'s `Retained<ProtocolObject<dyn T>>` wrapper is
//! unconditionally `!Send`/`!Sync` (it holds a `NonNull` pointer, and
//! `NonNull` is `!Send`/`!Sync` regardless of what it points to, forcing
//! any wrapper to opt back in explicitly) -- see the safety comment on
//! those impls for why sharing these specific object kinds across
//! threads is sound. `MTLCommandBuffer`/`MTLComputeCommandEncoder` are
//! *not* included in the cache and are still created fresh per call,
//! because (unlike the device/queue/library/pipeline/weight buffers)
//! Apple documents those two types as requiring single-threaded,
//! single-use access -- exactly matching what `launch_matvec` already
//! does (a fresh command buffer/encoder every call, never stored).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use std::cell::Cell;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("no Metal device available on this machine")]
    NoDevice,
    #[error("Metal kernel failed to compile: {0}")]
    CompileFailed(String),
    #[error("Metal function `{0}` not found in compiled library")]
    FunctionNotFound(&'static str),
    #[error("Metal compute pipeline creation failed: {0}")]
    PipelineFailed(String),
    #[error("Metal buffer allocation failed")]
    BufferAllocFailed,
    #[error("Metal command buffer/encoder creation failed")]
    CommandFailed,
}

/// Thread-local pointer + length for a Metal-resident activation buffer
/// (normalized hidden after final_norm in the dense stack). When set,
/// [`launch_matvec_fused`] can skip re-uploading if `x` matches.
#[derive(Clone, Copy)]
struct ResidentActivation {
    /// Raw pointer to the MTLBuffer (not retained — caller owns).
    buf_ptr: *const ProtocolObject<dyn MTLBuffer>,
    /// Number of f32 elements.
    len: usize,
}

thread_local! {
    /// Holds a resident activation buffer from the dense stack (final_norm
    /// output in `x_buf`) so the next `output_head.apply_gpu` can reuse it
    /// without uploading. Cleared after first read.
    static RESIDENT_ACT: Cell<Option<ResidentActivation>> = const { Cell::new(None) };
}

/// Stores a resident activation buffer pointer for the current thread.
/// Used by [`crate::attn::launch_decode_dense_stack`] when writing
/// final_norm output to scratch so the next matvec can skip upload.
pub(crate) fn set_resident_activation(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) {
    RESIDENT_ACT.set(Some(ResidentActivation {
        buf_ptr: buf as *const _,
        len,
    }));
}

/// Clears the resident activation buffer TLS. Used to ensure clean state
/// after a decode that doesn't consume the resident buffer.
pub fn clear_resident_activation() {
    RESIDENT_ACT.set(None);
}

/// Checks if a resident activation buffer matches `x`, and if so, returns
/// the buffer and clears the TLS. Used by [`launch_matvec_fused`].
fn take_resident_activation_if_matches(
    x: &[f32],
) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
    RESIDENT_ACT.take().and_then(|res| {
        if res.len == x.len() {
            // Safety: pointer came from a live buffer in the same thread's
            // dense stack call (still in scope). We use Retained::retain
            // to get a new strong reference.
            unsafe { Retained::retain(res.buf_ptr as *mut _) }
        } else {
            None
        }
    })
}

/// Returns the default Metal device's name, or `None` if this machine
/// has no Metal-capable GPU (real check, not a compile-time guess).
pub fn probe() -> Option<String> {
    let device = MTLCreateSystemDefaultDevice()?;
    Some(device.name().to_string())
}

/// ggml-metal `kernel_mul_mv_q8_0_f32` port: `N_R0=2` rows per
/// threadgroup, `NSG=4` simdgroups (128 threads) cooperating on the
/// same two rows, each thread owning `NQ=8` contiguous int8 quants of a
/// block per pass. Same dequant identity as
/// `ferrox_quant::dot_q8_0_f32_scalar` (34-byte block: 2-byte f16 scale
/// plus 32 int8 values). Replaces the legacy one-TG-per-row scalar kernel,
/// which left most of the memory system idle on Q8_0-heavy models
/// (TinyLlama Q8_0 decode was ~1.5x behind llama.cpp).
///
/// Cross-simdgroup reduction goes through 8 floats of threadgroup
/// memory (2 rows x 4 simdgroups). Host dispatches `ceil(n_rows/2)`
/// threadgroups of 128 threads with 32 bytes of TG memory.
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// module docs).
pub const Q8_0_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q8_0_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* partial [[threadgroup(0)]]
) {
    constexpr short NSG = 4;
    constexpr short nr0 = 2;
    constexpr short NQ = 8;

    const int nb = int(n_blocks_per_row);
    const int first_row = int(tgpig) * nr0;

    device const uchar* row_ptr[nr0];
    for (short row = 0; row < nr0; ++row) {
        row_ptr[row] = weights + (size_t)(first_row + row) * row_bytes;
    }

    // 4 threads per block (NQ=8 quants each), 8 blocks per simdgroup
    // pass, stride NSG*NQ = 32 blocks per threadgroup pass.
    const short ix = short(tiisg) / (32 / NQ); // 0..7: block within pass
    const short il = short(tiisg) % (32 / NQ); // 0..3: quant slice

    const int ib0 = int(sgitg) * NQ + ix;

    float sumf[nr0] = {0.0f, 0.0f};
    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uchar* block = row_ptr[row] + (size_t)ib * 34u;
            device const char* qs = (device const char*)(block + 2) + il * NQ;

            float sumq = 0.0f;
            #pragma clang loop unroll(full)
            for (short i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }

            sumf[row] += sumq * float(*(device const half*)(block));
        }

        yb += NSG * NQ * 32;
    }

    for (short row = 0; row < nr0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0) {
            partial[row * NSG + sgitg] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (short row = 0; row < nr0; ++row) {
            if (first_row + row < int(n_rows)) {
                float total = 0.0f;
                for (short sg = 0; sg < NSG; ++sg) {
                    total += partial[row * NSG + sg];
                }
                out[first_row + row] = total;
            }
        }
    }
}
"#;

/// Runs the Q8_0 matvec kernel: `weights` is `rows` rows of
/// `row_bytes`-byte Q8_0-quantized data (GGML layout), `x` is the
/// dense `[cols]` input vector, returns the `[rows]` dot-product
/// output. `row_bytes` must equal `(cols / 32) * 34`.
pub fn launch_q8_0_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        Q8_0_MATVEC_KERNEL_SRC,
        "q8_0_matvec",
        34,
        32,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// CUDA-kernel-equivalent MSL source for a fused Q4_0 dequant+dot
/// kernel: same one-threadgroup-per-row / threadgroup-reduction
/// structure as `Q8_0_MATVEC_KERNEL_SRC`, but unpacking Q4_0's 18-byte
/// blocks (2-byte `half` scale + 16 bytes of packed 4-bit nibbles, low
/// nibble = element `i`, high nibble = element `i+16`, both biased by
/// -8) to mirror `ferrox_quant::dot_q4_0_f32_scalar`'s exact math and
/// `ferrox_cuda::gpu::Q4_0_MATVEC_KERNEL_SRC`'s structure (using MSL's
/// native `half` instead of the CUDA kernel's manual f16 bit decode).
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// `launch_q4_0_matvec_matches_cpu_reference`).
pub const Q4_0_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4_0_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* partial [[threadgroup(0)]]
) {
    device const uchar* row_ptr = weights + (size_t)row * row_bytes;
    float acc = 0.0f;
    for (uint b = tid; b < n_blocks_per_row; b += tg_size) {
        device const uchar* block = row_ptr + b * 18;
        half scale_h = *(device const half*)(block);
        float scale = float(scale_h);
        uint base = b * 32;
        float block_acc = 0.0f;
        for (uint i = 0; i < 16; i++) {
            uchar byte = block[2 + i];
            int lo = (int)(byte & 0x0F) - 8;
            int hi = (int)((byte >> 4) & 0x0F) - 8;
            block_acc += float(lo) * x[base + i];
            block_acc += float(hi) * x[base + i + 16];
        }
        acc += block_acc * scale;
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = simd_sum(acc);
    if ((tid & 31u) == 0u) {
        partial[tid / 32u] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        uint nsg = (tg_size + 31u) / 32u;
        for (uint i = 0; i < nsg; i++) {
            total += partial[i];
        }
        out[row] = total;
    }
}
"#;

/// Launches the Q4_0 matvec kernel. Verified on a real Apple M2 Pro GPU
/// -- see `Q4_0_MATVEC_KERNEL_SRC`'s doc comment.
pub fn launch_q4_0_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        Q4_0_MATVEC_KERNEL_SRC,
        "q4_0_matvec",
        18,
        32,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// Q4_0 multi-activation matmul (`FERROX_METAL_MUL_MM` path).
///
/// Correctness-first: same dequant as [`Q4_0_MATVEC_KERNEL_SRC`]
/// (`scale * (nibble - 8)`), one threadgroup per weight row, threads
/// stride over Q4_0 blocks and accumulate into a small batch tile
/// (`NB=8`) before a simd_sum reduce. Not ggml-metal's simdgroup
/// `mul_mm` tile (that scaffold returned zeros on M2 — kept out until
/// indexing matches). Host flattens `x_batch` as `[batch, cols]`,
/// returns `[batch, rows]`.
pub const Q4_0_MUL_MM_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4_0_mul_mm(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* partial [[threadgroup(0)]]
) {
    if (row >= n_rows) {
        return;
    }
    constexpr short NB = 8;
    const int cols = int(n_blocks_per_row) * 32;
    device const uchar* row_ptr = weights + (size_t)row * row_bytes;

    for (int bt = 0; bt < int(batch_size); bt += NB) {
        float acc[8];
        for (short b = 0; b < NB; ++b) {
            acc[b] = 0.0f;
        }
        for (uint blk = tid; blk < n_blocks_per_row; blk += tg_size) {
            device const uchar* block = row_ptr + blk * 18u;
            const float scale = float(*(device const half*)block);
            const uint base = blk * 32u;
            for (short b = 0; b < NB; ++b) {
                const int batch_idx = bt + b;
                if (batch_idx >= int(batch_size)) {
                    break;
                }
                device const float* xb = x + (size_t)batch_idx * cols + base;
                float block_acc = 0.0f;
                for (uint i = 0; i < 16u; i++) {
                    const uchar byte = block[2 + i];
                    const int lo = (int)(byte & 0x0Fu) - 8;
                    const int hi = (int)((byte >> 4) & 0x0Fu) - 8;
                    block_acc += float(lo) * xb[i];
                    block_acc += float(hi) * xb[i + 16];
                }
                acc[b] += block_acc * scale;
            }
        }
        for (short b = 0; b < NB; ++b) {
            const int batch_idx = bt + b;
            if (batch_idx >= int(batch_size)) {
                break;
            }
            partial[tid] = acc[b];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float s = simd_sum(acc[b]);
            if ((tid & 31u) == 0u) {
                partial[tid / 32u] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float total = 0.0f;
                const uint nsg = (tg_size + 31u) / 32u;
                for (uint i = 0u; i < nsg; i++) {
                    total += partial[i];
                }
                out[(size_t)batch_idx * n_rows + row] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#;

/// Launches Q4_0 multi-activation matmul (see [`Q4_0_MUL_MM_KERNEL_SRC`]).
///
/// Dequant matches [`Q4_0_MATVEC_KERNEL_SRC`]. `x_batch` is `[batch, cols]`;
/// returns `[batch, rows]`.
pub fn launch_q4_0_mul_mm(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    let n_blocks_per_row = row_bytes / 18;
    let cols = n_blocks_per_row * 32;
    assert_eq!(weights.len(), rows * row_bytes);
    assert_eq!(x_batch.len(), batch_size * cols);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let mut x_owned = x_batch.to_vec();
    let x_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(x_owned.as_mut_ptr() as *mut _).unwrap(),
            x_owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;

    let weights_buf = resident_weight_buffer(device, weights)?;
    let out_elems = batch_size * rows;
    let out_buf = device
        .newBufferWithLength_options(out_elems * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;

    let pipeline = ensure_pipeline(device, Q4_0_MUL_MM_KERNEL_SRC, "q4_0_mul_mm")?;

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let enc = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    let tg = 64u32;
    unsafe {
        enc.setComputePipelineState(&pipeline.0);
        enc.setBuffer_offset_atIndex(Some(&weights_buf.buffer), weights_buf.weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
        let mut row_bytes_u32 = row_bytes as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut row_bytes_u32 as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut n_blocks_u32 = n_blocks_per_row as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut n_blocks_u32 as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        let mut n_rows_u32 = rows as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut n_rows_u32 as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut batch_u32 = batch_size as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut batch_u32 as *mut u32 as *mut _).unwrap(),
            4,
            6,
        );
        enc.setThreadgroupMemoryLength_atIndex((tg as usize) * 4, 0);
    }

    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let out_ptr = out_buf.contents();
    let out_slice =
        unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, out_elems) };
    Ok(out_slice.to_vec())
}

/// ggml-metal `kernel_mul_mv_q4_K_f32` port: `N_R0=2` rows per simdgroup,
/// `NSG=2` simdgroups per TG (4 rows / 64 threads). Register-local `yl`/`yh`
/// activation packs (no shared-`x` tile) with masked nibble dots — same
/// dequant identity as `ferrox_quant::dot_q4_k_f32_scalar`. Host dispatches
/// `ceil(n_rows/4)` threadgroups of 64 threads.
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// `launch_q4_k_matvec_matches_cpu_reference`).
pub const Q4_K_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;  // 0...3
    const short it = tiisg % 8;  // 0...7
    const short iq = it / 4;     // 0 or 1
    const short ir = it % 4;     // 0...3

    const int first_row = int(tgpig * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);

    float yl[16];
    float yh[16];
    float sumf[2] = {0.0f, 0.0f};

    device const float* y4 = x + ix * 256 + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = float4(0.0f);

        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];
            sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];
            sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128];
            sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160];
            sumy[3] += yh[i + 8];
        }

        device const uchar* block0 =
            weights + (size_t)first_row * row_bytes + (size_t)ib * 144u;
        device const uint16_t* sc =
            (device const uint16_t*)(block0 + 4) + iq;
        device const uint16_t* q1 =
            (device const uint16_t*)(block0 + 16) + 16 * iq + 4 * ir;
        device const half* dh = (device const half*)(block0);

        uint16_t sc16[4];
        thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t* q2 = q1 + 32;

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);

            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * float(q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * float(q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0])
                    * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0])
                        + (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1])
                            * (1.0f / 16.0f)
                        + (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4])
                        + (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5])
                            * (1.0f / 16.0f))
                - float(dh[1])
                    * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3])
                        + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

            q1 += row_bytes / 2;
            sc += row_bytes / 2;
            dh += row_bytes / 2;
        }

        y4 += 4 * 256;
    }

    for (int row = 0; row < nr0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < int(n_rows)) {
            out[first_row + row] = sum_all;
        }
    }
}
"#;

/// Launches the Q4_K matvec kernel. Verified on a real Apple M2 Pro GPU
/// -- see `Q4_K_MATVEC_KERNEL_SRC`'s doc comment.
pub fn launch_q4_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        Q4_K_MATVEC_KERNEL_SRC,
        "q4_k_matvec",
        144,
        256,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// Multi-activation Q4_K matmul (first-cut GEMM): weight matrix ×
/// `[cols, batch]` → `[rows, batch]` with layout `[batch, cols]` /
/// `[batch, rows]` on the host.
///
/// Same dequant/dot identity as [`Q4_K_MATVEC_KERNEL_SRC`], but each
/// threadgroup walks a batch tile (`NB=4`) inside the K-loop so one
/// weight-block load serves multiple activations — better than
/// [`launch_matvec_batch`]'s N separate matvec encodings (still N×
/// weight traffic). Not a full ggml-metal `mul_mm` (no simdgroup_matrix
/// / 64×32 tiles); correct first cut for prefill.
///
/// Host dispatches `ceil(n_rows/4)` threadgroups of 64 threads (1D grid;
/// batch tiling is inside the kernel).
pub const Q4_K_MATMUL_BATCH_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4_k_matmul_batch(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;
    constexpr short NW = 32;
    constexpr short NB = 4;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort tiisg = tid % NW;
    const ushort sgitg = tid / NW;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int first_row = int(tgpig * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);
    const int cols = nb * 256;

    float yl[16];
    float yh[16];

    for (int bt = 0; bt < int(batch_size); bt += NB) {
        float sumf[2][4];
        for (short r = 0; r < nr0; ++r) {
            for (short b = 0; b < NB; ++b) {
                sumf[r][b] = 0.0f;
            }
        }

        for (int ib = ix; ib < nb; ib += 4) {
            device const uchar* block0 =
                weights + (size_t)first_row * row_bytes + (size_t)ib * 144u;
            device const uint16_t* sc0 =
                (device const uint16_t*)(block0 + 4) + iq;
            device const uint16_t* q10 =
                (device const uint16_t*)(block0 + 16) + 16 * iq + 4 * ir;
            device const half* dh0 = (device const half*)(block0);

            for (short row = 0; row < nr0; row++) {
                device const uint16_t* sc =
                    (device const uint16_t*)((device const uchar*)sc0 + row * row_bytes);
                device const uint16_t* q1 =
                    (device const uint16_t*)((device const uchar*)q10 + row * row_bytes);
                device const half* dh =
                    (device const half*)((device const uchar*)dh0 + row * row_bytes);

                uint16_t sc16[4];
                thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                uint16_t q1r[4];
                uint16_t q2r[4];
                device const uint16_t* q2 = q1 + 32;
                for (short i = 0; i < 4; ++i) {
                    q1r[i] = q1[i];
                    q2r[i] = q2[i];
                }
                const float dscale = float(dh[0]);
                const float dmin = float(dh[1]);

                for (short b = 0; b < NB; ++b) {
                    const int batch_idx = bt + b;
                    if (batch_idx >= int(batch_size)) {
                        break;
                    }
                    device const float* y4 =
                        x + (size_t)batch_idx * cols + ib * 256 + 64 * iq + 8 * ir;

                    float4 sumy = float4(0.0f);
                    for (short i = 0; i < 8; ++i) {
                        yl[i + 0] = y4[i + 0];
                        sumy[0] += yl[i + 0];
                        yl[i + 8] = y4[i + 32];
                        sumy[1] += yl[i + 8];
                        yh[i + 0] = y4[i + 128];
                        sumy[2] += yh[i + 0];
                        yh[i + 8] = y4[i + 160];
                        sumy[3] += yh[i + 8];
                    }

                    float4 acc1 = float4(0.0f);
                    float4 acc2 = float4(0.0f);
                    for (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * float(q1r[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * float(q1r[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * float(q1r[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * float(q1r[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * float(q2r[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * float(q2r[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * float(q2r[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * float(q2r[i] & 0xF000);
                    }

                    sumf[row][b] += dscale
                            * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0])
                                + (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1])
                                    * (1.0f / 16.0f)
                                + (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4])
                                + (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5])
                                    * (1.0f / 16.0f))
                        - dmin
                            * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3])
                                + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
                }
            }
        }

        for (int row = 0; row < nr0; ++row) {
            for (short b = 0; b < NB; ++b) {
                const int batch_idx = bt + b;
                float sum_all = simd_sum(sumf[row][b]);
                if (tiisg == 0 && first_row + row < int(n_rows)
                    && batch_idx < int(batch_size)) {
                    out[(size_t)batch_idx * n_rows + first_row + row] = sum_all;
                }
            }
        }
    }
}
"#;

/// Launches the Q4_K multi-activation matmul. `x_batch` is `[batch, cols]`;
/// returns `[batch, rows]`.
pub fn launch_q4_k_matmul_batch(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matmul_batch(
        Q4_K_MATMUL_BATCH_KERNEL_SRC,
        "q4_k_matmul_batch",
        144,
        256,
        4, // rows_per_tg (NSG=2 × nr0=2)
        weights,
        x_batch,
        rows,
        row_bytes,
        batch_size,
    )
}

/// Q4_K multi-activation matmul (`FERROX_METAL_MUL_MM` prefill path).
///
/// Correctness-first, same shape as [`Q4_0_MUL_MM_KERNEL_SRC`]: one
/// threadgroup per weight row, threads stride over the row's Q4_K blocks
/// and accumulate an `NB`-wide batch tile before a `simd_sum` reduce, so
/// each row's quantized bytes are read once and reused across the batch.
/// The per-block dequant is the exact `ferrox_quant::dot_q4_k_f32_scalar`
/// identity (6-bit packed scales/mins via `q4_k_scale_min`), so results
/// match [`Q4_K_MATVEC_KERNEL_SRC`]. This is deliberately *not*
/// ggml-metal's `simdgroup_matrix` tile -- that scaffold produced wrong
/// dequant on M2, so it is kept out until its indexing is proven.
///
/// Host flattens `x_batch` as `[batch, cols]`; returns `[batch, rows]`.
pub const Q4_K_MUL_MM_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline uchar2 q4_k_scale_min(uint j, device const uchar* scales) {
    if (j < 4u) {
        return uchar2(scales[j] & 63u, scales[j + 4u] & 63u);
    }
    return uchar2(
        (scales[j + 4u] & 0x0Fu) | ((scales[j - 4u] >> 6u) << 4u),
        (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u));
}

kernel void q4_k_mul_mm(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* partial [[threadgroup(0)]]
) {
    if (row >= n_rows) {
        return;
    }
    constexpr short NB = 8;
    const int cols = int(n_blocks_per_row) * 256;
    device const uchar* row_ptr = weights + (size_t)row * row_bytes;

    for (int bt = 0; bt < int(batch_size); bt += NB) {
        float acc[8];
        for (short b = 0; b < NB; ++b) {
            acc[b] = 0.0f;
        }
        for (uint blk = tid; blk < n_blocks_per_row; blk += tg_size) {
            device const uchar* block = row_ptr + (size_t)blk * 144u;
            const float d = float(*(device const half*)block);
            const float dmin = float(*(device const half*)(block + 2));
            device const uchar* scales = block + 4;
            device const uchar* qs = block + 16;
            const uint base = blk * 256u;
            for (short b = 0; b < NB; ++b) {
                const int batch_idx = bt + b;
                if (batch_idx >= int(batch_size)) {
                    break;
                }
                device const float* xb = x + (size_t)batch_idx * cols + base;
                float block_acc = 0.0f;
                uint q_off = 0u;
                uint xoff = 0u;
                for (short is = 0; is < 8; is += 2) {
                    const uchar2 sm1 = q4_k_scale_min(uint(is), scales);
                    const uchar2 sm2 = q4_k_scale_min(uint(is) + 1u, scales);
                    const float d1 = d * float(sm1.x);
                    const float min1 = dmin * float(sm1.y);
                    const float d2 = d * float(sm2.x);
                    const float min2 = dmin * float(sm2.y);
                    for (short l = 0; l < 32; ++l) {
                        block_acc += (d1 * float(qs[q_off + l] & 0x0Fu) - min1) * xb[xoff + l];
                    }
                    for (short l = 0; l < 32; ++l) {
                        block_acc += (d2 * float(qs[q_off + l] >> 4) - min2) * xb[xoff + 32 + l];
                    }
                    q_off += 32u;
                    xoff += 64u;
                }
                acc[b] += block_acc;
            }
        }
        for (short b = 0; b < NB; ++b) {
            const int batch_idx = bt + b;
            if (batch_idx >= int(batch_size)) {
                break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float s = simd_sum(acc[b]);
            if ((tid & 31u) == 0u) {
                partial[tid / 32u] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float total = 0.0f;
                const uint nsg = (tg_size + 31u) / 32u;
                for (uint i = 0u; i < nsg; i++) {
                    total += partial[i];
                }
                out[(size_t)batch_idx * n_rows + row] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#;

/// Launches Q4_K multi-activation matmul (see [`Q4_K_MUL_MM_KERNEL_SRC`]).
///
/// Dequant matches [`Q4_K_MATVEC_KERNEL_SRC`] /
/// `ferrox_quant::dot_q4_k_f32`. `x_batch` is `[batch, cols]`; returns
/// `[batch, rows]`. One threadgroup per weight row, 64 threads striding
/// over the row's Q4_K blocks.
pub fn launch_q4_k_mul_mm(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    let n_blocks_per_row = row_bytes / 144;
    let cols = n_blocks_per_row * 256;
    assert_eq!(weights.len(), rows * row_bytes);
    assert_eq!(x_batch.len(), batch_size * cols);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let mut x_owned = x_batch.to_vec();
    let x_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(x_owned.as_mut_ptr() as *mut _).unwrap(),
            x_owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;

    let weights_buf = resident_weight_buffer(device, weights)?;
    let out_elems = batch_size * rows;
    let out_buf = device
        .newBufferWithLength_options(out_elems * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;

    let pipeline = ensure_pipeline(device, Q4_K_MUL_MM_KERNEL_SRC, "q4_k_mul_mm")?;

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let enc = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    let tg = 64u32;
    unsafe {
        enc.setComputePipelineState(&pipeline.0);
        enc.setBuffer_offset_atIndex(Some(&weights_buf.buffer), weights_buf.weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
        let mut row_bytes_u32 = row_bytes as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut row_bytes_u32 as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut n_blocks_u32 = n_blocks_per_row as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut n_blocks_u32 as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        let mut n_rows_u32 = rows as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut n_rows_u32 as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut batch_u32 = batch_size as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut batch_u32 as *mut u32 as *mut _).unwrap(),
            4,
            6,
        );
        enc.setThreadgroupMemoryLength_atIndex((tg as usize) * 4, 0);
    }

    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let out_ptr = out_buf.contents();
    let out_slice =
        unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, out_elems) };
    Ok(out_slice.to_vec())
}

/// Shared host plumbing for multi-activation matmul kernels (one CB,
/// `ceil(rows / rows_per_tg)` threadgroups × 64 threads, `batch_size`
/// bound as buffer 6).
#[allow(clippy::too_many_arguments)]
fn launch_matmul_batch(
    kernel_src: &'static str,
    fn_name: &'static str,
    block_bytes: usize,
    block_elems: usize,
    rows_per_tg: usize,
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    let n_blocks_per_row = row_bytes / block_bytes;
    let cols = n_blocks_per_row * block_elems;
    assert_eq!(weights.len(), rows * row_bytes);
    assert_eq!(x_batch.len(), batch_size * cols);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let mut x_owned = x_batch.to_vec();
    let x_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(x_owned.as_mut_ptr() as *mut _).unwrap(),
            x_owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;

    let weight_buf = resident_weight_buffer(device, weights)?;
    let out_elems = batch_size * rows;
    let out_buf = device
        .newBufferWithLength_options(out_elems * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;

    let cached_pipeline = ensure_pipeline(device, kernel_src, fn_name)?;
    let pipeline = &cached_pipeline.0;

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&weight_buf.buffer), weight_buf.weight_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(&x_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
        let mut row_bytes_u32 = row_bytes as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut row_bytes_u32 as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut n_blocks_u32 = n_blocks_per_row as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_blocks_u32 as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        let mut n_rows_u32 = rows as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_rows_u32 as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut batch_u32 = batch_size as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut batch_u32 as *mut u32 as *mut _).unwrap(),
            4,
            6,
        );
    }
    let n_tg = rows.div_ceil(rows_per_tg.max(1));
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let out_ptr = out_buf.contents();
    let out_slice =
        unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, out_elems) };
    Ok(out_slice.to_vec())
}

/// ggml-metal `kernel_mul_mv_q5_K_f32` port: `N_R0=1` row per simdgroup,
/// `NSG=2` simdgroups per TG (2 rows / 64 threads). Register-local
/// `yl`/`yh` activation packs with Q5_K's 5th-bit `qh` plane — same
/// dequant identity as `ferrox_quant::dot_q5_k_f32_scalar`. Host
/// dispatches `ceil(n_rows/2)` threadgroups of 64 threads.
///
/// `N_R0` stays at 1 (not 2 like Q4_K/Q6_K): ggml-metal reduced it after
/// a real register-spill regression (`llama.cpp` #20399).
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// `launch_q5_k_matvec_matches_cpu_reference`).
pub const Q5_K_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q5_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid_tg [[thread_position_in_threadgroup]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 1;
    constexpr short NW = 32;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort tiisg = tid_tg % NW;
    const ushort sgitg = tid_tg / NW;

    const short tid = tiisg / 4;
    const short ix = tiisg % 4;
    const short iq = tid / 4;
    const short ir = tid % 4;

    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uchar hm1 = uchar(1u << (2 * iq));
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    const int first_row = int(tgpig * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);

    if (first_row >= int(n_rows)) {
        return;
    }

    float sumf = 0.0f;
    float yl[16];
    float yh[16];

    uint16_t sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    device const float* y1 = x + ix * 256 + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uchar* block0 =
            weights + (size_t)first_row * row_bytes + (size_t)i * 176u;
        device const uchar* q1 = block0 + 48 + q_offset;
        device const uchar* qh = block0 + 16 + l0;
        device const half* dh = (device const half*)(block0);
        device const uint16_t* a =
            (device const uint16_t*)(block0 + 4) + iq;

        device const float* y2 = y1 + 128;
        float4 sumy = float4(0.0f);
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];
            sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32];
            sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l + 0];
            sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32];
            sumy[3] += yh[l + 8];
        }

        device const uchar* q2 = q1 + 64;

        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        float4 acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f);
        for (short l = 0; l < 8; ++l) {
            uchar h = qh[l];
            acc1[0] += yl[l + 0] * float(q1[l] & 0x0F);
            acc1[1] += yl[l + 8] * float(q1[l] & 0xF0);
            acc1[2] += yh[l + 0] * float(q2[l] & 0x0F);
            acc1[3] += yh[l + 8] * float(q2[l] & 0xF0);
            acc2[0] += (h & hm1) ? yl[l + 0] : 0.0f;
            acc2[1] += (h & hm2) ? yl[l + 8] : 0.0f;
            acc2[2] += (h & hm3) ? yh[l + 0] : 0.0f;
            acc2[3] += (h & hm4) ? yh[l + 8] : 0.0f;
        }

        sumf += float(dh[0])
                * (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0])
                    + float(sc8[1]) * (acc1[1] / 16.0f + 16.0f * acc2[1])
                    + float(sc8[4]) * (acc1[2] + 16.0f * acc2[2])
                    + float(sc8[5]) * (acc1[3] / 16.0f + 16.0f * acc2[3]))
            - float(dh[1])
                * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3])
                    + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

        y1 += 4 * 256;
    }

    float sum_all = simd_sum(sumf);
    if (tiisg == 0 && first_row < int(n_rows)) {
        out[first_row] = sum_all;
    }
}
"#;

/// Launches the Q5_K matvec kernel. Verified on a real Apple M2 Pro GPU
/// -- see `Q5_K_MATVEC_KERNEL_SRC`'s doc comment.
pub fn launch_q5_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        Q5_K_MATVEC_KERNEL_SRC,
        "q5_k_matvec",
        176,
        256,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// ggml-metal `kernel_mul_mv_q6_K_f32` port: `N_R0=2` rows per simdgroup,
/// `NSG=2` simdgroups per TG (4 rows / 64 threads). Register-local `yl`
/// packs; signed int8 sub-block scales match
/// `ferrox_quant::dot_q6_k_f32_scalar`. Host dispatches `ceil(n_rows/4)`
/// threadgroups of 64 threads.
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// `launch_q6_k_matvec_matches_cpu_reference`, which specifically
/// includes a negative-scale byte in its test data).
pub const Q6_K_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q6_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int first_row = int(tgpig * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);

    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8; // 0 or 1
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;

    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    float sumf[2] = {0.0f, 0.0f};
    float yl[16];

    for (int i = ix; i < nb; i += 2) {
        device const uchar* block0 =
            weights + (size_t)first_row * row_bytes + (size_t)i * 210u;
        device const uchar* q1 = block0 + q_offset_l;
        device const uchar* q2 = q1 + 32;
        device const uchar* qh = block0 + 128 + q_offset_h;
        device const char* sc = (device const char*)(block0 + 192 + is);
        device const half* dh = (device const half*)(block0 + 208);

        device const float* y = x + i * 256 + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l + 0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = float4(0.0f);

            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0]
                    * float(int((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1]
                    * float(int((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2]
                    * float(int((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3]
                    * float(int((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += float(dh[0])
                * (sums[0] * float(sc[0]) + sums[1] * float(sc[2])
                    + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));

            q1 += row_bytes;
            q2 += row_bytes;
            qh += row_bytes;
            sc += row_bytes;
            dh += row_bytes / 2;
        }
    }

    for (int row = 0; row < nr0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < int(n_rows)) {
            out[first_row + row] = sum_all;
        }
    }
}
"#;

/// Launches the Q6_K matvec kernel. Verified on a real Apple M2 Pro GPU
/// -- see `Q6_K_MATVEC_KERNEL_SRC`'s doc comment.
pub fn launch_q6_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        Q6_K_MATVEC_KERNEL_SRC,
        "q6_k_matvec",
        210,
        256,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// Multi-activation Q6_K matmul — same idea as
/// [`Q4_K_MATMUL_BATCH_KERNEL_SRC`]: matvec dequant identity with an
/// inner `NB=4` batch tile so weight blocks are reused across
/// activations. Host grid is still `ceil(n_rows/4)` × 64 threads.
pub const Q6_K_MATMUL_BATCH_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q6_k_matmul_batch(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid_tg [[thread_position_in_threadgroup]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;
    constexpr short NW = 32;
    constexpr short NB = 4;
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const ushort tiisg = tid_tg % NW;
    const ushort sgitg = tid_tg / NW;

    const int first_row = int(tgpig * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);
    const int cols = nb * 256;

    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;

    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    float yl[16];

    for (int bt = 0; bt < int(batch_size); bt += NB) {
        float sumf[2][4];
        for (short r = 0; r < nr0; ++r) {
            for (short b = 0; b < NB; ++b) {
                sumf[r][b] = 0.0f;
            }
        }

        for (int i = ix; i < nb; i += 2) {
            device const uchar* block0 =
                weights + (size_t)first_row * row_bytes + (size_t)i * 210u;

            for (short row = 0; row < nr0; ++row) {
                device const uchar* brow = block0 + (size_t)row * row_bytes;
                device const uchar* q1 = brow + q_offset_l;
                device const uchar* q2 = q1 + 32;
                device const uchar* qh = brow + 128 + q_offset_h;
                device const char* sc = (device const char*)(brow + 192 + is);
                device const half* dh = (device const half*)(brow + 208);

                uchar q1r[4];
                uchar q2r[4];
                uchar qhr[4];
                char scr[4];
                for (short l = 0; l < 4; ++l) {
                    q1r[l] = q1[l];
                    q2r[l] = q2[l];
                    qhr[l] = qh[l];
                }
                scr[0] = sc[0];
                scr[1] = sc[2];
                scr[2] = sc[4];
                scr[3] = sc[6];
                const float dscale = float(dh[0]);

                for (short b = 0; b < NB; ++b) {
                    const int batch_idx = bt + b;
                    if (batch_idx >= int(batch_size)) {
                        break;
                    }
                    device const float* y =
                        x + (size_t)batch_idx * cols + i * 256 + y_offset;

                    for (short l = 0; l < 4; ++l) {
                        yl[4 * l + 0] = y[l + 0];
                        yl[4 * l + 1] = y[l + 32];
                        yl[4 * l + 2] = y[l + 64];
                        yl[4 * l + 3] = y[l + 96];
                    }

                    float4 sums = float4(0.0f);
                    for (short l = 0; l < 4; ++l) {
                        sums[0] += yl[4 * l + 0]
                            * float(int((q1r[l] & 0xF) | ((qhr[l] & kmask1) << 4)) - 32);
                        sums[1] += yl[4 * l + 1]
                            * float(int((q2r[l] & 0xF) | ((qhr[l] & kmask2) << 2)) - 32);
                        sums[2] += yl[4 * l + 2]
                            * float(int((q1r[l] >> 4) | ((qhr[l] & kmask3) << 0)) - 32);
                        sums[3] += yl[4 * l + 3]
                            * float(int((q2r[l] >> 4) | ((qhr[l] & kmask4) >> 2)) - 32);
                    }

                    sumf[row][b] += dscale
                        * (sums[0] * float(scr[0]) + sums[1] * float(scr[1])
                            + sums[2] * float(scr[2]) + sums[3] * float(scr[3]));
                }
            }
        }

        for (int row = 0; row < nr0; ++row) {
            for (short b = 0; b < NB; ++b) {
                const int batch_idx = bt + b;
                float sum_all = simd_sum(sumf[row][b]);
                if (tiisg == 0 && first_row + row < int(n_rows)
                    && batch_idx < int(batch_size)) {
                    out[(size_t)batch_idx * n_rows + first_row + row] = sum_all;
                }
            }
        }
    }
}
"#;

/// Launches the Q6_K multi-activation matmul. `x_batch` is `[batch, cols]`;
/// returns `[batch, rows]`.
pub fn launch_q6_k_matmul_batch(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matmul_batch(
        Q6_K_MATMUL_BATCH_KERNEL_SRC,
        "q6_k_matmul_batch",
        210,
        256,
        4,
        weights,
        x_batch,
        rows,
        row_bytes,
        batch_size,
    )
}

/// ggml-metal `kernel_mul_mv_iq4_xs_f32` port: `N_R0=2` rows per
/// simdgroup, `NSG=2` simdgroups per TG (4 rows / 64 threads). IQ4_XS
/// blocks are 136 bytes / 256 elements: f16 super-scale `d`, 16 bits of
/// high scale bits, 4 bytes of low scale nibbles, then 128 bytes of
/// 4-bit indices into the shared IQ4_NL non-linear codebook (loaded
/// into 32 floats of threadgroup memory, one copy per 16 lanes). Each
/// lane owns 8 consecutive bytes of `qs` (16 elements: low nibbles →
/// elems j, high nibbles → elems j+16 of the 32-elem sub-block), with
/// odd/even lanes-of-16 walking odd/even blocks. Same dequant identity
/// as `ferrox_quant::dot_iq4_xs_f32`.
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference (see
/// `launch_iq4_xs_matvec_matches_cpu_reference`).
pub const IQ4_XS_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant float kvalues_iq4nl_f[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

kernel void iq4_xs_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* shmem_f32 [[threadgroup(0)]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;

    const int nb = int(n_blocks_per_row);
    const int first_row = int(tgpig * NSG + sgitg) * nr0;

    const short ix = short(tiisg) / 16; // 0/1: block parity
    const short it = short(tiisg) % 16;
    const short ib = it / 2;            // 0..7: 32-elem sub-block
    const short il = it % 2;            // 0/1: 8-byte half of qs sub-block

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[nr0] = {0.0f, 0.0f};

    device const float* yb = x + ix * 256 + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uchar* q8 = (thread const uchar*)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {
        device const float4* y4 = (device const float4*)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < nr0; ++row) {
            device const uchar* blk = weights
                + (size_t)(first_row + row) * row_bytes + (size_t)ibl * 136u;
            device const uint32_t* q4 =
                (device const uint32_t*)(blk + 8u + 16u * ib + 8u * il);

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = float4(shmem_f32[q8[0]], shmem_f32[q8[1]],
                         shmem_f32[q8[2]], shmem_f32[q8[3]]);
            qf2 = float4(shmem_f32[q8[4]], shmem_f32[q8[5]],
                         shmem_f32[q8[6]], shmem_f32[q8[7]]);
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = float4(shmem_f32[q8[0]], shmem_f32[q8[1]],
                         shmem_f32[q8[2]], shmem_f32[q8[3]]);
            qf2 = float4(shmem_f32[q8[4]], shmem_f32[q8[5]],
                         shmem_f32[q8[6]], shmem_f32[q8[7]]);
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const ushort scales_h = *(device const ushort*)(blk + 2);
            const int ls = int(((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf)
                | (((scales_h >> (2 * ib)) & 3) << 4)) - 32;
            sumf[row] += float(*(device const half*)blk) * float(ls)
                * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * 256;
    }

    for (short row = 0; row < nr0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < int(n_rows)) {
            out[first_row + row] = sum_all;
        }
    }
}
"#;

/// Launches the IQ4_XS matvec kernel. Verified on a real Apple M2 Pro
/// GPU -- see `IQ4_XS_MATVEC_KERNEL_SRC`'s doc comment.
pub fn launch_iq4_xs_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        IQ4_XS_MATVEC_KERNEL_SRC,
        "iq4_xs_matvec",
        136,
        256,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// Kernel metadata for building a [`MatvecLaunch`] from a GGML quant
/// tag name used by `WeightMatrix` (Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K).
/// The fifth field is rows-per-threadgroup (`1` for legacy one-row
/// kernels; `2` for Q5_K NSG=2/N_R0=1; `4` for Q4_K/Q6_K).
pub fn matvec_launch_meta(kind: &str) -> Option<(&'static str, &'static str, usize, usize, usize)> {
    match kind {
        "Q8_0" => Some((Q8_0_MATVEC_KERNEL_SRC, "q8_0_matvec", 34, 32, 2)),
        "Q4_0" => Some((Q4_0_MATVEC_KERNEL_SRC, "q4_0_matvec", 18, 32, 1)),
        "Q4_K" => Some((Q4_K_MATVEC_KERNEL_SRC, "q4_k_matvec", 144, 256, 4)),
        "Q5_K" => Some((Q5_K_MATVEC_KERNEL_SRC, "q5_k_matvec", 176, 256, 2)),
        "Q6_K" => Some((Q6_K_MATVEC_KERNEL_SRC, "q6_k_matvec", 210, 256, 4)),
        "IQ4_XS" => Some((IQ4_XS_MATVEC_KERNEL_SRC, "iq4_xs_matvec", 136, 256, 4)),
        _ => None,
    }
}

/// One process-wide `MTLDevice` + `MTLCommandQueue`, created once and
/// reused for every kernel launch (mirrors
/// `ferrox_cuda::gpu::shared_device`'s `Mutex<Option<Arc<...>>>`
/// pattern exactly, including the reason for `Mutex` over `OnceLock`:
/// this project's pinned minimum rustc predates
/// `OnceLock::get_or_try_init`).
pub(crate) struct SharedMetal {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

// SAFETY: Apple's Metal documentation states that `MTLDevice` and
// `MTLCommandQueue` objects are safe to use from multiple threads
// simultaneously (unlike `MTLCommandBuffer`/`MTLComputeCommandEncoder`,
// which are explicitly documented as requiring single-threaded,
// single-use access -- and which this module never caches, only ever
// creates fresh per call, exactly because of that distinction).
// `objc2-metal`'s `Retained<ProtocolObject<dyn T>>` is unconditionally
// `!Send`/`!Sync` regardless of `T` (it wraps a `NonNull`, which is
// `!Send`/`!Sync` no matter what it points to, forcing every crate that
// wraps an Objective-C object to explicitly assert thread-safety rather
// than getting it for free) -- so this is a deliberate, narrow opt-in
// for exactly the two object kinds Apple documents as safe to share,
// not a blanket assertion about arbitrary Objective-C objects.
unsafe impl Send for SharedMetal {}
unsafe impl Sync for SharedMetal {}

static SHARED_METAL: Mutex<Option<Arc<SharedMetal>>> = Mutex::new(None);

pub(crate) fn shared_metal() -> Result<Arc<SharedMetal>, MetalError> {
    let mut guard = SHARED_METAL.lock().unwrap();
    if let Some(shared) = guard.as_ref() {
        return Ok(shared.clone());
    }
    let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
    let queue = device.newCommandQueue().ok_or(MetalError::CommandFailed)?;
    let shared = Arc::new(SharedMetal { device, queue });
    *guard = Some(shared.clone());
    Ok(shared)
}

/// One compiled `MTLComputePipelineState`, cached by kernel function
/// name so a given kernel is only ever compiled once per process
/// (mirrors `ferrox_cuda::gpu::ensure_module_loaded`'s
/// compile-once/reuse behavior for NVRTC modules).
pub(crate) struct CachedPipeline(pub(crate) Retained<ProtocolObject<dyn MTLComputePipelineState>>);

// SAFETY: same justification as `SharedMetal` above -- Apple documents
// `MTLComputePipelineState` (like `MTLDevice`/`MTLCommandQueue`) as
// safe to use from multiple threads simultaneously once created.
unsafe impl Send for CachedPipeline {}
unsafe impl Sync for CachedPipeline {}

static PIPELINE_CACHE: Mutex<Option<HashMap<&'static str, Arc<CachedPipeline>>>> = Mutex::new(None);

pub(crate) fn ensure_pipeline(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    kernel_src: &'static str,
    fn_name: &'static str,
) -> Result<Arc<CachedPipeline>, MetalError> {
    let mut guard = PIPELINE_CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(HashMap::new);
    if let Some(cached) = cache.get(fn_name) {
        return Ok(cached.clone());
    }

    let src = NSString::from_str(kernel_src);
    let library = device
        .newLibraryWithSource_options_error(&src, None)
        .map_err(|e| MetalError::CompileFailed(e.to_string()))?;
    let func_name = NSString::from_str(fn_name);
    let function = library
        .newFunctionWithName(&func_name)
        .ok_or(MetalError::FunctionNotFound(fn_name))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| MetalError::PipelineFailed(e.to_string()))?;

    let cached = Arc::new(CachedPipeline(pipeline));
    cache.insert(fn_name, cached.clone());
    Ok(cached)
}

/// Process-wide cache of quantized weight `MTLBuffer`s, keyed by the
/// host slice's base pointer and length. Stable for mmap-backed and
/// owned `WeightBytes` after load (weights are not mutated in place).
pub(crate) struct ResidentWeightBuffer {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Byte offset of the weight bytes within `buffer`. Non-zero only on
    /// the zero-copy (`BytesNoCopy`) path, where the buffer wraps a whole
    /// registered GGUF mmap and the tensor starts at this file offset.
    /// Copy-path buffers always have offset 0. Every kernel that binds
    /// this buffer at argument slot 0 must bind it at this offset.
    pub(crate) weight_offset: usize,
    nbytes: usize,
    /// Keeps the registered mmap MTLBuffer entry alive for NoCopy aliases.
    _keepalive: Option<Arc<ResidentMmapFile>>,
}

// SAFETY: same justification as `SharedMetal` -- `MTLBuffer` created
// once and only read by compute kernels is safe to share across threads
// that each build their own command buffer/encoder.
unsafe impl Send for ResidentWeightBuffer {}
unsafe impl Sync for ResidentWeightBuffer {}

type WeightCacheKey = (usize, usize);
type WeightCacheMap = HashMap<WeightCacheKey, Arc<ResidentWeightBuffer>>;

static WEIGHT_CACHE: Mutex<Option<WeightCacheMap>> = Mutex::new(None);

fn weight_cache_budget_bytes() -> usize {
    match std::env::var("FERROX_METAL_WEIGHT_CACHE_BYTES") {
        Ok(v) => v.parse().unwrap_or(usize::MAX),
        // Default: effectively unlimited on unified memory; callers can
        // cap with FERROX_METAL_WEIGHT_CACHE_BYTES for smaller machines.
        Err(_) => usize::MAX,
    }
}

pub(crate) fn resident_weight_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    weights: &[u8],
) -> Result<Arc<ResidentWeightBuffer>, MetalError> {
    let key = (weights.as_ptr() as usize, weights.len());
    {
        let guard = WEIGHT_CACHE.lock().unwrap();
        if let Some(cache) = guard.as_ref() {
            if let Some(cached) = cache.get(&key) {
                return Ok(cached.clone());
            }
        }
    }

    let mut guard = WEIGHT_CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(HashMap::new);
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.clone());
    }

    let budget = weight_cache_budget_bytes();
    let used: usize = cache.values().map(|b| b.nbytes).sum();
    if used.saturating_add(weights.len()) > budget {
        // Drop everything and retry with a clean slate for this matrix.
        // Better than silently re-uploading forever under a tight budget.
        cache.clear();
    }
    if weights.len() > budget {
        // Matrix alone exceeds budget: one-shot upload, do not cache.
        return Ok(Arc::new(build_resident_weight_buffer(device, weights)?));
    }

    let cached = Arc::new(build_resident_weight_buffer(device, weights)?);
    cache.insert(key, cached.clone());
    Ok(cached)
}

/// VM page size used for `BytesNoCopy` alignment. Apple Silicon uses
/// 16 KiB pages; Intel Macs use 4 KiB. Over-aligning is never wrong (a
/// 16 KiB boundary is also a 4 KiB boundary); under-aligning makes
/// `newBufferWithBytesNoCopy` return nil, which we handle by falling
/// back to a copy, so a mismatch degrades gracefully rather than crashing.
#[cfg(target_arch = "aarch64")]
const METAL_VM_PAGE: usize = 16384;
#[cfg(not(target_arch = "aarch64"))]
const METAL_VM_PAGE: usize = 4096;

/// One MTLBuffer wrapping an entire GGUF mmap (page-aligned file image).
/// Tensor slices reuse this buffer at `range.start` offsets — same
/// residency model as llama.cpp's mmap-backed Metal buffers.
struct ResidentMmapFile {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Keeps the underlying mapping alive for as long as any weight
    /// buffer aliases it via `BytesNoCopy`.
    _mmap: Arc<memmap2::Mmap>,
    base_ptr: usize,
    len: usize,
}

// SAFETY: read-only MTLBuffer + immutable mmap, shared across threads
// that each build their own command buffer (same as ResidentWeightBuffer).
unsafe impl Send for ResidentMmapFile {}
unsafe impl Sync for ResidentMmapFile {}

type MmapFileCache = HashMap<usize, Arc<ResidentMmapFile>>;
static MMAP_FILE_CACHE: Mutex<Option<MmapFileCache>> = Mutex::new(None);

/// Register a GGUF mmap so later [`resident_weight_buffer`] calls whose
/// slices sit inside it can alias the file with `BytesNoCopy` instead of
/// copying. Safe to call multiple times for the same `Arc` (idempotent).
/// Call from the loader when taking [`WeightBytes::Mapped`] views.
pub fn register_weight_mmap(mmap: Arc<memmap2::Mmap>) {
    let force_copy = std::env::var("FERROX_METAL_WEIGHT_COPY")
        .map(|v| v != "0")
        .unwrap_or(false);
    if force_copy || mmap.is_empty() {
        return;
    }
    let key = mmap.as_ptr() as usize;
    {
        let guard = MMAP_FILE_CACHE.lock().unwrap();
        if let Some(cache) = guard.as_ref() {
            if cache.contains_key(&key) {
                return;
            }
        }
    }
    let Ok(shared) = shared_metal() else {
        return;
    };
    let device = &shared.device;
    let base = mmap.as_ptr() as usize;
    // mmap returns a page-aligned pointer; length must be a page multiple
    // for BytesNoCopy — round the file length up within the mapping's
    // VM region (the OS maps whole pages for the file).
    let buf_len = mmap.len().div_ceil(METAL_VM_PAGE) * METAL_VM_PAGE;
    // SAFETY: `mmap.as_ptr()` is page-aligned; `buf_len` is a page
    // multiple covering only pages the kernel already mapped for this
    // file. `_mmap` keepalive in ResidentMmapFile outlives the MTLBuffer.
    // `None` deallocator => Metal does not free host memory.
    let nocopy = unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            NonNull::new(base as *mut _).unwrap(),
            buf_len,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    };
    let Some(buffer) = nocopy else {
        return;
    };
    let entry = Arc::new(ResidentMmapFile {
        buffer,
        _mmap: mmap,
        base_ptr: base,
        len: buf_len,
    });
    let mut guard = MMAP_FILE_CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(HashMap::new);
    cache.entry(key).or_insert(entry);
}

fn find_registered_mmap(weights: &[u8]) -> Option<(Arc<ResidentMmapFile>, usize)> {
    let start = weights.as_ptr() as usize;
    let end = start + weights.len();
    let guard = MMAP_FILE_CACHE.lock().unwrap();
    let cache = guard.as_ref()?;
    for file in cache.values() {
        if start >= file.base_ptr && end <= file.base_ptr + file.len {
            return Some((file.clone(), start - file.base_ptr));
        }
    }
    None
}

/// Build a resident weight buffer for `weights`.
///
/// Prefer zero-copy: if the slice lives inside a mmap registered via
/// [`register_weight_mmap`], alias that file's MTLBuffer at the tensor
/// byte offset (no `to_vec` double). Owned / unregistered slices fall
/// back to a Shared copy. Set `FERROX_METAL_WEIGHT_COPY=1` to force copy.
fn build_resident_weight_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    weights: &[u8],
) -> Result<ResidentWeightBuffer, MetalError> {
    let force_copy = std::env::var("FERROX_METAL_WEIGHT_COPY")
        .map(|v| v != "0")
        .unwrap_or(false);

    if !force_copy && !weights.is_empty() {
        if let Some((file, offset)) = find_registered_mmap(weights) {
            return Ok(ResidentWeightBuffer {
                buffer: file.buffer.clone(),
                weight_offset: offset,
                nbytes: 0, // aliased: no cache-budget cost
                _keepalive: Some(file),
            });
        }
    }

    // Fallback: copy the bytes into a fresh Shared buffer (offset 0).
    let mut weights_owned = weights.to_vec();
    let buffer = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(weights_owned.as_mut_ptr() as *mut _).unwrap(),
            weights_owned.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;
    Ok(ResidentWeightBuffer {
        buffer,
        weight_offset: 0,
        nbytes: weights.len(),
        _keepalive: None,
    })
}

pub(crate) struct ResidentF32Buffer {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    nbytes: usize,
}

// SAFETY: same justification as `SharedMetal` -- `MTLBuffer` created
// once and only read by compute kernels is safe to share across threads
// that each build their own command buffer/encoder.
unsafe impl Send for ResidentF32Buffer {}
unsafe impl Sync for ResidentF32Buffer {}

type F32CacheKey = (usize, usize);
type F32CacheMap = HashMap<F32CacheKey, Arc<ResidentF32Buffer>>;

static F32_CACHE: Mutex<Option<F32CacheMap>> = Mutex::new(None);

pub(crate) fn resident_f32_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &[f32],
) -> Result<Arc<ResidentF32Buffer>, MetalError> {
    let key = (data.as_ptr() as usize, data.len());
    let nbytes = std::mem::size_of_val(data);
    {
        let guard = F32_CACHE.lock().unwrap();
        if let Some(cache) = guard.as_ref() {
            if let Some(cached) = cache.get(&key) {
                return Ok(cached.clone());
            }
        }
    }

    let mut guard = F32_CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(HashMap::new);
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.clone());
    }

    let budget = weight_cache_budget_bytes();
    let used: usize = cache.values().map(|b| b.nbytes).sum();
    if used.saturating_add(nbytes) > budget {
        // Drop everything and retry with a clean slate for this matrix.
        // Better than silently re-uploading forever under a tight budget.
        cache.clear();
    }
    if nbytes > budget {
        // Matrix alone exceeds budget: one-shot upload, do not cache.
        let mut data_owned = data.to_vec();
        let buffer = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(data_owned.as_mut_ptr() as *mut _).unwrap(),
                nbytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocFailed)?;
        return Ok(Arc::new(ResidentF32Buffer { buffer, nbytes }));
    }

    let mut data_owned = data.to_vec();
    let buffer = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(data_owned.as_mut_ptr() as *mut _).unwrap(),
            nbytes,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;
    let cached = Arc::new(ResidentF32Buffer { buffer, nbytes });
    cache.insert(key, cached.clone());
    Ok(cached)
}

/// One quantized matvec to encode into a fused Metal command buffer
/// (shared activation `x`, one `waitUntilCompleted` for the batch).
#[derive(Clone, Copy)]
pub struct MatvecLaunch<'a> {
    pub kernel_src: &'static str,
    pub fn_name: &'static str,
    pub block_bytes: usize,
    pub block_elems: usize,
    pub weights: &'a [u8],
    pub rows: usize,
    pub row_bytes: usize,
    /// Output rows owned by one threadgroup (`1` = legacy; `2` = Q5_K; `4` = Q4_K/Q6_K).
    pub rows_per_tg: usize,
}

/// Encodes every launch into a single compute command buffer sharing
/// one uploaded `x`, then waits once. Independent projections that
/// share an activation (e.g. Q/K/V) should use this instead of N
/// separate `launch_*_matvec` calls.
pub fn launch_matvec_fused(
    x: &[f32],
    launches: &[MatvecLaunch<'_>],
) -> Result<Vec<Vec<f32>>, MetalError> {
    if launches.is_empty() {
        return Ok(Vec::new());
    }
    for launch in launches {
        let n_blocks_per_row = launch.row_bytes / launch.block_bytes;
        assert_eq!(
            launch.weights.len(),
            launch.rows * launch.row_bytes,
            "weights must be exactly rows * row_bytes"
        );
        assert_eq!(
            x.len(),
            n_blocks_per_row * launch.block_elems,
            "x must have exactly n_blocks_per_row * block_elems elements"
        );
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    // Check if x is already resident from the dense stack (final_norm
    // output). If so, skip upload and use the resident buffer. Always
    // clear TLS afterward to prevent stale matches.
    let x_buf = if let Some(resident) = take_resident_activation_if_matches(x) {
        resident
    } else {
        // No match or no TLS set — clear any stale TLS and upload normally.
        clear_resident_activation();
        let mut x_owned = x.to_vec();
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(x_owned.as_mut_ptr() as *mut _).unwrap(),
                x_owned.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocFailed)?
    };

    let mut weight_bufs = Vec::with_capacity(launches.len());
    let mut out_bufs = Vec::with_capacity(launches.len());
    for launch in launches {
        weight_bufs.push(resident_weight_buffer(device, launch.weights)?);
        out_bufs.push(
            device
                .newBufferWithLength_options(launch.rows * 4, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
        );
    }

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    // One compute encoder for the whole fused batch — creating an
    // encoder per matvec (previous behavior) paid Metal encoder setup
    // cost N times and is a large share of the ~14× gap vs ggml-metal.
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    for (i, launch) in launches.iter().enumerate() {
        encode_matvec(
            &encoder,
            device,
            launch,
            &weight_bufs[i],
            &x_buf,
            &out_bufs[i],
        )?;
    }
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let mut outs = Vec::with_capacity(launches.len());
    for (i, launch) in launches.iter().enumerate() {
        let out_ptr = out_bufs[i].contents();
        let out_slice =
            unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, launch.rows) };
        outs.push(out_slice.to_vec());
    }
    Ok(outs)
}

/// Dense SwiGLU FFN on Metal with device-resident activations:
/// one upload of `x`, gate+up matvecs → SiLU×up → down, one download.
/// Matches CUDA [`ferrox_cuda::gpu::launch_dense_ffn_swiglu`] for MoE
/// experts; weights stay in the resident cache across calls.
pub fn launch_dense_ffn_swiglu(
    gate: &MatvecLaunch<'_>,
    up: &MatvecLaunch<'_>,
    down: &MatvecLaunch<'_>,
    x: &[f32],
) -> Result<Vec<f32>, MetalError> {
    assert_eq!(gate.rows, up.rows, "gate/up row counts must match");
    assert!(down.rows > 0);
    let n_blocks_gate = gate.row_bytes / gate.block_bytes;
    assert_eq!(
        x.len(),
        n_blocks_gate * gate.block_elems,
        "x length must match gate cols"
    );
    assert_eq!(
        up.row_bytes / up.block_bytes * up.block_elems,
        x.len(),
        "up cols must match x"
    );
    let n_blocks_down = down.row_bytes / down.block_bytes;
    assert_eq!(
        n_blocks_down * down.block_elems,
        gate.rows,
        "down cols must equal gate rows (SwiGLU width)"
    );

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let x_buf = if let Some(resident) = take_resident_activation_if_matches(x) {
        resident
    } else {
        clear_resident_activation();
        let mut x_owned = x.to_vec();
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(x_owned.as_mut_ptr() as *mut _).unwrap(),
                x_owned.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocFailed)?
    };

    let gate_w = resident_weight_buffer(device, gate.weights)?;
    let up_w = resident_weight_buffer(device, up.weights)?;
    let down_w = resident_weight_buffer(device, down.weights)?;
    let gate_buf = device
        .newBufferWithLength_options(gate.rows * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;
    let up_buf = device
        .newBufferWithLength_options(up.rows * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;
    let act_buf = device
        .newBufferWithLength_options(gate.rows * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;
    let out_buf = device
        .newBufferWithLength_options(down.rows * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_matvec(&encoder, device, gate, &gate_w, &x_buf, &gate_buf)?;
    encode_matvec(&encoder, device, up, &up_w, &x_buf, &up_buf)?;
    crate::elem::encode_silu_mul(
        &encoder,
        device,
        &gate_buf,
        &up_buf,
        &act_buf,
        gate.rows as u32,
    )?;
    encode_matvec(&encoder, device, down, &down_w, &act_buf, &out_buf)?;
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let out_ptr = out_buf.contents();
    Ok(unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, down.rows).to_vec() })
}

/// Encodes one quantized matvec into an existing compute encoder (no
/// commit/wait). Used by [`crate::attn`] to fuse QKV→RoPE→GQA→O.
pub(crate) fn encode_matvec(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    launch: &MatvecLaunch<'_>,
    weight: &ResidentWeightBuffer,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), MetalError> {
    encode_matvec_with_offsets(encoder, device, launch, weight, x_buf, 0, out_buf, 0)
}

/// Shared internal launch plumbing for every matvec kernel in this
/// module: the exact device/library/pipeline/buffer/dispatch sequence
/// `launch_q8_0_matvec` used on its own before this helper existed
/// (extracted once a second, third, fourth, and fifth near-identical
/// copy would otherwise have been needed) -- only the compiled kernel
/// source/function name and each format's per-row block byte/element
/// counts differ between formats. The device/command queue, each
/// kernel's compiled pipeline, and quantized weight buffers are
/// process-wide and persistent (see `shared_metal`/`ensure_pipeline`/
/// `resident_weight_buffer` above) -- only the per-call activation
/// upload, command buffer/encoder, and result download remain.
/// Single-matvec callers go through [`launch_matvec_fused`].
#[allow(clippy::too_many_arguments)]
fn launch_matvec(
    kernel_src: &'static str,
    fn_name: &'static str,
    block_bytes: usize,
    block_elems: usize,
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    let rows_per_tg = match fn_name {
        "q4_k_matvec" | "q6_k_matvec" | "iq4_xs_matvec" => 4,
        "q5_k_matvec" | "q8_0_matvec" => 2,
        _ => 1,
    };
    let mut outs = launch_matvec_fused(
        x,
        &[MatvecLaunch {
            kernel_src,
            fn_name,
            block_bytes,
            block_elems,
            weights,
            rows,
            row_bytes,
            rows_per_tg,
        }],
    )?;
    Ok(outs.pop().unwrap())
}

/// Like [`encode_matvec`], but binds `x` / `out` at byte offsets into
/// shared buffers. Used by [`launch_matvec_batch`] so N activations
/// share one uploaded `x_batch` buffer and one output buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_matvec_with_offsets(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    launch: &MatvecLaunch<'_>,
    weight: &ResidentWeightBuffer,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    x_byte_offset: usize,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    out_byte_offset: usize,
) -> Result<(), MetalError> {
    let cached_pipeline = ensure_pipeline(device, launch.kernel_src, launch.fn_name)?;
    let pipeline = &cached_pipeline.0;
    let n_blocks_per_row = launch.row_bytes / launch.block_bytes;
    let rows_per_tg = launch.rows_per_tg.max(1);
    let (tg_threads, tg_mem_bytes) = match launch.fn_name {
        // ggml Q4_K / Q5_K / Q6_K: 2 simdgroups × 32 lanes, register packs (no TG mem).
        "q4_k_matvec" | "q5_k_matvec" | "q6_k_matvec" => (64usize, 0usize),
        // ggml Q8_0: NSG=4 simdgroups on nr0=2 rows; 2x4 floats of TG
        // reduce scratch (min 16-byte TG allocation granularity).
        "q8_0_matvec" => (128usize, 32usize),
        // ggml IQ4_XS: 2 simdgroups x 32 lanes; 32 floats of TG memory
        // hold the non-linear codebook (one copy per 16 lanes).
        "iq4_xs_matvec" => (64usize, 128usize),
        _ if rows_per_tg > 1 => (32usize, 256 * 4),
        _ => {
            let tg = n_blocks_per_row.next_power_of_two().clamp(32, 256);
            (tg, tg * 4)
        }
    };
    let n_tg = launch.rows.div_ceil(rows_per_tg);
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&weight.buffer), weight.weight_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(x_buf), x_byte_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf), out_byte_offset, 2);
        let mut row_bytes_u32 = launch.row_bytes as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut row_bytes_u32 as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut n_blocks_u32 = n_blocks_per_row as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_blocks_u32 as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        if rows_per_tg > 1 {
            let mut n_rows_u32 = launch.rows as u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut n_rows_u32 as *mut u32 as *mut _).unwrap(),
                4,
                5,
            );
        }
        if tg_mem_bytes > 0 {
            encoder.setThreadgroupMemoryLength_atIndex(tg_mem_bytes, 0);
        }
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// One weight matrix × N activations in a single Metal command buffer
/// (shared resident weights, one `x_batch` upload, one
/// `waitUntilCompleted`).
///
/// Distinct from [`launch_matvec_fused`] (shared **one** `x`, different
/// weight matrices) and from `MatvecLaunch::rows_per_tg` (multiple
/// **weight rows** per threadgroup for a single activation). This is
/// multi-`x`: `x_batch` is layout `[batch, cols]`, returned `y` is
/// `[batch, rows]`.
///
/// For Q4_K / Q6_K with `batch_size >= 2`, prefer
/// [`launch_q4_k_matmul_batch`] / [`launch_q6_k_matmul_batch`] (real
/// multi-x kernel with weight reuse). This path encodes N matvecs.
pub fn launch_matvec_batch(
    launch: &MatvecLaunch<'_>,
    x_batch: &[f32],
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    let n_blocks_per_row = launch.row_bytes / launch.block_bytes;
    let cols = n_blocks_per_row * launch.block_elems;
    assert_eq!(
        launch.weights.len(),
        launch.rows * launch.row_bytes,
        "weights must be exactly rows * row_bytes"
    );
    assert_eq!(
        x_batch.len(),
        batch_size * cols,
        "x_batch must be batch_size * cols"
    );

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let mut x_owned = x_batch.to_vec();
    let x_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(x_owned.as_mut_ptr() as *mut _).unwrap(),
            x_owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;

    let weight_buf = resident_weight_buffer(device, launch.weights)?;
    let out_elems = batch_size * launch.rows;
    let out_buf = device
        .newBufferWithLength_options(out_elems * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    for b in 0..batch_size {
        encode_matvec_with_offsets(
            &encoder,
            device,
            launch,
            &weight_buf,
            &x_buf,
            b * cols * 4,
            &out_buf,
            b * launch.rows * 4,
        )?;
    }
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let out_ptr = out_buf.contents();
    let out_slice =
        unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, out_elems) };
    Ok(out_slice.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_q8_0_test_matrix(rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.037).sin()).collect();
        let mut weights = Vec::new();
        let mut expected = Vec::new();
        for r in 0..rows {
            let row_vals: Vec<f32> = (0..cols)
                .map(|i| ((r * 7 + i) as f32 * 0.013).cos())
                .collect();
            let q = ferrox_quant::quantize_q8_0(&row_vals);
            expected.push(ferrox_quant::dot_q8_0_f32_scalar(&q, &x));
            weights.extend_from_slice(&q);
        }
        (weights, x, expected)
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q8_0_matvec_matches_cpu_reference() {
        let rows = 8;
        let cols = 256;
        let row_bytes = (cols / ferrox_quant::Q8_0_BLOCK_ELEMS) * ferrox_quant::Q8_0_BLOCK_BYTES;
        let (weights, x, expected) = real_q8_0_test_matrix(rows, cols);

        let result = launch_q8_0_matvec(&weights, &x, rows, row_bytes).expect("kernel launch");

        assert_eq!(result.len(), expected.len());
        for (i, (a, b)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "row {i}: gpu={a} cpu={b} (diff too large)"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_iq4_xs_matvec_matches_cpu_reference() {
        // No IQ4_XS quantizer exists in `ferrox_quant` (encode is
        // llama.cpp-side); any bit pattern is a valid block, so build
        // deterministic pseudo-random blocks (finite small `d`) and
        // compare against the fused CPU dot. rows=6 exercises the
        // n_rows guard (not a multiple of rows_per_tg=4); 2 blocks/row
        // exercises the odd/even block split across lane groups.
        let rows = 6;
        let cols = 512;
        let blocks_per_row = cols / ferrox_quant::IQ4_XS_BLOCK_ELEMS;
        let row_bytes = blocks_per_row * ferrox_quant::IQ4_XS_BLOCK_BYTES;

        let mut state = 0x1234_5678u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let mut weights = vec![0u8; rows * row_bytes];
        for b in weights.iter_mut() {
            *b = next();
        }
        for r in 0..rows {
            for ib in 0..blocks_per_row {
                let off = r * row_bytes + ib * ferrox_quant::IQ4_XS_BLOCK_BYTES;
                let d = half::f16::from_f32(0.01 + 0.002 * (r * blocks_per_row + ib) as f32);
                weights[off..off + 2].copy_from_slice(&d.to_le_bytes());
            }
        }

        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.05).sin()).collect();
        let expected: Vec<f32> = (0..rows)
            .map(|r| ferrox_quant::dot_iq4_xs_f32(&weights[r * row_bytes..(r + 1) * row_bytes], &x))
            .collect();

        let result = launch_iq4_xs_matvec(&weights, &x, rows, row_bytes).expect("kernel launch");

        assert_eq!(result.len(), expected.len());
        for (i, (a, b)) in result.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-3 * b.abs().max(1.0);
            assert!((a - b).abs() < tol, "row {i}: gpu={a} cpu={b} tol={tol}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn probe_finds_a_real_device_name() {
        let name = probe().expect("this dev machine has a real Metal GPU");
        assert!(!name.is_empty());
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_0_matvec_matches_cpu_reference() {
        // Real, non-trivial Q4_0 rows built directly (no `quantize_q4_0`
        // producer exists in `ferrox_quant` -- Q4_0 is load-only in this
        // codebase, same convention `ferrox-cuda`'s Q4_0 test uses).
        // Built `blocks_per_row` blocks per row (not a single hard-coded
        // block regardless of `cols`) -- an earlier version of the
        // equivalent CUDA test got this wrong and only caught it via a
        // real out-of-bounds panic on real GPU hardware; built correctly
        // here from the start given that documented lesson.
        let rows = 4;
        let cols = 64;
        let blocks_per_row = cols / ferrox_quant::Q4_0_BLOCK_ELEMS;
        let row_bytes = blocks_per_row * ferrox_quant::Q4_0_BLOCK_BYTES;

        let mut weights = Vec::new();
        for r in 0..rows {
            for b in 0..blocks_per_row {
                weights.extend_from_slice(
                    &half::f16::from_f32(0.05 + (r * blocks_per_row + b) as f32 * 0.01)
                        .to_le_bytes(),
                );
                for i in 0..16u8 {
                    let lo = (i + r as u8 + b as u8) % 16;
                    let hi = (15 - i + r as u8 + b as u8) % 16;
                    weights.push(lo | (hi << 4));
                }
            }
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.09).sin()).collect();
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                let row_slice = &weights[r * row_bytes..(r + 1) * row_bytes];
                ferrox_quant::dot_q4_0_f32_scalar(row_slice, &x)
            })
            .collect();

        let result = launch_q4_0_matvec(&weights, &x, rows, row_bytes).expect("kernel launch");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "row {i}: GPU={got} CPU reference={want}"
            );
        }
    }

    /// Deterministic pseudo-random byte generator for building real,
    /// non-trivial K-quant block bytes -- no `quantize_qX_k` producer
    /// exists in `ferrox_quant` (these formats are load-only), the same
    /// convention `ferrox-cuda`'s equivalent tests use.
    fn pseudo_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    /// GPU-vs-CPU agreement check for the K-quant kernels, using a
    /// *relative* error bound rather than a fixed absolute one -- same
    /// reasoning and same tolerance as `ferrox-cuda::gpu::tests::assert_close_relative`:
    /// these kernels apply each block's scale/min *inside* a per-element
    /// `acc += (d1 * q - min1) * x[i]` accumulation (hundreds of float
    /// multiply-adds per row), so GPU-vs-CPU results can differ by
    /// float-rounding-order alone (Metal's default fast-math mode may
    /// contract `a*b+c` into a single-rounding fused multiply-add the
    /// same way NVRTC does; plain Rust `f32` arithmetic does not
    /// auto-contract). Also treats NaN==NaN as agreement, for the same
    /// reason the CUDA-side helper does: pseudo-random block bytes can
    /// happen to decode as a NaN/Inf `half` scale, and two backends that
    /// both produce NaN from the same degenerate input have actually
    /// agreed, even though IEEE754 NaN comparisons are always false.
    fn assert_close_relative(got: f32, want: f32, row: usize) {
        if want.is_nan() {
            assert!(
                got.is_nan(),
                "row {row}: CPU reference is NaN but GPU={got} is not"
            );
            return;
        }
        let tol = 1e-4 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "row {row}: GPU={got} CPU reference={want} (relative tolerance {tol})"
        );
    }

    /// Builds `rows` real (non-zero, non-trivial) blocks of `block_bytes`
    /// each for a K-quant format, and the matching `expected` output via
    /// `scalar_dot` (`ferrox_quant::dot_q{4,5,6}_k_f32_scalar` --
    /// independently verified elsewhere in this workspace), so the
    /// ignored GPU tests below check real numerical agreement with that
    /// trusted CPU reference, not just "the launch didn't error."
    fn real_k_quant_test_matrix(
        rows: usize,
        cols: usize,
        block_bytes: usize,
        scalar_dot: impl Fn(&[u8], &[f32]) -> f32,
    ) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let n_blocks_per_row = cols / 256;
        let row_bytes = n_blocks_per_row * block_bytes;
        let mut weights = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            weights.extend(pseudo_bytes(r as u32 + 1, row_bytes));
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.021).sin()).collect();
        let expected: Vec<f32> = (0..rows)
            .map(|r| scalar_dot(&weights[r * row_bytes..(r + 1) * row_bytes], &x))
            .collect();
        (weights, x, expected)
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_k_matvec_matches_cpu_reference() {
        // 5 rows exercises the multi-row TG (NR=4) plus a partial last group.
        let rows = 5;
        let cols = 512; // 2 Q4_K super-blocks per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            ferrox_quant::Q4_K_BLOCK_BYTES,
            ferrox_quant::dot_q4_k_f32_scalar,
        );

        let result = launch_q4_k_matvec(&weights, &x, rows, ferrox_quant::Q4_K_BLOCK_BYTES * 2)
            .expect("kernel launch");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q5_k_matvec_matches_cpu_reference() {
        // 5 rows exercises NSG=2 (2 rows/TG) plus a partial last group.
        let rows = 5;
        let cols = 512; // 2 Q5_K super-blocks per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            ferrox_quant::Q5_K_BLOCK_BYTES,
            ferrox_quant::dot_q5_k_f32_scalar,
        );

        let result = launch_q5_k_matvec(&weights, &x, rows, ferrox_quant::Q5_K_BLOCK_BYTES * 2)
            .expect("kernel launch");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q6_k_matvec_matches_cpu_reference() {
        let rows = 5; // multi-row TG (NR=4) + partial last group
        let cols = 512; // 2 Q6_K super-blocks per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            ferrox_quant::Q6_K_BLOCK_BYTES,
            ferrox_quant::dot_q6_k_f32_scalar,
        );

        let result = launch_q6_k_matvec(&weights, &x, rows, ferrox_quant::Q6_K_BLOCK_BYTES * 2)
            .expect("kernel launch");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_matvec_batch_matches_sequential_launches() {
        // Multi-x batch (distinct from rows_per_tg multi-row): N
        // activations share one weight matrix / one CB wait.
        let rows = 8;
        let cols = 256;
        let batch_size = 4;
        let row_bytes = (cols / ferrox_quant::Q8_0_BLOCK_ELEMS) * ferrox_quant::Q8_0_BLOCK_BYTES;
        let (weights, _x0, _) = real_q8_0_test_matrix(rows, cols);

        let mut x_batch = Vec::with_capacity(batch_size * cols);
        let mut expected = Vec::with_capacity(batch_size * rows);
        for b in 0..batch_size {
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i + b * 17) as f32 * 0.041).sin())
                .collect();
            let y = launch_q8_0_matvec(&weights, &x, rows, row_bytes).expect("single launch");
            expected.extend_from_slice(&y);
            x_batch.extend_from_slice(&x);
        }

        let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
            matvec_launch_meta("Q8_0").expect("Q8_0 meta");
        let launch = MatvecLaunch {
            kernel_src: src,
            fn_name,
            block_bytes,
            block_elems,
            weights: &weights,
            rows,
            row_bytes,
            rows_per_tg,
        };
        let got = launch_matvec_batch(&launch, &x_batch, batch_size).expect("batch launch");
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "elem {i}: batch={a} sequential={b} (diff too large)"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_k_matmul_batch_matches_sequential_apply() {
        // Multi-x Q4_K matmul vs N sequential matvecs (parity).
        let rows = 5;
        let cols = 512;
        let batch_size = 6; // exercises NB=4 tile + partial last tile
        let row_bytes = (cols / 256) * ferrox_quant::Q4_K_BLOCK_BYTES;
        let (weights, _x0, _) = real_k_quant_test_matrix(
            rows,
            cols,
            ferrox_quant::Q4_K_BLOCK_BYTES,
            ferrox_quant::dot_q4_k_f32_scalar,
        );

        let mut x_batch = Vec::with_capacity(batch_size * cols);
        let mut expected = Vec::with_capacity(batch_size * rows);
        for b in 0..batch_size {
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i + b * 19) as f32 * 0.023).sin())
                .collect();
            let y = launch_q4_k_matvec(&weights, &x, rows, row_bytes).expect("matvec");
            expected.extend_from_slice(&y);
            x_batch.extend_from_slice(&x);
        }

        let got = launch_q4_k_matmul_batch(&weights, &x_batch, rows, row_bytes, batch_size)
            .expect("matmul batch");
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*a, *b, i);
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q6_k_matmul_batch_matches_sequential_apply() {
        let rows = 5;
        let cols = 512;
        let batch_size = 6;
        let row_bytes = (cols / 256) * ferrox_quant::Q6_K_BLOCK_BYTES;
        let (weights, _x0, _) = real_k_quant_test_matrix(
            rows,
            cols,
            ferrox_quant::Q6_K_BLOCK_BYTES,
            ferrox_quant::dot_q6_k_f32_scalar,
        );

        let mut x_batch = Vec::with_capacity(batch_size * cols);
        let mut expected = Vec::with_capacity(batch_size * rows);
        for b in 0..batch_size {
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i + b * 23) as f32 * 0.027).sin())
                .collect();
            let y = launch_q6_k_matvec(&weights, &x, rows, row_bytes).expect("matvec");
            expected.extend_from_slice(&y);
            x_batch.extend_from_slice(&x);
        }

        let got = launch_q6_k_matmul_batch(&weights, &x_batch, rows, row_bytes, batch_size)
            .expect("matmul batch");
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*a, *b, i);
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_k_mul_mm_matches_cpu_matvec() {
        // Q4_K mul_mm vs N× matvec: same dequant identity, so the batched
        // path must reproduce each per-activation matvec.
        let rows = 9;
        let cols = 512; // 2 Q4_K blocks/row
        let batch_size = 7;
        let row_bytes = (cols / 256) * ferrox_quant::Q4_K_BLOCK_BYTES;
        let (weights, _x0, _) = real_k_quant_test_matrix(
            rows,
            cols,
            ferrox_quant::Q4_K_BLOCK_BYTES,
            ferrox_quant::dot_q4_k_f32_scalar,
        );

        let mut x_batch = Vec::with_capacity(batch_size * cols);
        let mut expected = Vec::with_capacity(batch_size * rows);
        for b in 0..batch_size {
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i + b * 23) as f32 * 0.027).sin())
                .collect();
            let y = launch_q4_k_matvec(&weights, &x, rows, row_bytes).expect("matvec");
            expected.extend_from_slice(&y);
            x_batch.extend_from_slice(&x);
        }

        let got =
            launch_q4_k_mul_mm(&weights, &x_batch, rows, row_bytes, batch_size).expect("mul_mm");
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*a, *b, i);
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_0_mul_mm_matches_cpu_matvec() {
        // Q4_0 mul_mm vs N× matvec. Finite f16 scales (same construction as
        // `launch_q4_0_matvec_matches_cpu_reference`) — raw pseudo_bytes can
        // decode to NaN halves and only exercise NaN==NaN agreement.
        let rows = 67;
        let cols = 320; // 10 Q4_0 blocks/row
        let batch_size = 9;
        let blocks_per_row = cols / ferrox_quant::Q4_0_BLOCK_ELEMS;
        let row_bytes = blocks_per_row * ferrox_quant::Q4_0_BLOCK_BYTES;

        let mut weights = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            for b in 0..blocks_per_row {
                weights.extend_from_slice(
                    &half::f16::from_f32(0.05 + (r * blocks_per_row + b) as f32 * 0.01)
                        .to_le_bytes(),
                );
                for i in 0..16u8 {
                    let lo = (i + r as u8 + b as u8) % 16;
                    let hi = (15 - i + r as u8 + b as u8) % 16;
                    weights.push(lo | (hi << 4));
                }
            }
        }

        let mut x_batch = Vec::with_capacity(batch_size * cols);
        let mut expected = Vec::with_capacity(batch_size * rows);
        for b in 0..batch_size {
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i + b * 23) as f32 * 0.027).sin())
                .collect();
            let y = launch_q4_0_matvec(&weights, &x, rows, row_bytes).expect("matvec");
            expected.extend_from_slice(&y);
            x_batch.extend_from_slice(&x);
        }

        let got =
            launch_q4_0_mul_mm(&weights, &x_batch, rows, row_bytes, batch_size).expect("mul_mm");
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*a, *b, i);
        }
    }
}
