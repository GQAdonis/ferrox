//! Device K/V caches default to **f16** (llama.cpp `-ctk f16 -ctv f16`).
//! Quantized stores: `FERROX_CTK=q8_0|turbo8|turbo4|fp8` (turbo3 still falls
//! back to f16). Quant caches dequant to a process-wide f16 scratch before
//! FA/GQA. Q/activations stay f32. Append converts on GPU.
//!
//! **Decode (B=1):** Q/K/V/O matvecs fused with RoPE→KV→GQA in one command
//! buffer ([`launch_decode_attn_block`]); dense layers can continue through
//! residual → FFN in the same CB ([`launch_decode_dense_layer`] /
//! [`launch_decode_dense_stack`]).
//!
//! **Prefill (T≥1):** dense layers with B≥4 can run
//! [`launch_prefill_dense_layer`] (one layer) or [`launch_prefill_dense_stack`]
//! (consecutive layers, one CB, one host readback) — RMSNorm → Q/K/V
//! `mul_mm_sg` → RoPE/KV/GQA → O → FFN (gate∥up→act→down) with activations
//! resident on GPU scratch (decode-stack barriers). Otherwise host (or
//! batched) Q/K/V feed [`launch_prefill_attn_block`] — multi-pos RoPE,
//! batch KV append into [`MetalKvBuffers`], causal GQA — so decode can
//! skip [`MetalKvBuffers::upload_from_host`] when seq_lens already match.
//!
//! Prefill GEMM timing: `FERROX_METAL_MM_TIMING=1` (see
//! [`crate::gpu::launch_dense_ffn_swiglu_batch`] / mul_mm_sg launches, and
//! [`launch_prefill_dense_layer`] / [`launch_prefill_dense_stack`]
//! setup/gpu/readback totals).
//!
//! Scope:
//! - `LLAMA_ROPE_TYPE_NORM` (interleaved) or `NEOX` ± Llama-3 freq factors
//! - Full-causal GQA (no sliding window); online-softmax kernels
//! - Prefill dense stack supports QKV bias + QK-norm via [`AttnExtras`]
//!   (same order as CPU: bias → norm → RoPE)
//!
//! Enable with `FERROX_METAL_ATTN=1` (also requires dense Metal / `FERROX_METAL`).
//! Optional `FERROX_METAL_LOGITS=1` folds final_norm + lm_head into the dense
//! or MoE stack CB (downloads vocab logits). Default off: host lm_head after
//! downloading hidden — measured ~2× faster on Llama-3.1-8B Q4_K_M.
//!
//! Greedy decode (`temperature<=0`, hooked from `ferrox-server::generate` /
//! `ferrox-cli`) can fold final_norm + lm_head + **argmax** into the same CB
//! (dense [`launch_decode_dense_stack`] or MoE [`launch_moe_decode_stack`])
//! and download only the top-1 token id — **default on** for greedy;
//! opt out with `FERROX_METAL_GREEDY_GPU=0`. Embedding gather can also run
//! on-GPU (`get_rows`) so the dense stack needs no host `dequant_row` upload.
//!
//! `FERROX_METAL_FA_VEC=0` disables llama-style FA-vec decode **and** prefill
//! (default **on** for `head_dim` in {64,96,128,256}). Other head dims keep
//! legacy online-softmax GQA.
//!
//! `FERROX_CTK` selects KV dtype ([`MetalKvDtype`]); see [`is_implemented`].
use crate::elem::{
    encode_add_rms_norm, encode_add_rms_norm_batch, encode_argmax, encode_f32_to_f16,
    encode_gelu_mul, encode_rms_norm, encode_rms_norm_at, encode_rms_norm_batch,
    encode_rms_norm_f32_to_f16_batch, encode_rms_norm_per_head_batch, encode_silu_mul,
    encode_vec_add, encode_vec_add_at, warm_prefill_elem_pipelines,
};
use crate::embd::{encode_get_rows, EmbdKind};
use crate::gpu::{
    compute_encoder_concurrent, encode_matvec, encode_matvec_with_offsets, encode_moe_topk_softmax,
    encode_mul_mm_sg_f16, encode_q4_0_moe_gate_then_up_silu, encode_q4_0_moe_gate_up_id,
    encode_q4_0_moe_gate_up_silu_fused, encode_q4_0_moe_id, encode_q4_0_moe_id_ex,
    encode_q4_0_moe_topk, encode_q4_0_mul_mm, ensure_pipeline, memory_barrier_buffers,
    memory_barrier_resources, resident_f32_buffer, resident_weight_buffer, shared_metal,
    warm_mul_mm_sg_pipeline, MatvecLaunch, MetalError, MoeExpertLaunch, MoePackedQ4, MulMmSgLaunch,
    ResidentF32Buffer, ResidentWeightBuffer,
};
use crate::moe_ranges::MoeMemRanges;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

/// RoPE pairing convention for Metal kernels. Mirrors
/// `ferrox_models::config::RopeLayout` / llama.cpp `llama_rope_type`
/// without pulling models into this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalRopeLayout {
    /// Adjacent pairs `(2*i, 2*i+1)` — `LLAMA_ROPE_TYPE_NORM`.
    Norm,
    /// Split-half pairs `(i, i+half)` — `LLAMA_ROPE_TYPE_NEOX`.
    Neox,
}

/// Whether the fused Metal attention block should run (in addition to
/// dense Metal matvecs). Default off until measured; `1|true|on` enables.
pub fn metal_attn_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("FERROX_METAL_ATTN").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("attn")
        )
    })
}

/// Keep MoE residual on Metal across attn→router→experts (default **on**).
/// Needs F32 router encode (OLMoE). Opt out: `FERROX_METAL_MOE_RESIDENT=0`.
pub fn metal_moe_resident_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("FERROX_METAL_MOE_RESIDENT").ok().as_deref(),
            Some("0") | Some("false") | Some("off")
        )
    })
}

/// Fold final RMSNorm + lm_head into the Metal dense stack (return vocab
/// logits). Default **off** — host lm_head after hidden download recovered
/// ~20 predicted tok/s vs ~10 with logits-in-stack on Llama-3.1-8B Q4_K_M.
pub fn metal_logits_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("FERROX_METAL_LOGITS").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("logits")
        )
    })
}

/// Whether greedy GPU argmax-in-stack is allowed. Default **on** when
/// Metal attn is in use for `temperature<=0` (parallel TG argmax ≈ host
/// lm_head on Host B). Opt out with `FERROX_METAL_GREEDY_GPU=0|false|off`.
pub fn metal_greedy_gpu_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        match std::env::var("FERROX_METAL_GREEDY_GPU").ok().as_deref() {
            Some("0") | Some("false") | Some("off") | Some("no") => false,
            Some("1") | Some("true") | Some("on") | Some("greedy") => true,
            // Unset: default on (was opt-in while sequential argmax regressed).
            None => true,
            Some(_) => true,
        }
    })
}

thread_local! {
    /// Per-request flag set by `ferrox-server::generate` when
    /// `temperature<=0`. Thread-local so concurrent requests sharing one
    /// `Arc<Decoder>` do not race.
    static GREEDY_ARGMAX: Cell<bool> = const { Cell::new(false) };
}

/// Enable/disable greedy GPU argmax for the current thread's decode steps.
pub fn set_metal_greedy_argmax(on: bool) {
    GREEDY_ARGMAX.with(|c| c.set(on));
}

/// True when this thread should fold final_norm+lm_head+argmax into the
/// dense/MoE stack and return a 1-element `[token_id as f32]` instead of
/// hidden or full vocab logits.
pub fn metal_greedy_argmax_active() -> bool {
    metal_greedy_gpu_enabled() && GREEDY_ARGMAX.with(|c| c.get())
}

// Norm (interleaved) and NeoX (split-half) kernels share the same buffer
// layout so `encode_rope` only swaps the entry point. Math mirrors
// `ferrox_core::attention::{apply_rope_interleaved, apply_rope}` —
// no Candle / third-party RoPE dependency.
const ROPE_NORM_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rope_interleaved_heads(
    device float* vecs [[buffer(0)]],
    constant uint& n_heads [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    constant uint& pos [[buffer(4)]],
    device const float* freq_factors [[buffer(5)]],
    constant uint& use_freq_factors [[buffer(6)]],
    uint h [[thread_position_in_grid]]
) {
    if (h >= n_heads) return;
    device float* vec = vecs + h * head_dim;
    uint half_dim = head_dim / 2u;
    for (uint i = 0; i < half_dim; i++) {
        float freq = 1.0f / pow(theta, (2.0f * float(i)) / float(head_dim));
        float angle = float(pos) * freq;
        if (use_freq_factors != 0u) {
            angle /= freq_factors[i];
        }
        float s = sin(angle);
        float c = cos(angle);
        float a = vec[2u * i];
        float b = vec[2u * i + 1u];
        vec[2u * i] = a * c - b * s;
        vec[2u * i + 1u] = a * s + b * c;
    }
}
"#;

const ROPE_NEOX_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rope_neox_heads(
    device float* vecs [[buffer(0)]],
    constant uint& n_heads [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    constant uint& pos [[buffer(4)]],
    device const float* freq_factors [[buffer(5)]],
    constant uint& use_freq_factors [[buffer(6)]],
    uint h [[thread_position_in_grid]]
) {
    if (h >= n_heads) return;
    device float* vec = vecs + h * head_dim;
    uint half_dim = head_dim / 2u;
    for (uint i = 0; i < half_dim; i++) {
        float freq = 1.0f / pow(theta, (2.0f * float(i)) / float(head_dim));
        float angle = float(pos) * freq;
        if (use_freq_factors != 0u) {
            angle /= freq_factors[i];
        }
        float s = sin(angle);
        float c = cos(angle);
        float a = vec[i];
        float b = vec[i + half_dim];
        vec[i] = a * c - b * s;
        vec[i + half_dim] = a * s + b * c;
    }
}
"#;

const KV_APPEND_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Append f32 K/V token into an f16-resident cache (llama.cpp default).
kernel void kv_append(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& offset_elems [[buffer(2)]],
    constant uint& n_elems [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n_elems) {
        dst[offset_elems + i] = half(src[i]);
    }
}
"#;

/// ggml Q8_0: 32 int8 values + one f16 scale (34 bytes). One thread / block.
const KV_APPEND_Q8_0_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kv_append_q8_0(
    device const float* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant uint& offset_elems [[buffer(2)]],
    constant uint& n_elems [[buffer(3)]],
    uint b [[thread_position_in_grid]]
) {
    const uint BLOCK = 32u;
    const uint BLOCK_BYTES = 34u;
    uint n_blocks = n_elems / BLOCK;
    if (b >= n_blocks) return;
    uint src_base = b * BLOCK;
    float amax = 0.0f;
    for (uint i = 0u; i < BLOCK; i++) {
        amax = fmax(amax, fabs(src[src_base + i]));
    }
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    uint dst_block = (offset_elems / BLOCK) + b;
    uint dst_base = dst_block * BLOCK_BYTES;
    half d_h = half(d);
    dst[dst_base + 0] = uchar(as_type<ushort>(d_h) & 0xFFu);
    dst[dst_base + 1] = uchar(as_type<ushort>(d_h) >> 8u);
    for (uint i = 0u; i < BLOCK; i++) {
        int q = int(round(src[src_base + i] * id));
        q = clamp(q, -127, 127);
        dst[dst_base + 2u + i] = uchar(char(q));
    }
}
"#;

const DEQUANT_Q8_0_TO_F16_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dequant_q8_0_to_f16(
    device const uchar* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n_elems [[buffer(2)]],
    uint b [[thread_position_in_grid]]
) {
    const uint BLOCK = 32u;
    const uint BLOCK_BYTES = 34u;
    uint n_blocks = n_elems / BLOCK;
    if (b >= n_blocks) return;
    uint src_base = b * BLOCK_BYTES;
    ushort d_bits = ushort(src[src_base]) | (ushort(src[src_base + 1u]) << 8u);
    float d = float(as_type<half>(d_bits));
    uint dst_base = b * BLOCK;
    for (uint i = 0u; i < BLOCK; i++) {
        char q = char(src[src_base + 2u + i]);
        dst[dst_base + i] = half(float(q) * d);
    }
}
"#;

/// TurboQuant-style 4-bit KV: f16 scale + 16 nibble bytes / 32 elems (18 B).
const KV_APPEND_TURBO4_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kv_append_turbo4(
    device const float* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant uint& offset_elems [[buffer(2)]],
    constant uint& n_elems [[buffer(3)]],
    uint b [[thread_position_in_grid]]
) {
    const uint BLOCK = 32u;
    const uint BLOCK_BYTES = 18u;
    uint n_blocks = n_elems / BLOCK;
    if (b >= n_blocks) return;
    uint src_base = b * BLOCK;
    float amax = 0.0f;
    for (uint i = 0u; i < BLOCK; i++) {
        amax = fmax(amax, fabs(src[src_base + i]));
    }
    float d = amax / 7.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    uint dst_block = (offset_elems / BLOCK) + b;
    uint dst_base = dst_block * BLOCK_BYTES;
    half d_h = half(d);
    dst[dst_base + 0] = uchar(as_type<ushort>(d_h) & 0xFFu);
    dst[dst_base + 1] = uchar(as_type<ushort>(d_h) >> 8u);
    for (uint i = 0u; i < 16u; i++) {
        int q0 = int(round(src[src_base + 2u * i] * id));
        int q1 = int(round(src[src_base + 2u * i + 1u] * id));
        q0 = clamp(q0, -8, 7);
        q1 = clamp(q1, -8, 7);
        dst[dst_base + 2u + i] = uchar((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
}
"#;

const DEQUANT_TURBO4_TO_F16_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dequant_turbo4_to_f16(
    device const uchar* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n_elems [[buffer(2)]],
    uint b [[thread_position_in_grid]]
) {
    const uint BLOCK = 32u;
    const uint BLOCK_BYTES = 18u;
    uint n_blocks = n_elems / BLOCK;
    if (b >= n_blocks) return;
    uint src_base = b * BLOCK_BYTES;
    ushort d_bits = ushort(src[src_base]) | (ushort(src[src_base + 1u]) << 8u);
    float d = float(as_type<half>(d_bits));
    uint dst_base = b * BLOCK;
    for (uint i = 0u; i < 16u; i++) {
        uchar byte = src[src_base + 2u + i];
        int q0 = int(char((byte & 0xFu) << 4) >> 4);
        int q1 = int(char((byte >> 4) << 4) >> 4);
        dst[dst_base + 2u * i] = half(float(q0) * d);
        dst[dst_base + 2u * i + 1u] = half(float(q1) * d);
    }
}
"#;

/// Whether FA-vec GQA decode is enabled.
///
/// Default: **on** for supported head dims (64 / 96 / 128 / 256) via
/// [`encode_gqa`]. `FERROX_METAL_FA_VEC=0|false|off` forces the legacy
/// online-softmax kernel. `=1|true|on|vec` keeps FA on (still only
/// dispatched for supported dims).
pub fn metal_fa_vec_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        match std::env::var("FERROX_METAL_FA_VEC").ok().as_deref() {
            Some("0") | Some("false") | Some("off") => false,
            Some("1") | Some("true") | Some("on") | Some("vec") => true,
            // Default on: llama-parity FA for d=128 is required to close the
            // ~1.15× decode gap (legacy GQA ≈ llama `-ctk f32`).
            _ => true,
        }
    })
}

/// Device KV cache element type (llama.cpp `-ctk` / `--kvcache-dtype` analogue).
///
/// Selected via `FERROX_CTK` ([`metal_kv_dtype`]). Implemented: F16, Q8_0,
/// Turbo8 (=Q8_0 wire), Turbo4, Fp8 (=Q8_0 wire). Turbo3 warns → F16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalKvDtype {
    F16,
    Q8_0,
    /// FP8-style KV (scaled int8 / Q8_0 layout).
    Fp8,
    /// TurboQuant 8-bit (Metal: same store as Q8_0; host WHT optional).
    Turbo8,
    /// TurboQuant 4-bit (WHT optional on host; Metal absmax nibble groups).
    Turbo4,
    /// TurboQuant 3-bit (experimental; not implemented yet).
    Turbo3,
}

impl MetalKvDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Fp8 => "fp8",
            Self::Turbo8 => "turbo8",
            Self::Turbo4 => "turbo4",
            Self::Turbo3 => "turbo3",
        }
    }

    pub fn is_implemented(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::Q8_0 | Self::Turbo8 | Self::Turbo4 | Self::Fp8
        )
    }

    /// True when attention must dequant store → f16 scratch before FA/GQA.
    pub fn needs_f16_scratch(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Turbo8 | Self::Turbo4 | Self::Fp8)
    }

    /// Uses ggml Q8_0 / fp8 34-byte blocks.
    fn is_q8_wire(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Turbo8 | Self::Fp8)
    }
}

/// True when `n_kv_heads * head_dim` is a multiple of ggml Q8_0 block size (32).
pub fn metal_kv_q8_0_viable(n_kv_heads: usize, head_dim: usize) -> bool {
    (n_kv_heads * head_dim).is_multiple_of(ferrox_quant::Q8_0_BLOCK_ELEMS)
}

/// turbo4 / fp8 share the 32-elem group alignment.
pub fn metal_kv_turbo4_viable(n_kv_heads: usize, head_dim: usize) -> bool {
    (n_kv_heads * head_dim).is_multiple_of(ferrox_quant::TURBO4_KV_GROUP)
}

/// Dtype actually used for new [`MetalKvBuffers`] (unimplemented / non-viable → F16).
pub fn effective_metal_kv_dtype(n_kv_heads: usize, head_dim: usize) -> MetalKvDtype {
    let requested = metal_kv_dtype();
    if !requested.is_implemented() {
        return MetalKvDtype::F16;
    }
    if requested.is_q8_wire() && !metal_kv_q8_0_viable(n_kv_heads, head_dim) {
        static WARNED: OnceLock<()> = OnceLock::new();
        let _ = WARNED.get_or_init(|| {
            eprintln!(
                "FERROX_CTK={}: n_kv_heads*head_dim={} not divisible by {}; using f16",
                requested.as_str(),
                n_kv_heads * head_dim,
                ferrox_quant::Q8_0_BLOCK_ELEMS
            );
        });
        return MetalKvDtype::F16;
    }
    if requested == MetalKvDtype::Turbo4 && !metal_kv_turbo4_viable(n_kv_heads, head_dim) {
        static WARNED: OnceLock<()> = OnceLock::new();
        let _ = WARNED.get_or_init(|| {
            eprintln!(
                "FERROX_CTK=turbo4: n_kv_heads*head_dim={} not divisible by {}; using f16",
                n_kv_heads * head_dim,
                ferrox_quant::TURBO4_KV_GROUP
            );
        });
        return MetalKvDtype::F16;
    }
    requested
}

/// Parse `FERROX_CTK` / `-ctk`-style strings. Unknown → [`MetalKvDtype::F16`].
pub fn parse_metal_kv_dtype(raw: Option<&str>) -> MetalKvDtype {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("q8_0") | Some("q8") => MetalKvDtype::Q8_0,
        Some("fp8") | Some("e4m3") => MetalKvDtype::Fp8,
        Some("turbo8") => MetalKvDtype::Turbo8,
        Some("turbo4") => MetalKvDtype::Turbo4,
        Some("turbo3") => MetalKvDtype::Turbo3,
        Some("f16") | Some("fp16") | Some("half") | Some("bf16") => MetalKvDtype::F16,
        _ => MetalKvDtype::F16,
    }
}

/// KV dtype requested by `FERROX_CTK` (default F16).
///
/// Unimplemented dtypes emit a one-time stderr warning; callers keep F16 buffers.
pub fn metal_kv_dtype() -> MetalKvDtype {
    static DTYPE: OnceLock<MetalKvDtype> = OnceLock::new();
    *DTYPE.get_or_init(|| {
        let dt = parse_metal_kv_dtype(std::env::var("FERROX_CTK").ok().as_deref());
        if !dt.is_implemented() {
            static WARNED: OnceLock<()> = OnceLock::new();
            let _ = WARNED.get_or_init(|| {
                eprintln!(
                    "FERROX_CTK={}: Metal {} KV cache not implemented yet; using f16 buffers",
                    dt.as_str(),
                    dt.as_str()
                );
            });
        }
        dt
    })
}

/// llama.cpp-style FA-vec decode for **head_dim=128**, f16 KV, NE=1, C=32.
/// One TG per head; NSG simdgroups each own every NSG-th KV tile, then
/// online-softmax merge. Replaces the old FA_VEC that recomputed V ×32.
const GQA_DECODE_FA_VEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_fa_vec(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Specialized for D=128 (host only dispatches when head_dim==128).
    constexpr uint D = 128u;
    constexpr uint D4 = 32u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    // Per-SG floats: C scores + D output.
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    so4[tiisg] = float4(0.0f);
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    // Each SG walks KV tiles: ic0 = sgitg, sgitg+nsg, ...
    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        // Q·K for all C positions: lane `ii` owns float4-slice `ii` of the head;
        // after simd_sum, every lane holds the full score for each cc.
        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = dot(sq4[tiisg], float4(k4[tiisg]));
            float sc = simd_sum(partial) * scale;
            if (softcap > 0.0f) {
                sc = softcap * tanh(sc / softcap);
            }
            scores[cc] = sc;
        }

        // Online softmax over this tile (one score per lane).
        float s_lane = (tiisg < chunk) ? scores[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        so4[tiisg] *= ms;
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O += P · V  (lane owns float4-slice tiisg of the output)
        float4 lo = float4(0.0f);
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* v4 =
                (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            lo += float4(v4[tiisg]) * ss[cc];
        }
        so4[tiisg] += lo;
    }

    // Publish S,M for cross-SG reduce (reuse ss[0], ss[1]).
    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG online-softmax merge.
    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + h * D);
        out4[tiisg] = so0[tiisg] * inv;
    }
}
"#;

/// llama.cpp-style FA-vec decode for **head_dim=64**, f16 KV, NE=2, C=32.
/// Same tile/merge structure as the d=128 kernel, but D4=16 float4
/// slices only cover half a simdgroup — so each warp processes **two**
/// KV positions per pass (half-warp `ty=0` gets even `cc`, `ty=1` odd),
/// with a 16-lane shuffle-xor dot reduce and a cross-half xor-16 merge
/// of the V accumulators. This keeps all 32 lanes busy where a naive
/// D4=16 port would idle half the warp (TinyLlama / Llama-3.2-1B are
/// d=64).
const GQA_DECODE_FA_VEC_D64_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_fa_vec_d64(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Specialized for D=64 (host only dispatches when head_dim==64).
    constexpr uint D = 64u;
    constexpr uint D4 = 16u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    // Per-SG floats: C scores + D output.
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;
    const uint tx = tiisg % D4; // float4 slice of the head
    const uint ty = tiisg / D4; // 0/1: token parity within a warp pass

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    if (tiisg < D4) {
        so4[tiisg] = float4(0.0f);
    }
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    // Each SG walks KV tiles: ic0 = sgitg, sgitg+nsg, ...
    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        // Q·K, two positions per warp pass: half-warp `ty` owns token
        // ic+cc; 16-lane xor reduce leaves the full dot in every lane
        // of that half; lane tx==0 publishes it.
        for (uint cc = ty; cc < chunk; cc += 2u) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float p = dot(sq4[tx], float4(k4[tx]));
            p += simd_shuffle_xor(p, 8u);
            p += simd_shuffle_xor(p, 4u);
            p += simd_shuffle_xor(p, 2u);
            p += simd_shuffle_xor(p, 1u);
            if (tx == 0u) {
                float sc = p * scale;
                if (softcap > 0.0f) {
                    sc = softcap * tanh(sc / softcap);
                }
                ss[cc] = sc;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax over this tile (one score per lane).
        float s_lane = (tiisg < chunk) ? ss[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        if (tiisg < D4) {
            so4[tiisg] *= ms;
        }
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O += P · V, two positions per warp pass; merge the two token
        // halves with an xor-16 shuffle before accumulating.
        float4 lo = float4(0.0f);
        for (uint cc = ty; cc < chunk; cc += 2u) {
            device const half4* v4 =
                (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            lo += float4(v4[tx]) * ss[cc];
        }
        lo += simd_shuffle_xor(lo, 16u);
        if (ty == 0u) {
            so4[tx] += lo;
        }
    }

    // Publish S,M for cross-SG reduce (reuse ss[0], ss[1]).
    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG online-softmax merge.
    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            if (tiisg < D4) {
                so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u && tiisg < D4) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + h * D);
        out4[tiisg] = so0[tiisg] * inv;
    }
}
"#;

/// FA-vec decode for **head_dim=96** (Phi-3-mini). D4=24 float4 slices —
/// lanes `tiisg < 24` own Q/K/V float4 work; remaining warp lanes contribute
/// zeros to the simd_sum score reduce so the tile/merge structure stays
/// identical to the d=128 kernel.
const GQA_DECODE_FA_VEC_D96_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_fa_vec_d96(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = 96u;
    constexpr uint D4 = 24u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    if (tiisg < D4) {
        so4[tiisg] = float4(0.0f);
    }
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = (tiisg < D4) ? dot(sq4[tiisg], float4(k4[tiisg])) : 0.0f;
            float sc = simd_sum(partial) * scale;
            if (softcap > 0.0f) {
                sc = softcap * tanh(sc / softcap);
            }
            scores[cc] = sc;
        }

        float s_lane = (tiisg < chunk) ? scores[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        if (tiisg < D4) {
            so4[tiisg] *= ms;
        }
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo = float4(0.0f);
        if (tiisg < D4) {
            for (uint cc = 0; cc < chunk; cc++) {
                device const half4* v4 =
                    (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
                lo += float4(v4[tiisg]) * ss[cc];
            }
            so4[tiisg] += lo;
        }
    }

    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            if (tiisg < D4) {
                so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u && tiisg < D4) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + h * D);
        out4[tiisg] = so0[tiisg] * inv;
    }
}
"#;

/// FA-vec decode for **head_dim=256** (Gemma-3). D4=64 float4 slices —
/// each warp lane owns two slices (`tiisg` and `tiisg+32`) so the simd
/// reduce still covers the full head.
const GQA_DECODE_FA_VEC_D256_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_fa_vec_d256(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = 256u;
    constexpr uint D4 = 64u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    so4[tiisg] = float4(0.0f);
    so4[tiisg + NW] = float4(0.0f);
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = 0.0f;
            for (uint i = tiisg; i < D4; i += NW) {
                partial += dot(sq4[i], float4(k4[i]));
            }
            float sc = simd_sum(partial) * scale;
            if (softcap > 0.0f) {
                sc = softcap * tanh(sc / softcap);
            }
            scores[cc] = sc;
        }

        float s_lane = (tiisg < chunk) ? scores[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        so4[tiisg] *= ms;
        so4[tiisg + NW] *= ms;
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo0 = float4(0.0f);
        float4 lo1 = float4(0.0f);
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* v4 =
                (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            lo0 += float4(v4[tiisg]) * ss[cc];
            lo1 += float4(v4[tiisg + NW]) * ss[cc];
        }
        so4[tiisg] += lo0;
        so4[tiisg + NW] += lo1;
    }

    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            so0[tiisg + NW] = so0[tiisg + NW] * a0 + so1[tiisg + NW] * a1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + h * D);
        out4[tiisg] = so0[tiisg] * inv;
        out4[tiisg + NW] = so0[tiisg + NW] * inv;
    }
}
"#;

const GQA_DECODE_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Online-softmax GQA decode (B=1, full causal).
// One threadgroup per query head. V accumulators stay register-local
// through the seq scan (float4 dots/axpy when head_dim % 4 == 0); each
// simdgroup reduces via simd_shuffle_xor, then N_SG partials merge in a
// compact TG footprint O(nsg * head_dim) instead of O(tg * head_dim).
// `tg` must be a multiple of 32 (host enforces). head_dim <= 256.

inline float online_rescale(float m_old, float m_new) {
    // exp(m_old - m_new); 0 when m_old is -inf (empty / inactive partial).
    return (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
}

kernel void gqa_decode(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    if (h >= n_heads || seq_len == 0u) return;

    constexpr uint MAX_D = 256u;
    constexpr uint NW = 32u;
    const uint nsg = tg / NW;
    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;

    // Compact layout after per-SG reduce: m[nsg] | s[nsg] | acc[nsg * head_dim]
    threadgroup float* m_sh = shared;
    threadgroup float* s_sh = shared + nsg;
    threadgroup float* acc_base = shared + 2u * nsg;

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(head_dim));
    device const float* q_h = q + h * head_dim;

    float m = -INFINITY;
    float s = 0.0f;
    float my_acc[MAX_D];
    for (uint d = 0; d < head_dim; d++) {
        my_acc[d] = 0.0f;
    }

    const bool vec4 = (head_dim & 3u) == 0u;
    // kv_start > 0 = sliding-window attention: only positions
    // [kv_start, seq_len) are visible (Gemma-style SWA).
    for (uint t = kv_start + tid; t < seq_len; t += tg) {
        device const half* k_t =
            k_cache + (t * n_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        if (vec4) {
            float4 acc4 = float4(0.0f);
            for (uint d = 0; d < head_dim; d += 4u) {
                acc4 += float4(
                    q_h[d], q_h[d + 1u], q_h[d + 2u], q_h[d + 3u])
                    * float4(
                        float(k_t[d]), float(k_t[d + 1u]), float(k_t[d + 2u]), float(k_t[d + 3u]));
            }
            dot = acc4[0] + acc4[1] + acc4[2] + acc4[3];
        } else {
            for (uint d = 0; d < head_dim; d++) {
                dot += q_h[d] * float(k_t[d]);
            }
        }
        float score = dot * scale;
        if (softcap > 0.0f) {
            score = softcap * tanh(score / softcap);
        }
        float m2 = max(m, score);
        float a = online_rescale(m, m2);
        float b = exp(score - m2);
        s = s * a + b;
        device const half* v_t =
            v_cache + (t * n_kv_heads + kv_h) * head_dim;
        if (vec4) {
            for (uint d = 0; d < head_dim; d += 4u) {
                my_acc[d] = my_acc[d] * a + b * float(v_t[d]);
                my_acc[d + 1u] = my_acc[d + 1u] * a + b * float(v_t[d + 1u]);
                my_acc[d + 2u] = my_acc[d + 2u] * a + b * float(v_t[d + 2u]);
                my_acc[d + 3u] = my_acc[d + 3u] * a + b * float(v_t[d + 3u]);
            }
        } else {
            for (uint d = 0; d < head_dim; d++) {
                my_acc[d] = my_acc[d] * a + b * float(v_t[d]);
            }
        }
        m = m2;
    }

    // Intra-simdgroup butterfly reduce (no TG traffic).
    for (ushort offset = ushort(NW >> 1); offset > 0u; offset >>= 1) {
        float m_o = simd_shuffle_xor(m, offset);
        float s_o = simd_shuffle_xor(s, offset);
        float m_new = max(m, m_o);
        float a = online_rescale(m, m_new);
        float a_o = online_rescale(m_o, m_new);
        s = s * a + s_o * a_o;
        for (uint d = 0; d < head_dim; d++) {
            float ao = simd_shuffle_xor(my_acc[d], offset);
            my_acc[d] = my_acc[d] * a + ao * a_o;
        }
        m = m_new;
    }

    if (tiisg == 0u) {
        m_sh[sgitg] = m;
        s_sh[sgitg] = s;
        threadgroup float* slot = acc_base + sgitg * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            slot[d] = my_acc[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG tree reduce over nsg << tg partials.
    for (uint stride = nsg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            uint other = tid + stride;
            float m1 = m_sh[tid];
            float m2 = m_sh[other];
            float s1 = s_sh[tid];
            float s2 = s_sh[other];
            float m_new = max(m1, m2);
            float a1 = online_rescale(m1, m_new);
            float a2 = online_rescale(m2, m_new);
            m_sh[tid] = m_new;
            s_sh[tid] = s1 * a1 + s2 * a2;
            threadgroup float* acc1 = acc_base + tid * head_dim;
            threadgroup float* acc2 = acc_base + other * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                acc1[d] = acc1[d] * a1 + acc2[d] * a2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_s = 1.0f / s_sh[0];
    threadgroup float* acc0 = acc_base;
    for (uint d = tid; d < head_dim; d += tg) {
        out[h * head_dim + d] = acc0[d] * inv_s;
    }
}
"#;

/// FA-vec multi-query causal prefill for head_dim **64 / 96 / 128**, f16
/// KV, C=32. One TG per `(head, query_token)`; same tile/merge as decode
/// FA-vec, but each query `qi` attends only over `[0 ..= kv_prefix_len + qi]`.
///
/// One body, three entry points. Only 128 existed before, which meant
/// every d=64 model (SmolLM2, TinyLlama, Llama-3.2-1B, Qwen2.5-0.5B) ran
/// the legacy `gqa_prefill` kernel instead. That kernel keeps a
/// *per-thread* accumulator in threadgroup memory — `tg * head_dim`
/// floats, ~27 KB at 108 threads — which caps occupancy at roughly one
/// threadgroup per core. This one needs `D + nsg*(C+D)` floats, 3.3 KB at
/// d=64. Profiling SmolLM2 metal `pp512` put 62% of prefill inside that
/// legacy kernel's `waitUntilCompleted`.
///
/// `D4 = D/4` is how many float4 lanes carry the query and the output.
/// At d=128 that is exactly the 32-lane simdgroup; at d=64 and d=96 it is
/// fewer, so the lanes above `D4` sit out the dot product and the
/// accumulator updates. They still participate in `simd_sum`/`simd_max`,
/// which is what makes the masking safe rather than merely lucky.
macro_rules! gqa_prefill_fa_vec_src {
    ($name:literal, $d:literal) => {
        concat!(
            r#"
#include <metal_stdlib>
using namespace metal;

kernel void "#,
            $name,
            r#"(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_q [[buffer(7)]],
    constant uint& kv_prefix_len [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = "#,
            $d,
            r#"u;
    constexpr uint D4 = D / 4u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    constexpr uint SG_F = C + D;

    uint h = tgpig.x;
    uint qi = tgpig.y;
    if (h >= n_heads || qi >= n_q || head_dim != D) return;

    uint causal_len = kv_prefix_len + qi + 1u;
    if (causal_len == 0u) return;

    uint tid = tid_tg.x;
    uint tg = tg_size.x;
    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;
    const bool own = tiisg < D4;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + (qi * n_heads + h) * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    if (own) {
        so4[tiisg] = float4(0.0f);
    }
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = ic0 * C;
        if (ic >= causal_len) break;
        uint chunk = min(C, causal_len - ic);

        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = own ? dot(sq4[tiisg], float4(k4[tiisg])) : 0.0f;
            float sc = simd_sum(partial) * scale;
            if (softcap > 0.0f) {
                sc = softcap * tanh(sc / softcap);
            }
            scores[cc] = sc;
        }

        float s_lane = (tiisg < chunk) ? scores[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        if (own) {
            so4[tiisg] *= ms;
        }
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (own) {
            float4 lo = float4(0.0f);
            for (uint cc = 0; cc < chunk; cc++) {
                device const half4* v4 =
                    (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
                lo += float4(v4[tiisg]) * ss[cc];
            }
            so4[tiisg] += lo;
        }
    }

    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            if (own) {
                so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u && own) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + (qi * n_heads + h) * D);
        out4[tiisg] = so0[tiisg] * inv;
    }
}
"#
        )
    };
}

const GQA_PREFILL_FA_VEC_KERNEL_SRC: &str = gqa_prefill_fa_vec_src!("gqa_prefill_fa_vec", "128");
const GQA_PREFILL_FA_VEC_D64_KERNEL_SRC: &str =
    gqa_prefill_fa_vec_src!("gqa_prefill_fa_vec_d64", "64");
const GQA_PREFILL_FA_VEC_D96_KERNEL_SRC: &str =
    gqa_prefill_fa_vec_src!("gqa_prefill_fa_vec_d96", "96");

/// Prefill FA with **4 queries per TG** (llama `OP_FLASH_ATTN_EXT_NQPSG`-lite).
/// Shares K/V tile traffic across queries. d=64 only (SmolLM2 / Tiny / 1B).
const GQA_PREFILL_FA_NQ4_D64_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_prefill_fa_nq4_d64(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_q [[buffer(7)]],
    constant uint& kv_prefix_len [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = 64u;
    constexpr uint D4 = 16u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    constexpr uint QN = 4u;
    // Per-SG scratch: C scores + QN*D output floats
    constexpr uint SG_F = C + QN * D;

    uint h = tgpig.x;
    uint qi0 = tgpig.y * QN;
    if (h >= n_heads || qi0 >= n_q || head_dim != D) return;

    uint tid = tid_tg.x;
    uint tg = tg_size.x;
    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;
    const bool own = tiisg < D4;

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    // shared: [QN * D] queries, then NSG * SG_F scratch
    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + QN * D + sgitg * SG_F;

    uint n_local = min(QN, n_q - qi0);
    for (uint j = 0; j < n_local; j++) {
        device const float4* q4 =
            (device const float4*)(q + ((qi0 + j) * n_heads + h) * D);
        for (uint i = tid; i < D4; i += tg) {
            sq4[j * D4 + i] = q4[i];
        }
    }
    // Zero unused query slots' shared Q (keeps masked scores clean).
    for (uint j = n_local; j < QN; j++) {
        for (uint i = tid; i < D4; i += tg) {
            sq4[j * D4 + i] = float4(0.0f);
        }
    }
    if (own) {
        for (uint j = 0; j < QN; j++) {
            ((threadgroup float4*)(ss + C + j * D))[tiisg] = float4(0.0f);
        }
    }
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[QN];
    float M[QN];
    uint causal[QN];
    for (uint j = 0; j < QN; j++) {
        S[j] = 0.0f;
        M[j] = -INFINITY;
        causal[j] = (j < n_local) ? (kv_prefix_len + qi0 + j + 1u) : 0u;
    }
    uint max_causal = 0u;
    for (uint j = 0; j < n_local; j++) {
        max_causal = max(max_causal, causal[j]);
    }
    if (max_causal == 0u) return;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = ic0 * C;
        if (ic >= max_causal) break;
        uint chunk = min(C, max_causal - ic);

        // scores[j][cc] staged in registers then written to ss for V pass
        float scores[QN][C];
        for (uint j = 0; j < QN; j++) {
            for (uint cc = 0; cc < C; cc++) {
                scores[j][cc] = -INFINITY;
            }
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            for (uint j = 0; j < n_local; j++) {
                if (ic + cc >= causal[j]) continue;
                float partial = own ? dot(sq4[j * D4 + tiisg], float4(k4[tiisg])) : 0.0f;
                float sc = simd_sum(partial) * scale;
                if (softcap > 0.0f) {
                    sc = softcap * tanh(sc / softcap);
                }
                scores[j][cc] = sc;
            }
        }

        for (uint j = 0; j < n_local; j++) {
            float s_lane = (tiisg < chunk) ? scores[j][tiisg] : -INFINITY;
            float M2 = simd_max(max(M[j], s_lane));
            float ms = (M[j] == -INFINITY) ? 0.0f : exp(M[j] - M2);
            float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
            S[j] = S[j] * ms + simd_sum(vs);
            // stash per-query score weights into ss[j*C + tiisg] — reuse ss
            // carefully: only C floats free at start of ss. Use so region
            // temporarily? Keep scores in thread registers for V: rewrite
            // scores[j][cc] as already-exp'd vs relative to M2.
            float inv_broadcast = 0.0f; // silence
            (void)inv_broadcast;
            for (uint cc = 0; cc < chunk; cc++) {
                float sc = scores[j][cc];
                scores[j][cc] = (sc == -INFINITY) ? 0.0f : exp(sc - M2);
            }
            if (own) {
                threadgroup float4* so4 = (threadgroup float4*)(ss + C + j * D);
                so4[tiisg] *= ms;
            }
            M[j] = M2;

            if (own) {
                threadgroup float4* so4 = (threadgroup float4*)(ss + C + j * D);
                float4 lo = float4(0.0f);
                for (uint cc = 0; cc < chunk; cc++) {
                    device const half4* v4 =
                        (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
                    lo += float4(v4[tiisg]) * scores[j][cc];
                }
                so4[tiisg] += lo;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce across simdgroups: store S/M into ss[0]/ss[1] per query via
    // a compact QN*2 header after scores region is free.
    // Layout: ss[0..QN) = S, ss[QN..2*QN) = M for this SG (tiisg==0 writes).
    if (tiisg == 0u) {
        for (uint j = 0; j < QN; j++) {
            ss[j] = S[j];
            ss[QN + j] = M[j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + QN * D + sgitg * SG_F;
            threadgroup float* ss1 = shared + QN * D + (sgitg + r) * SG_F;
            for (uint j = 0; j < n_local; j++) {
                float S0 = ss0[j];
                float S1 = ss1[j];
                float M0 = ss0[QN + j];
                float M1 = ss1[QN + j];
                float Mn = max(M0, M1);
                float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
                float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
                if (tiisg == 0u) {
                    ss0[j] = S0 * a0 + S1 * a1;
                    ss0[QN + j] = Mn;
                }
                if (own) {
                    threadgroup float4* so0 = (threadgroup float4*)(ss0 + C + j * D);
                    threadgroup float4* so1 = (threadgroup float4*)(ss1 + C + j * D);
                    so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u && own) {
        threadgroup float* ss0 = shared + QN * D;
        for (uint j = 0; j < n_local; j++) {
            float inv = (ss0[j] == 0.0f) ? 0.0f : (1.0f / ss0[j]);
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C + j * D);
            device float4* out4 =
                (device float4*)(out + ((qi0 + j) * n_heads + h) * D);
            out4[tiisg] = so0[tiisg] * inv;
        }
    }
}
"#;

/// llama `kernel_flash_attn_ext` tiling for d=64 (QN=8, C=64, NSG=4) with a
/// **scalar** `dot` + `simd_sum` score phase and a scalar P·V gather — despite
/// the name there is no simdgroup MMA in here.
///
/// Superseded by [`GQA_PREFILL_FA_EXT_MMA_D64_KERNEL_SRC`], which is now the
/// default. Kept reachable via `FERROX_METAL_FA_MMA=0`: it is the reference the
/// MMA kernel is diffed against in `gqa_prefill_fa_ext_mma_matches_scalar_d64`.
const GQA_PREFILL_FA_EXT_D64_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_prefill_fa_ext_d64(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_q [[buffer(7)]],
    constant uint& kv_prefix_len [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = 64u;
    constexpr uint D4 = 16u;
    constexpr uint D8 = 8u;
    constexpr uint QN = 8u;
    constexpr uint C = 64u;
    constexpr uint NW = 32u;
    constexpr uint NSG = 4u;
    constexpr uint NQ = QN / NSG;
    constexpr uint SH = 2u * C;
    constexpr uint NC = (C / 8u) / NSG;
    constexpr uint PV = 64u;
    constexpr uint NO = 8u / NSG;

    const uint h = tgpig.x;
    const uint qi0 = tgpig.y * QN;
    if (h >= n_heads || qi0 >= n_q || head_dim != D) return;

    const uint group_size = n_heads / max(n_kv_heads, 1u);
    const uint kv_h = h / max(group_size, 1u);
    const uint kv_stride = n_kv_heads * D;
    const float scale = 1.0f / sqrt(float(D));
    const uint n_local = min(QN, n_q - qi0);
    const bool own = tiisg < D4;

    // sq[QN,D] | so[QN,D] | ss[QN,SH]
    threadgroup float* sq = shared;
    threadgroup float* so = shared + QN * D;
    threadgroup float* ss = shared + 2u * QN * D;

    for (uint j = 0u; j < QN; j++) {
        const uint gqi = qi0 + j;
        threadgroup float4* sq4 = (threadgroup float4*)(sq + j * D);
        if (gqi < n_q) {
            device const float4* q4 =
                (device const float4*)(q + (gqi * n_heads + h) * D);
            for (uint i = tiisg; i < D4; i += NW) sq4[i] = q4[i];
        } else {
            for (uint i = tiisg; i < D4; i += NW) sq4[i] = float4(0.0f);
        }
        if (own) {
            threadgroup float4* so4 = (threadgroup float4*)(so + j * D);
            so4[tiisg] = float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (uint jj = 0u; jj < NQ; jj++) {
        S[jj] = 0.0f;
        M[jj] = -INFINITY;
    }

    uint max_causal = 0u;
    for (uint j = 0u; j < n_local; j++) {
        max_causal = max(max_causal, kv_prefix_len + qi0 + j + 1u);
    }
    if (max_causal == 0u) return;

    for (uint ic0 = 0u; ; ic0++) {
        const uint ic = ic0 * C;
        if (ic >= max_causal) break;
        const uint chunk = min(C, max_causal - ic);

        // Q·Kᵀ + scale/softcap/causal mask. Every ss slot is written exactly
        // once, from a register, by a single thread.
        //
        // Scale/softcap/mask used to live in a *separate* loop after the
        // barrier, run redundantly by all NSG simdgroups over the same
        // `cc = tiisg; cc < chunk; cc += NW` slots. That made each ss slot a
        // read-modify-write with no barrier between the four readers and the
        // four writers, so a simdgroup could load a score another had already
        // transformed and transform it a second time —
        //   softcap*tanh(softcap*tanh(x*scale/softcap)*scale/softcap)
        // instead of softcap*tanh(x*scale/softcap). Whether a given slot got
        // hit depended on simdgroup skew, which is why the corruption was
        // scattered over some (query, key) pairs and not others.
        //
        // The single-writer property is what the `sgitg == 0u` guard buys, and
        // it is now enforced for the whole score pipeline rather than for the
        // dot product alone. A future attempt to widen this phase across all
        // four simdgroups must keep it: partition `cc` so the simdgroups touch
        // disjoint ss slots, and keep the mask folded in here.
        if (sgitg == 0u) {
            for (uint cc = 0u; cc < chunk; cc++) {
                device const half4* k4 = (device const half4*)(k_cache
                    + ((ic + cc) * n_kv_heads + kv_h) * D);
                for (uint j = 0u; j < n_local; j++) {
                    threadgroup float4* sq4j = (threadgroup float4*)(sq + j * D);
                    float partial = own ? dot(sq4j[tiisg], float4(k4[tiisg])) : 0.0f;
                    float sc = simd_sum(partial);
                    if (tiisg == 0u) {
                        const uint clen = kv_prefix_len + qi0 + j + 1u;
                        if (ic + cc >= clen) {
                            sc = -INFINITY;
                        } else {
                            sc *= scale;
                            if (softcap > 0.0f) sc = softcap * tanh(sc / softcap);
                        }
                        ss[j * SH + cc] = sc;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax (llama float2 pairs over C=64).
        for (uint jj = 0u; jj < NQ; jj++) {
            const uint j = jj * NSG + sgitg;
            if (j >= n_local) continue;
            threadgroup float2* ss2 = (threadgroup float2*)(ss + j * SH);
            float2 s2 = ss2[tiisg];
            if (2u * tiisg + 1u >= chunk) s2[1] = -INFINITY;
            if (2u * tiisg >= chunk) s2[0] = -INFINITY;

            const float m = M[jj];
            M[jj] = simd_max(max(m, max(s2[0], s2[1])));
            const float ms = (m == -INFINITY) ? 0.0f : exp(m - M[jj]);
            const float2 vs2 = float2(
                (s2[0] == -INFINITY) ? 0.0f : exp(s2[0] - M[jj]),
                (s2[1] == -INFINITY) ? 0.0f : exp(s2[1] - M[jj])
            );
            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[tiisg] = vs2;

            if (own) {
                threadgroup float4* so4 = (threadgroup float4*)(so + j * D);
                so4[tiisg] *= ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P·V: scalar gather (V staging + MMA layout still WIP; Q·Kᵀ MMA is the win).
        for (uint jj = 0u; jj < NQ; jj++) {
            const uint j = jj * NSG + sgitg;
            if (j >= n_local || !own) continue;
            threadgroup float4* so4 = (threadgroup float4*)(so + j * D);
            float4 lo = float4(0.0f);
            for (uint cc = 0u; cc < chunk; cc++) {
                device const half4* v4 = (device const half4*)(v_cache
                    + (ic + cc) * kv_stride + kv_h * D);
                lo += float4(v4[tiisg]) * ss[j * SH + cc];
            }
            so4[tiisg] += lo;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Each SG writes its NQ queries (no cross-SG KV reduce — llama layout).
    for (uint jj = 0u; jj < NQ; jj++) {
        const uint j = jj * NSG + sgitg;
        if (j >= n_local || !own) continue;
        const float inv = (S[jj] == 0.0f) ? 0.0f : (1.0f / S[jj]);
        device float4* out4 =
            (device float4*)(out + ((qi0 + j) * n_heads + h) * D);
        threadgroup float4* so4 = (threadgroup float4*)(so + j * D);
        out4[tiisg] = so4[tiisg] * inv;
    }
}
"#;

/// llama `kernel_flash_attn_ext` for d=64, ported with **real 8×8 simdgroup
/// MMA** on both Q·Kᵀ and P·V (`ggml-metal.metal`, `kernel_flash_attn_ext_impl`
/// — Q·Kᵀ at :6693-6729, P·V at :6841-6910; d=64 instantiation at :7126).
///
/// Shape is llama's: QN=8 queries and C=64 keys per threadgroup, NSG=4
/// simdgroups, `ss[QN][2C]` scores, `so[QN][64]` accumulator. The predecessor
/// [`GQA_PREFILL_FA_EXT_D64_KERNEL_SRC`] had the same *tiling* but computed
/// scores with one `dot`+`simd_sum` per (query,key) inside a single simdgroup
/// with 16 of 32 lanes live — 16 of 128 threads doing arithmetic. Here all
/// four simdgroups run MMA over disjoint 8-key column blocks of `ss`, and P·V
/// runs over disjoint 8-wide column blocks of `so`.
///
/// Two things differ from llama and are load-bearing:
///
/// 1. **Where scale/softcap/mask are applied.** llama has an explicit mask
///    tensor and folds `scale`/`softcap`/`mask` into the online-softmax loop.
///    The predecessor kernel had to fold them into the *score* loop instead,
///    because an earlier version ran them as a separate all-simdgroups pass
///    over shared `ss` slots: a read-modify-write with four readers and four
///    writers and no barrier between them, which double-softcapped whichever
///    slots lost the race (see the comment on the scalar kernel). MMA scores
///    are written by `simdgroup_store`, so that fold is no longer possible —
///    they move back into the softmax loop, which is safe for the reason
///    llama's is: there each `ss` slot has exactly **one** owner
///    (`j = jj*NSG + sgitg` picks disjoint rows per simdgroup, `tiisg` picks
///    disjoint `float2` columns per lane), so the RMW has a single writer.
///    Any future edit must preserve that ownership, not the `sgitg == 0` guard
///    it replaces.
///
/// 2. **Tail handling.** `simdgroup_load` reads a full 8 rows of K/V, so the
///    last partial group of a cache whose length is not a multiple of 8 would
///    read past the end of the buffer. llama pre-pads its KV cache; Ferrox's
///    is exactly `kv_prefix_len + n_q` rows, so instead the ≤7 leftover rows
///    are staged once into a zero-filled `kpad`/`vpad` tile in threadgroup
///    memory and the MMA reads that. Groups entirely past the cache are
///    skipped: the causal mask forces their `ss` columns to `-INFINITY`
///    (`ic + cc >= kv_valid >= clen`), hence P = 0, hence no P·V contribution.
///
/// Parameterised over the head dim: every shape constant below is derived from
/// `D`, so the same body serves d=64 and d=128. Two limits bound it. `own`
/// assigns one `float4` of the output row per lane, so `D/4 <= 32`, i.e.
/// `D <= 128` — d=256 (Gemma-3) needs a lane loop and is not instantiated here.
/// And `D % 16 == 0`, because the Q·Kᵀ loop walks the head 16 columns at a
/// time as a pair of 8×8 MMAs.
macro_rules! gqa_prefill_fa_ext_mma_src {
    ($name:literal, $d:literal) => {
        concat!(
            r#"
#include <metal_stdlib>
using namespace metal;

kernel void "#,
            $name,
            r#"(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_q [[buffer(7)]],
    constant uint& kv_prefix_len [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = "#,
            $d,
            r#"u;          // DK == DV == PV
    constexpr uint D4 = D / 4u;      // output float4 columns == owning lanes
    constexpr uint D8 = D / 8u;      // D / 8: 8-wide MMA steps along the head
    constexpr uint QN = 8u;          // queries per threadgroup
    constexpr uint C = 64u;          // keys per threadgroup chunk
    constexpr uint NW = 32u;
    constexpr uint NSG = 4u;
    constexpr uint NQ = QN / NSG;    // softmax rows per simdgroup
    constexpr uint SH = 2u * C;      // ss row stride
    constexpr uint CB = C / 8u;      // 8-key blocks per chunk
    constexpr uint NC = CB / NSG;    // score blocks per simdgroup
    constexpr uint NO = D8 / NSG;    // output column blocks per simdgroup

    const uint h = tgpig.x;
    const uint qi0 = tgpig.y * QN;
    if (h >= n_heads || qi0 >= n_q || head_dim != D) return;

    const uint group_size = n_heads / max(n_kv_heads, 1u);
    const uint kv_h = h / max(group_size, 1u);
    const uint kv_stride = n_kv_heads * D;
    const float scale = 1.0f / sqrt(float(D));
    const uint n_local = min(QN, n_q - qi0);
    const bool own = tiisg < D4;

    // sq[QN,D] f32 | so[QN,D] f32 | ss[QN,SH] f32 | kpad[8,D] f16 | vpad[8,D] f16
    threadgroup float* sq = shared;
    threadgroup float* so = shared + QN * D;
    threadgroup float* ss = shared + 2u * QN * D;
    threadgroup half* kpad = (threadgroup half*)(shared + 2u * QN * D + QN * SH);
    threadgroup half* vpad = kpad + 8u * D;

    // Rows of K/V that physically exist, and the 8-row-aligned prefix of them
    // that `simdgroup_load` may read straight out of device memory.
    const uint kv_valid = kv_prefix_len + n_q;
    const uint kv_full = (kv_valid / 8u) * 8u;
    const uint kv_rem = kv_valid - kv_full;

    for (uint j = 0u; j < QN; j++) {
        const uint gqi = qi0 + j;
        threadgroup float4* sq4 = (threadgroup float4*)(sq + j * D);
        if (gqi < n_q) {
            device const float4* q4 =
                (device const float4*)(q + (gqi * n_heads + h) * D);
            for (uint i = tiisg; i < D4; i += NW) sq4[i] = q4[i];
        } else {
            // Zero query rows: their MMA scores are 0, the softmax skips them,
            // and nothing reads their `so` rows back out.
            for (uint i = tiisg; i < D4; i += NW) sq4[i] = float4(0.0f);
        }
        if (own) {
            threadgroup float4* so4 = (threadgroup float4*)(so + j * D);
            so4[tiisg] = float4(0.0f);
        }
    }

    if (kv_rem > 0u) {
        const uint tid = uint(sgitg) * NW + uint(tiisg);
        for (uint idx = tid; idx < 8u * D; idx += NSG * NW) {
            const uint r = idx / D;
            const uint c = idx - r * D;
            half kk = (half)0.0f;
            half vv = (half)0.0f;
            if (r < kv_rem) {
                const uint base = (kv_full + r) * kv_stride + kv_h * D + c;
                kk = k_cache[base];
                vv = v_cache[base];
            }
            kpad[idx] = kk;
            vpad[idx] = vv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (uint jj = 0u; jj < NQ; jj++) {
        S[jj] = 0.0f;
        M[jj] = -INFINITY;
    }

    uint max_causal = 0u;
    for (uint j = 0u; j < n_local; j++) {
        max_causal = max(max_causal, kv_prefix_len + qi0 + j + 1u);
    }
    if (max_causal == 0u) return;

    for (uint ic0 = 0u; ; ic0++) {
        const uint ic = ic0 * C;
        if (ic >= max_causal) break;

        // Q·Kᵀ — 8x8 MMA. Simdgroup `sgitg` owns key blocks
        // {sgitg, sgitg + NSG}, i.e. ss columns [8g, 8g+8): disjoint, so
        // `simdgroup_store` is a single writer per slot.
        for (uint cb = 0u; cb < NC; cb++) {
            const uint g = uint(sgitg) + cb * NSG;
            const uint key0 = ic + 8u * g;
            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
            if (key0 + 8u <= kv_full) {
                device const half* pk = k_cache + key0 * kv_stride + kv_h * D;
                for (uint i = 0u; i < D8 / 2u; i++) {
                    simdgroup_float8x8 mq0, mq1;
                    simdgroup_half8x8 mk0, mk1;
                    simdgroup_load(mq0, sq + 16u * i, D);
                    simdgroup_load(mq1, sq + 16u * i + 8u, D);
                    // transpose: [key,dim] -> [dim,key]
                    simdgroup_load(mk0, pk + 16u * i, kv_stride, 0, true);
                    simdgroup_load(mk1, pk + 16u * i + 8u, kv_stride, 0, true);
                    simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                    simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
                }
            } else if (key0 == kv_full && kv_rem > 0u) {
                threadgroup const half* pk = kpad;
                for (uint i = 0u; i < D8 / 2u; i++) {
                    simdgroup_float8x8 mq0, mq1;
                    simdgroup_half8x8 mk0, mk1;
                    simdgroup_load(mq0, sq + 16u * i, D);
                    simdgroup_load(mq1, sq + 16u * i + 8u, D);
                    simdgroup_load(mk0, pk + 16u * i, D, 0, true);
                    simdgroup_load(mk1, pk + 16u * i + 8u, D, 0, true);
                    simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                    simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
                }
            } else {
                // Entirely past the cache: every column here is masked below.
                continue;
            }
            simdgroup_store(mqk, ss + 8u * g, SH, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax. Single-writer per ss slot: row j is owned by one
        // simdgroup, column pair `tiisg` by one lane. scale/softcap/causal
        // mask are applied here, on a value read once and written once.
        for (uint jj = 0u; jj < NQ; jj++) {
            const uint j = jj * NSG + sgitg;
            if (j >= n_local) continue;
            threadgroup float2* ss2 = (threadgroup float2*)(ss + j * SH);
            float2 s2 = ss2[tiisg] * scale;
            if (softcap > 0.0f) s2 = softcap * tanh(s2 / softcap);
            const uint clen = kv_prefix_len + qi0 + j + 1u;
            const uint c0 = ic + 2u * uint(tiisg);
            // `cc >= chunk` is subsumed: it implies ic+cc >= max_causal >= clen.
            if (c0 >= clen) s2[0] = -INFINITY;
            if (c0 + 1u >= clen) s2[1] = -INFINITY;

            const float m = M[jj];
            M[jj] = simd_max(max(m, max(s2[0], s2[1])));
            const float ms = (m == -INFINITY) ? 0.0f : exp(m - M[jj]);
            const float2 vs2 = float2(
                (s2[0] == -INFINITY) ? 0.0f : exp(s2[0] - M[jj]),
                (s2[1] == -INFINITY) ? 0.0f : exp(s2[1] - M[jj])
            );
            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[tiisg] = vs2;

            if (own) {
                threadgroup float4* so4 = (threadgroup float4*)(so + j * D);
                so4[tiisg] *= ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += P·V — 8x8 MMA. Simdgroup `sgitg` owns output columns
        // {8*sgitg + 8*NSG*ii}, disjoint across simdgroups; every simdgroup
        // walks all CB key blocks.
        {
            simdgroup_float8x8 lo[NO];
            for (uint ii = 0u; ii < NO; ii++) {
                simdgroup_load(lo[ii], so + 8u * sgitg + 8u * NSG * ii, D, 0, false);
            }
            for (uint cc = 0u; cc < CB; cc++) {
                const uint key0 = ic + 8u * cc;
                const bool fullblk = (key0 + 8u <= kv_full);
                const bool padblk = (key0 == kv_full) && (kv_rem > 0u);
                if (!fullblk && !padblk) continue;
                simdgroup_float8x8 vs;
                simdgroup_load(vs, ss + 8u * cc, SH, 0, false);
                if (fullblk) {
                    device const half* pv =
                        v_cache + key0 * kv_stride + kv_h * D + 8u * sgitg;
                    for (uint ii = 0u; ii < NO; ii++) {
                        simdgroup_half8x8 mv;
                        simdgroup_load(mv, pv + 8u * NSG * ii, kv_stride, 0, false);
                        simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                    }
                } else {
                    threadgroup const half* pv = vpad + 8u * sgitg;
                    for (uint ii = 0u; ii < NO; ii++) {
                        simdgroup_half8x8 mv;
                        simdgroup_load(mv, pv + 8u * NSG * ii, D, 0, false);
                        simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                    }
                }
            }
            for (uint ii = 0u; ii < NO; ii++) {
                simdgroup_store(lo[ii], so + 8u * sgitg + 8u * NSG * ii, D, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Each SG writes its NQ queries (no cross-SG KV reduce — llama layout).
    for (uint jj = 0u; jj < NQ; jj++) {
        const uint j = jj * NSG + sgitg;
        if (j >= n_local || !own) continue;
        const float inv = (S[jj] == 0.0f) ? 0.0f : (1.0f / S[jj]);
        device float4* out4 =
            (device float4*)(out + ((qi0 + j) * n_heads + h) * D);
        threadgroup float4* so4 = (threadgroup float4*)(so + j * D);
        out4[tiisg] = so4[tiisg] * inv;
    }
}
"#
        )
    };
}

const GQA_PREFILL_FA_EXT_MMA_D64_KERNEL_SRC: &str =
    gqa_prefill_fa_ext_mma_src!("gqa_prefill_fa_ext_mma_d64", "64");
/// Qwen3-0.6B / Phi-4-mini / Mistral shape: head_dim 128. There is no scalar
/// `fa_ext` predecessor at this width (that kernel is d=64-only), so the A/B
/// reference for both correctness and timing is `gqa_prefill_fa_vec`.
const GQA_PREFILL_FA_EXT_MMA_D128_KERNEL_SRC: &str =
    gqa_prefill_fa_ext_mma_src!("gqa_prefill_fa_ext_mma_d128", "128");

const GQA_PREFILL_FA_VEC_D256_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_prefill_fa_vec_d256(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_q [[buffer(7)]],
    constant uint& kv_prefix_len [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = 256u;
    constexpr uint D4 = 64u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    constexpr uint SG_F = C + D;

    uint h = tgpig.x;
    uint qi = tgpig.y;
    if (h >= n_heads || qi >= n_q || head_dim != D) return;

    uint causal_len = kv_prefix_len + qi + 1u;
    if (causal_len == 0u) return;

    uint tid = tid_tg.x;
    uint tg = tg_size.x;
    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + (qi * n_heads + h) * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    // D=256 is 64 float4 spread over 32 lanes, so each lane owns *two*
    // of them: tiisg and tiisg+NW. Touching only the first truncated both
    // the Q.K dot and the output to the first 128 of 256 head dims. This
    // kernel was cloned from the d=128 one, where one float4 per lane is
    // exactly right; at d=256 it silently dropped half of every head.
    so4[tiisg] = float4(0.0f);
    so4[tiisg + NW] = float4(0.0f);
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = ic0 * C;
        if (ic >= causal_len) break;
        uint chunk = min(C, causal_len - ic);

        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = dot(sq4[tiisg], float4(k4[tiisg]))
                          + dot(sq4[tiisg + NW], float4(k4[tiisg + NW]));
            float sc = simd_sum(partial) * scale;
            if (softcap > 0.0f) {
                sc = softcap * tanh(sc / softcap);
            }
            scores[cc] = sc;
        }

        float s_lane = (tiisg < chunk) ? scores[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        so4[tiisg] *= ms;
        so4[tiisg + NW] *= ms;
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo0 = float4(0.0f);
        float4 lo1 = float4(0.0f);
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* v4 =
                (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            lo0 += float4(v4[tiisg]) * ss[cc];
            lo1 += float4(v4[tiisg + NW]) * ss[cc];
        }
        so4[tiisg] += lo0;
        so4[tiisg + NW] += lo1;
    }

    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            so0[tiisg + NW] = so0[tiisg + NW] * a0 + so1[tiisg + NW] * a1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + (qi * n_heads + h) * D);
        out4[tiisg] = so0[tiisg] * inv;
        out4[tiisg + NW] = so0[tiisg + NW] * inv;
    }
}
"#;

// Multi-token causal GQA prefill: one threadgroup per (query token, head).
// Query `qi` at absolute cache index `kv_prefix_len + qi` attends over
// K/V[0 ..= kv_prefix_len + qi] (inclusive), matching host
// `causal_gqa_attention` per position after the batch KV append.
// Used when FA-vec is off or head_dim lacks a specialized prefill kernel.
const GQA_PREFILL_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float online_rescale_prefill(float m_old, float m_new) {
    return (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
}

kernel void gqa_prefill(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_q [[buffer(7)]],
    constant uint& kv_prefix_len [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    uint h = tgpig.x;
    uint qi = tgpig.y;
    if (h >= n_heads || qi >= n_q) return;

    // Metal requires all thread-index attrs to share scalar vs vector shape.
    uint tid = tid_tg.x;
    uint tg = tg_size.x;
    uint causal_len = kv_prefix_len + qi + 1u;
    if (causal_len == 0u) return;

    threadgroup float* m_sh = shared;
    threadgroup float* s_sh = shared + tg;
    threadgroup float* acc_base = shared + 2u * tg;
    threadgroup float* my_acc = acc_base + tid * head_dim;

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(head_dim));
    device const float* q_h = q + (qi * n_heads + h) * head_dim;

    float m = -INFINITY;
    float s = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        my_acc[d] = 0.0f;
    }

    for (uint t = tid; t < causal_len; t += tg) {
        device const half* k_t =
            k_cache + (t * n_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_h[d] * float(k_t[d]);
        }
        float score = dot * scale;
        if (softcap > 0.0f) {
            score = softcap * tanh(score / softcap);
        }
        float m2 = max(m, score);
        float a = online_rescale_prefill(m, m2);
        float b = exp(score - m2);
        s = s * a + b;
        device const half* v_t =
            v_cache + (t * n_kv_heads + kv_h) * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            my_acc[d] = my_acc[d] * a + b * float(v_t[d]);
        }
        m = m2;
    }

    m_sh[tid] = m;
    s_sh[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            uint other = tid + stride;
            float m1 = m_sh[tid];
            float m2 = m_sh[other];
            float s1 = s_sh[tid];
            float s2 = s_sh[other];
            float m_new = max(m1, m2);
            float a1 = online_rescale_prefill(m1, m_new);
            float a2 = online_rescale_prefill(m2, m_new);
            m_sh[tid] = m_new;
            s_sh[tid] = s1 * a1 + s2 * a2;
            threadgroup float* acc1 = acc_base + tid * head_dim;
            threadgroup float* acc2 = acc_base + other * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                acc1[d] = acc1[d] * a1 + acc2[d] * a2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_s = 1.0f / s_sh[0];
    threadgroup float* acc0 = acc_base;
    device float* out_h = out + (qi * n_heads + h) * head_dim;
    for (uint d = tid; d < head_dim; d += tg) {
        out_h[d] = acc0[d] * inv_s;
    }
}
"#;

const ROPE_NORM_BATCH_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rope_interleaved_heads_batch(
    device float* vecs [[buffer(0)]],
    constant uint& n_heads [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    constant uint& base_pos [[buffer(4)]],
    device const float* freq_factors [[buffer(5)]],
    constant uint& use_freq_factors [[buffer(6)]],
    constant uint& n_tokens [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint t = gid.y;
    if (h >= n_heads || t >= n_tokens) return;
    uint pos = base_pos + t;
    device float* vec = vecs + (t * n_heads + h) * head_dim;
    uint half_dim = head_dim / 2u;
    for (uint i = 0; i < half_dim; i++) {
        float freq = 1.0f / pow(theta, (2.0f * float(i)) / float(head_dim));
        float angle = float(pos) * freq;
        if (use_freq_factors != 0u) {
            angle /= freq_factors[i];
        }
        float s = sin(angle);
        float c = cos(angle);
        float a = vec[2u * i];
        float b = vec[2u * i + 1u];
        vec[2u * i] = a * c - b * s;
        vec[2u * i + 1u] = a * s + b * c;
    }
}
"#;

const ROPE_NEOX_BATCH_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rope_neox_heads_batch(
    device float* vecs [[buffer(0)]],
    constant uint& n_heads [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    constant uint& base_pos [[buffer(4)]],
    device const float* freq_factors [[buffer(5)]],
    constant uint& use_freq_factors [[buffer(6)]],
    constant uint& n_tokens [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint t = gid.y;
    if (h >= n_heads || t >= n_tokens) return;
    uint pos = base_pos + t;
    device float* vec = vecs + (t * n_heads + h) * head_dim;
    uint half_dim = head_dim / 2u;
    for (uint i = 0; i < half_dim; i++) {
        float freq = 1.0f / pow(theta, (2.0f * float(i)) / float(head_dim));
        float angle = float(pos) * freq;
        if (use_freq_factors != 0u) {
            angle /= freq_factors[i];
        }
        float s = sin(angle);
        float c = cos(angle);
        float a = vec[i];
        float b = vec[i + half_dim];
        vec[i] = a * c - b * s;
        vec[i + half_dim] = a * s + b * c;
    }
}
"#;

/// Growable Metal-resident KV for one layer (`[seq, n_kv, head_dim]`).
///
/// Default **f16** matches llama.cpp `-ctk f16`. With `FERROX_CTK=q8_0` and a
/// viable head layout, stores ggml Q8_0 (~½ the bytes); attention kernels still
/// read f16 via a process-wide dequant scratch shared across layers.
pub struct MetalKvBuffers {
    dtype: MetalKvDtype,
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    capacity: usize,
}

// SAFETY: shared-mode MTLBuffers created once and mutated only from the
// decode thread that owns this cache (same justification as ResidentWeightBuffer).
unsafe impl Send for MetalKvBuffers {}
unsafe impl Sync for MetalKvBuffers {}

fn kv_store_nbytes(dtype: MetalKvDtype, elems: usize) -> Result<usize, MetalError> {
    match dtype {
        MetalKvDtype::F16 => Ok(elems * 2),
        d if d.is_q8_wire() => {
            if !elems.is_multiple_of(ferrox_quant::Q8_0_BLOCK_ELEMS) {
                return Err(MetalError::CommandFailed);
            }
            Ok((elems / ferrox_quant::Q8_0_BLOCK_ELEMS) * ferrox_quant::Q8_0_BLOCK_BYTES)
        }
        MetalKvDtype::Turbo4 => {
            if !elems.is_multiple_of(ferrox_quant::TURBO4_KV_GROUP) {
                return Err(MetalError::CommandFailed);
            }
            Ok((elems / ferrox_quant::TURBO4_KV_GROUP) * ferrox_quant::TURBO4_KV_BLOCK_BYTES)
        }
        _ => Ok(elems * 2),
    }
}

impl MetalKvBuffers {
    pub fn with_capacity(
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self, MetalError> {
        let dtype = effective_metal_kv_dtype(n_kv_heads, head_dim);
        Self::with_capacity_dtype(n_kv_heads, head_dim, max_seq_len, dtype)
    }

    /// Allocate KV buffers with an explicit store dtype (tests / callers that
    /// bypass `FERROX_CTK`).
    pub fn with_capacity_dtype(
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: MetalKvDtype,
    ) -> Result<Self, MetalError> {
        let shared = shared_metal()?;
        let dtype = match dtype {
            d if d.is_q8_wire() && !metal_kv_q8_0_viable(n_kv_heads, head_dim) => {
                return Err(MetalError::CommandFailed);
            }
            MetalKvDtype::Turbo4 if !metal_kv_turbo4_viable(n_kv_heads, head_dim) => {
                return Err(MetalError::CommandFailed);
            }
            d if d.is_implemented() => d,
            _ => MetalKvDtype::F16,
        };
        let capacity = max_seq_len.max(1);
        let elems = capacity * n_kv_heads * head_dim;
        let nbytes = kv_store_nbytes(dtype, elems)?;
        let k = shared
            .device
            .newBufferWithLength_options(nbytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocFailed)?;
        let v = shared
            .device
            .newBufferWithLength_options(nbytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocFailed)?;
        Ok(Self {
            dtype,
            k,
            v,
            n_kv_heads,
            head_dim,
            seq_len: 0,
            capacity,
        })
    }

    pub fn dtype(&self) -> MetalKvDtype {
        self.dtype
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn elems_per_token(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Overwrites device K/V from host f32 caches (e.g. after CPU prefill),
    /// converting to the store dtype on the host.
    pub fn upload_from_host(
        &mut self,
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<(), MetalError> {
        assert_eq!(k.len(), seq_len * self.elems_per_token());
        assert_eq!(v.len(), seq_len * self.elems_per_token());
        if seq_len > self.capacity {
            return Err(MetalError::CommandFailed);
        }
        let n = seq_len * self.elems_per_token();
        match self.dtype {
            d if d.is_q8_wire() => {
                let k_q = ferrox_quant::quantize_q8_0(&k[..n]);
                let v_q = ferrox_quant::quantize_q8_0(&v[..n]);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_q.as_ptr(),
                        self.k.contents().as_ptr() as *mut u8,
                        k_q.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        v_q.as_ptr(),
                        self.v.contents().as_ptr() as *mut u8,
                        v_q.len(),
                    );
                }
            }
            MetalKvDtype::Turbo4 => {
                let k_q = ferrox_quant::pack_turbo4_kv_blocks(&k[..n]);
                let v_q = ferrox_quant::pack_turbo4_kv_blocks(&v[..n]);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_q.as_ptr(),
                        self.k.contents().as_ptr() as *mut u8,
                        k_q.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        v_q.as_ptr(),
                        self.v.contents().as_ptr() as *mut u8,
                        v_q.len(),
                    );
                }
            }
            _ => {
                let k_f16: Vec<u16> = k[..n]
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let v_f16: Vec<u16> = v[..n]
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let nbytes = n * 2;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_f16.as_ptr() as *const u8,
                        self.k.contents().as_ptr() as *mut u8,
                        nbytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        v_f16.as_ptr() as *const u8,
                        self.v.contents().as_ptr() as *mut u8,
                        nbytes,
                    );
                }
            }
        }
        self.seq_len = seq_len;
        Ok(())
    }

    /// Copies the last appended token's K/V to host f32 (after a completed CB).
    pub fn last_token_host(&self) -> (Vec<f32>, Vec<f32>) {
        assert!(self.seq_len > 0);
        let (k, v) = self.tokens_host(self.seq_len - 1, 1);
        (k, v)
    }

    /// Downloads `n` tokens starting at `start` as f32 (after a completed CB).
    pub fn tokens_host(&self, start: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(start + n <= self.seq_len);
        let per = self.elems_per_token();
        let off = start * per;
        let elems = n * per;
        match self.dtype {
            d if d.is_q8_wire() => {
                let nbytes =
                    (elems / ferrox_quant::Q8_0_BLOCK_ELEMS) * ferrox_quant::Q8_0_BLOCK_BYTES;
                let byte_off =
                    (off / ferrox_quant::Q8_0_BLOCK_ELEMS) * ferrox_quant::Q8_0_BLOCK_BYTES;
                let k_ptr = self.k.contents();
                let v_ptr = self.v.contents();
                let k_bytes = unsafe {
                    std::slice::from_raw_parts(k_ptr.as_ptr().add(byte_off) as *const u8, nbytes)
                };
                let v_bytes = unsafe {
                    std::slice::from_raw_parts(v_ptr.as_ptr().add(byte_off) as *const u8, nbytes)
                };
                (
                    ferrox_quant::dequant_q8_0(k_bytes).expect("q8 k aligned"),
                    ferrox_quant::dequant_q8_0(v_bytes).expect("q8 v aligned"),
                )
            }
            MetalKvDtype::Turbo4 => {
                let nbytes =
                    (elems / ferrox_quant::TURBO4_KV_GROUP) * ferrox_quant::TURBO4_KV_BLOCK_BYTES;
                let byte_off =
                    (off / ferrox_quant::TURBO4_KV_GROUP) * ferrox_quant::TURBO4_KV_BLOCK_BYTES;
                let k_ptr = self.k.contents();
                let v_ptr = self.v.contents();
                let k_bytes = unsafe {
                    std::slice::from_raw_parts(k_ptr.as_ptr().add(byte_off) as *const u8, nbytes)
                };
                let v_bytes = unsafe {
                    std::slice::from_raw_parts(v_ptr.as_ptr().add(byte_off) as *const u8, nbytes)
                };
                (
                    ferrox_quant::unpack_turbo4_kv_blocks(k_bytes).expect("turbo4 k"),
                    ferrox_quant::unpack_turbo4_kv_blocks(v_bytes).expect("turbo4 v"),
                )
            }
            _ => {
                let k_ptr = self.k.contents();
                let v_ptr = self.v.contents();
                let k = unsafe {
                    std::slice::from_raw_parts(k_ptr.as_ptr() as *const u16, off + elems)
                };
                let v = unsafe {
                    std::slice::from_raw_parts(v_ptr.as_ptr() as *const u16, off + elems)
                };
                (
                    k[off..off + elems]
                        .iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .collect(),
                    v[off..off + elems]
                        .iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .collect(),
                )
            }
        }
    }
}

/// K or V plane when appending into [`MetalKvBuffers`].
#[derive(Clone, Copy)]
enum KvPlane {
    K,
    V,
}

/// Process-wide f16 view of Q8_0 KV for FA/GQA (one pair shared across layers).
struct Q8AttnScratch {
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    elems_cap: usize,
}

// SAFETY: gated by Q8_ATTN_SCRATCH mutex; encode holds the lock for the CB encode.
unsafe impl Send for Q8AttnScratch {}

static Q8_ATTN_SCRATCH: Mutex<Option<Q8AttnScratch>> = Mutex::new(None);

fn borrow_q8_attn_scratch(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    elems: usize,
) -> Result<std::sync::MutexGuard<'static, Option<Q8AttnScratch>>, MetalError> {
    let mut guard = Q8_ATTN_SCRATCH.lock().unwrap();
    let fits = guard.as_ref().is_some_and(|s| s.elems_cap >= elems);
    if !fits {
        let nbytes = elems.max(1) * 2;
        *guard = Some(Q8AttnScratch {
            k: device
                .newBufferWithLength_options(nbytes, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
            v: device
                .newBufferWithLength_options(nbytes, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
            elems_cap: elems.max(1),
        });
    }
    Ok(guard)
}

fn alloc_f32_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    device
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)
}

fn alloc_half_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    device
        .newBufferWithLength_options(n * 2, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)
}

fn alloc_u32_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    device
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)
}

fn upload_f32(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let mut owned = data.to_vec();
    unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(owned.as_mut_ptr() as *mut _).unwrap(),
            owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)
}

/// Upload host f32 as packed f16 (Metal KV / host-probe GQA).
fn upload_f16_from_f32(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let mut bits: Vec<u16> = data
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(bits.as_mut_ptr() as *mut _).unwrap(),
            bits.len() * 2,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)
}

fn copy_f32_into(buf: &ProtocolObject<dyn MTLBuffer>, data: &[f32]) {
    let nbytes = data.len() * 4;
    debug_assert!(buf.length() >= nbytes);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            nbytes,
        );
    }
}

/// Process-wide activation scratch for [`launch_decode_dense_stack`].
/// Avoids allocating ~10 MTLBuffers every decode token.
struct DecodeScratch {
    h: Retained<ProtocolObject<dyn MTLBuffer>>,
    x: Retained<ProtocolObject<dyn MTLBuffer>>,
    x2: Retained<ProtocolObject<dyn MTLBuffer>>,
    q: Retained<ProtocolObject<dyn MTLBuffer>>,
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn: Retained<ProtocolObject<dyn MTLBuffer>>,
    o: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    up: Retained<ProtocolObject<dyn MTLBuffer>>,
    act: Retained<ProtocolObject<dyn MTLBuffer>>,
    down: Retained<ProtocolObject<dyn MTLBuffer>>,
    logits: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Single u32 slot for greedy argmax-in-stack (always resident; 4 bytes).
    argmax_idx: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_cap: usize,
    max_q_cap: usize,
    max_kv_cap: usize,
    attn_cap: usize,
    max_gate_cap: usize,
    logits_cap: usize,
}

// SAFETY: scratch is gated by DECODE_SCRATCH mutex; only one encode at a time.
unsafe impl Send for DecodeScratch {}

static DECODE_SCRATCH: Mutex<Option<DecodeScratch>> = Mutex::new(None);

/// Process-wide activation scratch for [`launch_prefill_dense_layer`].
/// Same residency idea as [`DecodeScratch`], sized for batch B≥4.
struct PrefillScratch {
    h: Retained<ProtocolObject<dyn MTLBuffer>>,
    x: Retained<ProtocolObject<dyn MTLBuffer>>,
    x2: Retained<ProtocolObject<dyn MTLBuffer>>,
    q: Retained<ProtocolObject<dyn MTLBuffer>>,
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn: Retained<ProtocolObject<dyn MTLBuffer>>,
    o: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    up: Retained<ProtocolObject<dyn MTLBuffer>>,
    act: Retained<ProtocolObject<dyn MTLBuffer>>,
    down: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Reused f16 activation plane for `mul_mm_sg_f16` (max of hidden/q/gate).
    half_act: Retained<ProtocolObject<dyn MTLBuffer>>,
    half_act_cap: usize,
    batch_cap: usize,
    hidden_cap: usize,
    max_q_cap: usize,
    max_kv_cap: usize,
    max_gate_cap: usize,
}

// SAFETY: gated by PREFILL_SCRATCH mutex; one encode at a time.
unsafe impl Send for PrefillScratch {}

static PREFILL_SCRATCH: Mutex<Option<PrefillScratch>> = Mutex::new(None);

/// Shape key for a retained prefill command-buffer plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefillCbKey {
    pub layer: u32,
    pub batch: u32,
    pub hidden: u32,
    pub ffn: u32,
    pub q_rows: u32,
}

/// Shape key for a multi-layer prefill command-buffer plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefillStackCbKey {
    pub start_layer: u32,
    pub depth: u32,
    pub batch: u32,
    pub hidden: u32,
}

/// Stub cache for one-CB-per-layer encoding plans.
///
/// Partial step toward llama.cpp `ggml_metal_graph_compute`: retain encoded
/// `MTLCommandBuffer` templates (or ICB / graph nodes) keyed by
/// [`PrefillCbKey`] and replay with updated buffer bindings. Today we
/// record shape keys and which compute pipelines are warmed (compiled once
/// via [`MetalGraph::warm_prefill_pipelines`]) so the first dense-prefill
/// layer avoids repeated `ensure_pipeline` / Metal compile latency.
#[derive(Default, Debug)]
pub struct PrefillCbCache {
    keys: HashSet<PrefillCbKey>,
    stack_keys: HashSet<PrefillStackCbKey>,
    /// Kernel function names compiled and resident (process pipeline cache).
    hot_pipelines: HashSet<&'static str>,
}

impl PrefillCbCache {
    pub fn note(&mut self, key: PrefillCbKey) -> bool {
        self.keys.insert(key)
    }

    pub fn note_stack(&mut self, key: PrefillStackCbKey) -> bool {
        self.stack_keys.insert(key)
    }

    pub fn contains(&self, key: &PrefillCbKey) -> bool {
        self.keys.contains(key)
    }

    pub fn contains_stack(&self, key: &PrefillStackCbKey) -> bool {
        self.stack_keys.contains(key)
    }

    pub fn mark_pipeline_hot(&mut self, fn_name: &'static str) {
        self.hot_pipelines.insert(fn_name);
    }

    pub fn is_pipeline_hot(&self, fn_name: &str) -> bool {
        self.hot_pipelines.contains(fn_name)
    }

    pub fn hot_pipeline_count(&self) -> usize {
        self.hot_pipelines.len()
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Parameters for [`MetalGraph::warm_prefill_pipelines`].
pub struct PrefillWarmParams<'a> {
    pub layer: &'a PrefillDenseLayerMetal<'a>,
    pub rope_layout: MetalRopeLayout,
    pub head_dim: u32,
    pub gelu_ffn: bool,
    pub kv_dtype: MetalKvDtype,
}

/// Minimal Metal encode-plan holder (prefill CB cache; decode replay later).
#[derive(Default, Debug)]
pub struct MetalGraph {
    pub prefill: PrefillCbCache,
    prefill_pipelines_warmed: bool,
}

impl MetalGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn prefill_pipelines_warmed(&self) -> bool {
        self.prefill_pipelines_warmed
    }

    /// Compile mul_mm_sg, RMSNorm, RoPE, KV append/dequant, GQA, and FFN
    /// elementwise pipelines used by [`launch_prefill_dense_layer`]. Mirrors
    /// the pipeline residency half of llama.cpp `ggml_metal_graph_compute`
    /// (full CB / graph replay is still TODO).
    pub fn warm_prefill_pipelines(
        &mut self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        params: PrefillWarmParams<'_>,
    ) -> Result<(), MetalError> {
        let mut mark = |name: &'static str| self.prefill.mark_pipeline_hot(name);

        warm_prefill_elem_pipelines(device, params.gelu_ffn)?;
        mark("rms_norm_f32");
        mark("vec_add_f32");
        mark(if params.gelu_ffn {
            "gelu_mul_f32"
        } else {
            "silu_mul_f32"
        });

        for launch in [
            &params.layer.q,
            &params.layer.k,
            &params.layer.v,
            &params.layer.o,
            &params.layer.gate,
            &params.layer.up,
            &params.layer.down,
        ] {
            warm_mul_mm_sg_pipeline(device, launch.fn_name)?;
            mark(launch.fn_name);
            let f16_static: &'static str = match launch.fn_name {
                "q4_k_mul_mm_sg" => "q4_k_mul_mm_sg_f16",
                "q5_k_mul_mm_sg" => "q5_k_mul_mm_sg_f16",
                "q6_k_mul_mm_sg" => "q6_k_mul_mm_sg_f16",
                "q8_0_mul_mm_sg" => "q8_0_mul_mm_sg_f16",
                "q4_0_mul_mm_sg" => "q4_0_mul_mm_sg_f16",
                "q5_0_mul_mm_sg" => "q5_0_mul_mm_sg_f16",
                "iq4_xs_mul_mm_sg" => "iq4_xs_mul_mm_sg_f16",
                _ => continue,
            };
            warm_mul_mm_sg_pipeline(device, f16_static)?;
            mark(f16_static);
        }

        let (rope_src, rope_name) = match params.rope_layout {
            MetalRopeLayout::Norm => (ROPE_NORM_BATCH_KERNEL_SRC, "rope_interleaved_heads_batch"),
            MetalRopeLayout::Neox => (ROPE_NEOX_BATCH_KERNEL_SRC, "rope_neox_heads_batch"),
        };
        ensure_pipeline(device, rope_src, rope_name)?;
        mark(rope_name);

        match params.kv_dtype {
            d if d.is_q8_wire() => {
                ensure_pipeline(device, KV_APPEND_Q8_0_KERNEL_SRC, "kv_append_q8_0")?;
                mark("kv_append_q8_0");
                ensure_pipeline(
                    device,
                    DEQUANT_Q8_0_TO_F16_KERNEL_SRC,
                    "dequant_q8_0_to_f16",
                )?;
                mark("dequant_q8_0_to_f16");
            }
            MetalKvDtype::Turbo4 => {
                ensure_pipeline(device, KV_APPEND_TURBO4_KERNEL_SRC, "kv_append_turbo4")?;
                mark("kv_append_turbo4");
                ensure_pipeline(
                    device,
                    DEQUANT_TURBO4_TO_F16_KERNEL_SRC,
                    "dequant_turbo4_to_f16",
                )?;
                mark("dequant_turbo4_to_f16");
            }
            _ => {
                ensure_pipeline(device, KV_APPEND_KERNEL_SRC, "kv_append")?;
                mark("kv_append");
            }
        }

        warm_gqa_prefill_pipeline(device, params.head_dim, &mut mark)?;

        self.prefill_pipelines_warmed = true;
        Ok(())
    }
}

fn warm_gqa_prefill_pipeline(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    head_dim: u32,
    mark: &mut dyn FnMut(&'static str),
) -> Result<(), MetalError> {
    if metal_fa_vec_enabled() && gqa_prefill_fa_vec_supported(head_dim) {
        let (src, name) = match head_dim {
            64 => (GQA_PREFILL_FA_VEC_D64_KERNEL_SRC, "gqa_prefill_fa_vec_d64"),
            96 => (GQA_PREFILL_FA_VEC_D96_KERNEL_SRC, "gqa_prefill_fa_vec_d96"),
            128 => (GQA_PREFILL_FA_VEC_KERNEL_SRC, "gqa_prefill_fa_vec"),
            256 => (
                GQA_PREFILL_FA_VEC_D256_KERNEL_SRC,
                "gqa_prefill_fa_vec_d256",
            ),
            _ => return Err(MetalError::CommandFailed),
        };
        ensure_pipeline(device, src, name)?;
        mark(name);
    } else {
        ensure_pipeline(device, GQA_PREFILL_KERNEL_SRC, "gqa_prefill")?;
        mark("gqa_prefill");
    }
    Ok(())
}

static PREFILL_GRAPH: OnceLock<Mutex<MetalGraph>> = OnceLock::new();

/// Process-wide [`MetalGraph`] (stub). Intended for decode-stack replay
/// and future `ggml_metal_graph_compute`-style CB retention.
pub fn metal_graph() -> std::sync::MutexGuard<'static, MetalGraph> {
    PREFILL_GRAPH
        .get_or_init(|| Mutex::new(MetalGraph::new()))
        .lock()
        .unwrap()
}

struct ScratchCaps {
    hidden: usize,
    max_q: usize,
    max_kv: usize,
    attn: usize,
    max_gate: usize,
    logits: usize,
}

struct PrefillScratchCaps {
    batch: usize,
    hidden: usize,
    max_q: usize,
    max_kv: usize,
    max_gate: usize,
}

fn borrow_decode_scratch(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    caps: ScratchCaps,
) -> Result<std::sync::MutexGuard<'static, Option<DecodeScratch>>, MetalError> {
    let mut guard = DECODE_SCRATCH.lock().unwrap();
    let fits = match guard.as_ref() {
        Some(s) => {
            s.hidden_cap >= caps.hidden
                && s.max_q_cap >= caps.max_q
                && s.max_kv_cap >= caps.max_kv
                && s.attn_cap >= caps.attn
                && s.max_gate_cap >= caps.max_gate
                && s.logits_cap >= caps.logits
        }
        None => false,
    };
    if !fits {
        let logits = if caps.logits > 0 {
            Some(alloc_f32_buffer(device, caps.logits)?)
        } else {
            None
        };
        *guard = Some(DecodeScratch {
            h: alloc_f32_buffer(device, caps.hidden)?,
            x: alloc_f32_buffer(device, caps.hidden)?,
            x2: alloc_f32_buffer(device, caps.hidden)?,
            q: alloc_f32_buffer(device, caps.max_q)?,
            k: alloc_f32_buffer(device, caps.max_kv)?,
            v: alloc_f32_buffer(device, caps.max_kv)?,
            attn: alloc_f32_buffer(device, caps.attn)?,
            o: alloc_f32_buffer(device, caps.hidden)?,
            gate: alloc_f32_buffer(device, caps.max_gate)?,
            up: alloc_f32_buffer(device, caps.max_gate)?,
            act: alloc_f32_buffer(device, caps.max_gate)?,
            down: alloc_f32_buffer(device, caps.hidden)?,
            logits,
            argmax_idx: alloc_u32_buffer(device, 1)?,
            hidden_cap: caps.hidden,
            max_q_cap: caps.max_q,
            max_kv_cap: caps.max_kv,
            attn_cap: caps.attn,
            max_gate_cap: caps.max_gate,
            logits_cap: caps.logits,
        });
    }
    Ok(guard)
}

fn borrow_prefill_scratch(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    caps: PrefillScratchCaps,
) -> Result<std::sync::MutexGuard<'static, Option<PrefillScratch>>, MetalError> {
    let mut guard = PREFILL_SCRATCH.lock().unwrap();
    let fits = match guard.as_ref() {
        Some(s) => {
            s.batch_cap >= caps.batch
                && s.hidden_cap >= caps.hidden
                && s.max_q_cap >= caps.max_q
                && s.max_kv_cap >= caps.max_kv
                && s.max_gate_cap >= caps.max_gate
                && s.half_act_cap >= caps.batch * caps.hidden.max(caps.max_q).max(caps.max_gate)
        }
        None => false,
    };
    if !fits {
        let bh = caps.batch * caps.hidden;
        let half_cap = caps.batch * caps.hidden.max(caps.max_q).max(caps.max_gate);
        *guard = Some(PrefillScratch {
            h: alloc_f32_buffer(device, bh)?,
            x: alloc_f32_buffer(device, bh)?,
            x2: alloc_f32_buffer(device, bh)?,
            q: alloc_f32_buffer(device, caps.batch * caps.max_q)?,
            k: alloc_f32_buffer(device, caps.batch * caps.max_kv)?,
            v: alloc_f32_buffer(device, caps.batch * caps.max_kv)?,
            attn: alloc_f32_buffer(device, caps.batch * caps.max_q)?,
            o: alloc_f32_buffer(device, bh)?,
            gate: alloc_f32_buffer(device, caps.batch * caps.max_gate)?,
            up: alloc_f32_buffer(device, caps.batch * caps.max_gate)?,
            act: alloc_f32_buffer(device, caps.batch * caps.max_gate)?,
            down: alloc_f32_buffer(device, bh)?,
            half_act: alloc_half_buffer(device, half_cap)?,
            half_act_cap: half_cap,
            batch_cap: caps.batch,
            hidden_cap: caps.hidden,
            max_q_cap: caps.max_q,
            max_kv_cap: caps.max_kv,
            max_gate_cap: caps.max_gate,
        });
    }
    Ok(guard)
}

#[allow(clippy::too_many_arguments)]
fn encode_rope(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    layout: MetalRopeLayout,
    vecs: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    head_dim: u32,
    theta: f32,
    pos: u32,
    freq_factors: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<(), MetalError> {
    let (src, name) = match layout {
        MetalRopeLayout::Norm => (ROPE_NORM_KERNEL_SRC, "rope_interleaved_heads"),
        MetalRopeLayout::Neox => (ROPE_NEOX_KERNEL_SRC, "rope_neox_heads"),
    };
    let pipe = ensure_pipeline(device, src, name)?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(vecs), 0, 0);
        let mut n_heads_u = n_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_heads_u as *mut u32 as *mut _).unwrap(),
            4,
            1,
        );
        let mut head_dim_u = head_dim;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut head_dim_u as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
        let mut theta_f = theta;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut theta_f as *mut f32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut pos_u = pos;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut pos_u as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        if let Some(ff) = freq_factors {
            encoder.setBuffer_offset_atIndex(Some(ff), 0, 5);
            let mut use_ff = 1u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut use_ff as *mut u32 as *mut _).unwrap(),
                4,
                6,
            );
        } else {
            // Unused device buffer slot — bind a 4-byte scratch so index 5 is valid.
            let mut scratch = [0u8; 4];
            encoder.setBytes_length_atIndex(
                NonNull::new(scratch.as_mut_ptr() as *mut _).unwrap(),
                4,
                5,
            );
            let mut use_ff = 0u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut use_ff as *mut u32 as *mut _).unwrap(),
                4,
                6,
            );
        }
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_rope_batch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    layout: MetalRopeLayout,
    vecs: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    head_dim: u32,
    theta: f32,
    base_pos: u32,
    n_tokens: u32,
    freq_factors: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<(), MetalError> {
    if n_tokens == 0 {
        return Ok(());
    }
    let (src, name) = match layout {
        MetalRopeLayout::Norm => (ROPE_NORM_BATCH_KERNEL_SRC, "rope_interleaved_heads_batch"),
        MetalRopeLayout::Neox => (ROPE_NEOX_BATCH_KERNEL_SRC, "rope_neox_heads_batch"),
    };
    let pipe = ensure_pipeline(device, src, name)?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(vecs), 0, 0);
        let mut n_heads_u = n_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_heads_u as *mut u32 as *mut _).unwrap(),
            4,
            1,
        );
        let mut head_dim_u = head_dim;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut head_dim_u as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
        let mut theta_f = theta;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut theta_f as *mut f32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut base_pos_u = base_pos;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut base_pos_u as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        if let Some(ff) = freq_factors {
            encoder.setBuffer_offset_atIndex(Some(ff), 0, 5);
            let mut use_ff = 1u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut use_ff as *mut u32 as *mut _).unwrap(),
                4,
                6,
            );
        } else {
            let mut scratch = [0u8; 4];
            encoder.setBytes_length_atIndex(
                NonNull::new(scratch.as_mut_ptr() as *mut _).unwrap(),
                4,
                5,
            );
            let mut use_ff = 0u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut use_ff as *mut u32 as *mut _).unwrap(),
                4,
                6,
            );
        }
        let mut n_tok = n_tokens;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_tok as *mut u32 as *mut _).unwrap(),
            4,
            7,
        );
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads as usize,
            height: n_tokens as usize,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_kv_append(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    offset_elems: u32,
    n_elems: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, KV_APPEND_KERNEL_SRC, "kv_append")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let mut off = offset_elems;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut off as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
        let mut n = n_elems;
        encoder.setBytes_length_atIndex(NonNull::new(&mut n as *mut u32 as *mut _).unwrap(), 4, 3);
    }
    let tg = 256usize.min(n_elems as usize).max(1);
    let n_tg = (n_elems as usize).div_ceil(tg);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_kv_append_q8_0(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    offset_elems: u32,
    n_elems: u32,
) -> Result<(), MetalError> {
    if !offset_elems.is_multiple_of(ferrox_quant::Q8_0_BLOCK_ELEMS as u32)
        || !n_elems.is_multiple_of(ferrox_quant::Q8_0_BLOCK_ELEMS as u32)
    {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, KV_APPEND_Q8_0_KERNEL_SRC, "kv_append_q8_0")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let mut off = offset_elems;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut off as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
        let mut n = n_elems;
        encoder.setBytes_length_atIndex(NonNull::new(&mut n as *mut u32 as *mut _).unwrap(), 4, 3);
    }
    let n_blocks = (n_elems as usize) / ferrox_quant::Q8_0_BLOCK_ELEMS;
    let tg = 256usize.min(n_blocks).max(1);
    let n_tg = n_blocks.div_ceil(tg);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_dequant_q8_0_to_f16(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    n_elems: u32,
) -> Result<(), MetalError> {
    if !n_elems.is_multiple_of(ferrox_quant::Q8_0_BLOCK_ELEMS as u32) {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(
        device,
        DEQUANT_Q8_0_TO_F16_KERNEL_SRC,
        "dequant_q8_0_to_f16",
    )?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let mut n = n_elems;
        encoder.setBytes_length_atIndex(NonNull::new(&mut n as *mut u32 as *mut _).unwrap(), 4, 2);
    }
    let n_blocks = (n_elems as usize) / ferrox_quant::Q8_0_BLOCK_ELEMS;
    let tg = 256usize.min(n_blocks).max(1);
    let n_tg = n_blocks.div_ceil(tg);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_kv_append_turbo4(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    offset_elems: u32,
    n_elems: u32,
) -> Result<(), MetalError> {
    if !offset_elems.is_multiple_of(ferrox_quant::TURBO4_KV_GROUP as u32)
        || !n_elems.is_multiple_of(ferrox_quant::TURBO4_KV_GROUP as u32)
    {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, KV_APPEND_TURBO4_KERNEL_SRC, "kv_append_turbo4")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let mut off = offset_elems;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut off as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
        let mut n = n_elems;
        encoder.setBytes_length_atIndex(NonNull::new(&mut n as *mut u32 as *mut _).unwrap(), 4, 3);
    }
    let n_blocks = (n_elems as usize) / ferrox_quant::TURBO4_KV_GROUP;
    let tg = 256usize.min(n_blocks).max(1);
    let n_tg = n_blocks.div_ceil(tg);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_dequant_turbo4_to_f16(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    n_elems: u32,
) -> Result<(), MetalError> {
    if !n_elems.is_multiple_of(ferrox_quant::TURBO4_KV_GROUP as u32) {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(
        device,
        DEQUANT_TURBO4_TO_F16_KERNEL_SRC,
        "dequant_turbo4_to_f16",
    )?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let mut n = n_elems;
        encoder.setBytes_length_atIndex(NonNull::new(&mut n as *mut u32 as *mut _).unwrap(), 4, 2);
    }
    let n_blocks = (n_elems as usize) / ferrox_quant::TURBO4_KV_GROUP;
    let tg = 256usize.min(n_blocks).max(1);
    let n_tg = n_blocks.div_ceil(tg);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_kv_dequant_to_f16(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    dtype: MetalKvDtype,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    n_elems: u32,
) -> Result<(), MetalError> {
    match dtype {
        d if d.is_q8_wire() => encode_dequant_q8_0_to_f16(encoder, device, src, dst, n_elems),
        MetalKvDtype::Turbo4 => encode_dequant_turbo4_to_f16(encoder, device, src, dst, n_elems),
        _ => Err(MetalError::CommandFailed),
    }
}

fn encode_kv_store_append(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    kv: &MetalKvBuffers,
    plane: KvPlane,
    offset_elems: u32,
    n_elems: u32,
) -> Result<(), MetalError> {
    let dst: &ProtocolObject<dyn MTLBuffer> = match plane {
        KvPlane::K => &kv.k,
        KvPlane::V => &kv.v,
    };
    match kv.dtype {
        d if d.is_q8_wire() => {
            encode_kv_append_q8_0(encoder, device, src, dst, offset_elems, n_elems)
        }
        MetalKvDtype::Turbo4 => {
            encode_kv_append_turbo4(encoder, device, src, dst, offset_elems, n_elems)
        }
        _ => encode_kv_append(encoder, device, src, dst, offset_elems, n_elems),
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_gqa_with_kv(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    kv: &MetalKvBuffers,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_start: u32,
    softcap: Option<f32>,
) -> Result<(), MetalError> {
    if kv.dtype.needs_f16_scratch() {
        let elems = (seq_len as usize) * kv.elems_per_token();
        let mut guard = borrow_q8_attn_scratch(device, elems)?;
        let scratch = guard.as_mut().unwrap();
        encode_kv_dequant_to_f16(encoder, device, kv.dtype, &kv.k, &scratch.k, elems as u32)?;
        encode_kv_dequant_to_f16(encoder, device, kv.dtype, &kv.v, &scratch.v, elems as u32)?;
        memory_barrier_buffers(encoder);
        encode_gqa(
            encoder, device, q, &scratch.k, &scratch.v, out, n_heads, n_kv_heads, head_dim,
            seq_len, kv_start, softcap,
        )
    } else {
        encode_gqa(
            encoder, device, q, &kv.k, &kv.v, out, n_heads, n_kv_heads, head_dim, seq_len,
            kv_start, softcap,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_gqa_prefill_with_kv(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    kv: &MetalKvBuffers,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_q: u32,
    kv_prefix_len: u32,
    attn_softcap: Option<f32>,
) -> Result<(), MetalError> {
    let total_seq = kv_prefix_len + n_q;
    if kv.dtype.needs_f16_scratch() {
        let elems = (total_seq as usize) * kv.elems_per_token();
        let mut guard = borrow_q8_attn_scratch(device, elems)?;
        let scratch = guard.as_mut().unwrap();
        encode_kv_dequant_to_f16(encoder, device, kv.dtype, &kv.k, &scratch.k, elems as u32)?;
        encode_kv_dequant_to_f16(encoder, device, kv.dtype, &kv.v, &scratch.v, elems as u32)?;
        // Concurrent encode: GQA must not race the dequant writes.
        memory_barrier_buffers(encoder);
        encode_gqa_prefill(
            encoder,
            device,
            q,
            &scratch.k,
            &scratch.v,
            out,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
            attn_softcap,
        )
    } else {
        encode_gqa_prefill(
            encoder,
            device,
            q,
            &kv.k,
            &kv.v,
            out,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
            attn_softcap,
        )
    }
}

/// TG size for decode GQA: multiple of 32 with **power-of-two** N_SG
/// (cross-SG tree reduce). Compact TG mem after per-SG register reduce:
/// `(2 * nsg + nsg * head_dim) * 4`.
fn gqa_decode_threadgroup_size(seq_len: u32, head_dim: u32) -> u32 {
    const TG_BUDGET_BYTES: u32 = 28 * 1024;
    const NW: u32 = 32;
    let per_sg = head_dim.saturating_add(2).saturating_mul(4).max(1);
    let max_nsg = (TG_BUDGET_BYTES / per_sg).clamp(1, 8);
    let raw = seq_len.div_ceil(NW).clamp(1, 4).min(max_nsg);
    // Power-of-two N_SG only (1/2/4) so the cross-SG tree reduce is complete.
    let nsg = if raw >= 4 && max_nsg >= 4 {
        4
    } else if raw >= 2 && max_nsg >= 2 {
        2
    } else {
        1
    };
    nsg * NW
}

/// TG size for FA-vec decode (d=64/96/128/256): NSG=8 × NW=32.
fn gqa_fa_vec_threadgroup_size(_head_dim: u32) -> u32 {
    256
}

/// Prefill FA-vec TG size. d=64 wastes half of each simdgroup on Q·K
/// (`D4=16` of 32 lanes), so prefer fewer simdgroups → more concurrent
/// TGs (one TG still owns one query). Measured on SmolLM2 Metal pp512.
fn gqa_prefill_fa_vec_threadgroup_size(head_dim: u32) -> u32 {
    match head_dim {
        // d=64: D4=16 → half of each SG idle on Q·K. Prefer 2 SG (64
        // threads) so more (head,query) TGs stay in flight on tiny pp512.
        64 => 64,
        96 => 128,
        _ => 256,
    }
}

/// Head dims the FA-vec decode kernels cover (dedicated specializations).
fn gqa_fa_vec_supported(head_dim: u32) -> bool {
    matches!(head_dim, 64 | 96 | 128 | 256)
}

/// Prefill GQA keeps the legacy per-thread TG-acc layout when FA-vec is
/// off or head_dim is unsupported; TG must be a power of two for its tree
/// reduce. Separate from decode so NeoX/prefill agents can evolve this
/// without fighting decode occupancy tweaks.
fn gqa_prefill_threadgroup_size(seq_len: u32, head_dim: u32) -> u32 {
    const TG_BUDGET_BYTES: u32 = 28 * 1024;
    let per_thread = head_dim.saturating_add(2).saturating_mul(4).max(1);
    let max_by_mem = (TG_BUDGET_BYTES / per_thread).max(1);
    let want = seq_len.clamp(32, 128).min(max_by_mem).max(1);
    let mut tg = 1u32 << (31 - want.leading_zeros());
    while tg > max_by_mem && tg > 1 {
        tg >>= 1;
    }
    tg.max(1)
}

/// FA-vec prefill head dims: 64 (SmolLM2 / TinyLlama / Llama-3.2-1B),
/// 96, 128 (Llama-3), 256 (Gemma-2/3). Anything else falls back to the
/// legacy per-thread-accumulator `gqa_prefill`.
fn gqa_prefill_fa_vec_supported(head_dim: u32) -> bool {
    matches!(head_dim, 64 | 96 | 128 | 256)
}

#[allow(clippy::too_many_arguments)]
fn encode_gqa_prefill_fa_nq4_d64(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    v: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    n_q: u32,
    kv_prefix_len: u32,
    softcap: f32,
) -> Result<(), MetalError> {
    const QN: u32 = 4;
    const D: u32 = 64;
    let pipe = ensure_pipeline(
        device,
        GQA_PREFILL_FA_NQ4_D64_KERNEL_SRC,
        "gqa_prefill_fa_nq4_d64",
    )?;
    encoder.setComputePipelineState(&pipe.0);
    let tg = 64u32; // 2 SG — same as d64 FA-vec
    let nsg = tg / 32;
    // QN*D queries + NSG * (C + QN*D)
    let tg_mem = ((QN * D + nsg * (32 + QN * D)) * 4) as usize;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut nh = n_heads;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nh as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut nkv = n_kv_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut nkv as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut hd = D;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 6);
        let mut nq = n_q;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nq as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut prefix = kv_prefix_len;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut prefix as *mut u32 as *mut _).unwrap(),
            4,
            8,
        );
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    let n_tg_y = n_q.div_ceil(QN) as usize;
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads as usize,
            height: n_tg_y,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// llama `kernel_flash_attn_ext` dispatch: QN=8, C=64, NSG=4 (128 threads/TG).
///
/// `head_dim` must be 64 or 128 — the two widths the MMA kernel is
/// instantiated at. The scalar `fa_ext` predecessor only exists at d=64, so
/// `FERROX_METAL_FA_MMA=0` is honoured there and the caller keeps d=128 off
/// this path entirely when MMA is disabled.
#[allow(clippy::too_many_arguments)]
fn encode_gqa_prefill_fa_ext(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    v: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_q: u32,
    kv_prefix_len: u32,
    softcap: f32,
) -> Result<(), MetalError> {
    const QN: u32 = 8;
    const C: u32 = 64;
    const NSG: u32 = 4;
    const SH: u32 = 2 * C;
    let d = head_dim;
    let mma = gqa_prefill_fa_mma_enabled();
    let pipe = match (d, mma) {
        (64, true) => ensure_pipeline(
            device,
            GQA_PREFILL_FA_EXT_MMA_D64_KERNEL_SRC,
            "gqa_prefill_fa_ext_mma_d64",
        )?,
        (128, true) => ensure_pipeline(
            device,
            GQA_PREFILL_FA_EXT_MMA_D128_KERNEL_SRC,
            "gqa_prefill_fa_ext_mma_d128",
        )?,
        (64, false) => ensure_pipeline(
            device,
            GQA_PREFILL_FA_EXT_D64_KERNEL_SRC,
            "gqa_prefill_fa_ext_d64",
        )?,
        _ => return Err(MetalError::CommandFailed),
    };
    encoder.setComputePipelineState(&pipe.0);
    let tg = 32 * NSG;
    // sq[QN,D] + so[QN,D] + ss[QN,SH], plus kpad[8,D]+vpad[8,D] as f16 for the
    // MMA variant's ≤7-row cache tail. 10 KiB at d=64, 16 KiB at d=128.
    let tg_mem =
        ((2 * QN * d + QN * SH) * 4) as usize + if mma { (2 * 8 * d * 2) as usize } else { 0 };
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut nh = n_heads;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nh as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut nkv = n_kv_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut nkv as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut hd = d;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 6);
        let mut nq = n_q;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nq as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut prefix = kv_prefix_len;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut prefix as *mut u32 as *mut _).unwrap(),
            4,
            8,
        );
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads as usize,
            height: n_q.div_ceil(QN) as usize,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Default-on for d=64 prefill when n_q≥8 (beats FA-vec on SmolLM2 pp512).
/// Opt out: `FERROX_METAL_FA_EXT=0`.
fn gqa_prefill_fa_ext_d64_enabled() -> bool {
    !matches!(
        std::env::var("FERROX_METAL_FA_EXT").ok().as_deref(),
        Some("0") | Some("false") | Some("off") | Some("vec")
    )
}

/// Simdgroup-MMA score + P·V inside the `fa_ext` kernel. Default on.
/// `FERROX_METAL_FA_MMA=0` selects the scalar `dot`+`simd_sum` predecessor at
/// d=64, which computes the same thing and is the A/B reference for both the
/// correctness diff and any timing comparison. At d=128 there is no scalar
/// `fa_ext`, so the same knob sends that width back to FA-vec instead.
fn gqa_prefill_fa_mma_enabled() -> bool {
    !matches!(
        std::env::var("FERROX_METAL_FA_MMA").ok().as_deref(),
        Some("0") | Some("false") | Some("off") | Some("scalar")
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_gqa_prefill_fa_vec(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    v: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_q: u32,
    kv_prefix_len: u32,
    softcap: f32,
) -> Result<(), MetalError> {
    // llama flash_attn_ext (MMA Q·Kᵀ + P·V, QN=8/C=64): default for d=64 and
    // d=128 prefill. Opt out: FERROX_METAL_FA_EXT=0. Legacy NQ=4:
    // FERROX_METAL_FA_NQ=4 (d=64 only).
    //
    // d=128 has no scalar `fa_ext` kernel, so `FERROX_METAL_FA_MMA=0` sends it
    // back to FA-vec rather than to a variant that does not exist.
    if (head_dim == 64 || (head_dim == 128 && gqa_prefill_fa_mma_enabled())) && n_q >= 8 {
        if gqa_prefill_fa_ext_d64_enabled() {
            return encode_gqa_prefill_fa_ext(
                encoder,
                device,
                q,
                k,
                v,
                out,
                n_heads,
                n_kv_heads,
                head_dim,
                n_q,
                kv_prefix_len,
                softcap,
            );
        }
        let use_nq4 = head_dim == 64
            && matches!(
                std::env::var("FERROX_METAL_FA_NQ").ok().as_deref(),
                Some("4") | Some("nq4") | Some("on") | Some("true")
            );
        if use_nq4 {
            return encode_gqa_prefill_fa_nq4_d64(
                encoder,
                device,
                q,
                k,
                v,
                out,
                n_heads,
                n_kv_heads,
                n_q,
                kv_prefix_len,
                softcap,
            );
        }
    }
    let pipe = match head_dim {
        64 => ensure_pipeline(
            device,
            GQA_PREFILL_FA_VEC_D64_KERNEL_SRC,
            "gqa_prefill_fa_vec_d64",
        )?,
        96 => ensure_pipeline(
            device,
            GQA_PREFILL_FA_VEC_D96_KERNEL_SRC,
            "gqa_prefill_fa_vec_d96",
        )?,
        128 => ensure_pipeline(device, GQA_PREFILL_FA_VEC_KERNEL_SRC, "gqa_prefill_fa_vec")?,
        256 => ensure_pipeline(
            device,
            GQA_PREFILL_FA_VEC_D256_KERNEL_SRC,
            "gqa_prefill_fa_vec_d256",
        )?,
        _ => return Err(MetalError::CommandFailed),
    };
    encoder.setComputePipelineState(&pipe.0);
    let tg = gqa_prefill_fa_vec_threadgroup_size(head_dim);
    let nsg = tg / 32;
    // Q[D] + NSG * (C=32 scores + D output)
    let tg_mem = ((head_dim + nsg * (32 + head_dim)) * 4) as usize;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut nh = n_heads;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nh as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut nkv = n_kv_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut nkv as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut hd = head_dim;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 6);
        let mut nq = n_q;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nq as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut prefix = kv_prefix_len;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut prefix as *mut u32 as *mut _).unwrap(),
            4,
            8,
        );
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads as usize,
            height: n_q as usize,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_gqa_fa_vec(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    v: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_start: u32,
    softcap: f32,
) -> Result<(), MetalError> {
    let pipe = match head_dim {
        256 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_D256_KERNEL_SRC,
            "gqa_decode_fa_vec_d256",
        )?,
        128 => ensure_pipeline(device, GQA_DECODE_FA_VEC_KERNEL_SRC, "gqa_decode_fa_vec")?,
        96 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_D96_KERNEL_SRC,
            "gqa_decode_fa_vec_d96",
        )?,
        64 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_D64_KERNEL_SRC,
            "gqa_decode_fa_vec_d64",
        )?,
        _ => return Err(MetalError::CommandFailed),
    };
    encoder.setComputePipelineState(&pipe.0);
    let tg = gqa_fa_vec_threadgroup_size(head_dim);
    let nsg = tg / 32;
    // Q[D] + NSG * (C=32 scores + D output)
    let tg_mem = ((head_dim + nsg * (32 + head_dim)) * 4) as usize;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut nh = n_heads;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nh as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut nkv = n_kv_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut nkv as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut hd = head_dim;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 6);
        let mut sl = seq_len;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sl as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut ks = kv_start;
        encoder.setBytes_length_atIndex(NonNull::new(&mut ks as *mut u32 as *mut _).unwrap(), 4, 8);
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_gqa(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    v: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_start: u32,
    attn_softcap: Option<f32>,
) -> Result<(), MetalError> {
    let softcap = attn_softcap.filter(|&c| c > 0.0).unwrap_or(0.0);
    // FA-vec supports SWA (`kv_start`) and softcap; use it whenever
    // head_dim has a specialized kernel.
    if metal_fa_vec_enabled() && gqa_fa_vec_supported(head_dim) {
        return encode_gqa_fa_vec(
            encoder, device, q, k, v, out, n_heads, n_kv_heads, head_dim, seq_len, kv_start,
            softcap,
        );
    }
    if head_dim > 256 {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, GQA_DECODE_KERNEL_SRC, "gqa_decode")?;
    encoder.setComputePipelineState(&pipe.0);
    let tg = gqa_decode_threadgroup_size(seq_len, head_dim);
    let nsg = tg / 32;
    // m[nsg] + s[nsg] + acc[nsg * head_dim]
    let tg_mem = ((2 * nsg + nsg * head_dim) * 4) as usize;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut nh = n_heads;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nh as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut nkv = n_kv_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut nkv as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut hd = head_dim;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 6);
        let mut sl = seq_len;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sl as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut ks = kv_start;
        encoder.setBytes_length_atIndex(NonNull::new(&mut ks as *mut u32 as *mut _).unwrap(), 4, 8);
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_gqa_prefill(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    v: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_q: u32,
    kv_prefix_len: u32,
    attn_softcap: Option<f32>,
) -> Result<(), MetalError> {
    let softcap = attn_softcap.filter(|&c| c > 0.0).unwrap_or(0.0);
    if metal_fa_vec_enabled() && gqa_prefill_fa_vec_supported(head_dim) {
        return encode_gqa_prefill_fa_vec(
            encoder,
            device,
            q,
            k,
            v,
            out,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
            softcap,
        );
    }
    let pipe = ensure_pipeline(device, GQA_PREFILL_KERNEL_SRC, "gqa_prefill")?;
    encoder.setComputePipelineState(&pipe.0);
    let max_causal = kv_prefix_len + n_q;
    let tg = gqa_prefill_threadgroup_size(max_causal.max(1), head_dim);
    // Prefill kernel: per-thread TG acc — m[tg]|s[tg]|acc[tg*head_dim].
    let tg_mem = ((2 * tg + tg * head_dim) * 4) as usize;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut nh = n_heads;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nh as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut nkv = n_kv_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut nkv as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut hd = head_dim;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 6);
        let mut nq = n_q;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nq as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut prefix = kv_prefix_len;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut prefix as *mut u32 as *mut _).unwrap(),
            4,
            8,
        );
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads as usize,
            height: n_q as usize,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Encode the optional QKV bias adds + QK-RMSNorms (CPU-path order:
/// bias → norm → RoPE). Per-head (`weight.len() == head_dim`) or
/// whole-vector (`weight.len() == q_rows` / `k_rows`, OLMoE). No-ops
/// when `extras` is empty. Single-token path used by decode.
#[allow(clippy::too_many_arguments)]
fn encode_attn_extras(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    extras: &AttnExtras<'_>,
    q_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    k_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    v_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    q_rows: usize,
    k_rows: usize,
    v_rows: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) -> Result<(), MetalError> {
    let resident = resident_attn_extras(device, extras)?;
    encode_attn_extras_batch(
        encoder, device, extras, q_buf, k_buf, v_buf, q_rows, k_rows, v_rows, n_heads, n_kv_heads,
        head_dim, 1, rms_eps, &resident,
    )
}

/// Prefill (`batch ≥ 1`) extras using already-resident bias/norm buffers.
#[allow(clippy::too_many_arguments)]
fn encode_attn_extras_batch(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    extras: &AttnExtras<'_>,
    q_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    k_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    v_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    q_rows: usize,
    k_rows: usize,
    v_rows: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    batch: usize,
    rms_eps: f32,
    resident: &AttnExtrasResident,
) -> Result<(), MetalError> {
    if batch == 0 {
        return Ok(());
    }
    if let Some(bb) = resident.q_bias.as_ref() {
        debug_assert_eq!(extras.q_bias.map(|b| b.len()), Some(q_rows));
        for t in 0..batch {
            encode_vec_add_at(
                encoder,
                device,
                q_buf,
                t * q_rows * 4,
                &bb.buffer,
                q_rows as u32,
            )?;
        }
    }
    if let Some(bb) = resident.k_bias.as_ref() {
        debug_assert_eq!(extras.k_bias.map(|b| b.len()), Some(k_rows));
        for t in 0..batch {
            encode_vec_add_at(
                encoder,
                device,
                k_buf,
                t * k_rows * 4,
                &bb.buffer,
                k_rows as u32,
            )?;
        }
    }
    if let Some(bb) = resident.v_bias.as_ref() {
        debug_assert_eq!(extras.v_bias.map(|b| b.len()), Some(v_rows));
        for t in 0..batch {
            encode_vec_add_at(
                encoder,
                device,
                v_buf,
                t * v_rows * 4,
                &bb.buffer,
                v_rows as u32,
            )?;
        }
    }
    if let (Some(w), Some(wb)) = (extras.q_norm, resident.q_norm.as_ref()) {
        if w.len() == head_dim {
            encode_rms_norm_per_head_batch(
                encoder,
                device,
                q_buf,
                &wb.buffer,
                n_heads as u32,
                head_dim as u32,
                batch as u32,
                rms_eps,
            )?;
        } else {
            debug_assert_eq!(w.len(), q_rows, "Q norm weight must be head_dim or q_rows");
            for t in 0..batch {
                let off = t * q_rows * 4;
                encode_rms_norm_at(
                    encoder,
                    device,
                    q_buf,
                    off,
                    &wb.buffer,
                    q_buf,
                    off,
                    q_rows as u32,
                    rms_eps,
                )?;
            }
        }
    }
    if let (Some(w), Some(wb)) = (extras.k_norm, resident.k_norm.as_ref()) {
        if w.len() == head_dim {
            encode_rms_norm_per_head_batch(
                encoder,
                device,
                k_buf,
                &wb.buffer,
                n_kv_heads as u32,
                head_dim as u32,
                batch as u32,
                rms_eps,
            )?;
        } else {
            debug_assert_eq!(w.len(), k_rows, "K norm weight must be head_dim or k_rows");
            for t in 0..batch {
                let off = t * k_rows * 4;
                encode_rms_norm_at(
                    encoder,
                    device,
                    k_buf,
                    off,
                    &wb.buffer,
                    k_buf,
                    off,
                    k_rows as u32,
                    rms_eps,
                )?;
            }
        }
    }
    Ok(())
}

struct AttnExtrasResident {
    q_bias: Option<std::sync::Arc<ResidentF32Buffer>>,
    k_bias: Option<std::sync::Arc<ResidentF32Buffer>>,
    v_bias: Option<std::sync::Arc<ResidentF32Buffer>>,
    q_norm: Option<std::sync::Arc<ResidentF32Buffer>>,
    k_norm: Option<std::sync::Arc<ResidentF32Buffer>>,
}

fn resident_attn_extras(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    extras: &AttnExtras<'_>,
) -> Result<AttnExtrasResident, MetalError> {
    Ok(AttnExtrasResident {
        q_bias: extras
            .q_bias
            .map(|b| resident_f32_buffer(device, b))
            .transpose()?,
        k_bias: extras
            .k_bias
            .map(|b| resident_f32_buffer(device, b))
            .transpose()?,
        v_bias: extras
            .v_bias
            .map(|b| resident_f32_buffer(device, b))
            .transpose()?,
        q_norm: extras
            .q_norm
            .map(|b| resident_f32_buffer(device, b))
            .transpose()?,
        k_norm: extras
            .k_norm
            .map(|b| resident_f32_buffer(device, b))
            .transpose()?,
    })
}

/// Fused Q/K/V matvec → RoPE → KV append → GQA → O matvec.
/// Returns the O-projection output on the host. Updates `kv.seq_len`.
#[allow(clippy::too_many_arguments)]
pub fn launch_decode_attn_block(
    x: &[f32],
    q_launch: &MatvecLaunch<'_>,
    k_launch: &MatvecLaunch<'_>,
    v_launch: &MatvecLaunch<'_>,
    o_launch: &MatvecLaunch<'_>,
    kv: &mut MetalKvBuffers,
    n_heads: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    pos: usize,
    extras: &AttnExtras<'_>,
    rms_eps: f32,
) -> Result<Vec<f32>, MetalError> {
    let head_dim = kv.head_dim;
    let n_kv_heads = kv.n_kv_heads;
    assert_eq!(q_launch.rows, n_heads * head_dim);
    assert_eq!(k_launch.rows, n_kv_heads * head_dim);
    assert_eq!(v_launch.rows, n_kv_heads * head_dim);
    assert_eq!(o_launch.rows, n_heads * head_dim);
    assert_eq!(
        pos, kv.seq_len,
        "decode pos must equal current Metal KV length"
    );
    if kv.seq_len >= kv.capacity {
        return Err(MetalError::CommandFailed);
    }
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), head_dim / 2);
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let x_buf = upload_f32(device, x)?;
    let q_w = resident_weight_buffer(device, q_launch.weights)?;
    let k_w = resident_weight_buffer(device, k_launch.weights)?;
    let v_w = resident_weight_buffer(device, v_launch.weights)?;
    let o_w = resident_weight_buffer(device, o_launch.weights)?;

    let q_buf = alloc_f32_buffer(device, q_launch.rows)?;
    let k_buf = alloc_f32_buffer(device, k_launch.rows)?;
    let v_buf = alloc_f32_buffer(device, v_launch.rows)?;
    let attn_buf = alloc_f32_buffer(device, n_heads * head_dim)?;
    let o_buf = alloc_f32_buffer(device, o_launch.rows)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    encode_matvec(&encoder, device, q_launch, &q_w, &x_buf, &q_buf)?;
    encode_matvec(&encoder, device, k_launch, &k_w, &x_buf, &k_buf)?;
    encode_matvec(&encoder, device, v_launch, &v_w, &x_buf, &v_buf)?;
    encode_attn_extras(
        &encoder,
        device,
        extras,
        &q_buf,
        &k_buf,
        &v_buf,
        q_launch.rows,
        k_launch.rows,
        v_launch.rows,
        n_heads,
        n_kv_heads,
        head_dim,
        rms_eps,
    )?;

    encode_rope(
        &encoder,
        device,
        rope_layout,
        &q_buf,
        n_heads as u32,
        head_dim as u32,
        rope_theta,
        pos as u32,
        ff_buf.as_deref(),
    )?;
    encode_rope(
        &encoder,
        device,
        rope_layout,
        &k_buf,
        n_kv_heads as u32,
        head_dim as u32,
        rope_theta,
        pos as u32,
        ff_buf.as_deref(),
    )?;

    let token_elems = (n_kv_heads * head_dim) as u32;
    let offset = (kv.seq_len * n_kv_heads * head_dim) as u32;
    encode_kv_store_append(
        &encoder,
        device,
        &k_buf,
        kv,
        KvPlane::K,
        offset,
        token_elems,
    )?;
    encode_kv_store_append(
        &encoder,
        device,
        &v_buf,
        kv,
        KvPlane::V,
        offset,
        token_elems,
    )?;

    let new_seq = (kv.seq_len + 1) as u32;
    encode_gqa_with_kv(
        &encoder,
        device,
        &q_buf,
        kv,
        &attn_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        new_seq,
        0,
        extras.attn_logit_softcap,
    )?;

    // O matvec reads attn_buf as activation `x`.
    encode_matvec(&encoder, device, o_launch, &o_w, &attn_buf, &o_buf)?;

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    kv.seq_len += 1;

    let out_ptr = o_buf.contents();
    let out = unsafe {
        std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, o_launch.rows).to_vec()
    };
    Ok(out)
}

/// MoE pre-FFN on one CB: attn RMSNorm → QKV→RoPE→KV→GQA→O → residual →
/// FFN RMSNorm. Returns `(updated_hidden, ffn_normed)` for host routing +
/// [`crate::gpu::launch_moe_topk_swiglu`]. One wait replaces separate attn
/// download + host norms.
#[allow(clippy::too_many_arguments)]
pub fn launch_decode_moe_attn_ffn_pre(
    hidden: &[f32],
    attn_norm_w: &[f32],
    q_launch: &MatvecLaunch<'_>,
    k_launch: &MatvecLaunch<'_>,
    v_launch: &MatvecLaunch<'_>,
    o_launch: &MatvecLaunch<'_>,
    kv: &mut MetalKvBuffers,
    ffn_norm_w: &[f32],
    n_heads: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    pos: usize,
    rms_eps: f32,
    extras: &AttnExtras<'_>,
) -> Result<(Vec<f32>, Vec<f32>), MetalError> {
    let head_dim = kv.head_dim;
    let n_kv_heads = kv.n_kv_heads;
    let hidden_dim = hidden.len();
    assert_eq!(attn_norm_w.len(), hidden_dim);
    assert_eq!(ffn_norm_w.len(), hidden_dim);
    assert_eq!(q_launch.rows, n_heads * head_dim);
    assert_eq!(k_launch.rows, n_kv_heads * head_dim);
    assert_eq!(v_launch.rows, n_kv_heads * head_dim);
    assert_eq!(o_launch.rows, hidden_dim);
    assert_eq!(pos, kv.seq_len);
    if kv.seq_len >= kv.capacity {
        return Err(MetalError::CommandFailed);
    }
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), head_dim / 2);
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let h_buf = upload_f32(device, hidden)?;
    let attn_nw = resident_f32_buffer(device, attn_norm_w)?;
    let ffn_nw = resident_f32_buffer(device, ffn_norm_w)?;
    let x_buf = alloc_f32_buffer(device, hidden_dim)?;
    let x2_buf = alloc_f32_buffer(device, hidden_dim)?;

    let q_w = resident_weight_buffer(device, q_launch.weights)?;
    let k_w = resident_weight_buffer(device, k_launch.weights)?;
    let v_w = resident_weight_buffer(device, v_launch.weights)?;
    let o_w = resident_weight_buffer(device, o_launch.weights)?;

    let q_buf = alloc_f32_buffer(device, q_launch.rows)?;
    let k_buf = alloc_f32_buffer(device, k_launch.rows)?;
    let v_buf = alloc_f32_buffer(device, v_launch.rows)?;
    let attn_buf = alloc_f32_buffer(device, n_heads * head_dim)?;
    let o_buf = alloc_f32_buffer(device, o_launch.rows)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    encode_rms_norm(
        &encoder,
        device,
        &h_buf,
        &attn_nw.buffer,
        &x_buf,
        hidden_dim as u32,
        rms_eps,
    )?;
    encode_matvec(&encoder, device, q_launch, &q_w, &x_buf, &q_buf)?;
    encode_matvec(&encoder, device, k_launch, &k_w, &x_buf, &k_buf)?;
    encode_matvec(&encoder, device, v_launch, &v_w, &x_buf, &v_buf)?;
    encode_attn_extras(
        &encoder,
        device,
        extras,
        &q_buf,
        &k_buf,
        &v_buf,
        q_launch.rows,
        k_launch.rows,
        v_launch.rows,
        n_heads,
        n_kv_heads,
        head_dim,
        rms_eps,
    )?;
    encode_rope(
        &encoder,
        device,
        rope_layout,
        &q_buf,
        n_heads as u32,
        head_dim as u32,
        rope_theta,
        pos as u32,
        ff_buf.as_deref(),
    )?;
    encode_rope(
        &encoder,
        device,
        rope_layout,
        &k_buf,
        n_kv_heads as u32,
        head_dim as u32,
        rope_theta,
        pos as u32,
        ff_buf.as_deref(),
    )?;
    let token_elems = (n_kv_heads * head_dim) as u32;
    let offset = (kv.seq_len * n_kv_heads * head_dim) as u32;
    encode_kv_store_append(
        &encoder,
        device,
        &k_buf,
        kv,
        KvPlane::K,
        offset,
        token_elems,
    )?;
    encode_kv_store_append(
        &encoder,
        device,
        &v_buf,
        kv,
        KvPlane::V,
        offset,
        token_elems,
    )?;
    let new_seq = (kv.seq_len + 1) as u32;
    encode_gqa_with_kv(
        &encoder,
        device,
        &q_buf,
        kv,
        &attn_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        new_seq,
        0,
        extras.attn_logit_softcap,
    )?;
    encode_matvec(&encoder, device, o_launch, &o_w, &attn_buf, &o_buf)?;
    // h += o, then x2 = rms_norm(h) * ffn_γ
    encode_add_rms_norm(
        &encoder,
        device,
        &h_buf,
        &o_buf,
        &ffn_nw.buffer,
        &x2_buf,
        hidden_dim as u32,
        rms_eps,
    )?;

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    kv.seq_len += 1;

    let h_ptr = h_buf.contents();
    let x2_ptr = x2_buf.contents();
    let new_h =
        unsafe { std::slice::from_raw_parts(h_ptr.as_ptr() as *const f32, hidden_dim).to_vec() };
    let ffn_normed =
        unsafe { std::slice::from_raw_parts(x2_ptr.as_ptr() as *const f32, hidden_dim).to_vec() };
    Ok((new_h, ffn_normed))
}

/// Retained Metal buffers for MoE decode so residual stays on GPU across
/// layers. With packed-id MoE, host never sees activations mid-layer.
struct MoeDecodeScratch {
    h: Retained<ProtocolObject<dyn MTLBuffer>>,
    x_attn: Retained<ProtocolObject<dyn MTLBuffer>>,
    x2: Retained<ProtocolObject<dyn MTLBuffer>>,
    q: Retained<ProtocolObject<dyn MTLBuffer>>,
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn: Retained<ProtocolObject<dyn MTLBuffer>>,
    o: Retained<ProtocolObject<dyn MTLBuffer>>,
    router: Retained<ProtocolObject<dyn MTLBuffer>>,
    ids: Retained<ProtocolObject<dyn MTLBuffer>>,
    route: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Pre-SiLU gate projection (unfused MoE matvec_id).
    gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Pre-SiLU up projection (unfused MoE matvec_id).
    up: Retained<ProtocolObject<dyn MTLBuffer>>,
    act: Retained<ProtocolObject<dyn MTLBuffer>>,
    expert_out: Retained<ProtocolObject<dyn MTLBuffer>>,
    moe_out: Retained<ProtocolObject<dyn MTLBuffer>>,
    logits: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    argmax_idx: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_dim: usize,
    q_rows: usize,
    k_rows: usize,
    ffn_rows: usize,
    top_k_cap: usize,
    n_router: usize,
    logits_cap: usize,
}

thread_local! {
    static MOE_SCRATCH: RefCell<Option<MoeDecodeScratch>> = const { RefCell::new(None) };
}

/// Drop resident MoE decode buffers (CPU fallback / end of generate).
pub fn moe_decode_clear() {
    MOE_SCRATCH.with(|s| {
        *s.borrow_mut() = None;
    });
}

/// Ensure MoE decode scratch exists for `hidden_dim` (no host upload).
pub fn moe_decode_ensure(hidden_dim: usize) -> Result<(), MetalError> {
    let shared = shared_metal()?;
    let device = &shared.device;
    MOE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let need_new = match slot.as_ref() {
            None => true,
            Some(s) => s.hidden_dim != hidden_dim,
        };
        if need_new {
            // Caps match OLMoE-class decode (top-k≤8, ffn≤8×hidden, ≤256 experts).
            // Grown on demand in phase-1/2 if a layer exceeds them.
            let q_rows = hidden_dim * 2;
            let k_rows = hidden_dim;
            let ffn_rows = hidden_dim * 8;
            let top_k_cap = 8;
            let n_router = 256;
            *slot = Some(MoeDecodeScratch {
                h: alloc_f32_buffer(device, hidden_dim)?,
                x_attn: alloc_f32_buffer(device, hidden_dim)?,
                x2: alloc_f32_buffer(device, hidden_dim)?,
                q: alloc_f32_buffer(device, q_rows)?,
                k: alloc_f32_buffer(device, k_rows)?,
                v: alloc_f32_buffer(device, k_rows)?,
                attn: alloc_f32_buffer(device, q_rows)?,
                o: alloc_f32_buffer(device, hidden_dim)?,
                router: alloc_f32_buffer(device, n_router)?,
                ids: alloc_u32_buffer(device, top_k_cap)?,
                route: alloc_f32_buffer(device, top_k_cap)?,
                gate: alloc_f32_buffer(device, top_k_cap * ffn_rows)?,
                up: alloc_f32_buffer(device, top_k_cap * ffn_rows)?,
                act: alloc_f32_buffer(device, top_k_cap * ffn_rows)?,
                expert_out: alloc_f32_buffer(device, top_k_cap * hidden_dim)?,
                moe_out: alloc_f32_buffer(device, hidden_dim)?,
                logits: None,
                argmax_idx: alloc_u32_buffer(device, 1)?,
                hidden_dim,
                q_rows,
                k_rows,
                ffn_rows,
                top_k_cap,
                n_router,
                logits_cap: 0,
            });
        }
        Ok(())
    })
}

/// Seed residual `h` from host hidden (call once before the MoE layer loop).
pub fn moe_decode_seed(hidden: &[f32]) -> Result<(), MetalError> {
    moe_decode_ensure(hidden.len())?;
    MOE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let scratch = slot.as_mut().ok_or(MetalError::CommandFailed)?;
        let hidden_dim = scratch.hidden_dim;
        let dst = scratch.h.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(hidden.as_ptr(), dst.as_ptr() as *mut f32, hidden_dim);
        }
        Ok(())
    })
}

/// Download residual hidden. Keeps scratch buffers for the next token
/// (`moe_decode_seed` overwrites `h`). `None` if never seeded.
pub fn moe_decode_take_hidden() -> Option<Vec<f32>> {
    MOE_SCRATCH.with(|cell| {
        let slot = cell.borrow();
        let scratch = slot.as_ref()?;
        let ptr = scratch.h.contents();
        Some(unsafe {
            std::slice::from_raw_parts(ptr.as_ptr() as *const f32, scratch.hidden_dim).to_vec()
        })
    })
}

fn moe_scratch_ensure_caps(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    scratch: &mut MoeDecodeScratch,
    q_rows: usize,
    k_rows: usize,
    ffn_rows: usize,
    top_k: usize,
    n_router: usize,
) -> Result<(), MetalError> {
    if q_rows > scratch.q_rows {
        scratch.q = alloc_f32_buffer(device, q_rows)?;
        scratch.attn = alloc_f32_buffer(device, q_rows)?;
        scratch.q_rows = q_rows;
    }
    if k_rows > scratch.k_rows {
        scratch.k = alloc_f32_buffer(device, k_rows)?;
        scratch.v = alloc_f32_buffer(device, k_rows)?;
        scratch.k_rows = k_rows;
    }
    if n_router > scratch.n_router {
        scratch.router = alloc_f32_buffer(device, n_router)?;
        scratch.n_router = n_router;
    }
    if ffn_rows > scratch.ffn_rows || top_k > scratch.top_k_cap {
        let fk = ffn_rows.max(scratch.ffn_rows);
        let tk = top_k.max(scratch.top_k_cap);
        scratch.gate = alloc_f32_buffer(device, tk * fk)?;
        scratch.up = alloc_f32_buffer(device, tk * fk)?;
        scratch.act = alloc_f32_buffer(device, tk * fk)?;
        scratch.expert_out = alloc_f32_buffer(device, tk * scratch.hidden_dim)?;
        scratch.ids = alloc_u32_buffer(device, tk)?;
        scratch.route = alloc_f32_buffer(device, tk)?;
        scratch.ffn_rows = fk;
        scratch.top_k_cap = tk;
    }
    Ok(())
}

fn moe_scratch_ensure_logits(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    scratch: &mut MoeDecodeScratch,
    vocab: usize,
) -> Result<(), MetalError> {
    if vocab > scratch.logits_cap {
        scratch.logits = Some(alloc_f32_buffer(device, vocab)?);
        scratch.logits_cap = vocab;
    }
    Ok(())
}

/// Phase 1 (llama mul_mat_id graph style): on-device
/// `rms_attn → QKV→RoPE→KV→GQA→O → h+=o → rms_ffn → router`.
/// Downloads **only** router logits. Residual + FFN-normed stay resident.
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_decode_pre(
    attn_norm_w: &[f32],
    q_launch: &MatvecLaunch<'_>,
    k_launch: &MatvecLaunch<'_>,
    v_launch: &MatvecLaunch<'_>,
    o_launch: &MatvecLaunch<'_>,
    kv: &mut MetalKvBuffers,
    ffn_norm_w: &[f32],
    router_launch: &MatvecLaunch<'_>,
    n_heads: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    pos: usize,
    rms_eps: f32,
    extras: &AttnExtras<'_>,
) -> Result<Vec<f32>, MetalError> {
    let head_dim = kv.head_dim;
    let n_kv_heads = kv.n_kv_heads;
    let hidden_dim = attn_norm_w.len();
    assert_eq!(ffn_norm_w.len(), hidden_dim);
    assert_eq!(q_launch.rows, n_heads * head_dim);
    assert_eq!(k_launch.rows, n_kv_heads * head_dim);
    assert_eq!(v_launch.rows, n_kv_heads * head_dim);
    assert_eq!(o_launch.rows, hidden_dim);
    assert_eq!(pos, kv.seq_len);
    if kv.seq_len >= kv.capacity {
        return Err(MetalError::CommandFailed);
    }
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), head_dim / 2);
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    MOE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let scratch = slot.as_mut().ok_or(MetalError::CommandFailed)?;
        if scratch.hidden_dim != hidden_dim {
            return Err(MetalError::CommandFailed);
        }
        moe_scratch_ensure_caps(
            device,
            scratch,
            q_launch.rows,
            k_launch.rows,
            1, // ffn sized in experts phase
            1,
            router_launch.rows,
        )?;

        let attn_nw = resident_f32_buffer(device, attn_norm_w)?;
        let ffn_nw = resident_f32_buffer(device, ffn_norm_w)?;
        let q_w = resident_weight_buffer(device, q_launch.weights)?;
        let k_w = resident_weight_buffer(device, k_launch.weights)?;
        let v_w = resident_weight_buffer(device, v_launch.weights)?;
        let o_w = resident_weight_buffer(device, o_launch.weights)?;
        let r_w = resident_weight_buffer(device, router_launch.weights)?;
        let ff_buf = match freq_factors {
            Some(ff) => Some(upload_f32(device, ff)?),
            None => None,
        };

        let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?;

        encode_rms_norm(
            &encoder,
            device,
            &scratch.h,
            &attn_nw.buffer,
            &scratch.x_attn,
            hidden_dim as u32,
            rms_eps,
        )?;
        encode_matvec(
            &encoder,
            device,
            q_launch,
            &q_w,
            &scratch.x_attn,
            &scratch.q,
        )?;
        encode_matvec(
            &encoder,
            device,
            k_launch,
            &k_w,
            &scratch.x_attn,
            &scratch.k,
        )?;
        encode_matvec(
            &encoder,
            device,
            v_launch,
            &v_w,
            &scratch.x_attn,
            &scratch.v,
        )?;
        encode_attn_extras(
            &encoder,
            device,
            extras,
            &scratch.q,
            &scratch.k,
            &scratch.v,
            q_launch.rows,
            k_launch.rows,
            v_launch.rows,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_eps,
        )?;
        encode_rope(
            &encoder,
            device,
            rope_layout,
            &scratch.q,
            n_heads as u32,
            head_dim as u32,
            rope_theta,
            pos as u32,
            ff_buf.as_deref(),
        )?;
        encode_rope(
            &encoder,
            device,
            rope_layout,
            &scratch.k,
            n_kv_heads as u32,
            head_dim as u32,
            rope_theta,
            pos as u32,
            ff_buf.as_deref(),
        )?;
        let token_elems = (n_kv_heads * head_dim) as u32;
        let offset = (kv.seq_len * n_kv_heads * head_dim) as u32;
        encode_kv_store_append(
            &encoder,
            device,
            &scratch.k,
            kv,
            KvPlane::K,
            offset,
            token_elems,
        )?;
        encode_kv_store_append(
            &encoder,
            device,
            &scratch.v,
            kv,
            KvPlane::V,
            offset,
            token_elems,
        )?;
        let new_seq = (kv.seq_len + 1) as u32;
        encode_gqa_with_kv(
            &encoder,
            device,
            &scratch.q,
            kv,
            &scratch.attn,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            new_seq,
            0,
            extras.attn_logit_softcap,
        )?;
        encode_matvec(&encoder, device, o_launch, &o_w, &scratch.attn, &scratch.o)?;
        encode_add_rms_norm(
            &encoder,
            device,
            &scratch.h,
            &scratch.o,
            &ffn_nw.buffer,
            &scratch.x2,
            hidden_dim as u32,
            rms_eps,
        )?;
        encode_matvec(
            &encoder,
            device,
            router_launch,
            &r_w,
            &scratch.x2,
            &scratch.router,
        )?;

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        kv.seq_len += 1;

        let ptr = scratch.router.contents();
        let logits = unsafe {
            std::slice::from_raw_parts(ptr.as_ptr() as *const f32, router_launch.rows).to_vec()
        };
        Ok(logits)
    })
}

/// Phase 2: batched MoE SwiGLU on resident FFN-normed `x2`, then `h += moe`.
/// Prefers the Q4_0 top-k kernels (≤8 experts); otherwise falls back to
/// host [`crate::gpu::launch_moe_topk_swiglu`] + upload/add (slower).
pub fn launch_moe_decode_experts(experts: &[MoeExpertLaunch<'_>]) -> Result<(), MetalError> {
    if experts.is_empty() {
        return Ok(());
    }
    let hidden = experts[0].down.rows;
    let ffn = experts[0].gate.rows;
    let q4_0_batched = experts.len() <= 8
        && experts.iter().all(|ex| {
            ex.gate.fn_name == "q4_0_matvec"
                && ex.up.fn_name == "q4_0_matvec"
                && ex.down.fn_name == "q4_0_matvec"
                && ex.gate.block_bytes == 18
                && ex.up.block_bytes == 18
                && ex.down.block_bytes == 18
        });

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    MOE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let scratch = slot.as_mut().ok_or(MetalError::CommandFailed)?;
        if scratch.hidden_dim != hidden {
            return Err(MetalError::CommandFailed);
        }

        if q4_0_batched {
            moe_scratch_ensure_caps(
                device,
                scratch,
                scratch.q_rows,
                scratch.k_rows,
                ffn,
                experts.len(),
                scratch.n_router,
            )?;
            let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(MetalError::CommandFailed)?;
            encode_q4_0_moe_topk(
                &encoder,
                device,
                &scratch.x2,
                experts,
                &scratch.act,
                &scratch.expert_out,
                &scratch.moe_out,
            )?;
            encode_vec_add(
                &encoder,
                device,
                &scratch.h,
                &scratch.moe_out,
                hidden as u32,
            )?;
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            return Ok(());
        }

        // Generic path: download x2, run existing fuse, upload + add.
        let x2 = unsafe {
            std::slice::from_raw_parts(scratch.x2.contents().as_ptr() as *const f32, hidden)
                .to_vec()
        };
        drop(slot);
        let moe = crate::gpu::launch_moe_topk_swiglu(&x2, experts)?;
        MOE_SCRATCH.with(|cell| {
            let slot = cell.borrow();
            let scratch = slot.as_ref().ok_or(MetalError::CommandFailed)?;
            let moe_buf = upload_f32(device, &moe)?;
            let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(MetalError::CommandFailed)?;
            encode_vec_add(&encoder, device, &scratch.h, &moe_buf, hidden as u32)?;
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            Ok(())
        })
    })
}

/// One MoE layer's launches for [`launch_moe_decode_stack`].
pub struct MoeLayerMetal<'a> {
    pub attn_norm_w: &'a [f32],
    pub ffn_norm_w: &'a [f32],
    pub q: MatvecLaunch<'a>,
    pub k: MatvecLaunch<'a>,
    pub v: MatvecLaunch<'a>,
    pub o: MatvecLaunch<'a>,
    pub router: MatvecLaunch<'a>,
    pub packed: MoePackedQ4<'a>,
    pub extras: AttnExtras<'a>,
}

/// Pre-bound MTLBuffers for one MoE layer (llama: bind weights once).
struct MoeLayerResident {
    attn_key: usize,
    attn_nw: std::sync::Arc<ResidentF32Buffer>,
    ffn_nw: std::sync::Arc<ResidentF32Buffer>,
    q_w: std::sync::Arc<ResidentWeightBuffer>,
    k_w: std::sync::Arc<ResidentWeightBuffer>,
    v_w: std::sync::Arc<ResidentWeightBuffer>,
    o_w: std::sync::Arc<ResidentWeightBuffer>,
    r_w: std::sync::Arc<ResidentWeightBuffer>,
}

thread_local! {
    /// Hoisted per-layer QKV/router/norm buffers for the MoE stack.
    static TL_MOE_LAYER_RESIDENT: RefCell<Vec<Option<MoeLayerResident>>> =
        const { RefCell::new(Vec::new()) };
}

fn moe_layer_resident(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    layer_idx: usize,
    layer: &MoeLayerMetal<'_>,
) -> Result<MoeLayerResident, MetalError> {
    let attn_key = layer.attn_norm_w.as_ptr() as usize;
    if let Some(hit) = TL_MOE_LAYER_RESIDENT.with(|c| {
        c.borrow().get(layer_idx).and_then(|slot| {
            slot.as_ref()
                .filter(|r| r.attn_key == attn_key)
                .map(|r| MoeLayerResident {
                    attn_key: r.attn_key,
                    attn_nw: r.attn_nw.clone(),
                    ffn_nw: r.ffn_nw.clone(),
                    q_w: r.q_w.clone(),
                    k_w: r.k_w.clone(),
                    v_w: r.v_w.clone(),
                    o_w: r.o_w.clone(),
                    r_w: r.r_w.clone(),
                })
        })
    }) {
        return Ok(hit);
    }
    let bound = MoeLayerResident {
        attn_key,
        attn_nw: resident_f32_buffer(device, layer.attn_norm_w)?,
        ffn_nw: resident_f32_buffer(device, layer.ffn_norm_w)?,
        q_w: resident_weight_buffer(device, layer.q.weights)?,
        k_w: resident_weight_buffer(device, layer.k.weights)?,
        v_w: resident_weight_buffer(device, layer.v.weights)?,
        o_w: resident_weight_buffer(device, layer.o.weights)?,
        r_w: resident_weight_buffer(device, layer.router.weights)?,
    };
    TL_MOE_LAYER_RESIDENT.with(|c| {
        let mut v = c.borrow_mut();
        if v.len() <= layer_idx {
            v.resize_with(layer_idx + 1, || None);
        }
        v[layer_idx] = Some(MoeLayerResident {
            attn_key: bound.attn_key,
            attn_nw: bound.attn_nw.clone(),
            ffn_nw: bound.ffn_nw.clone(),
            q_w: bound.q_w.clone(),
            k_w: bound.k_w.clone(),
            v_w: bound.v_w.clone(),
            o_w: bound.o_w.clone(),
            r_w: bound.r_w.clone(),
        });
    });
    Ok(bound)
}

/// One MoE layer into a Concurrent encoder using llama-style [`MoeMemRanges`]
/// barriers (only on SRC↔DST / DST↔DST conflicts). Same shape as dense
/// [`launch_decode_dense_stack`] and llama `ggml_metal_op` + `mem_ranges`.
///
/// Fused Concurrent groups (fewer barriers): Q∥K∥V, extras+RoPE, KV stores.
#[allow(clippy::too_many_arguments)]
fn encode_moe_layer_fused(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut MoeMemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    scratch: &MoeDecodeScratch,
    layer_idx: usize,
    layer: &MoeLayerMetal<'_>,
    kv: &MetalKvBuffers,
    top_k: usize,
    norm_topk_prob: bool,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    ff_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    pos: usize,
    rms_eps: f32,
) -> Result<(), MetalError> {
    let bound = moe_layer_resident(device, layer_idx, layer)?;
    let attn_nw = &bound.attn_nw;
    let ffn_nw = &bound.ffn_nw;
    let q_w = &bound.q_w;
    let k_w = &bound.k_w;
    let v_w = &bound.v_w;
    let o_w = &bound.o_w;
    let r_w = &bound.r_w;

    // attn_norm: h → x_attn
    {
        let srcs = [scratch.h.as_ref()];
        let dsts = [scratch.x_attn.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_rms_norm(
            encoder,
            device,
            &scratch.h,
            &attn_nw.buffer,
            &scratch.x_attn,
            hidden_dim as u32,
            rms_eps,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    // Q∥K∥V — one Concurrent set (shared src, disjoint dsts).
    {
        let srcs = [scratch.x_attn.as_ref()];
        let dsts = [scratch.q.as_ref(), scratch.k.as_ref(), scratch.v.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_matvec(encoder, device, &layer.q, q_w, &scratch.x_attn, &scratch.q)?;
        encode_matvec(encoder, device, &layer.k, k_w, &scratch.x_attn, &scratch.k)?;
        encode_matvec(encoder, device, &layer.v, v_w, &scratch.x_attn, &scratch.v)?;
        mrs.end_op(&srcs, &dsts);
    }
    // extras + RoPE on q/k — fused group (one barrier before in-place chain).
    {
        let srcs = [scratch.q.as_ref(), scratch.k.as_ref(), scratch.v.as_ref()];
        let dsts = [scratch.q.as_ref(), scratch.k.as_ref(), scratch.v.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_attn_extras(
            encoder,
            device,
            &layer.extras,
            &scratch.q,
            &scratch.k,
            &scratch.v,
            layer.q.rows,
            layer.k.rows,
            layer.v.rows,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_eps,
        )?;
        // Concurrent: barrier before RoPE reads q/k/v written by extras.
        memory_barrier_resources(
            encoder,
            &[scratch.q.as_ref(), scratch.k.as_ref(), scratch.v.as_ref()],
        );
        encode_rope(
            encoder,
            device,
            rope_layout,
            &scratch.q,
            n_heads as u32,
            head_dim as u32,
            rope_theta,
            pos as u32,
            ff_buf,
        )?;
        encode_rope(
            encoder,
            device,
            rope_layout,
            &scratch.k,
            n_kv_heads as u32,
            head_dim as u32,
            rope_theta,
            pos as u32,
            ff_buf,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    let token_elems = (n_kv_heads * head_dim) as u32;
    let offset = (pos * n_kv_heads * head_dim) as u32;
    {
        let srcs = [scratch.k.as_ref(), scratch.v.as_ref()];
        let dsts = [kv.k.as_ref(), kv.v.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        // RoPE must complete before KV store (begin_op may already barrier).
        encode_kv_store_append(
            encoder,
            device,
            &scratch.k,
            kv,
            KvPlane::K,
            offset,
            token_elems,
        )?;
        encode_kv_store_append(
            encoder,
            device,
            &scratch.v,
            kv,
            KvPlane::V,
            offset,
            token_elems,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    {
        let srcs = [scratch.q.as_ref(), kv.k.as_ref(), kv.v.as_ref()];
        let dsts = [scratch.attn.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_gqa_with_kv(
            encoder,
            device,
            &scratch.q,
            kv,
            &scratch.attn,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            (pos + 1) as u32,
            0,
            layer.extras.attn_logit_softcap,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    {
        let srcs = [scratch.attn.as_ref()];
        let dsts = [scratch.o.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_matvec(encoder, device, &layer.o, o_w, &scratch.attn, &scratch.o)?;
        mrs.end_op(&srcs, &dsts);
    }
    {
        let srcs = [scratch.h.as_ref(), scratch.o.as_ref()];
        let dsts = [scratch.h.as_ref(), scratch.x2.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_add_rms_norm(
            encoder,
            device,
            &scratch.h,
            &scratch.o,
            &ffn_nw.buffer,
            &scratch.x2,
            hidden_dim as u32,
            rms_eps,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    {
        let srcs = [scratch.x2.as_ref()];
        let dsts = [scratch.router.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_matvec(
            encoder,
            device,
            &layer.router,
            r_w,
            &scratch.x2,
            &scratch.router,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    {
        let srcs = [scratch.router.as_ref()];
        let dsts = [scratch.ids.as_ref(), scratch.route.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_moe_topk_softmax(
            encoder,
            device,
            &scratch.router,
            &scratch.ids,
            &scratch.route,
            layer.router.rows as u32,
            top_k as u32,
            norm_topk_prob,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    let fused_gate_up = matches!(
        std::env::var("FERROX_METAL_MOE_FUSED_GATE_UP")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    );
    // Default: Concurrent gate∥up (Host B faster than gate→silu×up).
    // Opt into gate→silu×up with FERROX_METAL_MOE_GATE_THEN_SILU=1.
    let gate_then_silu = matches!(
        std::env::var("FERROX_METAL_MOE_GATE_THEN_SILU")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    );
    if fused_gate_up {
        {
            let srcs = [scratch.x2.as_ref(), scratch.ids.as_ref()];
            let dsts = [scratch.act.as_ref()];
            mrs.begin_op(encoder, &srcs, &dsts);
            encode_q4_0_moe_gate_up_silu_fused(
                encoder,
                device,
                &scratch.x2,
                &layer.packed,
                &scratch.ids,
                &scratch.act,
                top_k as u32,
                1,
            )?;
            mrs.end_op(&srcs, &dsts);
        }
        {
            let srcs = [
                scratch.ids.as_ref(),
                scratch.route.as_ref(),
                scratch.act.as_ref(),
                scratch.h.as_ref(),
            ];
            let dsts = [scratch.expert_out.as_ref(), scratch.h.as_ref()];
            mrs.begin_op(encoder, &srcs, &dsts);
            encode_q4_0_moe_id_ex(
                encoder,
                device,
                &scratch.x2,
                &layer.packed,
                &scratch.ids,
                &scratch.route,
                &scratch.gate,
                &scratch.up,
                &scratch.act,
                &scratch.expert_out,
                &scratch.h,
                top_k as u32,
                1,
                true,
                true,
                true,
                false,
            )?;
            mrs.end_op(&srcs, &dsts);
        }
    } else if gate_then_silu {
        let srcs = [scratch.x2.as_ref(), scratch.ids.as_ref()];
        let dsts = [scratch.gate.as_ref(), scratch.act.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_q4_0_moe_gate_then_up_silu(
            encoder,
            device,
            &scratch.x2,
            &layer.packed,
            &scratch.ids,
            &scratch.gate,
            &scratch.act,
            top_k as u32,
            1,
        )?;
        mrs.end_op(&srcs, &dsts);
        let srcs = [
            scratch.ids.as_ref(),
            scratch.route.as_ref(),
            scratch.act.as_ref(),
            scratch.h.as_ref(),
        ];
        let dsts = [scratch.expert_out.as_ref(), scratch.h.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_q4_0_moe_id_ex(
            encoder,
            device,
            &scratch.x2,
            &layer.packed,
            &scratch.ids,
            &scratch.route,
            &scratch.gate,
            &scratch.up,
            &scratch.act,
            &scratch.expert_out,
            &scratch.h,
            top_k as u32,
            1,
            true,
            true,
            true,
            false,
        )?;
        mrs.end_op(&srcs, &dsts);
    } else {
        let srcs = [scratch.x2.as_ref(), scratch.ids.as_ref()];
        let dsts = [scratch.gate.as_ref(), scratch.up.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_q4_0_moe_gate_up_id(
            encoder,
            device,
            &scratch.x2,
            &layer.packed,
            &scratch.ids,
            &scratch.gate,
            &scratch.up,
            top_k as u32,
            1,
        )?;
        mrs.end_op(&srcs, &dsts);
        let srcs = [
            scratch.x2.as_ref(),
            scratch.ids.as_ref(),
            scratch.route.as_ref(),
            scratch.gate.as_ref(),
            scratch.up.as_ref(),
            scratch.h.as_ref(),
        ];
        let dsts = [
            scratch.act.as_ref(),
            scratch.expert_out.as_ref(),
            scratch.h.as_ref(),
        ];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_q4_0_moe_id(
            encoder,
            device,
            &scratch.x2,
            &layer.packed,
            &scratch.ids,
            &scratch.route,
            &scratch.gate,
            &scratch.up,
            &scratch.act,
            &scratch.expert_out,
            &scratch.h,
            top_k as u32,
            1,
            true,
            true,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    Ok(())
}

/// One MoE layer, one CB (llama graph style): attn → residual → ffn_norm →
/// F32 router → GPU softmax top-k → packed Q4_0 experts → residual.
/// Returns selected expert ids (for hotness accounting). Updates `kv.seq_len`.
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_decode_layer_fused(
    attn_norm_w: &[f32],
    q_launch: &MatvecLaunch<'_>,
    k_launch: &MatvecLaunch<'_>,
    v_launch: &MatvecLaunch<'_>,
    o_launch: &MatvecLaunch<'_>,
    kv: &mut MetalKvBuffers,
    ffn_norm_w: &[f32],
    router_launch: &MatvecLaunch<'_>,
    packed: &MoePackedQ4<'_>,
    top_k: usize,
    norm_topk_prob: bool,
    n_heads: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    pos: usize,
    rms_eps: f32,
    extras: &AttnExtras<'_>,
) -> Result<Vec<usize>, MetalError> {
    let layer = MoeLayerMetal {
        attn_norm_w,
        ffn_norm_w,
        q: MatvecLaunch {
            kernel_src: q_launch.kernel_src,
            fn_name: q_launch.fn_name,
            block_bytes: q_launch.block_bytes,
            block_elems: q_launch.block_elems,
            weights: q_launch.weights,
            rows: q_launch.rows,
            row_bytes: q_launch.row_bytes,
            rows_per_tg: q_launch.rows_per_tg,
        },
        k: MatvecLaunch {
            kernel_src: k_launch.kernel_src,
            fn_name: k_launch.fn_name,
            block_bytes: k_launch.block_bytes,
            block_elems: k_launch.block_elems,
            weights: k_launch.weights,
            rows: k_launch.rows,
            row_bytes: k_launch.row_bytes,
            rows_per_tg: k_launch.rows_per_tg,
        },
        v: MatvecLaunch {
            kernel_src: v_launch.kernel_src,
            fn_name: v_launch.fn_name,
            block_bytes: v_launch.block_bytes,
            block_elems: v_launch.block_elems,
            weights: v_launch.weights,
            rows: v_launch.rows,
            row_bytes: v_launch.row_bytes,
            rows_per_tg: v_launch.rows_per_tg,
        },
        o: MatvecLaunch {
            kernel_src: o_launch.kernel_src,
            fn_name: o_launch.fn_name,
            block_bytes: o_launch.block_bytes,
            block_elems: o_launch.block_elems,
            weights: o_launch.weights,
            rows: o_launch.rows,
            row_bytes: o_launch.row_bytes,
            rows_per_tg: o_launch.rows_per_tg,
        },
        router: MatvecLaunch {
            kernel_src: router_launch.kernel_src,
            fn_name: router_launch.fn_name,
            block_bytes: router_launch.block_bytes,
            block_elems: router_launch.block_elems,
            weights: router_launch.weights,
            rows: router_launch.rows,
            row_bytes: router_launch.row_bytes,
            rows_per_tg: router_launch.rows_per_tg,
        },
        packed: MoePackedQ4 {
            gate: packed.gate,
            up: packed.up,
            down: packed.down,
            gate_stride: packed.gate_stride,
            up_stride: packed.up_stride,
            down_stride: packed.down_stride,
            n_experts: packed.n_experts,
            ffn_rows: packed.ffn_rows,
            hidden_rows: packed.hidden_rows,
            gate_row_bytes: packed.gate_row_bytes,
            down_row_bytes: packed.down_row_bytes,
            gate_kind: packed.gate_kind,
            up_kind: packed.up_kind,
            down_kind: packed.down_kind,
        },
        extras: AttnExtras {
            q_bias: extras.q_bias,
            k_bias: extras.k_bias,
            v_bias: extras.v_bias,
            q_norm: extras.q_norm,
            k_norm: extras.k_norm,
            attn_logit_softcap: extras.attn_logit_softcap,
        },
    };
    let out = launch_moe_decode_stack(
        &[], // seeded externally
        std::slice::from_ref(&layer),
        std::slice::from_mut(kv),
        top_k,
        norm_topk_prob,
        n_heads,
        rope_layout,
        rope_theta,
        freq_factors,
        pos,
        rms_eps,
        None,
        None,
        false,
        true, // reuse existing scratch.h
        None,
    )?;
    Ok(out.1.into_iter().next().unwrap_or_default())
}

/// All MoE layers in **one** command buffer (one wait) — dense-stack
/// equivalent for OLMoE. Returns `(hidden_or_logits_or_argmax, per_layer_expert_ids)`.
/// When `reuse_scratch_h` is true, `hidden` is ignored and scratch `h`
/// from a prior [`moe_decode_seed`] is used. When `embd` is `Some`,
/// gathers the token row into `h` on-GPU (no host seed upload).
///
/// With `final_norm_w` + `output`, folds lm_head on-GPU. `argmax_only`
/// downloads a 1-element `vec![token_id as f32]` (same contract as dense).
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_decode_stack(
    hidden: &[f32],
    layers: &[MoeLayerMetal<'_>],
    kvs: &mut [MetalKvBuffers],
    top_k: usize,
    norm_topk_prob: bool,
    n_heads: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    pos: usize,
    rms_eps: f32,
    final_norm_w: Option<&[f32]>,
    output: Option<&MatvecLaunch<'_>>,
    argmax_only: bool,
    reuse_scratch_h: bool,
    embd: Option<&EmbdGatherMetal<'_>>,
) -> Result<(Vec<f32>, Vec<Vec<usize>>), MetalError> {
    assert!(!layers.is_empty());
    assert_eq!(layers.len(), kvs.len());
    assert!(top_k > 0 && top_k <= 8);
    let hidden_dim = layers[0].packed.hidden_rows;
    let head_dim = kvs[0].head_dim;
    let n_kv_heads = kvs[0].n_kv_heads;
    for kv in kvs.iter() {
        assert_eq!(kv.head_dim, head_dim);
        assert_eq!(kv.n_kv_heads, n_kv_heads);
        assert_eq!(pos, kv.seq_len);
        if kv.seq_len >= kv.capacity {
            return Err(MetalError::CommandFailed);
        }
    }

    if let Some(e) = embd {
        assert_eq!(e.n_cols, hidden_dim);
        assert!(e.token_id < e.rows);
        moe_decode_ensure(hidden_dim)?;
    } else if !reuse_scratch_h {
        moe_decode_seed(hidden)?;
    } else {
        moe_decode_ensure(hidden_dim)?;
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;
    let ff_resident = match freq_factors {
        Some(ff) => Some(resident_f32_buffer(device, ff)?),
        None => None,
    };

    MOE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let scratch = slot.as_mut().ok_or(MetalError::CommandFailed)?;
        if scratch.hidden_dim != hidden_dim {
            return Err(MetalError::CommandFailed);
        }
        let max_q = layers.iter().map(|l| l.q.rows).max().unwrap();
        let max_k = layers.iter().map(|l| l.k.rows).max().unwrap();
        let max_ffn = layers.iter().map(|l| l.packed.ffn_rows).max().unwrap();
        let max_router = layers.iter().map(|l| l.router.rows).max().unwrap();
        moe_scratch_ensure_caps(device, scratch, max_q, max_k, max_ffn, top_k, max_router)?;
        if let Some(out_l) = output {
            moe_scratch_ensure_logits(device, scratch, out_l.rows)?;
        }

        let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
        // llama.cpp / dense-stack: one Concurrent encoder for the full graph,
        // barriers only via MoeMemRanges (ggml_mem_ranges).
        let encoder = compute_encoder_concurrent(&cmd_buf)?;
        let mut mrs = MoeMemRanges::new();

        let _embd_w = if let Some(e) = embd {
            let w = resident_weight_buffer(device, e.weights)?;
            let srcs: [&ProtocolObject<dyn MTLBuffer>; 0] = [];
            let dsts = [scratch.h.as_ref()];
            mrs.begin_op(&encoder, &srcs, &dsts);
            encode_get_rows(
                &encoder,
                device,
                e.kind,
                &w,
                &scratch.h,
                e.row_bytes as u32,
                e.n_cols as u32,
                e.token_id as u32,
            )?;
            mrs.end_op(&srcs, &dsts);
            Some(w)
        } else {
            None
        };

        for (layer_idx, (layer, kv)) in layers.iter().zip(kvs.iter()).enumerate() {
            encode_moe_layer_fused(
                &encoder,
                &mut mrs,
                device,
                scratch,
                layer_idx,
                layer,
                kv,
                top_k,
                norm_topk_prob,
                n_heads,
                n_kv_heads,
                head_dim,
                hidden_dim,
                rope_layout,
                rope_theta,
                ff_resident.as_ref().map(|b| b.buffer.as_ref()),
                pos,
                rms_eps,
            )?;
        }

        // final_norm → optional lm_head → optional argmax (dense-stack parity).
        let (download_n, download_logits, download_argmax) = if let Some(fnw) = final_norm_w {
            assert_eq!(fnw.len(), hidden_dim);
            let fn_buf = resident_f32_buffer(device, fnw)?;
            {
                let srcs = [scratch.h.as_ref()];
                let dsts = [scratch.x_attn.as_ref()];
                mrs.begin_op(&encoder, &srcs, &dsts);
                encode_rms_norm(
                    &encoder,
                    device,
                    &scratch.h,
                    &fn_buf.buffer,
                    &scratch.x_attn,
                    hidden_dim as u32,
                    rms_eps,
                )?;
                mrs.end_op(&srcs, &dsts);
            }
            if let Some(out_l) = output {
                let logits = scratch.logits.as_ref().ok_or(MetalError::CommandFailed)?;
                assert_eq!(out_l.rows, scratch.logits_cap);
                let out_w = resident_weight_buffer(device, out_l.weights)?;
                {
                    let srcs = [scratch.x_attn.as_ref()];
                    let dsts = [logits.as_ref()];
                    mrs.begin_op(&encoder, &srcs, &dsts);
                    encode_matvec(&encoder, device, out_l, &out_w, &scratch.x_attn, logits)?;
                    mrs.end_op(&srcs, &dsts);
                }
                if argmax_only {
                    {
                        let srcs = [logits.as_ref()];
                        let dsts = [scratch.argmax_idx.as_ref()];
                        mrs.begin_op(&encoder, &srcs, &dsts);
                        encode_argmax(
                            &encoder,
                            device,
                            logits,
                            &scratch.argmax_idx,
                            out_l.rows as u32,
                        )?;
                        mrs.end_op(&srcs, &dsts);
                    }
                    (1, false, true)
                } else {
                    (out_l.rows, true, false)
                }
            } else {
                (hidden_dim, false, false)
            }
        } else {
            assert!(output.is_none(), "MoE stack lm_head requires final_norm");
            (hidden_dim, false, false)
        };

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        if std::env::var_os("FERROX_METAL_GPU_TIMING").is_some() {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static GPU_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let dt = cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime();
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let acc = GPU_NS.fetch_add((dt * 1e9) as u64, std::sync::atomic::Ordering::Relaxed)
                + (dt * 1e9) as u64;
            if n.is_multiple_of(32) {
                eprintln!(
                    "ferrox: metal gpu {:.3} ms/tok avg over {} (last {:.3} ms)",
                    (acc as f64 / n as f64) / 1e6,
                    n,
                    dt * 1e3
                );
            }
        }

        for kv in kvs.iter_mut() {
            kv.seq_len = pos + 1;
        }

        // Skip expert-id host download on the hot path (sync tax). Hotness
        // tracking can be re-enabled later via a side channel if needed.
        let all_ids = vec![Vec::new(); layers.len()];

        if download_argmax {
            let ptr = scratch.argmax_idx.contents();
            let idx = unsafe { *(ptr.as_ptr() as *const u32) as usize };
            return Ok((vec![idx as f32], all_ids));
        }

        let src: &ProtocolObject<dyn MTLBuffer> = if download_logits {
            scratch
                .logits
                .as_ref()
                .ok_or(MetalError::CommandFailed)?
                .as_ref()
        } else if final_norm_w.is_some() {
            scratch.x_attn.as_ref()
        } else {
            scratch.h.as_ref()
        };
        let n = if download_logits {
            download_n
        } else {
            hidden_dim
        };
        let out = unsafe {
            std::slice::from_raw_parts(src.contents().as_ptr() as *const f32, n).to_vec()
        };
        Ok((out, all_ids))
    })
}

/// Full dense decode layer on one CB: RMSNorm → QKV→RoPE→KV→GQA→O →
/// residual → RMSNorm → gate/up → SiLU×up → down → residual.
/// Returns the updated residual `hidden` on the host. Updates `kv.seq_len`.
#[allow(clippy::too_many_arguments)]
pub fn launch_decode_dense_layer(
    hidden: &[f32],
    attn_norm_w: &[f32],
    q_launch: &MatvecLaunch<'_>,
    k_launch: &MatvecLaunch<'_>,
    v_launch: &MatvecLaunch<'_>,
    o_launch: &MatvecLaunch<'_>,
    kv: &mut MetalKvBuffers,
    ffn_norm_w: &[f32],
    gate_launch: &MatvecLaunch<'_>,
    up_launch: &MatvecLaunch<'_>,
    down_launch: &MatvecLaunch<'_>,
    n_heads: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    pos: usize,
    rms_eps: f32,
    extras: &AttnExtras<'_>,
) -> Result<Vec<f32>, MetalError> {
    let head_dim = kv.head_dim;
    let n_kv_heads = kv.n_kv_heads;
    let hidden_dim = hidden.len();
    assert_eq!(attn_norm_w.len(), hidden_dim);
    assert_eq!(ffn_norm_w.len(), hidden_dim);
    assert_eq!(q_launch.rows, n_heads * head_dim);
    assert_eq!(k_launch.rows, n_kv_heads * head_dim);
    assert_eq!(v_launch.rows, n_kv_heads * head_dim);
    assert_eq!(o_launch.rows, hidden_dim);
    assert_eq!(down_launch.rows, hidden_dim);
    assert_eq!(gate_launch.rows, up_launch.rows);
    assert_eq!(
        pos, kv.seq_len,
        "decode pos must equal current Metal KV length"
    );
    if kv.seq_len >= kv.capacity {
        return Err(MetalError::CommandFailed);
    }
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), head_dim / 2);
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let h_buf = upload_f32(device, hidden)?;
    let attn_nw = resident_f32_buffer(device, attn_norm_w)?;
    let ffn_nw = resident_f32_buffer(device, ffn_norm_w)?;
    let x_buf = alloc_f32_buffer(device, hidden_dim)?;
    let x2_buf = alloc_f32_buffer(device, hidden_dim)?;

    let q_w = resident_weight_buffer(device, q_launch.weights)?;
    let k_w = resident_weight_buffer(device, k_launch.weights)?;
    let v_w = resident_weight_buffer(device, v_launch.weights)?;
    let o_w = resident_weight_buffer(device, o_launch.weights)?;
    let gate_w = resident_weight_buffer(device, gate_launch.weights)?;
    let up_w = resident_weight_buffer(device, up_launch.weights)?;
    let down_w = resident_weight_buffer(device, down_launch.weights)?;

    let q_buf = alloc_f32_buffer(device, q_launch.rows)?;
    let k_buf = alloc_f32_buffer(device, k_launch.rows)?;
    let v_buf = alloc_f32_buffer(device, v_launch.rows)?;
    let attn_buf = alloc_f32_buffer(device, n_heads * head_dim)?;
    let o_buf = alloc_f32_buffer(device, o_launch.rows)?;
    let gate_buf = alloc_f32_buffer(device, gate_launch.rows)?;
    let up_buf = alloc_f32_buffer(device, up_launch.rows)?;
    let act_buf = alloc_f32_buffer(device, gate_launch.rows)?;
    let down_buf = alloc_f32_buffer(device, down_launch.rows)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    encode_rms_norm(
        &encoder,
        device,
        &h_buf,
        &attn_nw.buffer,
        &x_buf,
        hidden_dim as u32,
        rms_eps,
    )?;
    encode_matvec(&encoder, device, q_launch, &q_w, &x_buf, &q_buf)?;
    encode_matvec(&encoder, device, k_launch, &k_w, &x_buf, &k_buf)?;
    encode_matvec(&encoder, device, v_launch, &v_w, &x_buf, &v_buf)?;
    encode_attn_extras(
        &encoder,
        device,
        extras,
        &q_buf,
        &k_buf,
        &v_buf,
        q_launch.rows,
        k_launch.rows,
        v_launch.rows,
        n_heads,
        n_kv_heads,
        head_dim,
        rms_eps,
    )?;

    encode_rope(
        &encoder,
        device,
        rope_layout,
        &q_buf,
        n_heads as u32,
        head_dim as u32,
        rope_theta,
        pos as u32,
        ff_buf.as_deref(),
    )?;
    encode_rope(
        &encoder,
        device,
        rope_layout,
        &k_buf,
        n_kv_heads as u32,
        head_dim as u32,
        rope_theta,
        pos as u32,
        ff_buf.as_deref(),
    )?;

    let token_elems = (n_kv_heads * head_dim) as u32;
    let offset = (kv.seq_len * n_kv_heads * head_dim) as u32;
    encode_kv_store_append(
        &encoder,
        device,
        &k_buf,
        kv,
        KvPlane::K,
        offset,
        token_elems,
    )?;
    encode_kv_store_append(
        &encoder,
        device,
        &v_buf,
        kv,
        KvPlane::V,
        offset,
        token_elems,
    )?;

    let new_seq = (kv.seq_len + 1) as u32;
    encode_gqa_with_kv(
        &encoder,
        device,
        &q_buf,
        kv,
        &attn_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        new_seq,
        0,
        extras.attn_logit_softcap,
    )?;
    encode_matvec(&encoder, device, o_launch, &o_w, &attn_buf, &o_buf)?;
    encode_add_rms_norm(
        &encoder,
        device,
        &h_buf,
        &o_buf,
        &ffn_nw.buffer,
        &x2_buf,
        hidden_dim as u32,
        rms_eps,
    )?;
    encode_matvec(&encoder, device, gate_launch, &gate_w, &x2_buf, &gate_buf)?;
    encode_matvec(&encoder, device, up_launch, &up_w, &x2_buf, &up_buf)?;
    encode_silu_mul(
        &encoder,
        device,
        &gate_buf,
        &up_buf,
        &act_buf,
        gate_launch.rows as u32,
    )?;
    encode_matvec(&encoder, device, down_launch, &down_w, &act_buf, &down_buf)?;
    encode_vec_add(&encoder, device, &h_buf, &down_buf, hidden_dim as u32)?;

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    kv.seq_len += 1;

    let out_ptr = h_buf.contents();
    Ok(unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, hidden_dim).to_vec() })
}

/// Optional attention epilogue ops applied between the QKV matvecs and
/// RoPE, in CPU-path order: bias add (Qwen2-family `qkv_bias`), then
/// QK-RMSNorm — per-head (Qwen3 / Gemma-3, `weight.len() == head_dim`)
/// or whole-vector (OLMoE, `weight.len() == n_heads|n_kv_heads * head_dim`).
/// `attn_logit_softcap` is applied inside GQA after score scaling
/// (Gemma-2); when set, FA-vec is skipped in favour of the legacy kernel
/// unless the FA-vec softcap path is enabled.
#[derive(Default)]
pub struct AttnExtras<'a> {
    pub q_bias: Option<&'a [f32]>,
    pub k_bias: Option<&'a [f32]>,
    pub v_bias: Option<&'a [f32]>,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub attn_logit_softcap: Option<f32>,
}

impl AttnExtras<'_> {
    pub fn is_empty(&self) -> bool {
        self.q_bias.is_none()
            && self.k_bias.is_none()
            && self.v_bias.is_none()
            && self.q_norm.is_none()
            && self.k_norm.is_none()
            && self.attn_logit_softcap.is_none()
    }
}

/// Per-layer launches + norms for [`launch_decode_dense_stack`].
pub struct DenseLayerMetal<'a> {
    pub attn_norm_w: &'a [f32],
    pub ffn_norm_w: &'a [f32],
    pub q: MatvecLaunch<'a>,
    pub k: MatvecLaunch<'a>,
    pub v: MatvecLaunch<'a>,
    pub o: MatvecLaunch<'a>,
    pub gate: MatvecLaunch<'a>,
    pub up: MatvecLaunch<'a>,
    pub down: MatvecLaunch<'a>,
    pub extras: AttnExtras<'a>,
    /// Per-layer RoPE base override (Gemma-3 SWA layers use
    /// `rope_theta_swa`); `None` = the stack-wide theta.
    pub rope_theta: Option<f32>,
    /// Sliding-window size for this layer (`None` = full causal).
    pub window: Option<usize>,
    /// Gemma post-attention / post-FFN sandwich norms, applied to the
    /// block output *before* the residual add.
    pub post_attn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
}

/// Optional on-GPU embedding gather at the start of
/// [`launch_decode_dense_stack`] (skips host `dequant_row` + upload).
pub struct EmbdGatherMetal<'a> {
    pub kind: EmbdKind,
    pub weights: &'a [u8],
    pub rows: usize,
    pub row_bytes: usize,
    pub n_cols: usize,
    pub token_id: usize,
}

/// All dense layers in **one** command buffer (one wait). Hidden stays on
/// GPU across layers — Crane-style residency for B=1 decode.
/// When `embd` is `Some`, gathers that token row into scratch `h` on-GPU
/// instead of copying a host-provided `hidden` slice.
/// When `final_norm_w` + `output` are provided, also runs final RMSNorm +
/// lm_head on-GPU. With `argmax_only`, runs argmax and returns a
/// **1-element** `vec![token_id as f32]`; otherwise downloads vocab logits.
///
/// Chunked multi-CB early-commit (llama `n_main` style) was tried on Host B
/// and regressed decode tok/s — see `…_multicb*` receipts; kept single CB.
#[allow(clippy::too_many_arguments)]
pub fn launch_decode_dense_stack(
    hidden: &[f32],
    layers: &[DenseLayerMetal<'_>],
    kvs: &mut [MetalKvBuffers],
    n_heads: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    pos: usize,
    rms_eps: f32,
    final_norm_w: Option<&[f32]>,
    output: Option<&MatvecLaunch<'_>>,
    argmax_only: bool,
    embd: Option<&EmbdGatherMetal<'_>>,
    gelu_ffn: bool,
) -> Result<Vec<f32>, MetalError> {
    assert_eq!(layers.len(), kvs.len());
    assert!(!layers.is_empty());
    let hidden_dim = match embd {
        Some(e) => e.n_cols,
        None => hidden.len(),
    };
    assert!(hidden_dim > 0);
    let head_dim = kvs[0].head_dim;
    let n_kv_heads = kvs[0].n_kv_heads;
    for kv in kvs.iter() {
        assert_eq!(kv.head_dim, head_dim);
        assert_eq!(kv.n_kv_heads, n_kv_heads);
        assert_eq!(pos, kv.seq_len);
        if kv.seq_len >= kv.capacity {
            return Err(MetalError::CommandFailed);
        }
    }
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), head_dim / 2);
    }

    let max_q = layers.iter().map(|l| l.q.rows).max().unwrap();
    let max_kv = layers.iter().map(|l| l.k.rows).max().unwrap();
    let max_gate = layers.iter().map(|l| l.gate.rows).max().unwrap();
    let attn_elems = n_heads * head_dim;
    let logits_rows = output.map(|o| o.rows);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let scratch_guard = borrow_decode_scratch(
        device,
        ScratchCaps {
            hidden: hidden_dim,
            max_q,
            max_kv,
            attn: attn_elems,
            max_gate,
            logits: logits_rows.unwrap_or(0),
        },
    )?;
    let scratch = scratch_guard.as_ref().expect("scratch just ensured");
    let h_buf = &scratch.h;
    if let Some(e) = embd {
        assert_eq!(e.n_cols, hidden_dim);
        assert!(e.token_id < e.rows);
        assert_eq!(e.weights.len(), e.rows * e.row_bytes);
        // Gather runs in the same CB below (after encoder create).
    } else {
        assert_eq!(hidden.len(), hidden_dim);
        copy_f32_into(h_buf, hidden);
    }
    let x_buf = &scratch.x;
    let x2_buf = &scratch.x2;
    let q_buf = &scratch.q;
    let k_buf = &scratch.k;
    let v_buf = &scratch.v;
    let attn_buf = &scratch.attn;
    let o_buf = &scratch.o;
    let gate_buf = &scratch.gate;
    let up_buf = &scratch.up;
    let act_buf = &scratch.act;
    let down_buf = &scratch.down;
    let logits_buf = scratch.logits.as_ref();
    let argmax_idx_buf = &scratch.argmax_idx;

    let ff_resident = match freq_factors {
        Some(ff) => Some(resident_f32_buffer(device, ff)?),
        None => None,
    };
    let ff_buf = ff_resident.as_ref().map(|b| b.buffer.as_ref());

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    // Sandwich (Gemma post-norms): use the default serial encoder. Concurrent
    // dispatch + in-place RMSNorm/deferred residuals was measured to diverge
    // from CPU on Gemma-2 B=1 decode while the serial prefill-shaped residual
    // path stays coherent. Non-sandwich (SmolLM2) keeps Concurrent for Q∥K∥V.
    let sandwich = layers
        .iter()
        .any(|l| l.post_attn_norm.is_some() || l.post_ffn_norm.is_some());
    let encoder = if sandwich {
        cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?
    } else {
        // llama.cpp concurrent encode: gate∥up and Q∥K∥V overlap
        compute_encoder_concurrent(&cmd_buf)?
    };

    let embd_resident = if let Some(e) = embd {
        let w = resident_weight_buffer(device, e.weights)?;
        encode_get_rows(
            &encoder,
            device,
            e.kind,
            &w,
            h_buf,
            e.row_bytes as u32,
            e.n_cols as u32,
            e.token_id as u32,
        )?;
        memory_barrier_buffers(&encoder);
        Some(w)
    } else {
        None
    };
    let _embd_resident = embd_resident;

    // Gemma sandwich (post_attn / post_ffn) must apply residuals eagerly —
    // same shape as the working prefill stack / CPU path. Deferred
    // `h += down` fused into the next layer's attn_norm matches SmolLM2
    // (no post-norms) but diverges for Gemma-2 Metal greedy (BOS loops /
    // `*` spam) even when GQA unit tests pass.
    // `sandwich` was computed above (also selects serial encoder).

    for (layer_idx, (layer, kv)) in layers.iter().zip(kvs.iter_mut()).enumerate() {
        assert_eq!(layer.attn_norm_w.len(), hidden_dim);
        assert_eq!(layer.ffn_norm_w.len(), hidden_dim);
        assert_eq!(layer.o.rows, hidden_dim);
        assert_eq!(layer.down.rows, hidden_dim);
        assert_eq!(layer.gate.rows, layer.up.rows);

        let attn_nw = resident_f32_buffer(device, layer.attn_norm_w)?;
        let ffn_nw = resident_f32_buffer(device, layer.ffn_norm_w)?;
        let q_w = resident_weight_buffer(device, layer.q.weights)?;
        let k_w = resident_weight_buffer(device, layer.k.weights)?;
        let v_w = resident_weight_buffer(device, layer.v.weights)?;
        let o_w = resident_weight_buffer(device, layer.o.weights)?;
        let gate_w = resident_weight_buffer(device, layer.gate.weights)?;
        let up_w = resident_weight_buffer(device, layer.up.weights)?;
        let down_w = resident_weight_buffer(device, layer.down.weights)?;

        // Pre-LN: layer 0 norms raw hidden; later layers either fuse the
        // previous FFN residual into attn_norm (non-sandwich) or just
        // RMSNorm (sandwich already applied `h += down` eagerly).
        if layer_idx == 0 || sandwich {
            encode_rms_norm(
                &encoder,
                device,
                h_buf,
                &attn_nw.buffer,
                x_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
        } else {
            encode_add_rms_norm(
                &encoder,
                device,
                h_buf,
                down_buf,
                &attn_nw.buffer,
                x_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
        }
        memory_barrier_buffers(&encoder);
        // Q∥K∥V
        encode_matvec(&encoder, device, &layer.q, &q_w, x_buf, q_buf)?;
        encode_matvec(&encoder, device, &layer.k, &k_w, x_buf, k_buf)?;
        encode_matvec(&encoder, device, &layer.v, &v_w, x_buf, v_buf)?;
        memory_barrier_buffers(&encoder);
        encode_attn_extras(
            &encoder,
            device,
            &layer.extras,
            q_buf,
            k_buf,
            v_buf,
            layer.q.rows,
            layer.k.rows,
            layer.v.rows,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_eps,
        )?;
        memory_barrier_buffers(&encoder);
        let layer_theta = layer.rope_theta.unwrap_or(rope_theta);
        encode_rope(
            &encoder,
            device,
            rope_layout,
            q_buf,
            n_heads as u32,
            head_dim as u32,
            layer_theta,
            pos as u32,
            ff_buf,
        )?;
        encode_rope(
            &encoder,
            device,
            rope_layout,
            k_buf,
            n_kv_heads as u32,
            head_dim as u32,
            layer_theta,
            pos as u32,
            ff_buf,
        )?;
        memory_barrier_buffers(&encoder);
        let token_elems = (n_kv_heads * head_dim) as u32;
        let offset = (pos * n_kv_heads * head_dim) as u32;
        encode_kv_store_append(&encoder, device, k_buf, kv, KvPlane::K, offset, token_elems)?;
        encode_kv_store_append(&encoder, device, v_buf, kv, KvPlane::V, offset, token_elems)?;
        memory_barrier_buffers(&encoder);
        let new_seq = (pos + 1) as u32;
        // Sliding window: only the last `window` positions (incl. current)
        // are visible, matching `causal_gqa_attention_windowed`.
        let kv_start = match layer.window {
            Some(w) => (pos + 1).saturating_sub(w) as u32,
            None => 0,
        };
        encode_gqa_with_kv(
            &encoder,
            device,
            q_buf,
            kv,
            attn_buf,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            new_seq,
            kv_start,
            layer.extras.attn_logit_softcap,
        )?;
        memory_barrier_buffers(&encoder);
        encode_matvec(&encoder, device, &layer.o, &o_w, attn_buf, o_buf)?;
        memory_barrier_buffers(&encoder);
        // Gemma sandwich norm: normalize the attn block output *before*
        // the residual add (in-place: each thread reads x[i] only after
        // the barriered reduction, so out == x is safe).
        if let Some(post) = layer.post_attn_norm {
            assert_eq!(post.len(), hidden_dim);
            let pw = resident_f32_buffer(device, post)?;
            encode_rms_norm(
                &encoder,
                device,
                o_buf,
                &pw.buffer,
                o_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
            memory_barrier_buffers(&encoder);
        }
        if sandwich {
            // Eager residual + separate ffn_norm (prefill / CPU parity).
            encode_vec_add(&encoder, device, h_buf, o_buf, hidden_dim as u32)?;
            memory_barrier_buffers(&encoder);
            encode_rms_norm(
                &encoder,
                device,
                h_buf,
                &ffn_nw.buffer,
                x2_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
        } else {
            // Fuse attn residual + ffn_norm into one dispatch.
            encode_add_rms_norm(
                &encoder,
                device,
                h_buf,
                o_buf,
                &ffn_nw.buffer,
                x2_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
        }
        memory_barrier_buffers(&encoder);
        // gate ∥ up (llama concurrent)
        encode_matvec(&encoder, device, &layer.gate, &gate_w, x2_buf, gate_buf)?;
        encode_matvec(&encoder, device, &layer.up, &up_w, x2_buf, up_buf)?;
        memory_barrier_buffers(&encoder);
        if gelu_ffn {
            encode_gelu_mul(
                &encoder,
                device,
                gate_buf,
                up_buf,
                act_buf,
                layer.gate.rows as u32,
            )?;
        } else {
            encode_silu_mul(
                &encoder,
                device,
                gate_buf,
                up_buf,
                act_buf,
                layer.gate.rows as u32,
            )?;
        }
        memory_barrier_buffers(&encoder);
        encode_matvec(&encoder, device, &layer.down, &down_w, act_buf, down_buf)?;
        if let Some(post) = layer.post_ffn_norm {
            assert_eq!(post.len(), hidden_dim);
            let pw = resident_f32_buffer(device, post)?;
            encode_rms_norm(
                &encoder,
                device,
                down_buf,
                &pw.buffer,
                down_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
            memory_barrier_buffers(&encoder);
        }
        if sandwich {
            // Eager FFN residual — next layer attn_norm is plain RMSNorm.
            encode_vec_add(&encoder, device, h_buf, down_buf, hidden_dim as u32)?;
            memory_barrier_buffers(&encoder);
        } else {
            // Next layer's fused attn_norm reads prior `down_buf`.
            memory_barrier_buffers(&encoder);
            // Defer `h += down` until the next layer's attn_norm (or final_norm)
            // so it fuses with that RMSNorm. Last layer handled below.
        }
    }

    // Final norm / lm_head. Sandwich already applied every FFN residual;
    // non-sandwich still has a deferred last-layer `down` to fold in.
    let (download_n, norm_resident) = if let Some(fnw) = final_norm_w {
        assert_eq!(fnw.len(), hidden_dim);
        let fn_buf = resident_f32_buffer(device, fnw)?;
        if sandwich {
            encode_rms_norm(
                &encoder,
                device,
                h_buf,
                &fn_buf.buffer,
                x_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
        } else {
            encode_add_rms_norm(
                &encoder,
                device,
                h_buf,
                down_buf,
                &fn_buf.buffer,
                x_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
        }
        if let (Some(out_l), Some(logits)) = (output, logits_buf) {
            assert_eq!(out_l.rows, logits_rows.unwrap());
            // RAW: lm_head reads `x_buf` written by add_rms_norm. Without
            // this barrier Metal may overlap the matvec with the norm on
            // small hiddens (SmolLM2 h=576) and produce garbage logits /
            // greedy tokens while host lm_head after wait looks fine.
            memory_barrier_buffers(&encoder);
            let out_w = resident_weight_buffer(device, out_l.weights)?;
            encode_matvec(&encoder, device, out_l, &out_w, x_buf, logits)?;
            if argmax_only {
                memory_barrier_buffers(&encoder);
                encode_argmax(&encoder, device, logits, argmax_idx_buf, out_l.rows as u32)?;
                (1, false)
            } else {
                (out_l.rows, false)
            }
        } else {
            // final_norm ran but no lm_head — download normalized hidden
            // and mark x_buf resident for the next apply_gpu.
            (hidden_dim, true)
        }
    } else if sandwich {
        (hidden_dim, false)
    } else {
        // No final_norm: still apply the deferred last-layer FFN residual.
        encode_vec_add(&encoder, device, h_buf, down_buf, hidden_dim as u32)?;
        (hidden_dim, false)
    };

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    for kv in kvs.iter_mut() {
        kv.seq_len = pos + 1;
    }

    // If final_norm ran but no lm_head, mark normalized hidden (x_buf)
    // as resident so the next apply_gpu can skip re-upload.
    if norm_resident {
        crate::gpu::set_resident_activation(x_buf, hidden_dim);
    }

    if argmax_only && download_n == 1 && output.is_some() {
        let ptr = argmax_idx_buf.contents();
        let idx = unsafe { *(ptr.as_ptr() as *const u32) as usize };
        return Ok(vec![idx as f32]);
    }

    let src: &ProtocolObject<dyn MTLBuffer> = if norm_resident {
        x_buf
    } else if download_n == hidden_dim {
        h_buf
    } else {
        logits_buf.expect("logits buffer when downloading logits")
    };
    let out_ptr = src.contents();
    Ok(unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, download_n).to_vec() })
}

/// Per-layer `mul_mm_sg` launches for [`launch_prefill_dense_layer`] /
/// [`launch_prefill_dense_stack`].
pub struct PrefillDenseLayerMetal<'a> {
    pub attn_norm_w: &'a [f32],
    pub ffn_norm_w: &'a [f32],
    pub q: MulMmSgLaunch<'a>,
    pub k: MulMmSgLaunch<'a>,
    pub v: MulMmSgLaunch<'a>,
    pub o: MulMmSgLaunch<'a>,
    pub gate: MulMmSgLaunch<'a>,
    pub up: MulMmSgLaunch<'a>,
    pub down: MulMmSgLaunch<'a>,
    pub post_attn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
    /// QKV bias / QK-norm (Qwen2.5, Qwen3, Gemma-3). Applied after GEMM,
    /// before RoPE — same order as the CPU / decode paths.
    pub extras: AttnExtras<'a>,
    /// Layer index for [`PrefillCbCache`] keying only.
    pub layer_idx: u32,
}

struct PrefillScratchView<'a> {
    h: &'a ProtocolObject<dyn MTLBuffer>,
    x: &'a ProtocolObject<dyn MTLBuffer>,
    x2: &'a ProtocolObject<dyn MTLBuffer>,
    q: &'a ProtocolObject<dyn MTLBuffer>,
    k: &'a ProtocolObject<dyn MTLBuffer>,
    v: &'a ProtocolObject<dyn MTLBuffer>,
    attn: &'a ProtocolObject<dyn MTLBuffer>,
    o: &'a ProtocolObject<dyn MTLBuffer>,
    gate: &'a ProtocolObject<dyn MTLBuffer>,
    up: &'a ProtocolObject<dyn MTLBuffer>,
    act: &'a ProtocolObject<dyn MTLBuffer>,
    down: &'a ProtocolObject<dyn MTLBuffer>,
    half_act: &'a ProtocolObject<dyn MTLBuffer>,
}

struct PrefillDenseLayerResident {
    attn_nw: std::sync::Arc<ResidentF32Buffer>,
    ffn_nw: std::sync::Arc<ResidentF32Buffer>,
    q_w: std::sync::Arc<ResidentWeightBuffer>,
    k_w: std::sync::Arc<ResidentWeightBuffer>,
    v_w: std::sync::Arc<ResidentWeightBuffer>,
    o_w: std::sync::Arc<ResidentWeightBuffer>,
    gate_w: std::sync::Arc<ResidentWeightBuffer>,
    up_w: std::sync::Arc<ResidentWeightBuffer>,
    down_w: std::sync::Arc<ResidentWeightBuffer>,
    post_attn_w: Option<std::sync::Arc<ResidentF32Buffer>>,
    post_ffn_w: Option<std::sync::Arc<ResidentF32Buffer>>,
    extras: AttnExtrasResident,
}

fn resident_prefill_dense_layer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    layer: &PrefillDenseLayerMetal<'_>,
    hidden_dim: usize,
) -> Result<PrefillDenseLayerResident, MetalError> {
    let post_attn_w = if let Some(post) = layer.post_attn_norm {
        assert_eq!(post.len(), hidden_dim);
        Some(resident_f32_buffer(device, post)?)
    } else {
        None
    };
    let post_ffn_w = if let Some(post) = layer.post_ffn_norm {
        assert_eq!(post.len(), hidden_dim);
        Some(resident_f32_buffer(device, post)?)
    } else {
        None
    };
    Ok(PrefillDenseLayerResident {
        attn_nw: resident_f32_buffer(device, layer.attn_norm_w)?,
        ffn_nw: resident_f32_buffer(device, layer.ffn_norm_w)?,
        q_w: resident_weight_buffer(device, layer.q.weights)?,
        k_w: resident_weight_buffer(device, layer.k.weights)?,
        v_w: resident_weight_buffer(device, layer.v.weights)?,
        o_w: resident_weight_buffer(device, layer.o.weights)?,
        gate_w: resident_weight_buffer(device, layer.gate.weights)?,
        up_w: resident_weight_buffer(device, layer.up.weights)?,
        down_w: resident_weight_buffer(device, layer.down.weights)?,
        post_attn_w,
        post_ffn_w,
        extras: resident_attn_extras(device, &layer.extras)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_prefill_dense_layer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    layer: &PrefillDenseLayerMetal<'_>,
    resident: &PrefillDenseLayerResident,
    scratch: &PrefillScratchView<'_>,
    kv: &MetalKvBuffers,
    n_heads: usize,
    batch: usize,
    hidden_dim: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    ff_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    start_pos: usize,
    rms_eps: f32,
    gelu_ffn: bool,
    attn_softcap: Option<f32>,
) -> Result<(), MetalError> {
    let n_kv_heads = kv.n_kv_heads;
    let head_dim = kv.head_dim;
    let h_buf = scratch.h;
    let _x_buf = scratch.x;
    let x2_buf = scratch.x2;
    let q_buf = scratch.q;
    let k_buf = scratch.k;
    let v_buf = scratch.v;
    let attn_buf = scratch.attn;
    let o_buf = scratch.o;
    let gate_buf = scratch.gate;
    let up_buf = scratch.up;
    let act_buf = scratch.act;
    let down_buf = scratch.down;

    let half_act = scratch.half_act;
    encode_rms_norm_f32_to_f16_batch(
        encoder,
        device,
        h_buf,
        &resident.attn_nw.buffer,
        half_act,
        hidden_dim as u32,
        batch as u32,
        rms_eps,
    )?;
    memory_barrier_buffers(encoder);
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.q,
        &resident.q_w,
        half_act,
        q_buf,
        batch,
    )?;
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.k,
        &resident.k_w,
        half_act,
        k_buf,
        batch,
    )?;
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.v,
        &resident.v_w,
        half_act,
        v_buf,
        batch,
    )?;
    memory_barrier_buffers(encoder);

    encode_attn_extras_batch(
        encoder,
        device,
        &layer.extras,
        q_buf,
        k_buf,
        v_buf,
        layer.q.rows,
        layer.k.rows,
        layer.v.rows,
        n_heads,
        n_kv_heads,
        head_dim,
        batch,
        rms_eps,
        &resident.extras,
    )?;
    memory_barrier_buffers(encoder);

    encode_rope_batch(
        encoder,
        device,
        rope_layout,
        q_buf,
        n_heads as u32,
        head_dim as u32,
        rope_theta,
        start_pos as u32,
        batch as u32,
        ff_buf,
    )?;
    encode_rope_batch(
        encoder,
        device,
        rope_layout,
        k_buf,
        n_kv_heads as u32,
        head_dim as u32,
        rope_theta,
        start_pos as u32,
        batch as u32,
        ff_buf,
    )?;
    memory_barrier_buffers(encoder);

    let kv_width = n_kv_heads * head_dim;
    let token_elems = (batch * kv_width) as u32;
    let offset = (kv.seq_len * kv_width) as u32;
    encode_kv_store_append(encoder, device, k_buf, kv, KvPlane::K, offset, token_elems)?;
    encode_kv_store_append(encoder, device, v_buf, kv, KvPlane::V, offset, token_elems)?;
    memory_barrier_buffers(encoder);

    encode_gqa_prefill_with_kv(
        encoder,
        device,
        q_buf,
        kv,
        attn_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        batch as u32,
        start_pos as u32,
        attn_softcap,
    )?;
    memory_barrier_buffers(encoder);

    encode_f32_to_f16(
        encoder,
        device,
        attn_buf,
        half_act,
        (batch * layer.q.rows) as u32,
    )?;
    memory_barrier_buffers(encoder);
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.o,
        &resident.o_w,
        half_act,
        o_buf,
        batch,
    )?;
    memory_barrier_buffers(encoder);

    if let Some(pw) = resident.post_attn_w.as_ref() {
        encode_rms_norm_batch(
            encoder,
            device,
            o_buf,
            &pw.buffer,
            o_buf,
            hidden_dim as u32,
            batch as u32,
            rms_eps,
        )?;
        memory_barrier_buffers(encoder);
    }

    // Fuse residual add + FFN RMSNorm into one dispatch when possible.
    if resident.post_attn_w.is_none() {
        encode_add_rms_norm_batch(
            encoder,
            device,
            h_buf,
            o_buf,
            &resident.ffn_nw.buffer,
            x2_buf,
            hidden_dim as u32,
            batch as u32,
            rms_eps,
        )?;
    } else {
        encode_vec_add(encoder, device, h_buf, o_buf, (batch * hidden_dim) as u32)?;
        memory_barrier_buffers(encoder);
        encode_rms_norm_batch(
            encoder,
            device,
            h_buf,
            &resident.ffn_nw.buffer,
            x2_buf,
            hidden_dim as u32,
            batch as u32,
            rms_eps,
        )?;
    }
    memory_barrier_buffers(encoder);

    encode_f32_to_f16(
        encoder,
        device,
        x2_buf,
        half_act,
        (batch * hidden_dim) as u32,
    )?;
    memory_barrier_buffers(encoder);
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.gate,
        &resident.gate_w,
        half_act,
        gate_buf,
        batch,
    )?;
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.up,
        &resident.up_w,
        half_act,
        up_buf,
        batch,
    )?;
    memory_barrier_buffers(encoder);
    let ffn_elems = (batch * layer.gate.rows) as u32;
    if gelu_ffn {
        encode_gelu_mul(encoder, device, gate_buf, up_buf, act_buf, ffn_elems)?;
    } else {
        encode_silu_mul(encoder, device, gate_buf, up_buf, act_buf, ffn_elems)?;
    }
    memory_barrier_buffers(encoder);
    encode_f32_to_f16(
        encoder,
        device,
        act_buf,
        half_act,
        (batch * layer.gate.rows) as u32,
    )?;
    memory_barrier_buffers(encoder);
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.down,
        &resident.down_w,
        half_act,
        down_buf,
        batch,
    )?;
    memory_barrier_buffers(encoder);

    if let Some(pw) = resident.post_ffn_w.as_ref() {
        encode_rms_norm_batch(
            encoder,
            device,
            down_buf,
            &pw.buffer,
            down_buf,
            hidden_dim as u32,
            batch as u32,
            rms_eps,
        )?;
        memory_barrier_buffers(encoder);
    }

    encode_vec_add(
        encoder,
        device,
        h_buf,
        down_buf,
        (batch * hidden_dim) as u32,
    )?;
    memory_barrier_buffers(encoder);
    Ok(())
}

/// Consecutive dense prefill layers in **one** command buffer (B≥4).
///
/// Layer 0 copies `hidden` into scratch `h`; later layers leave `h` in
/// place after each residual add. Activations stay in [`PrefillScratch`]
/// across layers; host readback happens once at the end. Records
/// [`PrefillStackCbKey`] plus per-layer [`PrefillCbKey`] entries in the
/// process [`MetalGraph`].
///
/// Timing: `FERROX_METAL_MM_TIMING=1` logs setup/gpu/readback totals.
#[allow(clippy::too_many_arguments)]
pub fn launch_prefill_dense_stack(
    hidden: &[f32],
    layers: &[PrefillDenseLayerMetal<'_>],
    kvs: &mut [MetalKvBuffers],
    n_heads: usize,
    batch: usize,
    rope_layout: MetalRopeLayout,
    rope_thetas: &[f32],
    freq_factors: Option<&[f32]>,
    start_pos: usize,
    rms_eps: f32,
    gelu_ffn: bool,
    attn_softcap: Option<f32>,
) -> Result<Vec<f32>, MetalError> {
    if batch < 4 {
        return Err(MetalError::CommandFailed);
    }
    assert_eq!(layers.len(), kvs.len());
    assert_eq!(layers.len(), rope_thetas.len());
    assert!(!layers.is_empty());

    let hidden_dim = layers[0].attn_norm_w.len();
    assert_eq!(hidden.len(), batch * hidden_dim);
    let head_dim = kvs[0].head_dim;
    let n_kv_heads = kvs[0].n_kv_heads;
    for (layer, kv) in layers.iter().zip(kvs.iter()) {
        assert_eq!(layer.attn_norm_w.len(), hidden_dim);
        assert_eq!(layer.ffn_norm_w.len(), hidden_dim);
        assert_eq!(layer.q.rows, n_heads * head_dim);
        assert_eq!(layer.k.rows, n_kv_heads * head_dim);
        assert_eq!(layer.v.rows, n_kv_heads * head_dim);
        assert_eq!(layer.o.rows, hidden_dim);
        assert_eq!(layer.down.rows, hidden_dim);
        assert_eq!(layer.gate.rows, layer.up.rows);
        assert_eq!(start_pos, kv.seq_len);
        if kv.seq_len + batch > kv.capacity {
            return Err(MetalError::CommandFailed);
        }
        assert_eq!(kv.head_dim, head_dim);
        assert_eq!(kv.n_kv_heads, n_kv_heads);
    }
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), head_dim / 2);
    }

    let max_q = layers.iter().map(|l| l.q.rows).max().unwrap();
    let max_kv = layers.iter().map(|l| l.k.rows.max(l.v.rows)).max().unwrap();
    let max_gate = layers.iter().map(|l| l.gate.rows).max().unwrap();

    let timing = std::env::var_os("FERROX_METAL_MM_TIMING").is_some();
    let t_setup = std::time::Instant::now();

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let scratch_guard = borrow_prefill_scratch(
        device,
        PrefillScratchCaps {
            batch,
            hidden: hidden_dim,
            max_q,
            max_kv,
            max_gate,
        },
    )?;
    let scratch = scratch_guard.as_ref().expect("prefill scratch ensured");
    copy_f32_into(&scratch.h, hidden);

    let scratch_view = PrefillScratchView {
        h: &scratch.h,
        x: &scratch.x,
        x2: &scratch.x2,
        q: &scratch.q,
        k: &scratch.k,
        v: &scratch.v,
        attn: &scratch.attn,
        o: &scratch.o,
        gate: &scratch.gate,
        up: &scratch.up,
        act: &scratch.act,
        down: &scratch.down,
        half_act: &scratch.half_act,
    };

    let ff_resident = match freq_factors {
        Some(ff) => Some(resident_f32_buffer(device, ff)?),
        None => None,
    };
    let ff_buf = ff_resident.as_ref().map(|b| b.buffer.as_ref());

    {
        let mut graph = metal_graph();
        if !graph.prefill_pipelines_warmed() {
            graph.warm_prefill_pipelines(
                device,
                PrefillWarmParams {
                    layer: &layers[0],
                    rope_layout,
                    head_dim: head_dim as u32,
                    gelu_ffn,
                    kv_dtype: kvs[0].dtype,
                },
            )?;
        }
        graph.prefill.note_stack(PrefillStackCbKey {
            start_layer: layers[0].layer_idx,
            depth: layers.len() as u32,
            batch: batch as u32,
            hidden: hidden_dim as u32,
        });
        for layer in layers {
            graph.prefill.note(PrefillCbKey {
                layer: layer.layer_idx,
                batch: batch as u32,
                hidden: hidden_dim as u32,
                ffn: layer.gate.rows as u32,
                q_rows: layer.q.rows as u32,
            });
        }
    }

    let setup_us = t_setup.elapsed().as_micros();
    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = compute_encoder_concurrent(&cmd_buf)?;

    for (layer_idx, (layer, kv)) in layers.iter().zip(kvs.iter()).enumerate() {
        let resident = resident_prefill_dense_layer(device, layer, hidden_dim)?;
        encode_prefill_dense_layer(
            &encoder,
            device,
            layer,
            &resident,
            &scratch_view,
            kv,
            n_heads,
            batch,
            hidden_dim,
            rope_layout,
            rope_thetas[layer_idx],
            ff_buf,
            start_pos,
            rms_eps,
            gelu_ffn,
            attn_softcap,
        )?;
    }

    encoder.endEncoding();
    let t_gpu = std::time::Instant::now();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    let gpu_us = t_gpu.elapsed().as_micros();

    for kv in kvs.iter_mut() {
        kv.seq_len += batch;
    }

    let t_read = std::time::Instant::now();
    let out_ptr = scratch.h.contents();
    let out = unsafe {
        std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, batch * hidden_dim).to_vec()
    };
    if timing {
        crate::gpu::mm_timing_add(setup_us, gpu_us, t_read.elapsed().as_micros());
    }
    Ok(out)
}

/// One dense prefill layer in **one** command buffer (B≥4).
///
/// Encodes: attn RMSNorm (per row) → Q∥K∥V `mul_mm_sg` → RoPE → KV append →
/// causal GQA → O `mul_mm_sg` → residual → FFN RMSNorm → gate∥up → act →
/// down → residual. Activations stay in [`PrefillScratch`]; barriers match
/// the decode-stack Concurrent pattern. Records a [`PrefillCbKey`] in the
/// process [`MetalGraph`] and warms prefill pipelines on the first call
/// (llama.cpp `ggml_metal_graph_compute` residency; CB replay still TODO).
///
/// Rejects `batch < 4` and models that need QKV bias / QK-norm on this path
/// (caller should fall back to host proj + [`launch_prefill_attn_block`]).
///
/// Timing: `FERROX_METAL_MM_TIMING=1` logs setup/gpu/readback like mul_mm_sg.
#[allow(clippy::too_many_arguments)]
pub fn launch_prefill_dense_layer(
    hidden: &[f32],
    layer: &PrefillDenseLayerMetal<'_>,
    kv: &mut MetalKvBuffers,
    n_heads: usize,
    batch: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    start_pos: usize,
    rms_eps: f32,
    gelu_ffn: bool,
    attn_softcap: Option<f32>,
) -> Result<Vec<f32>, MetalError> {
    launch_prefill_dense_stack(
        hidden,
        std::slice::from_ref(layer),
        std::slice::from_mut(kv),
        n_heads,
        batch,
        rope_layout,
        std::slice::from_ref(&rope_theta),
        freq_factors,
        start_pos,
        rms_eps,
        gelu_ffn,
        attn_softcap,
    )
}

/// Host-upload RoPE only (parity testing). Applies `layout` RoPE in-place
/// across `n_heads` packed heads in `vecs` (`n_heads * head_dim`).
pub fn launch_rope_heads_host(
    vecs: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    layout: MetalRopeLayout,
    theta: f32,
    pos: usize,
    freq_factors: Option<&[f32]>,
) -> Result<(), MetalError> {
    assert_eq!(vecs.len(), n_heads * head_dim);
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), head_dim / 2);
    }
    let shared = shared_metal()?;
    let device = &shared.device;
    let buf = upload_f32(device, vecs)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };
    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_rope(
        &encoder,
        device,
        layout,
        &buf,
        n_heads as u32,
        head_dim as u32,
        theta,
        pos as u32,
        ff_buf.as_deref(),
    )?;
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    let ptr = buf.contents();
    let out = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, vecs.len()) };
    vecs.copy_from_slice(out);
    Ok(())
}

/// Host-upload GQA only (parity testing / fallback probe).
#[allow(clippy::too_many_arguments)]
pub fn launch_gqa_decode_host(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_gqa_decode_host_ex(
        q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, 0, None,
    )
}

/// Host-upload GQA with optional sliding-window start and logit softcap.
#[allow(clippy::too_many_arguments)]
pub fn launch_gqa_decode_host_ex(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_start: usize,
    attn_softcap: Option<f32>,
) -> Result<Vec<f32>, MetalError> {
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(k_cache.len(), seq_len * n_kv_heads * head_dim);
    assert_eq!(v_cache.len(), seq_len * n_kv_heads * head_dim);
    assert!(kv_start <= seq_len);

    let shared = shared_metal()?;
    let device = &shared.device;
    let q_buf = upload_f32(device, q)?;
    let k_buf = upload_f16_from_f32(device, k_cache)?;
    let v_buf = upload_f16_from_f32(device, v_cache)?;
    let out_buf = alloc_f32_buffer(device, n_heads * head_dim)?;

    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_gqa(
        &encoder,
        device,
        &q_buf,
        &k_buf,
        &v_buf,
        &out_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        seq_len as u32,
        kv_start as u32,
        attn_softcap,
    )?;
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let ptr = out_buf.contents();
    Ok(unsafe {
        std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n_heads * head_dim).to_vec()
    })
}

/// Multi-token RoPE → batch KV append → causal GQA prefill.
///
/// `q`/`k`/`v` are **pre-RoPE**, packed `[n_q, n_heads|n_kv_heads, head_dim]`.
/// `start_pos` must equal `kv.seq_len` (prefix already resident on Metal, or
/// empty). Returns attention output `[n_q, n_heads, head_dim]` and RoPE'd
/// K/V for host [`KvCache`] sync. Updates `kv.seq_len` by `n_q`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn launch_prefill_attn_block(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv: &mut MetalKvBuffers,
    n_heads: usize,
    n_q: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    start_pos: usize,
    attn_softcap: Option<f32>,
    return_kv: bool,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), MetalError> {
    let head_dim = kv.head_dim;
    let n_kv_heads = kv.n_kv_heads;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    assert_eq!(q.len(), n_q * q_width);
    assert_eq!(k.len(), n_q * kv_width);
    assert_eq!(v.len(), n_q * kv_width);
    assert_eq!(
        start_pos, kv.seq_len,
        "prefill start_pos must equal current Metal KV length"
    );
    if kv.seq_len + n_q > kv.capacity {
        return Err(MetalError::CommandFailed);
    }
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), head_dim / 2);
    }
    if n_q == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let q_buf = upload_f32(device, q)?;
    let k_buf = upload_f32(device, k)?;
    let v_buf = upload_f32(device, v)?;
    let attn_buf = alloc_f32_buffer(device, n_q * q_width)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    encode_rope_batch(
        &encoder,
        device,
        rope_layout,
        &q_buf,
        n_heads as u32,
        head_dim as u32,
        rope_theta,
        start_pos as u32,
        n_q as u32,
        ff_buf.as_deref(),
    )?;
    encode_rope_batch(
        &encoder,
        device,
        rope_layout,
        &k_buf,
        n_kv_heads as u32,
        head_dim as u32,
        rope_theta,
        start_pos as u32,
        n_q as u32,
        ff_buf.as_deref(),
    )?;

    let token_elems = (n_q * kv_width) as u32;
    let offset = (kv.seq_len * kv_width) as u32;
    encode_kv_store_append(
        &encoder,
        device,
        &k_buf,
        kv,
        KvPlane::K,
        offset,
        token_elems,
    )?;
    encode_kv_store_append(
        &encoder,
        device,
        &v_buf,
        kv,
        KvPlane::V,
        offset,
        token_elems,
    )?;

    let prefill_result = encode_gqa_prefill_with_kv(
        &encoder,
        device,
        &q_buf,
        kv,
        &attn_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        n_q as u32,
        start_pos as u32,
        attn_softcap,
    );
    encoder.endEncoding();
    prefill_result?;
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    kv.seq_len += n_q;

    let attn_ptr = attn_buf.contents();
    let attn = unsafe {
        std::slice::from_raw_parts(attn_ptr.as_ptr() as *const f32, n_q * q_width).to_vec()
    };
    if !return_kv {
        return Ok((attn, Vec::new(), Vec::new()));
    }
    let k_ptr = k_buf.contents();
    let v_ptr = v_buf.contents();
    let k_roped = unsafe {
        std::slice::from_raw_parts(k_ptr.as_ptr() as *const f32, n_q * kv_width).to_vec()
    };
    let v_roped = unsafe {
        std::slice::from_raw_parts(v_ptr.as_ptr() as *const f32, n_q * kv_width).to_vec()
    };
    Ok((attn, k_roped, v_roped))
}

/// Prefill attn + Q4_0 O projection + residual add in one CB.
/// Skips host K/V download (Metal KV is authoritative). Returns `h_out`.
#[allow(clippy::too_many_arguments)]
pub fn launch_prefill_attn_o_residual(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    h_in: &[f32],
    o: &MatvecLaunch<'_>,
    kv: &mut MetalKvBuffers,
    n_heads: usize,
    n_q: usize,
    rope_layout: MetalRopeLayout,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    start_pos: usize,
    attn_softcap: Option<f32>,
) -> Result<Vec<f32>, MetalError> {
    if o.fn_name != "q4_0_matvec" || o.block_bytes != 18 {
        return Err(MetalError::CommandFailed);
    }
    let head_dim = kv.head_dim;
    let n_kv_heads = kv.n_kv_heads;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let hidden = o.rows;
    assert_eq!(q.len(), n_q * q_width);
    assert_eq!(k.len(), n_q * kv_width);
    assert_eq!(v.len(), n_q * kv_width);
    assert_eq!(h_in.len(), n_q * hidden);
    assert_eq!(start_pos, kv.seq_len);
    if kv.seq_len + n_q > kv.capacity {
        return Err(MetalError::CommandFailed);
    }
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), head_dim / 2);
    }
    if n_q == 0 {
        return Ok(Vec::new());
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let q_buf = upload_f32(device, q)?;
    let k_buf = upload_f32(device, k)?;
    let v_buf = upload_f32(device, v)?;
    let mut h_owned = h_in.to_vec();
    let h_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(h_owned.as_mut_ptr() as *mut _).unwrap(),
            h_owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;
    let attn_buf = alloc_f32_buffer(device, n_q * q_width)?;
    let o_buf = alloc_f32_buffer(device, n_q * hidden)?;
    let o_w = resident_weight_buffer(device, o.weights)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    encode_rope_batch(
        &encoder,
        device,
        rope_layout,
        &q_buf,
        n_heads as u32,
        head_dim as u32,
        rope_theta,
        start_pos as u32,
        n_q as u32,
        ff_buf.as_deref(),
    )?;
    encode_rope_batch(
        &encoder,
        device,
        rope_layout,
        &k_buf,
        n_kv_heads as u32,
        head_dim as u32,
        rope_theta,
        start_pos as u32,
        n_q as u32,
        ff_buf.as_deref(),
    )?;

    let token_elems = (n_q * kv_width) as u32;
    let offset = (kv.seq_len * kv_width) as u32;
    encode_kv_store_append(
        &encoder,
        device,
        &k_buf,
        kv,
        KvPlane::K,
        offset,
        token_elems,
    )?;
    encode_kv_store_append(
        &encoder,
        device,
        &v_buf,
        kv,
        KvPlane::V,
        offset,
        token_elems,
    )?;

    encode_gqa_prefill_with_kv(
        &encoder,
        device,
        &q_buf,
        kv,
        &attn_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        n_q as u32,
        start_pos as u32,
        attn_softcap,
    )?;
    // Attn must finish before O reads it.
    memory_barrier_resources(&encoder, &[attn_buf.as_ref()]);
    // Small T: per-token matvec (mul_mm underperforms below ~32).
    // Large T: weight-reuse mul_mm.
    if n_q >= 32 {
        let n_blocks = o.row_bytes / o.block_bytes;
        encode_q4_0_mul_mm(
            &encoder,
            device,
            &o_w,
            &attn_buf,
            &o_buf,
            o.row_bytes,
            n_blocks,
            o.rows,
            n_q,
        )?;
    } else {
        for t in 0..n_q {
            encode_matvec_with_offsets(
                &encoder,
                device,
                o,
                &o_w,
                &attn_buf,
                t * q_width * 4,
                &o_buf,
                t * hidden * 4,
            )?;
        }
    }
    memory_barrier_resources(&encoder, &[o_buf.as_ref()]);
    encode_vec_add(&encoder, device, &h_buf, &o_buf, (n_q * hidden) as u32)?;
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    kv.seq_len += n_q;

    let h_ptr = h_buf.contents();
    Ok(unsafe { std::slice::from_raw_parts(h_ptr.as_ptr() as *const f32, n_q * hidden).to_vec() })
}

/// Host-upload multi-pos RoPE (parity testing). Layout `[n_tokens, n_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn launch_rope_heads_batch_host(
    vecs: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    n_tokens: usize,
    layout: MetalRopeLayout,
    theta: f32,
    base_pos: usize,
    freq_factors: Option<&[f32]>,
) -> Result<(), MetalError> {
    assert_eq!(vecs.len(), n_tokens * n_heads * head_dim);
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), head_dim / 2);
    }
    if n_tokens == 0 {
        return Ok(());
    }
    let shared = shared_metal()?;
    let device = &shared.device;
    let buf = upload_f32(device, vecs)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };
    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_rope_batch(
        &encoder,
        device,
        layout,
        &buf,
        n_heads as u32,
        head_dim as u32,
        theta,
        base_pos as u32,
        n_tokens as u32,
        ff_buf.as_deref(),
    )?;
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    let ptr = buf.contents();
    let out = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, vecs.len()) };
    vecs.copy_from_slice(out);
    Ok(())
}

/// Host-upload multi-query causal GQA (parity testing).
/// `q` is `[n_q, n_heads, head_dim]`; K/V are full caches of length
/// `kv_prefix_len + n_q` (already including the new tokens).
#[allow(clippy::too_many_arguments)]
pub fn launch_gqa_prefill_host(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_q: usize,
    kv_prefix_len: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_gqa_prefill_host_ex(
        q,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        n_q,
        kv_prefix_len,
        None,
    )
}

/// [`launch_gqa_prefill_host`] with an attention-logit softcap, so the
/// Gemma prefill path can be checked against the CPU reference.
#[allow(clippy::too_many_arguments)]
pub fn launch_gqa_prefill_host_ex(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_q: usize,
    kv_prefix_len: usize,
    attn_softcap: Option<f32>,
) -> Result<Vec<f32>, MetalError> {
    let total_seq = kv_prefix_len + n_q;
    assert_eq!(q.len(), n_q * n_heads * head_dim);
    assert_eq!(k_cache.len(), total_seq * n_kv_heads * head_dim);
    assert_eq!(v_cache.len(), total_seq * n_kv_heads * head_dim);
    if n_q == 0 {
        return Ok(Vec::new());
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let q_buf = upload_f32(device, q)?;
    let k_buf = upload_f16_from_f32(device, k_cache)?;
    let v_buf = upload_f16_from_f32(device, v_cache)?;
    let out_buf = alloc_f32_buffer(device, n_q * n_heads * head_dim)?;

    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    let enc_result = encode_gqa_prefill(
        &encoder,
        device,
        &q_buf,
        &k_buf,
        &v_buf,
        &out_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        n_q as u32,
        kv_prefix_len as u32,
        attn_softcap,
    );
    encoder.endEncoding();
    enc_result?;
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let ptr = out_buf.contents();
    Ok(unsafe {
        std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n_q * n_heads * head_dim).to_vec()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_cb_cache_tracks_keys_and_hot_pipelines() {
        let mut cache = PrefillCbCache::default();
        let key = PrefillCbKey {
            layer: 3,
            batch: 8,
            hidden: 4096,
            ffn: 14336,
            q_rows: 4096,
        };
        assert!(cache.note(key));
        assert!(!cache.note(key));
        assert!(cache.contains(&key));
        assert_eq!(cache.len(), 1);
        let stack = PrefillStackCbKey {
            start_layer: 0,
            depth: 32,
            batch: 8,
            hidden: 4096,
        };
        assert!(cache.note_stack(stack));
        assert!(!cache.note_stack(stack));
        assert!(cache.contains_stack(&stack));
        cache.mark_pipeline_hot("q4_k_mul_mm_sg");
        cache.mark_pipeline_hot("rms_norm_f32");
        assert!(cache.is_pipeline_hot("q4_k_mul_mm_sg"));
        assert!(!cache.is_pipeline_hot("gqa_prefill"));
        assert_eq!(cache.hot_pipeline_count(), 2);
    }

    #[test]
    fn metal_mm_timing_env_is_optional() {
        // Documented hook for [`launch_prefill_dense_layer`]: when set, setup/gpu/readback
        // microseconds accumulate via [`crate::gpu::mm_timing_add`].
        let enabled = std::env::var_os("FERROX_METAL_MM_TIMING").is_some();
        let _ = enabled;
    }

    #[test]
    fn parse_metal_kv_dtype_f16_default_and_aliases() {
        assert_eq!(parse_metal_kv_dtype(None), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(Some("")), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(Some("f16")), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(Some("FP16")), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(Some("half")), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(Some("bogus")), MetalKvDtype::F16);
    }

    #[test]
    fn parse_metal_kv_dtype_q8_0() {
        assert_eq!(parse_metal_kv_dtype(Some("q8_0")), MetalKvDtype::Q8_0);
        assert_eq!(parse_metal_kv_dtype(Some("Q8_0")), MetalKvDtype::Q8_0);
        assert_eq!(parse_metal_kv_dtype(Some("q8")), MetalKvDtype::Q8_0);
    }

    #[test]
    fn parse_metal_kv_dtype_turbo_family() {
        assert_eq!(parse_metal_kv_dtype(Some("fp8")), MetalKvDtype::Fp8);
        assert_eq!(parse_metal_kv_dtype(Some("turbo8")), MetalKvDtype::Turbo8);
        assert_eq!(parse_metal_kv_dtype(Some("turbo4")), MetalKvDtype::Turbo4);
        assert_eq!(parse_metal_kv_dtype(Some("turbo3")), MetalKvDtype::Turbo3);
        assert!(MetalKvDtype::Turbo4.is_implemented());
        assert!(MetalKvDtype::Turbo8.is_implemented());
        assert!(MetalKvDtype::Fp8.is_implemented());
        assert!(!MetalKvDtype::Turbo3.is_implemented());
        assert!(MetalKvDtype::F16.is_implemented());
        assert!(MetalKvDtype::Q8_0.is_implemented());
        assert!(metal_kv_q8_0_viable(4, 64));
        assert!(!metal_kv_q8_0_viable(2, 8));
        assert!(metal_kv_turbo4_viable(4, 64));
    }

    fn cpu_gqa(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        cpu_gqa_ex(
            q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, 0, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_gqa_ex(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_start: usize,
        softcap: Option<f32>,
    ) -> Vec<f32> {
        let group_size = n_heads / n_kv_heads.max(1);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let kv_h = h / group_size.max(1);
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for t in kv_start..seq_len {
                let k_t = &k_cache
                    [(t * n_kv_heads + kv_h) * head_dim..(t * n_kv_heads + kv_h + 1) * head_dim];
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += q_h[d] * k_t[d];
                }
                let mut score = dot * scale;
                if let Some(c) = softcap.filter(|&c| c > 0.0) {
                    score = c * (score / c).tanh();
                }
                scores[t] = score;
            }
            let max = scores[kv_start..]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in scores[kv_start..].iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in scores[kv_start..].iter_mut() {
                *s /= sum.max(f32::MIN_POSITIVE);
            }
            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            for t in kv_start..seq_len {
                let v_t = &v_cache
                    [(t * n_kv_heads + kv_h) * head_dim..(t * n_kv_heads + kv_h + 1) * head_dim];
                let w = scores[t];
                for d in 0..head_dim {
                    out_h[d] += w * v_t[d];
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_decode_matches_cpu() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let seq_len = 5;
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let cpu = cpu_gqa(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);
        let gpu = launch_gqa_decode_host(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len)
            .expect("metal gqa");
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            // Device KV is f16; allow round-trip vs f32 CPU reference.
            let tol = 2e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "elem {i}: cpu={a} gpu={b} tol={tol}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_fa_vec_matches_cpu() {
        // FA-vec dedicated kernels: d=128 (Llama-3.x), d=64 (TinyLlama /
        // Llama-3.2-1B), d=96 (Phi-3), d=256 (Gemma-3).
        let test_cases = vec![
            (4, 2, 128, 17),
            (8, 2, 128, 33),
            (8, 4, 128, 65),
            (8, 4, 128, 128),
            (4, 2, 64, 17),
            (8, 2, 64, 33),
            (8, 4, 64, 65),
            (32, 4, 64, 128),
            (8, 4, 64, 1),
            (4, 4, 96, 17),
            (8, 8, 96, 33),
            (32, 32, 96, 65),
            (4, 1, 256, 17),
            (8, 4, 256, 33),
            (4, 1, 256, 65),
        ];
        for (n_heads, n_kv_heads, head_dim, seq_len) in test_cases {
            let q: Vec<f32> = (0..n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let cpu = cpu_gqa(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);

            // FA is default-on for d=128; force-on for clarity.
            std::env::set_var("FERROX_METAL_FA_VEC", "1");
            let gpu = launch_gqa_decode_host(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len)
                .expect("metal gqa fa-vec");

            assert_eq!(
                cpu.len(),
                gpu.len(),
                "nh={n_heads} nkv={n_kv_heads} hd={head_dim} seq={seq_len}"
            );
            for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let tol = 2e-3 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "nh={n_heads} nkv={n_kv_heads} hd={head_dim} seq={seq_len} elem {i}: cpu={a} gpu={b} tol={tol}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_fa_vec_window_softcap_matches_cpu() {
        // Windowed (kv_start>0) + softcap paths through FA-vec kernels.
        let cases = [
            // (n_heads, n_kv, head_dim, seq, kv_start, softcap)
            (4, 2, 128, 65, 17, None),
            (8, 4, 128, 65, 33, Some(50.0f32)),
            (4, 2, 64, 48, 16, None),
            (8, 4, 64, 48, 8, Some(30.0)),
            (4, 4, 96, 40, 8, Some(50.0)),
            (4, 1, 256, 40, 12, Some(50.0)),
            (8, 4, 128, 33, 0, Some(50.0)), // softcap only
        ];
        std::env::set_var("FERROX_METAL_FA_VEC", "1");
        for (n_heads, n_kv_heads, head_dim, seq_len, kv_start, softcap) in cases {
            let q: Vec<f32> = (0..n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let cpu = cpu_gqa_ex(
                &q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_start, softcap,
            );
            let gpu = launch_gqa_decode_host_ex(
                &q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_start, softcap,
            )
            .expect("metal fa-vec window/softcap");
            assert_eq!(cpu.len(), gpu.len());
            for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let tol = 3e-3 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "hd={head_dim} seq={seq_len} ks={kv_start} sc={softcap:?} elem {i}: cpu={a} gpu={b}"
                );
            }
        }
    }

    fn cpu_rope_norm(vec: &mut [f32], pos: usize, theta: f32, freq_factors: Option<&[f32]>) {
        let dim = vec.len();
        let half = dim / 2;
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
            let angle = match freq_factors {
                Some(ff) => pos as f32 * freq / ff[i],
                None => pos as f32 * freq,
            };
            let (sin, cos) = angle.sin_cos();
            let a = vec[2 * i];
            let b = vec[2 * i + 1];
            vec[2 * i] = a * cos - b * sin;
            vec[2 * i + 1] = a * sin + b * cos;
        }
    }

    fn cpu_rope_neox(vec: &mut [f32], pos: usize, theta: f32, freq_factors: Option<&[f32]>) {
        let dim = vec.len();
        let half = dim / 2;
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
            let angle = match freq_factors {
                Some(ff) => pos as f32 * freq / ff[i],
                None => pos as f32 * freq,
            };
            let (sin, cos) = angle.sin_cos();
            let a = vec[i];
            let b = vec[i + half];
            vec[i] = a * cos - b * sin;
            vec[i + half] = a * sin + b * cos;
        }
    }

    fn assert_rope_parity(layout: MetalRopeLayout, with_ff: bool) {
        let n_heads = 3;
        let head_dim = 8;
        let pos = 5usize;
        let theta = 10000.0f32;
        let ff: Option<Vec<f32>> = if with_ff {
            Some((0..head_dim / 2).map(|i| 0.8 + i as f32 * 0.15).collect())
        } else {
            None
        };
        let mut cpu: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let mut gpu = cpu.clone();
        for h in 0..n_heads {
            let slice = &mut cpu[h * head_dim..(h + 1) * head_dim];
            match layout {
                MetalRopeLayout::Norm => cpu_rope_norm(slice, pos, theta, ff.as_deref()),
                MetalRopeLayout::Neox => cpu_rope_neox(slice, pos, theta, ff.as_deref()),
            }
        }
        launch_rope_heads_host(
            &mut gpu,
            n_heads,
            head_dim,
            layout,
            theta,
            pos,
            ff.as_deref(),
        )
        .expect("metal rope");
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "{layout:?} ff={with_ff} elem {i}: cpu={a} gpu={b} tol={tol}"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_norm_matches_cpu() {
        assert_rope_parity(MetalRopeLayout::Norm, false);
        assert_rope_parity(MetalRopeLayout::Norm, true);
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_neox_matches_cpu() {
        assert_rope_parity(MetalRopeLayout::Neox, false);
        assert_rope_parity(MetalRopeLayout::Neox, true);
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_gqa_prefill(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_q: usize,
        kv_prefix_len: usize,
    ) -> Vec<f32> {
        let q_width = n_heads * head_dim;
        let mut out = vec![0f32; n_q * q_width];
        for qi in 0..n_q {
            let causal_len = kv_prefix_len + qi + 1;
            let kv_elems = causal_len * n_kv_heads * head_dim;
            let row = cpu_gqa(
                &q[qi * q_width..(qi + 1) * q_width],
                &k_cache[..kv_elems],
                &v_cache[..kv_elems],
                n_heads,
                n_kv_heads,
                head_dim,
                causal_len,
            );
            out[qi * q_width..(qi + 1) * q_width].copy_from_slice(&row);
        }
        out
    }

    #[test]
    fn gqa_threadgroup_sizes_are_well_formed() {
        for seq in [1u32, 7, 32, 100, 512, 2048] {
            for hd in [8u32, 64, 128] {
                let pre = gqa_prefill_threadgroup_size(seq, hd);
                assert!(pre.is_power_of_two(), "prefill tg={pre} seq={seq} hd={hd}");
                assert!(pre >= 1);
                let dec = gqa_decode_threadgroup_size(seq, hd);
                assert_eq!(dec % 32, 0, "decode tg={dec} seq={seq} hd={hd}");
                assert!((dec / 32).is_power_of_two(), "decode nsg not pot: {dec}");
                assert!(dec >= 32);
            }
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_matches_cpu() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let n_q = 4;
        let kv_prefix_len = 2;
        let total = kv_prefix_len + n_q;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let cpu = cpu_gqa_prefill(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
        );
        let gpu = launch_gqa_prefill_host(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
        )
        .expect("metal gqa prefill");
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 2e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "elem {i}: cpu={a} gpu={b} tol={tol}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_vec_d128_matches_cpu() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 128;
        let n_q = 3;
        let kv_prefix_len = 5;
        let total = kv_prefix_len + n_q;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let cpu = cpu_gqa_prefill(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
        );
        let gpu = launch_gqa_prefill_host(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
        )
        .expect("metal gqa prefill fa-vec");
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 5e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "elem {i}: cpu={a} gpu={b} tol={tol}");
        }
    }

    /// Gemma-2 attends through the **prefill** FA-vec kernel at head_dim
    /// 256 with an attention-logit softcap, and nothing covered that: the
    /// only prefill parity test was d=128 without softcap, and every
    /// d=256 decode case fit inside a single 32-wide KV chunk. The kernel
    /// was dropping the upper half of every head and no test noticed.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_vec_softcap_matches_cpu() {
        std::env::set_var("FERROX_METAL_FA_VEC", "1");
        std::env::set_var("FERROX_METAL_FA_EXT", "0");
        // Every head dim the FA-vec prefill path claims to cover. d=64
        // and d=96 are the ones where fewer than 32 lanes own a float4 of
        // the output, so the lane masking is what these cases pin down --
        // an unmasked lane reads `sq4` past the end of the query and the
        // score is silently wrong for every token.
        // (n_heads, n_kv, head_dim, n_q, kv_prefix, softcap)
        let cases = [
            (8usize, 4usize, 256usize, 3usize, 5usize, Some(50.0f32)),
            (8, 4, 256, 16, 0, Some(50.0)),
            (8, 4, 256, 40, 9, Some(50.0)),
            (8, 4, 256, 3, 5, None),
            (8, 4, 128, 40, 9, Some(50.0)),
            (8, 4, 128, 33, 0, None),
            (9, 3, 64, 40, 9, Some(50.0)),
            (8, 4, 64, 65, 0, None),
            (8, 8, 64, 3, 5, None),
            (4, 2, 96, 40, 9, Some(50.0)),
            (4, 4, 96, 33, 7, None),
        ];
        for (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) in cases {
            let total = kv_prefix_len + n_q;
            let q: Vec<f32> = (0..n_q * n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();

            let q_width = n_heads * head_dim;
            let mut cpu = vec![0f32; n_q * q_width];
            for qi in 0..n_q {
                let causal_len = kv_prefix_len + qi + 1;
                let kv_elems = causal_len * n_kv_heads * head_dim;
                let row = cpu_gqa_ex(
                    &q[qi * q_width..(qi + 1) * q_width],
                    &k[..kv_elems],
                    &v[..kv_elems],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    causal_len,
                    0,
                    softcap,
                );
                cpu[qi * q_width..(qi + 1) * q_width].copy_from_slice(&row);
            }

            let gpu = launch_gqa_prefill_host_ex(
                &q,
                &k,
                &v,
                n_heads,
                n_kv_heads,
                head_dim,
                n_q,
                kv_prefix_len,
                softcap,
            )
            .expect("metal gqa prefill fa-vec d256");
            assert_eq!(cpu.len(), gpu.len());
            for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let tol = 5e-3 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "hd={head_dim} n_q={n_q} pre={kv_prefix_len} sc={softcap:?} elem {i}: cpu={a} gpu={b}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_ext_d64_matches_cpu() {
        std::env::set_var("FERROX_METAL_FA_VEC", "1");
        std::env::set_var("FERROX_METAL_FA_EXT", "1");
        // Shapes chosen around the MMA kernel's 8-row K/V granularity:
        // kv_valid = kv_prefix_len + n_q is 49 / 65 / 49 / 128 / 137 / 8 /
        // 128 / 64, covering both the padded tail (kv_valid % 8 != 0) and the
        // exact fit, at 1, 2 and 3 chunks of C=64 keys. The last two are the
        // prefix-cache shape: a long shared prefix with a short new batch, so
        // whole key blocks sit past `max_causal` and must contribute nothing.
        let cases = [
            (9usize, 3usize, 64usize, 40usize, 9usize, Some(50.0f32)),
            (8usize, 4usize, 64usize, 65usize, 0usize, None),
            (8usize, 4usize, 64usize, 40usize, 9usize, Some(50.0f32)),
            (8usize, 4usize, 64usize, 128usize, 0usize, None),
            (6usize, 2usize, 64usize, 130usize, 7usize, Some(30.0f32)),
            (4usize, 4usize, 64usize, 8usize, 0usize, None),
            (8usize, 4usize, 64usize, 8usize, 120usize, None),
            (8usize, 2usize, 64usize, 9usize, 55usize, Some(20.0f32)),
        ];
        for (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) in cases {
            let total = kv_prefix_len + n_q;
            let q: Vec<f32> = (0..n_q * n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let q_width = n_heads * head_dim;
            let mut cpu = vec![0f32; n_q * q_width];
            for qi in 0..n_q {
                let causal_len = kv_prefix_len + qi + 1;
                let kv_elems = causal_len * n_kv_heads * head_dim;
                let row = cpu_gqa_ex(
                    &q[qi * q_width..(qi + 1) * q_width],
                    &k[..kv_elems],
                    &v[..kv_elems],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    causal_len,
                    0,
                    softcap,
                );
                cpu[qi * q_width..(qi + 1) * q_width].copy_from_slice(&row);
            }
            let gpu = launch_gqa_prefill_host_ex(
                &q,
                &k,
                &v,
                n_heads,
                n_kv_heads,
                head_dim,
                n_q,
                kv_prefix_len,
                softcap,
            )
            .expect("fa_ext prefill");
            let mut max_diff = 0f32;
            let mut worst = (0usize, 0f32, 0f32);
            for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (i, *a, *b);
                }
            }
            let tol = 5e-3 * worst.1.abs().max(1.0);
            assert!(
                max_diff <= tol,
                "hd={head_dim} n_q={n_q} pre={kv_prefix_len} sc={softcap:?} max_diff={max_diff} worst={worst:?} tol={tol}"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_ext_matches_fa_vec_d64() {
        std::env::set_var("FERROX_METAL_FA_VEC", "1");
        let (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) =
            (9usize, 3usize, 64usize, 40usize, 9usize, Some(50.0f32));
        let total = kv_prefix_len + n_q;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        std::env::set_var("FERROX_METAL_FA_EXT", "0");
        let fa_vec = launch_gqa_prefill_host_ex(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
            softcap,
        )
        .expect("fa_vec");
        std::env::set_var("FERROX_METAL_FA_EXT", "1");
        let fa_ext = launch_gqa_prefill_host_ex(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
            softcap,
        )
        .expect("fa_ext");
        let mut max_diff = 0f32;
        let mut worst = (0usize, 0f32, 0f32);
        for (i, (a, b)) in fa_vec.iter().zip(fa_ext.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
                worst = (i, *a, *b);
            }
        }
        assert!(
            max_diff <= 1e-4,
            "fa_ext vs fa_vec max_diff={max_diff} worst={worst:?}"
        );
    }

    /// The MMA kernel against the scalar `dot`+`simd_sum` predecessor it
    /// replaces, on the same inputs. Both are `fa_ext`, so this isolates the
    /// MMA rewrite from every other difference (tiling, softmax, epilogue).
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_ext_mma_matches_scalar_d64() {
        std::env::set_var("FERROX_METAL_FA_VEC", "1");
        std::env::set_var("FERROX_METAL_FA_EXT", "1");
        // The MMA pipeline must exist: a compile failure here would otherwise
        // surface as a `.expect()` panic that reads like a device problem.
        {
            let shared = shared_metal().expect("metal device");
            ensure_pipeline(
                &shared.device,
                GQA_PREFILL_FA_EXT_MMA_D64_KERNEL_SRC,
                "gqa_prefill_fa_ext_mma_d64",
            )
            .expect("fa_ext MMA pipeline compiles");
        }
        let cases = [
            (9usize, 3usize, 64usize, 40usize, 9usize, Some(50.0f32)),
            (8usize, 4usize, 64usize, 128usize, 0usize, None),
            (6usize, 2usize, 64usize, 130usize, 7usize, Some(30.0f32)),
            (8usize, 8usize, 64usize, 65usize, 0usize, None),
            (8usize, 4usize, 64usize, 8usize, 120usize, None),
            (8usize, 2usize, 64usize, 9usize, 55usize, Some(20.0f32)),
        ];
        for (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) in cases {
            let total = kv_prefix_len + n_q;
            let q: Vec<f32> = (0..n_q * n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let run = |mma: &str| {
                std::env::set_var("FERROX_METAL_FA_MMA", mma);
                launch_gqa_prefill_host_ex(
                    &q,
                    &k,
                    &v,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    n_q,
                    kv_prefix_len,
                    softcap,
                )
                .expect("fa_ext")
            };
            let scalar = run("0");
            let mma = run("1");
            std::env::remove_var("FERROX_METAL_FA_MMA");
            let mut max_diff = 0f32;
            let mut worst = (0usize, 0f32, 0f32);
            for (i, (a, b)) in scalar.iter().zip(mma.iter()).enumerate() {
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (i, *a, *b);
                }
            }
            assert!(
                max_diff <= 1e-4,
                "mma vs scalar hd={head_dim} n_q={n_q} pre={kv_prefix_len} \
                 sc={softcap:?} max_diff={max_diff} worst={worst:?}"
            );
        }
    }

    /// The d=128 MMA kernel (Qwen3-0.6B / Phi-4-mini / Mistral shape) against
    /// the f32 CPU reference **and** against FA-vec, which is the only other
    /// kernel at this width and therefore the A/B baseline. Same shape sweep as
    /// the d=64 test: padded cache tails, exact 8-row fits, and the long-prefix
    /// / short-batch case where whole key blocks sit past `max_causal`.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_ext_mma_d128_matches_cpu_and_fa_vec() {
        std::env::set_var("FERROX_METAL_FA_VEC", "1");
        std::env::set_var("FERROX_METAL_FA_EXT", "1");
        {
            let shared = shared_metal().expect("metal device");
            ensure_pipeline(
                &shared.device,
                GQA_PREFILL_FA_EXT_MMA_D128_KERNEL_SRC,
                "gqa_prefill_fa_ext_mma_d128",
            )
            .expect("d128 fa_ext MMA pipeline compiles");
        }
        let cases = [
            (16usize, 8usize, 128usize, 40usize, 9usize, None),
            (16usize, 8usize, 128usize, 65usize, 0usize, None),
            (8usize, 4usize, 128usize, 128usize, 0usize, Some(50.0f32)),
            (6usize, 2usize, 128usize, 130usize, 7usize, Some(30.0f32)),
            (8usize, 4usize, 128usize, 8usize, 120usize, None),
            (8usize, 2usize, 128usize, 9usize, 55usize, Some(20.0f32)),
        ];
        for (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) in cases {
            let total = kv_prefix_len + n_q;
            let q: Vec<f32> = (0..n_q * n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let q_width = n_heads * head_dim;
            let mut cpu = vec![0f32; n_q * q_width];
            for qi in 0..n_q {
                let causal_len = kv_prefix_len + qi + 1;
                let kv_elems = causal_len * n_kv_heads * head_dim;
                let row = cpu_gqa_ex(
                    &q[qi * q_width..(qi + 1) * q_width],
                    &k[..kv_elems],
                    &v[..kv_elems],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    causal_len,
                    0,
                    softcap,
                );
                cpu[qi * q_width..(qi + 1) * q_width].copy_from_slice(&row);
            }
            let run = |mma: &str| {
                std::env::set_var("FERROX_METAL_FA_MMA", mma);
                launch_gqa_prefill_host_ex(
                    &q,
                    &k,
                    &v,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    n_q,
                    kv_prefix_len,
                    softcap,
                )
                .expect("d128 prefill")
            };
            let fa_vec = run("0");
            let mma = run("1");
            std::env::remove_var("FERROX_METAL_FA_MMA");
            let worst_of = |a: &[f32], b: &[f32]| {
                let mut max_diff = 0f32;
                let mut worst = (0usize, 0f32, 0f32);
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    let d = (x - y).abs();
                    if d > max_diff {
                        max_diff = d;
                        worst = (i, *x, *y);
                    }
                }
                (max_diff, worst)
            };
            let (d_cpu, w_cpu) = worst_of(&cpu, &mma);
            let tol = 5e-3 * w_cpu.1.abs().max(1.0);
            assert!(
                d_cpu <= tol,
                "mma vs cpu hd={head_dim} n_q={n_q} pre={kv_prefix_len} \
                 sc={softcap:?} max_diff={d_cpu} worst={w_cpu:?} tol={tol}"
            );
            let (d_vec, w_vec) = worst_of(&fa_vec, &mma);
            assert!(
                d_vec <= 1e-3,
                "mma vs fa_vec hd={head_dim} n_q={n_q} pre={kv_prefix_len} \
                 sc={softcap:?} max_diff={d_vec} worst={w_vec:?}"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_empty_prefix_matches_per_token_decode() {
        // n_q positions with no prior KV must match running decode GQA
        // at each causal length (same math as forward_batch).
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let n_q = 3;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.09).sin())
            .collect();
        let k: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.04).cos())
            .collect();
        let v: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.06).sin())
            .collect();
        let gpu = launch_gqa_prefill_host(&q, &k, &v, n_heads, n_kv_heads, head_dim, n_q, 0)
            .expect("metal gqa prefill");
        let q_width = n_heads * head_dim;
        for qi in 0..n_q {
            let causal = qi + 1;
            let kv_elems = causal * n_kv_heads * head_dim;
            let decode = launch_gqa_decode_host(
                &q[qi * q_width..(qi + 1) * q_width],
                &k[..kv_elems],
                &v[..kv_elems],
                n_heads,
                n_kv_heads,
                head_dim,
                causal,
            )
            .expect("metal gqa decode");
            for (i, (a, b)) in gpu[qi * q_width..(qi + 1) * q_width]
                .iter()
                .zip(decode.iter())
                .enumerate()
            {
                let tol = 1e-4 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "qi={qi} elem {i}: prefill={a} decode={b}"
                );
            }
        }
    }

    fn assert_rope_batch_parity(layout: MetalRopeLayout, with_ff: bool) {
        let n_heads = 3;
        let head_dim = 8;
        let n_tokens = 4;
        let base_pos = 2usize;
        let theta = 10000.0f32;
        let ff: Option<Vec<f32>> = if with_ff {
            Some((0..head_dim / 2).map(|i| 0.8 + i as f32 * 0.15).collect())
        } else {
            None
        };
        let mut cpu: Vec<f32> = (0..n_tokens * n_heads * head_dim)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let mut gpu = cpu.clone();
        for t in 0..n_tokens {
            let pos = base_pos + t;
            for h in 0..n_heads {
                let off = (t * n_heads + h) * head_dim;
                let slice = &mut cpu[off..off + head_dim];
                match layout {
                    MetalRopeLayout::Norm => cpu_rope_norm(slice, pos, theta, ff.as_deref()),
                    MetalRopeLayout::Neox => cpu_rope_neox(slice, pos, theta, ff.as_deref()),
                }
            }
        }
        launch_rope_heads_batch_host(
            &mut gpu,
            n_heads,
            head_dim,
            n_tokens,
            layout,
            theta,
            base_pos,
            ff.as_deref(),
        )
        .expect("metal rope batch");
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "{layout:?} batch ff={with_ff} elem {i}: cpu={a} gpu={b} tol={tol}"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_batch_norm_matches_cpu() {
        assert_rope_batch_parity(MetalRopeLayout::Norm, false);
        assert_rope_batch_parity(MetalRopeLayout::Norm, true);
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_batch_neox_matches_cpu() {
        assert_rope_batch_parity(MetalRopeLayout::Neox, false);
        assert_rope_batch_parity(MetalRopeLayout::Neox, true);
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn prefill_attn_block_matches_cpu_and_updates_kv() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let n_q = 3;
        let start_pos = 0usize;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.08).sin())
            .collect();
        let k: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.04).sin())
            .collect();
        let mut kv =
            MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 16).expect("alloc metal kv");
        let (attn, k_roped, v_roped) = launch_prefill_attn_block(
            &q,
            &k,
            &v,
            &mut kv,
            n_heads,
            n_q,
            MetalRopeLayout::Norm,
            10000.0,
            None,
            start_pos,
            None,
            true,
        )
        .expect("prefill attn");
        assert_eq!(kv.seq_len, n_q);
        assert_eq!(attn.len(), n_q * n_heads * head_dim);
        assert_eq!(k_roped.len(), n_q * n_kv_heads * head_dim);

        // CPU reference: per-token RoPE + causal GQA over growing cache.
        let mut k_cpu = k.clone();
        let v_cpu = v.clone();
        let mut q_cpu = q.clone();
        for t in 0..n_q {
            let pos = start_pos + t;
            for h in 0..n_heads {
                let off = (t * n_heads + h) * head_dim;
                cpu_rope_norm(&mut q_cpu[off..off + head_dim], pos, 10000.0, None);
            }
            for h in 0..n_kv_heads {
                let off = (t * n_kv_heads + h) * head_dim;
                cpu_rope_norm(&mut k_cpu[off..off + head_dim], pos, 10000.0, None);
            }
        }
        for (a, b) in k_cpu.iter().zip(k_roped.iter()) {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol);
        }
        for (a, b) in v_cpu.iter().zip(v_roped.iter()) {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol);
        }
        let cpu_attn = cpu_gqa_prefill(
            &q_cpu, &k_cpu, &v_cpu, n_heads, n_kv_heads, head_dim, n_q, 0,
        );
        for (i, (a, b)) in cpu_attn.iter().zip(attn.iter()).enumerate() {
            let tol = 2e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "attn elem {i}: cpu={a} gpu={b}");
        }
        let (k_dl, v_dl) = kv.tokens_host(0, n_q);
        for (i, (a, b)) in k_roped.iter().zip(k_dl.iter()).enumerate() {
            let tol = 2e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "k dl elem {i}: host={a} metal={b}");
        }
        for (i, (a, b)) in v_roped.iter().zip(v_dl.iter()).enumerate() {
            let tol = 2e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "v dl elem {i}: host={a} metal={b}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn q8_0_kv_prefill_matches_f16_path() {
        // elems/token = 2*64 = 128 (Q8_0 block-aligned).
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 64;
        let n_q = 2;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.04).sin())
            .collect();
        let mut kv_f16 =
            MetalKvBuffers::with_capacity_dtype(n_kv_heads, head_dim, 16, MetalKvDtype::F16)
                .expect("f16 kv");
        let mut kv_q8 =
            MetalKvBuffers::with_capacity_dtype(n_kv_heads, head_dim, 16, MetalKvDtype::Q8_0)
                .expect("q8 kv");
        assert_eq!(kv_q8.dtype(), MetalKvDtype::Q8_0);
        let (attn_f16, _, _) = launch_prefill_attn_block(
            &q,
            &k,
            &v,
            &mut kv_f16,
            n_heads,
            n_q,
            MetalRopeLayout::Norm,
            10000.0,
            None,
            0,
            None,
            false,
        )
        .expect("f16 prefill");
        let (attn_q8, _, _) = launch_prefill_attn_block(
            &q,
            &k,
            &v,
            &mut kv_q8,
            n_heads,
            n_q,
            MetalRopeLayout::Norm,
            10000.0,
            None,
            0,
            None,
            false,
        )
        .expect("q8 prefill");
        assert_eq!(attn_f16.len(), attn_q8.len());
        for (i, (a, b)) in attn_f16.iter().zip(attn_q8.iter()).enumerate() {
            // Q8_0 KV is lossy; keep a loose absolute+relative bound.
            let tol = 5e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "attn elem {i}: f16={a} q8={b} tol={tol}"
            );
        }
        let (k_dl, _) = kv_q8.tokens_host(0, n_q);
        assert_eq!(k_dl.len(), n_q * n_kv_heads * head_dim);
    }
}
