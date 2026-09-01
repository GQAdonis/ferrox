//! `mul_mm`: a batched quantized GEMM for CUDA -- the CUDA C source and
//! the per-quant-kind dispatch table.
//!
//! # UNRUN ON HARDWARE
//!
//! **No kernel in this module has ever executed on a GPU.** There is no
//! NVIDIA hardware in the environment it was written in, and this repo's
//! standing rule (`docs/plans/roadmap.md`, "CUDA stays at *must
//! compile*") is that a CUDA claim is worth nothing until someone
//! measures it. Nothing here may be described as a measured capability
//! in `docs/FEATURES.md` or `docs/MODELS.md` until
//! `cargo test -p ferrox-cuda --features cuda -- --ignored` has been run
//! on a real device and the result written down.
//!
//! What *is* established, and how:
//!
//! 1. **It is a port, not a design.** The arithmetic comes from
//!    `ferrox-metal`'s `mul_mm_sg_impl` (`crates/ferrox-metal/src/gpu.rs`),
//!    which is at parity with llama.cpp on Metal and has goldens. The
//!    per-kind unpack functions below are line-for-line transcriptions of
//!    that file's `Q8_0Dequant` / `Q4_0Dequant` functors, which are
//!    themselves llama's `dequantize_q8_0` / `dequantize_q4_0`.
//! 2. **It has a scalar twin.** [`crate::mul_mm_ref`] emulates this
//!    kernel on the CPU -- same tiling, same clamping, same index
//!    arithmetic, same accumulation order -- and its tests, which run in
//!    the default (no-GPU) build, hold it against an independent
//!    dequantize-then-GEMM built on `ferrox_quant`. A transcription
//!    error in the index math or the unpack shows up there.
//! 3. **Geometry cannot drift.** Every tile constant in the emitted CUDA
//!    is `#define`d from the Rust constants in this module, which are
//!    the same constants the twin uses. The kernel and its twin cannot
//!    disagree about a tile size; they can only disagree about the ~10
//!    lines of unpack expression, which is what (1) and (2) cover.
//!
//! 4. **The emitted C itself was executed.**
//!    `tools/mul_mm_host_check/run.sh` compiles each generated `.cu`
//!    against a keyword-and-barrier shim and runs it on the host CPU --
//!    one real thread per CUDA thread, a counting barrier for
//!    `__syncthreads()`, one block at a time so `__shared__` behaves --
//!    then compares it to the twin. Result on 2026-09-01, macOS/clang,
//!    both kinds, three shapes (exact tiles, partial on both axes,
//!    narrow batch): **zero mismatches, bit for bit**, including 1,458
//!    positions where a degenerate f16 scale made both sides NaN
//!    together. Deleting one term of the CUDA-only index arithmetic
//!    makes it fail, so it is a check and not a formality.
//!
//! What none of that covers, and what only hardware can settle: that
//! NVRTC accepts the source (clang and NVRTC are different front ends),
//! that the barrier placement survives a real warp scheduler, that the
//! launch configuration is valid on the target device, and what any of
//! it costs. A real GPU also contracts `acc += a * b` into an FMA, so
//! on-device results will be *close to* rather than equal to the twin's;
//! the hardware test compares with a relative tolerance for that reason.
//!
//! # Why this shape
//!
//! `ferrox-cuda` had no matrix-matrix product of any kind, so a batched
//! prefill decomposed into one matvec per position
//! (`crates/ferrox-core/src/weight_matrix.rs`, the CUDA batch arm). This
//! is the naive tiled `mul_mm` half of
//! `docs/plans/llama-cpp-gap-inventory.md` §2.7; the `dp4a` integer path
//! (llama's `mmq.cu`) is explicitly *not* attempted here.
//!
//! Unlike Metal's version this uses no matrix-fragment intrinsics
//! (`wmma`/`mma`): the K-loop is a plain fp32 FMA over shared-memory
//! tiles. That costs the constant factor and buys a kernel whose
//! arithmetic a CPU twin can reproduce exactly, which is the only kind
//! of correctness available without a device.

/// Rows of the weight matrix per threadblock tile.
pub const BM: usize = 64;
/// Tokens (batch entries) per threadblock tile.
pub const BN: usize = 32;
/// K-elements consumed per tile step. Must be a multiple of [`SUB`], and
/// every real quantized row length is a multiple of 32, so a K-loop that
/// steps 32 never straddles a partial block.
pub const BK: usize = 32;
/// Rows of the output micro-tile each thread owns.
pub const TM: usize = 4;
/// Columns of the output micro-tile each thread owns.
pub const TN: usize = 2;
/// Threads per block. `BM/TM * BN/TN` -- one thread per micro-tile.
pub const THREADS: usize = (BM / TM) * (BN / TN);
/// Elements produced by one call to the per-kind unpack function. This
/// is llama's (and `ferrox-metal`'s) sub-block granularity: `il` selects
/// which 16 consecutive elements of a super-block to decode.
pub const SUB: usize = 16;

// Compile-time geometry gates. These are the invariants the kernel's
// `tx`/`ty` decomposition and its A-tile loader assume; a retune that
// broke one would otherwise produce a kernel that launches and is
// wrong, so they fail the build rather than a test.
const _: () = assert!(THREADS == (BM / TM) * (BN / TN));
const _: () = assert!(BM.is_multiple_of(TM) && BN.is_multiple_of(TN));
const _: () = assert!(
    BK.is_multiple_of(SUB),
    "the K-tile must be whole sub-blocks"
);
const _: () = assert!(
    BM * (BK / SUB) <= THREADS,
    "the A-tile loader uses a prefix of the block"
);
const _: () = assert!(THREADS <= 1024, "CUDA caps a block at 1024 threads");

/// One quantized weight format the GEMM can consume.
///
/// Adding a format is a row in [`KINDS`] plus a `dequant_src` snippet --
/// never a second copy of the GEMM body. That is the seam
/// `ferrox-metal` proved: its `mul_mm_sg_impl` is one templated body
/// with seven `Dequant` functors, written that way *because* the
/// previous copy-per-format generation is how
/// `gqa_prefill_fa_vec_d256` ended up handling half a head.
#[derive(Debug, Clone, Copy)]
pub struct MulMmKind {
    /// GGUF quant name, for error messages.
    pub name: &'static str,
    /// NVRTC module cache key. Must be unique per kind.
    pub module_name: &'static str,
    /// `__global__` entry point name inside that module.
    pub fn_name: &'static str,
    /// On-disk stride of one super-block.
    pub block_bytes: usize,
    /// Elements one super-block decodes to.
    pub block_elems: usize,
    /// CUDA C defining
    /// `void ferrox_dequant_sub(const unsigned char* xb, int il, float* reg)`,
    /// writing `SUB` floats: the elements at `[SUB*il, SUB*il + SUB)`
    /// of the super-block at `xb`, in ascending element order.
    pub dequant_src: &'static str,
    /// The scalar twin of `dequant_src`: the same arithmetic in Rust,
    /// on the host, in the same order. It sits in this struct rather
    /// than in a parallel table so a kind cannot be added without one --
    /// the untestable half and the testable half are the same row.
    pub dequant_twin: fn(xb: &[u8], il: usize, reg: &mut [f32; SUB]),
}

impl MulMmKind {
    /// 16-value sub-blocks per super-block -- llama's `nl` template
    /// argument (2 for the 32-element legacy formats, 16 for the
    /// 256-element K-quants).
    pub const fn nl(&self) -> usize {
        self.block_elems / SUB
    }
}

/// Q8_0: `half d`, then 32 `int8` quants. Transcribed from
/// `ferrox-metal`'s `Q8_0Dequant::get`.
pub const Q8_0: MulMmKind = MulMmKind {
    name: "Q8_0",
    module_name: "ferrox_mul_mm_q8_0",
    fn_name: "q8_0_mul_mm",
    block_bytes: 34,
    block_elems: 32,
    dequant_src: r#"
__device__ __forceinline__ void ferrox_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = ferrox_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const signed char* qs = (const signed char*)(xb + 2) + 16 * il;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        reg[i] = (float)qs[i] * d;
    }
}
"#,
    dequant_twin: dequant_sub_q8_0,
};

/// Scalar twin of [`Q8_0`]'s `dequant_src`. Read the two side by side:
/// the loop bounds, the pointer offset and the multiply order are the
/// same statements in two languages.
fn dequant_sub_q8_0(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_to_f32(u16::from(xb[0]) | (u16::from(xb[1]) << 8));
    let qs = &xb[2 + SUB * il..2 + SUB * il + SUB];
    for (r, q) in reg.iter_mut().zip(qs.iter()) {
        *r = f32::from(*q as i8) * d;
    }
}

/// Q4_0: `half d`, then 16 bytes holding 32 nibbles (low nibble of byte
/// `j` is element `j`, high nibble is element `j + 16`), each biased by
/// -8. Transcribed from `ferrox-metal`'s `Q4_0Dequant::get`, which
/// composes llama's `uint16` pair reads out of bytes because a GGUF
/// tensor row is only 2-byte aligned.
///
/// Note the bias: `d1 * q + (-8 * d)`, not `d * (q - 8)`. That is
/// llama's order and the twin mirrors it, so the two agree bit for bit
/// where fp32 rounding would otherwise separate them.
pub const Q4_0: MulMmKind = MulMmKind {
    name: "Q4_0",
    module_name: "ferrox_mul_mm_q4_0",
    fn_name: "q4_0_mul_mm",
    block_bytes: 18,
    block_elems: 32,
    dequant_src: r#"
__device__ __forceinline__ void ferrox_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = ferrox_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const unsigned char* qs = xb + 2;
    const float d1 = il ? d / 16.0f : d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0f * d;
    const unsigned short mask0 = il ? 0x00F0 : 0x000F;
    const unsigned short mask1 = (unsigned short)(mask0 << 8);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned short w =
            (unsigned short)qs[2 * i] | ((unsigned short)qs[2 * i + 1] << 8);
        reg[2 * i + 0] = d1 * (float)(w & mask0) + md;
        reg[2 * i + 1] = d2 * (float)(w & mask1) + md;
    }
}
"#,
    dequant_twin: dequant_sub_q4_0,
};

/// Scalar twin of [`Q4_0`]'s `dequant_src`.
fn dequant_sub_q4_0(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_to_f32(u16::from(xb[0]) | (u16::from(xb[1]) << 8));
    let qs = &xb[2..2 + 16];
    let d1 = if il != 0 { d / 16.0 } else { d };
    let d2 = d1 / 256.0;
    let md = -8.0 * d;
    let mask0: u16 = if il != 0 { 0x00F0 } else { 0x000F };
    let mask1: u16 = mask0 << 8;
    for i in 0..8 {
        let w = u16::from(qs[2 * i]) | (u16::from(qs[2 * i + 1]) << 8);
        reg[2 * i] = d1 * f32::from(w & mask0) + md;
        reg[2 * i + 1] = d2 * f32::from(w & mask1) + md;
    }
}

/// Scalar twin of the CUDA `ferrox_f16_to_f32` in `F16_SRC`: the same
/// bit surgery, including the `exp == 31` NaN/Inf arm. `ldexpf(m, e)`
/// with an exact power of two is a multiply, so this is bit-identical
/// rather than merely close.
///
/// Held against `half::f16` by a test, which is what makes it a twin of
/// something and not a second guess.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);
    let scale = if exp == 0 {
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 31 {
        if mant != 0 {
            f32::from_bits(0x7fc0_0000)
        } else {
            f32::from_bits(0x7f80_0000)
        }
    } else {
        ((mant | 0x400) as f32) * 2f32.powi(exp as i32 - 25)
    };
    if sign != 0 {
        -scale
    } else {
        scale
    }
}

/// The dispatch table. A caller looks up by GGUF quant name; a new
/// format is one row here.
pub const KINDS: &[MulMmKind] = &[Q8_0, Q4_0];

/// Looks up a kind by its GGUF quant name (`"Q4_0"`, `"Q8_0"`).
/// `None` means this GEMM does not implement that format -- the caller
/// must fall back and say so, never compute something else.
pub fn kind_by_name(name: &str) -> Option<&'static MulMmKind> {
    KINDS.iter().find(|k| k.name == name)
}

/// f16 -> f32 by explicit bit surgery, shared with the matvec kernels in
/// `gpu.rs`. NVRTC has `__half` available but only with the CUDA headers
/// on the include path, which this crate deliberately does not require.
const F16_SRC: &str = r#"
__device__ __forceinline__ float ferrox_f16_to_f32(unsigned short bits) {
    unsigned int sign = (bits >> 15) & 0x1u;
    unsigned int exp = (bits >> 10) & 0x1Fu;
    unsigned int mant = bits & 0x3FFu;
    float scale;
    if (exp == 0) {
        scale = ldexpf((float)mant, -24);
    } else if (exp == 31) {
        scale = mant ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
    } else {
        scale = ldexpf((float)(mant | 0x400), (int)exp - 25);
    }
    return sign ? -scale : scale;
}
"#;

/// The GEMM body, identical for every quant kind.
///
/// `src0` is the quantized weight matrix, `n_rows` rows of `row_bytes`
/// each. `src1` is `batch` activation rows of `n_cols` f32. `dst` is
/// written as `dst[token * n_rows + row]`, which is the layout
/// `WeightMatrix::apply_batch` already returns.
///
/// Every `FX_*` name is `#define`d by [`kernel_src`] from this module's
/// Rust constants, so the emitted kernel and [`crate::mul_mm_ref`]
/// cannot disagree about geometry.
///
/// The out-of-range row clamp is llama's trick, kept from
/// `mul_mm_sg_impl`: a lane whose row does not exist re-reads the last
/// valid row rather than branching, so the load loop stays uniform, and
/// its result is thrown away by the bounds check at the store. Reading
/// a real row also means the dequant never touches unmapped bytes.
const BODY_SRC: &str = r#"
extern "C" __global__ void FX_FN_NAME(
    const unsigned char* __restrict__ src0,
    const float* __restrict__ src1,
    float* __restrict__ dst,
    int n_rows,
    int n_cols,
    int batch,
    int row_bytes
) {
    __shared__ float sa[FX_BK][FX_BM];
    __shared__ float sb[FX_BK][FX_BN];

    const int r0 = blockIdx.y * FX_BM;
    const int r1 = blockIdx.x * FX_BN;
    const int tid = threadIdx.x;

    // Micro-tile owner: `tx` walks rows, `ty` walks tokens.
    const int tx = tid % (FX_BM / FX_TM);
    const int ty = tid / (FX_BM / FX_TM);

    float acc[FX_TN][FX_TM];
#pragma unroll
    for (int n = 0; n < FX_TN; n++) {
#pragma unroll
        for (int m = 0; m < FX_TM; m++) {
            acc[n][m] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < n_cols; k0 += FX_BK) {
        // Guards the previous iteration's reads of sa/sb.
        __syncthreads();

        // A-tile: one thread decodes one FX_SUB-element sub-block, so
        // FX_BM * (FX_BK / FX_SUB) threads cover the tile. Stored
        // k-major so the K-loop below reads one row of sa per step.
        if (tid < FX_BM * (FX_BK / FX_SUB)) {
            const int lr = tid / (FX_BK / FX_SUB);
            const int ils = tid % (FX_BK / FX_SUB);
            int row = r0 + lr;
            if (row >= n_rows) {
                row = n_rows - 1;
            }
            const unsigned char* rp =
                src0 + (size_t)row * (size_t)row_bytes;
            const int sub = (k0 / FX_SUB) + ils;
            float reg[FX_SUB];
            ferrox_dequant_sub(
                rp + (size_t)(sub / FX_NL) * (size_t)FX_BLOCK_BYTES,
                sub % FX_NL,
                reg);
#pragma unroll
            for (int i = 0; i < FX_SUB; i++) {
                sa[FX_SUB * ils + i][lr] = reg[i];
            }
        }

        // B-tile: consecutive threads read consecutive k of one token.
        // Tokens past the end are zero-filled rather than skipped, so
        // the K-loop needs no per-token predicate.
        for (int idx = tid; idx < FX_BK * FX_BN; idx += FX_THREADS) {
            const int j = idx / FX_BK;
            const int kk = idx % FX_BK;
            const int col = r1 + j;
            sb[kk][j] = (col < batch)
                ? src1[(size_t)col * (size_t)n_cols + (size_t)(k0 + kk)]
                : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < FX_BK; kk++) {
            float a[FX_TM];
            float b[FX_TN];
#pragma unroll
            for (int m = 0; m < FX_TM; m++) {
                a[m] = sa[kk][tx * FX_TM + m];
            }
#pragma unroll
            for (int n = 0; n < FX_TN; n++) {
                b[n] = sb[kk][ty * FX_TN + n];
            }
#pragma unroll
            for (int n = 0; n < FX_TN; n++) {
#pragma unroll
                for (int m = 0; m < FX_TM; m++) {
                    acc[n][m] += a[m] * b[n];
                }
            }
        }
    }

    for (int n = 0; n < FX_TN; n++) {
        const int col = r1 + ty * FX_TN + n;
        if (col >= batch) {
            continue;
        }
        for (int m = 0; m < FX_TM; m++) {
            const int row = r0 + tx * FX_TM + m;
            if (row < n_rows) {
                dst[(size_t)col * (size_t)n_rows + (size_t)row] = acc[n][m];
            }
        }
    }
}
"#;

/// Emits the complete CUDA C translation unit for one quant kind.
///
/// Deterministic and side-effect free, which is what lets the tests in
/// [`crate::mul_mm_ref`] assert against it without a device.
pub fn kernel_src(kind: &MulMmKind) -> String {
    let defines = format!(
        "#define FX_BM {}\n\
         #define FX_BN {}\n\
         #define FX_BK {}\n\
         #define FX_TM {}\n\
         #define FX_TN {}\n\
         #define FX_THREADS {}\n\
         #define FX_SUB {}\n\
         #define FX_NL {}\n\
         #define FX_BLOCK_BYTES {}\n",
        BM,
        BN,
        BK,
        TM,
        TN,
        THREADS,
        SUB,
        kind.nl(),
        kind.block_bytes,
    );
    let body = BODY_SRC.replace("FX_FN_NAME", kind.fn_name);
    format!("{defines}{F16_SRC}{}{body}", kind.dequant_src)
}

/// Why a `mul_mm` dispatch was refused. Named rather than silent: a
/// shape this kernel cannot do must fall back to a path that can, and
/// the caller has to be able to say which.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MulMmUnsupported {
    /// `n_cols` is not a multiple of the K-tile. Every real GGUF row
    /// length is a multiple of 32, so this means a synthetic shape.
    ColsNotTileAligned { n_cols: usize, tile: usize },
    /// `n_cols` is not a whole number of super-blocks for this kind.
    ColsNotBlockAligned {
        n_cols: usize,
        block_elems: usize,
        kind: &'static str,
    },
    /// `row_bytes` does not match `n_cols` worth of super-blocks.
    RowBytesMismatch {
        row_bytes: usize,
        expected: usize,
        kind: &'static str,
    },
    /// The weight buffer is not `n_rows * row_bytes`.
    WeightsTooSmall { got: usize, want: usize },
    /// The activation buffer is not `batch * n_cols`.
    ActivationsTooSmall { got: usize, want: usize },
    /// A zero-sized dispatch. Not an error the caller must handle
    /// specially, but not something to launch a grid for either.
    EmptyShape,
}

impl std::fmt::Display for MulMmUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColsNotTileAligned { n_cols, tile } => {
                write!(f, "mul_mm: n_cols {n_cols} is not a multiple of the K-tile {tile}")
            }
            Self::ColsNotBlockAligned {
                n_cols,
                block_elems,
                kind,
            } => write!(
                f,
                "mul_mm: n_cols {n_cols} is not a whole number of {kind} blocks ({block_elems} elems)"
            ),
            Self::RowBytesMismatch {
                row_bytes,
                expected,
                kind,
            } => write!(
                f,
                "mul_mm: row_bytes {row_bytes} does not match {expected} for {kind} at this n_cols"
            ),
            Self::WeightsTooSmall { got, want } => {
                write!(f, "mul_mm: weight buffer is {got} bytes, needs {want}")
            }
            Self::ActivationsTooSmall { got, want } => {
                write!(f, "mul_mm: activation buffer is {got} floats, needs {want}")
            }
            Self::EmptyShape => write!(f, "mul_mm: empty shape"),
        }
    }
}

impl std::error::Error for MulMmUnsupported {}

/// The shape checks the kernel's index arithmetic assumes, in one place
/// so the launch path and the scalar twin cannot check different things.
pub fn validate_shape(
    kind: &MulMmKind,
    weights_len: usize,
    x_len: usize,
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
) -> Result<(), MulMmUnsupported> {
    if n_rows == 0 || n_cols == 0 || batch == 0 {
        return Err(MulMmUnsupported::EmptyShape);
    }
    if !n_cols.is_multiple_of(BK) {
        return Err(MulMmUnsupported::ColsNotTileAligned { n_cols, tile: BK });
    }
    if !n_cols.is_multiple_of(kind.block_elems) {
        return Err(MulMmUnsupported::ColsNotBlockAligned {
            n_cols,
            block_elems: kind.block_elems,
            kind: kind.name,
        });
    }
    let expected_row_bytes = (n_cols / kind.block_elems) * kind.block_bytes;
    if row_bytes != expected_row_bytes {
        return Err(MulMmUnsupported::RowBytesMismatch {
            row_bytes,
            expected: expected_row_bytes,
            kind: kind.name,
        });
    }
    let want_weights = n_rows * row_bytes;
    if weights_len < want_weights {
        return Err(MulMmUnsupported::WeightsTooSmall {
            got: weights_len,
            want: want_weights,
        });
    }
    let want_x = batch * n_cols;
    if x_len < want_x {
        return Err(MulMmUnsupported::ActivationsTooSmall {
            got: x_len,
            want: want_x,
        });
    }
    Ok(())
}

/// Whether a batched dispatch of this shape is worth a GEMM at all.
///
/// One token is a matvec, and `gpu.rs`'s matvec kernels are the arm that
/// has actually run on hardware; sending a single row through a 64x32
/// tile would waste 31 of every 32 output columns. The caller should
/// keep using `apply_gpu` below this threshold.
pub fn worth_a_gemm(batch: usize) -> bool {
    // `.max(2)` so this stays "never a single token" even if the tile
    // width is retuned downward.
    batch >= (BN / 4).max(2)
}

/// Grid dimensions for a dispatch, shared by the launch path and the
/// twin's block loop so they enumerate exactly the same tiles.
pub fn grid_dims(n_rows: usize, batch: usize) -> (usize, usize) {
    (batch.div_ceil(BN), n_rows.div_ceil(BM))
}
