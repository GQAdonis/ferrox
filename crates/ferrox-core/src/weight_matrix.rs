//! `WeightMatrix`: a weight matrix that may live either as plain f32
//! (small dims, embeddings, synthetic test weights) or as raw
//! Q8_0/Q4_0 block bytes loaded straight from a GGUF file, with no f32
//! expansion at load time. This is what lets ferrox load a
//! multi-billion-parameter checkpoint without first blowing it up 4x
//! in RAM: the loader (ferrox-models) hands tensors over still
//! quantized, and every matmul call here dispatches to the fused
//! dequant+dot kernels in ferrox-quant.

use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use crate::tensor::Tensor;

type Q4kRepackCache = Mutex<HashMap<(usize, usize), Arc<[u8]>>>;
type Q8x4RepackCache = Mutex<HashMap<(usize, usize), Arc<[u8]>>>;

/// Process-wide cache of interleaved Q4_K (`block_q4_Kx8`) bytes.
/// Keyed by canonical weight buffer pointer + row count so mmap stays
/// the source of truth and hot matrices share one repack.
fn q4k_repack_cache() -> &'static Q4kRepackCache {
    static CACHE: OnceLock<Q4kRepackCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_or_repack_q4k(data: &[u8], rows: usize, cols: usize) -> Arc<[u8]> {
    let key = (data.as_ptr() as usize, rows);
    {
        let cache = q4k_repack_cache().lock().unwrap();
        if let Some(hit) = cache.get(&key) {
            return Arc::clone(hit);
        }
    }
    let interleave = ferrox_quant::q4_kx8_interleave();
    let packed = ferrox_quant::pack_q4_k_matrix_x8(data, rows, cols, interleave);
    let arc: Arc<[u8]> = Arc::from(packed.into_boxed_slice());
    let mut cache = q4k_repack_cache().lock().unwrap();
    // Another thread may have won the race; prefer the existing entry.
    Arc::clone(cache.entry(key).or_insert_with(|| Arc::clone(&arc)))
}

/// Process-wide cache of interleaved Q8_0 (`block_q8_0x4`) bytes.
fn q8x4_repack_cache() -> &'static Q8x4RepackCache {
    static CACHE: OnceLock<Q8x4RepackCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_or_repack_q8x4(data: &[u8], rows: usize, cols: usize) -> Arc<[u8]> {
    let key = (data.as_ptr() as usize, rows);
    {
        let cache = q8x4_repack_cache().lock().unwrap();
        if let Some(hit) = cache.get(&key) {
            return Arc::clone(hit);
        }
    }
    let packed =
        ferrox_quant::pack_q8_0_matrix_x4(data, rows, cols, ferrox_quant::Q8_0X4_INTERLEAVE);
    let arc: Arc<[u8]> = Arc::from(packed.into_boxed_slice());
    let mut cache = q8x4_repack_cache().lock().unwrap();
    Arc::clone(cache.entry(key).or_insert_with(|| Arc::clone(&arc)))
}

/// Backing storage for a quantized weight matrix's raw bytes: either an
/// owned buffer (synthetic/test weights, or any tensor that had to be
/// copied for some other reason) or a zero-copy view into a shared
/// memory-mapped GGUF file. This is the fix for the "loader read
/// everything into a fresh Vec<u8>" inefficiency: a real checkpoint's
/// resident memory should be the mmap itself, not a second copy of it,
/// which is how llama.cpp's mmap-based loader both avoid
/// doubling a multi-hundred-gigabyte checkpoint's memory footprint.
pub enum WeightBytes {
    Owned(Vec<u8>),
    Mapped {
        mmap: Arc<memmap2::Mmap>,
        range: Range<usize>,
    },
    /// A sub-range of a shared, lease-style buffer (e.g. one matrix
    /// inside an `ferrox_core::expert_store::ExpertLease`'s combined
    /// gate/up/down bytes). Holding the `Arc` here is exactly what
    /// makes the store's lease pinning structural: as long as any
    /// `WeightMatrix` built over these bytes is alive, the cache entry's
    /// strong count stays >1 and eviction cannot reuse it.
    Shared {
        buf: Arc<Vec<u8>>,
        range: Range<usize>,
    },
}

impl WeightBytes {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            WeightBytes::Owned(v) => v,
            WeightBytes::Mapped { mmap, range } => &mmap[range.clone()],
            WeightBytes::Shared { buf, range } => &buf[range.clone()],
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True if this is a zero-copy mmap view rather than an owned
    /// heap allocation -- useful for tests/diagnostics asserting that
    /// the loader actually took the zero-copy path.
    pub fn is_mapped(&self) -> bool {
        matches!(self, WeightBytes::Mapped { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    Q8_0,
    Q4_0,
    /// The dominant real-world GGUF quantization formats (most
    /// published checkpoints ship as Q4_K_M or similar K-quant mixes,
    /// not the legacy Q4_0/Q8_0 formats above). See
    /// `ferrox_quant`'s module docs for the block layout and
    /// independent Python cross-validation.
    Q4K,
    Q5K,
    Q6K,
    /// The two more-aggressive K-quant tiers, used in Q2_K/Q3_K_M/
    /// Q3_K_L-style quant mixes (the far more common Q4_K_M/Q5_K_M
    /// mixes only combine with Q6_K, already covered above). See
    /// `ferrox_quant`'s module docs and independent Python
    /// cross-validation.
    Q2K,
    Q3K,
    /// Legacy, largely-obsolete-for-new-releases formats, still
    /// occasionally encountered. See `ferrox_quant`'s module docs;
    /// byte layouts verified against real `ggml-common.h` source.
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_1,
    /// Non-linear ("codebook") quants: a 4-bit index maps through a
    /// shared 16-entry signed lookup table instead of a linear
    /// `nibble*scale+min` transform. See `ferrox_quant`'s module docs
    /// and independent Python cross-validation.
    IQ4NL,
    IQ4XS,
    /// The codebook-grid low-bit formats used throughout published
    /// "Dynamic" low-bit GGUFs of large MoE models (grid-table
    /// magnitudes + shared sign patterns; scalar kernels only so far).
    /// See `ferrox_quant`'s module docs and the ggml-cross-validated
    /// independent Python reference.
    IQ1S,
    IQ2XXS,
    IQ3XXS,
    /// GGUF *block*-MXFP4 (17-byte interleaved blocks, ggml tag 39) --
    /// not the same layout as `WeightMatrix::Mxfp4`'s two-buffer
    /// safetensors form, though the math is identical. Scalar kernel
    /// only so far.
    Mxfp4Gguf,
}

/// A `ferrox_cuda::gpu::launch_*_matvec` function pointer's signature
/// -- named here purely to keep `apply_gpu`'s CUDA per-kind dispatch
/// table readable (all five real kernels share this exact signature).
#[cfg(feature = "cuda")]
type CudaMatvecLaunchFn =
    fn(&[u8], &[f32], usize, usize, usize) -> Result<Vec<f32>, ferrox_cuda::gpu::CudaError>;

/// Metal matvec launch signature (`weights`/`x` borrowed; row block
/// count is derived inside `ferrox_metal::gpu`).
#[cfg(feature = "metal")]
type MetalMatvecLaunchFn =
    fn(&[u8], &[f32], usize, usize) -> Result<Vec<f32>, ferrox_metal::gpu::MetalError>;

/// Whether dense [`WeightMatrix::apply`] / [`WeightMatrix::apply_batch`]
/// should try Metal first (when built with `--features metal`).
///
/// - `FERROX_METAL=0|false|off|cpu` — force CPU
/// - `FERROX_METAL=1|true|on|metal` — force Metal attempt
/// - unset / `auto` — Metal when [`ferrox_metal::gpu::probe`] finds a device
///
/// Decision is cached for the process lifetime (env read once).
#[cfg(feature = "metal")]
pub fn metal_dense_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("FERROX_METAL").ok().as_deref() {
        Some("0") | Some("false") | Some("off") | Some("cpu") => false,
        Some("1") | Some("true") | Some("on") | Some("metal") => true,
        _ => ferrox_metal::gpu::probe().is_some(),
    })
}

/// Whether dense [`WeightMatrix::apply`] should try CUDA first (when
/// built with `--features cuda`).
///
/// - `FERROX_CUDA=0|false|off|cpu` — force skip CUDA dense
/// - `FERROX_CUDA=1|true|on|cuda` — force CUDA attempt
/// - unset / `auto` — CUDA when a device probe succeeds
#[cfg(feature = "cuda")]
pub fn cuda_dense_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("FERROX_CUDA").ok().as_deref() {
        Some("0") | Some("false") | Some("off") | Some("cpu") => false,
        Some("1") | Some("true") | Some("on") | Some("cuda") => true,
        _ => ferrox_cuda::gpu::probe().is_some(),
    })
}

/// Whether CPU Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K matvec should quantize the
/// activation to int8 and use the integer `vec_dot` path. Q4_K
/// additionally lazy-repacks into interleaved `block_q4_Kx8` for 8-wide
/// GEMV; Q8_0 into `block_q8_0x4` for 4-wide GEMV.
///
/// Off by default *as a library*, and turned on by both binaries (see
/// `ferrox_core::threads`'s siblings in `ferrox-cli`/`ferrox-server`,
/// which set `FERROX_CPU_INT_DOT=1` unless the caller already chose).
/// The split is deliberate: this is what llama.cpp's CPU backend does
/// unconditionally -- quantize the activation to Q8, run integer
/// `vec_dot` -- and it is worth 28% of CPU decode on Host B
/// (Qwen2.5-0.5B Q8_0, `-ngl 0 -t 6`: 58.0 -> 80.5 tok/s). But it also
/// perturbs results below the f32 reference's precision, and this
/// crate's golden cross-validation against the independent NumPy
/// reference asserts exact agreement. So the *inference product*
/// defaults to fast and the *library default* stays reference-exact.
pub fn cpu_int_dot_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("FERROX_CPU_INT_DOT").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}

/// Sets `FERROX_CPU_INT_DOT=1` unless the caller already expressed a
/// preference. Call from a binary's startup, before any worker threads
/// exist. See [`cpu_int_dot_enabled`] for why the default lives here
/// rather than in the getter.
///
/// # Safety
/// Must be called while the process is still single-threaded, since it
/// mutates the process environment.
pub unsafe fn default_cpu_int_dot_on() {
    if std::env::var_os("FERROX_CPU_INT_DOT").is_none() {
        unsafe { std::env::set_var("FERROX_CPU_INT_DOT", "1") };
    }
}

pub enum WeightMatrix {
    F32(Tensor),
    Quantized {
        data: WeightBytes,
        rows: usize,
        cols: usize,
        kind: QuantKind,
    },
    /// MXFP4 (OCP Microscaling 4-bit float, Kimi K3's real routed-expert
    /// format): unlike every `Quantized` kind above, which store one
    /// interleaved block buffer per row, Kimi K3's real checkpoint
    /// stores the packed 4-bit codes and per-group E8M0 scales as two
    /// *separate* tensors (confirmed against a real shard header, see
    /// `ferrox_quant`'s MXFP4 module docs) -- so this variant holds two
    /// independently zero-copy-mappable buffers instead of `Quantized`'s
    /// single `data` buffer. `apply`/`apply_batch` dispatch to
    /// `ferrox_quant::dot_mxfp4_row_f32`, which reads directly from
    /// these buffers without ever materializing a dequantized f32 copy
    /// of the whole matrix -- the same zero-copy-mmap-plus-fused-dot
    /// discipline as every `Quantized` kind, letting a real MXFP4
    /// checkpoint's resident memory stay close to its on-disk size
    /// instead of the ~8x larger eager-f32-dequant footprint.
    Mxfp4 {
        packed: WeightBytes,
        scale: WeightBytes,
        rows: usize,
        cols: usize,
    },
}

impl WeightMatrix {
    pub fn rows(&self) -> usize {
        match self {
            WeightMatrix::F32(t) => t.rows(),
            WeightMatrix::Quantized { rows, .. } => *rows,
            WeightMatrix::Mxfp4 { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            WeightMatrix::F32(t) => t.cols(),
            WeightMatrix::Quantized { cols, .. } => *cols,
            WeightMatrix::Mxfp4 { cols, .. } => *cols,
        }
    }

    fn block_bytes_per_row(&self, kind: QuantKind, cols: usize) -> usize {
        match kind {
            QuantKind::Q8_0 => {
                (cols / ferrox_quant::Q8_0_BLOCK_ELEMS) * ferrox_quant::Q8_0_BLOCK_BYTES
            }
            QuantKind::Q4_0 => {
                (cols / ferrox_quant::Q4_0_BLOCK_ELEMS) * ferrox_quant::Q4_0_BLOCK_BYTES
            }
            QuantKind::Q4K => {
                (cols / ferrox_quant::Q4_K_BLOCK_ELEMS) * ferrox_quant::Q4_K_BLOCK_BYTES
            }
            QuantKind::Q5K => {
                (cols / ferrox_quant::Q5_K_BLOCK_ELEMS) * ferrox_quant::Q5_K_BLOCK_BYTES
            }
            QuantKind::Q6K => {
                (cols / ferrox_quant::Q6_K_BLOCK_ELEMS) * ferrox_quant::Q6_K_BLOCK_BYTES
            }
            QuantKind::Q2K => {
                (cols / ferrox_quant::Q2_K_BLOCK_ELEMS) * ferrox_quant::Q2_K_BLOCK_BYTES
            }
            QuantKind::Q3K => {
                (cols / ferrox_quant::Q3_K_BLOCK_ELEMS) * ferrox_quant::Q3_K_BLOCK_BYTES
            }
            QuantKind::Q4_1 => {
                (cols / ferrox_quant::Q4_1_BLOCK_ELEMS) * ferrox_quant::Q4_1_BLOCK_BYTES
            }
            QuantKind::Q5_0 => {
                (cols / ferrox_quant::Q5_0_BLOCK_ELEMS) * ferrox_quant::Q5_0_BLOCK_BYTES
            }
            QuantKind::Q5_1 => {
                (cols / ferrox_quant::Q5_1_BLOCK_ELEMS) * ferrox_quant::Q5_1_BLOCK_BYTES
            }
            QuantKind::Q8_1 => {
                (cols / ferrox_quant::Q8_1_BLOCK_ELEMS) * ferrox_quant::Q8_1_BLOCK_BYTES
            }
            QuantKind::IQ4NL => {
                (cols / ferrox_quant::IQ4_NL_BLOCK_ELEMS) * ferrox_quant::IQ4_NL_BLOCK_BYTES
            }
            QuantKind::IQ4XS => {
                (cols / ferrox_quant::IQ4_XS_BLOCK_ELEMS) * ferrox_quant::IQ4_XS_BLOCK_BYTES
            }
            QuantKind::IQ1S => {
                (cols / ferrox_quant::IQ1_S_BLOCK_ELEMS) * ferrox_quant::IQ1_S_BLOCK_BYTES
            }
            QuantKind::IQ2XXS => {
                (cols / ferrox_quant::IQ2_XXS_BLOCK_ELEMS) * ferrox_quant::IQ2_XXS_BLOCK_BYTES
            }
            QuantKind::IQ3XXS => {
                (cols / ferrox_quant::IQ3_XXS_BLOCK_ELEMS) * ferrox_quant::IQ3_XXS_BLOCK_BYTES
            }
            QuantKind::Mxfp4Gguf => {
                (cols / ferrox_quant::MXFP4_GGUF_BLOCK_ELEMS) * ferrox_quant::MXFP4_GGUF_BLOCK_BYTES
            }
        }
    }

    /// A reasonable minimum number of rows for one rayon task to
    /// process, to avoid rayon's work-stealing splitter fragmenting a
    /// matmul into tasks so small that scheduling/synchronization
    /// overhead dominates the real per-row work (a fused dequant+dot,
    /// not free). This is a real, measured fix, not speculative
    /// tuning: naive per-row splitting (rayon's default) caused a
    /// 13-16x throughput regression on a host configured with far more
    /// rayon threads than a small model's matrices have useful
    /// parallelism for (observed directly on a shared-core rented
    /// host, where auto-detected high thread counts collapsed
    /// throughput ~13-16x on a small model). Aims for ~4 tasks per thread
    /// -- enough that rayon's work-stealing can still load-balance
    /// across threads that finish early, without going all the way
    /// down to one task per row.
    ///
    /// Floor of 8 avoids Rayon thrash on tiny mats (SmolLM2 attn_kv
    /// has 192 rows → without a floor, ~48 one-row tasks on 10 cores).
    fn min_rows_per_task(rows: usize) -> usize {
        let threads = rayon::current_num_threads().max(1);
        (rows / (threads * 4)).max(8.min(rows.max(1)))
    }

    /// Prefer serial when the mat is too small for fork-join to pay off.
    fn prefer_serial_matvec(rows: usize, cols: usize) -> bool {
        // ~256k f32-equivalent ops: below this, Rayon overhead dominates
        // on Host B-class cores for Q8/Q4 decode GEMVs.
        rows.saturating_mul(cols) < 256_000
    }

    fn dot(kind: QuantKind, row: &[u8], x: &[f32]) -> f32 {
        match kind {
            QuantKind::Q8_0 => ferrox_quant::dot_q8_0_f32(row, x),
            QuantKind::Q4_0 => ferrox_quant::dot_q4_0_f32(row, x),
            QuantKind::Q4K => ferrox_quant::dot_q4_k_f32(row, x),
            QuantKind::Q5K => ferrox_quant::dot_q5_k_f32(row, x),
            QuantKind::Q6K => ferrox_quant::dot_q6_k_f32(row, x),
            QuantKind::Q2K => ferrox_quant::dot_q2_k_f32(row, x),
            QuantKind::Q3K => ferrox_quant::dot_q3_k_f32(row, x),
            QuantKind::Q4_1 => ferrox_quant::dot_q4_1_f32(row, x),
            QuantKind::Q5_0 => ferrox_quant::dot_q5_0_f32(row, x),
            QuantKind::Q5_1 => ferrox_quant::dot_q5_1_f32(row, x),
            QuantKind::Q8_1 => ferrox_quant::dot_q8_1_f32(row, x),
            QuantKind::IQ4NL => ferrox_quant::dot_iq4_nl_f32(row, x),
            QuantKind::IQ4XS => ferrox_quant::dot_iq4_xs_f32(row, x),
            QuantKind::IQ1S => ferrox_quant::dot_iq1_s_f32(row, x),
            QuantKind::IQ2XXS => ferrox_quant::dot_iq2_xxs_f32(row, x),
            QuantKind::IQ3XXS => ferrox_quant::dot_iq3_xxs_f32(row, x),
            QuantKind::Mxfp4Gguf => ferrox_quant::dot_mxfp4_gguf_f32(row, x),
        }
    }

    /// Per-kind full-buffer dequantization -- the row-lookup counterpart
    /// of `dot`'s fused per-kind dispatch below.
    fn dequant(kind: QuantKind, bytes: &[u8]) -> Vec<f32> {
        let out = match kind {
            QuantKind::Q8_0 => ferrox_quant::dequant_q8_0(bytes),
            QuantKind::Q4_0 => ferrox_quant::dequant_q4_0(bytes),
            QuantKind::Q4K => ferrox_quant::dequant_q4_k(bytes),
            QuantKind::Q5K => ferrox_quant::dequant_q5_k(bytes),
            QuantKind::Q6K => ferrox_quant::dequant_q6_k(bytes),
            QuantKind::Q2K => ferrox_quant::dequant_q2_k(bytes),
            QuantKind::Q3K => ferrox_quant::dequant_q3_k(bytes),
            QuantKind::Q4_1 => ferrox_quant::dequant_q4_1(bytes),
            QuantKind::Q5_0 => ferrox_quant::dequant_q5_0(bytes),
            QuantKind::Q5_1 => ferrox_quant::dequant_q5_1(bytes),
            QuantKind::Q8_1 => ferrox_quant::dequant_q8_1(bytes),
            QuantKind::IQ4NL => ferrox_quant::dequant_iq4_nl(bytes),
            QuantKind::IQ4XS => ferrox_quant::dequant_iq4_xs(bytes),
            QuantKind::IQ1S => ferrox_quant::dequant_iq1_s(bytes),
            QuantKind::IQ2XXS => ferrox_quant::dequant_iq2_xxs(bytes),
            QuantKind::IQ3XXS => ferrox_quant::dequant_iq3_xxs(bytes),
            QuantKind::Mxfp4Gguf => ferrox_quant::dequant_mxfp4_gguf(bytes),
        };
        out.expect("row byte length is block-aligned by construction (block_bytes_per_row)")
    }

    /// Dequantizes exactly one row to f32, without touching any other
    /// row's bytes. This is what makes a *quantized* embedding table
    /// usable directly: token lookup reads `row_bytes` bytes and
    /// dequantizes `cols` values, instead of the whole vocabulary
    /// tensor ever being widened to f32 (which for a large-vocab model
    /// is a multi-GB allocation that exists only to be indexed one row
    /// at a time).
    pub fn dequant_row(&self, r: usize) -> Vec<f32> {
        assert!(r < self.rows(), "row {r} out of range ({})", self.rows());
        match self {
            WeightMatrix::F32(t) => t.row(r).to_vec(),
            WeightMatrix::Quantized {
                data, cols, kind, ..
            } => {
                let row_bytes = self.block_bytes_per_row(*kind, *cols);
                let bytes = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                let out = Self::dequant(*kind, bytes);
                debug_assert_eq!(out.len(), *cols);
                out
            }
            WeightMatrix::Mxfp4 {
                packed,
                scale,
                cols,
                ..
            } => {
                let packed_per_row = cols / 2;
                let scales_per_row = cols / ferrox_quant::MXFP4_GROUP_SIZE;
                let p = &packed.as_slice()[r * packed_per_row..(r + 1) * packed_per_row];
                let sc = &scale.as_slice()[r * scales_per_row..(r + 1) * scales_per_row];
                ferrox_quant::dequant_mxfp4_row(p, sc)
                    .expect("row slices are group-aligned by construction")
            }
        }
    }

    /// Computes `W @ x` for a single activation vector `x` of length
    /// `self.cols()`, returning a vector of length `self.rows()`.
    /// Parallelized over output rows with rayon, same decomposition as
    /// `matmul_f32`.
    ///
    /// With `--features metal` / `--features cuda`, when the matching
    /// dense GPU env selects a device (see [`metal_dense_enabled`] /
    /// [`cuda_dense_enabled`]), quantized kinds that have a GPU kernel
    /// go through [`Self::apply_gpu`] first so dense Llama-class
    /// decode uses the GPU instead of only MoE expert placement.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.cols(),
            "activation length must match matrix column count"
        );
        #[cfg(feature = "cuda")]
        {
            if cuda_dense_enabled() {
                if let Some(out) = self.apply_gpu(x) {
                    return out;
                }
            }
        }
        #[cfg(feature = "metal")]
        {
            if metal_dense_enabled() {
                if let Some(out) = self.apply_gpu(x) {
                    return out;
                }
            }
        }
        self.apply_cpu(x)
    }

    /// CPU-only matvec (NEON/AVX/scalar via `ferrox-quant`). Used by
    /// [`Self::apply`] after Metal miss/disable, and by GPU parity tests
    /// that must not recurse into [`Self::apply_gpu`].
    pub fn apply_cpu(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.cols(),
            "activation length must match matrix column count"
        );
        match self {
            WeightMatrix::F32(t) => {
                let xt = Tensor::new(x.to_vec(), vec![1, x.len()]);
                crate::matmul::matmul_f32(&xt, t).data
            }
            WeightMatrix::Quantized {
                data,
                rows,
                cols,
                kind,
            } => {
                let row_bytes = self.block_bytes_per_row(*kind, *cols);
                let mut out = vec![0f32; *rows];
                // FERROX_CPU_INT_DOT=1: quantize the shared activation once,
                // then every row dot is int8×int8 → i32 (llama.cpp CPU matmul).
                // Q8_0/Q4_0 use 32-elem Q8_0 acts; Q4_K/Q5_K/Q6_K use Q8_K.
                if cpu_int_dot_enabled() {
                    match *kind {
                        QuantKind::Q8_0 if x.len().is_multiple_of(32) => {
                            let act = ferrox_quant::quantize_activations_q8(x);
                            let n_groups = *rows / ferrox_quant::Q8_0X4_NROWS;
                            let serial = Self::prefer_serial_matvec(*rows, *cols);
                            if n_groups > 0 {
                                let packed = get_or_repack_q8x4(data.as_slice(), *rows, *cols);
                                if serial {
                                    for (g, chunk) in out[..n_groups * ferrox_quant::Q8_0X4_NROWS]
                                        .chunks_mut(ferrox_quant::Q8_0X4_NROWS)
                                        .enumerate()
                                    {
                                        ferrox_quant::gemv_q8_0x4_group(
                                            &packed, g, &act, *cols, chunk,
                                        );
                                    }
                                } else {
                                    out[..n_groups * ferrox_quant::Q8_0X4_NROWS]
                                        .par_chunks_mut(ferrox_quant::Q8_0X4_NROWS)
                                        .with_min_len(Self::min_rows_per_task(n_groups).max(1))
                                        .enumerate()
                                        .for_each(|(g, chunk)| {
                                            ferrox_quant::gemv_q8_0x4_group(
                                                &packed, g, &act, *cols, chunk,
                                            );
                                        });
                                }
                                let data_slice = data.as_slice();
                                let tail_len = *rows - n_groups * ferrox_quant::Q8_0X4_NROWS;
                                if tail_len > 0 {
                                    let tail = &mut out[n_groups * ferrox_quant::Q8_0X4_NROWS..];
                                    if serial || Self::prefer_serial_matvec(tail_len, *cols) {
                                        for (i, o) in tail.iter_mut().enumerate() {
                                            let r = n_groups * ferrox_quant::Q8_0X4_NROWS + i;
                                            let row =
                                                &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                            *o = ferrox_quant::dot_q8_0_q8(row, &act);
                                        }
                                    } else {
                                        let min_len = Self::min_rows_per_task(tail_len);
                                        tail.par_iter_mut()
                                            .with_min_len(min_len)
                                            .enumerate()
                                            .for_each(|(i, o)| {
                                                let r = n_groups * ferrox_quant::Q8_0X4_NROWS + i;
                                                let row =
                                                    &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                                *o = ferrox_quant::dot_q8_0_q8(row, &act);
                                            });
                                    }
                                }
                                return out;
                            }
                            if serial {
                                for (r, o) in out.iter_mut().enumerate() {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    *o = ferrox_quant::dot_q8_0_q8(row, &act);
                                }
                            } else {
                                out.par_iter_mut()
                                    .with_min_len(Self::min_rows_per_task(*rows))
                                    .enumerate()
                                    .for_each(|(r, o)| {
                                        let row =
                                            &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                        *o = ferrox_quant::dot_q8_0_q8(row, &act);
                                    });
                            }
                            return out;
                        }
                        QuantKind::Q4_0 if x.len().is_multiple_of(32) => {
                            let act = ferrox_quant::quantize_activations_q8(x);
                            out.par_iter_mut()
                                .with_min_len(Self::min_rows_per_task(*rows))
                                .enumerate()
                                .for_each(|(r, o)| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    *o = ferrox_quant::dot_q4_0_q8(row, &act);
                                });
                            return out;
                        }
                        QuantKind::Q4K if x.len().is_multiple_of(256) => {
                            let act = ferrox_quant::quantize_activations_q8_k(x);
                            let n_groups = *rows / ferrox_quant::Q4_KX8_NROWS;
                            if n_groups > 0 {
                                let interleave = ferrox_quant::q4_kx8_interleave();
                                let packed = get_or_repack_q4k(data.as_slice(), *rows, *cols);
                                out[..n_groups * ferrox_quant::Q4_KX8_NROWS]
                                    .par_chunks_mut(ferrox_quant::Q4_KX8_NROWS)
                                    .with_min_len(Self::min_rows_per_task(n_groups).max(1))
                                    .enumerate()
                                    .for_each(|(g, chunk)| {
                                        ferrox_quant::gemv_q4_kx8_group(
                                            &packed, g, &act, *cols, interleave, chunk,
                                        );
                                    });
                                let data_slice = data.as_slice();
                                out[n_groups * ferrox_quant::Q4_KX8_NROWS..]
                                    .par_iter_mut()
                                    .with_min_len(Self::min_rows_per_task(
                                        *rows - n_groups * ferrox_quant::Q4_KX8_NROWS,
                                    ))
                                    .enumerate()
                                    .for_each(|(i, o)| {
                                        let r = n_groups * ferrox_quant::Q4_KX8_NROWS + i;
                                        let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                        *o = ferrox_quant::dot_q4_k_q8(row, &act);
                                    });
                                return out;
                            }
                            out.par_iter_mut()
                                .with_min_len(Self::min_rows_per_task(*rows))
                                .enumerate()
                                .for_each(|(r, o)| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    *o = ferrox_quant::dot_q4_k_q8(row, &act);
                                });
                            return out;
                        }
                        QuantKind::Q5K | QuantKind::Q6K if x.len().is_multiple_of(256) => {
                            let act = ferrox_quant::quantize_activations_q8_k(x);
                            let kind = *kind;
                            out.par_iter_mut()
                                .with_min_len(Self::min_rows_per_task(*rows))
                                .enumerate()
                                .for_each(|(r, o)| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    *o = match kind {
                                        QuantKind::Q5K => ferrox_quant::dot_q5_k_q8(row, &act),
                                        QuantKind::Q6K => ferrox_quant::dot_q6_k_q8(row, &act),
                                        _ => unreachable!(),
                                    };
                                });
                            return out;
                        }
                        _ => {}
                    }
                }
                out.par_iter_mut()
                    .with_min_len(Self::min_rows_per_task(*rows))
                    .enumerate()
                    .for_each(|(r, o)| {
                        let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                        *o = Self::dot(*kind, row, x);
                    });
                out
            }
            WeightMatrix::Mxfp4 {
                packed,
                scale,
                rows,
                cols,
            } => {
                let packed_row_bytes = cols / 2;
                let scale_row_bytes = cols / ferrox_quant::MXFP4_GROUP_SIZE;
                let mut out = vec![0f32; *rows];
                out.par_iter_mut()
                    .with_min_len(Self::min_rows_per_task(*rows))
                    .enumerate()
                    .for_each(|(r, o)| {
                        let prow =
                            &packed.as_slice()[r * packed_row_bytes..(r + 1) * packed_row_bytes];
                        let srow =
                            &scale.as_slice()[r * scale_row_bytes..(r + 1) * scale_row_bytes];
                        *o = ferrox_quant::dot_mxfp4_row_f32(prow, srow, x);
                    });
                out
            }
        }
    }

    /// INT_DOT matvec against a pre-quantized Q8_0 activation (shared gate/up).
    pub fn apply_cpu_q8(&self, act: &ferrox_quant::Q8Activations) -> Option<Vec<f32>> {
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        if !matches!(*kind, QuantKind::Q8_0 | QuantKind::Q4_0) || !cpu_int_dot_enabled() {
            return None;
        }
        if act.q.len() != *cols || !cols.is_multiple_of(32) {
            return None;
        }
        let row_bytes = self.block_bytes_per_row(*kind, *cols);
        let mut out = vec![0f32; *rows];
        let kind = *kind;
        let data = data.as_slice();
        // Q8_0×4 interleaved GEMV — same path as `apply_cpu` so dense
        // FFN gate+up (via `run_expert`) hit the fast kernel, not per-row
        // `dot_q8_0_q8` (SmolLM2/Qwen decode gap without this).
        if matches!(kind, QuantKind::Q8_0) {
            let n_groups = *rows / ferrox_quant::Q8_0X4_NROWS;
            if n_groups > 0 {
                let packed = get_or_repack_q8x4(data, *rows, *cols);
                let serial = Self::prefer_serial_matvec(*rows, *cols);
                let body = |g: usize, chunk: &mut [f32]| {
                    ferrox_quant::gemv_q8_0x4_group(&packed, g, act, *cols, chunk);
                };
                if serial {
                    for (g, chunk) in out[..n_groups * ferrox_quant::Q8_0X4_NROWS]
                        .chunks_mut(ferrox_quant::Q8_0X4_NROWS)
                        .enumerate()
                    {
                        body(g, chunk);
                    }
                } else {
                    out[..n_groups * ferrox_quant::Q8_0X4_NROWS]
                        .par_chunks_mut(ferrox_quant::Q8_0X4_NROWS)
                        .with_min_len(Self::min_rows_per_task(n_groups).max(1))
                        .enumerate()
                        .for_each(|(g, chunk)| body(g, chunk));
                }
                let tail_len = *rows - n_groups * ferrox_quant::Q8_0X4_NROWS;
                if tail_len > 0 {
                    let tail = &mut out[n_groups * ferrox_quant::Q8_0X4_NROWS..];
                    if serial || Self::prefer_serial_matvec(tail_len, *cols) {
                        for (i, o) in tail.iter_mut().enumerate() {
                            let r = n_groups * ferrox_quant::Q8_0X4_NROWS + i;
                            *o = ferrox_quant::dot_q8_0_q8(
                                &data[r * row_bytes..(r + 1) * row_bytes],
                                act,
                            );
                        }
                    } else {
                        let min_len = Self::min_rows_per_task(tail_len);
                        tail.par_iter_mut()
                            .with_min_len(min_len)
                            .enumerate()
                            .for_each(|(i, o)| {
                                let r = n_groups * ferrox_quant::Q8_0X4_NROWS + i;
                                *o = ferrox_quant::dot_q8_0_q8(
                                    &data[r * row_bytes..(r + 1) * row_bytes],
                                    act,
                                );
                            });
                    }
                }
                return Some(out);
            }
        }
        // Q4_0: pair rows for shared-act SDOT (llama-style). Odd tail solo.
        if matches!(kind, QuantKind::Q4_0) && *rows >= 2 {
            let n_pairs = *rows / 2;
            let serial = Self::prefer_serial_matvec(*rows, *cols);
            if serial {
                for p in 0..n_pairs {
                    let r = p * 2;
                    let (a, b) = ferrox_quant::dot_q4_0_q8_2row(
                        &data[r * row_bytes..(r + 1) * row_bytes],
                        &data[(r + 1) * row_bytes..(r + 2) * row_bytes],
                        act,
                    );
                    out[r] = a;
                    out[r + 1] = b;
                }
                if *rows % 2 == 1 {
                    let r = *rows - 1;
                    out[r] =
                        ferrox_quant::dot_q4_0_q8(&data[r * row_bytes..(r + 1) * row_bytes], act);
                }
                return Some(out);
            }
            out.par_chunks_mut(2)
                .with_min_len(Self::min_rows_per_task(n_pairs).max(1))
                .enumerate()
                .for_each(|(p, chunk)| {
                    if chunk.len() < 2 {
                        let r = p * 2;
                        chunk[0] = ferrox_quant::dot_q4_0_q8(
                            &data[r * row_bytes..(r + 1) * row_bytes],
                            act,
                        );
                        return;
                    }
                    let r = p * 2;
                    let (a, b) = ferrox_quant::dot_q4_0_q8_2row(
                        &data[r * row_bytes..(r + 1) * row_bytes],
                        &data[(r + 1) * row_bytes..(r + 2) * row_bytes],
                        act,
                    );
                    chunk[0] = a;
                    chunk[1] = b;
                });
            return Some(out);
        }
        if Self::prefer_serial_matvec(*rows, *cols) {
            for (r, o) in out.iter_mut().enumerate() {
                let row = &data[r * row_bytes..(r + 1) * row_bytes];
                *o = match kind {
                    QuantKind::Q8_0 => ferrox_quant::dot_q8_0_q8(row, act),
                    QuantKind::Q4_0 => ferrox_quant::dot_q4_0_q8(row, act),
                    _ => unreachable!(),
                };
            }
            return Some(out);
        }
        out.par_iter_mut()
            .with_min_len(Self::min_rows_per_task(*rows))
            .enumerate()
            .for_each(|(r, o)| {
                let row = &data[r * row_bytes..(r + 1) * row_bytes];
                *o = match kind {
                    QuantKind::Q8_0 => ferrox_quant::dot_q8_0_q8(row, act),
                    QuantKind::Q4_0 => ferrox_quant::dot_q4_0_q8(row, act),
                    _ => unreachable!(),
                };
            });
        Some(out)
    }

    /// Two contiguous rows × one Q8 act (shared act loads). Q4_0 uses
    /// [`ferrox_quant::dot_q4_0_q8_2row`]; Q8_0 falls back to two singles.
    pub fn dot_pair_cpu_q8(
        &self,
        row: usize,
        act: &ferrox_quant::Q8Activations,
    ) -> Option<(f32, f32)> {
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        if !matches!(*kind, QuantKind::Q8_0 | QuantKind::Q4_0) || !cpu_int_dot_enabled() {
            return None;
        }
        if act.q.len() != *cols || !cols.is_multiple_of(32) || row + 1 >= *rows {
            return None;
        }
        let row_bytes = self.block_bytes_per_row(*kind, *cols);
        let bytes = data.as_slice();
        let r0 = &bytes[row * row_bytes..(row + 1) * row_bytes];
        let r1 = &bytes[(row + 1) * row_bytes..(row + 2) * row_bytes];
        Some(match *kind {
            QuantKind::Q4_0 => ferrox_quant::dot_q4_0_q8_2row(r0, r1, act),
            QuantKind::Q8_0 => (
                ferrox_quant::dot_q8_0_q8(r0, act),
                ferrox_quant::dot_q8_0_q8(r1, act),
            ),
            _ => unreachable!(),
        })
    }

    /// Single-row INT_DOT against pre-quantized Q8_0 acts (llama `mul_mat_id`
    /// inner loop). Returns `None` if this matrix is not Q4_0/Q8_0 INT_DOT.
    pub fn dot_row_cpu_q8(&self, row: usize, act: &ferrox_quant::Q8Activations) -> Option<f32> {
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        if row >= *rows
            || !matches!(*kind, QuantKind::Q8_0 | QuantKind::Q4_0)
            || !cpu_int_dot_enabled()
            || act.q.len() != *cols
            || !cols.is_multiple_of(32)
        {
            return None;
        }
        let row_bytes = self.block_bytes_per_row(*kind, *cols);
        let bytes = &data.as_slice()[row * row_bytes..(row + 1) * row_bytes];
        Some(match *kind {
            QuantKind::Q8_0 => ferrox_quant::dot_q8_0_q8(bytes, act),
            QuantKind::Q4_0 => ferrox_quant::dot_q4_0_q8(bytes, act),
            _ => unreachable!(),
        })
    }

    /// Computes `W @ X` for a *batch* of activation vectors at once:
    /// `x_batch` is `batch_size` rows of `self.cols()` elements each,
    /// flattened row-major; returns `batch_size` rows of
    /// `self.rows()` elements each, flattened row-major (`[batch,
    /// rows]`, matching the layout `Tensor`/`Decoder` expect for
    /// chaining into further matmuls).
    ///
    /// This is not just a convenience wrapper: for a quantized matrix,
    /// each weight row's bytes are read from memory *once* and dotted
    /// against every activation in the batch, instead of once per
    /// `apply` call. For a memory-bandwidth-bound quantized matmul --
    /// which fused Q8_0/Q4_0 dot products are, since the whole point of
    /// keeping weights quantized is that reading them is the
    /// bottleneck, not the arithmetic -- processing `batch_size`
    /// positions this way costs roughly the same *memory traffic* as
    /// processing one position, not `batch_size` times as much. This
    /// is the same reason speculative-decoding verification and batched
    /// prefill are faster per-token than sequential single-token decode
    /// on real hardware: it turns `batch_size` separate reads of the
    /// same weights into one.
    ///
    /// With Metal dense enabled, dispatches a single batched Metal
    /// command buffer — Q4_K/Q6_K use
    /// [`ferrox_metal::gpu::launch_q4_k_matmul_batch`] /
    /// [`ferrox_metal::gpu::launch_q6_k_matmul_batch`] when
    /// `batch_size >= 2`; other kinds use
    /// [`ferrox_metal::gpu::launch_matvec_batch`]. Falls back to
    /// per-row [`Self::apply`] if the batch launch fails.
    pub fn apply_batch(&self, x_batch: &[f32], batch_size: usize) -> Vec<f32> {
        let cols = self.cols();
        assert_eq!(
            x_batch.len(),
            batch_size * cols,
            "x_batch length must be batch_size * cols"
        );
        if batch_size == 0 {
            return Vec::new();
        }

        #[cfg(feature = "metal")]
        {
            if metal_dense_enabled()
                && matches!(
                    self,
                    WeightMatrix::Quantized { kind, .. } if Self::metal_kind_supported(*kind)
                )
            {
                if let Some(out) = self.apply_gpu_batch(x_batch, batch_size) {
                    return out;
                }
                let rows = self.rows();
                let mut out = vec![0f32; batch_size * rows];
                for b in 0..batch_size {
                    let y = self.apply(&x_batch[b * cols..(b + 1) * cols]);
                    out[b * rows..(b + 1) * rows].copy_from_slice(&y);
                }
                return out;
            }
        }

        match self {
            WeightMatrix::F32(t) => {
                let xt = Tensor::new(x_batch.to_vec(), vec![batch_size, cols]);
                crate::matmul::matmul_f32(&xt, t).data
            }
            WeightMatrix::Quantized {
                data,
                rows,
                cols: _,
                kind,
            } => {
                let row_bytes = self.block_bytes_per_row(*kind, cols);
                // Laid out [rows, batch] during computation (each row
                // is one parallel unit of work, written to a
                // contiguous batch_size-sized chunk -- safe for
                // par_chunks_mut, unlike the [batch, rows] layout the
                // function returns), then transposed once at the end.
                let mut by_row = vec![0f32; rows * batch_size];

                // Prefill INT_DOT: quantize each activation once, then
                // reuse Q8 packs across all weight rows (llama CPU path).
                if cpu_int_dot_enabled() {
                    match *kind {
                        QuantKind::Q8_0 if cols.is_multiple_of(32) => {
                            let acts: Vec<_> = (0..batch_size)
                                .map(|b| {
                                    ferrox_quant::quantize_activations_q8(
                                        &x_batch[b * cols..(b + 1) * cols],
                                    )
                                })
                                .collect();
                            let n_groups = *rows / ferrox_quant::Q8_0X4_NROWS;
                            if n_groups > 0 {
                                let packed = get_or_repack_q8x4(data.as_slice(), *rows, cols);
                                let group_stride = ferrox_quant::Q8_0X4_NROWS * batch_size;
                                by_row[..n_groups * group_stride]
                                    .par_chunks_mut(group_stride)
                                    .with_min_len(Self::min_rows_per_task(n_groups).max(1))
                                    .enumerate()
                                    .for_each(|(g, group_out)| {
                                        // GEMM, not a GEMV per position:
                                        // `group_out` is already
                                        // `[row][batch]`, which is the
                                        // layout the batched kernel
                                        // writes, and the group's weight
                                        // vectors stay in registers
                                        // across a tile of activations.
                                        ferrox_quant::gemm_q8_0x4_group(
                                            &packed, g, &acts, cols, group_out,
                                        );
                                    });
                                let data_slice = data.as_slice();
                                by_row[n_groups * group_stride..]
                                    .par_chunks_mut(batch_size)
                                    .with_min_len(Self::min_rows_per_task(
                                        *rows - n_groups * ferrox_quant::Q8_0X4_NROWS,
                                    ))
                                    .enumerate()
                                    .for_each(|(i, out_row)| {
                                        let r = n_groups * ferrox_quant::Q8_0X4_NROWS + i;
                                        let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                        for (b, act) in acts.iter().enumerate() {
                                            out_row[b] = ferrox_quant::dot_q8_0_q8(row, act);
                                        }
                                    });
                            } else {
                                by_row
                                    .par_chunks_mut(batch_size)
                                    .with_min_len(Self::min_rows_per_task(*rows))
                                    .enumerate()
                                    .for_each(|(r, out_row)| {
                                        let row =
                                            &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                        for (b, act) in acts.iter().enumerate() {
                                            out_row[b] = ferrox_quant::dot_q8_0_q8(row, act);
                                        }
                                    });
                            }
                            let mut out = vec![0f32; batch_size * rows];
                            for r in 0..*rows {
                                for b in 0..batch_size {
                                    out[b * rows + r] = by_row[r * batch_size + b];
                                }
                            }
                            return out;
                        }
                        QuantKind::Q4_0 if cols.is_multiple_of(32) => {
                            let acts: Vec<_> = (0..batch_size)
                                .map(|b| {
                                    ferrox_quant::quantize_activations_q8(
                                        &x_batch[b * cols..(b + 1) * cols],
                                    )
                                })
                                .collect();
                            by_row
                                .par_chunks_mut(batch_size)
                                .with_min_len(Self::min_rows_per_task(*rows))
                                .enumerate()
                                .for_each(|(r, out_row)| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    for (b, act) in acts.iter().enumerate() {
                                        out_row[b] = ferrox_quant::dot_q4_0_q8(row, act);
                                    }
                                });
                            let mut out = vec![0f32; batch_size * rows];
                            for r in 0..*rows {
                                for b in 0..batch_size {
                                    out[b * rows + r] = by_row[r * batch_size + b];
                                }
                            }
                            return out;
                        }
                        QuantKind::Q4K if cols.is_multiple_of(256) => {
                            let acts: Vec<_> = (0..batch_size)
                                .map(|b| {
                                    ferrox_quant::quantize_activations_q8_k(
                                        &x_batch[b * cols..(b + 1) * cols],
                                    )
                                })
                                .collect();
                            let n_groups = *rows / ferrox_quant::Q4_KX8_NROWS;
                            if n_groups > 0 {
                                let interleave = ferrox_quant::q4_kx8_interleave();
                                let packed = get_or_repack_q4k(data.as_slice(), *rows, cols);
                                let group_stride = ferrox_quant::Q4_KX8_NROWS * batch_size;
                                by_row[..n_groups * group_stride]
                                    .par_chunks_mut(group_stride)
                                    .with_min_len(Self::min_rows_per_task(n_groups).max(1))
                                    .enumerate()
                                    .for_each(|(g, group_out)| {
                                        let mut tmp = [0f32; ferrox_quant::Q4_KX8_NROWS];
                                        for (b, act) in acts.iter().enumerate() {
                                            ferrox_quant::gemv_q4_kx8_group(
                                                &packed, g, act, cols, interleave, &mut tmp,
                                            );
                                            for r in 0..ferrox_quant::Q4_KX8_NROWS {
                                                group_out[r * batch_size + b] = tmp[r];
                                            }
                                        }
                                    });
                                let data_slice = data.as_slice();
                                by_row[n_groups * group_stride..]
                                    .par_chunks_mut(batch_size)
                                    .with_min_len(Self::min_rows_per_task(
                                        *rows - n_groups * ferrox_quant::Q4_KX8_NROWS,
                                    ))
                                    .enumerate()
                                    .for_each(|(i, out_row)| {
                                        let r = n_groups * ferrox_quant::Q4_KX8_NROWS + i;
                                        let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                        for (b, act) in acts.iter().enumerate() {
                                            out_row[b] = ferrox_quant::dot_q4_k_q8(row, act);
                                        }
                                    });
                            } else {
                                by_row
                                    .par_chunks_mut(batch_size)
                                    .with_min_len(Self::min_rows_per_task(*rows))
                                    .enumerate()
                                    .for_each(|(r, out_row)| {
                                        let row =
                                            &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                        for (b, act) in acts.iter().enumerate() {
                                            out_row[b] = ferrox_quant::dot_q4_k_q8(row, act);
                                        }
                                    });
                            }
                            let mut out = vec![0f32; batch_size * rows];
                            for r in 0..*rows {
                                for b in 0..batch_size {
                                    out[b * rows + r] = by_row[r * batch_size + b];
                                }
                            }
                            return out;
                        }
                        QuantKind::Q5K | QuantKind::Q6K if cols.is_multiple_of(256) => {
                            let acts: Vec<_> = (0..batch_size)
                                .map(|b| {
                                    ferrox_quant::quantize_activations_q8_k(
                                        &x_batch[b * cols..(b + 1) * cols],
                                    )
                                })
                                .collect();
                            let kind = *kind;
                            by_row
                                .par_chunks_mut(batch_size)
                                .with_min_len(Self::min_rows_per_task(*rows))
                                .enumerate()
                                .for_each(|(r, out_row)| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    for (b, act) in acts.iter().enumerate() {
                                        out_row[b] = match kind {
                                            QuantKind::Q5K => ferrox_quant::dot_q5_k_q8(row, act),
                                            QuantKind::Q6K => ferrox_quant::dot_q6_k_q8(row, act),
                                            _ => unreachable!(),
                                        };
                                    }
                                });
                            let mut out = vec![0f32; batch_size * rows];
                            for r in 0..*rows {
                                for b in 0..batch_size {
                                    out[b * rows + r] = by_row[r * batch_size + b];
                                }
                            }
                            return out;
                        }
                        _ => {}
                    }
                }

                by_row
                    .par_chunks_mut(batch_size)
                    .with_min_len(Self::min_rows_per_task(*rows))
                    .enumerate()
                    .for_each(|(r, out_row)| {
                        let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                        for b in 0..batch_size {
                            let x = &x_batch[b * cols..(b + 1) * cols];
                            out_row[b] = Self::dot(*kind, row, x);
                        }
                    });

                let mut out = vec![0f32; batch_size * rows];
                for r in 0..*rows {
                    for b in 0..batch_size {
                        out[b * rows + r] = by_row[r * batch_size + b];
                    }
                }
                out
            }
            WeightMatrix::Mxfp4 {
                packed,
                scale,
                rows,
                cols: _,
            } => {
                let packed_row_bytes = cols / 2;
                let scale_row_bytes = cols / ferrox_quant::MXFP4_GROUP_SIZE;
                let mut by_row = vec![0f32; rows * batch_size];
                by_row
                    .par_chunks_mut(batch_size)
                    .with_min_len(Self::min_rows_per_task(*rows))
                    .enumerate()
                    .for_each(|(r, out_row)| {
                        let prow =
                            &packed.as_slice()[r * packed_row_bytes..(r + 1) * packed_row_bytes];
                        let srow =
                            &scale.as_slice()[r * scale_row_bytes..(r + 1) * scale_row_bytes];
                        for b in 0..batch_size {
                            let x = &x_batch[b * cols..(b + 1) * cols];
                            out_row[b] = ferrox_quant::dot_mxfp4_row_f32(prow, srow, x);
                        }
                    });

                let mut out = vec![0f32; batch_size * rows];
                for r in 0..*rows {
                    for b in 0..batch_size {
                        out[b * rows + r] = by_row[r * batch_size + b];
                    }
                }
                out
            }
        }
    }

    /// Bytes actually resident in memory for this matrix -- the number
    /// that matters for "can this model's weights fit in RAM/VRAM at
    /// all," as opposed to the always-4x-larger f32-expanded size.
    pub fn resident_bytes(&self) -> usize {
        match self {
            WeightMatrix::F32(t) => t.len() * 4,
            WeightMatrix::Quantized { data, .. } => data.len(),
            WeightMatrix::Mxfp4 { packed, scale, .. } => packed.len() + scale.len(),
        }
    }

    /// Dispatches a single matvec through a real GPU kernel when a GPU
    /// feature is compiled in (`cuda` and/or `metal`) and this matrix
    /// is one of the five GPU-accelerated quant kinds (Q8_0, Q4_0,
    /// Q4_K, Q5_K, Q6_K). Returns `None` for every other case (no GPU
    /// feature, `F32`/`Mxfp4`/`Mxfp4Gguf`, or a `Quantized` kind other
    /// than the five below), so the caller falls back to `apply()` on
    /// the CPU -- this is a real dispatch decision
    /// (`ferrox_moe::run_expert_placed` uses it exactly this way), not
    /// a stub. Metal weight buffers are process-resident after the first
    /// upload (`ferrox_metal::gpu` weight cache); activations still
    /// upload per call. When both `cuda` and `metal` are enabled, CUDA
    /// is tried first and Metal is the fallback.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn apply_gpu(&self, x: &[f32]) -> Option<Vec<f32>> {
        assert_eq!(
            x.len(),
            self.cols(),
            "activation length must match matrix column count"
        );

        // F32 stays on CPU in apply_gpu: a lone small router matvec is
        // faster as host GEMV than a Metal sync. F32 Metal launches are
        // used when fused into MoE resident decode (encode_matvec).
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        let row_bytes = self.block_bytes_per_row(*kind, *cols);

        #[cfg(feature = "cuda")]
        {
            let launch: Option<CudaMatvecLaunchFn> = match kind {
                QuantKind::Q8_0 => Some(ferrox_cuda::gpu::launch_q8_0_matvec),
                QuantKind::Q4_0 => Some(ferrox_cuda::gpu::launch_q4_0_matvec),
                QuantKind::Q4K => Some(ferrox_cuda::gpu::launch_q4_k_matvec),
                QuantKind::Q5K => Some(ferrox_cuda::gpu::launch_q5_k_matvec),
                QuantKind::Q6K => Some(ferrox_cuda::gpu::launch_q6_k_matvec),
                _ => None,
            };
            if let Some(launch) = launch {
                let n_blocks_per_row = row_bytes / Self::block_bytes_for_kind(*kind);
                match launch(data.as_slice(), x, *rows, row_bytes, n_blocks_per_row) {
                    Ok(out) => return Some(out),
                    Err(e) => {
                        eprintln!(
                            "ferrox: CUDA matvec dispatch failed, trying next backend / CPU: {e}"
                        );
                    }
                }
            }
        }

        #[cfg(feature = "metal")]
        {
            let launch: Option<MetalMatvecLaunchFn> = match kind {
                QuantKind::Q8_0 => Some(ferrox_metal::gpu::launch_q8_0_matvec),
                QuantKind::Q4_0 => Some(ferrox_metal::gpu::launch_q4_0_matvec),
                QuantKind::Q4K => Some(ferrox_metal::gpu::launch_q4_k_matvec),
                QuantKind::Q5K => Some(ferrox_metal::gpu::launch_q5_k_matvec),
                QuantKind::Q6K => Some(ferrox_metal::gpu::launch_q6_k_matvec),
                QuantKind::IQ4XS => Some(ferrox_metal::gpu::launch_iq4_xs_matvec),
                _ => None,
            };
            if let Some(launch) = launch {
                match launch(data.as_slice(), x, *rows, row_bytes) {
                    Ok(out) => return Some(out),
                    Err(e) => {
                        eprintln!("ferrox: Metal matvec dispatch failed, falling back to CPU: {e}");
                    }
                }
            }
        }

        None
    }

    /// Runs several independent matvecs that share the same activation
    /// `x` in one GPU dispatch (one upload of `x`, one wait). Tries
    /// CUDA first (when `cuda_dense_enabled()`), then Metal (when
    /// `metal_dense_enabled()`). Intended for Q/K/V (and similar)
    /// projections. Returns `None` if no GPU backend is enabled, any
    /// matrix lacks a GPU kernel, or all fused launches fail — caller
    /// should fall back to sequential [`Self::apply`].
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn apply_gpu_multi(mats: &[&WeightMatrix], x: &[f32]) -> Option<Vec<Vec<f32>>> {
        if mats.is_empty() {
            return None;
        }
        assert_eq!(
            x.len(),
            mats[0].cols(),
            "activation length must match matrix column count"
        );

        // Try CUDA first if enabled.
        #[cfg(feature = "cuda")]
        if cuda_dense_enabled() {
            let mut launches = Vec::with_capacity(mats.len());
            for m in mats {
                assert_eq!(m.cols(), mats[0].cols());
                let WeightMatrix::Quantized {
                    data,
                    rows,
                    cols,
                    kind,
                } = m
                else {
                    return None;
                };
                let (kernel_src, module_name, fn_name) = match kind {
                    QuantKind::Q8_0 => (
                        ferrox_cuda::gpu::Q8_0_MATVEC_KERNEL_SRC,
                        "ferrox_q8_0",
                        "q8_0_matvec",
                    ),
                    QuantKind::Q4_0 => (
                        ferrox_cuda::gpu::Q4_0_MATVEC_KERNEL_SRC,
                        "ferrox_q4_0",
                        "q4_0_matvec",
                    ),
                    QuantKind::Q4K => (
                        ferrox_cuda::gpu::Q4_K_MATVEC_KERNEL_SRC,
                        "ferrox_q4_k",
                        "q4_k_matvec",
                    ),
                    QuantKind::Q5K => (
                        ferrox_cuda::gpu::Q5_K_MATVEC_KERNEL_SRC,
                        "ferrox_q5_k",
                        "q5_k_matvec",
                    ),
                    QuantKind::Q6K => (
                        ferrox_cuda::gpu::Q6_K_MATVEC_KERNEL_SRC,
                        "ferrox_q6_k",
                        "q6_k_matvec",
                    ),
                    _ => return None,
                };
                let row_bytes = m.block_bytes_per_row(*kind, *cols);
                let n_blocks_per_row = row_bytes / Self::block_bytes_for_kind(*kind);
                launches.push(ferrox_cuda::gpu::MatvecLaunch {
                    kernel_src,
                    module_name,
                    fn_name,
                    // Borrow mmap/owned storage — never to_vec() (breaks
                    // resident_cuda_weights pointer cache; re-uploads GB).
                    weights: data.as_slice(),
                    rows: *rows,
                    row_bytes,
                    n_blocks_per_row,
                });
            }
            match ferrox_cuda::gpu::launch_matvec_multi(x, &launches) {
                Ok(outs) => return Some(outs),
                Err(e) => {
                    eprintln!("ferrox: CUDA multi-matvec failed, trying next backend: {e}");
                }
            }
        }

        // Try Metal if CUDA didn't return or failed.
        #[cfg(feature = "metal")]
        if metal_dense_enabled() {
            let mut launches = Vec::with_capacity(mats.len());
            let mut held: Vec<(&[u8], usize, usize, &'static str)> = Vec::with_capacity(mats.len());
            for m in mats {
                assert_eq!(m.cols(), mats[0].cols());
                let WeightMatrix::Quantized {
                    data,
                    rows,
                    cols,
                    kind,
                } = m
                else {
                    return None;
                };
                let kind_name = match kind {
                    QuantKind::Q8_0 => "Q8_0",
                    QuantKind::Q4_0 => "Q4_0",
                    QuantKind::Q4K => "Q4_K",
                    QuantKind::Q5K => "Q5_K",
                    QuantKind::Q6K => "Q6_K",
                    QuantKind::IQ4XS => "IQ4_XS",
                    _ => return None,
                };
                let row_bytes = m.block_bytes_per_row(*kind, *cols);
                held.push((data.as_slice(), *rows, row_bytes, kind_name));
            }
            for (weights, rows, row_bytes, kind_name) in &held {
                let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
                    ferrox_metal::gpu::matvec_launch_meta(kind_name)?;
                launches.push(ferrox_metal::gpu::MatvecLaunch {
                    kernel_src: src,
                    fn_name,
                    block_bytes,
                    block_elems,
                    weights,
                    rows: *rows,
                    row_bytes: *row_bytes,
                    rows_per_tg,
                });
            }
            match ferrox_metal::gpu::launch_matvec_fused(x, &launches) {
                Ok(outs) => return Some(outs),
                Err(e) => {
                    eprintln!("ferrox: Metal fused matvec failed, falling back to CPU: {e}");
                }
            }
        }

        None
    }

    /// Dense SwiGLU FFN on GPU with device-resident activations:
    /// one upload of `x`, gate+up+silu×up+down on device, one download.
    /// Tries CUDA first when enabled, then Metal. Returns `None` if
    /// no GPU path applies — caller falls back to [`Self::apply`] /
    /// multi-matvec.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn apply_gpu_dense_ffn_swiglu(
        gate: &WeightMatrix,
        up: &WeightMatrix,
        down: &WeightMatrix,
        x: &[f32],
    ) -> Option<Vec<f32>> {
        #[cfg(feature = "cuda")]
        {
            if cuda_dense_enabled() {
                fn cuda_launch(m: &WeightMatrix) -> Option<ferrox_cuda::gpu::MatvecLaunch<'_>> {
                    let WeightMatrix::Quantized {
                        data,
                        rows,
                        cols,
                        kind,
                    } = m
                    else {
                        return None;
                    };
                    let (kernel_src, module_name, fn_name) = match kind {
                        QuantKind::Q8_0 => (
                            ferrox_cuda::gpu::Q8_0_MATVEC_KERNEL_SRC,
                            "ferrox_q8_0",
                            "q8_0_matvec",
                        ),
                        QuantKind::Q4_0 => (
                            ferrox_cuda::gpu::Q4_0_MATVEC_KERNEL_SRC,
                            "ferrox_q4_0",
                            "q4_0_matvec",
                        ),
                        QuantKind::Q4K => (
                            ferrox_cuda::gpu::Q4_K_MATVEC_KERNEL_SRC,
                            "ferrox_q4_k",
                            "q4_k_matvec",
                        ),
                        QuantKind::Q5K => (
                            ferrox_cuda::gpu::Q5_K_MATVEC_KERNEL_SRC,
                            "ferrox_q5_k",
                            "q5_k_matvec",
                        ),
                        QuantKind::Q6K => (
                            ferrox_cuda::gpu::Q6_K_MATVEC_KERNEL_SRC,
                            "ferrox_q6_k",
                            "q6_k_matvec",
                        ),
                        _ => return None,
                    };
                    let row_bytes = m.block_bytes_per_row(*kind, *cols);
                    let n_blocks_per_row = row_bytes / WeightMatrix::block_bytes_for_kind(*kind);
                    Some(ferrox_cuda::gpu::MatvecLaunch {
                        kernel_src,
                        module_name,
                        fn_name,
                        weights: data.as_slice(),
                        rows: *rows,
                        row_bytes,
                        n_blocks_per_row,
                    })
                }
                if let (Some(g), Some(u), Some(d)) =
                    (cuda_launch(gate), cuda_launch(up), cuda_launch(down))
                {
                    assert_eq!(gate.cols(), x.len());
                    assert_eq!(up.cols(), x.len());
                    assert_eq!(down.cols(), gate.rows());
                    match ferrox_cuda::gpu::launch_dense_ffn_swiglu(&g, &u, &d, x) {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!("ferrox: CUDA dense FFN fuse failed, trying next: {e}");
                        }
                    }
                }
            }
        }
        #[cfg(feature = "metal")]
        {
            if metal_dense_enabled() {
                fn metal_launch(m: &WeightMatrix) -> Option<ferrox_metal::gpu::MatvecLaunch<'_>> {
                    let WeightMatrix::Quantized {
                        data,
                        rows,
                        cols: _,
                        kind,
                    } = m
                    else {
                        return None;
                    };
                    let kind_name = match kind {
                        QuantKind::Q8_0 => "Q8_0",
                        QuantKind::Q4_0 => "Q4_0",
                        QuantKind::Q4K => "Q4_K",
                        QuantKind::Q5K => "Q5_K",
                        QuantKind::Q6K => "Q6_K",
                        QuantKind::IQ4XS => "IQ4_XS",
                        _ => return None,
                    };
                    let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
                        ferrox_metal::gpu::matvec_launch_meta(kind_name)?;
                    let row_bytes = if *rows == 0 {
                        0
                    } else {
                        data.as_slice().len() / *rows
                    };
                    Some(ferrox_metal::gpu::MatvecLaunch {
                        kernel_src: src,
                        fn_name,
                        block_bytes,
                        block_elems,
                        weights: data.as_slice(),
                        rows: *rows,
                        row_bytes,
                        rows_per_tg,
                    })
                }
                if let (Some(g), Some(u), Some(d)) =
                    (metal_launch(gate), metal_launch(up), metal_launch(down))
                {
                    assert_eq!(gate.cols(), x.len());
                    assert_eq!(up.cols(), x.len());
                    assert_eq!(down.cols(), gate.rows());
                    match ferrox_metal::gpu::launch_dense_ffn_swiglu(&g, &u, &d, x) {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!("ferrox: Metal dense FFN fuse failed, falling back: {e}");
                        }
                    }
                }
            }
        }
        None
    }

    /// Runs one weight matrix against `batch_size` activations in a
    /// single Metal command buffer (shared resident weights, one
    /// upload of `x_batch`, one GPU wait). `x_batch` / return layout
    /// match [`Self::apply_batch`]: `[batch, cols]` → `[batch, rows]`.
    /// Returns `None` if Metal dense is off, the kind lacks a Metal
    /// kernel, or the launch fails.
    ///
    /// Q4_K / Q6_K with `batch_size >= 2` use
    /// [`ferrox_metal::gpu::launch_q4_k_matmul_batch`] /
    /// [`ferrox_metal::gpu::launch_q6_k_matmul_batch`]; other kinds
    /// fall through to [`ferrox_metal::gpu::launch_matvec_batch`].
    #[cfg(feature = "metal")]
    pub fn apply_gpu_batch(&self, x_batch: &[f32], batch_size: usize) -> Option<Vec<f32>> {
        if !metal_dense_enabled() || batch_size == 0 {
            return None;
        }
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        let kind_name = match kind {
            QuantKind::Q8_0 => "Q8_0",
            QuantKind::Q4_0 => "Q4_0",
            QuantKind::Q4K => "Q4_K",
            QuantKind::Q5K => "Q5_K",
            QuantKind::Q6K => "Q6_K",
            QuantKind::IQ4XS => "IQ4_XS",
            _ => return None,
        };
        let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
            ferrox_metal::gpu::matvec_launch_meta(kind_name)?;
        let row_bytes = self.block_bytes_per_row(*kind, *cols);
        // First-cut Q4/Q6 matmul kernels can lose to N× matvec on Host B
        // for typical chat prompts (8B fair: ~17 vs ~21 prompt tok/s).
        // Opt in with FERROX_METAL_MATMUL=1 once tiling improves.
        let use_matmul = batch_size >= 2
            && matches!(
                std::env::var("FERROX_METAL_MATMUL").ok().as_deref(),
                Some("1") | Some("true") | Some("on")
            );
        // Weight-reuse mul_mm for prefill batch ≥ 4 (Q4_0 / Q4_K / Q6_K).
        // Default **on**; `FERROX_METAL_MUL_MM=0` forces N× matvec batch.
        // Threshold 4 (was 8) covers shorter prompts without changing the
        // decode path (batch_size == 1 still uses matvec).
        let use_mul_mm = batch_size >= 4
            && !matches!(
                std::env::var("FERROX_METAL_MUL_MM").ok().as_deref(),
                Some("0") | Some("false") | Some("off")
            );
        if use_mul_mm {
            match kind {
                QuantKind::Q4_0 => {
                    match ferrox_metal::gpu::launch_q4_0_mul_mm(
                        data.as_slice(),
                        x_batch,
                        *rows,
                        row_bytes,
                        batch_size,
                    ) {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!("ferrox: Metal Q4_0 mul_mm failed, matvec fallback: {e}");
                        }
                    }
                }
                QuantKind::Q4K => {
                    // True simdgroup GEMM: each 64x32 output tile reads its
                    // weight slice once into threadgroup memory instead of
                    // once per token. `launch_q4_k_mul_mm` below is the
                    // batched-matvec fallback it replaces -- correct, but it
                    // re-reads the whole matrix for every token, which is why
                    // Metal `pp512` was 14-99x behind llama.cpp.
                    match ferrox_metal::gpu::launch_q4_k_mul_mm_sg(
                        data.as_slice(),
                        x_batch,
                        *rows,
                        row_bytes,
                        batch_size,
                    ) {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!(
                                "ferrox: Metal Q4_K simdgroup mul_mm failed, batched-matvec fallback: {e}"
                            );
                        }
                    }
                    match ferrox_metal::gpu::launch_q4_k_mul_mm(
                        data.as_slice(),
                        x_batch,
                        *rows,
                        row_bytes,
                        batch_size,
                    ) {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!(
                                "ferrox: Metal Q4_K mul_mm (MUL_MM path) failed, matvec fallback: {e}"
                            );
                        }
                    }
                }
                QuantKind::Q6K => {
                    match ferrox_metal::gpu::launch_q6_k_matmul_batch(
                        data.as_slice(),
                        x_batch,
                        *rows,
                        row_bytes,
                        batch_size,
                    ) {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!(
                                "ferrox: Metal Q6_K matmul batch (MUL_MM path) failed, matvec fallback: {e}"
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        if use_matmul {
            match kind {
                QuantKind::Q4K => {
                    match ferrox_metal::gpu::launch_q4_k_matmul_batch(
                        data.as_slice(),
                        x_batch,
                        *rows,
                        row_bytes,
                        batch_size,
                    ) {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!(
                                "ferrox: Metal Q4_K matmul batch failed, matvec fallback: {e}"
                            );
                        }
                    }
                }
                QuantKind::Q6K => {
                    match ferrox_metal::gpu::launch_q6_k_matmul_batch(
                        data.as_slice(),
                        x_batch,
                        *rows,
                        row_bytes,
                        batch_size,
                    ) {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!(
                                "ferrox: Metal Q6_K matmul batch failed, matvec fallback: {e}"
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        let launch = ferrox_metal::gpu::MatvecLaunch {
            kernel_src: src,
            fn_name,
            block_bytes,
            block_elems,
            weights: data.as_slice(),
            rows: *rows,
            row_bytes,
            rows_per_tg,
        };
        match ferrox_metal::gpu::launch_matvec_batch(&launch, x_batch, batch_size) {
            Ok(out) => Some(out),
            Err(e) => {
                eprintln!("ferrox: Metal batch matvec failed, falling back: {e}");
                None
            }
        }
    }

    #[cfg(feature = "metal")]
    fn metal_kind_supported(kind: QuantKind) -> bool {
        matches!(
            kind,
            QuantKind::Q8_0 | QuantKind::Q4_0 | QuantKind::Q4K | QuantKind::Q5K | QuantKind::Q6K
        )
    }

    /// The block size (in bytes) for exactly the quant kinds
    /// `apply_gpu` dispatches to a real kernel for -- a small,
    /// deliberately partial mirror of `block_bytes_per_row`'s per-kind
    /// match (only these five formats have a real GPU kernel today).
    #[cfg(feature = "cuda")]
    fn block_bytes_for_kind(kind: QuantKind) -> usize {
        match kind {
            QuantKind::Q8_0 => ferrox_quant::Q8_0_BLOCK_BYTES,
            QuantKind::Q4_0 => ferrox_quant::Q4_0_BLOCK_BYTES,
            QuantKind::Q4K => ferrox_quant::Q4_K_BLOCK_BYTES,
            QuantKind::Q5K => ferrox_quant::Q5_K_BLOCK_BYTES,
            QuantKind::Q6K => ferrox_quant::Q6_K_BLOCK_BYTES,
            _ => unreachable!("apply_gpu only calls this for the five GPU-dispatchable kinds"),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// `dequant_row` must reproduce exactly the values a full-buffer
    /// dequantization of the same row produces, for every storage
    /// variant -- and read only that row's bytes (each row here has
    /// distinct values, so an off-by-one-row slice fails loudly).
    #[test]
    fn dequant_row_matches_full_dequant_per_row() {
        // F32 variant.
        let rows = 3;
        let cols = 64;
        let f32_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let m = WeightMatrix::F32(Tensor::new(f32_data.clone(), vec![rows, cols]));
        for r in 0..rows {
            assert_eq!(m.dequant_row(r), &f32_data[r * cols..(r + 1) * cols]);
        }

        // Quantized (Q8_0) variant: quantize each row independently and
        // compare dequant_row against dequantizing that row's bytes.
        let mut packed = Vec::new();
        for r in 0..rows {
            packed.extend(make_q8_0_row(&f32_data[r * cols..(r + 1) * cols]));
        }
        let row_bytes = packed.len() / rows;
        let q = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed.clone()),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        for r in 0..rows {
            let expected =
                ferrox_quant::dequant_q8_0(&packed[r * row_bytes..(r + 1) * row_bytes]).unwrap();
            assert_eq!(q.dequant_row(r), expected, "Q8_0 row {r}");
        }

        // Mxfp4 (two-buffer) variant: arbitrary valid bytes, compare
        // against the row-level reference dequantizer directly.
        let cols = 64;
        let packed: Vec<u8> = pseudo_bytes(7, rows * cols / 2);
        let scales: Vec<u8> = pseudo_bytes(11, rows * cols / 32);
        let m = WeightMatrix::Mxfp4 {
            packed: WeightBytes::Owned(packed.clone()),
            scale: WeightBytes::Owned(scales.clone()),
            rows,
            cols,
        };
        for r in 0..rows {
            let expected = ferrox_quant::dequant_mxfp4_row(
                &packed[r * cols / 2..(r + 1) * cols / 2],
                &scales[r * cols / 32..(r + 1) * cols / 32],
            )
            .unwrap();
            assert_eq!(m.dequant_row(r), expected, "Mxfp4 row {r}");
        }
    }

    /// A quantized matrix used as an embedding table: `dequant_row`
    /// then a dot product must agree with `apply` against a one-hot...
    /// no -- more directly, with the fused `dot` of that row, proving
    /// row lookup and matmul read identical bytes.
    #[test]
    fn dequant_row_agrees_with_fused_dot_on_the_same_row() {
        let rows = 4;
        let cols = 64;
        let f32_data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        let mut packed = Vec::new();
        for r in 0..rows {
            packed.extend(make_q8_0_row(&f32_data[r * cols..(r + 1) * cols]));
        }
        let q = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.031).cos()).collect();
        let applied = q.apply(&x);
        for (r, &got) in applied.iter().enumerate() {
            let via_row: f32 = q.dequant_row(r).iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!(
                (got - via_row).abs() < 1e-4,
                "row {r}: apply={got} via dequant_row={via_row}"
            );
        }
    }

    fn make_q8_0_row(values: &[f32]) -> Vec<u8> {
        ferrox_quant::quantize_q8_0(values)
    }

    /// Deterministic byte generator for MXFP4 test fixtures (no
    /// quantizer exists in `ferrox_quant` -- MXFP4 is only ever a
    /// real, already-quantized checkpoint format, never produced by
    /// ferrox -- so tests build arbitrary-but-valid-shaped bytes
    /// directly, same convention as `ferrox-models::kimi_loader`'s
    /// tests).
    fn pseudo_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    /// Clamped to a realistic E8M0 scale range -- see
    /// `ferrox-models::kimi_loader`'s identical helper for why (byte
    /// 255 is OCP-spec-reserved for NaN, and bytes above ~252 can
    /// legitimately overflow f32::MAX when combined with E2M1's max
    /// magnitude; neither is representative of a real trained weight).
    fn pseudo_mxfp4_scale_bytes(seed: u32, len: usize) -> Vec<u8> {
        pseudo_bytes(seed, len)
            .into_iter()
            .map(|b| b % 180)
            .collect()
    }

    #[test]
    fn f32_and_mxfp4_paths_agree() {
        let rows = 2;
        let cols = 64; // 2 MXFP4 groups of 32 per row
        let packed = pseudo_bytes(1, rows * (cols / 2));
        let scale = pseudo_mxfp4_scale_bytes(2, rows * (cols / ferrox_quant::MXFP4_GROUP_SIZE));
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.3).collect();

        // Independent reference: dequantize each row to plain f32 (the
        // already-tested `dequant_mxfp4_row`), then use the ordinary
        // F32 matmul path.
        let mut f32_weights = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let prow = &packed[r * (cols / 2)..(r + 1) * (cols / 2)];
            let srow = &scale[r * (cols / ferrox_quant::MXFP4_GROUP_SIZE)
                ..(r + 1) * (cols / ferrox_quant::MXFP4_GROUP_SIZE)];
            f32_weights.extend(ferrox_quant::dequant_mxfp4_row(prow, srow).unwrap());
        }
        let f32_matrix = WeightMatrix::F32(Tensor::new(f32_weights, vec![rows, cols]));
        let f32_out = f32_matrix.apply(&x);

        let mxfp4_matrix = WeightMatrix::Mxfp4 {
            packed: WeightBytes::Owned(packed),
            scale: WeightBytes::Owned(scale),
            rows,
            cols,
        };
        let mxfp4_out = mxfp4_matrix.apply(&x);

        assert_eq!(f32_out.len(), rows);
        assert_eq!(mxfp4_out.len(), rows);
        for (f, m) in f32_out.iter().zip(mxfp4_out.iter()) {
            assert!((f - m).abs() < 1e-3, "f32={f} mxfp4={m}");
        }
    }

    #[test]
    fn mxfp4_apply_batch_matches_sequential_apply_calls() {
        let rows = 3;
        let cols = 64;
        let packed = pseudo_bytes(3, rows * (cols / 2));
        let scale = pseudo_mxfp4_scale_bytes(4, rows * (cols / ferrox_quant::MXFP4_GROUP_SIZE));
        let matrix = WeightMatrix::Mxfp4 {
            packed: WeightBytes::Owned(packed),
            scale: WeightBytes::Owned(scale),
            rows,
            cols,
        };

        let batch_size = 4;
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| ((i % 13) as f32) * 0.02 - 0.15)
            .collect();

        let batched = matrix.apply_batch(&x_batch, batch_size);
        assert_eq!(batched.len(), batch_size * rows);

        for b in 0..batch_size {
            let x = &x_batch[b * cols..(b + 1) * cols];
            let sequential = matrix.apply(x);
            let from_batch = &batched[b * rows..(b + 1) * rows];
            assert_eq!(
                sequential, from_batch,
                "batch row {b} disagrees with sequential apply()"
            );
        }
    }

    #[test]
    fn mxfp4_resident_bytes_matches_the_packed_plus_scale_byte_count_not_eager_f32() {
        let rows = 2;
        let cols = 64;
        let packed = pseudo_bytes(5, rows * (cols / 2));
        let scale = pseudo_mxfp4_scale_bytes(6, rows * (cols / ferrox_quant::MXFP4_GROUP_SIZE));
        let packed_len = packed.len();
        let scale_len = scale.len();
        let matrix = WeightMatrix::Mxfp4 {
            packed: WeightBytes::Owned(packed),
            scale: WeightBytes::Owned(scale),
            rows,
            cols,
        };

        assert_eq!(matrix.resident_bytes(), packed_len + scale_len);
        // Real MXFP4 packs 2 values/byte plus 1 scale byte per 32
        // values -- resident_bytes should be far below the 4-bytes-
        // per-value eager-f32 footprint.
        let eager_f32_bytes = rows * cols * 4;
        assert!(
            matrix.resident_bytes() * 4 < eager_f32_bytes,
            "expected MXFP4 resident bytes well under 1/4 of eager f32: got {} vs {}",
            matrix.resident_bytes(),
            eager_f32_bytes
        );
    }

    #[test]
    fn f32_and_quantized_paths_agree_within_quant_error() {
        // 1 row, 32 cols, values chosen to keep Q8_0 error small.
        let weights: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.2).collect();
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05 - 0.8).collect();

        let f32_matrix = WeightMatrix::F32(Tensor::new(weights.clone(), vec![1, 32]));
        let f32_out = f32_matrix.apply(&x);

        let packed = make_q8_0_row(&weights);
        let quant_matrix = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows: 1,
            cols: 32,
            kind: QuantKind::Q8_0,
        };
        let quant_out = quant_matrix.apply(&x);

        assert_eq!(f32_out.len(), 1);
        assert_eq!(quant_out.len(), 1);
        assert!(
            (f32_out[0] - quant_out[0]).abs() < 0.05,
            "f32={} quant={}",
            f32_out[0],
            quant_out[0]
        );
    }

    #[test]
    fn quantized_resident_bytes_is_smaller_than_f32() {
        let weights = vec![0.1f32; 64]; // 2 rows x 32 cols
        let f32_matrix = WeightMatrix::F32(Tensor::new(weights.clone(), vec![2, 32]));

        let mut packed = Vec::new();
        for chunk in weights.chunks(32) {
            packed.extend(ferrox_quant::quantize_q8_0(chunk));
        }
        let quant_matrix = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows: 2,
            cols: 32,
            kind: QuantKind::Q8_0,
        };

        assert_eq!(f32_matrix.resident_bytes(), 64 * 4); // 256 bytes
        assert_eq!(quant_matrix.resident_bytes(), 2 * 34); // 68 bytes
        assert!(quant_matrix.resident_bytes() < f32_matrix.resident_bytes());
        // Q8_0 should be close to the theoretical ~4x reduction vs f32.
        let ratio = f32_matrix.resident_bytes() as f32 / quant_matrix.resident_bytes() as f32;
        assert!(ratio > 3.5, "expected ~4x reduction, got {ratio}x");
    }

    #[test]
    fn rows_and_cols_report_correctly_for_both_variants() {
        let f32_matrix = WeightMatrix::F32(Tensor::new(vec![0.0; 6], vec![2, 3]));
        assert_eq!(f32_matrix.rows(), 2);
        assert_eq!(f32_matrix.cols(), 3);

        let quant_matrix = WeightMatrix::Quantized {
            data: WeightBytes::Owned(vec![0u8; 34]),
            rows: 1,
            cols: 32,
            kind: QuantKind::Q8_0,
        };
        assert_eq!(quant_matrix.rows(), 1);
        assert_eq!(quant_matrix.cols(), 32);
    }

    #[test]
    #[should_panic]
    fn apply_panics_on_activation_length_mismatch() {
        let f32_matrix = WeightMatrix::F32(Tensor::new(vec![0.0; 6], vec![2, 3]));
        f32_matrix.apply(&[1.0, 2.0]); // wrong length (needs 3)
    }

    #[test]
    fn apply_batch_with_batch_size_one_matches_apply() {
        let weights: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.13).collect();
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.02 - 0.3).collect();

        let f32_matrix = WeightMatrix::F32(Tensor::new(weights.clone(), vec![1, 32]));
        let single = f32_matrix.apply(&x);
        let batched = f32_matrix.apply_batch(&x, 1);
        assert_eq!(single, batched);

        let packed = ferrox_quant::quantize_q8_0(&weights);
        let quant_matrix = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows: 1,
            cols: 32,
            kind: QuantKind::Q8_0,
        };
        let single_q = quant_matrix.apply(&x);
        let batched_q = quant_matrix.apply_batch(&x, 1);
        assert_eq!(single_q, batched_q);
    }

    #[test]
    fn apply_batch_matches_sequential_apply_calls_for_each_row_f32() {
        let rows = 3;
        let cols = 32;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let matrix = WeightMatrix::F32(Tensor::new(weights, vec![rows, cols]));

        let batch_size = 4;
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| ((i % 13) as f32) * 0.03 - 0.2)
            .collect();

        let batched = matrix.apply_batch(&x_batch, batch_size);
        assert_eq!(batched.len(), batch_size * rows);

        for b in 0..batch_size {
            let x = &x_batch[b * cols..(b + 1) * cols];
            let sequential = matrix.apply(x);
            let from_batch = &batched[b * rows..(b + 1) * rows];
            assert_eq!(
                sequential, from_batch,
                "batch row {b} disagrees with sequential apply()"
            );
        }
    }

    #[test]
    fn apply_batch_matches_sequential_apply_calls_for_each_row_quantized() {
        let rows = 3;
        let cols = 32;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.07)
            .collect();
        let mut packed = Vec::new();
        for row in weights.chunks(cols) {
            packed.extend(ferrox_quant::quantize_q8_0(row));
        }
        let matrix = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };

        let batch_size = 5;
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| ((i % 11) as f32) * 0.04 - 0.25)
            .collect();

        let batched = matrix.apply_batch(&x_batch, batch_size);
        assert_eq!(batched.len(), batch_size * rows);

        for b in 0..batch_size {
            let x = &x_batch[b * cols..(b + 1) * cols];
            let sequential = matrix.apply(x);
            let from_batch = &batched[b * rows..(b + 1) * rows];
            for (s, fb) in sequential.iter().zip(from_batch.iter()) {
                assert!(
                    (s - fb).abs() < 1e-4,
                    "batch row {b}: sequential={s} batched={fb}"
                );
            }
        }
    }

    #[test]
    fn apply_batch_with_zero_batch_size_returns_empty() {
        let matrix = WeightMatrix::F32(Tensor::new(vec![0.0; 6], vec![2, 3]));
        let out = matrix.apply_batch(&[], 0);
        assert!(out.is_empty());
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    mod gpu_dispatch {
        use super::*;

        /// `apply_gpu` must return `None` for `F32` -- and, crucially,
        /// without ever touching the CUDA driver at all (this runs on
        /// every CI machine, none of which have a GPU): the `let ...
        /// else { return None }` pattern match happens before any
        /// `ferrox_cuda` call, so this is a real, meaningful assertion
        /// about dispatch behavior, not a stub.
        #[test]
        fn apply_gpu_returns_none_for_f32() {
            let matrix = WeightMatrix::F32(Tensor::new(vec![0.0; 6], vec![2, 3]));
            assert!(matrix.apply_gpu(&[0.0, 0.0, 0.0]).is_none());
        }

        #[test]
        fn apply_gpu_returns_none_for_mxfp4() {
            let matrix = WeightMatrix::Mxfp4 {
                packed: WeightBytes::Owned(vec![0u8; 32]),
                scale: WeightBytes::Owned(vec![0u8; 2]),
                rows: 1,
                cols: 64,
            };
            assert!(matrix.apply_gpu(&vec![0.0; 64]).is_none());
        }

        /// A `Quantized` matrix whose `kind` has no real CUDA kernel
        /// (only Q8_0/Q4_0/Q4_K/Q5_K/Q6_K do) must also fall back to
        /// `None`, not panic on the `unreachable!()` in
        /// `block_bytes_for_kind` -- proving the two match arms
        /// (`apply_gpu`'s early match, `block_bytes_for_kind`'s
        /// exhaustive one) stay in sync.
        #[test]
        fn apply_gpu_returns_none_for_an_unsupported_quant_kind() {
            let matrix = WeightMatrix::Quantized {
                data: WeightBytes::Owned(vec![0u8; ferrox_quant::Q2_K_BLOCK_BYTES]),
                rows: 1,
                cols: ferrox_quant::Q2_K_BLOCK_ELEMS,
                kind: QuantKind::Q2K,
            };
            assert!(matrix
                .apply_gpu(&vec![0.0; ferrox_quant::Q2_K_BLOCK_ELEMS])
                .is_none());
        }

        #[test]
        #[ignore = "requires real GPU hardware (CUDA or Metal) -- run with --ignored"]
        fn apply_gpu_matches_apply_for_q8_0_on_real_hardware() {
            let weights: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.05).collect();
            let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
            let packed = ferrox_quant::quantize_q8_0(&weights);
            let matrix = WeightMatrix::Quantized {
                data: WeightBytes::Owned(packed),
                rows: 1,
                cols: 64,
                kind: QuantKind::Q8_0,
            };

            let cpu = matrix.apply_cpu(&x);
            let gpu = matrix
                .apply_gpu(&x)
                .expect("Q8_0 must dispatch to a real GPU kernel");
            assert_eq!(cpu.len(), gpu.len());
            for (c, g) in cpu.iter().zip(gpu.iter()) {
                assert!((c - g).abs() < 1e-2, "cpu={c} gpu={g}");
            }
        }
    }
}
