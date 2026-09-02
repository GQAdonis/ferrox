//! The Q4_K weight encoder: a transcription of llama.cpp's
//! `quantize_row_q4_K_ref` (`ggml/src/ggml-quants.c`), not a
//! reimplementation of it.
//!
//! A K-quant is NOT min/max over a block. Q4_K's 256-element
//! super-block is fitted in three stages, and every one of them has to
//! be reproduced exactly or the file differs:
//!
//! 1. Each of the 8 sub-blocks of 32 gets an **iterative** affine fit
//!    (`make_qkx2_quants`): 21 candidate inverse scales are tried, each
//!    one re-solves a weighted least-squares for (scale, min) from the
//!    integer codes it produced, and the lowest weighted squared error
//!    wins. The weights are `sqrt(mean(x^2)) + |x|`, so a sub-block's
//!    large values pull the fit toward themselves.
//! 2. The 8 scales and 8 mins are themselves quantized to 6 bits
//!    against the super-block's `d`/`dmin` and packed into 12 bytes.
//! 3. The 4-bit codes are then recomputed **against the 6-bit-rounded**
//!    scale and min, not against the fit from stage 1 -- so stage 3
//!    sees a slightly different affine map than stage 1 did.
//!
//! A naive min/max encoder skips all three and produces a file that
//! loads and generates measurably worse text. That is the failure this
//! module exists to not ship, so the arithmetic below is deliberately
//! the same shape as the C, down to the operation order in the
//! least-squares accumulation.
//!
//! Deviations from upstream, all of them shown not to change a byte by
//! `q4_k_matches_llama_cpp_quantize_row_q4_k_ref` in `tests`:
//!
//! * `nearest_int`'s `assert(fabsf(fval) <= 4194303.f)` is not
//!   reproduced. It is compiled out of the release `libggml` that
//!   `llama-quantize` actually links, so asserting here would make
//!   ferrox stop where llama.cpp proceeds -- a refusal that fires on
//!   input llama.cpp handles is not coverage, it is a different tool.
//! * The 6-bit scale/min are unpacked for stage 3 by the same
//!   [`crate::q4_k_scale_min`] the *reader* uses, rather than by a
//!   second copy of `get_scale_min_k4`. Two copies of that bit-packing
//!   is precisely the shape of bug this repo keeps finding; one
//!   function means the encoder and the decoder cannot disagree about
//!   what was packed.

use half::f16;

use crate::{q4_k_scale_min, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q4_K_SCALE_BYTES};

/// Sub-blocks per Q4_K super-block, and elements in each.
const SUB: usize = 8;
const SUB_ELEMS: usize = Q4_K_BLOCK_ELEMS / SUB; // 32

/// ggml's `nearest_int`: add 1.5 * 2^23 so the mantissa's low bits hold
/// the rounded integer, then read them back out.
///
/// This is **round-half-to-even**, because it is the FPU's own rounding
/// mode that does the work. `f32::round` is round-half-away-from-zero
/// and disagrees on every exact tie -- and ties are not rare here: the
/// candidate inverse scales in [`make_qkx2_quants`] walk a 0.1-wide
/// grid, so `iscale * (x - min)` lands on `.5` constantly.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// llama.cpp's `make_qkx2_quants`: fit `x[i] ~= scale * L[i] - the_min`
/// with `L[i]` in `0..=nmax`, minimising the `weights`-weighted error.
///
/// Returns `(scale, the_min)` and fills `l`. `laux` is scratch, passed
/// in rather than allocated because the C does the same and this runs
/// once per 32 weights of the checkpoint.
///
/// The signature is upstream's, `use_mad` and all: Q2_K passes `true`
/// with `n = 16`, Q5_K passes `nmax = 31`. Keeping the parameters means
/// the next K-quant is a call, not a copy of this function with two
/// constants changed -- which is how this repo has lost a model feature
/// eight times.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    laux: &mut [u8],
    nmax: i32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> (f32, f32) {
    let n = x.len();
    debug_assert_eq!(weights.len(), n);
    debug_assert_eq!(l.len(), n);
    debug_assert!(laux.len() >= n);

    // Deliberately not `min.min(x[i])` / `max.max(x[i])`. They differ
    // from the C comparisons only when `x[0]` is NaN -- Rust's
    // `f32::min` returns the non-NaN operand, `x[i] < NaN` is false and
    // keeps the NaN -- so no fixture can tell them apart on a real
    // checkpoint. The C's shape is kept anyway, because a checkpoint
    // with a NaN weight should produce llama.cpp's bytes rather than
    // politely different ones. Same choice, same reason, as the `amax`
    // fold in the Q8_0 encoder next door.
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l[..n].fill(0);
        return (0.0, -min);
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = li as u8;
        let diff = scale * l[i] as f32 + min - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }

    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = li as u8;
            let w = weights[i];
            sum_l += w * li as f32;
            sum_l2 += w * li as f32 * li as f32;
            sum_xl += w * li as f32 * x[i];
        }
        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// Encodes one Q4_K super-block (exactly [`Q4_K_BLOCK_ELEMS`] values)
/// and appends its [`Q4_K_BLOCK_BYTES`] bytes to `out`.
pub fn encode_block_q4_k(block: &[f32; Q4_K_BLOCK_ELEMS], out: &mut Vec<u8>) {
    // `l` is deliberately carried from stage 1 into stage 3. Stage 3
    // skips any sub-block whose reconstructed `d` rounded to zero (`if
    // (!d) continue;` upstream), and the codes then written are the
    // ones stage 1 left behind -- NOT zeros. Clearing `l` per sub-block
    // reads as tidier and writes a different file.
    let mut l = [0u8; Q4_K_BLOCK_ELEMS];
    let mut laux = [0u8; SUB_ELEMS];
    let mut weights = [0f32; SUB_ELEMS];
    let mut mins = [0f32; SUB];
    let mut scales = [0f32; SUB];

    let mut max_scale = 0f32; // deducting the min keeps scales positive
    let mut max_min = 0f32;
    for j in 0..SUB {
        let lo = SUB_ELEMS * j;
        let xs = &block[lo..lo + SUB_ELEMS];
        let mut sum_x2 = 0f32;
        for &v in xs {
            sum_x2 += v * v;
        }
        let av_x = (sum_x2 / SUB_ELEMS as f32).sqrt();
        for (w, &v) in weights.iter_mut().zip(xs) {
            *w = av_x + v.abs();
        }
        let (scale, min) = make_qkx2_quants(
            xs,
            &weights,
            &mut l[lo..lo + SUB_ELEMS],
            &mut laux,
            15,
            -1.0,
            0.1,
            20,
            false,
        );
        scales[j] = scale;
        mins[j] = min;
        if scale > max_scale {
            max_scale = scale;
        }
        if min > max_min {
            max_min = min;
        }
    }

    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut packed = [0u8; Q4_K_SCALE_BYTES];
    for j in 0..SUB {
        // Upstream's `MIN(63, ls)`. It cannot fire on THIS path:
        // `inv_scale` is `63/max_scale` and `max_scale` is the largest
        // of `scales`, so the product is at most 63 plus an ulp and
        // rounds to 63. It is kept because it is what the C says and
        // because the imatrix variant of this encoder
        // (`quantize_row_q4_K_impl`) reaches the same packing from
        // `make_qp_quants`, where the bound is not automatic -- but no
        // fixture here can turn its removal red, and saying so is
        // better than implying the golden covers it.
        // The cast comes BEFORE the clamp, because upstream's does:
        //
        //     uint8_t ls = nearest_int(inv_scale*scales[j]);
        //     ls = MIN(63, ls);
        //
        // `nearest_int` returns `int`, and storing it in a `uint8_t`
        // truncates to eight bits FIRST. Clamping to 63 and casting
        // afterwards is the same for every value in `0..=255` and
        // different for a negative one: C wraps -1 to 255 and then
        // clamps to 63, this order clamps -1 to -1 and casts to 255.
        //
        // A negative reaches here when a sub-block's least-squares fit
        // returns a negative scale while some other sub-block's is
        // positive, so `inv_scale` is positive and the product is not.
        // Upstream's comment says scales are always positive "as we are
        // deducting the min", which is the assumption this arithmetic
        // quietly does not rely on. Rare, and it was 0.55% of the
        // super-blocks in a real Qwen3-0.6B tensor.
        let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
        let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
        if j < 4 {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            packed[j - 4] |= (ls >> 4) << 6;
            packed[j] |= (lm >> 4) << 6;
        }
    }
    let d = f16::from_f32(max_scale / 63.0);
    let dmin = f16::from_f32(max_min / 63.0);

    for j in 0..SUB {
        let (sc, m) = q4_k_scale_min(j, &packed);
        let dj = d.to_f32() * sc as f32;
        if dj == 0.0 {
            continue;
        }
        let dm = dmin.to_f32() * m as f32;
        for ii in 0..SUB_ELEMS {
            let idx = SUB_ELEMS * j + ii;
            l[idx] = nearest_int((block[idx] + dm) / dj).clamp(0, 15) as u8;
        }
    }

    out.reserve(Q4_K_BLOCK_BYTES);
    out.extend_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&dmin.to_le_bytes());
    out.extend_from_slice(&packed);
    for j in (0..Q4_K_BLOCK_ELEMS).step_by(64) {
        for i in 0..32 {
            out.push(l[j + i] | (l[j + i + 32] << 4));
        }
    }
}

/// Encodes a whole row (or any slice whose length is a multiple of
/// [`Q4_K_BLOCK_ELEMS`]) into Q4_K super-blocks, appending to `out`.
///
/// Returns `None` when `src.len()` is not a multiple of the super-block
/// size. llama.cpp handles that case by silently *changing type* --
/// `tensor_type_fallback` rewrites a Q4_K tensor with an awkward row
/// length to Q5_0, and to F16 if that does not fit either -- and ferrox
/// has neither encoder, so this refuses instead of padding. Padding
/// would write more elements than the tensor's shape declares and every
/// following row would decode shifted.
pub fn encode_row_q4_k(src: &[f32], out: &mut Vec<u8>) -> Option<()> {
    let (blocks, rest) = src.as_chunks::<Q4_K_BLOCK_ELEMS>();
    if !rest.is_empty() {
        return None;
    }
    out.reserve(blocks.len() * Q4_K_BLOCK_BYTES);
    for block in blocks {
        encode_block_q4_k(block, out);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant_q4_k;

    /// Four super-blocks of deterministic, **f16-shaped** input, built
    /// so that every branch of the reference a plausible rewrite would
    /// get wrong is exercised at least once.
    ///
    /// f16-shaped is not decoration. Step 1 of this work learned it the
    /// expensive way: its Q8_0 golden was documented as catching
    /// `v * (1/d)` versus `v / d` and did not, because over uniform f32
    /// noise the two spellings agree for 8192 consecutive values. f16's
    /// 11-bit mantissa lands on rounding boundaries constantly, and
    /// real weights are f16, so the fixture is f16.
    ///
    /// The sub-block roster, by index (32 sub-blocks of 32 values):
    ///
    /// * 8 -- all zero: `max == min`, the early return that fills the
    ///   codes with 0 and reports a scale of 0.
    /// * 9 -- constant non-zero: `max == min` again, but with a min
    ///   that is clamped to 0 because it is positive.
    /// * 10 -- all positive: exercises `if (min > 0) min = 0`.
    /// * 11 -- all negative: `max` is negative and `min` is not clamped.
    /// * 17 -- four orders of magnitude smaller than its super-block's
    ///   neighbours, so its 6-bit scale rounds to **zero** and stage 3
    ///   skips it. The codes written for it are the ones stage 1 left
    ///   in `l`; an encoder that clears `l` per sub-block writes 32
    ///   different bytes here and nowhere else.
    /// * everything else -- weight-like noise at one of four gains, so
    ///   sub-blocks within a super-block disagree about scale and the
    ///   6-bit scale quantization actually has to do something.
    ///
    /// **The seed is not decorative either.** Two of the reference's
    /// decisions -- `nearest_int`'s round-half-to-even and the
    /// `this_min > 0` clamp inside the least-squares step -- only show
    /// up on some data, and the first seed tried exercised neither: the
    /// whole golden stayed green with `f32::round` substituted for
    /// `nearest_int`. This one was picked by encoding 3999 candidate
    /// fixtures twice, once with each spelling of every decision in the
    /// reference, and keeping a seed where all of them differ. 255 of
    /// the 3999 qualify, so this is a fixture chosen to be able to
    /// fail, not a seed fitted to one assertion.
    fn sample_input() -> Vec<f32> {
        const GAINS: [f32; 4] = [0.02, 0.05, 0.1, 0.25];
        let mut state: u32 = 0xb54c_da26;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // [-1, 1)
            ((state >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let mut out = Vec::with_capacity(4 * Q4_K_BLOCK_ELEMS);
        for sub in 0..4 * SUB {
            for _ in 0..SUB_ELEMS {
                let v = next();
                let shaped = match sub {
                    8 => 0.0,
                    9 => 0.125,
                    10 => v.abs() * 0.05 + 0.01,
                    11 => -(v.abs() * 0.05 + 0.01),
                    17 => v * 1e-4,
                    _ => v * GAINS[sub % GAINS.len()],
                };
                out.push(f16::from_f32(shaped).to_f32());
            }
        }
        out
    }

    /// Regenerates the golden below. Ignored, because it needs a
    /// llama.cpp checkout: it writes [`sample_input`] as raw
    /// little-endian f32 to `$FERROX_Q4_K_FIXTURE_OUT`, which the C
    /// harness described in the PR body then feeds to llama.cpp's own
    /// encoder.
    ///
    /// The input lives here and only here. A C harness that re-derived
    /// the same values from a copy of the generator would be two
    /// structures that must agree with nothing enforcing it -- this
    /// repo's dominant bug shape -- and it would silently compare two
    /// different inputs the day one copy drifted.
    #[test]
    #[ignore = "developer tool: regenerates LLAMA_CPP_Q4_K_GOLDEN"]
    fn dump_the_fixture_the_c_harness_reads() {
        let path = std::env::var("FERROX_Q4_K_FIXTURE_OUT")
            .expect("set FERROX_Q4_K_FIXTURE_OUT to the path to write");
        let mut bytes = Vec::new();
        for v in sample_input() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// llama.cpp's own bytes for [`sample_input`].
    ///
    /// Produced by linking `.scratch/llama.cpp/build/bin/libggml-base`
    /// and calling the exported `quantize_row_q4_K_ref` on the f32s
    /// [`dump_the_fixture_the_c_harness_reads`] writes. The same
    /// harness also calls `ggml_quantize_chunk(GGML_TYPE_Q4_K, ...)` --
    /// the entry point `llama-quantize` itself goes through -- and
    /// asserts the two agree, so this is what the real tool writes and
    /// not merely what a reference function does.
    const LLAMA_CPP_Q4_K_GOLDEN: [u8; 4 * Q4_K_BLOCK_BYTES] = [
        0x32, 0x10, 0x14, 0x1c, 0x05, 0x0c, 0x59, 0xff, 0x04, 0x0b, 0x58, 0xff, 0x55, 0xcc, 0x8a,
        0xb3, 0xed, 0xc8, 0xba, 0x4e, 0xeb, 0x91, 0x85, 0xa6, 0x9c, 0x87, 0xd8, 0xab, 0x42, 0xe9,
        0x87, 0x0b, 0xb3, 0x82, 0x59, 0xb2, 0xc0, 0x80, 0x87, 0xa7, 0x98, 0x62, 0x75, 0x94, 0x31,
        0x0a, 0x89, 0xda, 0xc5, 0x32, 0xd4, 0xfa, 0xf6, 0xd6, 0xc1, 0xbd, 0xf1, 0xc8, 0x6c, 0xbf,
        0xc4, 0xa0, 0xeb, 0x46, 0x7d, 0xb0, 0xf4, 0xb7, 0x95, 0xbc, 0xd1, 0xe6, 0x84, 0x8d, 0x77,
        0x1d, 0x01, 0xd7, 0x1f, 0xda, 0x1b, 0xf6, 0x4f, 0x62, 0x3f, 0xce, 0x28, 0x47, 0x5b, 0xba,
        0xeb, 0xfc, 0x04, 0xb3, 0xba, 0x44, 0x94, 0xe0, 0xd6, 0xbf, 0x7e, 0x02, 0xf0, 0xac, 0x4c,
        0xda, 0xbf, 0x21, 0x4d, 0xc7, 0xd1, 0xb0, 0x6b, 0xf0, 0xb2, 0x0a, 0x8d, 0x25, 0xbc, 0x2c,
        0xda, 0xd7, 0xa9, 0x51, 0x32, 0xa2, 0xc0, 0x5e, 0x1c, 0x86, 0x95, 0x53, 0x40, 0x7d, 0xf1,
        0xf5, 0x34, 0xf8, 0x9c, 0xf0, 0x9f, 0xa8, 0x4c, 0x48, 0x2d, 0x10, 0xeb, 0x1b, 0x00, 0x10,
        0x48, 0xc6, 0x00, 0x00, 0x40, 0xcf, 0x45, 0xcc, 0xaa, 0xff, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
        0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
        0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xe6, 0xba, 0x13,
        0x8b, 0x45, 0x3b, 0x24, 0x4e, 0x69, 0xbb, 0x96, 0xd7, 0xc9, 0xe9, 0x13, 0xad, 0x75, 0xeb,
        0xeb, 0x8a, 0x0e, 0x9e, 0x76, 0xfb, 0x0d, 0x17, 0xfe, 0x5a, 0x07, 0x7e, 0x6d, 0x95, 0xa5,
        0x30, 0x33, 0xe7, 0xf1, 0x3d, 0x29, 0xfa, 0x07, 0x84, 0x99, 0xea, 0xca, 0x06, 0xe3, 0x27,
        0xa5, 0x40, 0x07, 0x12, 0xf3, 0x55, 0x72, 0xe5, 0xa1, 0x12, 0x35, 0xee, 0x06, 0xf2, 0x51,
        0x0d, 0x11, 0x70, 0x92, 0xc1, 0xd3, 0x7c, 0x99, 0x33, 0x74, 0x49, 0x6e, 0x6d, 0x2a, 0xdc,
        0xa2, 0x57, 0x35, 0xbe, 0xd3, 0x65, 0xbb, 0xf1, 0x14, 0x05, 0x09, 0xd4, 0x87, 0x3e, 0xbf,
        0x4c, 0xc8, 0xd1, 0x30, 0x10, 0xdc, 0x1b, 0x05, 0x00, 0x58, 0xff, 0x05, 0x00, 0x58, 0xff,
        0x55, 0xdd, 0x89, 0xce, 0x44, 0xa8, 0xc2, 0x9c, 0xec, 0x84, 0x2c, 0x8f, 0x6f, 0x30, 0x74,
        0xae, 0x66, 0x2b, 0x16, 0x5b, 0xe6, 0xe0, 0x96, 0x69, 0x66, 0x8f, 0xe4, 0x5c, 0x57, 0x24,
        0x06, 0x52, 0x67, 0xa1, 0xa0, 0x43, 0xaa, 0x5d, 0x43, 0x4d, 0x7c, 0xbf, 0x78, 0x16, 0x59,
        0xf9, 0x30, 0x58, 0x03, 0x12, 0x73, 0xed, 0x8d, 0x00, 0xdd, 0x49, 0xe2, 0xf9, 0xa1, 0x88,
        0x2c, 0x80, 0x90, 0x0f, 0xb3, 0x2b, 0xf5, 0xc9, 0x72, 0x61, 0x6a, 0x85, 0x99, 0xc3, 0x02,
        0xd4, 0xd8, 0x2a, 0xee, 0x20, 0xa9, 0xcd, 0x9a, 0xa8, 0xed, 0x6b, 0x95, 0x98, 0x8c, 0x96,
        0x6f, 0x1f, 0xda, 0x13, 0xf9, 0xc7, 0x75, 0xdd, 0x55, 0x17, 0x71, 0xd4, 0xbd, 0xc5, 0x79,
        0xa0, 0x2d, 0xcb, 0x7b, 0x40, 0x76, 0x1b, 0xf4, 0x04, 0x56, 0xd2, 0x1b, 0x24, 0x44, 0x25,
        0x68, 0x01, 0x33, 0xa1, 0x92, 0xf5, 0x1f, 0x69, 0xed, 0xd1, 0xa8, 0x28, 0x3c, 0x10, 0x2d,
        0x1c, 0x05, 0x0c, 0x58, 0xff, 0x05, 0x0c, 0x53, 0xff, 0x55, 0xcc, 0x88, 0x9f, 0xba, 0xf2,
        0xc5, 0x51, 0xfe, 0x43, 0xec, 0x47, 0x96, 0x64, 0x14, 0x78, 0xf3, 0x6b, 0x46, 0x52, 0x79,
        0x15, 0x26, 0x05, 0x50, 0x9f, 0xdd, 0xec, 0x0b, 0x0d, 0x5a, 0x8f, 0xe1, 0x15, 0x76, 0x87,
        0x1c, 0x6a, 0xf7, 0xe1, 0xe2, 0x46, 0xc4, 0xcc, 0x90, 0x95, 0x40, 0x67, 0xdb, 0x70, 0x53,
        0xd4, 0x70, 0xb4, 0xcd, 0x80, 0x52, 0xb4, 0x0b, 0xc1, 0xd5, 0xda, 0x17, 0x15, 0x1e, 0x99,
        0x57, 0x22, 0x9c, 0x58, 0xc3, 0xc4, 0x5e, 0xd2, 0x78, 0x37, 0x69, 0xe2, 0x21, 0xf7, 0x83,
        0x5c, 0xa1, 0x6a, 0xbd, 0xbe, 0x72, 0xa4, 0x3d, 0x61, 0x76, 0xcb, 0x55, 0x2a, 0x01, 0x8d,
        0x14, 0xcb, 0xdc, 0x4f, 0x6f, 0x15, 0x46, 0x6e, 0xe8, 0x5d, 0x6d, 0xf0, 0xea, 0x0d, 0xaa,
        0x8f, 0xdd, 0xd7, 0x3e, 0x52, 0x20, 0x24, 0x1b, 0x15, 0x62, 0x98, 0x0a, 0xf4, 0x42, 0x9b,
        0xdc, 0x8a, 0xb7, 0xab, 0xae, 0x57,
    ];

    /// The property that makes `ferrox quantize --type q4_k_s --pure`'s
    /// output a file llama.cpp would have written, rather than one that
    /// merely decodes to similar numbers.
    ///
    /// An encoder that is within Q4_K's error bound passes any
    /// tolerance test and still writes a different file. Only this
    /// catches that.
    #[test]
    fn q4_k_matches_llama_cpp_quantize_row_q4_k_ref() {
        let x = sample_input();
        let mut got = Vec::new();
        encode_row_q4_k(&x, &mut got).unwrap();
        assert_eq!(got.len(), LLAMA_CPP_Q4_K_GOLDEN.len());
        for (b, (g, w)) in got
            .as_chunks::<Q4_K_BLOCK_BYTES>()
            .0
            .iter()
            .zip(LLAMA_CPP_Q4_K_GOLDEN.as_chunks::<Q4_K_BLOCK_BYTES>().0)
            .enumerate()
        {
            assert_eq!(g, w, "super-block {b} disagrees with llama.cpp");
        }
    }

    /// A row that is not a whole number of super-blocks is refused, not
    /// padded. llama.cpp answers this case by changing the tensor's
    /// TYPE (Q4_K -> Q5_0 -> F16); ferrox has neither encoder, and
    /// padding would shift every following row on decode.
    #[test]
    fn a_row_that_is_not_a_whole_number_of_super_blocks_is_refused() {
        let mut out = Vec::new();
        assert!(encode_row_q4_k(&[0.5; Q4_K_BLOCK_ELEMS + 1], &mut out).is_none());
        // 32 is a Q8_0 block and a Q4_K sub-block, and still not a
        // Q4_K row: the block size that matters here is 256.
        assert!(encode_row_q4_k(&[0.5; 32], &mut out).is_none());
        assert!(encode_row_q4_k(&[], &mut out).is_some());
    }

    /// Round trip through this crate's own reader, against an exact
    /// property rather than a tolerance: for every element, **no
    /// representable level is strictly closer** than the one the
    /// encoder chose.
    ///
    /// A tolerance would have to be invented, and an invented tolerance
    /// is what this whole issue exists to avoid. This is a fact instead:
    /// stage 3 rounds to the nearest of the 16 levels `d*sc*k -
    /// dmin*m`, so a nibble packed into the wrong half-byte, a scale
    /// unpacked from the wrong bits, or an off-by-one in the sub-block
    /// stride all move some element off its nearest level and turn this
    /// red. It says nothing about whether the *fit* is good -- that is
    /// what the golden above is for, and this is the weak half.
    ///
    /// (A sub-block whose 6-bit scale rounded to zero has all 16 levels
    /// equal, so it passes trivially. Sub-block 17 is that case, on
    /// purpose.)
    #[test]
    fn every_element_lands_on_its_nearest_representable_level() {
        let x = sample_input();
        let mut bytes = Vec::new();
        encode_row_q4_k(&x, &mut bytes).unwrap();
        let back = dequant_q4_k(&bytes).unwrap();
        assert_eq!(back.len(), x.len());

        for (b, block) in bytes.as_chunks::<Q4_K_BLOCK_BYTES>().0.iter().enumerate() {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let packed: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            for j in 0..SUB {
                let (sc, m) = q4_k_scale_min(j, &packed);
                let (dj, dm) = (d * sc as f32, dmin * m as f32);
                for ii in 0..SUB_ELEMS {
                    let idx = b * Q4_K_BLOCK_ELEMS + SUB_ELEMS * j + ii;
                    let chosen = (x[idx] - back[idx]).abs();
                    for k in 0..=15u8 {
                        let level = dj * k as f32 - dm;
                        assert!(
                            (x[idx] - level).abs() >= chosen,
                            "block {b} sub-block {j} element {ii}: {} is closer to {} than to the \
                             chosen {}",
                            x[idx],
                            level,
                            back[idx]
                        );
                    }
                }
            }
        }
    }
}
