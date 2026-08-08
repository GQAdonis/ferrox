//! Interleaved Q4_K × Q8_K and Q8_0 × Q8 GEMV (llama.cpp repack layouts).
//!
//! - Q4_K: packs 8 rows into `block_q4_Kx8` (`make_block_q4_Kx8`).
//! - Q8_0: packs 4 rows into `block_q8_0x4` (`make_block_q8_0x4`) with
//!   4-byte interleave for NEON SDOT `ggml_gemv_q8_0_4x4_q8_0`.
//! - Q4_0: packs 4 rows into `block_q4_0x4` (`make_block_q4_0x4`) with
//!   4-byte interleave + XOR `0x88888888` for `ggml_gemv_q4_0_4x4_q8_0`.
//!
//! Gated on `FERROX_CPU_INT_DOT`, which `ferrox` and `ferrox-server`
//! turn on by default (`=0` opts out); off in the library so golden
//! cross-validation stays reference-exact.

use crate::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q8Activations, Q8KActivations, Q4_K_BLOCK_BYTES,
    Q4_K_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use half::f16;

/// Bytes per interleaved `block_q4_Kx8` (8 × f16 d + 8 × f16 dmin + 96 scales + 1024 qs).
pub const Q4_KX8_BLOCK_BYTES: usize = 1152;
/// Number of Q4_K rows packed into one interleaved block.
pub const Q4_KX8_NROWS: usize = 8;

const KMASK1: u32 = 0x3f3f_3f3f;
const KMASK2: u32 = 0x0f0f_0f0f;
const KMASK3: u32 = 0x0303_0303;

/// Preferred qs interleave width for this CPU: 4 on Apple DotProd NEON, 8 on AVX2.
#[inline]
pub fn q4_kx8_interleave() -> usize {
    if cfg!(target_arch = "x86_64") {
        8
    } else {
        4
    }
}

#[inline]
fn f16_from_bytes(b: &[u8]) -> f32 {
    f16::from_le_bytes([b[0], b[1]]).to_f32()
}

/// Pack eight canonical Q4_K super-blocks (same column-block index) into
/// one `block_q4_Kx8`. `interleave` is 4 (ARM) or 8 (x86).
pub fn make_block_q4_kx8(
    rows: [&[u8]; Q4_KX8_NROWS],
    interleave: usize,
) -> [u8; Q4_KX8_BLOCK_BYTES] {
    debug_assert!(interleave == 4 || interleave == 8);
    for r in &rows {
        debug_assert_eq!(r.len(), Q4_K_BLOCK_BYTES);
    }
    let mut out = [0u8; Q4_KX8_BLOCK_BYTES];
    // d[8] at 0, dmin[8] at 16, scales[96] at 32, qs[1024] at 128.
    for (i, row) in rows.iter().enumerate() {
        out[i * 2] = row[0];
        out[i * 2 + 1] = row[1];
        out[16 + i * 2] = row[2];
        out[16 + i * 2 + 1] = row[3];
    }

    let end = (Q4_K_BLOCK_ELEMS * 4) / interleave; // qs bytes * 8 rows / interleave
    let qs_out = &mut out[128..];
    for i in 0..end {
        let src_id = i % Q4_KX8_NROWS;
        let src_offset = (i / Q4_KX8_NROWS) * interleave;
        let dst_offset = i * interleave;
        let src_qs = &rows[src_id][16..144];
        qs_out[dst_offset..dst_offset + interleave]
            .copy_from_slice(&src_qs[src_offset..src_offset + interleave]);
    }

    // Rearrange 6-bit scales/mins across 8 rows into 96 packed bytes
    // (llama.cpp `make_block_q4_Kx8`).
    let mut s = [0u8; 8];
    let mut m = [0u8; 8];
    let scales_out = &mut out[32..128];

    for i in 0..4 {
        for j in 0..8 {
            let sc = &rows[j][4..16];
            s[j] = sc[i] & 63;
            m[j] = sc[i + 4] & 63;
        }
        let base = i * 12;
        scales_out[base] = (s[0] & 63) + ((s[4] & 48) << 2);
        scales_out[base + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        scales_out[base + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        scales_out[base + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        scales_out[base + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        scales_out[base + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        scales_out[base + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        scales_out[base + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        scales_out[base + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        scales_out[base + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        scales_out[base + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        scales_out[base + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    for i in 0..4 {
        for j in 0..8 {
            let sc = &rows[j][4..16];
            s[j] = ((sc[i] & 192) >> 2) | (sc[i + 8] & 15);
            m[j] = ((sc[i + 4] & 192) >> 2) | ((sc[i + 8] & 240) >> 4);
        }
        let base = 48 + i * 12;
        scales_out[base] = (s[0] & 63) + ((s[4] & 48) << 2);
        scales_out[base + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        scales_out[base + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        scales_out[base + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        scales_out[base + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        scales_out[base + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        scales_out[base + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        scales_out[base + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        scales_out[base + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        scales_out[base + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        scales_out[base + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        scales_out[base + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    out
}

/// Repack a full Q4_K matrix (row-major canonical blocks) into interleaved
/// `block_q4_Kx8` groups. Rows not divisible by 8 are left out (caller
/// handles the tail with per-row dots). `interleave` defaults via
/// [`q4_kx8_interleave`].
pub fn pack_q4_k_matrix_x8(data: &[u8], rows: usize, cols: usize, interleave: usize) -> Vec<u8> {
    assert!(cols.is_multiple_of(Q4_K_BLOCK_ELEMS));
    let n_blocks = cols / Q4_K_BLOCK_ELEMS;
    let row_bytes = n_blocks * Q4_K_BLOCK_BYTES;
    assert_eq!(data.len(), rows * row_bytes);
    let n_groups = rows / Q4_KX8_NROWS;
    let mut out = Vec::with_capacity(n_groups * n_blocks * Q4_KX8_BLOCK_BYTES);
    for g in 0..n_groups {
        for b in 0..n_blocks {
            let mut row_refs: [&[u8]; Q4_KX8_NROWS] = [&[]; Q4_KX8_NROWS];
            for (r, slot) in row_refs.iter_mut().enumerate() {
                let base = (g * Q4_KX8_NROWS + r) * row_bytes + b * Q4_K_BLOCK_BYTES;
                *slot = &data[base..base + Q4_K_BLOCK_BYTES];
            }
            out.extend_from_slice(&make_block_q4_kx8(row_refs, interleave));
        }
    }
    out
}

/// Decode one 12-byte packed scale/min group into 8 scales + 8 mins (u8).
#[inline]
fn decode_scales_mins(scales12: &[u8], scales_out: &mut [u8; 8], mins_out: &mut [u8; 8]) {
    debug_assert!(scales12.len() >= 12);
    let mut utmp = [0u32; 4];
    utmp[0] = u32::from_le_bytes(scales12[0..4].try_into().unwrap());
    utmp[1] = u32::from_le_bytes(scales12[4..8].try_into().unwrap());
    utmp[2] = u32::from_le_bytes(scales12[8..12].try_into().unwrap());
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux_0 = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux_0;
    utmp[0] &= KMASK1;
    let bytes = unsafe { std::slice::from_raw_parts(utmp.as_ptr() as *const u8, 16) };
    scales_out.copy_from_slice(&bytes[0..8]);
    mins_out.copy_from_slice(&bytes[8..16]);
}

/// Scalar GEMV for interleave=4 (`ggml_gemv_q4_K_8x4_q8_K_generic`).
fn gemv_q4_kx8_q8_k_scalar_4(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    let blocklen = 4;
    let ncols_interleaved = Q4_KX8_NROWS;
    debug_assert_eq!(act.n_blocks(), nb);
    debug_assert_eq!(out.len(), n_row_groups * ncols_interleaved);
    debug_assert_eq!(packed.len(), n_row_groups * nb * Q4_KX8_BLOCK_BYTES);

    for x in 0..n_row_groups {
        let mut sumf = [0f32; 8];
        let mut sum_minf = [0f32; 8];
        let group_off = x * nb * Q4_KX8_BLOCK_BYTES;
        for l in 0..nb {
            let blk = &packed[group_off + l * Q4_KX8_BLOCK_BYTES..][..Q4_KX8_BLOCK_BYTES];
            let d = &blk[0..16];
            let dmin = &blk[16..32];
            let scales = &blk[32..128];
            let qs = &blk[128..];
            let da = act.d[l];
            let q8 = &act.q[l * Q4_K_BLOCK_ELEMS..(l + 1) * Q4_K_BLOCK_ELEMS];
            let bsums = &act.bsums[l * 16..(l + 1) * 16];

            let mut all_scales = [[0u8; 8]; 8];
            let mut all_mins = [[0u8; 8]; 8];
            for sb in 0..8 {
                decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
            }

            let n_k = Q4_K_BLOCK_ELEMS / (2 * blocklen); // 32
            for k in 0..n_k {
                let sb_pair = k / 8;
                let sc0 = &all_scales[sb_pair * 2];
                let sc1 = &all_scales[sb_pair * 2 + 1];
                for j in 0..ncols_interleaved {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let qbyte = qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        let v0 = (qbyte & 0x0F) as i32;
                        let v1 = (qbyte >> 4) as i32;
                        let a0 = q8[(k / 8) * 64 + (k % 8) * blocklen + i] as i32;
                        let a1 = q8[(k / 8) * 64 + (k % 8) * blocklen + i + 32] as i32;
                        sumi += v0 * a0 * sc0[j] as i32 + v1 * a1 * sc1[j] as i32;
                    }
                    sumf[j] += sumi as f32 * f16_from_bytes(&d[j * 2..]) * da;
                }
            }
            for sb in 0..8 {
                let mins = &all_mins[sb];
                let bsum = bsums[sb * 2] as i32 + bsums[sb * 2 + 1] as i32;
                for j in 0..ncols_interleaved {
                    sum_minf[j] +=
                        mins[j] as f32 * bsum as f32 * f16_from_bytes(&dmin[j * 2..]) * da;
                }
            }
        }
        let base = x * ncols_interleaved;
        for j in 0..ncols_interleaved {
            out[base + j] = sumf[j] - sum_minf[j];
        }
    }
}

/// Scalar GEMV for interleave=8 (`ggml_gemv_q4_K_8x8_q8_K_generic`).
fn gemv_q4_kx8_q8_k_scalar_8(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    let blocklen = 8;
    let ncols_interleaved = Q4_KX8_NROWS;
    debug_assert_eq!(act.n_blocks(), nb);
    debug_assert_eq!(out.len(), n_row_groups * ncols_interleaved);

    for x in 0..n_row_groups {
        let mut sumf = [0f32; 8];
        let mut sum_minf = [0f32; 8];
        let group_off = x * nb * Q4_KX8_BLOCK_BYTES;
        for l in 0..nb {
            let blk = &packed[group_off + l * Q4_KX8_BLOCK_BYTES..][..Q4_KX8_BLOCK_BYTES];
            let d = &blk[0..16];
            let dmin = &blk[16..32];
            let scales = &blk[32..128];
            let qs = &blk[128..];
            let da = act.d[l];
            let q8 = &act.q[l * Q4_K_BLOCK_ELEMS..(l + 1) * Q4_K_BLOCK_ELEMS];
            let bsums = &act.bsums[l * 16..(l + 1) * 16];

            let mut all_scales = [[0u8; 8]; 8];
            let mut all_mins = [[0u8; 8]; 8];
            for sb in 0..8 {
                decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
            }

            let n_k = Q4_K_BLOCK_ELEMS / (2 * blocklen); // 16
            for k in 0..n_k {
                let sb_pair = k / 4;
                let sc0 = &all_scales[sb_pair * 2];
                let sc1 = &all_scales[sb_pair * 2 + 1];
                for j in 0..ncols_interleaved {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let qbyte = qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        let v0 = (qbyte & 0x0F) as i32;
                        let v1 = (qbyte >> 4) as i32;
                        let a0 = q8[(k >> 2) * 64 + (k % 4) * blocklen + i] as i32;
                        let a1 = q8[(k >> 2) * 64 + (k % 4) * blocklen + i + 32] as i32;
                        sumi += v0 * a0 * sc0[j] as i32 + v1 * a1 * sc1[j] as i32;
                    }
                    sumf[j] += sumi as f32 * f16_from_bytes(&d[j * 2..]) * da;
                }
            }
            for sb in 0..8 {
                let mins = &all_mins[sb];
                let bsum = bsums[sb * 2] as i32 + bsums[sb * 2 + 1] as i32;
                for j in 0..ncols_interleaved {
                    sum_minf[j] +=
                        mins[j] as f32 * bsum as f32 * f16_from_bytes(&dmin[j * 2..]) * da;
                }
            }
        }
        let base = x * ncols_interleaved;
        for j in 0..ncols_interleaved {
            out[base + j] = sumf[j] - sum_minf[j];
        }
    }
}

/// GEMV: interleaved Q4_K weights × Q8_K activation → `n_row_groups * 8` f32s.
/// Dispatches to NEON (interleave 4) / AVX2 (interleave 8) when available.
pub fn gemv_q4_kx8_q8_k(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    interleave: usize,
    out: &mut [f32],
) {
    assert!(n_cols.is_multiple_of(Q4_K_BLOCK_ELEMS));
    assert_eq!(out.len(), n_row_groups * Q4_KX8_NROWS);
    match interleave {
        4 => {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    unsafe {
                        neon::gemv_q4_kx8_q8_k_neon_sdot(packed, act, n_cols, n_row_groups, out);
                    }
                    return;
                }
            }
            gemv_q4_kx8_q8_k_scalar_4(packed, act, n_cols, n_row_groups, out);
        }
        8 => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        avx2::gemv_q4_kx8_q8_k_avx2(packed, act, n_cols, n_row_groups, out);
                    }
                    return;
                }
            }
            gemv_q4_kx8_q8_k_scalar_8(packed, act, n_cols, n_row_groups, out);
        }
        _ => panic!("q4_kx8 interleave must be 4 or 8, got {interleave}"),
    }
}

/// One row-group (8 outputs) starting at `group` within a packed matrix.
#[inline]
pub fn gemv_q4_kx8_group(
    packed: &[u8],
    group: usize,
    act: &Q8KActivations,
    n_cols: usize,
    interleave: usize,
    out8: &mut [f32],
) {
    debug_assert_eq!(out8.len(), Q4_KX8_NROWS);
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    let off = group * nb * Q4_KX8_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q4_KX8_BLOCK_BYTES];
    gemv_q4_kx8_q8_k(slice, act, n_cols, 1, interleave, out8);
}

// ---------------------------------------------------------------------------
// Q8_0 ×4 interleaved GEMV (llama.cpp `block_q8_0x4` / `ggml_gemv_q8_0_4x4`)
// ---------------------------------------------------------------------------

/// Bytes per interleaved `block_q8_0x4` (4 × f16 d + 128 qs).
pub const Q8_0X4_BLOCK_BYTES: usize = 136;
/// Number of Q8_0 rows packed into one interleaved block.
pub const Q8_0X4_NROWS: usize = 4;
/// qs interleave width for `ggml_gemv_q8_0_4x4_q8_0` (NEON SDOT).
pub const Q8_0X4_INTERLEAVE: usize = 4;

/// Pack four canonical Q8_0 blocks (same column-block) into one
/// `block_q8_0x4`. `interleave` is 4 (ARM 4x4) or 8 (4x8).
pub fn make_block_q8_0x4(
    rows: [&[u8]; Q8_0X4_NROWS],
    interleave: usize,
) -> [u8; Q8_0X4_BLOCK_BYTES] {
    debug_assert!(interleave == 4 || interleave == 8);
    for r in &rows {
        debug_assert_eq!(r.len(), Q8_0_BLOCK_BYTES);
    }
    let mut out = [0u8; Q8_0X4_BLOCK_BYTES];
    for (i, row) in rows.iter().enumerate() {
        out[i * 2] = row[0];
        out[i * 2 + 1] = row[1];
    }
    let end = (Q8_0_BLOCK_ELEMS * Q8_0X4_NROWS) / interleave;
    let qs_out = &mut out[8..];
    for i in 0..end {
        let src_id = i % Q8_0X4_NROWS;
        let src_offset = (i / Q8_0X4_NROWS) * interleave;
        let dst_offset = i * interleave;
        let src_qs = &rows[src_id][2..34];
        qs_out[dst_offset..dst_offset + interleave]
            .copy_from_slice(&src_qs[src_offset..src_offset + interleave]);
    }
    out
}

/// Repack a Q8_0 matrix into interleaved `block_q8_0x4` groups. Tail rows
/// (not divisible by 4) are omitted; caller dots them with [`crate::dot_q8_0_q8`].
pub fn pack_q8_0_matrix_x4(data: &[u8], rows: usize, cols: usize, interleave: usize) -> Vec<u8> {
    assert!(cols.is_multiple_of(Q8_0_BLOCK_ELEMS));
    let n_blocks = cols / Q8_0_BLOCK_ELEMS;
    let row_bytes = n_blocks * Q8_0_BLOCK_BYTES;
    assert_eq!(data.len(), rows * row_bytes);
    let n_groups = rows / Q8_0X4_NROWS;
    let mut out = Vec::with_capacity(n_groups * n_blocks * Q8_0X4_BLOCK_BYTES);
    for g in 0..n_groups {
        for b in 0..n_blocks {
            let mut row_refs: [&[u8]; Q8_0X4_NROWS] = [&[]; Q8_0X4_NROWS];
            for (r, slot) in row_refs.iter_mut().enumerate() {
                let base = (g * Q8_0X4_NROWS + r) * row_bytes + b * Q8_0_BLOCK_BYTES;
                *slot = &data[base..base + Q8_0_BLOCK_BYTES];
            }
            out.extend_from_slice(&make_block_q8_0x4(row_refs, interleave));
        }
    }
    out
}

/// Scalar GEMV for interleave=4 (`ggml_gemv_q8_0_4x4_q8_0_generic`).
fn gemv_q8_0x4_q8_0_scalar(
    packed: &[u8],
    act: &Q8Activations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let blocklen = Q8_0X4_INTERLEAVE;
    let ncols = Q8_0X4_NROWS;
    debug_assert_eq!(act.n_blocks(), nb);
    debug_assert_eq!(out.len(), n_row_groups * ncols);
    debug_assert_eq!(packed.len(), n_row_groups * nb * Q8_0X4_BLOCK_BYTES);

    for x in 0..n_row_groups {
        let mut sumf = [0f32; 4];
        let group_off = x * nb * Q8_0X4_BLOCK_BYTES;
        for l in 0..nb {
            let blk = &packed[group_off + l * Q8_0X4_BLOCK_BYTES..][..Q8_0X4_BLOCK_BYTES];
            let qs = &blk[8..];
            let da = act.d[l];
            let q8 = &act.q[l * Q8_0_BLOCK_ELEMS..(l + 1) * Q8_0_BLOCK_ELEMS];
            for k in 0..(Q8_0_BLOCK_ELEMS / blocklen) {
                for j in 0..ncols {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let v0 = qs[k * ncols * blocklen + j * blocklen + i] as i8 as i32;
                        sumi += v0 * q8[k * blocklen + i] as i32;
                    }
                    sumf[j] += sumi as f32 * f16_from_bytes(&blk[j * 2..]) * da;
                }
            }
        }
        let base = x * ncols;
        out[base..base + ncols].copy_from_slice(&sumf);
    }
}

/// GEMV: interleaved Q8_0 weights × Q8 activation → `n_row_groups * 4` f32s.
pub fn gemv_q8_0x4_q8_0(
    packed: &[u8],
    act: &Q8Activations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    assert!(n_cols.is_multiple_of(Q8_0_BLOCK_ELEMS));
    assert_eq!(out.len(), n_row_groups * Q8_0X4_NROWS);
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            unsafe {
                neon::gemv_q8_0x4_q8_0_neon_sdot(packed, act, n_cols, n_row_groups, out);
            }
            return;
        }
    }
    gemv_q8_0x4_q8_0_scalar(packed, act, n_cols, n_row_groups, out);
}

/// How many activations one [`gemm_q8_0x4_group`] pass keeps in flight.
/// Four f32x4 accumulators plus the eight loaded weight vectors fit
/// comfortably in NEON's register file, so each weight load is amortized
/// over four activations instead of being repeated per activation.
pub const Q8_0X4_GEMM_NC: usize = 4;

/// GEMM counterpart of [`gemv_q8_0x4_group`]: one row-group (4 rows)
/// against `acts.len()` activations at once.
///
/// The difference that matters is register blocking over the *batch*
/// dimension. Calling the GEMV once per activation reloads the group's
/// eight `int8x16` weight vectors for every activation; this loads them
/// once per `Q8_0X4_GEMM_NC` activations and issues the dot products
/// back to back. That is the same reason llama.cpp ships
/// `ggml_gemm_q8_0_4x4_q8_0` next to `ggml_gemv_q8_0_4x4_q8_0` rather
/// than looping the GEMV.
///
/// `out` is `[row][act]`: `out[r * acts.len() + j]`, which is the layout
/// `WeightMatrix::apply_batch` accumulates into.
pub fn gemm_q8_0x4_group(
    packed: &[u8],
    group: usize,
    acts: &[Q8Activations],
    n_cols: usize,
    out: &mut [f32],
) {
    assert_eq!(out.len(), Q8_0X4_NROWS * acts.len());
    assert!(n_cols.is_multiple_of(Q8_0_BLOCK_ELEMS));
    if acts.is_empty() {
        return;
    }
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let off = group * nb * Q8_0X4_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q8_0X4_BLOCK_BYTES];

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            unsafe {
                neon::gemm_q8_0x4_q8_0_neon_sdot(slice, acts, n_cols, out);
            }
            return;
        }
    }
    // Portable fallback: the GEMV, once per activation. Same results,
    // none of the reuse.
    let mut tmp = [0f32; Q8_0X4_NROWS];
    for (j, act) in acts.iter().enumerate() {
        gemv_q8_0x4_q8_0(slice, act, n_cols, 1, &mut tmp);
        for (r, v) in tmp.iter().enumerate() {
            out[r * acts.len() + j] = *v;
        }
    }
}

/// How many activations one [`gemm_q4_kx8_group`] pass keeps in flight.
///
/// Four is llama's shape for `ggml_gemm_q4_K_8x4_q8_K` (`q8_k_blocklen`),
/// and it is what the register file allows here: eight `uint8x16` weight
/// columns plus one activation's four `int8x16` and its accumulator pair
/// stay resident while the batch loop turns.
pub const Q4_KX8_GEMM_NC: usize = 4;

/// GEMM counterpart of [`gemv_q4_kx8_group`]: one row-group (8 rows)
/// against `acts.len()` activations at once.
///
/// Q4_K was the expensive omission. The GEMV repeats, *per activation*,
/// work that depends only on the weights: 16 f16 scale conversions, 8
/// `decode_scales_mins` calls and 16 `q4_cols` loads per 256-element
/// super-block. At batch 512 that is the same 6-bit scale decode run 512
/// times. Q8_0 already had `gemm_q8_0x4_group`; this is the same idea for
/// the format that carries every `*_Q4_K_M` checkpoint's FFN.
///
/// `out` is `[row][act]`: `out[r * acts.len() + j]`, matching
/// [`gemm_q8_0x4_group`] and what `WeightMatrix::apply_batch` writes.
pub fn gemm_q4_kx8_group(
    packed: &[u8],
    group: usize,
    acts: &[Q8KActivations],
    n_cols: usize,
    interleave: usize,
    out: &mut [f32],
) {
    assert_eq!(out.len(), Q4_KX8_NROWS * acts.len());
    assert!(n_cols.is_multiple_of(Q4_K_BLOCK_ELEMS));
    if acts.is_empty() {
        return;
    }
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    let off = group * nb * Q4_KX8_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q4_KX8_BLOCK_BYTES];

    #[cfg(target_arch = "aarch64")]
    {
        // The NEON kernel is written against the interleave-4 layout the
        // GEMV uses; any other packing falls through to the GEMV loop.
        if interleave == 4 && acts.len() <= Q4_KX8_GEMM_NC {
            if std::arch::is_aarch64_feature_detected!("i8mm") {
                unsafe {
                    neon::gemm_q4_kx8_q8_k_neon_i8mm(slice, acts, n_cols, out);
                }
                return;
            }
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                unsafe {
                    neon::gemm_q4_kx8_q8_k_neon_sdot(slice, acts, n_cols, out);
                }
                return;
            }
        }
    }
    // Portable fallback: the GEMV, once per activation. Same results,
    // none of the reuse.
    let mut tmp = [0f32; Q4_KX8_NROWS];
    for (j, act) in acts.iter().enumerate() {
        gemv_q4_kx8_q8_k(slice, act, n_cols, 1, interleave, &mut tmp);
        for (r, v) in tmp.iter().enumerate() {
            out[r * acts.len() + j] = *v;
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_0 ×4 interleaved GEMV/GEMM (llama.cpp `block_q4_0x4` / `ggml_gemv_q4_0_4x4`)
// ---------------------------------------------------------------------------

/// Bytes per interleaved `block_q4_0x4` (4 × f16 d + 64 qs).
pub const Q4_0X4_BLOCK_BYTES: usize = 72;
/// Number of Q4_0 rows packed into one interleaved block.
pub const Q4_0X4_NROWS: usize = 4;
/// qs interleave width for `ggml_gemv_q4_0_4x4_q8_0` (NEON SDOT).
pub const Q4_0X4_INTERLEAVE: usize = 4;

const Q4_0X4_XOR_MASK_U32: u32 = 0x8888_8888;
const Q4_0X4_XOR_MASK_U64: u64 = 0x8888_8888_8888_8888;

/// Pack four canonical Q4_0 blocks (same column-block) into one
/// `block_q4_0x4`. Nibble bytes are XOR-masked during interleave so
/// NEON can unpack without explicit `- 8` bias subtraction.
pub fn make_block_q4_0x4(
    rows: [&[u8]; Q4_0X4_NROWS],
    interleave: usize,
) -> [u8; Q4_0X4_BLOCK_BYTES] {
    debug_assert!(interleave == 4 || interleave == 8);
    for r in &rows {
        debug_assert_eq!(r.len(), Q4_0_BLOCK_BYTES);
    }
    let mut out = [0u8; Q4_0X4_BLOCK_BYTES];
    for (i, row) in rows.iter().enumerate() {
        out[i * 2] = row[0];
        out[i * 2 + 1] = row[1];
    }
    let end = (Q4_0_BLOCK_ELEMS * 2) / interleave;
    let qs_out = &mut out[8..];
    for i in 0..end {
        let src_id = i % Q4_0X4_NROWS;
        let src_offset = (i / Q4_0X4_NROWS) * interleave;
        let dst_offset = i * interleave;
        let src_qs = &rows[src_id][2..18];
        if interleave == 4 {
            let mut elems = u32::from_le_bytes(
                src_qs[src_offset..src_offset + 4]
                    .try_into()
                    .expect("4-byte interleave chunk"),
            );
            elems ^= Q4_0X4_XOR_MASK_U32;
            qs_out[dst_offset..dst_offset + 4].copy_from_slice(&elems.to_le_bytes());
        } else {
            let mut elems = u64::from_le_bytes(
                src_qs[src_offset..src_offset + 8]
                    .try_into()
                    .expect("8-byte interleave chunk"),
            );
            elems ^= Q4_0X4_XOR_MASK_U64;
            qs_out[dst_offset..dst_offset + 8].copy_from_slice(&elems.to_le_bytes());
        }
    }
    out
}

/// Repack a Q4_0 matrix into interleaved `block_q4_0x4` groups. Tail rows
/// (not divisible by 4) are omitted; caller dots them with [`crate::dot_q4_0_q8`].
pub fn pack_q4_0_matrix_x4(data: &[u8], rows: usize, cols: usize, interleave: usize) -> Vec<u8> {
    assert!(cols.is_multiple_of(Q4_0_BLOCK_ELEMS));
    let n_blocks = cols / Q4_0_BLOCK_ELEMS;
    let row_bytes = n_blocks * Q4_0_BLOCK_BYTES;
    assert_eq!(data.len(), rows * row_bytes);
    let n_groups = rows / Q4_0X4_NROWS;
    let mut out = Vec::with_capacity(n_groups * n_blocks * Q4_0X4_BLOCK_BYTES);
    for g in 0..n_groups {
        for b in 0..n_blocks {
            let mut row_refs: [&[u8]; Q4_0X4_NROWS] = [&[]; Q4_0X4_NROWS];
            for (r, slot) in row_refs.iter_mut().enumerate() {
                let base = (g * Q4_0X4_NROWS + r) * row_bytes + b * Q4_0_BLOCK_BYTES;
                *slot = &data[base..base + Q4_0_BLOCK_BYTES];
            }
            out.extend_from_slice(&make_block_q4_0x4(row_refs, interleave));
        }
    }
    out
}

#[inline]
fn q4_0x4_nibble_dot(byte: u8, q8_lo: i32, q8_hi: i32) -> i32 {
    let v0 = ((byte << 4) as i8) as i32;
    let v1 = ((byte & 0xF0) as i8) as i32;
    ((v0 * q8_lo) + (v1 * q8_hi)) >> 4
}

/// Scalar GEMV for interleave=4 (`ggml_gemv_q4_0_4x4_q8_0_generic`).
fn gemv_q4_0x4_q8_0_scalar(
    packed: &[u8],
    act: &Q8Activations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_0_BLOCK_ELEMS;
    let blocklen = Q4_0X4_INTERLEAVE;
    let ncols = Q4_0X4_NROWS;
    debug_assert_eq!(act.n_blocks(), nb);
    debug_assert_eq!(out.len(), n_row_groups * ncols);
    debug_assert_eq!(packed.len(), n_row_groups * nb * Q4_0X4_BLOCK_BYTES);

    for x in 0..n_row_groups {
        let mut sumf = [0f32; 4];
        let group_off = x * nb * Q4_0X4_BLOCK_BYTES;
        for l in 0..nb {
            let blk = &packed[group_off + l * Q4_0X4_BLOCK_BYTES..][..Q4_0X4_BLOCK_BYTES];
            let da = act.d[l];
            let q8 = &act.q[l * Q4_0_BLOCK_ELEMS..(l + 1) * Q4_0_BLOCK_ELEMS];
            for k in 0..(Q4_0_BLOCK_ELEMS / (2 * blocklen)) {
                for j in 0..ncols {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let byte = blk[8 + k * ncols * blocklen + j * blocklen + i];
                        sumi += q4_0x4_nibble_dot(
                            byte,
                            q8[k * blocklen + i] as i32,
                            q8[k * blocklen + i + Q4_0_BLOCK_ELEMS / 2] as i32,
                        );
                    }
                    sumf[j] += sumi as f32 * f16_from_bytes(&blk[j * 2..]) * da;
                }
            }
        }
        let base = x * ncols;
        out[base..base + ncols].copy_from_slice(&sumf);
    }
}

/// GEMV: interleaved Q4_0 weights × Q8 activation → `n_row_groups * 4` f32s.
pub fn gemv_q4_0x4_q8_0(
    packed: &[u8],
    act: &Q8Activations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    assert!(n_cols.is_multiple_of(Q4_0_BLOCK_ELEMS));
    assert_eq!(out.len(), n_row_groups * Q4_0X4_NROWS);
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            unsafe {
                neon::gemv_q4_0x4_q8_0_neon_sdot(packed, act, n_cols, n_row_groups, out);
            }
            return;
        }
    }
    gemv_q4_0x4_q8_0_scalar(packed, act, n_cols, n_row_groups, out);
}

/// How many activations one [`gemm_q4_0x4_group`] pass keeps in flight.
pub const Q4_0X4_GEMM_NC: usize = 4;

/// GEMM counterpart of [`gemv_q4_0x4_group`]: one row-group (4 rows)
/// against `acts.len()` activations at once. `out` is `[row][act]`:
/// `out[r * acts.len() + j]`.
pub fn gemm_q4_0x4_group(
    packed: &[u8],
    group: usize,
    acts: &[Q8Activations],
    n_cols: usize,
    out: &mut [f32],
) {
    assert_eq!(out.len(), Q4_0X4_NROWS * acts.len());
    assert!(n_cols.is_multiple_of(Q4_0_BLOCK_ELEMS));
    if acts.is_empty() {
        return;
    }
    let nb = n_cols / Q4_0_BLOCK_ELEMS;
    let off = group * nb * Q4_0X4_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q4_0X4_BLOCK_BYTES];

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            unsafe {
                neon::gemm_q4_0x4_q8_0_neon_sdot(slice, acts, n_cols, out);
            }
            return;
        }
    }
    let mut tmp = [0f32; Q4_0X4_NROWS];
    for (j, act) in acts.iter().enumerate() {
        gemv_q4_0x4_q8_0(slice, act, n_cols, 1, &mut tmp);
        for (r, v) in tmp.iter().enumerate() {
            out[r * acts.len() + j] = *v;
        }
    }
}

/// One row-group (4 outputs) starting at `group` within a packed Q4_0x4 matrix.
#[inline]
pub fn gemv_q4_0x4_group(
    packed: &[u8],
    group: usize,
    act: &Q8Activations,
    n_cols: usize,
    out4: &mut [f32],
) {
    debug_assert_eq!(out4.len(), Q4_0X4_NROWS);
    let nb = n_cols / Q4_0_BLOCK_ELEMS;
    let off = group * nb * Q4_0X4_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q4_0X4_BLOCK_BYTES];
    gemv_q4_0x4_q8_0(slice, act, n_cols, 1, out4);
}

/// One row-group (4 outputs) starting at `group` within a packed Q8_0x4 matrix.
#[inline]
pub fn gemv_q8_0x4_group(
    packed: &[u8],
    group: usize,
    act: &Q8Activations,
    n_cols: usize,
    out4: &mut [f32],
) {
    debug_assert_eq!(out4.len(), Q8_0X4_NROWS);
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let off = group * nb * Q8_0X4_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q8_0X4_BLOCK_BYTES];
    gemv_q8_0x4_q8_0(slice, act, n_cols, 1, out4);
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon,dotprod")]
    unsafe fn sdot_lane(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t, lane: u32) -> int32x4_t {
        // sdot Vd.4S, Vn.16B, Vm.4B[lane]
        match lane {
            0 => std::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[0]",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack),
            ),
            1 => std::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[1]",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack),
            ),
            2 => std::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[2]",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack),
            ),
            3 => std::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[3]",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack),
            ),
            _ => unreachable!(),
        }
        acc
    }

    /// NEON DotProd GEMV for interleave-4 packed weights (Apple Silicon path).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemv_q4_kx8_q8_k_neon_sdot(
        packed: &[u8],
        act: &Q8KActivations,
        n_cols: usize,
        n_row_groups: usize,
        out: &mut [f32],
    ) {
        let nb = n_cols / Q4_K_BLOCK_ELEMS;
        let m4b = vdupq_n_u8(0x0f);

        for x in 0..n_row_groups {
            let mut acc_f32 = [vdupq_n_f32(0.0), vdupq_n_f32(0.0)];
            let group_off = x * nb * Q4_KX8_BLOCK_BYTES;

            for b in 0..nb {
                let blk = packed.as_ptr().add(group_off + b * Q4_KX8_BLOCK_BYTES);
                let mut d_arr = [0f32; 8];
                let mut dmin_arr = [0f32; 8];
                for j in 0..8 {
                    d_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
                    dmin_arr[j] =
                        f16_from_bytes(std::slice::from_raw_parts(blk.add(16 + j * 2), 2));
                }
                let q8_d = act.d[b];
                let sb_scale_0123 = vmulq_n_f32(vld1q_f32(d_arr.as_ptr()), q8_d);
                let sb_scale_4567 = vmulq_n_f32(vld1q_f32(d_arr.as_ptr().add(4)), q8_d);
                let sb_min_0123 = vmulq_n_f32(vld1q_f32(dmin_arr.as_ptr()), q8_d);
                let sb_min_4567 = vmulq_n_f32(vld1q_f32(dmin_arr.as_ptr().add(4)), q8_d);

                let mut bias_acc = [vdupq_n_s32(0), vdupq_n_s32(0)];
                let q8_base = act.q.as_ptr().add(b * Q4_K_BLOCK_ELEMS);
                let bsums_ptr = act.bsums.as_ptr().add(b * 16);
                // Pairwise-add 16 bsums → 8 (matching llama vpaddq_s16).
                let mut bsums_arr = [0i16; 8];
                for (i, slot) in bsums_arr.iter_mut().enumerate() {
                    *slot = *bsums_ptr.add(2 * i) + *bsums_ptr.add(2 * i + 1);
                }

                let scales_base = blk.add(32);
                let qs_base = blk.add(128);

                for sb in 0..4 {
                    let mut acc_lo = [vdupq_n_s32(0), vdupq_n_s32(0)];
                    let mut acc_hi = [vdupq_n_s32(0), vdupq_n_s32(0)];

                    let mut q4sb_mins = [vdupq_n_s16(0); 2];
                    let mut q4sb_scales = [vdupq_n_s16(0); 2];
                    for i in 0..2 {
                        let mut sc = [0u8; 8];
                        let mut mn = [0u8; 8];
                        let offset = sb * 24 + i * 12;
                        decode_scales_mins(
                            std::slice::from_raw_parts(scales_base.add(offset), 12),
                            &mut sc,
                            &mut mn,
                        );
                        let mut sc_i8 = [0i8; 8];
                        let mut mn_i8 = [0i8; 8];
                        for t in 0..8 {
                            sc_i8[t] = sc[t] as i8;
                            mn_i8[t] = mn[t] as i8;
                        }
                        q4sb_scales[i] = vmovl_s8(vld1_s8(sc_i8.as_ptr()));
                        q4sb_mins[i] = vmovl_s8(vld1_s8(mn_i8.as_ptr()));
                    }

                    let mut q8_qs = [vdupq_n_s8(0); 4];
                    for (i, slot) in q8_qs.iter_mut().enumerate() {
                        *slot = vld1q_s8(q8_base.add(sb * 64 + i * 16));
                    }

                    for c in 0..2 {
                        let mut q4_cols = [vdupq_n_u8(0); 8];
                        for (i, slot) in q4_cols.iter_mut().enumerate() {
                            *slot = vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + i * 32 + 16 * c));
                        }

                        acc_lo[c] = sdot_lane(
                            acc_lo[c],
                            vreinterpretq_s8_u8(vandq_u8(q4_cols[0], m4b)),
                            q8_qs[0],
                            0,
                        );
                        acc_lo[c] = sdot_lane(
                            acc_lo[c],
                            vreinterpretq_s8_u8(vandq_u8(q4_cols[1], m4b)),
                            q8_qs[0],
                            1,
                        );
                        acc_lo[c] = sdot_lane(
                            acc_lo[c],
                            vreinterpretq_s8_u8(vandq_u8(q4_cols[2], m4b)),
                            q8_qs[0],
                            2,
                        );
                        acc_lo[c] = sdot_lane(
                            acc_lo[c],
                            vreinterpretq_s8_u8(vandq_u8(q4_cols[3], m4b)),
                            q8_qs[0],
                            3,
                        );
                        acc_lo[c] = sdot_lane(
                            acc_lo[c],
                            vreinterpretq_s8_u8(vandq_u8(q4_cols[4], m4b)),
                            q8_qs[1],
                            0,
                        );
                        acc_lo[c] = sdot_lane(
                            acc_lo[c],
                            vreinterpretq_s8_u8(vandq_u8(q4_cols[5], m4b)),
                            q8_qs[1],
                            1,
                        );
                        acc_lo[c] = sdot_lane(
                            acc_lo[c],
                            vreinterpretq_s8_u8(vandq_u8(q4_cols[6], m4b)),
                            q8_qs[1],
                            2,
                        );
                        acc_lo[c] = sdot_lane(
                            acc_lo[c],
                            vreinterpretq_s8_u8(vandq_u8(q4_cols[7], m4b)),
                            q8_qs[1],
                            3,
                        );

                        acc_hi[c] = sdot_lane(
                            acc_hi[c],
                            vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[0], 4)),
                            q8_qs[2],
                            0,
                        );
                        acc_hi[c] = sdot_lane(
                            acc_hi[c],
                            vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[1], 4)),
                            q8_qs[2],
                            1,
                        );
                        acc_hi[c] = sdot_lane(
                            acc_hi[c],
                            vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[2], 4)),
                            q8_qs[2],
                            2,
                        );
                        acc_hi[c] = sdot_lane(
                            acc_hi[c],
                            vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[3], 4)),
                            q8_qs[2],
                            3,
                        );
                        acc_hi[c] = sdot_lane(
                            acc_hi[c],
                            vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[4], 4)),
                            q8_qs[3],
                            0,
                        );
                        acc_hi[c] = sdot_lane(
                            acc_hi[c],
                            vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[5], 4)),
                            q8_qs[3],
                            1,
                        );
                        acc_hi[c] = sdot_lane(
                            acc_hi[c],
                            vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[6], 4)),
                            q8_qs[3],
                            2,
                        );
                        acc_hi[c] = sdot_lane(
                            acc_hi[c],
                            vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[7], 4)),
                            q8_qs[3],
                            3,
                        );
                    }

                    let sc_0123_lo = vget_low_s16(q4sb_scales[0]);
                    let sc_0123_hi = vget_low_s16(q4sb_scales[1]);
                    let sumf_0123 = vcvtq_f32_s32(vaddq_s32(
                        vmulq_s32(vmovl_s16(sc_0123_lo), acc_lo[0]),
                        vmulq_s32(vmovl_s16(sc_0123_hi), acc_hi[0]),
                    ));
                    acc_f32[0] = vfmaq_f32(acc_f32[0], sb_scale_0123, sumf_0123);

                    let sc_4567_lo = vget_high_s16(q4sb_scales[0]);
                    let sc_4567_hi = vget_high_s16(q4sb_scales[1]);
                    let sumf_4567 = vcvtq_f32_s32(vaddq_s32(
                        vmulq_s32(vmovl_s16(sc_4567_lo), acc_lo[1]),
                        vmulq_s32(vmovl_s16(sc_4567_hi), acc_hi[1]),
                    ));
                    acc_f32[1] = vfmaq_f32(acc_f32[1], sb_scale_4567, sumf_4567);

                    let bsums_vec_lo = vdup_n_s16(bsums_arr[2 * sb]);
                    let bsums_vec_hi = vdup_n_s16(bsums_arr[2 * sb + 1]);
                    bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_lo, vget_low_s16(q4sb_mins[0]));
                    bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_hi, vget_low_s16(q4sb_mins[1]));
                    bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_lo, vget_high_s16(q4sb_mins[0]));
                    bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_hi, vget_high_s16(q4sb_mins[1]));
                }

                acc_f32[0] = vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sb_min_0123);
                acc_f32[1] = vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sb_min_4567);
            }

            let base = x * Q4_KX8_NROWS;
            vst1q_f32(out.as_mut_ptr().add(base), acc_f32[0]);
            vst1q_f32(out.as_mut_ptr().add(base + 4), acc_f32[1]);
        }
    }

    /// NEON DotProd **GEMM** for interleave-4 packed Q4_K weights: one
    /// row-group (8 rows) against up to [`Q4_KX8_GEMM_NC`] activations.
    ///
    /// Same arithmetic as [`gemv_q4_kx8_q8_k_neon_sdot`], reordered so
    /// the weight-side unpack happens once per activation *tile* rather
    /// than once per activation. Per 256-element super-block that hoists
    /// 16 f16 scale conversions, 8 `decode_scales_mins` calls and 16
    /// `q4_cols` loads out of the batch loop -- which is the whole point,
    /// and the same reason llama.cpp ships `ggml_gemm_q4_K_8x4_q8_K`
    /// beside its GEMV rather than looping the GEMV.
    ///
    /// `out` is `[row][act]`: `out[r * na + j]`.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemm_q4_kx8_q8_k_neon_sdot(
        packed: &[u8],
        acts: &[Q8KActivations],
        n_cols: usize,
        out: &mut [f32],
    ) {
        let na = acts.len();
        debug_assert!(na <= Q4_KX8_GEMM_NC);
        let nb = n_cols / Q4_K_BLOCK_ELEMS;
        let m4b = vdupq_n_u8(0x0f);

        // [act][row-half]; row-half 0 is rows 0..3, 1 is rows 4..7.
        let mut acc_f32 = [[vdupq_n_f32(0.0); 2]; Q4_KX8_GEMM_NC];
        let mut bias_acc = [[vdupq_n_s32(0); 2]; Q4_KX8_GEMM_NC];

        for b in 0..nb {
            let blk = packed.as_ptr().add(b * Q4_KX8_BLOCK_BYTES);

            // --- weight-side, once per block (was once per activation) ---
            let mut d_arr = [0f32; 8];
            let mut dmin_arr = [0f32; 8];
            for j in 0..8 {
                d_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
                dmin_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(16 + j * 2), 2));
            }
            let d_lo = vld1q_f32(d_arr.as_ptr());
            let d_hi = vld1q_f32(d_arr.as_ptr().add(4));
            let dmin_lo = vld1q_f32(dmin_arr.as_ptr());
            let dmin_hi = vld1q_f32(dmin_arr.as_ptr().add(4));

            // Per-activation scaling of those, plus the pairwise-added
            // bsums this block needs (llama's vpaddq_s16).
            let mut sb_scale = [[vdupq_n_f32(0.0); 2]; Q4_KX8_GEMM_NC];
            let mut sb_min = [[vdupq_n_f32(0.0); 2]; Q4_KX8_GEMM_NC];
            let mut bsums_arr = [[0i16; 8]; Q4_KX8_GEMM_NC];
            for (a, act) in acts.iter().enumerate() {
                let q8_d = act.d[b];
                sb_scale[a] = [vmulq_n_f32(d_lo, q8_d), vmulq_n_f32(d_hi, q8_d)];
                sb_min[a] = [vmulq_n_f32(dmin_lo, q8_d), vmulq_n_f32(dmin_hi, q8_d)];
                let bsums_ptr = act.bsums.as_ptr().add(b * 16);
                for (i, slot) in bsums_arr[a].iter_mut().enumerate() {
                    *slot = *bsums_ptr.add(2 * i) + *bsums_ptr.add(2 * i + 1);
                }
            }

            let scales_base = blk.add(32);
            let qs_base = blk.add(128);

            for sb in 0..4 {
                // 6-bit scale/min decode: once per block-quarter, not
                // once per (block-quarter, activation).
                let mut q4sb_mins = [vdupq_n_s16(0); 2];
                let mut q4sb_scales = [vdupq_n_s16(0); 2];
                for i in 0..2 {
                    let mut sc = [0u8; 8];
                    let mut mn = [0u8; 8];
                    let offset = sb * 24 + i * 12;
                    decode_scales_mins(
                        std::slice::from_raw_parts(scales_base.add(offset), 12),
                        &mut sc,
                        &mut mn,
                    );
                    let mut sc_i8 = [0i8; 8];
                    let mut mn_i8 = [0i8; 8];
                    for t in 0..8 {
                        sc_i8[t] = sc[t] as i8;
                        mn_i8[t] = mn[t] as i8;
                    }
                    q4sb_scales[i] = vmovl_s8(vld1_s8(sc_i8.as_ptr()));
                    q4sb_mins[i] = vmovl_s8(vld1_s8(mn_i8.as_ptr()));
                }

                // `c` selects the row half, so each pass owns one output
                // quad and the accumulators can be consumed immediately
                // instead of all eight staying live.
                for c in 0..2 {
                    let mut q4_cols = [vdupq_n_u8(0); 8];
                    for (i, slot) in q4_cols.iter_mut().enumerate() {
                        *slot = vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + i * 32 + 16 * c));
                    }
                    let (sc_lo, sc_hi) = if c == 0 {
                        (vget_low_s16(q4sb_scales[0]), vget_low_s16(q4sb_scales[1]))
                    } else {
                        (vget_high_s16(q4sb_scales[0]), vget_high_s16(q4sb_scales[1]))
                    };

                    // Mask once per weight tile, not once per
                    // activation, and keep the lane indices literal --
                    // a runtime lane forces a real call per `sdot`
                    // instead of the single instruction it should be.
                    let lo0 = vreinterpretq_s8_u8(vandq_u8(q4_cols[0], m4b));
                    let lo1 = vreinterpretq_s8_u8(vandq_u8(q4_cols[1], m4b));
                    let lo2 = vreinterpretq_s8_u8(vandq_u8(q4_cols[2], m4b));
                    let lo3 = vreinterpretq_s8_u8(vandq_u8(q4_cols[3], m4b));
                    let lo4 = vreinterpretq_s8_u8(vandq_u8(q4_cols[4], m4b));
                    let lo5 = vreinterpretq_s8_u8(vandq_u8(q4_cols[5], m4b));
                    let lo6 = vreinterpretq_s8_u8(vandq_u8(q4_cols[6], m4b));
                    let lo7 = vreinterpretq_s8_u8(vandq_u8(q4_cols[7], m4b));
                    let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[0], 4));
                    let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[1], 4));
                    let hi2 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[2], 4));
                    let hi3 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[3], 4));
                    let hi4 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[4], 4));
                    let hi5 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[5], 4));
                    let hi6 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[6], 4));
                    let hi7 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[7], 4));
                    let sc_lo_w = vmovl_s16(sc_lo);
                    let sc_hi_w = vmovl_s16(sc_hi);

                    for a in 0..na {
                        let q8_base = acts[a].q.as_ptr().add(b * Q4_K_BLOCK_ELEMS);
                        let y0 = vld1q_s8(q8_base.add(sb * 64));
                        let y1 = vld1q_s8(q8_base.add(sb * 64 + 16));
                        let y2 = vld1q_s8(q8_base.add(sb * 64 + 32));
                        let y3 = vld1q_s8(q8_base.add(sb * 64 + 48));
                        let mut acc_lo = vdupq_n_s32(0);
                        let mut acc_hi = vdupq_n_s32(0);
                        acc_lo = sdot_lane(acc_lo, lo0, y0, 0);
                        acc_lo = sdot_lane(acc_lo, lo1, y0, 1);
                        acc_lo = sdot_lane(acc_lo, lo2, y0, 2);
                        acc_lo = sdot_lane(acc_lo, lo3, y0, 3);
                        acc_lo = sdot_lane(acc_lo, lo4, y1, 0);
                        acc_lo = sdot_lane(acc_lo, lo5, y1, 1);
                        acc_lo = sdot_lane(acc_lo, lo6, y1, 2);
                        acc_lo = sdot_lane(acc_lo, lo7, y1, 3);
                        acc_hi = sdot_lane(acc_hi, hi0, y2, 0);
                        acc_hi = sdot_lane(acc_hi, hi1, y2, 1);
                        acc_hi = sdot_lane(acc_hi, hi2, y2, 2);
                        acc_hi = sdot_lane(acc_hi, hi3, y2, 3);
                        acc_hi = sdot_lane(acc_hi, hi4, y3, 0);
                        acc_hi = sdot_lane(acc_hi, hi5, y3, 1);
                        acc_hi = sdot_lane(acc_hi, hi6, y3, 2);
                        acc_hi = sdot_lane(acc_hi, hi7, y3, 3);
                        let sumf = vcvtq_f32_s32(vaddq_s32(
                            vmulq_s32(sc_lo_w, acc_lo),
                            vmulq_s32(sc_hi_w, acc_hi),
                        ));
                        acc_f32[a][c] = vfmaq_f32(acc_f32[a][c], sb_scale[a][c], sumf);
                    }
                }

                for a in 0..na {
                    let bs_lo = vdup_n_s16(bsums_arr[a][2 * sb]);
                    let bs_hi = vdup_n_s16(bsums_arr[a][2 * sb + 1]);
                    bias_acc[a][0] = vmlal_s16(bias_acc[a][0], bs_lo, vget_low_s16(q4sb_mins[0]));
                    bias_acc[a][0] = vmlal_s16(bias_acc[a][0], bs_hi, vget_low_s16(q4sb_mins[1]));
                    bias_acc[a][1] = vmlal_s16(bias_acc[a][1], bs_lo, vget_high_s16(q4sb_mins[0]));
                    bias_acc[a][1] = vmlal_s16(bias_acc[a][1], bs_hi, vget_high_s16(q4sb_mins[1]));
                }
            }

            for a in 0..na {
                for c in 0..2 {
                    acc_f32[a][c] =
                        vmlsq_f32(acc_f32[a][c], vcvtq_f32_s32(bias_acc[a][c]), sb_min[a][c]);
                    bias_acc[a][c] = vdupq_n_s32(0);
                }
            }
        }

        for a in 0..na {
            let mut row = [0f32; Q4_KX8_NROWS];
            vst1q_f32(row.as_mut_ptr(), acc_f32[a][0]);
            vst1q_f32(row.as_mut_ptr().add(4), acc_f32[a][1]);
            for (r, v) in row.iter().enumerate() {
                out[r * na + a] = *v;
            }
        }
    }

    /// i8mm GEMM hook for interleave-4 Q4_K×8 tiles. SMMLA needs repacked
    /// weights (llama.cpp `ggml_gemm_q4_K_8x8_q8_K` / repack.cpp); until ferrox
    /// ships that layout this delegates to the dotprod GEMM with identical
    /// numerics.
    #[target_feature(enable = "neon,i8mm")]
    pub unsafe fn gemm_q4_kx8_q8_k_neon_i8mm(
        packed: &[u8],
        acts: &[Q8KActivations],
        n_cols: usize,
        out: &mut [f32],
    ) {
        gemm_q4_kx8_q8_k_neon_sdot(packed, acts, n_cols, out);
    }

    /// NEON DotProd GEMV for `block_q8_0x4` (llama `ggml_gemv_q8_0_4x4_q8_0`).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemv_q8_0x4_q8_0_neon_sdot(
        packed: &[u8],
        act: &Q8Activations,
        n_cols: usize,
        n_row_groups: usize,
        out: &mut [f32],
    ) {
        let nb = n_cols / Q8_0_BLOCK_ELEMS;
        for x in 0..n_row_groups {
            let mut acc = vdupq_n_f32(0.0);
            let group_off = x * nb * Q8_0X4_BLOCK_BYTES;
            for b in 0..nb {
                let blk = packed.as_ptr().add(group_off + b * Q8_0X4_BLOCK_BYTES);
                let qs = blk.add(8);
                // Four int8x16: first 64 qs bytes (k=0..3 × 4 rows × 4).
                let b0 = vld1q_s8(qs as *const i8);
                let b1 = vld1q_s8(qs.add(16) as *const i8);
                let b2 = vld1q_s8(qs.add(32) as *const i8);
                let b3 = vld1q_s8(qs.add(48) as *const i8);
                let b4 = vld1q_s8(qs.add(64) as *const i8);
                let b5 = vld1q_s8(qs.add(80) as *const i8);
                let b6 = vld1q_s8(qs.add(96) as *const i8);
                let b7 = vld1q_s8(qs.add(112) as *const i8);

                let a_ptr = act.q.as_ptr().add(b * Q8_0_BLOCK_ELEMS);
                let a0 = vld1q_s8(a_ptr);
                let a1 = vld1q_s8(a_ptr.add(16));

                let mut ret = vdupq_n_s32(0);
                ret = sdot_lane(ret, b0, a0, 0);
                ret = sdot_lane(ret, b1, a0, 1);
                ret = sdot_lane(ret, b2, a0, 2);
                ret = sdot_lane(ret, b3, a0, 3);
                ret = sdot_lane(ret, b4, a1, 0);
                ret = sdot_lane(ret, b5, a1, 1);
                ret = sdot_lane(ret, b6, a1, 2);
                ret = sdot_lane(ret, b7, a1, 3);

                // Four f16 weight scales at blk[0..8] — load as u16 then
                // convert (avoids 4× scalar half::f16 path per block).
                let d_bits = vld1_u16(blk as *const u16);
                let mut dw = [0f32; 4];
                dw[0] = f16::from_bits(vget_lane_u16(d_bits, 0)).to_f32();
                dw[1] = f16::from_bits(vget_lane_u16(d_bits, 1)).to_f32();
                dw[2] = f16::from_bits(vget_lane_u16(d_bits, 2)).to_f32();
                dw[3] = f16::from_bits(vget_lane_u16(d_bits, 3)).to_f32();
                let scale = vmulq_n_f32(vld1q_f32(dw.as_ptr()), act.d[b]);
                acc = vfmaq_f32(acc, vcvtq_f32_s32(ret), scale);
            }
            vst1q_f32(out.as_mut_ptr().add(x * Q8_0X4_NROWS), acc);
        }
    }

    /// NEON DotProd GEMM for one `block_q8_0x4` row-group against
    /// several activations (llama `ggml_gemm_q8_0_4x4_q8_0` in shape).
    ///
    /// The eight weight vectors of a block are loaded once and reused
    /// across a tile of [`Q8_0X4_GEMM_NC`] activations, which is the
    /// whole point of having a GEMM rather than a loop over the GEMV.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemm_q8_0x4_q8_0_neon_sdot(
        group: &[u8],
        acts: &[Q8Activations],
        n_cols: usize,
        out: &mut [f32],
    ) {
        let nb = n_cols / Q8_0_BLOCK_ELEMS;
        let n_acts = acts.len();
        let mut j0 = 0;
        while j0 < n_acts {
            let tile = Q8_0X4_GEMM_NC.min(n_acts - j0);
            let mut acc = [vdupq_n_f32(0.0); Q8_0X4_GEMM_NC];
            for b in 0..nb {
                let blk = group.as_ptr().add(b * Q8_0X4_BLOCK_BYTES);
                let qs = blk.add(8);
                let w = [
                    vld1q_s8(qs as *const i8),
                    vld1q_s8(qs.add(16) as *const i8),
                    vld1q_s8(qs.add(32) as *const i8),
                    vld1q_s8(qs.add(48) as *const i8),
                    vld1q_s8(qs.add(64) as *const i8),
                    vld1q_s8(qs.add(80) as *const i8),
                    vld1q_s8(qs.add(96) as *const i8),
                    vld1q_s8(qs.add(112) as *const i8),
                ];
                let d_bits = vld1_u16(blk as *const u16);
                let mut dw = [0f32; 4];
                dw[0] = f16::from_bits(vget_lane_u16(d_bits, 0)).to_f32();
                dw[1] = f16::from_bits(vget_lane_u16(d_bits, 1)).to_f32();
                dw[2] = f16::from_bits(vget_lane_u16(d_bits, 2)).to_f32();
                dw[3] = f16::from_bits(vget_lane_u16(d_bits, 3)).to_f32();
                let dw_v = vld1q_f32(dw.as_ptr());

                for t in 0..tile {
                    let act = &acts[j0 + t];
                    let a_ptr = act.q.as_ptr().add(b * Q8_0_BLOCK_ELEMS);
                    let a0 = vld1q_s8(a_ptr);
                    let a1 = vld1q_s8(a_ptr.add(16));
                    let mut ret = vdupq_n_s32(0);
                    ret = sdot_lane(ret, w[0], a0, 0);
                    ret = sdot_lane(ret, w[1], a0, 1);
                    ret = sdot_lane(ret, w[2], a0, 2);
                    ret = sdot_lane(ret, w[3], a0, 3);
                    ret = sdot_lane(ret, w[4], a1, 0);
                    ret = sdot_lane(ret, w[5], a1, 1);
                    ret = sdot_lane(ret, w[6], a1, 2);
                    ret = sdot_lane(ret, w[7], a1, 3);
                    let scale = vmulq_n_f32(dw_v, act.d[b]);
                    acc[t] = vfmaq_f32(acc[t], vcvtq_f32_s32(ret), scale);
                }
            }
            for t in 0..tile {
                let mut lanes = [0f32; Q8_0X4_NROWS];
                vst1q_f32(lanes.as_mut_ptr(), acc[t]);
                for (r, v) in lanes.iter().enumerate() {
                    out[r * n_acts + j0 + t] = *v;
                }
            }
            j0 += tile;
        }
    }

    /// NEON DotProd GEMV for `block_q4_0x4` (llama `ggml_gemv_q4_0_4x4_q8_0`).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemv_q4_0x4_q8_0_neon_sdot(
        packed: &[u8],
        act: &Q8Activations,
        n_cols: usize,
        n_row_groups: usize,
        out: &mut [f32],
    ) {
        let nb = n_cols / Q4_0_BLOCK_ELEMS;
        let maskf0 = vdupq_n_u8(0xF0);
        for x in 0..n_row_groups {
            let mut acc = vdupq_n_f32(0.0);
            let group_off = x * nb * Q4_0X4_BLOCK_BYTES;
            for b in 0..nb {
                let blk = packed.as_ptr().add(group_off + b * Q4_0X4_BLOCK_BYTES);
                let qs = blk.add(8);
                let a_ptr = act.q.as_ptr().add(b * Q4_0_BLOCK_ELEMS);
                let a0 = vld1q_s8(a_ptr);
                let a1 = vld1q_s8(a_ptr.add(16));

                let mut ret = vdupq_n_s32(0);
                for wi in 0..4u32 {
                    let w = vld1q_u8(qs.add(wi as usize * 16));
                    let hi = vreinterpretq_s8_u8(vshlq_n_u8(w, 4));
                    let lo = vreinterpretq_s8_u8(vandq_u8(w, maskf0));
                    ret = sdot_lane(ret, hi, a0, wi);
                    ret = sdot_lane(ret, lo, a1, wi);
                }

                let d_bits = vld1_u16(blk as *const u16);
                let mut dw = [0f32; 4];
                dw[0] = f16::from_bits(vget_lane_u16(d_bits, 0)).to_f32();
                dw[1] = f16::from_bits(vget_lane_u16(d_bits, 1)).to_f32();
                dw[2] = f16::from_bits(vget_lane_u16(d_bits, 2)).to_f32();
                dw[3] = f16::from_bits(vget_lane_u16(d_bits, 3)).to_f32();
                let scale = vmulq_n_f32(vld1q_f32(dw.as_ptr()), act.d[b]);
                acc = vfmaq_f32(acc, vcvtq_f32_s32(vshrq_n_s32(ret, 4)), scale);
            }
            vst1q_f32(out.as_mut_ptr().add(x * Q4_0X4_NROWS), acc);
        }
    }

    /// NEON DotProd GEMM for one `block_q4_0x4` row-group against several
    /// activations (llama `ggml_gemm_q4_0_4x4_q8_0` in shape).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemm_q4_0x4_q8_0_neon_sdot(
        group: &[u8],
        acts: &[Q8Activations],
        n_cols: usize,
        out: &mut [f32],
    ) {
        let nb = n_cols / Q4_0_BLOCK_ELEMS;
        let n_acts = acts.len();
        let maskf0 = vdupq_n_u8(0xF0);
        let mut j0 = 0;
        while j0 < n_acts {
            let tile = Q4_0X4_GEMM_NC.min(n_acts - j0);
            let mut acc = [vdupq_n_f32(0.0); Q4_0X4_GEMM_NC];
            for b in 0..nb {
                let blk = group.as_ptr().add(b * Q4_0X4_BLOCK_BYTES);
                let qs = blk.add(8);
                let w = [
                    vld1q_u8(qs),
                    vld1q_u8(qs.add(16)),
                    vld1q_u8(qs.add(32)),
                    vld1q_u8(qs.add(48)),
                ];
                let d_bits = vld1_u16(blk as *const u16);
                let mut dw = [0f32; 4];
                dw[0] = f16::from_bits(vget_lane_u16(d_bits, 0)).to_f32();
                dw[1] = f16::from_bits(vget_lane_u16(d_bits, 1)).to_f32();
                dw[2] = f16::from_bits(vget_lane_u16(d_bits, 2)).to_f32();
                dw[3] = f16::from_bits(vget_lane_u16(d_bits, 3)).to_f32();
                let dw_v = vld1q_f32(dw.as_ptr());

                for t in 0..tile {
                    let act = &acts[j0 + t];
                    let a_ptr = act.q.as_ptr().add(b * Q4_0_BLOCK_ELEMS);
                    let a0 = vld1q_s8(a_ptr);
                    let a1 = vld1q_s8(a_ptr.add(16));
                    let mut ret = vdupq_n_s32(0);
                    for (wi, wchunk) in w.iter().enumerate() {
                        let hi = vreinterpretq_s8_u8(vshlq_n_u8(*wchunk, 4));
                        let lo = vreinterpretq_s8_u8(vandq_u8(*wchunk, maskf0));
                        ret = sdot_lane(ret, hi, a0, wi as u32);
                        ret = sdot_lane(ret, lo, a1, wi as u32);
                    }
                    let scale = vmulq_n_f32(dw_v, act.d[b]);
                    acc[t] = vfmaq_f32(acc[t], vcvtq_f32_s32(vshrq_n_s32(ret, 4)), scale);
                }
            }
            for t in 0..tile {
                let mut lanes = [0f32; Q4_0X4_NROWS];
                vst1q_f32(lanes.as_mut_ptr(), acc[t]);
                for (r, v) in lanes.iter().enumerate() {
                    out[r * n_acts + j0 + t] = *v;
                }
            }
            j0 += tile;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// AVX2 GEMV for interleave-8 packed weights. Accumulates 8 f32 outputs
    /// in `__m256` lanes; inner int dots use maddubs over nibble×act pairs.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemv_q4_kx8_q8_k_avx2(
        packed: &[u8],
        act: &Q8KActivations,
        n_cols: usize,
        n_row_groups: usize,
        out: &mut [f32],
    ) {
        let nb = n_cols / Q4_K_BLOCK_ELEMS;
        let blocklen = 8;
        let ncols = Q4_KX8_NROWS;

        for x in 0..n_row_groups {
            let mut acc = _mm256_setzero_ps();
            let mut acc_min = _mm256_setzero_ps();
            let group_off = x * nb * Q4_KX8_BLOCK_BYTES;

            for l in 0..nb {
                let blk = packed.as_ptr().add(group_off + l * Q4_KX8_BLOCK_BYTES);
                let mut d_arr = [0f32; 8];
                let mut dmin_arr = [0f32; 8];
                for j in 0..8 {
                    d_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
                    dmin_arr[j] =
                        f16_from_bytes(std::slice::from_raw_parts(blk.add(16 + j * 2), 2));
                }
                let da = act.d[l];
                let d_vec = _mm256_mul_ps(_mm256_loadu_ps(d_arr.as_ptr()), _mm256_set1_ps(da));
                let dmin_vec =
                    _mm256_mul_ps(_mm256_loadu_ps(dmin_arr.as_ptr()), _mm256_set1_ps(da));

                let scales = std::slice::from_raw_parts(blk.add(32), 96);
                let qs = std::slice::from_raw_parts(blk.add(128), 1024);
                let q8 = &act.q[l * Q4_K_BLOCK_ELEMS..(l + 1) * Q4_K_BLOCK_ELEMS];
                let bsums = &act.bsums[l * 16..(l + 1) * 16];

                let mut all_scales = [[0u8; 8]; 8];
                let mut all_mins = [[0u8; 8]; 8];
                for sb in 0..8 {
                    decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
                }

                let mut isum = [0i32; 8];
                let n_k = Q4_K_BLOCK_ELEMS / (2 * blocklen);
                for k in 0..n_k {
                    let sb_pair = k / 4;
                    let sc0 = &all_scales[sb_pair * 2];
                    let sc1 = &all_scales[sb_pair * 2 + 1];
                    for j in 0..ncols {
                        let mut s = 0i32;
                        for i in 0..blocklen {
                            let qbyte = qs[k * ncols * blocklen + j * blocklen + i];
                            let v0 = (qbyte & 0x0F) as i32;
                            let v1 = (qbyte >> 4) as i32;
                            let a0 = q8[(k >> 2) * 64 + (k % 4) * blocklen + i] as i32;
                            let a1 = q8[(k >> 2) * 64 + (k % 4) * blocklen + i + 32] as i32;
                            s += v0 * a0 * sc0[j] as i32 + v1 * a1 * sc1[j] as i32;
                        }
                        isum[j] += s;
                    }
                }

                let isum_ps =
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(isum.as_ptr() as *const __m256i));
                acc = _mm256_fmadd_ps(isum_ps, d_vec, acc);

                let mut minsum = [0i32; 8];
                for sb in 0..8 {
                    let bsum = bsums[sb * 2] as i32 + bsums[sb * 2 + 1] as i32;
                    for j in 0..ncols {
                        minsum[j] += all_mins[sb][j] as i32 * bsum;
                    }
                }
                let minsum_ps =
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(minsum.as_ptr() as *const __m256i));
                acc_min = _mm256_fmadd_ps(minsum_ps, dmin_vec, acc_min);
            }

            _mm256_storeu_ps(out.as_mut_ptr().add(x * ncols), _mm256_sub_ps(acc, acc_min));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dot_q4_0_q8_scalar, dot_q4_k_q8_scalar, dot_q8_0_q8_scalar, quantize_activations_q8,
        quantize_activations_q8_k, Q4_0_BLOCK_BYTES,
    };

    fn synth_q4_k_row(n_blocks: usize, seed: u8) -> Vec<u8> {
        let mut weights = Vec::with_capacity(n_blocks * Q4_K_BLOCK_BYTES);
        for b in 0..n_blocks {
            weights.extend_from_slice(
                &f16::from_f32(0.05 + (b as f32 + seed as f32) * 0.01).to_le_bytes(),
            );
            weights.extend_from_slice(
                &f16::from_f32(0.01 + (b as f32 + seed as f32) * 0.002).to_le_bytes(),
            );
            for i in 0..12u8 {
                weights.push(20 + i.wrapping_mul(3).wrapping_add(seed));
            }
            for i in 0..128u8 {
                weights.push(i.wrapping_mul(17).wrapping_add(b as u8).wrapping_add(seed));
            }
        }
        weights
    }

    fn synth_q4_0_row(n_blocks: usize, seed: u8) -> Vec<u8> {
        let mut weights = Vec::with_capacity(n_blocks * Q4_0_BLOCK_BYTES);
        for b in 0..n_blocks {
            weights.extend_from_slice(
                &f16::from_f32(0.05 + (b as f32 + seed as f32) * 0.012).to_le_bytes(),
            );
            for i in 0..16u8 {
                weights.push(i.wrapping_mul(23).wrapping_add(b as u8).wrapping_add(seed));
            }
        }
        weights
    }

    fn synth_q8_0_row(n_blocks: usize, seed: u8) -> Vec<u8> {
        let mut weights = Vec::with_capacity(n_blocks * Q8_0_BLOCK_BYTES);
        for b in 0..n_blocks {
            weights.extend_from_slice(
                &f16::from_f32(0.04 + (b as f32 + seed as f32) * 0.008).to_le_bytes(),
            );
            for i in 0..32u8 {
                // signed i8 stored as u8 bytes
                let q = ((i as i8)
                    .wrapping_mul(3)
                    .wrapping_add(seed as i8)
                    .wrapping_add(b as i8)) as u8;
                weights.push(q);
            }
        }
        weights
    }

    #[test]
    fn pack_and_gemv_matches_scalar_row_dots() {
        let n_blocks = 2;
        let cols = n_blocks * Q4_K_BLOCK_ELEMS;
        let rows = 16; // two full groups
        let mut matrix = Vec::new();
        for r in 0..rows {
            matrix.extend_from_slice(&synth_q4_k_row(n_blocks, r as u8));
        }
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.017 - 2.1).sin() * 1.8)
            .collect();
        let act = quantize_activations_q8_k(&x);

        let mut reference = vec![0f32; rows];
        let row_bytes = n_blocks * Q4_K_BLOCK_BYTES;
        for r in 0..rows {
            reference[r] = dot_q4_k_q8_scalar(&matrix[r * row_bytes..(r + 1) * row_bytes], &act);
        }

        for &interleave in &[4usize, 8] {
            let packed = pack_q4_k_matrix_x8(&matrix, rows, cols, interleave);
            let n_groups = rows / Q4_KX8_NROWS;
            let mut out = vec![0f32; rows];
            gemv_q4_kx8_q8_k(&packed, &act, cols, n_groups, interleave, &mut out);
            for r in 0..rows {
                let err = (out[r] - reference[r]).abs();
                let scale = reference[r].abs().max(1.0);
                assert!(
                    err / scale < 1e-4 || err < 1e-3,
                    "interleave={interleave} row {r}: got {} want {} err={err}",
                    out[r],
                    reference[r]
                );
            }
        }
    }

    #[test]
    fn q4_0x4_pack_and_gemv_matches_scalar_row_dots() {
        let n_blocks = 3;
        let cols = n_blocks * Q4_0_BLOCK_ELEMS;
        let rows = 12;
        let mut matrix = Vec::new();
        for r in 0..rows {
            matrix.extend_from_slice(&synth_q4_0_row(n_blocks, r as u8));
        }
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.019 - 1.2).sin() * 2.1)
            .collect();
        let act = quantize_activations_q8(&x);

        let row_bytes = n_blocks * Q4_0_BLOCK_BYTES;
        let mut reference = vec![0f32; rows];
        for r in 0..rows {
            reference[r] = dot_q4_0_q8_scalar(&matrix[r * row_bytes..(r + 1) * row_bytes], &act);
        }

        let packed = pack_q4_0_matrix_x4(&matrix, rows, cols, Q4_0X4_INTERLEAVE);
        let n_groups = rows / Q4_0X4_NROWS;
        let mut out = vec![0f32; rows];
        gemv_q4_0x4_q8_0(&packed, &act, cols, n_groups, &mut out);
        for r in 0..rows {
            let err = (out[r] - reference[r]).abs();
            let scale = reference[r].abs().max(1.0);
            assert!(
                err / scale < 1e-4 || err < 1e-3,
                "Q4_0x4 row {r}: got {} want {} err={err}",
                out[r],
                reference[r]
            );
        }
    }

    #[test]
    fn q4_0x4_gemm_matches_the_gemv_run_once_per_activation() {
        let n_blocks = 4;
        let cols = n_blocks * Q4_0_BLOCK_ELEMS;
        let rows = 8;
        let mut matrix = Vec::new();
        for r in 0..rows {
            matrix.extend_from_slice(&synth_q4_0_row(n_blocks, (r * 2 + 5) as u8));
        }
        let packed = pack_q4_0_matrix_x4(&matrix, rows, cols, Q4_0X4_INTERLEAVE);

        let n_acts = 7;
        let acts: Vec<Q8Activations> = (0..n_acts)
            .map(|j| {
                let x: Vec<f32> = (0..cols)
                    .map(|i| (((i + j * 11) as f32) * 0.021 - 0.7).cos() * 1.9)
                    .collect();
                quantize_activations_q8(&x)
            })
            .collect();

        for group in 0..rows / Q4_0X4_NROWS {
            let mut gemm_out = vec![0f32; Q4_0X4_NROWS * n_acts];
            gemm_q4_0x4_group(&packed, group, &acts, cols, &mut gemm_out);

            for (j, act) in acts.iter().enumerate() {
                let mut gemv_out = [0f32; Q4_0X4_NROWS];
                gemv_q4_0x4_group(&packed, group, act, cols, &mut gemv_out);
                for r in 0..Q4_0X4_NROWS {
                    assert_eq!(
                        gemm_out[r * n_acts + j],
                        gemv_out[r],
                        "group {group} row {r} act {j}: Q4_0 GEMM and GEMV disagree"
                    );
                }
            }
        }
    }

    #[test]
    fn q4_0x4_gemm_with_no_activations_is_a_no_op() {
        let n_blocks = 2;
        let cols = n_blocks * Q4_0_BLOCK_ELEMS;
        let mut matrix = Vec::new();
        for r in 0..Q4_0X4_NROWS {
            matrix.extend_from_slice(&synth_q4_0_row(n_blocks, r as u8));
        }
        let packed = pack_q4_0_matrix_x4(&matrix, Q4_0X4_NROWS, cols, Q4_0X4_INTERLEAVE);
        let mut out: Vec<f32> = Vec::new();
        gemm_q4_0x4_group(&packed, 0, &[], cols, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn q8_0x4_pack_and_gemv_matches_scalar_row_dots() {
        let n_blocks = 3;
        let cols = n_blocks * Q8_0_BLOCK_ELEMS;
        let rows = 12; // three full groups of 4
        let mut matrix = Vec::new();
        for r in 0..rows {
            matrix.extend_from_slice(&synth_q8_0_row(n_blocks, r as u8));
        }
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.023 - 1.4).cos() * 2.2)
            .collect();
        let act = quantize_activations_q8(&x);

        let row_bytes = n_blocks * Q8_0_BLOCK_BYTES;
        let mut reference = vec![0f32; rows];
        for r in 0..rows {
            reference[r] = dot_q8_0_q8_scalar(&matrix[r * row_bytes..(r + 1) * row_bytes], &act);
        }

        let packed = pack_q8_0_matrix_x4(&matrix, rows, cols, Q8_0X4_INTERLEAVE);
        let n_groups = rows / Q8_0X4_NROWS;
        let mut out = vec![0f32; rows];
        gemv_q8_0x4_q8_0(&packed, &act, cols, n_groups, &mut out);
        for r in 0..rows {
            let err = (out[r] - reference[r]).abs();
            let scale = reference[r].abs().max(1.0);
            assert!(
                err / scale < 1e-4 || err < 1e-3,
                "Q8_0x4 row {r}: got {} want {} err={err}",
                out[r],
                reference[r]
            );
        }
    }

    /// The GEMM exists purely to reuse weight loads across activations,
    /// so it must produce exactly what the per-activation GEMV produces
    /// -- not merely something close. Any divergence would be a
    /// batch-size-dependent numeric difference, i.e. prefill and decode
    /// disagreeing about the same prompt.
    #[test]
    fn q8_0x4_gemm_matches_the_gemv_run_once_per_activation() {
        let n_blocks = 4;
        let cols = n_blocks * Q8_0_BLOCK_ELEMS;
        let rows = 8;
        let mut matrix = Vec::new();
        for r in 0..rows {
            matrix.extend_from_slice(&synth_q8_0_row(n_blocks, (r * 3 + 1) as u8));
        }
        let packed = pack_q8_0_matrix_x4(&matrix, rows, cols, Q8_0X4_INTERLEAVE);

        // Deliberately not a multiple of the tile width, so the tail
        // path is covered too.
        let n_acts = 7;
        let acts: Vec<Q8Activations> = (0..n_acts)
            .map(|j| {
                let x: Vec<f32> = (0..cols)
                    .map(|i| (((i + j * 13) as f32) * 0.017 - 0.9).sin() * 1.7)
                    .collect();
                quantize_activations_q8(&x)
            })
            .collect();

        for group in 0..rows / Q8_0X4_NROWS {
            let mut gemm_out = vec![0f32; Q8_0X4_NROWS * n_acts];
            gemm_q8_0x4_group(&packed, group, &acts, cols, &mut gemm_out);

            for (j, act) in acts.iter().enumerate() {
                let mut gemv_out = [0f32; Q8_0X4_NROWS];
                gemv_q8_0x4_group(&packed, group, act, cols, &mut gemv_out);
                for r in 0..Q8_0X4_NROWS {
                    assert_eq!(
                        gemm_out[r * n_acts + j],
                        gemv_out[r],
                        "group {group} row {r} act {j}: GEMM and GEMV disagree"
                    );
                }
            }
        }
    }

    /// The Q4_K GEMM must agree with the GEMV **exactly**, for the same
    /// reason as the Q8_0 pair above: the two run on the same prompt in
    /// different batch regimes (prefill vs the `< 4` tail vs decode), so
    /// any divergence is prefill and decode disagreeing about the same
    /// tokens. The GEMM only reorders which loop the weight unpack sits
    /// in — every multiply-accumulate happens in the same order and the
    /// same precision — so equality is the right assertion, not
    /// closeness.
    #[test]
    fn q4_kx8_gemm_matches_the_gemv_run_once_per_activation() {
        let n_blocks = 3;
        let cols = n_blocks * Q4_K_BLOCK_ELEMS;
        let rows = 2 * Q4_KX8_NROWS;
        let mut matrix = Vec::new();
        for r in 0..rows {
            matrix.extend_from_slice(&synth_q4_k_row(n_blocks, (r * 5 + 3) as u8));
        }
        let interleave = 4;
        let packed = pack_q4_k_matrix_x8(&matrix, rows, cols, interleave);

        // Not a multiple of the tile width, so the ragged tail the
        // caller has to chunk around is covered too.
        let n_acts = 6;
        let acts: Vec<Q8KActivations> = (0..n_acts)
            .map(|j| {
                let x: Vec<f32> = (0..cols)
                    .map(|i| (((i + j * 29) as f32) * 0.011 - 0.4).cos() * 2.3)
                    .collect();
                quantize_activations_q8_k(&x)
            })
            .collect();

        for group in 0..rows / Q4_KX8_NROWS {
            for chunk in acts.chunks(Q4_KX8_GEMM_NC) {
                let mut gemm_out = vec![0f32; Q4_KX8_NROWS * chunk.len()];
                gemm_q4_kx8_group(&packed, group, chunk, cols, interleave, &mut gemm_out);

                for (j, act) in chunk.iter().enumerate() {
                    let mut gemv_out = [0f32; Q4_KX8_NROWS];
                    gemv_q4_kx8_group(&packed, group, act, cols, interleave, &mut gemv_out);
                    for r in 0..Q4_KX8_NROWS {
                        assert_eq!(
                            gemm_out[r * chunk.len() + j],
                            gemv_out[r],
                            "group {group} row {r} act {j}: Q4_K GEMM and GEMV disagree"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q4_kx8_gemm_with_no_activations_is_a_no_op() {
        let n_blocks = 2;
        let cols = n_blocks * Q4_K_BLOCK_ELEMS;
        let mut matrix = Vec::new();
        for r in 0..Q4_KX8_NROWS {
            matrix.extend_from_slice(&synth_q4_k_row(n_blocks, r as u8));
        }
        let packed = pack_q4_k_matrix_x8(&matrix, Q4_KX8_NROWS, cols, 4);
        let mut out: Vec<f32> = Vec::new();
        gemm_q4_kx8_group(&packed, 0, &[], cols, 4, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn q8_0x4_gemm_with_no_activations_is_a_no_op() {
        let n_blocks = 2;
        let cols = n_blocks * Q8_0_BLOCK_ELEMS;
        let mut matrix = Vec::new();
        for r in 0..Q8_0X4_NROWS {
            matrix.extend_from_slice(&synth_q8_0_row(n_blocks, r as u8));
        }
        let packed = pack_q8_0_matrix_x4(&matrix, Q8_0X4_NROWS, cols, Q8_0X4_INTERLEAVE);
        let mut out: Vec<f32> = Vec::new();
        gemm_q8_0x4_group(&packed, 0, &[], cols, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn block_size_matches_ggml() {
        assert_eq!(Q4_KX8_BLOCK_BYTES, 16 + 16 + 96 + 1024);
        assert_eq!(Q8_0X4_BLOCK_BYTES, 4 * 2 + Q8_0_BLOCK_ELEMS * Q8_0X4_NROWS);
        assert_eq!(Q4_0X4_BLOCK_BYTES, 4 * 2 + Q4_0_BLOCK_ELEMS * 2);
    }
}
