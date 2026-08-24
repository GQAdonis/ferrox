//! Awkward-shape parity for the CPU integer-dot repack path
//! (`FERROX_CPU_INT_DOT=1`).
//!
//! Every other `apply_batch` test in the tree runs with the flag at its
//! library default, which is **off** (see
//! `ferrox_core::weight_matrix::cpu_int_dot_enabled` — the library stays
//! reference-exact so the NumPy goldens can assert bit equality). That
//! means none of them ever touch `pack_q*_matrix_x*`, the interleave-8
//! layout, or the ARM i8mm SMMLA GEMMs
//! (`ggml_gemm_q{4,5,6}_K_8x8_q8_K`, `ggml_gemm_q{8,4}_0_4x8_q8_0`).
//! This file turns the flag on and drives those kernels through the
//! shapes that the fixed-shape unit tests in `ferrox-quant`'s
//! `repack.rs` cannot reach:
//!
//! - **row counts that are not a multiple of the tile** — 7, 9, 19 rows
//!   against an 8-row `block_q*_Kx8` tile, 3, 5, 11 against a 4-row
//!   `block_q*_0x4` tile. These split the matmul between the packed
//!   GEMM and the leftover per-row dot tail, and a seam error there is
//!   invisible to any test whose row count divides the tile.
//! - **a single row** — `rows = 1` means *no* row-group at all, so the
//!   packed path must not be entered and the tail must carry the whole
//!   matmul.
//! - **a single column of activations** — `batch_size = 1`, the
//!   degenerate `Q8*ActsX4` quad with `na = 1`.
//! - **batch sizes that straddle the 4-wide activation quad** — 3, 5, 9,
//!   so the last tile is partial and the output scatter writes a short
//!   stride.
//! - **the minimum block count in K** — one 256-element super-block for
//!   the K-quants, one 32-element block for Q8_0/Q4_0. K itself is
//!   always a whole number of blocks: a GGUF row is stored as whole
//!   blocks, and `cpu_int_dot_kind_supported` refuses anything else, so
//!   "K not a multiple of the block" is unrepresentable rather than
//!   untested — the single-block case is the awkward end of that axis.
//!
//! Two references are checked per shape, because they fail differently:
//!
//! 1. `apply_batch` (the packed **GEMM**, i8mm on an i8mm host) against
//!    `apply` run once per batch row (the packed **GEMV**). Both quantize
//!    the activation identically, so only f32 accumulation order differs
//!    and the bound is tight. This is what catches a bad GEMM.
//! 2. Both against an f32 dequantize-and-dot reference built from
//!    `dequant_row`, which never sees the packed buffer. Loose (the Q8
//!    activation quantization is the error floor) but it is the only leg
//!    that catches a bad *pack* — a mis-interleave that GEMV and GEMM
//!    would agree on because they read the same wrong bytes.
//!
//! The flag is a process-wide `OnceLock`, so this file deliberately
//! holds exactly one `#[test]`: flipping it in a shared test binary
//! would silently change the behaviour of every other test in it.

use ferrox_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};

/// Minimal f16 encode for small positive normals (test fixtures only).
fn f16_le(x: f32) -> [u8; 2] {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    (((exp as u16) << 10) | mant as u16).to_le_bytes()
}

/// Deterministic pseudo-random quantized matrix. Every byte pattern is a
/// valid weight payload for these kinds; only the f16 scale fields need
/// sane values, so the quantizer itself is not in the loop.
fn synth_quant_matrix(kind: QuantKind, rows: usize, cols: usize, seed: u32) -> WeightMatrix {
    let mut state = seed | 1;
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 24) as u8
    };
    let mut data = Vec::new();
    match kind {
        QuantKind::Q8_0 | QuantKind::Q4_0 => {
            let qs = if kind == QuantKind::Q8_0 { 32 } else { 16 };
            for _ in 0..rows * (cols / 32) {
                data.extend_from_slice(&f16_le(0.02 + f32::from(next()) * 0.0004));
                for _ in 0..qs {
                    data.push(next());
                }
            }
        }
        QuantKind::Q4K | QuantKind::Q5K => {
            let body = if kind == QuantKind::Q4K {
                12 + 128
            } else {
                12 + 32 + 128
            };
            for _ in 0..rows * (cols / 256) {
                data.extend_from_slice(&f16_le(0.01 + f32::from(next()) * 0.0002));
                data.extend_from_slice(&f16_le(0.005 + f32::from(next()) * 0.0001));
                for _ in 0..body {
                    data.push(next());
                }
            }
        }
        QuantKind::Q6K => {
            for _ in 0..rows * (cols / 256) {
                for _ in 0..128 + 64 + 16 {
                    data.push(next());
                }
                data.extend_from_slice(&f16_le(0.01 + f32::from(next()) * 0.0002));
            }
        }
        _ => unreachable!("synth_quant_matrix: unsupported kind"),
    }
    WeightMatrix::Quantized {
        data: WeightBytes::Owned(data),
        rows,
        cols,
        kind,
    }
}

fn synth_activations(batch: usize, cols: usize, seed: usize) -> Vec<f32> {
    (0..batch * cols)
        .map(|i| ((((i + seed) * 31 + 7) % 97) as f32) * 0.021 - 1.0)
        .collect()
}

/// f32 dequantize-and-dot, independent of every packed layout.
fn dequant_reference(
    matrix: &WeightMatrix,
    x_batch: &[f32],
    batch: usize,
    cols: usize,
) -> Vec<f32> {
    let rows = matrix.rows();
    let mut out = vec![0f32; batch * rows];
    for r in 0..rows {
        let w = matrix.dequant_row(r);
        assert_eq!(w.len(), cols);
        for b in 0..batch {
            let x = &x_batch[b * cols..(b + 1) * cols];
            out[b * rows + r] = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }
    }
    out
}

#[test]
fn int_dot_batch_matches_gemv_and_dequant_across_awkward_shapes() {
    // Must happen before any dispatch decision is cached. This is the
    // only test in this binary, so nothing else can be mid-flight.
    std::env::set_var("FERROX_CPU_INT_DOT", "1");
    std::env::set_var("FERROX_METAL", "0");
    std::env::set_var("FERROX_METAL_ATTN", "0");
    std::env::set_var("FERROX_CUDA", "0");
    assert!(
        ferrox_core::weight_matrix::cpu_int_dot_enabled(),
        "this test is meaningless unless the integer-dot path is live"
    );

    // (kind, cols, row counts). Row counts bracket the tile width: below
    // it, one short of it, exactly it, one past it, and a multi-group
    // count with a tail.
    let cases: &[(QuantKind, &[usize], &[usize])] = &[
        (QuantKind::Q4K, &[256, 512], &[1, 7, 8, 9, 19]),
        (QuantKind::Q5K, &[256, 512], &[1, 7, 8, 9, 19]),
        (QuantKind::Q6K, &[256, 512], &[1, 7, 8, 9, 19]),
        (QuantKind::Q8_0, &[32, 96], &[1, 3, 4, 5, 11]),
        (QuantKind::Q4_0, &[32, 96], &[1, 3, 4, 5, 11]),
    ];
    let batches = [1usize, 2, 3, 4, 5, 9];

    // The repack cache is keyed by the weight buffer's address, so every
    // matrix stays alive for the whole test rather than being dropped
    // into an address a later matrix could be handed back.
    let mut alive: Vec<WeightMatrix> = Vec::new();
    let mut seed = 0x1234_5678u32;
    for (kind, colss, rowss) in cases {
        for &cols in *colss {
            for &rows in *rowss {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                alive.push(synth_quant_matrix(*kind, rows, cols, seed));
            }
        }
    }

    let mut idx = 0usize;
    for (kind, colss, rowss) in cases {
        for &cols in *colss {
            for &rows in *rowss {
                let matrix = &alive[idx];
                idx += 1;
                for &batch in &batches {
                    let x_batch = synth_activations(batch, cols, rows + cols);
                    let got = matrix.apply_batch(&x_batch, batch);
                    assert_eq!(
                        got.len(),
                        batch * rows,
                        "{kind:?} rows {rows} cols {cols} batch {batch}: wrong output length"
                    );

                    // Leg 1: packed GEMM vs packed GEMV, same activation
                    // quantization, so only the f32 fold order differs.
                    for b in 0..batch {
                        let gemv = matrix.apply(&x_batch[b * cols..(b + 1) * cols]);
                        assert_eq!(gemv.len(), rows);
                        for r in 0..rows {
                            let g = got[b * rows + r];
                            let v = gemv[r];
                            let err = (g - v).abs();
                            assert!(
                                err / v.abs().max(1.0) < 1e-4 || err < 1e-3,
                                "{kind:?} rows {rows} cols {cols} batch {batch} \
                                 [b{b} r{r}]: apply_batch={g} apply={v}"
                            );
                        }
                    }

                    // Leg 2: both against f32 dequantize-and-dot. The
                    // bound is the Q8/Q8_K *activation* quantization
                    // floor, not the kernel's, so it is scaled by the
                    // RMS of the reference outputs rather than by each
                    // output: these synthetic weights are uniform random
                    // bytes, so individual dots cancel to near zero and a
                    // per-element relative bound would be meaningless.
                    // Worst deviation measured over every shape below on
                    // an M2 Pro (i8mm) is 0.0395 x RMS; 0.12 keeps a 3x
                    // margin. It is a coarse bound on purpose — its job
                    // is to catch a mis-*pack*, which leg 1 cannot see
                    // because GEMM and GEMV read the same packed bytes,
                    // and a mis-pack decorrelates the output from the
                    // reference entirely (~1.4 x RMS), an order of
                    // magnitude past this bound.
                    let want = dequant_reference(matrix, &x_batch, batch, cols);
                    let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
                    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        let err = (g - w).abs();
                        assert!(
                            err < 0.12 * rms.max(1e-3),
                            "{kind:?} rows {rows} cols {cols} batch {batch} \
                             [flat {i}]: int-dot={g} dequant-dot={w} \
                             (err {err}, rms {rms})"
                        );
                    }
                }
            }
        }
    }
}
