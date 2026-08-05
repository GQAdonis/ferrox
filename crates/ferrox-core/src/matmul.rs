use rayon::prelude::*;

use crate::tensor::Tensor;

/// Row-major matmul: `a` is [m, k], `b_t` is [n, k] (i.e. already
/// transposed, which is how GGUF stores weight matrices for a `y = W x`
/// projection). Output is [m, n]. Parallelized over output rows with
/// rayon, matching the "each output row is independent" decomposition
/// used across llama.cpp / ggml's matmul kernels -- except for the
/// single-token decode case (`m == 1`, the common case for this path,
/// since `WeightMatrix::F32` is only used for small tensors like
/// embeddings/synthetic weights), where parallelizing over `m` would
/// give exactly one chunk regardless of thread count, i.e. no
/// parallelism at all no matter how large `n` is. That case instead
/// parallelizes over `n` (output features) directly, since `out` is
/// exactly `n` elements long when `m == 1` and needs no layout
/// transpose to do so.
pub fn matmul_f32(a: &Tensor, b_t: &Tensor) -> Tensor {
    let m = a.rows();
    let k = a.cols();
    let n = b_t.rows();
    assert_eq!(
        b_t.cols(),
        k,
        "matmul shape mismatch: a is [{m},{k}], b_t is [{},{}]",
        b_t.rows(),
        b_t.cols()
    );

    let mut out = vec![0f32; m * n];
    if m == 1 {
        let a_row = a.row(0);
        out.par_iter_mut().enumerate().for_each(|(col, out_val)| {
            let b_row = b_t.row(col);
            let mut acc = 0f32;
            for i in 0..k {
                acc += a_row[i] * b_row[i];
            }
            *out_val = acc;
        });
    } else {
        out.par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, out_row)| {
                let a_row = a.row(row);
                for (col, out_val) in out_row.iter_mut().enumerate() {
                    let b_row = b_t.row(col);
                    let mut acc = 0f32;
                    for i in 0..k {
                        acc += a_row[i] * b_row[i];
                    }
                    *out_val = acc;
                }
            });
    }

    Tensor::new(out, vec![m, n])
}

/// RMSNorm as used by LLaMA-family and DeepSeek/GLM/Kimi-family decoders:
/// x_normalized = x / sqrt(mean(x^2) + eps) * weight
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len());
    let mean_sq = sum_sq(x) / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    let mut out = vec![0f32; x.len()];
    mul3_scale(x, weight, scale, &mut out);
    out
}

/// Per-head RMSNorm (Qwen3 / Gemma3 `attn_q_norm` / `attn_k_norm`):
/// `weight` has length `head_dim` and is reused for every head in
/// `x` (layout `[n_heads, head_dim]` row-major).
pub fn rms_norm_per_head(x: &[f32], weight: &[f32], head_dim: usize, eps: f32) -> Vec<f32> {
    assert_eq!(weight.len(), head_dim);
    assert_eq!(x.len() % head_dim, 0);
    let mut out = vec![0f32; x.len()];
    for (head, out_h) in x.chunks_exact(head_dim).zip(out.chunks_exact_mut(head_dim)) {
        let mean_sq = sum_sq(head) / head_dim as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        mul3_scale(head, weight, scale, out_h);
    }
    out
}

#[inline]
fn sum_sq(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { sum_sq_neon(x) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { sum_sq_avx2(x) };
        }
    }
    x.iter().map(|v| v * v).sum()
}

#[inline]
fn mul3_scale(x: &[f32], w: &[f32], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), w.len());
    debug_assert_eq!(x.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { mul3_scale_neon(x, w, scale, out) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { mul3_scale_avx2(x, w, scale, out) };
            return;
        }
    }
    for ((o, &xv), &wv) in out.iter_mut().zip(x).zip(w) {
        *o = xv * scale * wv;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_sq_neon(x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let v = vld1q_f32(x.as_ptr().add(i));
        acc = vfmaq_f32(acc, v, v);
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum += x[i] * x[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mul3_scale_neon(x: &[f32], w: &[f32], scale: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let vs = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let wv = vld1q_f32(w.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(vmulq_f32(xv, vs), wv));
        i += 4;
    }
    while i < n {
        out[i] = x[i] * scale * w[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_sq_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        acc = _mm256_fmadd_ps(v, v, acc);
        i += 8;
    }
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut s128 = _mm_add_ps(lo, hi);
    s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    s128 = _mm_add_ss(s128, _mm_shuffle_ps(s128, s128, 1));
    let mut sum = _mm_cvtss_f32(s128);
    while i < n {
        sum += x[i] * x[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul3_scale_avx2(x: &[f32], w: &[f32], scale: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let vs = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        _mm256_storeu_ps(
            out.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_mul_ps(xv, vs), wv),
        );
        i += 8;
    }
    while i < n {
        out[i] = x[i] * scale * w[i];
        i += 1;
    }
}

/// Soft-cap used by Gemma 2+ attention / final logits:
/// `softcap * tanh(x / softcap)`.
pub fn softcap_inplace(x: &mut [f32], softcap: f32) {
    if softcap <= 0.0 {
        return;
    }
    let inv = 1.0 / softcap;
    for v in x.iter_mut() {
        *v = softcap * (*v * inv).tanh();
    }
}

/// GELU (tanh approximation) used by Gemma GeGLU FFNs.
pub fn gelu(x: f32) -> f32 {
    // HuggingFace / llama.cpp GELU tanh approx.
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    const C: f32 = 0.044_715;
    0.5 * x * (1.0 + (K * (x + C * x * x * x)).tanh())
}

/// Elementwise gated FFN combine: gelu(gate) * up (Gemma GeGLU).
pub fn geglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| gelu(*g) * u)
        .collect()
}

/// Plain (non-RMS) LayerNorm -- ggml's `LLM_NORM` (as opposed to
/// `LLM_NORM_RMS`, what [`rms_norm`] implements): subtract the mean,
/// divide by the standard deviation, then apply an elementwise
/// affine `* weight + bias`. GLM-5.2's real DSA lightning indexer
/// normalizes its compressed key through exactly this
/// (`indexer_k_norm` carries both a `weight` *and* a `bias` GGUF
/// tensor -- confirmed against llama.cpp PR #23346/#25407's real
/// `create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight"|"bias", i),
/// ...)` calls and the `build_norm(indexer_k, ..., LLM_NORM, il)` call
/// site, `LLM_NORM` being ggml's plain-LayerNorm op, distinct from
/// every other norm in this codebase so far, which are all RMSNorm).
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), bias.len());
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .map(|((v, w), b)| (v - mean) * inv_std * w + b)
        .collect()
}

/// SiLU / swish activation: x * sigmoid(x). Used by the SwiGLU-style
/// gated MLP and MoE expert feed-forward blocks in this family of models.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Elementwise gated FFN combine: silu(gate) * up, the standard SwiGLU
/// pairing used inside both dense and MoE-expert feed-forward blocks.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| silu(*g) * u)
        .collect()
}

/// Kimi K3's `situ` activation (`hidden_act: "situ"` in its real
/// `config.json`, registered as `ACT2FN["situ"] -> SituAndMul` in
/// `modeling_kimi_linear.py`): `beta*tanh(gate/beta)*sigmoid(gate) *
/// linear_beta*tanh(up/linear_beta)`. Not SiLU/SwiGLU -- a real,
/// non-obvious fact confirmed by reading Kimi K3's actual reference
/// source and config (`activation_situ_beta`=4.0,
/// `activation_situ_linear_beta`=25.0) rather than assuming the more
/// common SwiGLU convention every other model in this codebase uses.
pub fn situ_and_mul(gate: &[f32], up: &[f32], beta: f32, linear_beta: f32) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let situ_a = beta * (g / beta).tanh() * (1.0 / (1.0 + (-g).exp()));
            let up_t = linear_beta * (u / linear_beta).tanh();
            situ_a * up_t
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_zero_mean_unit_var_input_is_unchanged_by_weight_one_bias_zero() {
        // A hand-picked vector with mean 0 and population variance 1:
        // [-1, 1] has mean 0, var = ((1)+(1))/2 = 1.
        let x = vec![-1.0, 1.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![0.0, 0.0];
        let out = layer_norm(&x, &weight, &bias, 0.0);
        assert!((out[0] - (-1.0)).abs() < 1e-4);
        assert!((out[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_applies_affine_weight_and_bias_after_normalizing() {
        let x = vec![-1.0, 1.0];
        let weight = vec![2.0, 3.0];
        let bias = vec![10.0, -10.0];
        let out = layer_norm(&x, &weight, &bias, 0.0);
        // normalized = [-1, 1] (same as above), then * weight + bias:
        assert!((out[0] - (-2.0 + 10.0)).abs() < 1e-4);
        assert!((out[1] - (3.0 - 10.0)).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_constant_input_is_zero_before_bias() {
        // Zero variance input: every normalized value must be exactly 0
        // (mean-subtracted, so all zero) regardless of eps, then affine.
        let x = vec![5.0, 5.0, 5.0];
        let weight = vec![1.0, 1.0, 1.0];
        let bias = vec![0.25, 0.25, 0.25];
        let out = layer_norm(&x, &weight, &bias, 1e-5);
        for v in out {
            assert!((v - 0.25).abs() < 1e-4);
        }
    }

    #[test]
    fn matmul_identity_returns_input() {
        // a = [[1,2],[3,4]], b_t = identity transposed = identity
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let identity = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let out = matmul_f32(&a, &identity);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_known_values() {
        // a = [1, 2, 3] (1x3), b_t = [[1,1,1]] (1x3) => dot = 6
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let b_t = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]);
        let out = matmul_f32(&a, &b_t);
        assert_eq!(out.shape, vec![1, 1]);
        assert_eq!(out.data[0], 6.0);
    }

    #[test]
    fn matmul_single_row_batch_matches_sequential_dot_products() {
        // m=1 exercises the dedicated single-token-decode path
        // (parallelized over n, not m) -- check it against a plain
        // sequential dot product per output column, not just m=1/n=1.
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let b_t = Tensor::new(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0],
            vec![4, 3],
        );
        let out = matmul_f32(&a, &b_t);
        assert_eq!(out.shape, vec![1, 4]);
        assert_eq!(out.data, vec![1.0, 2.0, 6.0, 2.0]);
    }

    #[test]
    fn rms_norm_unit_weight_preserves_direction() {
        let x = vec![3.0, 4.0];
        let w = vec![1.0, 1.0];
        let out = rms_norm(&x, &w, 1e-6);
        // ratio between components should be preserved
        assert!((out[0] / out[1] - 3.0 / 4.0).abs() < 1e-4);
    }

    #[test]
    fn silu_is_zero_at_zero_and_monotonic_ish() {
        assert!((silu(0.0)).abs() < 1e-6);
        assert!(silu(5.0) > silu(1.0));
    }

    // Golden values independently computed in Python from the same
    // formula transcribed from Kimi K3's real `SituAndMul` source
    // (`beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)`),
    // using its real config values (`activation_situ_beta`=4.0,
    // `activation_situ_linear_beta`=25.0).
    #[test]
    fn situ_and_mul_matches_independent_python_reference() {
        let cases = [
            (0.0f32, 0.0f32, 0.0f32),
            (2.0, -3.0, -4.861_066_3),
            (-1.5, 10.0, -2.483_860_7),
        ];
        for (gate, up, expected) in cases {
            let got = situ_and_mul(&[gate], &[up], 4.0, 25.0)[0];
            assert!(
                (got - expected).abs() < 1e-4,
                "situ({gate},{up}): rust={got} python={expected}"
            );
        }
    }
}
