//! Qwen-style Gated Delta Net (GDN) — linear-attention / SSM recurrent
//! primitive for hybrid arches (`qwen35`, `qwen35moe`, `qwen3next`, …).
//!
//! Distinct from Kimi KDA (`kda.rs`): GDN uses a **fused** QKV projection,
//! a **single** depthwise `ssm_conv1d` over the concatenated QKV channels,
//! per-head `ssm_alpha` / `ssm_beta` gates, and decay
//! `exp(softplus(α + ssm_dt) · ssm_a)` (GGUF `ssm_a` is typically
//! `-exp(A_log)`). KDA is not a drop-in for this graph.
//!
//! ## GGUF tensor name mapping (per layer `L`)
//!
//! | Role | GGUF name |
//! |---|---|
//! | Fused Q‖K‖V | `blk.{L}.attn_qkv.weight` |
//! | Output / z gate | `blk.{L}.attn_gate.weight` |
//! | Depthwise causal conv | `blk.{L}.ssm_conv1d.weight` |
//! | Decay bias | `blk.{L}.ssm_dt.bias` (alt: `ssm_dt`) |
//! | Decay scale | `blk.{L}.ssm_a` |
//! | Input gate β | `blk.{L}.ssm_beta.weight` |
//! | Forget raw α | `blk.{L}.ssm_alpha.weight` |
//! | Output RMSNorm | `blk.{L}.ssm_norm.weight` |
//! | Output projection | `blk.{L}.ssm_out.weight` |
//!
//! Legacy `qwen3next` may pack β/α into `ssm_ba` or fuse QKV+z into
//! `ssm_in`; this module implements the split qwen35 layout only.
//!
//! Not wired into GGUF load / [`crate::hybrid_engine::HybridEngine`] serve
//! yet — factory still [`HybridEngine::reject`](crate::hybrid_engine::HybridEngine::reject).

use ferrox_core::matmul::{rms_norm, silu};
use ferrox_core::weight_matrix::WeightMatrix;

/// Tiny-config dims for the Qwen35-style GDN step (equal K/V heads).
#[derive(Debug, Clone, Copy)]
pub struct GdnConfig {
    pub hidden_dim: usize,
    pub num_v_heads: usize,
    pub head_dim: usize,
    pub conv_kernel_size: usize,
    pub rms_norm_eps: f32,
}

impl GdnConfig {
    pub fn qkv_dim(&self) -> usize {
        // Q + K + V with num_k_heads == num_v_heads, head_k == head_v.
        3 * self.num_v_heads * self.head_dim
    }

    pub fn v_dim(&self) -> usize {
        self.num_v_heads * self.head_dim
    }
}

/// Weights matching the qwen35 GGUF layout (see module docs).
pub struct GdnWeights {
    pub attn_qkv: WeightMatrix,  // [qkv_dim, hidden]
    pub attn_gate: WeightMatrix, // [v_dim, hidden]
    /// Depthwise taps, row-major `[qkv_dim, conv_kernel_size]`.
    pub ssm_conv1d: Vec<f32>,
    pub ssm_dt: Vec<f32>,          // [num_v_heads]
    pub ssm_a: Vec<f32>,           // [num_v_heads]
    pub ssm_beta: WeightMatrix,    // [num_v_heads, hidden]
    pub ssm_alpha: WeightMatrix,   // [num_v_heads, hidden]
    pub ssm_norm: Vec<f32>,        // [head_dim]
    pub ssm_out: WeightMatrix,     // [hidden, v_dim]
}

/// Fixed-size recurrent + short-conv state (unlike growing KV).
pub struct GdnState {
    conv_hist: Vec<f32>,
    /// Flat `[num_v_heads, head_dim, head_dim]` — state[v, k] per head.
    recurrent: Vec<f32>,
}

impl GdnState {
    pub fn new(cfg: &GdnConfig) -> Self {
        Self {
            conv_hist: Vec::new(),
            recurrent: vec![0f32; cfg.num_v_heads * cfg.head_dim * cfg.head_dim],
        }
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn l2_normalize(v: &mut [f32], eps: f32) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let scale = 1.0 / (norm_sq + eps).sqrt();
    for x in v.iter_mut() {
        *x *= scale;
    }
}

/// Depthwise causal conv over `dim` channels + SiLU (padding = kernel−1).
fn causal_conv_step(
    weight: &[f32],
    history: &mut Vec<f32>,
    current: &[f32],
    kernel_size: usize,
    dim: usize,
) -> Vec<f32> {
    let hist_len = history.len() / dim.max(1);
    let missing = (kernel_size - 1).saturating_sub(hist_len);

    let mut y = vec![0f32; dim];
    for j in 0..kernel_size {
        if j < missing {
            continue;
        }
        let src: &[f32] = if j == kernel_size - 1 {
            current
        } else {
            let hist_idx = j - missing;
            &history[hist_idx * dim..(hist_idx + 1) * dim]
        };
        for d in 0..dim {
            y[d] += weight[d * kernel_size + j] * src[d];
        }
    }
    for v in y.iter_mut() {
        *v = silu(*v);
    }

    history.extend_from_slice(current);
    let max_hist_len = (kernel_size - 1) * dim;
    if history.len() > max_hist_len {
        let excess = history.len() - max_hist_len;
        history.drain(0..excess);
    }
    y
}

/// One decode step. Assumes `num_k_heads == num_v_heads` and
/// `head_k_dim == head_v_dim == cfg.head_dim`.
pub fn gdn_forward_token(
    weights: &GdnWeights,
    cfg: &GdnConfig,
    hidden: &[f32],
    state: &mut GdnState,
) -> Vec<f32> {
    assert_eq!(hidden.len(), cfg.hidden_dim);
    let qkv_dim = cfg.qkv_dim();
    let v_dim = cfg.v_dim();
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_v_heads;

    let qkv_lin = weights.attn_qkv.apply(hidden);
    let z = weights.attn_gate.apply(hidden);
    let beta_raw = weights.ssm_beta.apply(hidden);
    let alpha_raw = weights.ssm_alpha.apply(hidden);

    let qkv = causal_conv_step(
        &weights.ssm_conv1d,
        &mut state.conv_hist,
        &qkv_lin,
        cfg.conv_kernel_size,
        qkv_dim,
    );

    let qk_dim = n_heads * head_dim;
    let (q_all, rest) = qkv.split_at(qk_dim);
    let (k_all, v_all) = rest.split_at(qk_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut y_flat = vec![0f32; v_dim];

    #[allow(clippy::needless_range_loop)]
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut q_h = q_all[base..base + head_dim].to_vec();
        let mut k_h = k_all[base..base + head_dim].to_vec();
        let v_h = &v_all[base..base + head_dim];

        l2_normalize(&mut q_h, 1e-6);
        l2_normalize(&mut k_h, 1e-6);
        for x in q_h.iter_mut() {
            *x *= scale;
        }

        // g = exp(softplus(α + dt) * A); A = ssm_a (often negative).
        let gate = softplus(alpha_raw[h] + weights.ssm_dt[h]) * weights.ssm_a[h];
        let decay = gate.exp();
        let beta = sigmoid(beta_raw[h]);

        let s_base = h * head_dim * head_dim;
        let s = &mut state.recurrent[s_base..s_base + head_dim * head_dim];

        // state *= decay
        for cell in s.iter_mut() {
            *cell *= decay;
        }

        // kv_mem[v] = sum_k state[v,k] * k[k]
        let mut kv_mem = vec![0f32; head_dim];
        for v_idx in 0..head_dim {
            let mut acc = 0f32;
            for k_idx in 0..head_dim {
                acc += s[v_idx * head_dim + k_idx] * k_h[k_idx];
            }
            kv_mem[v_idx] = acc;
        }

        // state[v,k] += beta * (v - kv_mem)[v] * k[k]
        for v_idx in 0..head_dim {
            let delta = (v_h[v_idx] - kv_mem[v_idx]) * beta;
            for k_idx in 0..head_dim {
                s[v_idx * head_dim + k_idx] += delta * k_h[k_idx];
            }
        }

        // y[v] = sum_k state[v,k] * q[k]
        for v_idx in 0..head_dim {
            let mut acc = 0f32;
            for k_idx in 0..head_dim {
                acc += s[v_idx * head_dim + k_idx] * q_h[k_idx];
            }
            y_flat[base + v_idx] = acc;
        }
    }

    // Per-head RMSNorm on y, then SiLU(z) * normed.
    let mut gated = vec![0f32; v_dim];
    for h in 0..n_heads {
        let base = h * head_dim;
        let normed = rms_norm(
            &y_flat[base..base + head_dim],
            &weights.ssm_norm,
            cfg.rms_norm_eps,
        );
        for i in 0..head_dim {
            gated[base + i] = silu(z[base + i]) * normed[i];
        }
    }

    weights.ssm_out.apply(&gated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_core::tensor::Tensor;

    const HIDDEN: usize = 4;
    const N_HEADS: usize = 2;
    const HEAD_DIM: usize = 2;
    const CONV_K: usize = 2;
    const QKV_DIM: usize = 3 * N_HEADS * HEAD_DIM; // 12
    const V_DIM: usize = N_HEADS * HEAD_DIM; // 4

    fn wm(data: &[f32], rows: usize, cols: usize) -> WeightMatrix {
        assert_eq!(data.len(), rows * cols);
        WeightMatrix::F32(Tensor::new(data.to_vec(), vec![rows, cols]))
    }

    fn cfg() -> GdnConfig {
        GdnConfig {
            hidden_dim: HIDDEN,
            num_v_heads: N_HEADS,
            head_dim: HEAD_DIM,
            conv_kernel_size: CONV_K,
            rms_norm_eps: 1e-5,
        }
    }

    fn make_weights() -> GdnWeights {
        // Deterministic tiny synthetic weights (not a golden oracle).
        let mut qkv = Vec::with_capacity(QKV_DIM * HIDDEN);
        for i in 0..QKV_DIM * HIDDEN {
            qkv.push(((i % 7) as f32 - 3.0) * 0.1);
        }
        let mut gate = Vec::with_capacity(V_DIM * HIDDEN);
        for i in 0..V_DIM * HIDDEN {
            gate.push(((i % 5) as f32 - 2.0) * 0.08);
        }
        let mut conv = Vec::with_capacity(QKV_DIM * CONV_K);
        for i in 0..QKV_DIM * CONV_K {
            conv.push(if i % CONV_K == CONV_K - 1 { 1.0 } else { 0.1 });
        }
        let mut beta = Vec::with_capacity(N_HEADS * HIDDEN);
        let mut alpha = Vec::with_capacity(N_HEADS * HIDDEN);
        for i in 0..N_HEADS * HIDDEN {
            beta.push(((i % 3) as f32 - 1.0) * 0.2);
            alpha.push(((i % 4) as f32 - 1.5) * 0.15);
        }
        let mut out = Vec::with_capacity(HIDDEN * V_DIM);
        for i in 0..HIDDEN * V_DIM {
            out.push(((i % 6) as f32 - 2.5) * 0.12);
        }
        GdnWeights {
            attn_qkv: wm(&qkv, QKV_DIM, HIDDEN),
            attn_gate: wm(&gate, V_DIM, HIDDEN),
            ssm_conv1d: conv,
            ssm_dt: vec![0.1, -0.05],
            // Negative A → decay ∈ (0, 1] after softplus·A + exp.
            ssm_a: vec![-0.5, -0.75],
            ssm_beta: wm(&beta, N_HEADS, HIDDEN),
            ssm_alpha: wm(&alpha, N_HEADS, HIDDEN),
            ssm_norm: vec![1.0, 1.0],
            ssm_out: wm(&out, HIDDEN, V_DIM),
        }
    }

    #[test]
    fn gdn_forward_token_tiny_dims_finite_and_shaped() {
        let weights = make_weights();
        let cfg = cfg();
        let mut state = GdnState::new(&cfg);
        let hidden = [0.2f32, -0.1, 0.3, -0.4];

        let out0 = gdn_forward_token(&weights, &cfg, &hidden, &mut state);
        assert_eq!(out0.len(), HIDDEN);
        assert!(out0.iter().all(|x| x.is_finite()));

        let out1 = gdn_forward_token(&weights, &cfg, &hidden, &mut state);
        assert_eq!(out1.len(), HIDDEN);
        assert!(out1.iter().all(|x| x.is_finite()));
        // Second step must see non-zero recurrent state → different output.
        assert!(
            out0
                .iter()
                .zip(out1.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "recurrent state should change the second token"
        );
    }

    #[test]
    fn softplus_matches_closed_form_at_zero() {
        assert!((softplus(0.0) - (2.0f32).ln()).abs() < 1e-6);
    }
}
