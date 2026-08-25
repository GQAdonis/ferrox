//! Qwen-style Gated Delta Net (GDN) — linear-attention / SSM recurrent
//! primitive for hybrid arches (`qwen35`, `qwen35moe`, `qwen3next`, …).
//!
//! Distinct from Kimi KDA (`kda.rs`): GDN uses a **fused** QKV projection,
//! a **single** depthwise `ssm_conv1d` over the concatenated QKV channels,
//! per-head `ssm_alpha` / `ssm_beta` gates, and decay
//! `exp(softplus(α + ssm_dt) · ssm_a)` (GGUF `ssm_a` is typically
//! `-exp(A_log)`). KDA is not a drop-in for this graph.
//!
//! ## GQA geometry (the shape rule this module implements)
//!
//! GDN is **GQA-shaped**: a real checkpoint has *fewer K heads than V
//! heads* (`num_value_heads % num_key_heads == 0`) and the K and V head
//! dims need not be equal (`key_head_dim ≠ value_head_dim` is legal).
//! Transcribed from FreeToken's `qwen3_5_moe/gdn_reference.py`
//! (`Qwen3_5GatedDeltaNetReference.forward`, no-cache path) and
//! `LinearGatedDeltaGroupConfig`, where `num_key_heads` /
//! `num_value_heads` / `key_head_dim` / `value_head_dim` are four
//! independent numbers, not two:
//!
//! * **Split offsets.** The fused projection produces `conv_dim =
//!   2·key_dim + value_dim` channels, with `key_dim = num_key_heads ·
//!   key_head_dim` and `value_dim = num_value_heads · value_head_dim`,
//!   and is split as `[key_dim, key_dim, value_dim]` — *not* into three
//!   equal thirds. Assuming equality computes the K and V offsets wrong,
//!   so **every** head reads a slice straddling the wrong tensor: the
//!   layer stays finite and correctly shaped and silently returns
//!   garbage rather than failing.
//! * **Replication.** Each K head's q/k pair is replicated
//!   `num_value_heads / num_key_heads` times (`repeat_interleave` on the
//!   head axis) *before* the recurrence, so V head `h` reads K head
//!   `h / rep`. Skipping the replication indexes q/k past the end of the
//!   Q slice (into K, then into V) instead of reusing the shared head.
//! * **Rectangular state.** The recurrent state is `[num_value_heads,
//!   key_head_dim, value_head_dim]`, not square, and the q scale is
//!   `key_head_dim^-0.5` (the *key* dim — the reference takes `dk` from
//!   `key.shape[-1]`). Sizing the state from one head dim under-allocates
//!   whenever `key_head_dim > value_head_dim` and mis-strides the
//!   read-out in either direction.
//!
//! With `num_key_heads == num_value_heads` and `key_head_dim ==
//! value_head_dim` every rule above collapses to the older equal-head
//! path, bit for bit — pinned by
//! `equal_head_geometry_stays_bit_identical_to_the_pre_generalization_output`.
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
//! GGUF weight load skeleton: [`crate::hybrid_gguf_loader`]. Serve still
//! fail-closed — factory [`HybridEngine::reject`](crate::hybrid_engine::HybridEngine::reject).

use ferrox_core::matmul::{rms_norm, silu};
use ferrox_core::weight_matrix::WeightMatrix;

/// Dims for one Qwen35-style GDN block, with **independent** K/V head
/// counts and K/V head dims (see the module docs for why all four are
/// separate numbers).
///
/// Invariant: `num_key_heads > 0` and `num_value_heads % num_key_heads ==
/// 0`. The quotient is the q/k replication factor; a config that violated
/// it would leave V heads with no K head to read from, so
/// [`gdn_forward_token`] asserts it instead of flooring the division and
/// silently pairing a V head with a K head that never fed it.
#[derive(Debug, Clone, Copy)]
pub struct GdnConfig {
    pub hidden_dim: usize,
    /// Number of Q/K heads in the checkpoint (`≤ num_value_heads`).
    pub num_key_heads: usize,
    /// Number of V heads — also the length of `ssm_dt` / `ssm_a` and the
    /// row count of `ssm_beta` / `ssm_alpha`.
    pub num_value_heads: usize,
    /// Width of one Q/K head (`dk`; sets the `dk^-0.5` q scale).
    pub key_head_dim: usize,
    /// Width of one V head (`dv`; also the width of `ssm_norm`).
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
    pub rms_norm_eps: f32,
}

impl GdnConfig {
    /// Channels occupied by Q (and, separately, by K) in the fused
    /// projection: `num_key_heads · key_head_dim`.
    pub fn key_dim(&self) -> usize {
        self.num_key_heads * self.key_head_dim
    }

    /// Channels occupied by V in the fused projection — also the width of
    /// the `attn_gate` (z) projection and of `ssm_out`'s input.
    pub fn value_dim(&self) -> usize {
        self.num_value_heads * self.value_head_dim
    }

    /// Fused Q‖K‖V width: `key_dim + key_dim + value_dim`.
    ///
    /// This is the number the equal-head formula got wrong: `3 ·
    /// num_value_heads · head_dim` only coincides with the real width
    /// when both head counts *and* both head dims match. It is the same
    /// arithmetic that yields the split offsets, so a wrong total means
    /// wrong Q/K/V slices — not a length mismatch anyone would notice.
    pub fn qkv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// q/k replication factor: how many V heads share one K head (the
    /// `repeat_interleave` count in the reference). `1` for the
    /// equal-head geometry.
    pub fn heads_per_key_group(&self) -> usize {
        self.num_value_heads / self.num_key_heads
    }
}

/// Weights matching the qwen35 GGUF layout (see module docs).
pub struct GdnWeights {
    pub attn_qkv: WeightMatrix,  // [qkv_dim, hidden]
    pub attn_gate: WeightMatrix, // [value_dim, hidden]
    /// Depthwise taps, row-major `[qkv_dim, conv_kernel_size]`.
    pub ssm_conv1d: Vec<f32>,
    pub ssm_dt: Vec<f32>,        // [num_value_heads]
    pub ssm_a: Vec<f32>,         // [num_value_heads]
    pub ssm_beta: WeightMatrix,  // [num_value_heads, hidden]
    pub ssm_alpha: WeightMatrix, // [num_value_heads, hidden]
    pub ssm_norm: Vec<f32>,      // [value_head_dim]
    pub ssm_out: WeightMatrix,   // [hidden, value_dim]
}

/// Fixed-size recurrent + short-conv state (unlike growing KV).
pub struct GdnState {
    conv_hist: Vec<f32>,
    /// Flat `[num_value_heads, value_head_dim, key_head_dim]` — one
    /// **rectangular** `state[v, k]` block per V head. The reference
    /// stores the transpose (`[dk, dv]`); same elements, different
    /// traversal order. Sizing this from a single `head_dim` truncates
    /// the block whenever `key_head_dim != value_head_dim`, so the second
    /// and later heads would read another head's memory.
    recurrent: Vec<f32>,
}

impl GdnState {
    pub fn new(cfg: &GdnConfig) -> Self {
        Self {
            conv_hist: Vec::new(),
            recurrent: vec![0f32; cfg.num_value_heads * cfg.value_head_dim * cfg.key_head_dim],
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

/// One decode step of the GQA-shaped Gated Delta Net.
///
/// Handles `num_key_heads ≤ num_value_heads` and `key_head_dim ≠
/// value_head_dim`: the fused projection is split at
/// `[key_dim, key_dim, value_dim]`, V head `h` reads K head
/// `h / heads_per_key_group()` (the reference's `repeat_interleave`), and
/// each head's state is the rectangular `[value_head_dim, key_head_dim]`
/// block. The equal-head geometry is the `heads_per_key_group() == 1`
/// special case and runs identical arithmetic in identical order.
pub fn gdn_forward_token(
    weights: &GdnWeights,
    cfg: &GdnConfig,
    hidden: &[f32],
    state: &mut GdnState,
) -> Vec<f32> {
    assert_eq!(hidden.len(), cfg.hidden_dim);
    assert!(
        cfg.num_key_heads > 0 && cfg.num_value_heads.is_multiple_of(cfg.num_key_heads),
        "GDN needs num_value_heads ({}) to be a positive multiple of num_key_heads ({}); \
         otherwise repeat_interleave has no whole replication factor and some V heads would \
         silently read a K head that never fed them",
        cfg.num_value_heads,
        cfg.num_key_heads
    );

    let qkv_dim = cfg.qkv_dim();
    let key_dim = cfg.key_dim();
    let value_dim = cfg.value_dim();
    let key_head_dim = cfg.key_head_dim;
    let value_head_dim = cfg.value_head_dim;
    let rep = cfg.heads_per_key_group();

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

    // torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1).
    let (q_all, rest) = qkv.split_at(key_dim);
    let (k_all, v_all) = rest.split_at(key_dim);

    let scale = 1.0 / (key_head_dim as f32).sqrt();
    let mut y_flat = vec![0f32; value_dim];

    #[allow(clippy::needless_range_loop)]
    for h in 0..cfg.num_value_heads {
        // repeat_interleave(rep, dim=head): V head h consumes K head h / rep.
        let k_base = (h / rep) * key_head_dim;
        let v_base = h * value_head_dim;
        let mut q_h = q_all[k_base..k_base + key_head_dim].to_vec();
        let mut k_h = k_all[k_base..k_base + key_head_dim].to_vec();
        let v_h = &v_all[v_base..v_base + value_head_dim];

        l2_normalize(&mut q_h, 1e-6);
        l2_normalize(&mut k_h, 1e-6);
        for x in q_h.iter_mut() {
            *x *= scale;
        }

        // g = exp(softplus(α + dt) * A); A = ssm_a (often negative).
        let gate = softplus(alpha_raw[h] + weights.ssm_dt[h]) * weights.ssm_a[h];
        let decay = gate.exp();
        let beta = sigmoid(beta_raw[h]);

        let block = value_head_dim * key_head_dim;
        let s_base = h * block;
        let s = &mut state.recurrent[s_base..s_base + block];

        // state *= decay
        for cell in s.iter_mut() {
            *cell *= decay;
        }

        // kv_mem[v] = sum_k state[v,k] * k[k]
        let mut kv_mem = vec![0f32; value_head_dim];
        for v_idx in 0..value_head_dim {
            let mut acc = 0f32;
            for k_idx in 0..key_head_dim {
                acc += s[v_idx * key_head_dim + k_idx] * k_h[k_idx];
            }
            kv_mem[v_idx] = acc;
        }

        // state[v,k] += beta * (v - kv_mem)[v] * k[k]
        for v_idx in 0..value_head_dim {
            let delta = (v_h[v_idx] - kv_mem[v_idx]) * beta;
            for k_idx in 0..key_head_dim {
                s[v_idx * key_head_dim + k_idx] += delta * k_h[k_idx];
            }
        }

        // y[v] = sum_k state[v,k] * q[k]
        for v_idx in 0..value_head_dim {
            let mut acc = 0f32;
            for k_idx in 0..key_head_dim {
                acc += s[v_idx * key_head_dim + k_idx] * q_h[k_idx];
            }
            y_flat[v_base + v_idx] = acc;
        }
    }

    // Per-head RMSNorm on y (over value_head_dim), then SiLU(z) * normed.
    let mut gated = vec![0f32; value_dim];
    for h in 0..cfg.num_value_heads {
        let base = h * value_head_dim;
        let normed = rms_norm(
            &y_flat[base..base + value_head_dim],
            &weights.ssm_norm,
            cfg.rms_norm_eps,
        );
        for i in 0..value_head_dim {
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
            num_key_heads: N_HEADS,
            num_value_heads: N_HEADS,
            key_head_dim: HEAD_DIM,
            value_head_dim: HEAD_DIM,
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

    /// Raw (un-wrapped) tensor data for one GDN layer, so the reference
    /// oracle below can run its own matvecs instead of borrowing this
    /// module's [`WeightMatrix`] path.
    struct RawGdn {
        qkv: Vec<f32>,   // [qkv_dim, hidden]
        gate: Vec<f32>,  // [value_dim, hidden]
        conv: Vec<f32>,  // [qkv_dim, kernel]
        dt: Vec<f32>,    // [num_value_heads]
        a: Vec<f32>,     // [num_value_heads]
        beta: Vec<f32>,  // [num_value_heads, hidden]
        alpha: Vec<f32>, // [num_value_heads, hidden]
        norm: Vec<f32>,  // [value_head_dim]
        out: Vec<f32>,   // [hidden, value_dim]
    }

    impl RawGdn {
        fn to_weights(&self, cfg: &GdnConfig) -> GdnWeights {
            let h = cfg.hidden_dim;
            GdnWeights {
                attn_qkv: wm(&self.qkv, cfg.qkv_dim(), h),
                attn_gate: wm(&self.gate, cfg.value_dim(), h),
                ssm_conv1d: self.conv.clone(),
                ssm_dt: self.dt.clone(),
                ssm_a: self.a.clone(),
                ssm_beta: wm(&self.beta, cfg.num_value_heads, h),
                ssm_alpha: wm(&self.alpha, cfg.num_value_heads, h),
                ssm_norm: self.norm.clone(),
                ssm_out: wm(&self.out, h, cfg.value_dim()),
            }
        }
    }

    /// Deterministic spread of distinct, bounded, sign-alternating values:
    /// no two channels of the fused projection carry the same number, so a
    /// wrong split offset cannot accidentally alias onto a right answer.
    fn fill(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let k = (i * 37 + seed * 101) % 23;
                (k as f32 - 11.0)
                    * 0.043
                    * if (i + seed).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
            })
            .collect()
    }

    fn matvec(rows_data: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        assert_eq!(rows_data.len(), rows * cols);
        assert_eq!(x.len(), cols);
        (0..rows)
            .map(|r| (0..cols).map(|c| rows_data[r * cols + c] * x[c]).sum())
            .collect()
    }

    fn ref_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn ref_softplus(x: f32) -> f32 {
        (1.0 + x.exp()).ln()
    }

    fn ref_silu(x: f32) -> f32 {
        x * ref_sigmoid(x)
    }

    fn ref_l2norm(v: &[f32]) -> Vec<f32> {
        let sum_sq: f32 = v.iter().map(|x| x * x).sum();
        let inv = 1.0 / (sum_sq + 1e-6).sqrt();
        v.iter().map(|x| x * inv).collect()
    }

    /// Independent transcription of `gdn_reference.py`'s
    /// `Qwen3_5GatedDeltaNetReference.forward` +
    /// `recurrent_gated_delta_rule`, run over a whole token sequence in
    /// the reference's own index order: the split is literally
    /// `[key_dim, key_dim, value_dim]` (`:152`), q/k are materialized
    /// through an explicit `repeat_interleave` (`:163`), and state is
    /// `[dk, dv]` (this module stores the transpose). Nothing here calls
    /// [`gdn_forward_token`], so it is an oracle rather than a restatement
    /// of the code under test.
    ///
    /// `ssm_a` follows the GGUF convention this module documents
    /// (`ssm_a == -exp(A_log)`), so `g = softplus(α + dt) · ssm_a` spells
    /// the reference's `-A_log.exp() * softplus(a + dt_bias)`.
    fn reference_forward(raw: &RawGdn, cfg: &GdnConfig, tokens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let hidden = cfg.hidden_dim;
        let qkv_dim = cfg.qkv_dim();
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let dk = cfg.key_head_dim;
        let dv = cfg.value_head_dim;
        let kernel = cfg.conv_kernel_size;
        let rep = cfg.num_value_heads / cfg.num_key_heads;
        let scale = 1.0 / (dk as f32).sqrt();

        // in_proj_qkv over the sequence, then the depthwise causal conv
        // with padding = kernel-1 (positions before 0 read as zero) + silu.
        let mixed: Vec<Vec<f32>> = tokens
            .iter()
            .map(|h| matvec(&raw.qkv, qkv_dim, hidden, h))
            .collect();
        let mut conved = vec![vec![0f32; qkv_dim]; tokens.len()];
        for (t, conved_t) in conved.iter_mut().enumerate() {
            for (d, out_d) in conved_t.iter_mut().enumerate() {
                let mut acc = 0f32;
                for j in 0..kernel {
                    let src = t as isize - (kernel as isize - 1) + j as isize;
                    if src < 0 {
                        continue;
                    }
                    acc += raw.conv[d * kernel + j] * mixed[src as usize][d];
                }
                *out_d = ref_silu(acc);
            }
        }

        // state[h][k][v] — the reference's [num_v_heads, dk, dv] layout.
        let mut state = vec![vec![vec![0f32; dv]; dk]; cfg.num_value_heads];
        let mut outputs = Vec::with_capacity(tokens.len());

        for (t, token) in tokens.iter().enumerate() {
            let z = matvec(&raw.gate, value_dim, hidden, token);
            let a_raw = matvec(&raw.alpha, cfg.num_value_heads, hidden, token);
            let b_raw = matvec(&raw.beta, cfg.num_value_heads, hidden, token);

            let q_slice = &conved[t][0..key_dim];
            let k_slice = &conved[t][key_dim..2 * key_dim];
            let v_slice = &conved[t][2 * key_dim..];

            // repeat_interleave(rep) on the head axis of q and k.
            let mut q_heads: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_value_heads);
            let mut k_heads: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_value_heads);
            for kh in 0..cfg.num_key_heads {
                for _ in 0..rep {
                    q_heads.push(q_slice[kh * dk..(kh + 1) * dk].to_vec());
                    k_heads.push(k_slice[kh * dk..(kh + 1) * dk].to_vec());
                }
            }

            let mut core = vec![0f32; value_dim];
            for h in 0..cfg.num_value_heads {
                let q = ref_l2norm(&q_heads[h]);
                let k = ref_l2norm(&k_heads[h]);
                let v = &v_slice[h * dv..(h + 1) * dv];

                let decay = (ref_softplus(a_raw[h] + raw.dt[h]) * raw.a[h]).exp();
                let beta = ref_sigmoid(b_raw[h]);

                for row in state[h].iter_mut() {
                    for cell in row.iter_mut() {
                        *cell *= decay;
                    }
                }
                // kv_mem = (state * k[:, None]).sum(dim=-2)
                let mut kv_mem = vec![0f32; dv];
                for (k_idx, row) in state[h].iter().enumerate() {
                    for (v_idx, cell) in row.iter().enumerate() {
                        kv_mem[v_idx] += cell * k[k_idx];
                    }
                }
                // state += k[:, None] * ((v - kv_mem) * beta)[None, :]
                let delta: Vec<f32> = (0..dv).map(|i| (v[i] - kv_mem[i]) * beta).collect();
                for (k_idx, row) in state[h].iter_mut().enumerate() {
                    for (v_idx, cell) in row.iter_mut().enumerate() {
                        *cell += k[k_idx] * delta[v_idx];
                    }
                }
                // out = (state * (q * scale)[:, None]).sum(dim=-2)
                for (k_idx, row) in state[h].iter().enumerate() {
                    for (v_idx, cell) in row.iter().enumerate() {
                        core[h * dv + v_idx] += cell * q[k_idx] * scale;
                    }
                }
            }

            // RMSNormGated over head_v_dim groups, norm_before_gate=True.
            let mut gated = vec![0f32; value_dim];
            for h in 0..cfg.num_value_heads {
                let base = h * dv;
                let mean_sq = core[base..base + dv].iter().map(|x| x * x).sum::<f32>() / dv as f32;
                let inv = 1.0 / (mean_sq + cfg.rms_norm_eps).sqrt();
                for i in 0..dv {
                    gated[base + i] = core[base + i] * inv * raw.norm[i] * ref_silu(z[base + i]);
                }
            }
            outputs.push(matvec(&raw.out, hidden, value_dim, &gated));
        }
        outputs
    }

    /// A GQA-shaped geometry: one K head feeding two V heads,
    /// `key_head_dim` 2 against `value_head_dim` 3. `qkv_dim` is
    /// `2·2 + 6 = 10` and the split offsets are 0 / 2 / 4.
    fn unequal_cfg() -> GdnConfig {
        GdnConfig {
            hidden_dim: 3,
            num_key_heads: 1,
            num_value_heads: 2,
            key_head_dim: 2,
            value_head_dim: 3,
            conv_kernel_size: 3,
            rms_norm_eps: 1e-5,
        }
    }

    fn unequal_raw(cfg: &GdnConfig) -> RawGdn {
        RawGdn {
            qkv: fill(cfg.qkv_dim() * cfg.hidden_dim, 1),
            gate: fill(cfg.value_dim() * cfg.hidden_dim, 2),
            conv: fill(cfg.qkv_dim() * cfg.conv_kernel_size, 3),
            dt: vec![0.1, -0.05],
            a: vec![-0.5, -0.75],
            beta: fill(cfg.num_value_heads * cfg.hidden_dim, 4),
            alpha: fill(cfg.num_value_heads * cfg.hidden_dim, 5),
            norm: vec![1.1, 0.9, 1.3],
            out: fill(cfg.hidden_dim * cfg.value_dim(), 6),
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
            out0.iter()
                .zip(out1.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "recurrent state should change the second token"
        );
    }

    #[test]
    fn softplus_matches_closed_form_at_zero() {
        assert!((softplus(0.0) - (2.0f32).ln()).abs() < 1e-6);
    }

    /// **The central test — it fails against the pre-generalization
    /// implementation.** The geometry is one K head / two V heads with
    /// `key_head_dim = 2` and `value_head_dim = 3`, so the fused
    /// projection is `2·2 + 2·3 = 10` wide and splits at `[2, 2, 6]`. The
    /// equal-head formula (`3 · num_v_heads · head_dim`) claims 12 wide
    /// splitting at `[4, 4, 4]`: it reads Q/K/V from offsets that belong
    /// to the neighbouring tensor, and no single `head_dim` rescues it,
    /// because K and V head dims genuinely differ here.
    ///
    /// Expected values come from [`reference_forward`] — a transcription
    /// of `gdn_reference.py:152` (the `[key_dim, key_dim, value_dim]`
    /// split) and `:163` (`repeat_interleave` of q/k up to `num_v_heads`)
    /// in the reference's own `[dk, dv]` state layout — **not** from this
    /// module's own output.
    #[test]
    fn unequal_head_geometry_matches_the_reference_split_and_replication() {
        let cfg = unequal_cfg();
        // Offsets straight from the reference formula, spelled out.
        assert_eq!(cfg.key_dim(), 2, "key_dim = num_key_heads * key_head_dim");
        assert_eq!(
            cfg.value_dim(),
            6,
            "value_dim = num_value_heads * value_head_dim"
        );
        assert_eq!(
            cfg.qkv_dim(),
            10,
            "conv_dim = 2*key_dim + value_dim; the equal-head formula would say 12"
        );
        assert_eq!(cfg.heads_per_key_group(), 2);

        let raw = unequal_raw(&cfg);
        let tokens = vec![
            vec![0.2f32, -0.1, 0.3],
            vec![-0.4f32, 0.25, 0.05],
            vec![0.15f32, 0.35, -0.2],
        ];
        let expected = reference_forward(&raw, &cfg, &tokens);

        let weights = raw.to_weights(&cfg);
        let mut state = GdnState::new(&cfg);
        for (t, token) in tokens.iter().enumerate() {
            let got = gdn_forward_token(&weights, &cfg, token, &mut state);
            assert_eq!(got.len(), cfg.hidden_dim);
            for (i, (g, e)) in got.iter().zip(expected[t].iter()).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-6 + 1e-5 * e.abs(),
                    "token {t} dim {i}: got {g}, reference {e}"
                );
            }
        }
    }

    /// Replicating one K head across `rep` V heads must be *exactly* the
    /// same computation as a checkpoint that stored those `rep` K rows
    /// duplicated on disk. If the replication indexed `h * key_head_dim`
    /// instead of `(h / rep) * key_head_dim`, this equivalence breaks —
    /// and for the one-K-head config it would read past the Q slice into
    /// K, which no shape check would catch.
    #[test]
    fn replicating_one_key_head_equals_a_checkpoint_with_duplicated_key_rows() {
        let shared = unequal_cfg(); // 1 K head → 2 V heads
        let mut duplicated = shared;
        duplicated.num_key_heads = 2; // same math, K/Q rows stored twice

        let raw_shared = unequal_raw(&shared);
        let hidden = shared.hidden_dim;
        let dk = shared.key_head_dim;
        let kernel = shared.conv_kernel_size;

        // Rebuild the fused projection (and its conv taps) with the single
        // K head's Q and K rows physically duplicated; V rows untouched.
        let mut qkv_dup = Vec::with_capacity(duplicated.qkv_dim() * hidden);
        let mut conv_dup = Vec::with_capacity(duplicated.qkv_dim() * kernel);
        for part in 0..2 {
            // Q block, then K block.
            let w_src = part * shared.key_dim() * hidden;
            let c_src = part * shared.key_dim() * kernel;
            for _ in 0..2 {
                qkv_dup.extend_from_slice(&raw_shared.qkv[w_src..w_src + dk * hidden]);
                conv_dup.extend_from_slice(&raw_shared.conv[c_src..c_src + dk * kernel]);
            }
        }
        qkv_dup.extend_from_slice(&raw_shared.qkv[2 * shared.key_dim() * hidden..]);
        conv_dup.extend_from_slice(&raw_shared.conv[2 * shared.key_dim() * kernel..]);

        let raw_dup = RawGdn {
            qkv: qkv_dup,
            conv: conv_dup,
            gate: raw_shared.gate.clone(),
            dt: raw_shared.dt.clone(),
            a: raw_shared.a.clone(),
            beta: raw_shared.beta.clone(),
            alpha: raw_shared.alpha.clone(),
            norm: raw_shared.norm.clone(),
            out: raw_shared.out.clone(),
        };

        let w_shared = raw_shared.to_weights(&shared);
        let w_dup = raw_dup.to_weights(&duplicated);
        let mut s_shared = GdnState::new(&shared);
        let mut s_dup = GdnState::new(&duplicated);
        for token in [
            vec![0.2f32, -0.1, 0.3],
            vec![-0.4f32, 0.25, 0.05],
            vec![0.15f32, 0.35, -0.2],
        ] {
            let a = gdn_forward_token(&w_shared, &shared, &token, &mut s_shared);
            let b = gdn_forward_token(&w_dup, &duplicated, &token, &mut s_dup);
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-6, "{a:?} vs {b:?}");
            }
        }
    }

    /// The regression this generalization must not break: with
    /// `num_key_heads == num_value_heads` and `key_head_dim ==
    /// value_head_dim` the layer must return **bit-identical** floats to
    /// the pre-generalization equal-head implementation. The constants are
    /// the raw `f32` bit patterns that implementation produced on
    /// [`make_weights`] / [`cfg`], so a reordered accumulation, a scale
    /// taken from the wrong head dim, or an altered state stride shows up
    /// here instead of silently shifting every equal-head checkpoint's
    /// output.
    #[test]
    fn equal_head_geometry_stays_bit_identical_to_the_pre_generalization_output() {
        const GOLDEN_STEP0: [u32; HIDDEN] = [1006633802, 998763940, 3163995192, 1006633802];
        const GOLDEN_STEP1: [u32; HIDDEN] = [1006151492, 1000425832, 3164698040, 1006151492];

        let weights = make_weights();
        let cfg = cfg();
        assert_eq!(cfg.qkv_dim(), QKV_DIM, "equal heads keep 3 * n_heads * dim");
        assert_eq!(cfg.value_dim(), V_DIM);
        assert_eq!(cfg.heads_per_key_group(), 1, "no replication when K == V");

        let mut state = GdnState::new(&cfg);
        let hidden = [0.2f32, -0.1, 0.3, -0.4];
        let out0 = gdn_forward_token(&weights, &cfg, &hidden, &mut state);
        let out1 = gdn_forward_token(&weights, &cfg, &hidden, &mut state);

        for (i, (got, want)) in out0.iter().zip(GOLDEN_STEP0.iter()).enumerate() {
            assert_eq!(got.to_bits(), *want, "step 0 dim {i}: {got}");
        }
        for (i, (got, want)) in out1.iter().zip(GOLDEN_STEP1.iter()).enumerate() {
            assert_eq!(got.to_bits(), *want, "step 1 dim {i}: {got}");
        }
    }

    /// The recurrent state is `[num_value_heads, value_head_dim,
    /// key_head_dim]`. A square `head_dim × head_dim` block would
    /// allocate 2·2·2 = 8 floats here instead of 2·3·2 = 12, and the
    /// second head's read-out would run off the end of the buffer.
    #[test]
    fn recurrent_state_is_rectangular_when_key_and_value_head_dims_differ() {
        let cfg = unequal_cfg();
        let state = GdnState::new(&cfg);
        assert_eq!(state.recurrent.len(), 2 * 3 * 2);
        assert!(state.recurrent.iter().all(|x| *x == 0.0));
    }

    #[test]
    #[should_panic(expected = "positive multiple")]
    fn value_heads_not_a_multiple_of_key_heads_is_rejected_not_floored() {
        let cfg = GdnConfig {
            hidden_dim: 3,
            num_key_heads: 3,
            num_value_heads: 4,
            key_head_dim: 2,
            value_head_dim: 2,
            conv_kernel_size: 2,
            rms_norm_eps: 1e-5,
        };
        let raw = RawGdn {
            qkv: fill(cfg.qkv_dim() * cfg.hidden_dim, 1),
            gate: fill(cfg.value_dim() * cfg.hidden_dim, 2),
            conv: fill(cfg.qkv_dim() * cfg.conv_kernel_size, 3),
            dt: vec![0.0; cfg.num_value_heads],
            a: vec![-0.5; cfg.num_value_heads],
            beta: fill(cfg.num_value_heads * cfg.hidden_dim, 4),
            alpha: fill(cfg.num_value_heads * cfg.hidden_dim, 5),
            norm: vec![1.0; cfg.value_head_dim],
            out: fill(cfg.hidden_dim * cfg.value_dim(), 6),
        };
        let weights = raw.to_weights(&cfg);
        let mut state = GdnState::new(&cfg);
        gdn_forward_token(&weights, &cfg, &[0.1, 0.2, 0.3], &mut state);
    }
}
