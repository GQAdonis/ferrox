//! The Mamba-2 selective-state-space step, as ggml computes it.
//!
//! Two kernels and a state. `ggml_ssm_conv` (`ggml-cpu/ops.cpp:9557-9608`)
//! is a causal depthwise convolution of width `d_conv` over the
//! projected `xBC` rows, with the previous `d_conv - 1` rows as the
//! state; `ggml_ssm_scan` (`:9627-9850`, the `src3->ne[0] == 1` arm) is
//! the per-head recurrence
//!
//! ```text
//! dt'      = softplus(dt_h)                     (ggml-impl.h:107-109)
//! dA       = exp(dt' * A_h)                     one scalar per head
//! S[h,d,:] = S[h,d,:] * dA + B[g,:] * (x[h,d] * dt')
//! y[h,d]   = S[h,d,:] . C[g,:]
//! ```
//!
//! with `g = h / (n_head / n_group)` (`repeat_interleave`) and a float
//! accumulator. The state is `[n_head][head_dim][d_state]` with the
//! state index fastest, ggml's `{d_state, head_dim, n_head}`.
//!
//! This file is the arithmetic only: no weights, no norms, no
//! projections. `ferrox_models::mamba2` owns those and the residual
//! topology; [`crate::recurrent_state::RecurrentState`] owns the two
//! buffers between tokens.

/// `log(1 + exp(x))`, in ggml's precision (`ggml_compute_softplus_f32`).
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// One token of the causal conv: `out[c] = sum_i taps[c][i] * window[i][c]`,
/// where `window` is the previous `d_conv - 1` rows (the state, oldest
/// first) followed by this token's `x`, and the state is then shifted
/// by one row with `x` appended.
///
/// `state` is `[(d_conv - 1)][width]`, `taps` is `[width][d_conv]`
/// (ggml `{d_conv, width}`: channel `c`'s taps contiguous, oldest input
/// on tap 0), `x` and `out` are `[width]`.
pub fn conv_step(state: &mut [f32], taps: &[f32], d_conv: usize, x: &[f32], out: &mut [f32]) {
    let width = x.len();
    assert_eq!(state.len(), (d_conv - 1) * width);
    assert_eq!(taps.len(), width * d_conv);
    assert_eq!(out.len(), width);
    for c in 0..width {
        let t = &taps[c * d_conv..(c + 1) * d_conv];
        // ops.cpp:9598-9603: a float accumulator over the window in
        // order, the newest input last.
        let mut acc = 0.0f32;
        for (i, tap) in t.iter().enumerate().take(d_conv - 1) {
            acc += state[i * width + c] * tap;
        }
        acc += x[c] * t[d_conv - 1];
        out[c] = acc;
    }
    // Shift: drop the oldest row, append `x`.
    if d_conv > 1 {
        state.copy_within(width.., 0);
        state[(d_conv - 2) * width..].copy_from_slice(x);
    }
}

/// The geometry one scan step needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanDims {
    pub n_head: usize,
    pub head_dim: usize,
    pub d_state: usize,
    pub n_group: usize,
}

impl ScanDims {
    /// Floats in one sequence's SSM state.
    pub fn state_len(self) -> usize {
        self.n_head * self.head_dim * self.d_state
    }
}

/// One token of the Mamba-2 scan, in place on `state`
/// (`[n_head][head_dim][d_state]`).
///
/// `x` is `[n_head][head_dim]`, `dt` is `[n_head]` (BEFORE softplus,
/// the bias already added), `a` is `[n_head]` (the stored `ssm_a`,
/// already negative), `b` and `c` are `[n_group][d_state]`, `y` is
/// `[n_head][head_dim]`.
#[allow(clippy::too_many_arguments)] // the seven operands ggml_ssm_scan takes, plus the dims
pub fn scan_step(
    dims: ScanDims,
    state: &mut [f32],
    x: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    y: &mut [f32],
) {
    let ScanDims {
        n_head,
        head_dim,
        d_state,
        n_group,
    } = dims;
    assert_eq!(state.len(), dims.state_len());
    assert_eq!(x.len(), n_head * head_dim);
    assert_eq!(dt.len(), n_head);
    assert_eq!(a.len(), n_head);
    assert_eq!(b.len(), n_group * d_state);
    assert_eq!(c.len(), n_group * d_state);
    assert_eq!(y.len(), n_head * head_dim);
    assert_eq!(n_head % n_group, 0, "ops.cpp:9659");
    let heads_per_group = n_head / n_group;
    for h in 0..n_head {
        let dt_sp = softplus(dt[h]);
        let da = (dt_sp * a[h]).exp();
        let g = h / heads_per_group;
        let (bg, cg) = (
            &b[g * d_state..(g + 1) * d_state],
            &c[g * d_state..(g + 1) * d_state],
        );
        for d in 0..head_dim {
            let ii = h * head_dim + d;
            let x_dt = x[ii] * dt_sp;
            let s = &mut state[ii * d_state..(ii + 1) * d_state];
            let mut sum = 0.0f32;
            for k in 0..d_state {
                let v = s[k] * da + bg[k] * x_dt;
                sum += v * cg[k];
                s[k] = v;
            }
            y[ii] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_is_ggml_s() {
        assert!((softplus(0.0) - 2f32.ln()).abs() < 1e-7);
        assert_eq!(softplus(25.0), 25.0);
        assert!((softplus(-30.0)).abs() < 1e-6);
    }

    /// Two channels, taps `[1, 2, 3]` and `[0, 0, 1]`: the newest input
    /// on the LAST tap, the state shifting one row per step.
    #[test]
    fn conv_step_puts_the_newest_input_on_the_last_tap_and_shifts() {
        let taps = [1.0, 2.0, 3.0, 0.0, 0.0, 1.0];
        let mut state = vec![0.0f32; 2 * 2];
        let mut out = [0.0f32; 2];
        conv_step(&mut state, &taps, 3, &[1.0, 5.0], &mut out);
        assert_eq!(out, [3.0, 5.0]);
        assert_eq!(state, vec![0.0, 0.0, 1.0, 5.0]);
        conv_step(&mut state, &taps, 3, &[2.0, 6.0], &mut out);
        assert_eq!(out, [3.0 * 2.0 + 2.0 * 1.0, 6.0]);
        conv_step(&mut state, &taps, 3, &[1.0, 7.0], &mut out);
        assert_eq!(out, [3.0 + 4.0 + 1.0, 7.0]);
        assert_eq!(state, vec![2.0, 6.0, 1.0, 7.0]);
    }

    /// One head, one channel, one state: the recurrence by hand.
    #[test]
    fn scan_step_is_the_recurrence() {
        let dims = ScanDims {
            n_head: 1,
            head_dim: 1,
            d_state: 2,
            n_group: 1,
        };
        let mut state = vec![0.0f32; 2];
        let mut y = [0.0f32];
        let a = [-1.0f32];
        // dt = 0 -> softplus = ln 2; dA = exp(-ln 2) = 0.5.
        scan_step(
            dims,
            &mut state,
            &[2.0],
            &[0.0],
            &a,
            &[1.0, 3.0],
            &[1.0, 1.0],
            &mut y,
        );
        let x_dt = 2.0 * 2f32.ln();
        assert!((state[0] - x_dt).abs() < 1e-6 && (state[1] - 3.0 * x_dt).abs() < 1e-6);
        assert!((y[0] - 4.0 * x_dt).abs() < 1e-5);
        scan_step(
            dims,
            &mut state,
            &[0.0],
            &[0.0],
            &a,
            &[1.0, 3.0],
            &[1.0, 0.0],
            &mut y,
        );
        assert!((state[0] - 0.5 * x_dt).abs() < 1e-6, "decayed by dA");
        assert!((y[0] - 0.5 * x_dt).abs() < 1e-6, "C selects state 0");
    }

    /// Groups: head `h` reads B/C group `h / (n_head / n_group)`.
    #[test]
    fn heads_read_their_group_s_b_and_c() {
        let dims = ScanDims {
            n_head: 4,
            head_dim: 1,
            d_state: 1,
            n_group: 2,
        };
        let mut state = vec![0.0f32; 4];
        let mut y = [0.0f32; 4];
        scan_step(
            dims,
            &mut state,
            &[1.0; 4],
            &[30.0; 4],
            &[0.0; 4],
            &[1.0, 10.0],
            &[1.0, 1.0],
            &mut y,
        );
        // softplus(30) = 30, dA = 1: state = B[g] * 30.
        assert_eq!(y, [30.0, 30.0, 300.0, 300.0]);
    }
}
