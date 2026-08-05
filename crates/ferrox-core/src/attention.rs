//! Rotary position embedding (RoPE, both the split-half `apply_rope`
//! and interleaved `apply_rope_interleaved` conventions) and
//! grouped-query causal attention (GQA). This is the "vanilla"
//! attention path used as the correctness baseline.
//! `causal_mla_attention`/`causal_mla_attention_sparse` add
//! DeepSeek-style latent attention and its DSA sparse-selection variant
//! (GLM-5.2, DeepSeek V3.2/V4); both mechanisms are now backed by real,
//! public reference implementations (see docs/MODELS.md).
//! `ferrox_models::mla`/`ferrox_models::glm_dsa` compose these
//! primitives into full RoPE-carrying MLA forward passes.

use crate::cache::PagedKvStore;

/// Applies rotary position embedding in place to a single head's vector,
/// split-half (GPT-NeoX / `LLAMA_ROPE_TYPE_NEOX`) style: each pair
/// `(i, i+half)` is rotated together, for position `pos` with base
/// `theta`. This is what llama.cpp calls NEOX-style RoPE (used by e.g.
/// DeepSeek-V3.2's lightning indexer); see [`apply_rope_interleaved`]
/// for the other real convention.
pub fn apply_rope(vec: &mut [f32], pos: usize, theta: f32) {
    let dim = vec.len();
    let half = dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = vec[i];
        let b = vec[i + half];
        vec[i] = a * cos - b * sin;
        vec[i + half] = a * sin + b * cos;
    }
}

/// Inverse of [`apply_rope`] (split-half / NeoX): rotates each pair by
/// `-angle`. DeepSeek V4 applies this ("derope" / `ggml_rope_ext_back`)
/// to the rope slice of attention output before the grouped `wo_a`
/// projection — see `.scratch/NOTES_DS4_INFERENCE.md`.
pub fn apply_rope_back(vec: &mut [f32], pos: usize, theta: f32) {
    let dim = vec.len();
    let half = dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = vec[i];
        let b = vec[i + half];
        // Inverse of (a,b) -> (a cos - b sin, a sin + b cos).
        vec[i] = a * cos + b * sin;
        vec[i + half] = -a * sin + b * cos;
    }
}

/// Inverse of [`apply_rope_interleaved`] (adjacent-pair / Norm RoPE).
pub fn apply_rope_interleaved_back(vec: &mut [f32], pos: usize, theta: f32) {
    let dim = vec.len();
    let half = dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = vec[2 * i];
        let b = vec[2 * i + 1];
        vec[2 * i] = a * cos + b * sin;
        vec[2 * i + 1] = -a * sin + b * cos;
    }
}

/// SIMD `q·k` for one attention head. NEON (`vfmaq_f32`) / AVX2+FMA
/// (`_mm256_fmadd_ps`) 4-/8-wide accumulation with a scalar tail, falling
/// back to a plain scalar sum elsewhere. The grouped accumulation
/// reassociates the sum, so results match the scalar dot only within
/// float noise -- which the online-softmax path already tolerates.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { dot_f32_neon(a, b) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc = vfmaq_f32(acc, va, vb);
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    // horizontal sum of the 8 lanes
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut s128 = _mm_add_ps(lo, hi);
    s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    s128 = _mm_add_ss(s128, _mm_shuffle_ps(s128, s128, 0x55));
    let mut sum = _mm_cvtss_f32(s128);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Online (flash-style) softmax·V accumulate for one head: one pass over
/// K/V, no `seq_len` score buffer. Numerically matches classic
/// max-subtract softmax within float noise (see unit tests).
///
/// When `attn_softcap` is `Some(sc)` with `sc > 0`, each score is remapped
/// with Gemma-2-style `sc * tanh(score / sc)` before the online softmax
/// (llama.cpp `attention.logit_softcapping`).
fn online_attn_accumulate(
    q_h: &[f32],
    scale: f32,
    head_dim: usize,
    out_h: &mut [f32],
    attn_softcap: Option<f32>,
    mut for_each_kv: impl FnMut(&mut dyn FnMut(&[f32], &[f32])),
) {
    debug_assert_eq!(q_h.len(), head_dim);
    debug_assert_eq!(out_h.len(), head_dim);
    let mut m = f32::NEG_INFINITY;
    let mut l = 0f32;
    out_h.fill(0.0);
    for_each_kv(&mut |k_t, v_t| {
        let mut s = dot_f32(q_h, k_t) * scale;
        if let Some(sc) = attn_softcap.filter(|&c| c > 0.0) {
            s = sc * (s / sc).tanh();
        }
        let m_new = m.max(s);
        let alpha = (m - m_new).exp();
        let p = (s - m_new).exp();
        l = l * alpha + p;
        for d in 0..head_dim {
            out_h[d] = out_h[d] * alpha + p * v_t[d];
        }
        m = m_new;
    });
    if l > 0.0 {
        let inv = 1.0 / l;
        for x in out_h.iter_mut() {
            *x *= inv;
        }
    }
}

/// Same split-half rotation as [`apply_rope`], but each frequency band
/// `i` has its angle divided by `freq_factors[i]` before the rotation --
/// Llama 3/3.1/3.2's real per-band RoPE frequency correction (the
/// `rope_freqs.weight` GGUF tensor, `n_rot/2` elements, `TENSOR_NOT_REQUIRED`
/// so most non-Llama-3 checkpoints don't carry it). Confirmed against
/// real llama.cpp source, not guessed: `ggml_rope_cache_init`
/// (`ggml/src/ggml-cpu/ops.cpp`) computes `theta/freq_factors[i0/2]`
/// per band before `rope_yarn`. `freq_factors` all-`1.0` is
/// mathematically identical to plain `apply_rope` (pinned by
/// `rope_with_all_ones_freq_factors_matches_plain_rope`).
///
/// Found via a real end-to-end run: serving a real
/// Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf checkpoint (this tensor
/// present and non-trivial) produced short answers correctly but
/// degenerated into a spurious early EOS a few dozen tokens into a
/// longer generation -- a real, independent llama.cpp oracle run
/// against the exact same file continued correctly with no early stop.
/// Root cause: every RoPE angle was computed without this per-band
/// correction, an error that compounds with position and eventually
/// produces wrong logits.
pub fn apply_rope_with_freq_factors(vec: &mut [f32], pos: usize, theta: f32, freq_factors: &[f32]) {
    let dim = vec.len();
    let half = dim / 2;
    assert_eq!(
        freq_factors.len(),
        half,
        "freq_factors must have one entry per rotation band (dim/2)"
    );
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        let angle = pos as f32 * freq / freq_factors[i];
        let (sin, cos) = angle.sin_cos();
        let a = vec[i];
        let b = vec[i + half];
        vec[i] = a * cos - b * sin;
        vec[i + half] = a * sin + b * cos;
    }
}

/// Applies rotary position embedding in place, interleaved (GPT-J /
/// llama.cpp's `LLAMA_ROPE_TYPE_NORM`) style: adjacent pairs
/// `(2*i, 2*i+1)` are rotated together, rather than `apply_rope`'s
/// split-half pairing. GLM-5.2 uses this convention for both its main
/// attention (`rope_interleave: true`) and its lightning indexer
/// (`indexer_rope_interleave: true`) per its real `config.json`
/// (`huggingface.co/zai-org/GLM-5.2`) — confirmed against llama.cpp PR
/// #25407, which rotates the indexer with `LLAMA_ROPE_TYPE_NORM` where
/// DeepSeek-V3.2's PR #23346 uses `LLAMA_ROPE_TYPE_NEOX`.
pub fn apply_rope_interleaved(vec: &mut [f32], pos: usize, theta: f32) {
    let dim = vec.len();
    let half = dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = vec[2 * i];
        let b = vec[2 * i + 1];
        vec[2 * i] = a * cos - b * sin;
        vec[2 * i + 1] = a * sin + b * cos;
    }
}

/// Interleaved (GPT-J / `LLAMA_ROPE_TYPE_NORM`) RoPE with Llama 3/3.1/3.2's
/// per-band frequency correction -- the combination real llama.cpp uses
/// for `general.architecture = "llama"` checkpoints that carry
/// `rope_freqs.weight`. Pairing is adjacent `(2*i, 2*i+1)` as in
/// [`apply_rope_interleaved`]; each band's angle is divided by
/// `freq_factors[i]` as in [`apply_rope_with_freq_factors`].
/// `freq_factors` all-`1.0` is mathematically identical to plain
/// `apply_rope_interleaved` (pinned by
/// `rope_interleaved_with_all_ones_freq_factors_matches_plain_interleaved`).
pub fn apply_rope_interleaved_with_freq_factors(
    vec: &mut [f32],
    pos: usize,
    theta: f32,
    freq_factors: &[f32],
) {
    let dim = vec.len();
    let half = dim / 2;
    assert_eq!(
        freq_factors.len(),
        half,
        "freq_factors must have one entry per rotation band (dim/2)"
    );
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        let angle = pos as f32 * freq / freq_factors[i];
        let (sin, cos) = angle.sin_cos();
        let a = vec[2 * i];
        let b = vec[2 * i + 1];
        vec[2 * i] = a * cos - b * sin;
        vec[2 * i + 1] = a * sin + b * cos;
    }
}

/// Single-token causal attention for one query against all cached
/// key/value positions (0..=pos), grouped-query style: `n_kv_heads` may
/// be fewer than `n_heads`, with each KV head shared by
/// `n_heads / n_kv_heads` query heads.
///
/// `q` is [n_heads, head_dim]; `k_cache`/`v_cache` are
/// [seq_len, n_kv_heads, head_dim] flattened row-major. Returns
/// [n_heads, head_dim].
pub fn causal_gqa_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    causal_gqa_attention_softcap(
        q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, None,
    )
}

/// [`causal_gqa_attention`] with optional Gemma-2 attention logit softcap.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_softcap(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    attn_softcap: Option<f32>,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(k_cache.len(), seq_len * n_kv_heads * head_dim);
    assert_eq!(v_cache.len(), seq_len * n_kv_heads * head_dim);

    let group_size = n_heads / n_kv_heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / group_size.max(1);
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        online_attn_accumulate(q_h, scale, head_dim, out_h, attn_softcap, |visit| {
            for t in 0..seq_len {
                let k_t = &k_cache
                    [(t * n_kv_heads + kv_h) * head_dim..(t * n_kv_heads + kv_h + 1) * head_dim];
                let v_t = &v_cache
                    [(t * n_kv_heads + kv_h) * head_dim..(t * n_kv_heads + kv_h + 1) * head_dim];
                visit(k_t, v_t);
            }
        });
    }

    out
}

/// Same computation as `causal_gqa_attention`, but each query only
/// attends to the last `window` cached positions (inclusive of itself)
/// instead of the full causal history -- Mistral/Mixtral/Qwen2-family
/// sliding-window attention. Confirmed against the real
/// `sliding_window` config field used by those models (real
/// `transformers` source for `Qwen2MoeAttention`/Mixtral's equivalent)
/// and against candle-transformers' `mixtral.rs`/`qwen2_moe.rs`, which
/// both mask scores where `key_pos + sliding_window < query_pos` --
/// i.e. only the most recent `window` positions (including the
/// query's own) stay unmasked. `window >= seq_len` degenerates to
/// exactly `causal_gqa_attention`'s full-causal behavior (pinned by
/// `windowed_attention_with_window_covering_full_history_matches_full_causal`).
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_windowed(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    window: usize,
) -> Vec<f32> {
    causal_gqa_attention_windowed_softcap(
        q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, window, None,
    )
}

/// [`causal_gqa_attention_windowed`] with optional attention logit softcap.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_windowed_softcap(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    window: usize,
    attn_softcap: Option<f32>,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(k_cache.len(), seq_len * n_kv_heads * head_dim);
    assert_eq!(v_cache.len(), seq_len * n_kv_heads * head_dim);
    assert!(window > 0, "window must be positive");

    let group_size = n_heads / n_kv_heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; n_heads * head_dim];
    // The current query is the last position in the cache (position
    // seq_len - 1); only the most recent `window` positions, including
    // this one, are visible.
    let window_start = seq_len.saturating_sub(window);

    for h in 0..n_heads {
        let kv_h = h / group_size.max(1);
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        online_attn_accumulate(q_h, scale, head_dim, out_h, attn_softcap, |visit| {
            for t in window_start..seq_len {
                let k_t = &k_cache
                    [(t * n_kv_heads + kv_h) * head_dim..(t * n_kv_heads + kv_h + 1) * head_dim];
                let v_t = &v_cache
                    [(t * n_kv_heads + kv_h) * head_dim..(t * n_kv_heads + kv_h + 1) * head_dim];
                visit(k_t, v_t);
            }
        });
    }

    out
}

/// Prefill (multi-query) causal GQA: `q`/`k_cache`/`v_cache` are all length
/// `seq_len` in the time dimension. Query at position `t` attends only to
/// keys/values `0..=t` (same math as looping [`causal_gqa_attention`] per
/// token). Layout: q/out `[seq_len, n_heads, head_dim]`; k/v
/// `[seq_len, n_kv_heads, head_dim]`. Metal prefill kernels must match.
pub fn causal_gqa_attention_prefill(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * n_heads * head_dim);
    assert_eq!(k_cache.len(), seq_len * n_kv_heads * head_dim);
    assert_eq!(v_cache.len(), seq_len * n_kv_heads * head_dim);

    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv_heads * head_dim;
    let mut out = vec![0f32; seq_len * q_stride];
    for t in 0..seq_len {
        let q_t = &q[t * q_stride..(t + 1) * q_stride];
        let k_prefix = &k_cache[..(t + 1) * kv_stride];
        let v_prefix = &v_cache[..(t + 1) * kv_stride];
        let attn = causal_gqa_attention(
            q_t,
            k_prefix,
            v_prefix,
            n_heads,
            n_kv_heads,
            head_dim,
            t + 1,
        );
        out[t * q_stride..(t + 1) * q_stride].copy_from_slice(&attn);
    }
    out
}

/// Same math as `causal_gqa_attention`, but K/V positions are read
/// through a `PagedKvStore` block table instead of one contiguous
/// slice: position `t` lives in block `block_table[t / block_size]`
/// at offset `t % block_size`, so blocks need not be physically
/// adjacent or in order. Must match `causal_gqa_attention` given the
/// same logical K/V contents (float noise only) — the block table is a
/// storage-layout detail, not a math change.
pub fn causal_gqa_attention_paged(
    q: &[f32],
    store: &PagedKvStore,
    block_table: &[usize],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * head_dim);
    let block_size = store.block_size();
    assert!(
        block_table.len() * block_size >= seq_len,
        "block table too short for seq_len"
    );

    let group_size = n_heads / n_kv_heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / group_size.max(1);
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        online_attn_accumulate(q_h, scale, head_dim, out_h, None, |visit| {
            for t in 0..seq_len {
                let block_id = block_table[t / block_size];
                let offset = t % block_size;
                let k_row = store.k_row(block_id, offset);
                let v_row = store.v_row(block_id, offset);
                let k_t = &k_row[kv_h * head_dim..(kv_h + 1) * head_dim];
                let v_t = &v_row[kv_h * head_dim..(kv_h + 1) * head_dim];
                visit(k_t, v_t);
            }
        });
    }

    out
}

/// Single-token causal attention for DeepSeek/Kimi-style Multi-head
/// Latent Attention (MLA): every query head has its own key/value (no
/// GQA-style grouping -- verified directly against Kimi K3's real
/// `KimiMLAAttention.forward`, where `kv_b_proj` expands to the full
/// `num_heads` count and the `num_key_value_heads`/`num_key_value_groups`
/// fields computed in `__init__` go unused), but the key/query head
/// dimension (`qk_head_dim` = `qk_nope_head_dim + qk_rope_head_dim`) can
/// differ from the value head dimension (`v_head_dim`) -- unlike
/// `causal_gqa_attention`, which assumes one shared `head_dim` for both.
///
/// `q` is [n_heads, qk_head_dim]; `k_cache` is [seq_len, n_heads,
/// qk_head_dim]; `v_cache` is [seq_len, n_heads, v_head_dim]. Returns
/// [n_heads, v_head_dim].
pub fn causal_mla_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * qk_head_dim);
    assert_eq!(k_cache.len(), seq_len * n_heads * qk_head_dim);
    assert_eq!(v_cache.len(), seq_len * n_heads * v_head_dim);

    let scale = 1.0 / (qk_head_dim as f32).sqrt();
    let mut out = vec![0f32; n_heads * v_head_dim];

    for h in 0..n_heads {
        let q_h = &q[h * qk_head_dim..(h + 1) * qk_head_dim];

        let mut scores = vec![0f32; seq_len];
        for t in 0..seq_len {
            let k_t =
                &k_cache[(t * n_heads + h) * qk_head_dim..(t * n_heads + h + 1) * qk_head_dim];
            let mut dot = 0f32;
            for d in 0..qk_head_dim {
                dot += q_h[d] * k_t[d];
            }
            scores[t] = dot * scale;
        }

        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }

        let out_h = &mut out[h * v_head_dim..(h + 1) * v_head_dim];
        for t in 0..seq_len {
            let v_t = &v_cache[(t * n_heads + h) * v_head_dim..(t * n_heads + h + 1) * v_head_dim];
            let w = scores[t];
            for d in 0..v_head_dim {
                out_h[d] += w * v_t[d];
            }
        }
    }

    out
}

/// The DeepSeek-V3.2 / GLM-5.2 "lightning indexer" (arXiv 2512.02556;
/// real, merged, tested reference implementations in llama.cpp PR
/// #23346 and PR #25407): scores every causally-visible key position
/// against the query using a cheap multi-head dot-product indexer, then
/// keeps only the `top_k` highest-scoring positions.
///
/// `indexer_q` is `[n_index_heads][index_head_dim]` for this query
/// position; `indexer_keys` is `[num_causal_positions][index_head_dim]`
/// (one MQA key per causal position, `0..=query_pos`); `indexer_weights`
/// is `[n_index_heads]`. Returns the kept key positions, ascending.
///
/// The real implementation additionally rotates `indexer_q`/`indexer_k`
/// through a fixed orthogonal Hadamard matrix before the dot product (to
/// spread values evenly for FP8 quantization on real hardware). An
/// orthogonal transform applied identically to both operands leaves
/// their dot product unchanged in exact arithmetic
/// (`(Hq)·(Hk) = q^T H^T H k = q^T k`), so this f32 CPU path omits it —
/// the score computed here is exact, not an approximation of the real
/// one.
pub fn lightning_indexer_topk(
    indexer_q: &[Vec<f32>],
    indexer_keys: &[Vec<f32>],
    indexer_weights: &[f32],
    top_k: usize,
) -> Vec<usize> {
    let n_heads = indexer_q.len();
    assert_eq!(indexer_weights.len(), n_heads);
    let index_head_dim = indexer_q.first().map_or(0, |q| q.len());
    let scale = 1.0 / ((index_head_dim * n_heads) as f32).sqrt();

    let mut scored: Vec<(usize, f32)> = indexer_keys
        .iter()
        .enumerate()
        .map(|(j, k)| {
            let score: f32 = indexer_q
                .iter()
                .zip(indexer_weights.iter())
                .map(|(q, w)| {
                    let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                    dot.max(0.0) * w * scale
                })
                .sum();
            (j, score)
        })
        .collect();

    let keep = top_k.min(scored.len());
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut kept: Vec<usize> = scored.into_iter().take(keep).map(|(j, _)| j).collect();
    kept.sort_unstable();
    kept
}

/// Same as [`causal_mla_attention`] for a single query position, but
/// attention is restricted to the explicit `visible` key positions
/// (ascending, a subset of `0..seq_len`) rather than the full causal
/// history — the sparse-attention half of GLM-5.2/DeepSeek-V3.2's DSA,
/// applied after [`lightning_indexer_topk`] selects `visible`.
#[allow(clippy::too_many_arguments)]
pub fn causal_mla_attention_sparse(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
    visible: &[usize],
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * qk_head_dim);
    assert_eq!(k_cache.len(), seq_len * n_heads * qk_head_dim);
    assert_eq!(v_cache.len(), seq_len * n_heads * v_head_dim);
    assert!(
        visible.iter().all(|&t| t < seq_len),
        "visible positions must be within seq_len"
    );

    let scale = 1.0 / (qk_head_dim as f32).sqrt();
    let mut out = vec![0f32; n_heads * v_head_dim];

    for h in 0..n_heads {
        let q_h = &q[h * qk_head_dim..(h + 1) * qk_head_dim];

        let mut scores = vec![0f32; visible.len()];
        for (i, &t) in visible.iter().enumerate() {
            let k_t =
                &k_cache[(t * n_heads + h) * qk_head_dim..(t * n_heads + h + 1) * qk_head_dim];
            let mut dot = 0f32;
            for d in 0..qk_head_dim {
                dot += q_h[d] * k_t[d];
            }
            scores[i] = dot * scale;
        }

        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }

        let out_h = &mut out[h * v_head_dim..(h + 1) * v_head_dim];
        for (i, &t) in visible.iter().enumerate() {
            let v_t = &v_cache[(t * n_heads + h) * v_head_dim..(t * n_heads + h + 1) * v_head_dim];
            let w = scores[i];
            for d in 0..v_head_dim {
                out_h[d] += w * v_t[d];
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_dot_f32_matches_scalar_across_lengths() {
        // Cover exact-multiple and tail lengths around the SIMD width.
        for n in [1usize, 3, 4, 7, 8, 15, 16, 63, 128, 129] {
            let a: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.31 - 2.0).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.17 + 1.0).cos()).collect();
            let simd = dot_f32(&a, &b);
            let scalar: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!(
                (simd - scalar).abs() <= 1e-4 * scalar.abs().max(1.0),
                "n={n} simd={simd} scalar={scalar}"
            );
        }
    }

    #[test]
    fn rope_preserves_vector_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope(&mut v, 5, 10000.0);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE is a rotation and must preserve norm"
        );
    }

    #[test]
    fn rope_back_inverts_rope() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut v = original.clone();
        apply_rope(&mut v, 11, 10000.0);
        apply_rope_back(&mut v, 11, 10000.0);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_interleaved_back_inverts_interleaved() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut v = original.clone();
        apply_rope_interleaved(&mut v, 11, 10000.0);
        apply_rope_interleaved_back(&mut v, 11, 10000.0);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let original = v.clone();
        apply_rope(&mut v, 0, 10000.0);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_with_all_ones_freq_factors_matches_plain_rope() {
        let mut with_factors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut plain = with_factors.clone();
        let ones = vec![1.0; 3];
        apply_rope_with_freq_factors(&mut with_factors, 7, 10000.0, &ones);
        apply_rope(&mut plain, 7, 10000.0);
        for (a, b) in with_factors.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_with_freq_factors_diverges_from_plain_rope_when_factors_are_not_one() {
        let mut with_factors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut plain = with_factors.clone();
        let factors = vec![0.5, 2.0, 1.0];
        apply_rope_with_freq_factors(&mut with_factors, 7, 10000.0, &factors);
        apply_rope(&mut plain, 7, 10000.0);
        let differs = with_factors
            .iter()
            .zip(plain.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "non-1.0 freq_factors must change the rotation");
    }

    #[test]
    fn rope_with_freq_factors_preserves_vector_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_with_freq_factors(&mut v, 5, 10000.0, &[0.8, 1.3]);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-4);
    }

    #[test]
    fn rope_interleaved_preserves_vector_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_interleaved(&mut v, 5, 10000.0);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE is a rotation and must preserve norm"
        );
    }

    #[test]
    fn rope_interleaved_at_position_zero_is_identity() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let original = v.clone();
        apply_rope_interleaved(&mut v, 0, 10000.0);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_interleaved_rotates_adjacent_pairs_not_split_halves() {
        // With a single frequency band (dim=2), interleaved and split-half
        // RoPE are mathematically identical (both rotate the one (v[0],
        // v[1]) pair). The two conventions only diverge once dim > 2 and
        // there's more than one frequency band to route pairs into --
        // that's the real bug class this test guards against: mixing up
        // which components get paired together.
        let mut interleaved = vec![1.0, 0.0, 0.0, 1.0];
        let mut split_half = interleaved.clone();
        apply_rope_interleaved(&mut interleaved, 3, 10000.0);
        apply_rope(&mut split_half, 3, 10000.0);
        // Different frequency assigned to each pair in the two
        // conventions (interleaved pairs (0,1)+(2,3), split-half pairs
        // (0,2)+(1,3)) so with two distinct frequency bands the outputs
        // must differ.
        let differs = interleaved
            .iter()
            .zip(split_half.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "the two RoPE conventions must not coincide here");
    }

    #[test]
    fn rope_interleaved_with_all_ones_freq_factors_matches_plain_interleaved() {
        let mut with_factors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut plain = with_factors.clone();
        let ones = vec![1.0; 3];
        apply_rope_interleaved_with_freq_factors(&mut with_factors, 7, 10000.0, &ones);
        apply_rope_interleaved(&mut plain, 7, 10000.0);
        for (a, b) in with_factors.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_interleaved_with_freq_factors_diverges_when_factors_are_not_one() {
        let mut with_factors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut plain = with_factors.clone();
        let factors = vec![0.5, 2.0, 1.0];
        apply_rope_interleaved_with_freq_factors(&mut with_factors, 7, 10000.0, &factors);
        apply_rope_interleaved(&mut plain, 7, 10000.0);
        let differs = with_factors
            .iter()
            .zip(plain.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "non-1.0 freq_factors must change the rotation");
    }

    #[test]
    fn rope_interleaved_with_freq_factors_preserves_vector_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_interleaved_with_freq_factors(&mut v, 5, 10000.0, &[0.8, 1.3]);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-4);
    }

    #[test]
    fn attention_with_single_position_returns_that_value() {
        // With one cached position, attention weight is trivially 1.0,
        // so output must equal that single V vector regardless of Q/K.
        let q = vec![1.0, 0.0]; // 1 head, head_dim=2
        let k_cache = vec![0.5, 0.5]; // seq_len=1, 1 kv head
        let v_cache = vec![9.0, -3.0];
        let out = causal_gqa_attention(&q, &k_cache, &v_cache, 1, 1, 2, 1);
        assert!((out[0] - 9.0).abs() < 1e-4);
        assert!((out[1] - (-3.0)).abs() < 1e-4);
    }

    #[test]
    fn gqa_group_mapping_shares_kv_heads_correctly() {
        // 4 query heads, 2 kv heads -> heads 0,1 use kv head 0; heads 2,3 use kv head 1.
        let head_dim = 2;
        let q = vec![
            1.0, 0.0, // head 0
            1.0, 0.0, // head 1
            1.0, 0.0, // head 2
            1.0, 0.0, // head 3
        ];
        // seq_len = 1, 2 kv heads
        let k_cache = vec![1.0, 0.0, 1.0, 0.0];
        let v_cache = vec![100.0, 100.0, 200.0, 200.0];
        let out = causal_gqa_attention(&q, &k_cache, &v_cache, 4, 2, head_dim, 1);
        // heads 0,1 -> kv head 0 -> v = [100,100]; heads 2,3 -> kv head 1 -> v=[200,200]
        assert_eq!(&out[0..2], &[100.0, 100.0][..]);
        assert_eq!(&out[2..4], &[100.0, 100.0][..]);
        assert_eq!(&out[4..6], &[200.0, 200.0][..]);
        assert_eq!(&out[6..8], &[200.0, 200.0][..]);
    }

    #[test]
    fn prefill_gqa_matches_per_token_causal() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 5;
        let q: Vec<f32> = (0..seq_len * n_heads * head_dim)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.19).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let batched =
            causal_gqa_attention_prefill(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);
        let q_stride = n_heads * head_dim;
        let kv_stride = n_kv_heads * head_dim;
        for t in 0..seq_len {
            let expect = causal_gqa_attention(
                &q[t * q_stride..(t + 1) * q_stride],
                &k[..(t + 1) * kv_stride],
                &v[..(t + 1) * kv_stride],
                n_heads,
                n_kv_heads,
                head_dim,
                t + 1,
            );
            let got = &batched[t * q_stride..(t + 1) * q_stride];
            for (a, b) in got.iter().zip(expect.iter()) {
                assert!((a - b).abs() < 1e-5, "t={t}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn windowed_attention_with_window_covering_full_history_matches_full_causal() {
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 3;
        let seq_len = 4;

        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let k_cache: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.17).cos())
            .collect();
        let v_cache: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();

        let full = causal_gqa_attention(
            &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, seq_len,
        );
        let windowed = causal_gqa_attention_windowed(
            &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, seq_len, seq_len,
        );
        assert_eq!(full.len(), windowed.len());
        for (a, b) in full.iter().zip(windowed.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "window >= seq_len must be bit-identical to full causal"
            );
        }
    }

    #[test]
    fn windowed_attention_ignores_positions_outside_the_window() {
        // 1 head, seq_len=3, window=1: only the current position (t=2)
        // should ever be attended to, so the output must equal exactly
        // that position's V vector regardless of Q/K -- masking every
        // earlier position out means there is exactly one candidate
        // left, and softmax over one candidate is trivially 1.0.
        let head_dim = 2;
        let q = vec![1.0, 0.0];
        let k_cache = vec![9.0, -9.0, 0.5, 0.5, -3.0, 7.0]; // seq_len=3, 1 kv head
        let v_cache = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let out = causal_gqa_attention_windowed(&q, &k_cache, &v_cache, 1, 1, head_dim, 3, 1);
        assert!((out[0] - 50.0).abs() < 1e-4);
        assert!((out[1] - 60.0).abs() < 1e-4);
    }

    #[test]
    fn paged_attention_matches_contiguous_attention_bit_identical() {
        use crate::cache::{PagedKvCache, PagedKvStore};

        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 3;
        let block_size = 2;
        let seq_len = 5;

        // Deterministic pseudo-random-ish K/V/Q values, no RNG needed.
        let k_flat: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| ((i * 7 + 1) % 13) as f32 * 0.1)
            .collect();
        let v_flat: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| ((i * 5 + 3) % 11) as f32 * 0.1)
            .collect();
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i * 3 + 2) % 9) as f32 * 0.1)
            .collect();

        let contiguous =
            causal_gqa_attention(&q, &k_flat, &v_flat, n_heads, n_kv_heads, head_dim, seq_len);

        let mut store = PagedKvStore::new(block_size, seq_len, n_kv_heads, head_dim);
        let mut cache = PagedKvCache::new();
        for t in 0..seq_len {
            let start = t * n_kv_heads * head_dim;
            let end = start + n_kv_heads * head_dim;
            cache
                .push(&mut store, &k_flat[start..end], &v_flat[start..end])
                .expect("store sized for seq_len blocks, must not exhaust");
        }

        let paged = causal_gqa_attention_paged(
            &q,
            &store,
            cache.block_table(),
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
        );

        assert_eq!(contiguous.len(), paged.len());
        for (a, b) in contiguous.iter().zip(paged.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "paged path must be bit-identical");
        }
    }

    #[test]
    fn mla_attention_with_single_position_returns_that_value() {
        // Same reasoning as `attention_with_single_position_returns_that_value`,
        // but with distinct qk/v head dims (5 vs 3) to exercise the one real
        // difference from `causal_gqa_attention`.
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0]; // 1 head, qk_head_dim=5
        let k_cache = vec![0.2, 0.2, 0.2, 0.2, 0.2]; // seq_len=1
        let v_cache = vec![9.0, -3.0, 1.0]; // v_head_dim=3
        let out = causal_mla_attention(&q, &k_cache, &v_cache, 1, 5, 3, 1);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 9.0).abs() < 1e-4);
        assert!((out[1] - (-3.0)).abs() < 1e-4);
        assert!((out[2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn mla_attention_every_head_gets_its_own_kv_no_grouping() {
        // Unlike GQA, MLA has no shared-kv-head grouping: with 2 heads and
        // 2 cached kv-head-slots, head 0 must only ever see kv slot 0 and
        // head 1 only kv slot 1.
        let qk_head_dim = 2;
        let v_head_dim = 2;
        let q = vec![1.0, 0.0, 1.0, 0.0]; // 2 heads
        let k_cache = vec![1.0, 0.0, 1.0, 0.0]; // seq_len=1, 2 heads
        let v_cache = vec![100.0, 100.0, 200.0, 200.0];
        let out = causal_mla_attention(&q, &k_cache, &v_cache, 2, qk_head_dim, v_head_dim, 1);
        assert_eq!(&out[0..2], &[100.0, 100.0][..]);
        assert_eq!(&out[2..4], &[200.0, 200.0][..]);
    }

    #[test]
    fn lightning_indexer_topk_keeps_all_positions_when_top_k_covers_them() {
        let indexer_q = vec![vec![1.0, 0.0]];
        let indexer_keys = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.1, 0.9]];
        let indexer_weights = vec![1.0];
        let kept = lightning_indexer_topk(&indexer_q, &indexer_keys, &indexer_weights, 10);
        assert_eq!(kept, vec![0, 1, 2]);
    }

    #[test]
    fn lightning_indexer_topk_selects_highest_scoring_positions() {
        // Query aligned with key 0 (dot=1.0), partially with key 2 (dot=0.9),
        // orthogonal to key 1 (dot=0.0, relu'd score 0). Top-2 must be {0, 2}.
        let indexer_q = vec![vec![1.0, 0.0]];
        let indexer_keys = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.9, 0.1]];
        let indexer_weights = vec![1.0];
        let kept = lightning_indexer_topk(&indexer_q, &indexer_keys, &indexer_weights, 2);
        assert_eq!(kept, vec![0, 2]);
    }

    #[test]
    fn lightning_indexer_topk_relu_zeroes_negative_dot_products() {
        // Key 1's dot product with the query is negative; ReLU floors its
        // score at 0, so it must lose to key 0 (positive) even at top_k=1.
        let indexer_q = vec![vec![1.0, 0.0]];
        let indexer_keys = vec![vec![0.3, 0.0], vec![-1.0, 0.0]];
        let indexer_weights = vec![1.0];
        let kept = lightning_indexer_topk(&indexer_q, &indexer_keys, &indexer_weights, 1);
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn mla_attention_sparse_with_all_positions_visible_matches_full_causal() {
        let qk_head_dim = 3;
        let v_head_dim = 2;
        let seq_len = 4;
        let n_heads = 2;
        let q: Vec<f32> = (0..n_heads * qk_head_dim).map(|i| i as f32 * 0.1).collect();
        let k_cache: Vec<f32> = (0..seq_len * n_heads * qk_head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let v_cache: Vec<f32> = (0..seq_len * n_heads * v_head_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();

        let full = causal_mla_attention(
            &q,
            &k_cache,
            &v_cache,
            n_heads,
            qk_head_dim,
            v_head_dim,
            seq_len,
        );
        let visible: Vec<usize> = (0..seq_len).collect();
        let sparse = causal_mla_attention_sparse(
            &q,
            &k_cache,
            &v_cache,
            n_heads,
            qk_head_dim,
            v_head_dim,
            seq_len,
            &visible,
        );

        assert_eq!(full.len(), sparse.len());
        for (a, b) in full.iter().zip(sparse.iter()) {
            assert!((a - b).abs() < 1e-6, "full={a} sparse={b}");
        }
    }

    #[test]
    fn mla_attention_sparse_ignores_positions_outside_visible_set() {
        // Only position 0 is visible; a wildly different value at position 1
        // must have zero influence on the output.
        let qk_head_dim = 2;
        let v_head_dim = 1;
        let q = vec![1.0, 0.0];
        let k_cache = vec![1.0, 0.0, 1.0, 0.0]; // seq_len=2, identical keys
        let v_cache = vec![5.0, 999.0]; // position 0 -> 5.0, position 1 -> 999.0
        let out = causal_mla_attention_sparse(
            &q,
            &k_cache,
            &v_cache,
            1,
            qk_head_dim,
            v_head_dim,
            2,
            &[0],
        );
        assert_eq!(out.len(), 1);
        assert!((out[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn attn_logit_softcap_changes_output_vs_uncapped() {
        // Softcap must change the attended output relative to the uncapped
        // path (and must not be a no-op identity for large scores).
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let seq_len = 3;
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 + 1.0) * 2.5)
            .collect();
        let k: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.7).sin() * 3.0)
            .collect();
        let v: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.3).cos())
            .collect();
        let plain = causal_gqa_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);
        let capped = causal_gqa_attention_softcap(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            Some(30.0),
        );
        assert_eq!(plain.len(), capped.len());
        let differs = plain
            .iter()
            .zip(capped.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(differs, "softcap must change attention output");
        // Softcap None / <=0 must match the uncapped path.
        let none =
            causal_gqa_attention_softcap(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, None);
        for (a, b) in plain.iter().zip(none.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
