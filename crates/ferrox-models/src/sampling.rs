//! Token sampling from a decoder's output logits: temperature, top-k,
//! top-p (nucleus), and repetition penalty, on top of the greedy argmax
//! ferrox previously always used unconditionally.
//!
//! `crate::speculative` verifies draft tokens against
//! [`sampling_distribution`] -- the exact distribution [`Sampler`]
//! draws from for a given `SamplingParams` -- so speculation is
//! lossless with respect to whatever sampling configuration the caller
//! asked for, rather than only at temperature 0.
//!
//! No external `rand` dependency: a small xorshift64* generator (the
//! same algorithm `Decoder::new_random_small`'s test-only `Lcg` already
//! uses in `decoder.rs`) is enough for sampling and keeps the
//! dependency tree the same minimal, pure-Rust shape as the rest of
//! this crate.

/// Sampling parameters for one generation request. `temperature <= 0.0`
/// means "sample nothing, take the greedy argmax" -- the same
/// deterministic behavior ferrox always had before this module existed.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    /// Nucleus sampling threshold in (0.0, 1.0]. 1.0 disables top-p
    /// filtering (every token with nonzero probability is eligible).
    pub top_p: f32,
    /// Keep only the `top_k` highest-probability tokens before
    /// sampling. 0 disables top-k filtering.
    pub top_k: usize,
    /// > 1.0 discourages repeating a token already in `history`; 1.0
    /// > disables repetition penalty. Uses the standard convention
    /// > (divide positive logits, multiply negative ones) so the penalty
    /// > always pushes toward *less* likely, regardless of logit sign.
    pub repetition_penalty: f32,
    /// OpenAI-style presence penalty: subtract from logits of tokens
    /// that already appeared in `history` (once per distinct token).
    pub presence_penalty: f32,
    /// OpenAI-style frequency penalty: subtract `frequency_penalty *
    /// count` from logits for each token id seen in `history`.
    pub frequency_penalty: f32,
}

impl Default for SamplingParams {
    /// Greedy decoding: identical behavior to ferrox's original
    /// argmax-only generation loop.
    fn default() -> Self {
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

/// Zeroes out logits a caller wants to forbid, in place, before the
/// sampler looks at them. Used by JSON-object mode to keep generation
/// inside the grammar.
pub type LogitMask<'a> = &'a mut dyn FnMut(&mut [f32]);

/// A small, seedable xorshift64* generator. Not cryptographically
/// secure -- sampling doesn't need that -- but reproducible given a
/// seed, which greedy argmax already was for free.
pub struct Sampler {
    state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        // xorshift64* requires a nonzero seed.
        Sampler {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        // The `*` in xorshift64*. Without it this is plain xorshift64,
        // whose state IS its output, and a small seed's first output is
        // therefore still small: for every seed below ~4000 the first
        // draw landed in the bottom eighth of [0, 1), so a request that
        // asked for `seed: 42` always got its first token from the
        // bottom of the CDF. The multiply is what decorrelates the
        // output from a low-entropy state; see
        // `low_seeds_do_not_bias_the_first_draw`.
        self.state.wrapping_mul(0x2545F491_4F6CDD1D)
    }

    /// Uniform float in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Samples one token id from `logits`, given `params` and the
    /// already-generated `history` (for repetition penalty). Falls back
    /// to plain greedy argmax when `params.temperature <= 0.0`.
    ///
    /// A length-1 `logits` vector is treated as a precomputed greedy token
    /// id (`logits[0] as usize`) — used by the Metal dense-stack path that
    /// returns GPU argmax instead of downloading the full vocab.
    pub fn sample(&mut self, logits: &[f32], params: &SamplingParams, history: &[usize]) -> usize {
        self.sample_with_mask(logits, params, history, None)
    }

    /// Like [`Self::sample`], but optionally zeroes disallowed logits via
    /// `mask` before argmax / nucleus sampling (used for JSON-object mode).
    pub fn sample_with_mask(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        history: &[usize],
        mut mask: Option<LogitMask<'_>>,
    ) -> usize {
        if params.temperature <= 0.0 && mask.is_none() {
            if logits.len() == 1 {
                return logits[0] as usize;
            }
            let mut scores = logits.to_vec();
            apply_history_penalties(&mut scores, params, history);
            return argmax(&scores);
        }

        let mut scores: Vec<f32> = logits.to_vec();
        apply_history_penalties(&mut scores, params, history);

        if let Some(m) = mask.as_mut() {
            m(&mut scores);
        }

        if params.temperature <= 0.0 {
            if scores.len() == 1 {
                return scores[0] as usize;
            }
            return argmax(&scores);
        }

        let probs = filtered_distribution(scores, logits, params);
        self.sample_from(&probs)
    }

    /// A uniform draw in `[0.0, 1.0)`.
    ///
    /// Exposed because speculative decoding's accept test is a coin
    /// flip against `p_target(x) / p_draft(x)` rather than a draw from
    /// a distribution, and it must come off the same seeded stream as
    /// every other draw in the run or a "reproducible given a seed"
    /// generation stops being reproducible.
    pub fn uniform(&mut self) -> f32 {
        self.next_f32()
    }

    /// Draws one index from an already-normalised distribution.
    ///
    /// Split out of [`Self::sample_with_mask`] so speculative decoding
    /// can sample from a distribution it had to compute anyway (the
    /// rejection rule needs `p_target` itself, not just a draw from it)
    /// and still go through *exactly* the same draw as ordinary
    /// sampling. Two separate copies of this loop would be two chances
    /// to be subtly non-lossless.
    pub fn sample_from(&mut self, probs: &[f32]) -> usize {
        let draw = self.next_f32();
        let mut cumulative = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if draw < cumulative {
                return i;
            }
        }
        // Floating-point rounding may leave `draw` fractionally above
        // the final cumulative sum; the last nonzero-probability token
        // is the correct fallback, not index 0.
        probs
            .iter()
            .enumerate()
            .rev()
            .find(|&(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// The **exact** distribution [`Sampler::sample`] draws from for these
/// logits, params and history: penalties applied, temperature divided
/// in, top-k and top-p filtered, renormalised to sum to 1.
///
/// This is what makes lossless speculative verification possible. The
/// speculative-sampling rejection rule compares `p_target(x)` against
/// the draft's `q(x)`, and "the target's probability" is meaningless
/// unless it is the probability the *configured sampler* would actually
/// have used -- a rule that compared against the raw softmax while the
/// server sampled with `top_p = 0.9` would be lossless with respect to
/// a model nobody is running.
///
/// Greedy (`temperature <= 0.0`) is a distribution too: the point mass
/// on the argmax. Returning it as one rather than as a special case is
/// why the same verification code is correct at every temperature.
pub fn sampling_distribution(
    logits: &[f32],
    params: &SamplingParams,
    history: &[usize],
) -> Vec<f32> {
    let mut scores = logits.to_vec();
    apply_history_penalties(&mut scores, params, history);
    if params.temperature <= 0.0 {
        let mut probs = vec![0.0f32; scores.len()];
        if let Some(p) = probs.get_mut(argmax(&scores)) {
            *p = 1.0;
        }
        return probs;
    }
    filtered_distribution(scores, logits, params)
}

/// Shared tail of [`Sampler::sample_with_mask`] and
/// [`sampling_distribution`]: divide the already-penalised `scores` by
/// the temperature, softmax, apply top-k and top-p, and renormalise.
/// `raw_logits` is only consulted for the degenerate
/// everything-filtered-to-zero fallback.
///
/// Both callers go through here rather than each doing their own
/// temperature-then-filter, because a difference between the two is
/// exactly the kind of silent non-losslessness speculative
/// verification is supposed to rule out.
fn filtered_distribution(
    mut scores: Vec<f32>,
    raw_logits: &[f32],
    params: &SamplingParams,
) -> Vec<f32> {
    for s in scores.iter_mut() {
        *s /= params.temperature;
    }
    let mut probs = softmax(&scores);

    if params.top_k > 0 && params.top_k < probs.len() {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        for &i in idx.iter().skip(params.top_k) {
            probs[i] = 0.0;
        }
    }

    if params.top_p < 1.0 {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut cumulative = 0.0f32;
        let mut cutoff = idx.len();
        for (rank, &i) in idx.iter().enumerate() {
            cumulative += probs[i];
            if cumulative >= params.top_p {
                cutoff = rank + 1;
                break;
            }
        }
        for &i in idx.iter().skip(cutoff) {
            probs[i] = 0.0;
        }
    }

    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        // Every candidate got filtered to zero (degenerate params);
        // fall back to greedy rather than sampling from nothing.
        let mut point = vec![0.0f32; probs.len()];
        if let Some(p) = point.get_mut(argmax(raw_logits)) {
            *p = 1.0;
        }
        return point;
    }
    for p in probs.iter_mut() {
        *p /= total;
    }
    probs
}

fn apply_history_penalties(scores: &mut [f32], params: &SamplingParams, history: &[usize]) {
    if params.repetition_penalty != 1.0 {
        for &tok in history {
            if let Some(s) = scores.get_mut(tok) {
                *s = if *s > 0.0 {
                    *s / params.repetition_penalty
                } else {
                    *s * params.repetition_penalty
                };
            }
        }
    }
    if params.presence_penalty != 0.0 || params.frequency_penalty != 0.0 {
        let mut counts = std::collections::HashMap::<usize, usize>::new();
        for &tok in history {
            *counts.entry(tok).or_insert(0) += 1;
        }
        for (tok, count) in counts {
            if let Some(s) = scores.get_mut(tok) {
                *s -= params.frequency_penalty * count as f32;
                *s -= params.presence_penalty;
            }
        }
    }
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        vec![1.0 / logits.len().max(1) as f32; logits.len()]
    } else {
        exps.into_iter().map(|e| e / sum).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_zero_accepts_precomputed_argmax_singleton() {
        let mut sampler = Sampler::new(1);
        let params = SamplingParams::default();
        assert_eq!(sampler.sample(&[42.0], &params, &[]), 42);
        // Non-greedy must not treat a singleton as a token id.
        let sampled = SamplingParams {
            temperature: 0.8,
            ..SamplingParams::default()
        };
        // Softmax of a single logit → only token 0 is eligible.
        assert_eq!(sampler.sample(&[42.0], &sampled, &[]), 0);
    }

    #[test]
    fn temperature_zero_is_deterministic_greedy_argmax() {
        let logits = vec![0.1, 0.9, 0.3, -0.2];
        let params = SamplingParams::default();
        let mut sampler = Sampler::new(42);
        assert_eq!(sampler.sample(&logits, &params, &[]), 1);
        // Must be deterministic regardless of RNG state advancing.
        assert_eq!(sampler.sample(&logits, &params, &[]), 1);
    }

    #[test]
    fn high_temperature_can_pick_a_non_argmax_token_over_many_draws() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(7);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(sampler.sample(&logits, &params, &[]));
        }
        assert!(
            seen.len() > 1,
            "uniform logits at temperature=1.0 must produce more than one distinct token across 200 draws"
        );
    }

    #[test]
    fn top_k_one_is_equivalent_to_greedy() {
        let logits = vec![0.1, 0.9, 0.3, -0.2];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(123);
        for _ in 0..20 {
            assert_eq!(sampler.sample(&logits, &params, &[]), 1);
        }
    }

    #[test]
    fn top_p_near_zero_is_equivalent_to_greedy() {
        let logits = vec![0.1, 5.0, 0.3, -0.2];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.001,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(9);
        for _ in 0..20 {
            assert_eq!(sampler.sample(&logits, &params, &[]), 1);
        }
    }

    #[test]
    fn presence_and_frequency_penalties_reduce_seen_token_logits() {
        let logits = vec![0.0, 5.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            presence_penalty: 10.0,
            frequency_penalty: 0.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(1);
        let mut counts = [0usize; 3];
        for _ in 0..500 {
            counts[sampler.sample(&logits, &params, &[1])] += 1;
        }
        assert!(
            counts[1] < 250,
            "presence_penalty should discourage token 1; counts={counts:?}"
        );

        let params = SamplingParams {
            temperature: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 10.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(2);
        counts = [0; 3];
        for _ in 0..500 {
            counts[sampler.sample(&logits, &params, &[1, 1, 1])] += 1;
        }
        assert!(
            counts[1] < 250,
            "frequency_penalty should discourage repeated token 1; counts={counts:?}"
        );
    }

    #[test]
    fn repetition_penalty_reduces_probability_of_recently_seen_token() {
        let logits = vec![0.0, 5.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            repetition_penalty: 1000.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(3);
        let mut counts = [0usize; 3];
        for _ in 0..500 {
            counts[sampler.sample(&logits, &params, &[1])] += 1;
        }
        assert!(
            counts[1] < 250,
            "heavily penalizing token 1 (already in history) should make it far less likely than its raw logit alone would suggest; got counts={counts:?}"
        );
    }

    #[test]
    fn low_seeds_do_not_bias_the_first_draw() {
        // Every generation seeds a fresh `Sampler` (the server does it
        // per request, from the caller's `seed`), so the FIRST draw off
        // a freshly seeded generator is the one users actually see.
        // Plain xorshift64 returns its own state, so seeds 1..4000 all
        // produced a first draw in the bottom eighth of [0, 1) -- the
        // first sampled token of every seeded request came off the
        // bottom of the CDF.
        let vocab = 8;
        let logits = vec![0.0f32; vocab];
        let params = SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let seeds = 4_000u64;
        let mut counts = vec![0usize; vocab];
        for seed in 1..=seeds {
            counts[Sampler::new(seed).sample(&logits, &params, &[])] += 1;
        }
        let expected = seeds as f64 / vocab as f64;
        for (token, &c) in counts.iter().enumerate() {
            assert!(
                (c as f64 - expected).abs() < expected * 0.25,
                "uniform logits: token {token} came up {c} times across {seeds} seeds, \
                 expected about {expected:.0} (counts={counts:?})"
            );
        }
    }

    #[test]
    fn the_published_distribution_is_the_one_sample_actually_draws_from() {
        // `sampling_distribution` is load-bearing for lossless
        // speculative verification: if it disagreed with what `sample`
        // draws from, every accept/reject decision would be measured
        // against the wrong target. Check them against each other
        // empirically, with filters on so the two code paths have
        // something to disagree about.
        let logits = vec![0.4, 2.0, -1.0, 1.2, 0.9, -0.3];
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 4,
            repetition_penalty: 1.3,
            ..SamplingParams::default()
        };
        let history = [1usize, 4];
        let claimed = sampling_distribution(&logits, &params, &history);
        assert!((claimed.iter().sum::<f32>() - 1.0).abs() < 1e-5);

        let draws = 100_000;
        let mut counts = vec![0usize; logits.len()];
        let mut sampler = Sampler::new(0xC0FFEE);
        for _ in 0..draws {
            counts[sampler.sample(&logits, &params, &history)] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            let empirical = c as f64 / draws as f64;
            assert!(
                (empirical - claimed[i] as f64).abs() < 0.01,
                "token {i}: sample() draws it {empirical:.4} of the time but \
                 sampling_distribution claims {:.4}",
                claimed[i]
            );
        }
    }

    #[test]
    fn greedy_is_published_as_a_point_mass_not_a_special_case() {
        let logits = vec![0.1, 0.9, 0.3, -0.2];
        let probs = sampling_distribution(&logits, &SamplingParams::default(), &[]);
        assert_eq!(probs, vec![0.0, 1.0, 0.0, 0.0]);
        // Penalties still apply at temperature 0, so the point mass
        // moves with them.
        let penalized = sampling_distribution(
            &logits,
            &SamplingParams {
                repetition_penalty: 100.0,
                ..SamplingParams::default()
            },
            &[1],
        );
        assert_eq!(penalized[1], 0.0);
        assert_eq!(penalized.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn degenerate_all_zero_probability_falls_back_to_greedy() {
        // top_k=1 combined with a top_p that would exclude even that
        // one surviving token is a contradictory/degenerate
        // configuration; must not panic or sample index 0 blindly.
        let logits = vec![0.1, 0.9, 0.3, -0.2];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(1);
        assert_eq!(sampler.sample(&logits, &params, &[]), 1);
    }
}
