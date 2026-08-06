//! Token sampling from a decoder's output logits: temperature, top-k,
//! top-p (nucleus), and repetition penalty, on top of the greedy argmax
//! ferrox previously always used unconditionally (see
//! `crate::speculative`, which still uses plain greedy argmax directly
//! since its quality-neutrality proof depends on exactly matching
//! greedy decode -- sampling is a deliberately separate, opt-in path).
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
        self.state
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
            return argmax(logits);
        }
        for p in probs.iter_mut() {
            *p /= total;
        }

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
