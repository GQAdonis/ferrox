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

use crate::sampler_chain::Candidates;

/// Sampling parameters for one generation request. `temperature <= 0.0`
/// means "sample nothing, take the greedy argmax" -- the same
/// deterministic behavior ferrox always had before this module existed.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    /// Nucleus sampling threshold in (0.0, 1.0]. 1.0 disables top-p
    /// filtering (every token with nonzero probability is eligible).
    pub top_p: f32,
    /// Keep only candidates at least `min_p` times as likely as the most
    /// likely one. `0.0` disables it; llama.cpp's `--min-p`, whose
    /// default is **0.05** (`common/common.h:231`) rather than off.
    ///
    /// That default is why this is a parity item and not a feature:
    /// llama.cpp truncates with min-p on every run nobody configured,
    /// so without it ferrox could not reproduce llama.cpp's *own*
    /// out-of-the-box output for any prompt.
    ///
    /// The struct default here stays `0.0` (disabled) for the same
    /// reason `temperature` defaults to greedy: `SamplingParams::default`
    /// is ferrox's "do nothing the caller did not ask for" baseline, and
    /// llama.cpp's CLI numbers live on the CLI flags.
    pub min_p: f32,
    /// Keep only the `top_k` highest-probability tokens before
    /// sampling. 0 disables top-k filtering.
    pub top_k: usize,
    /// > 1.0 discourages repeating a token already in `history`; 1.0
    /// > disables repetition penalty. Uses the standard convention
    /// > (divide positive logits, multiply negative ones) so the penalty
    /// > always pushes toward *less* likely, regardless of logit sign.
    pub repetition_penalty: f32,
    /// How many of the most recent tokens the penalties look at, as
    /// llama.cpp's `penalty_last_n` (`common/common.h:238`, default 64).
    ///
    /// `0` disables the penalties entirely. ferrox had no window at all
    /// and scanned the WHOLE history, so on a long generation it
    /// penalised a steadily growing set of tokens where llama.cpp
    /// penalises the last 64 -- the divergence grew with output length,
    /// which is exactly when a repetition penalty matters most.
    pub penalty_last_n: usize,
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
            min_p: 0.0,
            top_k: 0,
            repetition_penalty: 1.0,
            penalty_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

/// The sampling a **checkpoint recommends for itself**, one `Option`
/// per field so that "this model says nothing about top_p" stays
/// distinguishable from "this model recommends top_p = 1.0". This is
/// sglang's `sampling_defaults='model'`, ported from FreeToken
/// `python/freetoken/utils/hf.py:92 load_generation_sampling`.
///
/// Every field is `None` for a checkpoint that recommends nothing,
/// which is the overwhelming majority, and
/// [`RecommendedSampling::resolve`] then reproduces ferrox's existing
/// defaults exactly -- a recommendation may only fill a gap the request
/// left, never override it.
///
/// Why this exists at all: reasoning checkpoints are tuned for a
/// specific sampler (Qwen3.5 ships temperature 1.0, top_k 20, top_p
/// 0.95) and ship those numbers with the weights. Served under a
/// generic greedy-or-0.8 default they fall into repetition loops --
/// fluent output that never terminates -- which reads as a broken model
/// rather than as a serving default nobody read off the file.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RecommendedSampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
}

/// The sampling fields **one request** actually specified. `None` means
/// the request said nothing about that field, so the checkpoint's
/// recommendation (and then the framework default) may speak for it.
///
/// Collapsing this to a plain [`SamplingParams`] at the wire boundary
/// -- `temperature: req.temperature.unwrap_or(0.0)` -- is what destroys
/// the distinction: a request that omitted `temperature` becomes
/// indistinguishable from one that explicitly asked for greedy, and no
/// recommendation can ever apply.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RequestedSampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
}

impl RecommendedSampling {
    /// True when the checkpoint recommended nothing at all, i.e.
    /// [`Self::resolve`] is guaranteed to return the framework defaults
    /// for any request. Useful for telling an operator whether
    /// "model defaults" had anything to act on.
    pub fn is_empty(&self) -> bool {
        *self == RecommendedSampling::default()
    }

    /// Precedence, exactly as FreeToken's `resolve_sampling.pick`
    /// (`python/freetoken/server/generation.py:170`) applies it: the
    /// **request's** own value, else the **checkpoint's**
    /// recommendation, else the **framework** default carried by
    /// `framework` (ferrox's `SamplingParams::default()` unless a caller
    /// has its own).
    ///
    /// The penalty fields are taken from `framework` untouched: nothing
    /// in the reference reads a recommended penalty, and inventing one
    /// here would be this function changing generation on its own.
    ///
    /// Getting the order wrong in either direction is a silent
    /// behaviour change: recommendation-over-request makes a client's
    /// explicit `temperature: 0` unreachable on a model that recommends
    /// 1.0, and framework-over-recommendation is the greedy repetition
    /// loop this whole path exists to avoid.
    pub fn resolve(
        &self,
        requested: RequestedSampling,
        framework: SamplingParams,
    ) -> SamplingParams {
        SamplingParams {
            temperature: requested
                .temperature
                .or(self.temperature)
                .unwrap_or(framework.temperature),
            top_p: requested.top_p.or(self.top_p).unwrap_or(framework.top_p),
            top_k: requested.top_k.or(self.top_k).unwrap_or(framework.top_k),
            ..framework
        }
    }

    /// The recommendation in a HuggingFace-style `generation_config.json`
    /// body.
    ///
    /// Two rules, both from the reference
    /// (`hf.py:92 load_generation_sampling`):
    ///
    /// * `do_sample: false` means the checkpoint recommends **greedy**,
    ///   which is returned as `temperature = 0` and *nothing else* --
    ///   the top_k/top_p in such a file describe a sampler the model
    ///   asks not to be used.
    /// * otherwise only the keys **actually present** are returned. An
    ///   absent key stays `None`; filling it with a house default (the
    ///   naive reading, and what HF's own `GenerationConfig` object does
    ///   for you) would turn silence into a recommendation and let a
    ///   file that says only `temperature: 0.6` also pin top_p to 1.0,
    ///   overriding the server's own default for a value the checkpoint
    ///   never expressed.
    ///
    /// A file that does not parse, or is not a JSON object, recommends
    /// nothing -- a malformed sidecar must not be able to change how a
    /// model is sampled.
    pub fn from_generation_config(json: &str) -> Self {
        let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(json)
        else {
            return RecommendedSampling::default();
        };
        if map.get("do_sample").and_then(|v| v.as_bool()) == Some(false) {
            return RecommendedSampling {
                temperature: Some(0.0),
                ..RecommendedSampling::default()
            };
        }
        RecommendedSampling {
            temperature: map
                .get("temperature")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            top_p: map.get("top_p").and_then(|v| v.as_f64()).map(|v| v as f32),
            top_k: map
                .get("top_k")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        }
    }

    /// [`Self::from_generation_config`] for the `generation_config.json`
    /// beside a checkpoint's weights (an HF-format model directory).
    ///
    /// A directory with no such file recommends nothing, exactly like a
    /// GGUF with no `general.sampling.*` keys: the absence of a
    /// recommendation is the normal case and must never be an error that
    /// stops a model from loading.
    pub fn from_model_dir(dir: &std::path::Path) -> Self {
        match std::fs::read_to_string(dir.join("generation_config.json")) {
            Ok(text) => Self::from_generation_config(&text),
            Err(_) => RecommendedSampling::default(),
        }
    }
}

/// Sets logits a caller wants to forbid to `-inf`, in place, before the
/// sampler looks at them.
///
/// Two callers today, and they COMPOSE rather than exclude each other --
/// a masked logit stays masked, so the order they run in cannot matter:
/// JSON-object mode's character-class filter
/// (`ferrox_server::json_mode`), and grammar-constrained decoding
/// ([`crate::grammar_sampler::GrammarSampler::mask_logits`]).
///
/// The signature returns nothing because the callback runs from inside
/// the sampler, which has no error to return one through. A mask that
/// CAN fail -- a grammar that dead-ends leaves every logit at `-inf`,
/// and sampling from that is how an "impossible" request becomes
/// arbitrary text with a 200 -- records its refusal in the closure's own
/// captured state, and the decode loop reads it after the sample and
/// throws the token away. `ferrox_server::sample_step::sample_next` is
/// the one place that pairing lives.
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

        let probs = filtered_distribution(scores, params);
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
    filtered_distribution(scores, params)
}

/// Shared tail of [`Sampler::sample_with_mask`] and
/// [`sampling_distribution`]: run the already-penalised `scores` through
/// llama.cpp's sampler chain and return the resulting full-vocabulary
/// distribution.
///
/// # Order, and why it is a specification
///
/// llama.cpp's default chain is `penalties, dry, top_n_sigma, top_k,
/// typical_p, top_p, min_p, xtc, temperature` (`common/common.h:259-269`,
/// consumed by `common/sampling.cpp:349-397`). The penalties already ran
/// in [`apply_history_penalties`]; this function is the rest of it, in
/// that order, and **temperature is last**.
///
/// ferrox used to divide by the temperature FIRST and filter afterwards.
/// That is not a reordering of independent steps. Top-p selects the
/// smallest set of candidates whose probabilities sum to `p`, and
/// temperature changes those probabilities: a high temperature flattens
/// the distribution so the nucleus grows, a low one sharpens it so the
/// nucleus shrinks. Min-p compares each candidate's logit against
/// `max + ln(p)`, and temperature scales exactly the gap being compared.
/// Filtering before scaling and filtering after scaling therefore keep
/// DIFFERENT candidate sets for the same flags.
///
/// Both callers go through here rather than each running their own
/// chain, because a difference between the two is exactly the kind of
/// silent non-losslessness speculative verification is supposed to rule
/// out.
///
/// The filters themselves live in [`crate::sampler_chain`], which models
/// the shrinking candidate list llama.cpp passes down the chain --
/// including the renormalisation between steps that a keep-mask cannot
/// express. See that module's header.
fn filtered_distribution(scores: Vec<f32>, params: &SamplingParams) -> Vec<f32> {
    let vocab = scores.len();
    let mut candidates = Candidates::new(&scores);
    candidates.top_k(params.top_k);
    candidates.top_p(params.top_p);
    candidates.min_p(params.min_p);
    candidates.temperature(params.temperature);
    candidates.into_distribution(vocab)
}

/// Penalise tokens that already appear in `history`, once each.
///
/// ONCE EACH is the whole subtlety, and ferrox used to get it wrong.
/// llama.cpp walks the CANDIDATE list and looks each candidate up in a
/// count map (`llama-sampler.cpp:2735-2756`), so a token repeated `n`
/// times is divided by `penalty_repeat` exactly once. ferrox walked the
/// HISTORY, so the same token was divided `n` times and the effective
/// penalty was `penalty^n`.
///
/// That was live on every `ferrox run`: `--repeat-penalty` defaults to
/// 1.1, so a token seen five times was penalised 1.61x rather than
/// 1.1x, and the divergence grew with the length of the output.
///
/// The sign convention is llama.cpp's and its comment explains it:
/// dividing alone would make tokens with NEGATIVE logits more likely,
/// so negatives are multiplied instead.
fn apply_history_penalties(scores: &mut [f32], params: &SamplingParams, history: &[usize]) {
    if params.repetition_penalty == 1.0
        && params.presence_penalty == 0.0
        && params.frequency_penalty == 0.0
    {
        return;
    }
    // Only the last `penalty_last_n`, as llama.cpp's ring buffer does.
    if params.penalty_last_n == 0 {
        return;
    }
    let window = history.len().saturating_sub(params.penalty_last_n);
    let mut counts = std::collections::HashMap::<usize, usize>::new();
    for &tok in &history[window..] {
        *counts.entry(tok).or_insert(0) += 1;
    }
    for (tok, count) in counts {
        let Some(s) = scores.get_mut(tok) else {
            continue;
        };
        if params.repetition_penalty != 1.0 {
            *s = if *s > 0.0 {
                *s / params.repetition_penalty
            } else {
                *s * params.repetition_penalty
            };
        }
        *s -= params.frequency_penalty * count as f32;
        *s -= params.presence_penalty;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The repetition penalty is applied ONCE per token, however many
    /// times that token appears in the history.
    ///
    /// ferrox walked the history and divided once per OCCURRENCE, so the
    /// effective penalty was `penalty^n`. llama.cpp walks the candidates
    /// and looks each up in a count map, so it is `penalty` flat
    /// (`llama-sampler.cpp:2735-2756`).
    ///
    /// Live on every `ferrox run`: `--repeat-penalty` defaults to 1.1,
    /// so a token seen five times was penalised 1.61x, and the
    /// divergence grew with the length of the output. Twenty-four
    /// sampling tests passed with the bug in place, which is why this
    /// one exists.
    #[test]
    fn the_repetition_penalty_does_not_compound_with_repeats() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 2.0,
            ..SamplingParams::default()
        };
        let logits = vec![4.0f32, 1.0, 1.0];

        // Token 0 appears five times. Penalised once, its score is 2.0;
        // compounded it would be 4 / 2^5 = 0.125.
        let mut scores = logits.clone();
        apply_history_penalties(&mut scores, &params, &[0, 0, 0, 0, 0]);
        assert!(
            (scores[0] - 2.0).abs() < 1e-6,
            "expected one division (2.0), got {} -- {} would be 2^5",
            scores[0],
            4.0f32 / 32.0
        );

        // And once really is once: one occurrence and five occurrences
        // must land on the same score, or the count still leaks in.
        let mut once = logits.clone();
        apply_history_penalties(&mut once, &params, &[0]);
        assert_eq!(once[0].to_bits(), scores[0].to_bits());

        // A NEGATIVE logit is multiplied rather than divided, or the
        // penalty would make it more likely -- llama.cpp's own comment.
        let mut negative = vec![-4.0f32];
        apply_history_penalties(&mut negative, &params, &[0, 0, 0]);
        assert!((negative[0] + 8.0).abs() < 1e-6, "got {}", negative[0]);
    }

    /// The penalties look at the last `penalty_last_n` tokens, not the
    /// whole history.
    ///
    /// llama.cpp keeps a ring buffer of `penalty_last_n` (default 64,
    /// `common/common.h:238`); ferrox scanned everything generated so
    /// far. On a long generation that is a steadily growing set of
    /// penalised tokens against llama.cpp's fixed 64 -- the divergence
    /// grows with output length, which is when a repetition penalty
    /// matters most.
    #[test]
    fn the_penalties_only_see_the_last_n_tokens() {
        let params = SamplingParams {
            repetition_penalty: 2.0,
            penalty_last_n: 2,
            ..SamplingParams::default()
        };
        let mut scores = vec![8.0f32, 8.0, 8.0];
        // Token 0 fell out of the window; tokens 1 and 2 are in it.
        apply_history_penalties(&mut scores, &params, &[0, 1, 2]);
        assert_eq!(
            scores[0].to_bits(),
            8.0f32.to_bits(),
            "token 0 is outside the window"
        );
        assert!((scores[1] - 4.0).abs() < 1e-6, "got {}", scores[1]);
        assert!((scores[2] - 4.0).abs() < 1e-6, "got {}", scores[2]);

        // `0` disables the penalties outright, as llama.cpp documents.
        let off = SamplingParams {
            penalty_last_n: 0,
            ..params
        };
        let mut untouched = vec![8.0f32; 3];
        apply_history_penalties(&mut untouched, &off, &[0, 1, 2]);
        assert_eq!(untouched, vec![8.0f32; 3]);

        // A window longer than the history is not an overflow.
        let wide = SamplingParams {
            penalty_last_n: 1000,
            ..params
        };
        let mut short = vec![8.0f32];
        apply_history_penalties(&mut short, &wide, &[0]);
        assert!((short[0] - 4.0).abs() < 1e-6);
    }

    /// Frequency penalty still scales with the count, while the
    /// repetition penalty does not.
    ///
    /// Both live in the same loop, so a fix that made the repetition
    /// penalty flat by dropping the counts would break this one.
    #[test]
    fn the_frequency_penalty_still_counts_repeats() {
        let params = SamplingParams {
            frequency_penalty: 0.5,
            presence_penalty: 0.25,
            ..SamplingParams::default()
        };
        let mut scores = vec![10.0f32];
        apply_history_penalties(&mut scores, &params, &[0, 0, 0, 0]);
        // 10 - 0.5*4 - 0.25 = 7.75
        assert!((scores[0] - 7.75).abs() < 1e-6, "got {}", scores[0]);
    }

    /// Top-p cuts the UNSCALED distribution; the temperature reshapes
    /// only the survivors.
    ///
    /// llama.cpp's default chain runs temperature LAST
    /// (`common/common.h:259-269`); ferrox divided first and filtered
    /// afterwards. Not an innocuous reordering: temperature changes the
    /// probabilities top-p sums over, so a high temperature flattens the
    /// distribution and grows the nucleus. The two orders keep different
    /// candidate sets for identical flags.
    #[test]
    fn temperature_does_not_change_which_candidates_top_p_keeps() {
        let logits = vec![3.0f32, 2.0, 1.0, 0.0];
        let at = |temperature: f32| -> Vec<bool> {
            let params = SamplingParams {
                temperature,
                top_p: 0.9,
                top_k: 0,
                ..SamplingParams::default()
            };
            sampling_distribution(&logits, &params, &[])
                .iter()
                .map(|&p| p > 0.0)
                .collect()
        };

        let cold = at(0.5);
        let hot = at(4.0);
        assert_eq!(
            cold, hot,
            "the surviving set must not depend on the temperature: \
             cold={cold:?} hot={hot:?}"
        );
        // And the cut must actually bite, or the equality above is
        // satisfied by keeping everything.
        assert!(
            cold.iter().any(|&k| !k),
            "top_p = 0.9 must drop at least one of these four candidates"
        );
    }

    /// min-p truncates, and it truncates on llama.cpp's threshold.
    ///
    /// llama.cpp enables min-p **by default** at 0.05
    /// (`common/common.h:231`), so until this existed ferrox could not
    /// reproduce llama.cpp's own out-of-the-box output on any prompt --
    /// a parity gap, not a missing feature.
    ///
    /// Logits `[4, 3, 2, 1]` at `min_p = 0.2`: the threshold is
    /// `4 + ln(0.2) = 2.3905`, so exactly the candidates at 4 and 3
    /// survive. Arithmetic done by hand from
    /// `src/llama-sampler.cpp:1556`, not read back off the code.
    #[test]
    fn min_p_truncates_at_ln_p_below_the_top_logit() {
        let logits = vec![4.0f32, 3.0, 2.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            min_p: 0.2,
            ..SamplingParams::default()
        };
        let probs = sampling_distribution(&logits, &params, &[]);
        assert!(probs[0] > 0.0 && probs[1] > 0.0);
        assert_eq!(probs[2], 0.0, "2.0 is below 4 + ln(0.2) = 2.3905");
        assert_eq!(probs[3], 0.0);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // The two survivors are renormalised against each other:
        // e^4 / (e^4 + e^3) = 0.7311.
        assert!((probs[0] - 0.731_059).abs() < 1e-5, "got {}", probs[0]);

        // 0.0 disables it, which is ferrox's struct default -- adding
        // min-p must not change any existing caller's distribution.
        let off = SamplingParams {
            min_p: 0.0,
            ..params.clone()
        };
        let unfiltered = sampling_distribution(&logits, &off, &[]);
        assert!(unfiltered.iter().all(|&p| p > 0.0));
    }

    /// min-p runs BEFORE the temperature, so the set it keeps does not
    /// depend on `--temp`.
    ///
    /// This is the same trap as E4 and it bites harder here. min-p's
    /// test is `logit_i >= logit_max + ln(p)`, and temperature divides
    /// **both** logits, so it scales the very gap being compared against
    /// a fixed `ln(p)`. On these logits at `min_p = 0.2`, running min-p
    /// after a temperature of 0.5 would keep one candidate and after 2.0
    /// would keep all four; llama.cpp keeps two at every temperature
    /// (`common/common.h:259-269` puts `MIN_P` before `TEMPERATURE`).
    ///
    /// Move `candidates.min_p(..)` after `candidates.temperature(..)` in
    /// `filtered_distribution` and this goes red.
    #[test]
    fn temperature_does_not_change_which_candidates_min_p_keeps() {
        let logits = vec![3.0f32, 2.0, 1.0, 0.0];
        let survivors = |temperature: f32| -> Vec<bool> {
            let params = SamplingParams {
                temperature,
                min_p: 0.2,
                ..SamplingParams::default()
            };
            sampling_distribution(&logits, &params, &[])
                .iter()
                .map(|&p| p > 0.0)
                .collect()
        };

        let cold = survivors(0.5);
        let warm = survivors(1.0);
        let hot = survivors(2.0);
        assert_eq!(cold, warm, "cold={cold:?} warm={warm:?}");
        assert_eq!(warm, hot, "warm={warm:?} hot={hot:?}");
        // 3 + ln(0.2) = 1.3905, so exactly the 3.0 and 2.0 candidates.
        assert_eq!(warm, vec![true, true, false, false]);
    }

    /// min-p sits AFTER top-p in the chain, and both may bite on the
    /// same call.
    ///
    /// `top_p = 0.95` on this distribution keeps three candidates
    /// (0.6337 + 0.2331 + 0.0857 = 0.9525); min-p at 0.2 then drops the
    /// third, whose probability is 0.135 of the top one. Getting only
    /// one of the two filters gives a different answer either way, so
    /// this fails if either is dropped or if min-p is skipped when top-p
    /// already truncated.
    #[test]
    fn top_p_and_min_p_both_apply() {
        let logits = vec![3.0f32, 2.0, 1.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.95,
            min_p: 0.2,
            ..SamplingParams::default()
        };
        let probs = sampling_distribution(&logits, &params, &[]);
        assert_eq!(
            probs.iter().map(|&p| p > 0.0).collect::<Vec<_>>(),
            vec![true, true, false, false]
        );

        // top-p alone keeps three; min-p alone also keeps two here, so
        // pin the top-p-only case to prove the two filters are distinct
        // and that this test is not satisfied by min-p doing all the
        // work.
        let top_p_only = SamplingParams {
            min_p: 0.0,
            ..params.clone()
        };
        assert_eq!(
            sampling_distribution(&logits, &top_p_only, &[])
                .iter()
                .filter(|&&p| p > 0.0)
                .count(),
            3
        );
    }

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

    /// Only the keys the file actually carries become a recommendation.
    ///
    /// **This test fails if an absent key is filled with a house
    /// default** (the naive reading, and what HF's own
    /// `GenerationConfig` object does): `top_p` and `top_k` would come
    /// back as `Some(1.0)` / `Some(0)` and would then override whatever
    /// the server itself defaults to, for values this checkpoint never
    /// expressed.
    #[test]
    fn an_absent_generation_config_key_stays_absent_rather_than_taking_a_default() {
        let recommended = RecommendedSampling::from_generation_config(r#"{"temperature": 0.6}"#);
        assert_eq!(recommended.temperature, Some(0.6));
        assert_eq!(recommended.top_p, None, "top_p was not in the file");
        assert_eq!(recommended.top_k, None, "top_k was not in the file");
        // An explicit JSON null is silence too (the reference's
        // `if val is not None`).
        let nulled = RecommendedSampling::from_generation_config(r#"{"top_p": null}"#);
        assert_eq!(nulled, RecommendedSampling::default());
    }

    /// A reasoning checkpoint's full recommendation survives intact --
    /// the case the whole path exists for (Qwen3.5: temp 1.0, top_k 20,
    /// top_p 0.95).
    #[test]
    fn every_generation_config_key_present_is_recommended() {
        let recommended = RecommendedSampling::from_generation_config(
            r#"{"do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}"#,
        );
        assert_eq!(
            recommended,
            RecommendedSampling {
                temperature: Some(1.0),
                top_p: Some(0.95),
                top_k: Some(20),
            }
        );
    }

    /// `do_sample: false` recommends greedy, expressed as temperature 0
    /// and *nothing else*: the top_k/top_p such a file also carries
    /// describe a sampler it is asking not to be used, so returning them
    /// would filter a distribution the model wants collapsed to its
    /// argmax.
    #[test]
    fn do_sample_false_recommends_greedy_and_no_other_field() {
        let recommended = RecommendedSampling::from_generation_config(
            r#"{"do_sample": false, "temperature": 0.7, "top_k": 50, "top_p": 0.9}"#,
        );
        assert_eq!(recommended.temperature, Some(0.0));
        assert_eq!(recommended.top_p, None);
        assert_eq!(recommended.top_k, None);
    }

    /// A sidecar that does not parse must not be able to change how the
    /// model is sampled.
    #[test]
    fn a_malformed_generation_config_recommends_nothing() {
        for text in ["", "not json", "[1, 2, 3]", "null"] {
            assert!(
                RecommendedSampling::from_generation_config(text).is_empty(),
                "{text:?} must recommend nothing"
            );
        }
    }

    /// Precedence: the request wins over the checkpoint, and the
    /// checkpoint only fills what the request left unset. An explicit
    /// `temperature: 0` from a client must stay reachable on a model
    /// that recommends 1.0.
    #[test]
    fn a_request_outranks_the_recommendation_which_outranks_the_framework_default() {
        let recommended = RecommendedSampling {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(20),
        };
        let resolved = recommended.resolve(
            RequestedSampling {
                temperature: Some(0.0),
                ..RequestedSampling::default()
            },
            SamplingParams::default(),
        );
        assert_eq!(resolved.temperature, 0.0, "the request asked for greedy");
        assert_eq!(resolved.top_p, 0.95, "the request said nothing about top_p");
        assert_eq!(resolved.top_k, 20, "the request said nothing about top_k");
        // Penalties are never recommended, only carried through.
        assert_eq!(resolved.repetition_penalty, 1.0);
    }

    /// A checkpoint that recommends nothing must leave ferrox's existing
    /// behaviour bit-identical: greedy, unfiltered, exactly
    /// `SamplingParams::default()`.
    #[test]
    fn a_checkpoint_that_recommends_nothing_leaves_the_framework_defaults_alone() {
        let resolved = RecommendedSampling::default()
            .resolve(RequestedSampling::default(), SamplingParams::default());
        let default = SamplingParams::default();
        assert_eq!(resolved.temperature, default.temperature);
        assert_eq!(resolved.top_p, default.top_p);
        assert_eq!(resolved.top_k, default.top_k);
    }

    /// A model directory with no `generation_config.json` recommends
    /// nothing rather than failing: the absence of a recommendation is
    /// the normal case for most checkpoints.
    #[test]
    fn a_model_directory_without_a_generation_config_recommends_nothing() {
        let dir = std::env::temp_dir().join(format!(
            "ferrox_test_no_generation_config_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(RecommendedSampling::from_model_dir(&dir).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The sidecar is read from the directory beside the weights, the
    /// same place HF's `GenerationConfig.from_pretrained` looks.
    #[test]
    fn a_model_directory_generation_config_is_read_from_beside_the_weights() {
        let dir = std::env::temp_dir().join(format!(
            "ferrox_test_generation_config_dir_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"temperature": 0.6, "top_p": 0.95}"#,
        )
        .unwrap();
        let recommended = RecommendedSampling::from_model_dir(&dir);
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(recommended.temperature, Some(0.6));
        assert_eq!(recommended.top_p, Some(0.95));
        assert_eq!(recommended.top_k, None);
    }
}
