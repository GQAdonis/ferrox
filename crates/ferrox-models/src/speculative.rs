//! Prompt-lookup speculative decoding: propose several candidate next
//! tokens by finding a repeat of the current context elsewhere in the
//! token history (no separate draft model needed, unlike classic
//! speculative decoding), then verify all candidates in a single
//! `Decoder::forward_batch` call instead of one `forward_token` call
//! per candidate.
//!
//! This is the CPU-only, no-draft-model variant of speculative
//! decoding (the same idea as vLLM's "prompt lookup decoding"), chosen
//! specifically because it needs no GPU and no second model to be
//! useful -- unlike tree-based speculative decoding with a real draft
//! model, which needs real hardware to actually pay off.
//!
//! # Quality-neutrality is the property that matters most here
//!
//! Speculative decoding is only worth having if it produces *exactly*
//! the same output as plain greedy decode, just potentially faster.
//! `speculative_decode`'s accept/reject protocol is designed so that
//! every accepted token is one `forward_batch` would have produced
//! anyway on its own path: a candidate is only kept if the model's own
//! argmax at that position agrees with it. This is checked directly by
//! `speculative_decode_matches_greedy_token_for_token`, which runs both
//! this module's decode and a plain sequential `forward_token` loop
//! against the same decoder and asserts the exact same token sequence
//! comes out either way.

use crate::decoder::Decoder;
use ferrox_core::cache::KvCache;

/// Proposes candidate continuation tokens by looking for the longest
/// available match of the most recent `ngram_size` tokens earlier in
/// `history`, and returning up to `max_draft_len` tokens that followed
/// that earlier occurrence. Returns an empty vector if no match is
/// found or `history` is too short to contain one.
///
/// This is deliberately simple (last-match-wins, not best-match or a
/// frequency-weighted choice): the whole point of prompt-lookup
/// decoding is that it's nearly free to compute, since a wrong guess
/// costs nothing but a rejected batch position, not a correctness bug.
#[derive(Debug, Clone, Copy)]
pub struct PromptLookupSpeculator {
    pub ngram_size: usize,
    pub max_draft_len: usize,
}

impl PromptLookupSpeculator {
    pub fn new(ngram_size: usize, max_draft_len: usize) -> Self {
        assert!(ngram_size >= 1, "ngram_size must be at least 1");
        assert!(max_draft_len >= 1, "max_draft_len must be at least 1");
        PromptLookupSpeculator {
            ngram_size,
            max_draft_len,
        }
    }

    /// Looks for the most recent earlier occurrence of `history`'s
    /// last `ngram_size` tokens, scanning from the end backwards so
    /// the *most recent* match wins (most likely to reflect current
    /// context, e.g. a loop the model is currently in). Returns the
    /// tokens that followed that occurrence, truncated to
    /// `max_draft_len`.
    pub fn propose(&self, history: &[usize]) -> Vec<usize> {
        if history.len() < self.ngram_size + 1 {
            return Vec::new();
        }
        let needle = &history[history.len() - self.ngram_size..];

        // Search every earlier start position, latest first. The last
        // possible start that still leaves room for the needle without
        // overlapping into the needle itself is history.len() -
        // ngram_size - 1 (exclusive of the needle's own occurrence).
        let last_possible_start = history.len() - self.ngram_size - 1;
        for start in (0..=last_possible_start).rev() {
            if &history[start..start + self.ngram_size] == needle {
                let continuation_start = start + self.ngram_size;
                let available = history.len() - continuation_start;
                let take = available.min(self.max_draft_len);
                return history[continuation_start..continuation_start + take].to_vec();
            }
        }
        Vec::new()
    }
}

/// Result of a speculative decode run, with the counters that make its
/// actual savings observable rather than just assumed.
#[derive(Debug, Clone)]
pub struct SpeculativeDecodeResult {
    pub generated_tokens: Vec<usize>,
    /// Number of `Decoder::forward_batch` calls made (prefill counts as
    /// one call, each subsequent accept/reject round counts as one
    /// more, regardless of how many tokens that round produced).
    pub forward_calls: usize,
    /// Total tokens produced across all rounds -- always equal to
    /// `generated_tokens.len()`, kept as a separate field so the ratio
    /// `tokens_generated / forward_calls` (the actual speedup metric)
    /// is easy to read directly off this struct.
    pub tokens_generated: usize,
}

impl SpeculativeDecodeResult {
    /// Average tokens produced per `forward_batch` call. 1.0 means
    /// speculation never helped (every round produced exactly the
    /// anchor token); higher means draft tokens were accepted.
    pub fn tokens_per_call(&self) -> f64 {
        if self.forward_calls == 0 {
            0.0
        } else {
            self.tokens_generated as f64 / self.forward_calls as f64
        }
    }
}

/// Runs greedy decoding for `max_new_tokens` steps, using
/// `speculator` to propose candidate continuations and verifying them
/// in batches. `prompt_tokens` is processed as a single prefill batch
/// (one `forward_batch` call for the whole prompt, not one per
/// prompt token -- itself a real saving independent of speculation).
///
/// `kv_caches` must be freshly initialized (empty) for this call;
/// continuing an already-populated cache across multiple calls isn't
/// supported by this function today.
pub fn speculative_decode(
    decoder: &Decoder,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
    kv_caches: &mut [KvCache],
    speculator: &PromptLookupSpeculator,
) -> SpeculativeDecodeResult {
    assert!(!prompt_tokens.is_empty(), "prompt must not be empty");

    let mut history: Vec<usize> = prompt_tokens.to_vec();
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut forward_calls = 0usize;

    // Prefill: one batched call over the whole prompt instead of one
    // forward_token call per prompt token.
    let prefill_logits = decoder.forward_batch(prompt_tokens, 0, kv_caches);
    forward_calls += 1;
    let mut pending_logits = prefill_logits
        .last()
        .expect("prompt_tokens is non-empty, so forward_batch returns at least one logits vector")
        .clone();
    let mut pos = prompt_tokens.len();

    while generated.len() < max_new_tokens {
        let real_tok = argmax(&pending_logits);

        let remaining_budget = max_new_tokens - generated.len() - 1; // -1 for real_tok itself
        let mut guesses = speculator.propose(&history);
        guesses.truncate(remaining_budget);

        if guesses.is_empty() {
            // No draft: verify just the anchor token, batch size 1.
            let logits = decoder.forward_batch(&[real_tok], pos, kv_caches);
            forward_calls += 1;
            generated.push(real_tok);
            history.push(real_tok);
            pending_logits = logits.into_iter().next().unwrap();
            pos += 1;
            continue;
        }

        let mut batch = Vec::with_capacity(1 + guesses.len());
        batch.push(real_tok);
        batch.extend_from_slice(&guesses);

        let batch_logits = decoder.forward_batch(&batch, pos, kv_caches);
        forward_calls += 1;

        // accepted_count = length of the longest prefix of `guesses`
        // whose predicted-by-the-real-path argmax matches.
        let mut accepted_count = 0usize;
        for (i, &guess) in guesses.iter().enumerate() {
            if argmax(&batch_logits[i]) == guess {
                accepted_count += 1;
            } else {
                break;
            }
        }

        if accepted_count < guesses.len() {
            // Some guess was wrong: roll the cache back to keep only
            // the anchor token plus the accepted guesses.
            let committed_len = pos + 1 + accepted_count;
            for cache in kv_caches.iter_mut() {
                cache.truncate(committed_len);
            }
        }

        generated.push(real_tok);
        generated.extend_from_slice(&guesses[..accepted_count]);
        history.push(real_tok);
        history.extend_from_slice(&guesses[..accepted_count]);

        // batch_logits[accepted_count] was computed from a token that
        // is definitely correct (either the anchor itself, if
        // accepted_count == 0, or the last accepted guess), so it
        // correctly predicts the next not-yet-filled position
        // regardless of whether every guess was accepted.
        pending_logits = batch_logits[accepted_count].clone();
        pos += 1 + accepted_count;
    }

    generated.truncate(max_new_tokens);
    SpeculativeDecodeResult {
        tokens_generated: generated.len(),
        generated_tokens: generated,
        forward_calls,
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
    use crate::config::glm_5_2;
    use crate::ModelConfig;

    fn tiny_test_config() -> ModelConfig {
        let mut cfg = glm_5_2();
        cfg.hidden_dim = 16;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 4;
        cfg.moe.hidden_dim = 16;
        cfg.moe.n_experts = 6;
        cfg.moe.n_experts_active = 2;
        cfg.moe.n_shared_experts = 1;
        cfg.moe.expert_ffn_dim = 8;
        cfg
    }

    // ---- PromptLookupSpeculator tests ----

    #[test]
    fn proposes_the_continuation_after_a_real_repeat() {
        let spec = PromptLookupSpeculator::new(2, 4);
        // "...1 2 3 4 5 9 9 9 1 2" -> earlier "1 2" occurs at the very
        // start (indices 0-1); the 4 tokens that followed it are
        // "3 4 5 9" (capped at max_draft_len=4).
        let history = vec![1, 2, 3, 4, 5, 9, 9, 9, 1, 2];
        assert_eq!(spec.propose(&history), vec![3, 4, 5, 9]);
    }

    #[test]
    fn respects_max_draft_len() {
        let spec = PromptLookupSpeculator::new(2, 2);
        let history = vec![1, 2, 3, 4, 5, 6, 7, 1, 2];
        assert_eq!(spec.propose(&history), vec![3, 4]);
    }

    #[test]
    fn returns_empty_when_no_earlier_match_exists() {
        let spec = PromptLookupSpeculator::new(2, 4);
        let history = vec![1, 2, 3, 4, 5];
        assert_eq!(spec.propose(&history), Vec::<usize>::new());
    }

    #[test]
    fn returns_empty_when_history_too_short() {
        let spec = PromptLookupSpeculator::new(3, 4);
        let history = vec![1, 2, 3];
        assert_eq!(spec.propose(&history), Vec::<usize>::new());
    }

    #[test]
    fn finds_the_most_recent_match_when_several_exist() {
        let spec = PromptLookupSpeculator::new(1, 3);
        // needle = [9]. Earlier occurrences at index 0 (-> [8,7,6]) and
        // index 4 (-> [5,4,9]); most recent (index 4) should win.
        let history = vec![9, 8, 7, 6, 9, 5, 4, 9];
        assert_eq!(spec.propose(&history), vec![5, 4, 9]);
    }

    // ---- speculative_decode correctness tests ----

    #[test]
    fn speculative_decode_matches_greedy_token_for_token() {
        // The property that matters most: speculative decoding must be
        // quality-neutral. Build a prompt with a real repeated pattern
        // so the speculator actually proposes something non-trivial,
        // then check token-for-token identity against plain greedy
        // forward_token decoding on a separately constructed but
        // identically-seeded decoder.
        let cfg = tiny_test_config();
        let vocab = 8;
        let prompt = vec![1usize, 2, 3, 4, 1, 2];
        let max_new = 6;

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut caches_a: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_a.config.n_kv_heads, decoder_a.config.head_dim))
            .collect();
        let speculator = PromptLookupSpeculator::new(2, 3);
        let result = speculative_decode(&decoder_a, &prompt, max_new, &mut caches_a, &speculator);

        let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        let mut caches_b: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_b.config.n_kv_heads, decoder_b.config.head_dim))
            .collect();
        let mut pending = decoder_b
            .forward_batch(&prompt, 0, &mut caches_b)
            .pop()
            .unwrap();
        let mut greedy = Vec::with_capacity(max_new);
        for pos in (prompt.len()..).take(max_new) {
            let tok = argmax(&pending);
            greedy.push(tok);
            pending = decoder_b.forward_token(tok, pos, &mut caches_b);
        }

        assert_eq!(
            result.generated_tokens, greedy,
            "speculative decode must produce exactly the same tokens as plain greedy decode"
        );
    }

    #[test]
    fn speculative_decode_saves_real_calls_when_drafts_hit() {
        // Craft a scenario where the FIRST round's draft is guaranteed
        // to be checked against a real repeat (prompt = "A B A B",
        // ngram_size=2 will find "A B" repeating and propose whatever
        // came before, if anything did). This test doesn't assert the
        // draft is *accepted* (that depends on the random weights'
        // actual predictions, which this test doesn't control) --
        // it asserts the WEAKER but still meaningful property that
        // forward_calls is never more than max_new_tokens (i.e.
        // speculation is never worse than plain sequential decode) and
        // reports tokens_per_call for visibility.
        let cfg = tiny_test_config();
        let vocab = 8;
        let prompt = vec![1usize, 2, 3, 1, 2];
        let max_new = 8;

        let decoder = Decoder::new_random_small(cfg, 2, vocab);
        let mut caches: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let speculator = PromptLookupSpeculator::new(2, 4);
        let result = speculative_decode(&decoder, &prompt, max_new, &mut caches, &speculator);

        assert_eq!(result.tokens_generated, max_new);
        assert!(
            result.forward_calls <= max_new,
            "speculative decode must never need MORE forward_batch calls than plain sequential decode would (calls={}, tokens={})",
            result.forward_calls,
            max_new
        );
    }

    #[test]
    fn speculative_decode_with_no_repeats_falls_back_to_one_token_per_call() {
        // A prompt with no internal repeats at all (each token
        // distinct, ngram_size=3 means nothing can ever match) must
        // still work correctly, just without any speedup: forward_calls
        // should equal tokens_generated exactly (batch size 1 every
        // round), matching plain sequential decode's call count.
        let cfg = tiny_test_config();
        let vocab = 8;
        let prompt = vec![1usize, 2, 3];
        let max_new = 5;

        let decoder = Decoder::new_random_small(cfg, 2, vocab);
        let mut caches: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let speculator = PromptLookupSpeculator::new(10, 4); // ngram far longer than any possible history
        let result = speculative_decode(&decoder, &prompt, max_new, &mut caches, &speculator);

        assert_eq!(result.tokens_generated, max_new);
        assert_eq!(
            result.forward_calls,
            1 + max_new,
            "prefill (1 call) + one call per token when nothing ever matches"
        );
    }
}
