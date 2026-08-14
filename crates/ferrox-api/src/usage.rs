//! OpenAI-convention token accounting plus llama.cpp-style timings.
//!
//! Counted from the exact token ids the generation loop processed
//! (prompt after BOS insertion, and every generated id), not
//! re-tokenized after the fact -- re-tokenizing decoded text is not
//! guaranteed to round-trip to the same count.
//!
//! Why the server reports timings at all, when a client can hold a
//! stopwatch: the client's stopwatch measures the network, the proxy's
//! buffer and its own event loop. More importantly it cannot separate
//! **prefill from decode**, and a UI that divides total tokens by total
//! wall time reports a 50 tok/s model as 5 tok/s whenever the prompt is
//! long. Every downstream number built on that is then wrong in the
//! same direction. So the phases are reported separately and the client
//! is never asked to infer one from the other.
//!
//! Every timing is optional: a cached response, a batched decode, or an
//! engine path that does not time itself must be able to answer
//! honestly rather than emit a plausible zero.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Prefill throughput (prompt tokens / prefill seconds), when timed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_per_second: Option<f64>,
    /// Decode throughput (completion tokens / decode seconds), when timed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_per_second: Option<f64>,
    /// Wall time spent processing the prompt, in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration_ms: Option<f64>,
    /// Wall time spent in the decode loop, in milliseconds. Kept
    /// separate from `prompt_eval_duration_ms` on purpose (see the
    /// module docs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_duration_ms: Option<f64>,
    /// Time to first token: from the start of prefill to the moment the
    /// first token was produced. `None` when no token was produced at
    /// all (an immediate EOS), because a zero there would read as an
    /// instantaneous response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_ms: Option<f64>,
    /// Prompt tokens served from the KV prefix cache instead of being
    /// recomputed. `Some(0)` means "the cache was consulted and missed";
    /// `None` means "no prefix cache is configured" -- a distinction the
    /// UI needs to decide whether to show the row at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_per_second: None,
            predicted_per_second: None,
            prompt_eval_duration_ms: None,
            generation_duration_ms: None,
            time_to_first_token_ms: None,
            cached_tokens: None,
        }
    }

    /// Records the two phase durations, in seconds, and the rates they
    /// imply. A zero-length phase leaves the rate unset rather than
    /// dividing by zero into infinity.
    pub fn with_timings(mut self, prompt_secs: f64, predicted_secs: f64) -> Self {
        self.prompt_eval_duration_ms = Some(prompt_secs * 1000.0);
        self.generation_duration_ms = Some(predicted_secs * 1000.0);
        if prompt_secs > 0.0 && self.prompt_tokens > 0 {
            self.prompt_per_second = Some(self.prompt_tokens as f64 / prompt_secs);
        }
        if predicted_secs > 0.0 && self.completion_tokens > 0 {
            self.predicted_per_second = Some(self.completion_tokens as f64 / predicted_secs);
        }
        self
    }

    /// Time-to-first-token, in seconds, measured from the start of
    /// prefill. Ignored when no token was generated.
    pub fn with_ttft(mut self, secs: f64) -> Self {
        if self.completion_tokens > 0 {
            self.time_to_first_token_ms = Some(secs * 1000.0);
        }
        self
    }

    /// Prompt tokens that came from the prefix cache. Call this only
    /// when a prefix cache actually exists (see `cached_tokens`).
    pub fn with_cached_tokens(mut self, cached: usize) -> Self {
        self.cached_tokens = Some(cached);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn totals_are_the_sum_of_the_two_phases() {
        let usage = Usage::new(7, 3);
        assert_eq!(usage.total_tokens, 10);
    }

    #[test]
    fn phase_durations_stay_separate() {
        // 100 prompt tokens in 1s, 10 generated in 1s. A client that
        // conflated the phases would report 110/2 = 55 tok/s for both.
        let usage = Usage::new(100, 10).with_timings(1.0, 1.0);
        assert_eq!(usage.prompt_per_second, Some(100.0));
        assert_eq!(usage.predicted_per_second, Some(10.0));
        assert_eq!(usage.prompt_eval_duration_ms, Some(1000.0));
        assert_eq!(usage.generation_duration_ms, Some(1000.0));
    }

    #[test]
    fn zero_length_phases_do_not_become_infinite_rates() {
        let usage = Usage::new(5, 5).with_timings(0.0, 0.0);
        assert_eq!(usage.prompt_per_second, None);
        assert_eq!(usage.predicted_per_second, None);
        assert_eq!(usage.prompt_eval_duration_ms, Some(0.0));
    }

    #[test]
    fn ttft_is_unset_when_nothing_was_generated() {
        let usage = Usage::new(5, 0).with_ttft(0.25);
        assert_eq!(usage.time_to_first_token_ms, None);
        assert_eq!(
            Usage::new(5, 1).with_ttft(0.25).time_to_first_token_ms,
            Some(250.0)
        );
    }

    #[test]
    fn untimed_usage_serializes_to_the_plain_openai_shape() {
        // Older clients must not start seeing null-valued extras.
        let json = serde_json::to_string(&Usage::new(2, 3)).unwrap();
        assert_eq!(
            json,
            "{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}"
        );
    }

    #[test]
    fn a_prefix_cache_miss_is_distinguishable_from_no_prefix_cache() {
        assert_eq!(Usage::new(2, 3).cached_tokens, None);
        assert_eq!(
            Usage::new(2, 3).with_cached_tokens(0).cached_tokens,
            Some(0)
        );
    }
}
