//! The sampler knobs an OpenAI request body carries, resolved to
//! [`SamplingParams`] in exactly one place.
//!
//! `/v1/chat/completions` and `/v1/completions` cannot share a request
//! struct -- their bodies are genuinely different shapes -- so each
//! names its own wire fields. What they must NOT each own is the
//! mapping: which knobs exist, and what each one means when the caller
//! said nothing. That copy had already drifted. `/v1/completions`
//! hardcoded `top_k: 0` and `repetition_penalty: 1.0` while the chat
//! route read both off the request, so four real sampler fields were
//! dropped by serde on one route and honoured on the other -- the same
//! defect as `logit_bias`, four more times.
//!
//! So the knobs are one struct and the defaults are one function. A knob
//! added here is added to both routes at once, or to neither.

use ferrox_models::sampling::SamplingParams;

/// How many recent tokens the repetition/presence/frequency penalties
/// look at. llama.cpp's `penalty_last_n` default (`common/common.h:238`);
/// the OpenAI wire has no field for it on either route.
const DEFAULT_PENALTY_LAST_N: usize = 64;

/// One request's sampler knobs, each `None` when the caller said
/// nothing about it.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SamplingKnobs {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    /// llama.cpp's `--min-p`: keep only candidates at least this
    /// fraction as likely as the most likely one.
    ///
    /// `None` resolves to `0.0` (off), NOT to llama.cpp's CLI default of
    /// 0.05. A server that quietly truncated every request nobody
    /// configured would be a behaviour change no caller asked for, and
    /// an HTTP client is not running llama.cpp's command line. The
    /// number lives on the CLI flag, where the person who typed it can
    /// see it.
    pub(crate) min_p: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) repetition_penalty: Option<f32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
}

impl SamplingKnobs {
    /// Fill every gap the request left with this server's default.
    pub(crate) fn resolve(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature.unwrap_or(0.0),
            top_p: self.top_p.unwrap_or(1.0),
            min_p: self.min_p.unwrap_or(0.0),
            top_k: self.top_k.unwrap_or(0),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            penalty_last_n: DEFAULT_PENALTY_LAST_N,
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty request is greedy decoding with every filter off: the
    /// same "do nothing the caller did not ask for" baseline
    /// `SamplingParams::default` is.
    #[test]
    fn an_empty_request_resolves_to_the_do_nothing_baseline() {
        let resolved = SamplingKnobs::default().resolve();
        let baseline = SamplingParams::default();
        assert_eq!(resolved.temperature, baseline.temperature);
        assert_eq!(resolved.top_p, baseline.top_p);
        assert_eq!(resolved.min_p, baseline.min_p);
        assert_eq!(resolved.top_k, baseline.top_k);
        assert_eq!(resolved.repetition_penalty, baseline.repetition_penalty);
        assert_eq!(resolved.penalty_last_n, baseline.penalty_last_n);
    }

    /// Specifically: a server that shipped llama.cpp's 0.05 would
    /// truncate the distribution of every request nobody configured.
    #[test]
    fn min_p_defaults_to_off_rather_than_to_llama_cpps_cli_number() {
        assert_eq!(SamplingKnobs::default().resolve().min_p, 0.0);
        assert_eq!(
            SamplingKnobs {
                min_p: Some(0.05),
                ..SamplingKnobs::default()
            }
            .resolve()
            .min_p,
            0.05
        );
    }
}
