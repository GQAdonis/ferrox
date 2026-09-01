//! Sampler knobs a caller may send that this server does not implement,
//! refused BY NAME rather than dropped.
//!
//! Serde drops an undeclared field silently, and a caller cannot tell
//! that apart from having had it honoured: they get a 200 and an answer
//! computed under different rules than they asked for, which is worse
//! than an error. So the field is deserialized purely in order to be
//! refused.
//!
//! **One implementation, two routes.** `/v1/chat/completions` and
//! `/v1/completions` cannot share a request struct -- their bodies are
//! genuinely different shapes -- but they decide here, so they cannot
//! disagree about the same field again. `/v1/completions` refused
//! `logit_bias` from the start; `/v1/chat/completions` did not even
//! declare it, and that split is the defect this module closes.

use serde_json::Value;

use crate::{unsupported_feature, ApiError};

/// Refuse `logit_bias` by name, or accept a bias that would move
/// nothing.
///
/// ## Why refused rather than implemented
///
/// [`ferrox_models::sampling::Sampler::sample_with_mask`] already takes
/// an arbitrary `FnMut(&mut [f32])`, so *the sampler* could apply a bias
/// with no change to `sampling.rs`. Honouring it end to end is the part
/// that is not cheap, and each piece left out would be a fresh instance
/// of the same silent-wrong bug:
///
/// - The whole-response cache keys on the sampler settings
///   (`ChatCompletionRequest::cache_key`). A bias outside that key means
///   two requests differing only in their bias share one cached answer.
/// - The continuous-batching worker samples through its own call site,
///   so a bias wired into the private decode loop alone would be honoured
///   or ignored depending on `FERROX_CONTINUOUS_BATCHING`.
/// - On Metal at `temperature <= 0` the decoder folds `lm_head` and
///   argmax into the GPU stack and returns a *one-element* logits vector
///   holding the chosen id (see `generate::generate`'s greedy guard).
///   There is no vocabulary-shaped logit vector left to bias.
///
/// A refusal is coverage. Implementing it is tracked as its own item
/// (`docs/plans/llama-cpp-gap-inventory.md`, the `logit_bias` sampler
/// row), not smuggled in half-done.
///
/// `{}` is accepted: several OpenAI clients send an empty map on every
/// request as a default, and there is no token whose logit it would
/// have moved. Refusing it would be a false refusal.
pub(crate) fn refuse_logit_bias(value: Option<&Value>, route: &str) -> Result<(), ApiError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_null() || value.as_object().is_some_and(|m| m.is_empty()) {
        return Ok(());
    }
    Err(unsupported_feature(&format!(
        "`logit_bias` is not implemented on {route} (see docs/API.md). \
         It is refused rather than ignored: a dropped bias is \
         indistinguishable from an honoured one."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn a_real_bias_is_refused_by_name() {
        let bias = serde_json::json!({"50256": -100.0});
        let (status, body) =
            refuse_logit_bias(Some(&bias), "/v1/chat/completions").expect_err("refused");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let message = body["error"]["message"].as_str().expect("message");
        assert!(message.contains("logit_bias"), "{message}");
        assert!(message.contains("/v1/chat/completions"), "{message}");
    }

    /// An absent bias and a bias that would move nothing are the same
    /// request as far as the sampler is concerned. Refusing `{}` would
    /// reject clients that send it as a default on every call.
    #[test]
    fn an_empty_or_absent_bias_is_served() {
        assert!(refuse_logit_bias(None, "/v1/completions").is_ok());
        assert!(refuse_logit_bias(Some(&serde_json::json!({})), "/v1/completions").is_ok());
    }

    /// A bias that is not an object at all is still a bias the caller
    /// meant; it must not fall through the empty-map hole.
    #[test]
    fn a_malformed_bias_is_refused_rather_than_read_as_empty() {
        assert!(refuse_logit_bias(Some(&serde_json::json!([])), "/v1/completions").is_err());
        assert!(refuse_logit_bias(Some(&serde_json::json!("none")), "/v1/completions").is_err());
    }

    /// The defect this module exists for: `/v1/completions` refused
    /// `logit_bias` and `/v1/chat/completions` did not even declare it,
    /// so the SAME body got a 501 on one route and a 200 with unbiased
    /// output on the other. Asserted through both real request types,
    /// because a shared helper only helps if both routes call it.
    #[test]
    fn both_routes_answer_a_logit_bias_the_same_way() {
        for (bias, expected_refusal) in [
            (serde_json::json!({"50256": -100.0}), true),
            (serde_json::json!({}), false),
        ] {
            let chat: crate::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "logit_bias": bias,
            }))
            .expect("chat request");
            let completion: crate::openai_extra::CompletionsRequest =
                serde_json::from_value(serde_json::json!({
                    "prompt": "hi",
                    "logit_bias": bias,
                }))
                .expect("completions request");

            let chat_status = chat.validate_supported_fields().err().map(|(s, _)| s);
            let completion_status = completion.validate().err().map(|(s, _)| s);
            assert_eq!(
                chat_status, completion_status,
                "the two routes disagree about logit_bias {bias}"
            );
            assert_eq!(
                chat_status.is_some(),
                expected_refusal,
                "wrong verdict for logit_bias {bias}"
            );
        }
    }
}
