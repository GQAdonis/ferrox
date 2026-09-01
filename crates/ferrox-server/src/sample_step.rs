//! One token, chosen under every per-request constraint on the logits.
//!
//! There are two decode loops in this server -- the private
//! `generate::sample_until_stop` loop and the continuous batcher's
//! per-row step in `serving::batch::worker` -- and for as long as each
//! held its own call to `Sampler`, they disagreed about what a request
//! meant. `response_format: {"type": "json_object"}` was honoured on one
//! and dropped on the other, so the same body got structured output or
//! not depending on `FERROX_CONTINUOUS_BATCHING`: an env var the caller
//! cannot see deciding a per-request feature.
//!
//! So the choice lives here once and both loops call it. Adding a
//! constraint means adding it to [`sample_next`], and it is then in
//! effect on both paths or on neither -- which is the only way the two
//! can be made to agree about a constraint that does not exist yet.

use ferrox_models::sampling::Sampler;

use crate::generate::GenerationParams;
use crate::json_mode::mask_logits_for_json;

/// Sample the next token id from `logits` under `params`.
///
/// `decode_token` renders one vocabulary entry as text. It is required
/// rather than optional on purpose: the previous shape took an
/// `Option`, and the `None` arm silently fell through to *unconstrained*
/// sampling for a request that had asked to be constrained. Both callers
/// already hold a tokenizer, so there is no arm to fall through to.
pub(crate) fn sample_next(
    sampler: &mut Sampler,
    logits: &[f32],
    params: &GenerationParams,
    history: &[usize],
    decode_token: &dyn Fn(usize) -> String,
) -> usize {
    if !params.needs_vocab_logits() {
        return sampler.sample(logits, &params.sampling, history);
    }
    // A backend that folded lm_head+argmax onto the device returns one
    // element holding the chosen id, not a vocabulary, and masking it
    // would zero a token id rather than a logit. That is unreachable:
    // `generate::greedy_gpu_fold_allowed` refuses the fold for exactly
    // the requests that reach this branch. Asserted rather than assumed,
    // because the failure it guards is silent -- plausible-looking
    // unstructured text, served with a 200.
    debug_assert!(
        logits.len() > 1,
        "a request needing vocabulary-shaped logits was handed {} of them; \
         greedy_gpu_fold_allowed and needs_vocab_logits have drifted apart",
        logits.len()
    );
    let mut mask = |scores: &mut [f32]| mask_logits_for_json(scores, decode_token);
    sampler.sample_with_mask(logits, &params.sampling, history, Some(&mut mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_models::sampling::SamplingParams;

    /// Token 0 renders as something no JSON document may contain;
    /// token 1 is a brace. Any masked sampler must refuse 0.
    fn decode_token(id: usize) -> String {
        match id {
            0 => "<html>".to_string(),
            1 => "{".to_string(),
            _ => "0".to_string(),
        }
    }

    fn params(json_object: bool, temperature: f32) -> GenerationParams {
        GenerationParams {
            max_tokens: 8,
            sampling: SamplingParams {
                temperature,
                ..SamplingParams::default()
            },
            seed: 7,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            json_object,
            cancel: None,
            ignore_eos: false,
        }
    }

    /// The assertion neither decode path had. Token 0 wins the argmax by
    /// a mile and is not JSON-safe; a masked greedy sample must not
    /// return it. `temperature: 0` because that is how callers actually
    /// ask for structured output, and it is the case the Metal greedy
    /// fold used to break.
    #[test]
    fn json_mode_masks_at_temperature_zero() {
        let logits = vec![9.0, 1.0, 0.5];
        let mut sampler = Sampler::new(7);
        let chosen = sample_next(
            &mut sampler,
            &logits,
            &params(true, 0.0),
            &[],
            &decode_token,
        );
        assert_ne!(chosen, 0, "a non-JSON-safe token was sampled in json mode");
        assert_eq!(chosen, 1);
    }

    /// Same logits, sampled rather than greedy: the mask is not a
    /// greedy-only path.
    #[test]
    fn json_mode_masks_when_sampling_too() {
        let logits = vec![9.0, 1.0, 0.5];
        for seed in 0..16u64 {
            let mut sampler = Sampler::new(seed);
            let chosen = sample_next(
                &mut sampler,
                &logits,
                &params(true, 1.0),
                &[],
                &decode_token,
            );
            assert_ne!(chosen, 0, "seed {seed} sampled a non-JSON-safe token");
        }
    }

    /// And the mask is not applied to a request that never asked for it:
    /// the unconstrained argmax is token 0.
    #[test]
    fn a_plain_request_is_not_masked() {
        let logits = vec![9.0, 1.0, 0.5];
        let mut sampler = Sampler::new(7);
        let chosen = sample_next(
            &mut sampler,
            &logits,
            &params(false, 0.0),
            &[],
            &decode_token,
        );
        assert_eq!(chosen, 0);
    }

    /// The coupling that keeps [`sample_next`]'s `debug_assert` true,
    /// checked in the default build because the fold it constrains is
    /// `#[cfg(feature = "metal")]`.
    ///
    /// `temperature <= 0` alone used to permit the fold. It must not: a
    /// JSON-mode request at temperature 0 would then be handed a
    /// one-element vector and masked into nothing.
    #[test]
    fn a_json_mode_request_may_never_fold_lm_head_into_a_gpu_argmax() {
        use crate::generate::greedy_gpu_fold_allowed;

        assert!(greedy_gpu_fold_allowed(&params(false, 0.0)));
        assert!(!greedy_gpu_fold_allowed(&params(true, 0.0)));
        // Not greedy: never folded either way, whatever the constraints.
        assert!(!greedy_gpu_fold_allowed(&params(false, 0.8)));
        assert!(!greedy_gpu_fold_allowed(&params(true, 0.8)));
    }
}
