//! Minimal greedy decode used by [`crate::verify`].
//!
//! Deliberately independent of `run.rs`: this is a reference harness, and
//! it should not drift when the CLI's prompt handling, chat templating or
//! sampling changes. Raw prompt, no chat template, greedy argmax.

use anyhow::Context;
use ferrox_core::cache::KvCache;
use ferrox_gguf::ShardedGguf;
use ferrox_models::config::ModelConfig;
use ferrox_models::decoder::Decoder;
use ferrox_models::tokenizer::{GgufBpeTokenizer, GgufSpmTokenizer, GgufUnigramTokenizer};
use std::path::Path;

enum Tok {
    Bpe(Box<GgufBpeTokenizer>),
    Spm(GgufSpmTokenizer),
    Unigram(GgufUnigramTokenizer),
}

impl Tok {
    fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            Tok::Bpe(t) => t.encode(text).into_iter().map(|i| i as usize).collect(),
            Tok::Spm(t) => t.encode(text).into_iter().map(|i| i as usize).collect(),
            Tok::Unigram(t) => t.encode(text).into_iter().map(|i| i as usize).collect(),
        }
    }
}

/// Greedy-decodes `n` tokens from `prompt` and returns their ids plus
/// the prompt's tokenized length.
///
/// Stops early at EOS — a shorter sequence still compares correctly,
/// because the caller checks length as well as contents.
///
/// `prompt_tokens`, when set, stretches the prompt to exactly that many
/// tokens by repeating it (BOS kept once, at the front). This exists
/// because the prefill attention kernels — the ones this tool is meant
/// to catch — are gated on batch size (`n_q >= 8` for `fa_ext`), so a
/// short prompt verifies the decode path twice and the prefill path
/// never. Repeated text is fine: the check is kernel agreement, not
/// output quality.
pub fn greedy_token_ids(
    path: &Path,
    prompt: &str,
    n: usize,
    prompt_tokens: Option<usize>,
) -> anyhow::Result<(Vec<u32>, usize)> {
    let file = ShardedGguf::open(path)?;
    let config = ModelConfig::from_gguf(&file).context("reading model config")?;
    let tokenizer = match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => Tok::Bpe(Box::new(GgufBpeTokenizer::from_gguf(&file)?)),
        Some("llama") => Tok::Spm(GgufSpmTokenizer::from_gguf(&file)?),
        Some("t5") => Tok::Unigram(GgufUnigramTokenizer::from_gguf(&file)?),
        other => anyhow::bail!("verify does not cover tokenizer {other:?}"),
    };
    let eos = file
        .metadata_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize);
    let bos = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);

    let decoder = Decoder::from_gguf(path, config)?;

    let mut tokens = tokenizer.encode(prompt);
    if ferrox_models::tokenizer::should_add_bos_token(&file) {
        if let Some(b) = bos {
            if tokens.first() != Some(&b) {
                tokens.insert(0, b);
            }
        }
    }
    if tokens.is_empty() {
        anyhow::bail!("prompt tokenized to nothing");
    }
    if let Some(want) = prompt_tokens {
        tokens = stretch_prompt(tokens, want, bos)?;
    }

    let mut caches: Vec<KvCache> = (0..decoder.layers.len())
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect();

    let prompt_len = tokens.len();
    let mut logits = decoder.forward_batch_last(&tokens, 0, &mut caches);
    let mut pos = tokens.len();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let next = argmax(&logits);
        out.push(next as u32);
        if Some(next) == eos {
            break;
        }
        logits = decoder.forward_token(next, pos, &mut caches);
        pos += 1;
    }
    Ok((out, prompt_len))
}

/// Repeat `tokens` until it is exactly `want` long, keeping at most one
/// leading BOS. Errors rather than silently shortening when `want` is
/// smaller than the tokenized prompt, so `--prompt-tokens` never reports
/// a length the run did not use.
fn stretch_prompt(
    tokens: Vec<usize>,
    want: usize,
    bos: Option<usize>,
) -> anyhow::Result<Vec<usize>> {
    if want == 0 {
        anyhow::bail!("--prompt-tokens must be at least 1");
    }
    if want < tokens.len() {
        anyhow::bail!(
            "--prompt-tokens {want} is shorter than the tokenized prompt ({}); \
             pass a shorter --prompt instead",
            tokens.len()
        );
    }
    let leading_bos = (bos.is_some() && tokens.first().copied() == bos).then(|| tokens[0]);
    let body = &tokens[leading_bos.iter().count()..];
    if body.is_empty() {
        anyhow::bail!("prompt is BOS only; nothing to repeat");
    }
    let mut out = Vec::with_capacity(want);
    out.extend(leading_bos);
    while out.len() < want {
        let take = (want - out.len()).min(body.len());
        out.extend_from_slice(&body[..take]);
    }
    Ok(out)
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::stretch_prompt;

    #[test]
    fn stretch_repeats_the_body_and_keeps_one_bos() {
        // [BOS, a, b, c] stretched to 8 -> BOS once, body cycled.
        let got = stretch_prompt(vec![1, 10, 11, 12], 8, Some(1)).unwrap();
        assert_eq!(got, vec![1, 10, 11, 12, 10, 11, 12, 10]);
        assert_eq!(got.len(), 8);
        assert_eq!(got.iter().filter(|&&t| t == 1).count(), 1);
    }

    #[test]
    fn stretch_without_bos_cycles_the_whole_prompt() {
        assert_eq!(
            stretch_prompt(vec![7, 8], 5, None).unwrap(),
            vec![7, 8, 7, 8, 7]
        );
    }

    #[test]
    fn stretch_is_a_no_op_at_the_current_length() {
        assert_eq!(
            stretch_prompt(vec![1, 4, 5], 3, Some(1)).unwrap(),
            vec![1, 4, 5]
        );
    }

    #[test]
    fn stretch_refuses_to_shorten_or_empty_a_prompt() {
        // Silently truncating would report a prompt length the run did
        // not use, which is exactly the vacuous-pass bug this flag fixes.
        assert!(stretch_prompt(vec![1, 4, 5, 6], 2, Some(1)).is_err());
        assert!(stretch_prompt(vec![1, 4], 0, Some(1)).is_err());
        assert!(stretch_prompt(vec![1], 8, Some(1)).is_err());
    }

    #[test]
    fn a_repeated_token_that_equals_bos_is_only_stripped_at_the_front() {
        // Body containing the BOS id mid-prompt must survive: only a
        // leading BOS is special.
        let got = stretch_prompt(vec![1, 9, 1, 9], 6, Some(1)).unwrap();
        assert_eq!(got, vec![1, 9, 1, 9, 9, 1]);
    }
}
