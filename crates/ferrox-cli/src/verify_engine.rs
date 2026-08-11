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

/// Greedy-decodes `n` tokens from `prompt` and returns their ids.
///
/// Stops early at EOS — a shorter sequence still compares correctly,
/// because the caller checks length as well as contents.
pub fn greedy_token_ids(path: &Path, prompt: &str, n: usize) -> anyhow::Result<Vec<u32>> {
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

    let mut caches: Vec<KvCache> = (0..decoder.layers.len())
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect();

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
