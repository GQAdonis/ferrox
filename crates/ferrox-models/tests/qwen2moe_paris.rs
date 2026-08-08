//! Greedy CPU regression for Qwen1.5-MoE-A2.7B Q4_K_M: prompt
//! `The capital of France is` must continue with `Paris`.
//!
//! Requires:
//! - `cargo test -p ferrox-models --test qwen2moe_paris -- --ignored`
//! - `models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf` (or `FERROX_QWEN2MOE_GGUF`)

use std::path::{Path, PathBuf};

use ferrox_core::cache::KvCache;
use ferrox_gguf::ShardedGguf;
use ferrox_models::config::ModelConfig;
use ferrox_models::decoder::Decoder;
use ferrox_models::tokenizer::GgufBpeTokenizer;

const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 16;

fn qwen2moe_gguf_path() -> PathBuf {
    if let Ok(p) = std::env::var("FERROX_QWEN2MOE_GGUF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf")
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
#[ignore = "needs models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf (or FERROX_QWEN2MOE_GGUF)"]
fn qwen2moe_cpu_greedy_paris_regression() {
    let path = qwen2moe_gguf_path();
    if !path.exists() {
        eprintln!("skip: Qwen2-MoE GGUF missing at {}", path.display());
        return;
    }

    let file = ShardedGguf::open(&path).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("from_gguf");
    eprintln!(
        "config: sliding_window={:?} expert_ffn_dim={} norm_topk_prob={} n_shared={}",
        config.sliding_window, config.moe.expert_ffn_dim, config.moe.norm_topk_prob, config.moe.n_shared_experts
    );
    let tok = GgufBpeTokenizer::from_gguf(&file).expect("tokenizer");
    let decoder = Decoder::from_gguf(&path, config.clone()).expect("load decoder");

    let mut tokens: Vec<usize> = tok
        .encode(PROMPT)
        .into_iter()
        .map(|t| t as usize)
        .collect();
    // llama.cpp qwen2 pre: add_bos=false — do not prepend bos_token_id.

    let mut caches: Vec<KvCache> = (0..config.n_layers)
        .map(|_| KvCache::new(config.n_kv_heads, config.head_dim))
        .collect();

    let logits_per_pos = decoder.forward_batch(&tokens, 0, &mut caches);
    let mut last_logits = logits_per_pos.last().expect("non-empty prompt").clone();

    let mut generated = Vec::with_capacity(MAX_NEW_TOKENS);
    for step in 0..MAX_NEW_TOKENS {
        let next = argmax(&last_logits);
        generated.push(next);
        let pos = tokens.len() + step;
        last_logits = decoder.forward_token(next, pos, &mut caches);
    }

    tokens.extend(&generated);
    let ids: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
    let text = tok.decode(&ids);
    eprintln!("decoded={text:?}");

    assert!(
        text.to_ascii_lowercase().contains("paris"),
        "expected Paris in greedy CPU continuation, got {text:?}"
    );
}
