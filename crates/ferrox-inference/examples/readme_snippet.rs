//! The library example from the repository README, compiled.
//!
//! Documentation examples rot silently: nothing fails when a signature
//! changes under them, so the first person to notice is a reader who
//! pastes it and gets a compile error. This file is the same snippet,
//! built by `cargo check --workspace --all-targets`, so that stops
//! being possible.
//!
//! It is never RUN in CI -- it wants a real checkpoint on disk -- so
//! keep it to API shape rather than behaviour. If you edit the README
//! example, edit this too; they are meant to be the same text.

#![allow(unused)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ferrox_inference::core::cache::KvCache;
    use ferrox_inference::gguf::ShardedGguf;
    use ferrox_inference::models::{Decoder, ModelConfig};

    let path = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

    // Inspect a checkpoint without loading a single weight: GGUF metadata
    // and tensor descriptors come from the header.
    let file = ShardedGguf::open(path)?;
    println!(
        "{} tensors, arch {:?}",
        file.tensor_count(),
        file.metadata_str("general.architecture"),
    );

    // Hyperparameters are derived from the file, not hand-written. Fields
    // that had to be guessed are listed in `config.best_effort_fields`.
    let config = ModelConfig::from_gguf(&file)?;

    // Load. Weights stay quantized and mmap-resident; dequant happens
    // inside the matmul. This REFUSES a checkpoint carrying tensors this
    // build never reads, rather than computing a different graph.
    let decoder = Decoder::from_gguf(path, config)?;

    // Prefill a prompt and read the logits at its last position.
    let mut caches: Vec<KvCache> = (0..decoder.layers.len())
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect();
    let logits = decoder.forward_batch_last(&[1, 2, 3], 0, &mut caches);

    Ok(())
}
