//! Print MoE packed quant kinds per layer for Qwen1.5-MoE.

use std::path::{Path, PathBuf};

use ferrox_gguf::ShardedGguf;
use ferrox_models::config::ModelConfig;
use ferrox_models::decoder::Decoder;

fn qwen2moe_gguf_path() -> PathBuf {
    if let Ok(p) = std::env::var("FERROX_TEST_QWEN2MOE_GGUF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf")
}

#[test]
#[ignore = "needs Qwen1.5-MoE GGUF"]
fn qwen15_moe_packed_kinds_per_layer() {
    let path = qwen2moe_gguf_path();
    if !path.exists() {
        eprintln!("skip: {}", path.display());
        return;
    }
    let file = ShardedGguf::open(&path).expect("open");
    let config = ModelConfig::from_gguf(&file).expect("config");
    let decoder = Decoder::from_gguf(&path, config).expect("decoder");
    for (i, layer) in decoder.layers.iter().enumerate() {
        match &layer.moe.packed_q4 {
            Some(p) => {
                let v = p.view();
                eprintln!(
                    "layer {i}: gate={} up={} down={} experts={}",
                    v.gate_kind, v.up_kind, v.down_kind, v.n_experts
                );
            }
            None => eprintln!("layer {i}: no packed_q4"),
        }
    }
}
