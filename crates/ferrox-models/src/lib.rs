//! ferrox-models: GGUF decoder, architecture registry, and structural
//! presets for frontier stacks (GLM / DeepSeek V4 / Kimi).
//!
//! Unconfirmed preset fields go in `best_effort_fields` and must be
//! overwritten from real `config.json` / GGUF metadata. Status of what
//! actually runs: `docs/MODELS.md`. Presets `glm_5_2` / `deepseek_v4_pro`
//! / `kimi_k3` are sketches for synthetic tests — not real-checkpoint
//! support. Dedicated primitives live in `glm52_*`, `deepseek_v4_*`,
//! `kimi_*` modules.

pub mod block_residual;
pub mod capability;
pub mod config;
pub mod decoder;
pub mod deepseek_v4_decoder;
pub mod engine;
pub mod engine_factory;
pub mod execution_plan;
pub mod glm52_decoder;
pub mod glm52_gguf_loader;
pub mod gdn;
pub mod hf_pull;
pub mod glm_dsa;
pub mod hybrid_engine;
pub mod hybrid_gguf_loader;
pub mod hyper_connections;
pub mod kda;
pub mod kimi_decoder;
pub mod kimi_generate;
pub mod kimi_gguf_loader;
pub mod kimi_loader;
pub mod kimi_tokenizer;
pub mod kimi_validate;
pub mod latent_moe;
pub mod llama4_engine;
pub mod loader;
pub mod minimax_engine;
pub mod mla;
pub mod mla_gguf_loader;
pub mod mmproj;
pub mod output_projection;
pub mod prefix_cache;
pub mod recurrent_engine;
pub mod residency_report;
pub mod sampling;
pub mod speculative;
pub mod t5_engine;
pub mod tensor_role;
pub mod tokenizer;
pub mod vision;
pub mod vl_engine;

pub use capability::{
    architecture_catalog, coverage_report_markdown, resolve_architecture, resolve_profile,
    ArchPath, ArchProfile, ArchScope, DecoderFamily, MemoryKind, QkNormStyle,
};
pub use config::{deepseek_v4_pro, glm_5_2, kimi_k3, FfnActivation, ModelConfig, RopeLayout};
pub use decoder::Decoder;
pub use engine::{
    DeepseekV4Engine, Engine, Glm52Engine, KimiEngine, MlaEngine, MlaLayerWeights, TextTokenizer,
};
pub use engine_factory::{
    ensure_generic_decoder, load_glm52_engine_from_path, load_mla_engine_from_path,
    select_engine_kind, EngineSelectError, SelectedEngineKind, ServedEngine,
};
pub use execution_plan::{ExecutionPlan, FusedOpCaps, MemoryPlan, PlanGeometry};
pub use loader::LoadError;
pub use output_projection::grouped_output_projection;
pub use prefix_cache::{PrefixCache, PrefixCacheStats, PrefixMatch};
pub use sampling::{Sampler, SamplingParams};
pub use speculative::{speculative_decode, PromptLookupSpeculator, SpeculativeDecodeResult};
pub use tensor_role::TensorRole;
pub use tokenizer::{
    ByteTokenizer, GgufBpeTokenizer, GgufSpmTokenizer, GgufUnigramTokenizer, TokenizerLoadError,
};

#[cfg(feature = "metal")]
pub use ferrox_metal::attn::{
    metal_greedy_argmax_active, metal_greedy_gpu_enabled, set_metal_greedy_argmax,
};
