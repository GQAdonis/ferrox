//! Loads the model `ferrox-server` actually serves.
//!
//! If `FERROX_MODEL_PATH` is set, this loads a real checkpoint: a
//! `.gguf` **file** through either the generic `Decoder` path or
//! [`ferrox_models::load_mla_engine_from_path`] for MLA architectures
//! (deepseek2 / mistral4 dense-lead), or [`ferrox_models::load_glm52_engine_from_path`]
//! for GLM-5.2 / GLM4-family GGUFs (`glm-dsa`, `glm4`, `glm4moe`) when
//! the file itself names via `tokenizer.ggml.model`
//! (`gpt2` -> `GgufBpeTokenizer`, `llama` -> `GgufSpmTokenizer`, `t5` ->
//! `GgufUnigramTokenizer`); or a Kimi K3 checkpoint **directory** (real
//! `model.safetensors.index.json` + shards, `tiktoken.model`, optional
//! `tokenizer_config.json`) through `kimi_loader::load_kimi_checkpoint`
//! and `KimiTokenizer`, wrapped in the generic `ferrox_models::Engine`/
//! `TextTokenizer` traits (see `engine` module) so `ferrox-server` can
//! serve either through the same `/v1/chat/completions` endpoint (see
//! `LoadedModel`/`crate::Model` in `main.rs`). A GGUF checkpoint is a
//! single file; a Kimi K3 checkpoint is always a directory (its shards,
//! index, and tokenizer files can't be one file) -- `FERROX_MODEL_PATH`
//! pointing at a directory vs. a file is what selects between them.
//!
//! If `FERROX_MODEL_PATH` is unset, falls back to the previous
//! synthetic-weight demo behavior (small random weights for one of the
//! three built-in presets, byte tokenizer, always the GGUF-shaped path)
//! so the server still starts and demonstrates the request/response
//! pipeline without requiring a checkpoint on disk -- but logs a loud
//! warning that this is not real model output.

use std::path::Path;

use ferrox_gguf::ShardedGguf;
use ferrox_models::{
    deepseek_v4_pro, glm_5_2, kimi_k3, load_gemma4_engine_from_path, load_glm52_engine_from_path,
    load_mla_engine_from_path, select_engine_kind, ByteTokenizer, Decoder, Gemma4Engine,
    GgufBpeTokenizer, GgufSpmTokenizer, GgufUnigramTokenizer, Glm52Engine, KimiEngine, MlaEngine,
    ModelConfig, SelectedEngineKind, ServedEngine, TextTokenizer,
};

/// Default expert-cache budget when `FERROX_SSD_STREAMING` is set without
/// an explicit `FERROX_EXPERT_CACHE_BYTES`.
const DEFAULT_SSD_STREAMING_CACHE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Resolves the opt-in expert-streaming cache budget from
/// `FERROX_EXPERT_CACHE_BYTES` and/or `FERROX_SSD_STREAMING`.
fn resolve_expert_cache_bytes() -> anyhow::Result<Option<u64>> {
    let ssd_streaming = env_truthy("FERROX_SSD_STREAMING");
    let explicit = match std::env::var("FERROX_EXPERT_CACHE_BYTES") {
        Ok(v) => Some(v.parse::<u64>().map_err(|_| {
            anyhow::anyhow!("FERROX_EXPERT_CACHE_BYTES must be a non-negative integer, got {v:?}")
        })?),
        Err(_) => None,
    };
    if ssd_streaming {
        let budget = explicit.unwrap_or(DEFAULT_SSD_STREAMING_CACHE_BYTES);
        tracing::info!(
            "SSD streaming mode: expert cache streaming enabled with bounded cache of {budget} bytes"
        );
        Ok(Some(budget))
    } else if let Some(b) = explicit {
        tracing::info!("expert streaming enabled: bounded cache of {b} bytes");
        Ok(Some(b))
    } else {
        Ok(None)
    }
}

/// Whichever real tokenizer a loaded GGUF file's own metadata named, or
/// the byte-level fallback when no real vocabulary is available.
pub enum ServerTokenizer {
    // Boxed: GgufBpeTokenizer's merge-rank table makes it far larger
    // than the other variants (~1.2KB vs Spm's ~96 bytes vs Byte's 0),
    // so leaving it unboxed would size every ServerTokenizer value --
    // even Byte -- to the size of the largest variant.
    Bpe(Box<GgufBpeTokenizer>),
    Spm(GgufSpmTokenizer),
    Unigram(GgufUnigramTokenizer),
    Byte,
}

impl ServerTokenizer {
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            ServerTokenizer::Bpe(t) => t.encode(text).into_iter().map(|id| id as usize).collect(),
            ServerTokenizer::Spm(t) => t.encode(text).into_iter().map(|id| id as usize).collect(),
            ServerTokenizer::Unigram(t) => {
                t.encode(text).into_iter().map(|id| id as usize).collect()
            }
            ServerTokenizer::Byte => ByteTokenizer::encode(text)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
        }
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        match self {
            ServerTokenizer::Bpe(t) => t.decode(&ids32),
            ServerTokenizer::Spm(t) => t.decode(&ids32),
            ServerTokenizer::Unigram(t) => t.decode(&ids32),
            ServerTokenizer::Byte => ByteTokenizer::decode(&ids32),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            ServerTokenizer::Bpe(_) => "gguf-bpe",
            ServerTokenizer::Spm(_) => "gguf-spm",
            ServerTokenizer::Unigram(_) => "gguf-unigram",
            ServerTokenizer::Byte => "byte (no real vocabulary loaded)",
        }
    }
}

impl TextTokenizer for ServerTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        ServerTokenizer::encode(self, text)
    }

    fn decode(&self, ids: &[usize]) -> String {
        ServerTokenizer::decode(self, ids)
    }
}

fn tokenizer_from_gguf(file: &ShardedGguf) -> anyhow::Result<ServerTokenizer> {
    match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2") => Ok(ServerTokenizer::Bpe(Box::new(GgufBpeTokenizer::from_gguf(
            file,
        )?))),
        Some("llama") => Ok(ServerTokenizer::Spm(GgufSpmTokenizer::from_gguf(file)?)),
        // "t5" is the real GGUF tag for a SentencePiece Unigram (ULM)
        // vocabulary -- confirmed against llama.cpp's own real
        // vocab-type-loading source, not guessed (see
        // GgufUnigramTokenizer's doc comment).
        Some("t5") => Ok(ServerTokenizer::Unigram(GgufUnigramTokenizer::from_gguf(
            file,
        )?)),
        other => {
            tracing::warn!(
                "unrecognized or missing tokenizer.ggml.model ({other:?}) -- falling back to \
                 a raw byte tokenizer, which will not match this checkpoint's real vocabulary"
            );
            Ok(ServerTokenizer::Byte)
        }
    }
}

pub struct GgufLoaded {
    pub decoder: Decoder,
    pub tokenizer: ServerTokenizer,
    pub eos_id: Option<usize>,
    pub bos_id: Option<usize>,
    /// True when this is the synthetic-random-weights fallback rather
    /// than a real checkpoint -- surfaced in `/v1/models` and logged so
    /// nobody mistakes demo output for a real completion.
    pub is_synthetic: bool,
    /// Detected from the GGUF's own `tokenizer.chat_template` metadata
    /// (see `crate::chat_template`); `Plain` for the synthetic fallback,
    /// since there's no real checkpoint to carry one.
    pub chat_template: crate::chat_template::ChatTemplate,
}

pub struct KimiLoaded {
    pub engine: KimiEngine,
    pub tokenizer: ferrox_models::kimi_tokenizer::KimiTokenizer,
    pub eos_id: Option<usize>,
    pub chat_template: crate::chat_template::ChatTemplate,
}

pub struct MlaLoaded {
    pub engine: MlaEngine,
    pub tokenizer: ServerTokenizer,
    pub eos_id: Option<usize>,
    pub bos_id: Option<usize>,
    pub name: String,
    pub chat_template: crate::chat_template::ChatTemplate,
}

pub struct Gemma4Loaded {
    pub engine: Gemma4Engine,
    pub tokenizer: ServerTokenizer,
    pub eos_id: Option<usize>,
    pub bos_id: Option<usize>,
    pub name: String,
    pub chat_template: crate::chat_template::ChatTemplate,
}

pub struct Glm52Loaded {
    pub engine: Glm52Engine,
    pub tokenizer: ServerTokenizer,
    pub eos_id: Option<usize>,
    pub bos_id: Option<usize>,
    pub name: String,
    pub chat_template: crate::chat_template::ChatTemplate,
}

/// Either real checkpoint shape this server can serve. See this
/// module's doc comment for how `load()` picks between them.
#[allow(clippy::large_enum_variant)]
pub enum LoadedModel {
    Gguf(GgufLoaded),
    Kimi(KimiLoaded),
    Mla(MlaLoaded),
    Gemma4(Gemma4Loaded),
    Glm52(Glm52Loaded),
}

fn is_glm52_arch(arch: &str) -> bool {
    matches!(arch, "glm-dsa" | "glm4" | "glm4moe")
}

pub fn load() -> anyhow::Result<LoadedModel> {
    match std::env::var("FERROX_MODEL_PATH") {
        Ok(path) => {
            let path = ferrox_models::hf_pull::resolve_model_path(&path)?;
            if Path::new(&path).is_dir() {
                load_real_kimi_checkpoint(&path).map(LoadedModel::Kimi)
            } else {
                load_gguf_file(&path)
            }
        }
        Err(_) => {
            let preset = std::env::var("FERROX_PRESET").unwrap_or_else(|_| "glm-5.2".to_string());
            tracing::warn!(
                "FERROX_MODEL_PATH not set -- serving synthetic random weights for preset \
                 '{preset}' with a byte tokenizer, not a real checkpoint. Set \
                 FERROX_MODEL_PATH=/path/to/model.gguf (or a Kimi K3 checkpoint directory) to \
                 serve a real model."
            );
            Ok(LoadedModel::Gguf(GgufLoaded {
                decoder: build_synthetic_decoder(&preset)?,
                tokenizer: ServerTokenizer::Byte,
                eos_id: None,
                bos_id: None,
                is_synthetic: true,
                chat_template: crate::chat_template::ChatTemplate::Plain,
            }))
        }
    }
}

fn load_gguf_file(path: &str) -> anyhow::Result<LoadedModel> {
    let file = ShardedGguf::open(path)?;
    if file.shard_count() > 1 {
        tracing::info!(
            "split GGUF checkpoint: {} shards, {} tensors total",
            file.shard_count(),
            file.tensor_count()
        );
    }
    let arch = file.metadata_str("general.architecture");
    ferrox_models::mmproj::warn_mmproj_if_present(Path::new(path), arch);
    if let Some(arch) = arch {
        if is_glm52_arch(arch) {
            return load_glm52_checkpoint(path, &file).map(LoadedModel::Glm52);
        }
        if matches!(select_engine_kind(arch), Ok(SelectedEngineKind::Mla)) {
            return load_mla_checkpoint(path, &file).map(LoadedModel::Mla);
        }
        if matches!(select_engine_kind(arch), Ok(SelectedEngineKind::Gemma4)) {
            return load_gemma4_checkpoint(path, &file).map(LoadedModel::Gemma4);
        }
    }
    load_real_gguf_checkpoint(path, &file).map(LoadedModel::Gguf)
}

fn load_glm52_checkpoint(path: &str, file: &ShardedGguf) -> anyhow::Result<Glm52Loaded> {
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("glm-dsa")
        .to_string();
    let name = file
        .metadata_str("general.name")
        .unwrap_or(arch.as_str())
        .to_string();
    tracing::info!("loading GGUF as GLM-5.2 engine (arch={arch}, name={name})");
    let served =
        load_glm52_engine_from_path(Path::new(path)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Glm52(engine) = served else {
        anyhow::bail!("expected ServedEngine::Glm52 for architecture {arch}");
    };

    let tokenizer = tokenizer_from_gguf(file)?;
    let eos_id = file
        .metadata_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize);
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);
    let byte_tokenizer = matches!(tokenizer, ServerTokenizer::Byte);
    let chat_template = crate::chat_template::ChatTemplate::detect_for_gguf(
        file.metadata_str("tokenizer.chat_template"),
        Some(arch.as_str()),
        byte_tokenizer,
    );
    tracing::info!("detected chat template: {chat_template:?}");

    Ok(Glm52Loaded {
        engine,
        tokenizer,
        eos_id,
        bos_id,
        name,
        chat_template,
    })
}

fn load_mla_checkpoint(path: &str, file: &ShardedGguf) -> anyhow::Result<MlaLoaded> {
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("mla")
        .to_string();
    let name = file
        .metadata_str("general.name")
        .unwrap_or(arch.as_str())
        .to_string();
    tracing::info!("loading GGUF as MLA engine (arch={arch}, name={name})");
    let served = load_mla_engine_from_path(Path::new(path)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Mla(engine) = served else {
        anyhow::bail!("expected ServedEngine::Mla for architecture {arch}");
    };

    let tokenizer = tokenizer_from_gguf(file)?;
    let eos_id = file
        .metadata_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize);
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);
    let byte_tokenizer = matches!(tokenizer, ServerTokenizer::Byte);
    let chat_template = crate::chat_template::ChatTemplate::detect_for_gguf(
        file.metadata_str("tokenizer.chat_template"),
        Some(arch.as_str()),
        byte_tokenizer,
    );
    tracing::info!("detected chat template: {chat_template:?}");

    Ok(MlaLoaded {
        engine,
        tokenizer,
        eos_id,
        bos_id,
        name,
        chat_template,
    })
}
fn load_gemma4_checkpoint(path: &str, file: &ShardedGguf) -> anyhow::Result<Gemma4Loaded> {
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("gemma4")
        .to_string();
    let name = file
        .metadata_str("general.name")
        .unwrap_or(arch.as_str())
        .to_string();
    tracing::info!("loading GGUF as Gemma4 engine (arch={arch}, name={name})");
    let served = load_gemma4_engine_from_path(Path::new(path)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Gemma4(engine) = served else {
        anyhow::bail!("expected ServedEngine::Gemma4 for architecture {arch}");
    };

    let tokenizer = tokenizer_from_gguf(file)?;
    let eos_id = file
        .metadata_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize);
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);
    let byte_tokenizer = matches!(tokenizer, ServerTokenizer::Byte);
    let chat_template = crate::chat_template::ChatTemplate::detect_for_gguf(
        file.metadata_str("tokenizer.chat_template"),
        Some(arch.as_str()),
        byte_tokenizer,
    );
    tracing::info!("detected chat template: {chat_template:?}");

    Ok(Gemma4Loaded {
        engine,
        tokenizer,
        eos_id,
        bos_id,
        name,
        chat_template,
    })
}

fn load_real_gguf_checkpoint(path: &str, file: &ShardedGguf) -> anyhow::Result<GgufLoaded> {
    let config = ModelConfig::from_gguf(file)?;
    if let Some(arch) = file.metadata_str("general.architecture") {
        ferrox_models::ensure_generic_decoder(arch).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let all_confirmed =
        config.best_effort_fields.len() == 1 && config.best_effort_fields[0].starts_with("none --");
    if !all_confirmed {
        tracing::warn!(
            "some model config fields could not be read from this GGUF file's own metadata \
             and were inferred instead: {:?}",
            config.best_effort_fields
        );
    }

    let tokenizer = tokenizer_from_gguf(file)?;

    let eos_id = file
        .metadata_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize);
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);

    let byte_tokenizer = matches!(tokenizer, ServerTokenizer::Byte);
    let chat_template = crate::chat_template::ChatTemplate::detect_for_gguf(
        file.metadata_str("tokenizer.chat_template"),
        file.metadata_str("general.architecture"),
        byte_tokenizer,
    );
    tracing::info!("detected chat template: {chat_template:?}");

    // Opt-in expert streaming: with FERROX_EXPERT_CACHE_BYTES set (or
    // FERROX_SSD_STREAMING=1, which defaults the cache to 2 GiB),
    // routed experts are not loaded resident -- they stream through a
    // bounded, lease-protected cache of that many bytes (one global
    // budget across all layers), reading from the checkpoint file on
    // miss. Output is bit-identical to resident loading (same bytes,
    // same kernels); the trade is RAM footprint vs. read I/O.
    let expert_cache_bytes = resolve_expert_cache_bytes()?;
    let decoder = Decoder::from_gguf_with_expert_cache(path, config, expert_cache_bytes)?;

    Ok(GgufLoaded {
        decoder,
        tokenizer,
        eos_id,
        bos_id,
        is_synthetic: false,
        chat_template,
    })
}

/// Loads a real Kimi K3 checkpoint directory, mirroring
/// `ferrox-cli`'s `run-kimi` command exactly (same real file layout,
/// same real hparams/config assembly) so this is genuinely new
/// plumbing wired into the server, not a reimplementation with its own
/// untested assumptions.
fn load_real_kimi_checkpoint(dir: &str) -> anyhow::Result<KimiLoaded> {
    load_kimi_checkpoint_with_config(
        dir,
        kimi_k3(),
        ferrox_models::kimi_loader::KimiRealHparams::real(),
    )
}

/// The real loading logic, parametrized over `model_cfg`/`hp` so it can
/// be exercised end to end against a small synthetic checkpoint in
/// tests (Kimi K3's real shape -- 7168 hidden dim, 896 experts -- makes
/// a real-shaped on-disk fixture impractical for a unit test) --
/// mirrors how `kimi_loader::load_kimi_checkpoint` itself is already
/// generic over these two, rather than hardcoding the real preset the
/// way `ferrox-cli`'s `run-kimi` command does. `load_real_kimi_checkpoint`
/// (the real entry point `load()` calls) always passes the real
/// `kimi_k3()`/`KimiRealHparams::real()` -- this function's
/// parametrization is purely for testability, not a behavior change.
pub(crate) fn load_kimi_checkpoint_with_config(
    dir: &str,
    model_cfg: ModelConfig,
    hp: ferrox_models::kimi_loader::KimiRealHparams,
) -> anyhow::Result<KimiLoaded> {
    let dir_path = Path::new(dir);
    let index_path = dir_path.join("model.safetensors.index.json");
    tracing::info!(
        "loading real Kimi K3 checkpoint index: {}",
        index_path.display()
    );
    let shard = ferrox_safetensors::ShardedSafetensors::open_index(&index_path)?;

    tracing::info!(
        "loading all {} real layers (eagerly touches every routed expert's mmap range, \
         never materializes a dequantized f32 copy)...",
        model_cfg.n_layers
    );
    // Same opt-in as the GGUF path: FERROX_EXPERT_CACHE_BYTES (or
    // FERROX_SSD_STREAMING) streams routed experts through one bounded,
    // lease-protected store instead of holding every expert object
    // resident (bit-identical output).
    let expert_cache_bytes = resolve_expert_cache_bytes()?;
    let weights = ferrox_models::kimi_loader::load_kimi_checkpoint_with_expert_cache(
        &shard,
        &model_cfg,
        &hp,
        expert_cache_bytes,
    )?;
    tracing::info!("loaded. vocab size: {}", weights.output_head.rows());

    let vocab_path = dir_path.join("tiktoken.model");
    let vocab_text = std::fs::read_to_string(&vocab_path)?;
    let ranks = ferrox_models::kimi_tokenizer::parse_tiktoken_vocab(&vocab_text)?;
    let tokenizer_config_path = dir_path.join("tokenizer_config.json");
    let tokenizer_config_text = if tokenizer_config_path.exists() {
        Some(std::fs::read_to_string(&tokenizer_config_path)?)
    } else {
        None
    };
    let special_tokens = match &tokenizer_config_text {
        Some(text) => ferrox_models::kimi_tokenizer::parse_special_tokens(text)?,
        None => std::collections::HashMap::new(),
    };
    let eos_id = special_tokens.get("[EOS]").copied().map(|v| v as usize);
    let tokenizer =
        ferrox_models::kimi_tokenizer::KimiTokenizer::new(ranks, special_tokens.clone())?;
    tracing::info!(
        "loaded real tokenizer: {} base tokens, eos_id={eos_id:?}",
        tokenizer.vocab_size()
    );

    // Kimi K3's checkpoint carries its chat template as a top-level
    // string field in `tokenizer_config.json` (the real HuggingFace
    // convention), not GGUF metadata -- read it the same way, then
    // reuse the exact same `ChatTemplate::detect` sniffing logic.
    let chat_template_str = tokenizer_config_text.as_deref().and_then(|text| {
        serde_json::from_str::<serde_json::Value>(text)
            .ok()
            .and_then(|v| v.get("chat_template")?.as_str().map(str::to_string))
    });
    let chat_template = crate::chat_template::ChatTemplate::detect(chat_template_str.as_deref());
    tracing::info!("detected chat template: {chat_template:?}");

    let ferrox_models::config::AttentionKind::KimiHybrid(hybrid) = &model_cfg.attention else {
        anyhow::bail!("kimi_k3() preset must use AttentionKind::KimiHybrid");
    };
    let decoder_cfg = ferrox_models::kimi_decoder::KimiDecoderConfig {
        attn_res_block_size: 12,
        rms_norm_eps: model_cfg.rms_norm_eps,
        situ_beta: 4.0,
        situ_linear_beta: 25.0,
        moe: ferrox_models::latent_moe::KimiMoeConfig {
            n_experts_active: model_cfg.moe.n_experts_active,
            moe_renormalize: true,
            routed_scaling_factor: 1.0,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            rms_norm_eps: model_cfg.rms_norm_eps,
        },
    };
    let engine = KimiEngine {
        weights,
        cfg: decoder_cfg,
        mla_cfg: hybrid.mla.clone(),
        kda_cfg: hybrid.kda.clone(),
    };

    Ok(KimiLoaded {
        engine,
        tokenizer,
        eos_id,
        chat_template,
    })
}

fn build_synthetic_decoder(preset: &str) -> anyhow::Result<Decoder> {
    let base_cfg: ModelConfig = match preset {
        "glm-5.2" => glm_5_2(),
        "deepseek-v4-pro" => deepseek_v4_pro(),
        "kimi-k3" => kimi_k3(),
        other => anyhow::bail!("unknown preset '{other}'"),
    };
    // Small synthetic dims: this fallback demonstrates the
    // request/response/decode plumbing when no real checkpoint is
    // configured; it does not produce meaningful completions.
    let mut cfg = base_cfg;
    cfg.hidden_dim = 32;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 8;
    cfg.moe.hidden_dim = 32;
    cfg.moe.n_experts = cfg.moe.n_experts.min(16);
    cfg.moe.expert_ffn_dim = 16;

    Ok(Decoder::new_random_small(cfg, 2, 256))
}
