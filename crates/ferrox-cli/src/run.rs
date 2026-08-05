//! llama.cpp-style GGUF completion (`-m` / `-p` / `-n` / …).

use std::fmt;
use std::io::{self, Read, Write};
use std::path::Path;
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use clap::{Args, ValueEnum};
use ferrox_core::cache::KvCache;
use ferrox_gguf::ShardedGguf;
use ferrox_models::{
    ensure_generic_decoder, load_mla_engine_from_path, select_engine_kind, ByteTokenizer, Decoder,
    Engine, GgufBpeTokenizer, GgufSpmTokenizer, GgufUnigramTokenizer, ModelConfig, Sampler,
    SamplingParams, SelectedEngineKind, ServedEngine,
};

/// llama.cpp-compatible completion flags.
#[derive(Args, Debug, Clone)]
pub struct InferArgs {
    /// Model path (GGUF). Alias of llama.cpp `-m`.
    #[arg(
        short = 'm',
        long = "model",
        value_name = "FILE",
        required_unless_present = "list_devices"
    )]
    pub model: Option<String>,

    /// Prompt string. Alias of llama.cpp `-p`.
    #[arg(short = 'p', long = "prompt", default_value = "")]
    pub prompt: String,

    /// Prompt from file. Alias of llama.cpp `-f`.
    #[arg(short = 'f', long = "file", value_name = "FILE")]
    pub file: Option<String>,

    /// Number of tokens to predict (`-1` = fill remaining context).
    #[arg(short = 'n', long = "n-predict", default_value_t = 128)]
    pub n_predict: i64,

    /// Context size (0 = use GGUF `{arch}.context_length`, else 4096).
    #[arg(short = 'c', long = "ctx-size", default_value_t = 0)]
    pub ctx_size: usize,

    /// CPU threads (0 = leave rayon / env defaults). Sets `RAYON_NUM_THREADS`.
    #[arg(short = 't', long = "threads", default_value_t = 0)]
    pub threads: usize,

    /// Sampling temperature (`0` = greedy).
    #[arg(long = "temp", default_value_t = 0.8)]
    pub temperature: f32,

    /// Top-k sampling (`0` = disabled).
    #[arg(long = "top-k", default_value_t = 40)]
    pub top_k: usize,

    /// Top-p nucleus sampling.
    #[arg(long = "top-p", default_value_t = 0.95)]
    pub top_p: f32,

    /// Repetition penalty (`1.0` = off).
    #[arg(long = "repeat-penalty", default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// RNG seed (`-1` = time-based).
    #[arg(short = 's', long = "seed", default_value_t = -1)]
    pub seed: i64,

    /// Devices used for offloading (`none` disables GPU use).
    #[arg(
        long = "device",
        visible_alias = "dev",
        value_name = "DEVICE",
        ignore_case = true
    )]
    pub device: Option<OffloadDevice>,

    /// Print available offload devices and exit.
    #[arg(long = "list-devices", default_value_t = false)]
    pub list_devices: bool,

    /// GPU layers: `0`, a positive number, `auto`, or `all`.
    ///
    /// Partial placement is not implemented yet; any value above zero
    /// currently enables all supported operations on the selected backend.
    #[arg(
        long = "n-gpu-layers",
        visible_aliases = ["gpu-layers", "ngl"],
        default_value = "auto",
        value_name = "N"
    )]
    pub n_gpu_layers: GpuLayers,

    /// Optional system prompt (chat mode only).
    #[arg(long = "system")]
    pub system: Option<String>,

    /// Raw prompt: skip chat-template wrap (llama.cpp `--no-cnv`).
    #[arg(long = "no-cnv", default_value_t = false)]
    pub no_cnv: bool,

    /// Process `\\n` / `\\t` / `\\r` / `\\\\` escapes in `-p`.
    #[arg(short = 'e', long = "escape", default_value_t = false)]
    pub escape: bool,

    /// Ignore EOS and always emit up to `-n` tokens.
    #[arg(long = "ignore-eos", default_value_t = false)]
    pub ignore_eos: bool,

    /// Print the final prompt before generation.
    #[arg(long = "verbose-prompt", default_value_t = false)]
    pub verbose_prompt: bool,

    /// KV cache dtype (llama.cpp `-ctk` analogue). Sets `FERROX_CTK`.
    /// Values: `f16` (default), `q8_0`, `fp8`, `turbo8`, `turbo4`, `turbo3`.
    /// Non-f16 paths warn and fall back to f16 until Metal kernels land.
    #[arg(long = "ctk", value_name = "TYPE", default_value = "f16")]
    pub ctk: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OffloadDevice {
    Auto,
    None,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLayers {
    Auto,
    All,
    Count(u32),
}

impl GpuLayers {
    fn offload_enabled(self) -> bool {
        !matches!(self, Self::Count(0))
    }
}

impl FromStr for GpuLayers {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "all" => Ok(Self::All),
            _ => value
                .parse::<u32>()
                .map(Self::Count)
                .map_err(|_| "expected 0, a positive integer, 'auto', or 'all'".into()),
        }
    }
}

impl fmt::Display for GpuLayers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::All => f.write_str("all"),
            Self::Count(value) => value.fmt(f),
        }
    }
}

enum CliTokenizer {
    Bpe(Box<GgufBpeTokenizer>),
    Spm(GgufSpmTokenizer),
    Unigram(GgufUnigramTokenizer),
    Byte,
}

impl CliTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            CliTokenizer::Bpe(t) => t.encode(text).into_iter().map(|id| id as usize).collect(),
            CliTokenizer::Spm(t) => t.encode(text).into_iter().map(|id| id as usize).collect(),
            CliTokenizer::Unigram(t) => t.encode(text).into_iter().map(|id| id as usize).collect(),
            CliTokenizer::Byte => ByteTokenizer::encode(text)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
        }
    }

    fn decode(&self, ids: &[usize]) -> String {
        let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        match self {
            CliTokenizer::Bpe(t) => t.decode(&ids32),
            CliTokenizer::Spm(t) => t.decode(&ids32),
            CliTokenizer::Unigram(t) => t.decode(&ids32),
            CliTokenizer::Byte => ByteTokenizer::decode(&ids32),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            CliTokenizer::Bpe(_) => "gguf-bpe",
            CliTokenizer::Spm(_) => "gguf-spm",
            CliTokenizer::Unigram(_) => "gguf-unigram",
            CliTokenizer::Byte => "byte",
        }
    }
}

#[derive(Clone, Copy)]
enum ChatKind {
    ChatMl,
    GenericRoleMarkers,
    Llama3,
    Gemma,
    Plain,
}

impl ChatKind {
    fn detect(template: Option<&str>) -> Self {
        match template {
            Some(t) if t.contains("<|im_start|>") => ChatKind::ChatMl,
            Some(t) if t.contains("<|start_header_id|>") => ChatKind::Llama3,
            Some(t) if t.contains("<|user|>") || t.contains("<|assistant|>") => {
                ChatKind::GenericRoleMarkers
            }
            Some(t) if t.contains("<start_of_turn>") => ChatKind::Gemma,
            _ => ChatKind::Plain,
        }
    }

    /// Match `ferrox-server::chat_template::ChatTemplate::detect_for_gguf`.
    fn detect_for_gguf(template: Option<&str>, arch: Option<&str>, byte_tokenizer: bool) -> Self {
        match template.filter(|t| !t.is_empty()) {
            Some(t) => Self::detect(Some(t)),
            None if byte_tokenizer || arch.is_none() => Self::Plain,
            None => Self::ChatMl,
        }
    }

    fn wrap_user(&self, system: Option<&str>, user: &str) -> String {
        match self {
            ChatKind::ChatMl => {
                let mut out = String::new();
                if let Some(sys) = system {
                    out.push_str("<|im_start|>system\n");
                    out.push_str(sys);
                    out.push_str("<|im_end|>\n");
                }
                out.push_str("<|im_start|>user\n");
                out.push_str(user);
                out.push_str("<|im_end|>\n<|im_start|>assistant\n");
                out
            }
            ChatKind::GenericRoleMarkers => {
                let mut out = String::new();
                if let Some(sys) = system {
                    out.push_str("<|system|>\n");
                    out.push_str(sys);
                    out.push_str("</s>\n");
                }
                out.push_str("<|user|>\n");
                out.push_str(user);
                out.push_str("</s>\n<|assistant|>\n");
                out
            }
            ChatKind::Llama3 => {
                let mut out = String::new();
                if let Some(sys) = system {
                    out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                    out.push_str(sys);
                    out.push_str("<|eot_id|>");
                }
                out.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                out.push_str(user);
                out.push_str("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
                out
            }
            ChatKind::Gemma => {
                let mut out = String::new();
                if let Some(sys) = system {
                    // Gemma often folds system into the first user turn.
                    out.push_str("<start_of_turn>user\n");
                    out.push_str(sys);
                    out.push('\n');
                    out.push_str(user);
                    out.push_str("<end_of_turn>\n<start_of_turn>model\n");
                } else {
                    out.push_str("<start_of_turn>user\n");
                    out.push_str(user);
                    out.push_str("<end_of_turn>\n<start_of_turn>model\n");
                }
                out
            }
            // Match `ferrox-server::chat_template::ChatTemplate::Plain`
            // (`role: content` lines) so CLI and `/v1/chat/completions`
            // share the same prompt framing when GGUF has no template.
            ChatKind::Plain => {
                if let Some(sys) = system {
                    format!("system: {sys}\nuser: {user}")
                } else {
                    format!("user: {user}")
                }
            }
        }
    }
}

fn apply_escapes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn resolve_prompt(args: &InferArgs) -> anyhow::Result<String> {
    let mut prompt = if let Some(path) = &args.file {
        let mut buf = String::new();
        let mut f = std::fs::File::open(path)?;
        f.read_to_string(&mut buf)?;
        buf
    } else {
        args.prompt.clone()
    };
    if args.escape {
        prompt = apply_escapes(&prompt);
    }
    Ok(prompt)
}

fn print_available_devices() {
    println!("Available devices:");
    println!("  CPU");

    let metal = ferrox_metal::MetalProfile::detect();
    if let Some(name) = metal.device_name {
        println!("  Metal: {name}");
    }

    let cuda = ferrox_cuda::HardwareProfile::detect();
    if cuda.cuda_available {
        let name = cuda.cuda_device_name.as_deref().unwrap_or("unknown device");
        println!("  CUDA: {name}");
        if cuda.cuda_device_count > 1 {
            println!("        ({} devices detected)", cuda.cuda_device_count);
        }
    }
}

fn apply_backend_env(args: &InferArgs) -> anyhow::Result<()> {
    if args.threads > 0 {
        // SAFETY: single-threaded init before rayon workers spawn.
        unsafe {
            std::env::set_var("RAYON_NUM_THREADS", args.threads.to_string());
            std::env::set_var("FERROX_CPU_THREADS", args.threads.to_string());
        }
    }

    let device = if args.n_gpu_layers.offload_enabled() {
        args.device.unwrap_or(OffloadDevice::Auto)
    } else {
        OffloadDevice::None
    };

    match device {
        OffloadDevice::None | OffloadDevice::Cpu => unsafe {
            std::env::set_var("FERROX_METAL", "0");
            std::env::set_var("FERROX_METAL_ATTN", "0");
            std::env::set_var("FERROX_CUDA", "0");
        },
        OffloadDevice::Auto => unsafe {
            std::env::set_var("FERROX_METAL", "auto");
            std::env::set_var("FERROX_CUDA", "auto");
            std::env::set_var("FERROX_METAL_ATTN", "1");
        },
        OffloadDevice::Metal => {
            #[cfg(not(feature = "metal"))]
            {
                anyhow::bail!("Metal requested but this binary was built without --features metal");
            }
            #[cfg(feature = "metal")]
            {
                if !ferrox_metal::MetalProfile::detect().available {
                    anyhow::bail!("Metal requested but no Metal device is available");
                }
                unsafe {
                    std::env::set_var("FERROX_METAL", "1");
                    std::env::set_var("FERROX_METAL_ATTN", "1");
                    std::env::set_var("FERROX_CUDA", "0");
                }
            }
        }
        OffloadDevice::Cuda => {
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA requested but this binary was built without --features cuda");
            }
            #[cfg(feature = "cuda")]
            {
                if !ferrox_cuda::HardwareProfile::detect().cuda_available {
                    anyhow::bail!("CUDA requested but no CUDA device is available");
                }
                unsafe {
                    std::env::set_var("FERROX_CUDA", "1");
                    std::env::set_var("FERROX_METAL", "0");
                    std::env::set_var("FERROX_METAL_ATTN", "0");
                }
            }
        }
    }

    // SAFETY: single-threaded init before Metal/CUDA workers spawn.
    unsafe {
        std::env::set_var("FERROX_CTK", args.ctk.trim());
    }

    eprintln!(
        "ferrox: device={} gpu-layers={} ctk={}",
        match device {
            OffloadDevice::Auto => "auto",
            OffloadDevice::None | OffloadDevice::Cpu => "none",
            OffloadDevice::Metal => "Metal",
            OffloadDevice::Cuda => "CUDA",
        },
        args.n_gpu_layers,
        args.ctk.trim()
    );
    Ok(())
}

fn seed_from_args(seed: i64) -> u64 {
    if seed < 0 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1)
    } else {
        seed as u64
    }
}

/// Run llama.cpp-style GGUF completion.
pub fn run_infer(args: InferArgs) -> anyhow::Result<()> {
    if args.list_devices {
        print_available_devices();
        return Ok(());
    }
    apply_backend_env(&args)?;

    let model = args
        .model
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--model is required"))?;
    let path = Path::new(&model);
    if !path.exists() {
        anyhow::bail!("model not found: {model}");
    }

    let file = ShardedGguf::open(path)?;
    let arch_early = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    if matches!(
        select_engine_kind(&arch_early),
        Ok(SelectedEngineKind::Mla)
    ) {
        return run_mla_infer(args, path, &file);
    }

    let config = ModelConfig::from_gguf(&file)?;
    if let Some(arch) = file.metadata_str("general.architecture") {
        ensure_generic_decoder(arch).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    if !(config.best_effort_fields.is_empty()
        || (config.best_effort_fields.len() == 1
            && config.best_effort_fields[0].starts_with("none --")))
    {
        eprintln!(
            "ferrox: inferred config fields: {:?}",
            config.best_effort_fields
        );
    }

    let tokenizer = match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2") => CliTokenizer::Bpe(Box::new(GgufBpeTokenizer::from_gguf(&file)?)),
        Some("llama") => CliTokenizer::Spm(GgufSpmTokenizer::from_gguf(&file)?),
        Some("t5") => CliTokenizer::Unigram(GgufUnigramTokenizer::from_gguf(&file)?),
        other => {
            eprintln!(
                "ferrox: unrecognized tokenizer.ggml.model ({other:?}); using byte tokenizer"
            );
            CliTokenizer::Byte
        }
    };
    let eos_id = file
        .metadata_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize);
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);

    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown");
    let gguf_ctx = file
        .metadata_u64(&format!("{arch}.context_length"))
        .map(|v| v as usize)
        .unwrap_or(4096);
    let ctx_size = if args.ctx_size > 0 {
        args.ctx_size
    } else {
        gguf_ctx
    };

    let chat = ChatKind::detect_for_gguf(
        file.metadata_str("tokenizer.chat_template"),
        file.metadata_str("general.architecture"),
        matches!(tokenizer, CliTokenizer::Byte),
    );
    let user_prompt = resolve_prompt(&args)?;
    let prompt = if args.no_cnv {
        user_prompt
    } else {
        chat.wrap_user(args.system.as_deref(), &user_prompt)
    };

    if args.verbose_prompt {
        eprintln!("----- prompt -----");
        eprintln!("{prompt}");
        eprintln!("------------------");
    }

    eprintln!(
        "ferrox: loading {} (tokenizer={}, ctx={ctx_size})",
        model,
        tokenizer.kind()
    );
    let load_t = Instant::now();
    let decoder = Decoder::from_gguf(path, config)?;
    eprintln!("ferrox: loaded in {:.2}s", load_t.elapsed().as_secs_f64());

    let mut tokens = tokenizer.encode(&prompt);
    if let Some(bos) = bos_id {
        if tokens.first() != Some(&bos) {
            tokens.insert(0, bos);
        }
    }
    let vocab_size = decoder.config.vocab_size;
    if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab_size) {
        anyhow::bail!("prompt token {bad} outside vocab_size {vocab_size}");
    }
    if tokens.len() >= ctx_size {
        anyhow::bail!(
            "prompt length {} >= context size {ctx_size}; raise -c or shorten prompt",
            tokens.len()
        );
    }

    let room = ctx_size - tokens.len();
    let max_new = if args.n_predict < 0 {
        room
    } else {
        (args.n_predict as usize).min(room)
    };

    let sampling = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repeat_penalty,
    };
    let seed = seed_from_args(args.seed);
    let mut sampler = Sampler::new(seed);

    #[cfg(feature = "metal")]
    let _metal_greedy_guard = {
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                ferrox_models::set_metal_greedy_argmax(false);
            }
        }
        if sampling.temperature <= 0.0 && ferrox_models::metal_greedy_gpu_enabled() {
            ferrox_models::set_metal_greedy_argmax(true);
            Some(Guard)
        } else {
            None
        }
    };

    let mut caches: Vec<KvCache> = decoder
        .layers
        .iter()
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect();

    let prefill_t = Instant::now();
    let mut pos;
    let mut logits = if tokens.is_empty() {
        let l = decoder.forward_token(0, 0, &mut caches);
        pos = 1;
        l
    } else {
        let rows = decoder.forward_batch(&tokens, 0, &mut caches);
        pos = tokens.len();
        rows.into_iter()
            .last()
            .expect("forward_batch returns one logits row per prompt token")
    };
    let prefill_secs = prefill_t.elapsed().as_secs_f64();

    let mut generated: Vec<usize> = Vec::with_capacity(max_new);
    let mut stdout = io::stdout().lock();
    let decode_t = Instant::now();
    for _ in 0..max_new {
        let next = sampler.sample(&logits, &sampling, &generated);
        if !args.ignore_eos && Some(next) == eos_id {
            break;
        }
        generated.push(next);
        let piece = tokenizer.decode(&[next]);
        stdout.write_all(piece.as_bytes())?;
        stdout.flush()?;
        logits = decoder.forward_token(next, pos, &mut caches);
        pos += 1;
    }
    let decode_secs = decode_t.elapsed().as_secs_f64();
    writeln!(stdout)?;

    let prompt_n = tokens.len();
    let gen_n = generated.len();
    let prompt_tps = if prefill_secs > 0.0 {
        prompt_n as f64 / prefill_secs
    } else {
        0.0
    };
    let pred_tps = if decode_secs > 0.0 {
        gen_n as f64 / decode_secs
    } else {
        0.0
    };
    eprintln!(
        "ferrox: prompt {prompt_n} tokens, {prompt_tps:.2} t/s; \
         predict {gen_n} tokens, {pred_tps:.2} t/s"
    );

    Ok(())
}

/// Dense-lead DeepSeek-2 / Mistral-4 path via [`MlaEngine`].
fn run_mla_infer(
    args: InferArgs,
    path: &Path,
    file: &ShardedGguf,
) -> anyhow::Result<()> {
    let tokenizer = match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2") => CliTokenizer::Bpe(Box::new(GgufBpeTokenizer::from_gguf(file)?)),
        Some("llama") => CliTokenizer::Spm(GgufSpmTokenizer::from_gguf(file)?),
        Some("t5") => CliTokenizer::Unigram(GgufUnigramTokenizer::from_gguf(file)?),
        other => {
            eprintln!(
                "ferrox: unrecognized tokenizer.ggml.model ({other:?}); using byte tokenizer"
            );
            CliTokenizer::Byte
        }
    };
    let eos_id = file
        .metadata_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize);
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown");
    let gguf_ctx = file
        .metadata_u64(&format!("{arch}.context_length"))
        .map(|v| v as usize)
        .unwrap_or(4096);
    let ctx_size = if args.ctx_size > 0 {
        args.ctx_size
    } else {
        gguf_ctx
    };

    let chat = ChatKind::detect_for_gguf(
        file.metadata_str("tokenizer.chat_template"),
        file.metadata_str("general.architecture"),
        matches!(tokenizer, CliTokenizer::Byte),
    );
    let user_prompt = resolve_prompt(&args)?;
    let prompt = if args.no_cnv {
        user_prompt
    } else {
        chat.wrap_user(args.system.as_deref(), &user_prompt)
    };

    eprintln!(
        "ferrox: loading {} as MLA engine (tokenizer={}, ctx={ctx_size})",
        args.model.as_deref().unwrap_or("?"),
        tokenizer.kind()
    );
    let load_t = Instant::now();
    let served = load_mla_engine_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Mla(engine) = served else {
        anyhow::bail!("expected ServedEngine::Mla");
    };
    eprintln!("ferrox: loaded in {:.2}s", load_t.elapsed().as_secs_f64());

    let mut tokens = tokenizer.encode(&prompt);
    if let Some(bos) = bos_id {
        if tokens.first() != Some(&bos) {
            tokens.insert(0, bos);
        }
    }
    let vocab_size = Engine::vocab_size(&engine);
    if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab_size) {
        anyhow::bail!("prompt token {bad} outside vocab_size {vocab_size}");
    }
    if tokens.len() >= ctx_size {
        anyhow::bail!(
            "prompt length {} >= context size {ctx_size}; raise -c or shorten prompt",
            tokens.len()
        );
    }

    let room = ctx_size - tokens.len();
    let max_new = if args.n_predict < 0 {
        room
    } else {
        (args.n_predict as usize).min(room)
    };

    let sampling = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repeat_penalty,
    };
    let mut sampler = Sampler::new(seed_from_args(args.seed));
    let mut state = Engine::new_state(&engine);

    let prefill_t = Instant::now();
    let mut pos = 0usize;
    let mut logits = if tokens.is_empty() {
        let l = engine.forward_token(0, 0, &mut state);
        pos = 1;
        l
    } else {
        let mut last = Vec::new();
        for &tok in &tokens {
            last = engine.forward_token(tok, pos, &mut state);
            pos += 1;
        }
        last
    };
    let prefill_secs = prefill_t.elapsed().as_secs_f64();

    let mut generated: Vec<usize> = Vec::with_capacity(max_new);
    let mut stdout = io::stdout().lock();
    let decode_t = Instant::now();
    for _ in 0..max_new {
        let next = sampler.sample(&logits, &sampling, &generated);
        if !args.ignore_eos && Some(next) == eos_id {
            break;
        }
        generated.push(next);
        let piece = tokenizer.decode(&[next]);
        stdout.write_all(piece.as_bytes())?;
        stdout.flush()?;
        logits = engine.forward_token(next, pos, &mut state);
        pos += 1;
    }
    let decode_secs = decode_t.elapsed().as_secs_f64();
    writeln!(stdout)?;

    let prompt_n = tokens.len();
    let gen_n = generated.len();
    let prompt_tps = if prefill_secs > 0.0 {
        prompt_n as f64 / prefill_secs
    } else {
        0.0
    };
    let pred_tps = if decode_secs > 0.0 {
        gen_n as f64 / decode_secs
    } else {
        0.0
    };
    eprintln!(
        "ferrox: prompt {prompt_n} tokens, {prompt_tps:.2} t/s; \
         predict {gen_n} tokens, {pred_tps:.2} t/s"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::GpuLayers;
    use std::str::FromStr;

    #[test]
    fn parses_llama_gpu_layer_values() {
        assert_eq!(GpuLayers::from_str("0"), Ok(GpuLayers::Count(0)));
        assert_eq!(GpuLayers::from_str("42"), Ok(GpuLayers::Count(42)));
        assert_eq!(GpuLayers::from_str("auto"), Ok(GpuLayers::Auto));
        assert_eq!(GpuLayers::from_str("all"), Ok(GpuLayers::All));
        assert!(GpuLayers::from_str("-1").is_err());
        assert!(GpuLayers::from_str("some").is_err());
    }
}
