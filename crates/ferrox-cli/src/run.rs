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
    ensure_generic_decoder, load_gemma4_engine_from_path, load_glm52_engine_from_path,
    load_mla_engine_from_path, select_engine_kind, ByteTokenizer, Decoder, Engine,
    GgufBpeTokenizer, GgufSpmTokenizer, GgufUnigramTokenizer, ModelConfig, Sampler, SamplingParams,
    SelectedEngineKind, ServedEngine,
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

    /// Context size: `auto` = largest that fits the device memory
    /// budget, `0` = the GGUF's own `{arch}.context_length` (else
    /// 4096), or an explicit token count.
    #[arg(short = 'c', long = "ctx-size", default_value_t = ContextSize::FromModel)]
    pub ctx_size: ContextSize,

    /// Refuse to load (exit 1) when the pre-load budget says the
    /// requested context will not fit, instead of warning and trying
    /// anyway. Off by default because ferrox mmaps its weights: an
    /// over-budget model really can run, page-faulting, so the check
    /// is advisory unless you say otherwise.
    #[arg(long = "strict-budget", default_value_t = false)]
    pub strict_budget: bool,

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

    /// Multi-token prediction (MTP) draft heads — not loaded from GGUF yet.
    #[arg(long = "mtp", default_value_t = false)]
    pub mtp: bool,

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

/// What `-c` / `--ctx-size` was asked for, before any model is opened.
/// Same shape as [`GpuLayers`]: a symbolic value alongside the literal
/// one, resolved once the header and the device budget are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextSize {
    /// Largest context that fits the device memory budget.
    Auto,
    /// llama.cpp's `-c 0`: whatever the GGUF says it was trained for.
    FromModel,
    Tokens(usize),
}

impl FromStr for ContextSize {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim() {
            "auto" => Ok(Self::Auto),
            "0" => Ok(Self::FromModel),
            other => other
                .parse::<usize>()
                .map(Self::Tokens)
                .map_err(|_| "expected 'auto', 0, or a positive token count".into()),
        }
    }
}

impl fmt::Display for ContextSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::FromModel => f.write_str("0"),
            Self::Tokens(n) => n.fmt(f),
        }
    }
}

/// Which memory pool the resolved backend draws from, so the budget is
/// probed against the device that will actually hold the KV cache.
/// `Auto` is reported as CPU: without a `--device`/`--ngl` choice the
/// generic decoder keeps its host `KvCache`, and claiming a GPU budget
/// we may never use would be the wrong kind of optimism.
fn budget_backend_for(args: &InferArgs) -> ferrox_models::BudgetBackend {
    use ferrox_models::BudgetBackend;
    let offload = args.n_gpu_layers.offload_enabled();
    match args.device {
        Some(OffloadDevice::Metal) => BudgetBackend::Metal,
        Some(OffloadDevice::Cuda) => BudgetBackend::Cuda,
        None | Some(OffloadDevice::Auto) if offload && cfg!(feature = "metal") => {
            BudgetBackend::Metal
        }
        None | Some(OffloadDevice::Auto) if offload && cfg!(feature = "cuda") => {
            BudgetBackend::Cuda
        }
        _ => BudgetBackend::Cpu,
    }
}

/// Width of the KV store the selected backend will really keep. The
/// host `ferrox_core::cache::KvCache` is `Vec<f32>`; only the Metal
/// path has a device KV whose dtype `--ctk` selects.
fn kv_elem_for(args: &InferArgs) -> ferrox_models::KvElem {
    use ferrox_models::{BudgetBackend, KvElem};
    match budget_backend_for(args) {
        BudgetBackend::Metal => KvElem::from_ctk(&args.ctk),
        // CUDA has no device KV store of its own yet, and CPU is the
        // f32 host cache.
        BudgetBackend::Cuda | BudgetBackend::Cpu => KvElem::F32,
    }
}

/// Resolves `-c/--ctx-size` against the pre-load budget, printing the
/// arithmetic behind the answer.
///
/// This is the whole point of Phase 2: the terms are exact in the GGUF
/// header, so the check happens *before* the weights load rather than
/// being discovered as an allocation failure later. `auto` picks the
/// largest fitting context; an explicit context that does not fit is
/// reported as a typed rejection naming the estimate, the limit and
/// which ceiling binds -- fatal under `--strict-budget`, a warning
/// otherwise (see that flag's doc comment for why the default is
/// advisory).
fn resolve_ctx_size(args: &InferArgs, path: &Path, gguf_ctx: usize) -> anyhow::Result<usize> {
    use ferrox_models::residency_report::{ResidencyAssumptions, ResidencyReport};
    use ferrox_models::DeviceBudget;

    let backend = budget_backend_for(args);
    let budget = DeviceBudget::detect(backend);
    let assumptions = ResidencyAssumptions {
        context_tokens: gguf_ctx,
        concurrent_requests: 1,
        expert_cache_bytes: expert_cache_bytes_from_env(),
        kv_elem: kv_elem_for(args),
        prefill_chunk: prefill_chunk_from_env(),
        ..ResidencyAssumptions::default()
    };

    // No probe, no ceiling: fall back to the requested context rather
    // than refusing on the strength of a number we do not have.
    if budget.is_unknown() {
        let requested = match args.ctx_size {
            ContextSize::Tokens(n) => n,
            ContextSize::Auto | ContextSize::FromModel => gguf_ctx,
        };
        eprintln!("ferrox: {budget}; using ctx={requested} unchecked");
        return Ok(requested);
    }

    let report = match ResidencyReport::from_gguf(path, assumptions, budget.usable_bytes) {
        Ok(r) => r,
        // A header this planner cannot read is not a reason to refuse
        // a run the loader may well handle (MLA/Gemma4/GLM stacks have
        // their own hparams and do not go through `ModelConfig`).
        Err(e) => {
            let requested = match args.ctx_size {
                ContextSize::Tokens(n) => n,
                ContextSize::Auto | ContextSize::FromModel => gguf_ctx,
            };
            eprintln!("ferrox: KV budget not computed for this checkpoint ({e}); ctx={requested}");
            return Ok(requested);
        }
    };
    let priced = report.kv_budget();

    let tokens = match args.ctx_size {
        ContextSize::Auto => {
            let fit = report.auto_context(gguf_ctx);
            eprintln!("ferrox: {budget}");
            eprintln!("ferrox: {fit}");
            eprintln!("ferrox: {}", budget.caveat());
            if fit.tokens == 0 {
                anyhow::bail!(
                    "{}: no context fits -- {} of weights leave nothing for KV inside the \
                     {} budget. Quantize further, stream experts \
                     (FERROX_EXPERT_CACHE_BYTES), or raise FERROX_DEVICE_BUDGET_BYTES.",
                    ferrox_models::Ceiling::DeviceMemory.code(),
                    report.weights_bytes,
                    budget.usable_bytes,
                );
            }
            fit.tokens
        }
        ContextSize::FromModel => gguf_ctx,
        ContextSize::Tokens(n) => n,
    };

    if let Err(e) = priced.check(tokens) {
        let fit = report.auto_context(gguf_ctx);
        let message = format!(
            "{}: {} bytes estimated at ctx={tokens} against a {} byte {} budget ({}); \
             {} bytes over. `--ctx-size auto` would pick {}. {}",
            e.code(),
            e.estimated_bytes,
            e.limit_bytes,
            backend,
            budget.source,
            e.overage_bytes(),
            fit.tokens,
            budget.caveat(),
        );
        if args.strict_budget {
            anyhow::bail!("{message}");
        }
        eprintln!("ferrox: WARNING {message}");
        eprintln!("ferrox: continuing anyway (pass --strict-budget to refuse instead)");
    }
    Ok(tokens)
}

/// `FERROX_EXPERT_CACHE_BYTES`, so the plan charges streamed routed
/// experts at their cache budget rather than fully resident.
fn expert_cache_bytes_from_env() -> Option<u64> {
    std::env::var("FERROX_EXPERT_CACHE_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
}

/// `FERROX_CHUNKED_PREFILL`; `1` (token-at-a-time) when unset. Only
/// affects sliding-window layers, whose resident positions are
/// `window + chunk - 1`.
fn prefill_chunk_from_env() -> usize {
    std::env::var("FERROX_CHUNKED_PREFILL")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1)
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

/// The checkpoint's own chat template, evaluated.
///
/// This used to be a near-identical copy of `ferrox-server`'s marker
/// sniffer -- six hand-written renderers picked by which literal marker
/// a template string happened to contain. Both are gone; both now
/// compile the real Jinja source with
/// [`ferrox_models::chat_template`], so `ferrox -m mistral.gguf -p hi`
/// and `POST /v1/chat/completions` frame the same conversation the same
/// way, including for the families the sniffer never recognised.
struct ChatKind {
    template: ferrox_models::chat_template::ChatTemplate,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatKind {
    fn detect_for_gguf(file: &ShardedGguf, byte_tokenizer: bool) -> Self {
        ChatKind {
            template: ferrox_models::chat_template::ChatTemplate::from_gguf_metadata(
                file.metadata_str("tokenizer.chat_template"),
                file.metadata_str("general.architecture"),
                byte_tokenizer,
            ),
            bos_token: file.token_text("tokenizer.ggml.bos_token_id"),
            eos_token: file.token_text("tokenizer.ggml.eos_token_id"),
        }
    }

    /// One conversation turn, framed the way this checkpoint expects.
    ///
    /// A template that will not render is an error, never a fallback to
    /// a guessed framing: a silently mis-framed prompt is exactly the
    /// bug that made this stop sniffing, and it shows up as degenerate
    /// output rather than as a message.
    fn wrap_user(&self, system: Option<&str>, user: &str) -> anyhow::Result<String> {
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(serde_json::json!({"role": "system", "content": sys}));
        }
        messages.push(serde_json::json!({"role": "user", "content": user}));
        let opts = ferrox_models::chat_template::RenderOptions {
            add_generation_prompt: true,
            bos_token: self.bos_token.clone(),
            eos_token: self.eos_token.clone(),
            ..Default::default()
        };
        self.template
            .render(&messages, &opts)
            .map_err(|e| anyhow::anyhow!("chat template failed to render: {e}"))
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
    // SAFETY: still single-threaded here; rayon/Metal workers below.
    unsafe { ferrox_core::weight_matrix::default_cpu_int_dot_on() };
    // Same pool policy as `ferrox-server`: explicit width (performance
    // cores by default, as llama.cpp does) and explicit QoS.
    ferrox_core::threads::init_cpu_pool();

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
            // Honor a pre-set FERROX_METAL_ATTN so ablations like
            // `FERROX_METAL_ATTN=0 … --ngl 99` actually disable attn.
            if std::env::var_os("FERROX_METAL_ATTN").is_none() {
                std::env::set_var("FERROX_METAL_ATTN", "1");
            }
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
                    if std::env::var_os("FERROX_METAL_ATTN").is_none() {
                        std::env::set_var("FERROX_METAL_ATTN", "1");
                    }
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
    if args.mtp {
        anyhow::bail!(
            "--mtp: MTP draft heads not yet loaded from GGUF (num_nextn_predict_layers); \
             prompt-lookup speculative decoding remains available via `ferrox speculative`"
        );
    }
    apply_backend_env(&args)?;

    let model = args
        .model
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--model is required"))?;
    let model = crate::pull::resolve_model_path(&model)?;
    let path = Path::new(&model);
    if !path.exists() {
        anyhow::bail!("model not found: {model}");
    }

    let file = ShardedGguf::open(path)?;
    let arch_early = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    ferrox_models::mmproj::eprint_mmproj_if_present(path, Some(arch_early.as_str()));
    if matches!(select_engine_kind(&arch_early), Ok(SelectedEngineKind::Mla)) {
        return run_mla_infer(args, path, &file);
    }
    if matches!(
        select_engine_kind(&arch_early),
        Ok(SelectedEngineKind::Gemma4)
    ) {
        return run_gemma4_infer(args, path, &file);
    }
    if matches!(arch_early.as_str(), "glm-dsa" | "glm4" | "glm4moe") {
        return run_glm52_infer(args, path, &file);
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
        Some("gpt2" | "gemma4") => CliTokenizer::Bpe(Box::new(GgufBpeTokenizer::from_gguf(&file)?)),
        Some("llama") => CliTokenizer::Spm(GgufSpmTokenizer::from_gguf(&file)?),
        Some("t5") => CliTokenizer::Unigram(GgufUnigramTokenizer::from_gguf(&file)?),
        other => {
            eprintln!(
                "ferrox: unrecognized tokenizer.ggml.model ({other:?}); using byte tokenizer"
            );
            CliTokenizer::Byte
        }
    };
    // Not just `eos_token_id`: Llama-3 ends a turn with `<|eot_id|>` and
    // gemma-4 with `<turn|>`, neither of which is the metadata EOS.
    let stop_tokens = ferrox_models::tokenizer::StopTokens::from_gguf(&file);
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
    let ctx_size = resolve_ctx_size(&args, path, gguf_ctx)?;

    let chat = ChatKind::detect_for_gguf(&file, matches!(tokenizer, CliTokenizer::Byte));
    let user_prompt = resolve_prompt(&args)?;
    let prompt = if args.no_cnv {
        user_prompt
    } else {
        chat.wrap_user(args.system.as_deref(), &user_prompt)?
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
    // LongRoPE picks its factor set from the run's context size, not the
    // checkpoint's advertised maximum (llama.cpp does the same, per
    // request, from `cparams.n_ctx_seq`).
    let mut config = config;
    config.apply_runtime_context(ctx_size);
    let decoder = Decoder::from_gguf(path, config)?;
    eprintln!("ferrox: loaded in {:.2}s", load_t.elapsed().as_secs_f64());

    let mut tokens = tokenizer.encode(&prompt);
    // Match llama.cpp vocab add_bos (qwen2/BPE default false). Blindly
    // prepending bos_token_id poisons Qwen2-MoE (`<|endoftext|>`).
    ferrox_models::tokenizer::prepend_bos(
        &mut tokens,
        bos_id.filter(|_| ferrox_models::tokenizer::should_add_bos_token(&file)),
    );
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
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
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
        if sampling.temperature <= 0.0 {
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
        let l = decoder.forward_batch_last(&tokens, 0, &mut caches);
        pos = tokens.len();
        l
    };
    let prefill_secs = prefill_t.elapsed().as_secs_f64();

    let mut generated: Vec<usize> = Vec::with_capacity(max_new);
    let mut stdout = io::stdout().lock();
    let decode_t = Instant::now();
    for _ in 0..max_new {
        let next = sampler.sample(&logits, &sampling, &generated);
        if !args.ignore_eos && stop_tokens.contains(next) {
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
fn run_mla_infer(args: InferArgs, path: &Path, file: &ShardedGguf) -> anyhow::Result<()> {
    let tokenizer = match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => CliTokenizer::Bpe(Box::new(GgufBpeTokenizer::from_gguf(file)?)),
        Some("llama") => CliTokenizer::Spm(GgufSpmTokenizer::from_gguf(file)?),
        Some("t5") => CliTokenizer::Unigram(GgufUnigramTokenizer::from_gguf(file)?),
        other => {
            eprintln!(
                "ferrox: unrecognized tokenizer.ggml.model ({other:?}); using byte tokenizer"
            );
            CliTokenizer::Byte
        }
    };
    // Not just `eos_token_id`: Llama-3 ends a turn with `<|eot_id|>` and
    // gemma-4 with `<turn|>`, neither of which is the metadata EOS.
    let stop_tokens = ferrox_models::tokenizer::StopTokens::from_gguf(file);
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
    let ctx_size = resolve_ctx_size(&args, path, gguf_ctx)?;

    let chat = ChatKind::detect_for_gguf(file, matches!(tokenizer, CliTokenizer::Byte));
    let user_prompt = resolve_prompt(&args)?;
    let prompt = if args.no_cnv {
        user_prompt
    } else {
        chat.wrap_user(args.system.as_deref(), &user_prompt)?
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
    ferrox_models::tokenizer::prepend_bos(
        &mut tokens,
        bos_id.filter(|_| ferrox_models::tokenizer::should_add_bos_token(file)),
    );
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
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
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
        if !args.ignore_eos && stop_tokens.contains(next) {
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

/// GLM-5.2 / GLM4-family path via [`Glm52Engine`]./// Gemma-4 dedicated path via [`ferrox_models::Gemma4Engine`].
fn run_gemma4_infer(args: InferArgs, path: &Path, file: &ShardedGguf) -> anyhow::Result<()> {
    let tokenizer = match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => CliTokenizer::Bpe(Box::new(GgufBpeTokenizer::from_gguf(file)?)),
        Some("llama") => CliTokenizer::Spm(GgufSpmTokenizer::from_gguf(file)?),
        Some("t5") => CliTokenizer::Unigram(GgufUnigramTokenizer::from_gguf(file)?),
        other => {
            eprintln!(
                "ferrox: unrecognized tokenizer.ggml.model ({other:?}); using byte tokenizer"
            );
            CliTokenizer::Byte
        }
    };
    // Not just `eos_token_id`: Llama-3 ends a turn with `<|eot_id|>` and
    // gemma-4 with `<turn|>`, neither of which is the metadata EOS.
    let stop_tokens = ferrox_models::tokenizer::StopTokens::from_gguf(file);
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
    let ctx_size = resolve_ctx_size(&args, path, gguf_ctx)?;

    let chat = ChatKind::detect_for_gguf(file, matches!(tokenizer, CliTokenizer::Byte));
    let user_prompt = resolve_prompt(&args)?;
    let prompt = if args.no_cnv {
        user_prompt
    } else {
        chat.wrap_user(args.system.as_deref(), &user_prompt)?
    };

    eprintln!(
        "ferrox: loading {} as Gemma4 engine (tokenizer={}, ctx={ctx_size})",
        args.model.as_deref().unwrap_or("?"),
        tokenizer.kind()
    );
    let load_t = Instant::now();
    let served = load_gemma4_engine_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Gemma4(engine) = served else {
        anyhow::bail!("expected ServedEngine::Gemma4");
    };
    let engine = *engine;
    eprintln!("ferrox: loaded in {:.2}s", load_t.elapsed().as_secs_f64());

    let mut tokens = tokenizer.encode(&prompt);
    ferrox_models::tokenizer::prepend_bos(
        &mut tokens,
        bos_id.filter(|_| ferrox_models::tokenizer::should_add_bos_token(file)),
    );
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
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
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
        if !args.ignore_eos && stop_tokens.contains(next) {
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

/// GLM-5.2 / GLM4-family path via [`Glm52Engine`].
fn run_glm52_infer(args: InferArgs, path: &Path, file: &ShardedGguf) -> anyhow::Result<()> {
    let tokenizer = match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => CliTokenizer::Bpe(Box::new(GgufBpeTokenizer::from_gguf(file)?)),
        Some("llama") => CliTokenizer::Spm(GgufSpmTokenizer::from_gguf(file)?),
        Some("t5") => CliTokenizer::Unigram(GgufUnigramTokenizer::from_gguf(file)?),
        other => {
            eprintln!(
                "ferrox: unrecognized tokenizer.ggml.model ({other:?}); using byte tokenizer"
            );
            CliTokenizer::Byte
        }
    };
    // Not just `eos_token_id`: Llama-3 ends a turn with `<|eot_id|>` and
    // gemma-4 with `<turn|>`, neither of which is the metadata EOS.
    let stop_tokens = ferrox_models::tokenizer::StopTokens::from_gguf(file);
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
    let ctx_size = resolve_ctx_size(&args, path, gguf_ctx)?;

    let chat = ChatKind::detect_for_gguf(file, matches!(tokenizer, CliTokenizer::Byte));
    let user_prompt = resolve_prompt(&args)?;
    let prompt = if args.no_cnv {
        user_prompt
    } else {
        chat.wrap_user(args.system.as_deref(), &user_prompt)?
    };

    eprintln!(
        "ferrox: loading {} as GLM-5.2 engine (tokenizer={}, ctx={ctx_size})",
        args.model.as_deref().unwrap_or("?"),
        tokenizer.kind()
    );
    let load_t = Instant::now();
    let served = load_glm52_engine_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Glm52(engine) = served else {
        anyhow::bail!("expected ServedEngine::Glm52");
    };
    eprintln!("ferrox: loaded in {:.2}s", load_t.elapsed().as_secs_f64());

    let mut tokens = tokenizer.encode(&prompt);
    ferrox_models::tokenizer::prepend_bos(
        &mut tokens,
        bos_id.filter(|_| ferrox_models::tokenizer::should_add_bos_token(file)),
    );
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
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
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
        if !args.ignore_eos && stop_tokens.contains(next) {
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
