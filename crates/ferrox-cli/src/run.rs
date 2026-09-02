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
    load_mla_engine_from_path, select_engine_kind, Decoder, Engine, GgufBpeTokenizer,
    GgufSpmTokenizer, GgufUnigramTokenizer, ModelConfig, Sampler, SamplingParams,
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

    /// Constrain generation to a GBNF grammar (llama.cpp's `--grammar`).
    #[arg(long = "grammar")]
    pub grammar: Option<String>,

    /// Read the GBNF grammar from a file (llama.cpp's `--grammar-file`).
    #[arg(long = "grammar-file")]
    pub grammar_file: Option<std::path::PathBuf>,

    /// Constrain generation to a JSON Schema, converted to GBNF
    /// (llama.cpp's `-j` / `--json-schema`).
    #[arg(short = 'j', long = "json-schema")]
    pub json_schema: Option<String>,

    /// Min-p sampling: drop every candidate less than this fraction as
    /// likely as the most likely one (`0.0` = disabled).
    ///
    /// llama.cpp's `--min-p`, and its default is **0.05**, not off
    /// (`common/common.h:231`, `common/arg.cpp:1987`). ferrox had no
    /// min-p at all, so it could not reproduce llama.cpp's own
    /// out-of-the-box output for any prompt.
    #[arg(long = "min-p", default_value_t = 0.05)]
    pub min_p: f32,

    /// How many recent tokens the penalties consider (`0` = off).
    ///
    /// llama.cpp's `--repeat-last-n`, default 64
    /// (`common/common.h:238`). ferrox had no window and scanned the
    /// whole history.
    #[arg(long = "repeat-last-n", default_value_t = 64)]
    pub repeat_last_n: usize,

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

    /// GPU layers: `0`, `auto`, `all`, or a count at or above the
    /// model's layer count.
    ///
    /// Partial placement is not implemented, and a PARTIAL count is now
    /// REFUSED rather than silently rounded up -- see
    /// [`GpuLayers::check_supported`]. This comment used to say "any
    /// value above zero currently enables all supported operations",
    /// which described the behaviour that was the bug: llama.cpp's
    /// `-ngl N` offloads exactly N layers, so accepting the count and
    /// offloading everything turned the flag into an out-of-memory on
    /// the machine it exists to accommodate.
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

    /// Process `\\n` / `\\t` / `\\r` / `\\\\` escapes in `-p`. Use
    /// `--no-escape` to pass the prompt through literally.
    ///
    /// Defaults TRUE, matching llama.cpp (`common/common.h:563`), which
    /// also spells the negation `--no-escape` (`common/arg.cpp:1799`).
    /// ferrox defaulted false, so `-p "line one\\nline two"` reached the
    /// model as a literal backslash-n on ferrox and as a newline on
    /// llama.cpp -- the same command, a different prompt, and no error
    /// either way.
    #[arg(
        short = 'e',
        long = "escape",
        default_value_t = true,
        overrides_with = "no_escape"
    )]
    pub escape: bool,

    /// Pass the prompt through literally, without expanding escapes.
    #[arg(long = "no-escape", action = clap::ArgAction::SetTrue)]
    pub no_escape: bool,

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

/// Build the shared decode step, compiling the grammar against this
/// model's vocabulary if one was asked for.
///
/// Takes the tokenizer rather than a closure so the vocabulary view is
/// built once per run, not once per token.
fn token_step(
    args: &InferArgs,
    sampler: Sampler,
    tokenizer: &CliTokenizer,
    stop_tokens: &ferrox_models::tokenizer::StopTokens,
    vocab_size: usize,
) -> anyhow::Result<TokenStep> {
    let Some(src) = args.grammar_source()? else {
        return Ok(TokenStep::new(sampler, None));
    };
    // Compiled here, before the decode loop, so a grammar that does not
    // parse fails the command rather than the first token.
    let grammar = ferrox_models::grammar::Grammar::from_str_with_root(&src, "root")
        .map_err(|e| anyhow::anyhow!("grammar does not parse: {e}"))?;
    let grammar = ferrox_models::grammar_sampler::GrammarSampler::new(
        grammar,
        vocab_size,
        |id| tokenizer.decode(&[id]).into_bytes(),
        |id| stop_tokens.contains(id),
    );
    Ok(TokenStep::new(sampler, Some(grammar)))
}

/// One decode step, shared by every generation loop in this file.
///
/// There were FOUR byte-identical `sampler.sample(&logits, &sampling,
/// &generated)` call sites here -- the dense path, the engine path and
/// two chat paths. Adding a grammar to three of them and missing the
/// fourth would have produced unconstrained output on one code path with
/// every test still green, which is this repo's most-repeated bug and
/// was the same shape `InferArgs::sampling()` was introduced to kill.
///
/// Holds the grammar because the mask and the accept are two halves of
/// one hook: a caller that could take the mask without the accept would
/// keep asking "what may the FIRST token be" forever.
pub struct TokenStep {
    sampler: ferrox_models::sampling::Sampler,
    grammar: Option<ferrox_models::grammar_sampler::GrammarSampler>,
}

impl TokenStep {
    pub fn new(
        sampler: ferrox_models::sampling::Sampler,
        grammar: Option<ferrox_models::grammar_sampler::GrammarSampler>,
    ) -> Self {
        Self { sampler, grammar }
    }

    /// Whether this step must see one logit per vocabulary entry.
    ///
    /// A backend may fold `lm_head + argmax` into its decode stack and
    /// return a single token id instead of logits. That is sound only
    /// when nothing needs to look at the vocabulary first, and a grammar
    /// does. Read by the Metal greedy guard, which used to test the
    /// temperature alone.
    // Read only by the Metal greedy guard, so a CPU-only build has no
    // fold to refuse and this is genuinely dead there. Same shape and
    // same reason as `ferrox-models`'s `FoldedLmHead`.
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    pub fn needs_vocab_logits(&self) -> bool {
        self.grammar.is_some()
    }

    /// `Ok(None)` means the grammar is SATISFIED and has no legal
    /// continuation -- a finished answer, not a failure. An unsatisfied
    /// dead end is the `Err`.
    pub fn next(
        &mut self,
        logits: &[f32],
        sampling: &ferrox_models::sampling::SamplingParams,
        generated: &[usize],
    ) -> anyhow::Result<Option<usize>> {
        let Some(grammar) = self.grammar.as_mut() else {
            return Ok(Some(self.sampler.sample(logits, sampling, generated)));
        };
        let mut refusal = None;
        let mut outcome = ferrox_models::grammar_sampler::MaskOutcome::Allowed;
        let next = {
            let g = &*grammar;
            let mut mask = |scores: &mut [f32]| match g.mask_logits(scores) {
                Ok(o) => outcome = o,
                Err(e) => refusal = Some(e),
            };
            self.sampler
                .sample_with_mask(logits, sampling, generated, Some(&mut mask))
        };
        if let Some(e) = refusal {
            anyhow::bail!("grammar refused every continuation: {e}");
        }
        if outcome == ferrox_models::grammar_sampler::MaskOutcome::Complete {
            return Ok(None);
        }
        grammar.accept(next)?;
        Ok(Some(next))
    }
}

impl InferArgs {
    /// The grammar these flags describe, if any.
    ///
    /// The three spellings are llama.cpp's and are MUTUALLY EXCLUSIVE
    /// there. Refused together rather than silently picking one, because
    /// a caller who passed both asked for two different constraints and
    /// honouring either is answering a question they did not ask.
    pub fn grammar_source(&self) -> anyhow::Result<Option<String>> {
        let given = [
            self.grammar.is_some(),
            self.grammar_file.is_some(),
            self.json_schema.is_some(),
        ]
        .iter()
        .filter(|b| **b)
        .count();
        if given > 1 {
            anyhow::bail!(
                "--grammar, --grammar-file and --json-schema are mutually exclusive; \
                 pass exactly one"
            );
        }
        if let Some(g) = &self.grammar {
            return Ok(Some(g.clone()));
        }
        if let Some(path) = &self.grammar_file {
            return Ok(Some(std::fs::read_to_string(path).map_err(|e| {
                anyhow::anyhow!("--grammar-file {}: {e}", path.display())
            })?));
        }
        if let Some(schema) = &self.json_schema {
            // Converted here rather than at the sampler, so a schema that
            // cannot be expressed fails BEFORE the model is loaded.
            return Ok(Some(
                ferrox_models::grammar::json_schema_to_grammar(schema)
                    .map_err(|e| anyhow::anyhow!("--json-schema: {e}"))?,
            ));
        }
        Ok(None)
    }

    /// The sampler these flags describe.
    ///
    /// One function rather than one copy per generation path. There were
    /// four identical literals here (the dense path, the engine path and
    /// two chat paths), which is the shape `CLAUDE.md` names as this
    /// repo's most expensive failure: adding `--min-p` meant editing
    /// four places, and a sampler added to three of them would be
    /// silently absent from the fourth with every test still green.
    pub fn sampling(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            min_p: self.min_p,
            top_k: self.top_k,
            repetition_penalty: self.repeat_penalty,
            penalty_last_n: self.repeat_last_n,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
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

    /// Reject a PARTIAL offload rather than silently offloading
    /// everything.
    ///
    /// llama.cpp's `-ngl N` puts exactly `N` layers in VRAM and runs the
    /// rest on the CPU (`common/arg.cpp`), which is how people fit a
    /// model that does not otherwise fit. ferrox parses the count and
    /// then reads only `offload_enabled()`, a bool -- so `--ngl 10` on a
    /// 32-layer model offloaded all 32.
    ///
    /// That is the worst shape of divergence: same flag, same value, no
    /// error, and the failure lands as an out-of-memory on the machine
    /// the flag existed to accommodate.
    ///
    /// Partial offload is a real feature and not implemented here, so
    /// this REFUSES and names it. `0` (all CPU) and any count at or
    /// above the layer count (all GPU) are exact, and stay accepted.
    fn check_supported(self, n_layers: usize) -> anyhow::Result<()> {
        if let Self::Count(n) = self {
            let n = n as usize;
            if n > 0 && n < n_layers {
                anyhow::bail!(
                    "--ngl {n} asks for a PARTIAL offload ({n} of {n_layers} layers), which \
                     ferrox does not implement -- it would silently offload all {n_layers}. \
                     Use `--ngl 0` for CPU only, or `--ngl {n_layers}` / `--ngl all` for \
                     every layer."
                );
            }
        }
        Ok(())
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

/// The startup banner, as a value so a test can hold it to the
/// same `kv_elem_for` the budget prices with.
fn banner_line(args: &InferArgs, device: OffloadDevice) -> String {
    // The banner reports what this run WILL DO, not what was typed.
    // It used to echo `--ctk` verbatim, so a CPU run printed `ctk=f16`
    // while the host `KvCache` is `Vec<f32>` and the budget priced it
    // at f32 -- the same two-structures-must-agree shape as everywhere
    // else in this repo, and it made the memory warning look wrong
    // (double the KV bytes the banner implied) when the warning was
    // the only honest line of the two. `kv_elem_for` is now the single
    // source, so the number in the banner is the number in the budget.
    let effective_ctk = kv_elem_for(args);
    let requested_ctk = args.ctk.trim();
    let ctk_note = if effective_ctk.as_str() == requested_ctk {
        String::new()
    } else {
        format!(" (--ctk {requested_ctk} ignored: only the Metal KV store has a selectable dtype)")
    };

    format!(
        "ferrox: device={} gpu-layers={} ctk={}{}",
        match device {
            OffloadDevice::Auto => "auto",
            OffloadDevice::None => "none",
            OffloadDevice::Cpu => "cpu",
            OffloadDevice::Metal => "Metal",
            OffloadDevice::Cuda => "CUDA",
        },
        gpu_layers_note(args, device),
        effective_ctk.as_str(),
        ctk_note
    )
}

/// `-ngl` as the run will honour it. `-dev cpu -ngl all` offloads
/// nothing, and printing a bare `gpu-layers=all` there reads as a
/// promise the run does not keep.
fn gpu_layers_note(args: &InferArgs, device: OffloadDevice) -> String {
    let requested = args.n_gpu_layers.to_string();
    match device {
        OffloadDevice::None | OffloadDevice::Cpu if args.n_gpu_layers.offload_enabled() => {
            format!("{requested} (ignored, no GPU offload on this device)")
        }
        _ => requested,
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
             {} bytes over. That estimate is {}. `--ctx-size auto` would pick {}. {}",
            e.code(),
            e.estimated_bytes,
            e.limit_bytes,
            backend,
            budget.usable_provenance(),
            e.overage_bytes(),
            e.detail,
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

/// The tokenizer a GGUF's own metadata names, or a refusal.
///
/// Four call sites had a byte-by-byte copy of this match, each falling
/// back to `CliTokenizer::Byte` on anything unrecognised. That fallback
/// produced FLUENT GARBAGE: the model was fed ids from a vocabulary it
/// was never trained on, so it generated confidently and wrongly with
/// nothing in the output saying so. This project refuses everywhere
/// else rather than compute something different; the tokenizer was the
/// one place that did not.
///
/// The `Byte` variant went with it: the CLI has no synthetic-weight
/// path, so with the fallback gone nothing could construct it.
fn cli_tokenizer_from_gguf(file: &ShardedGguf) -> anyhow::Result<CliTokenizer> {
    match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => Ok(CliTokenizer::Bpe(Box::new(GgufBpeTokenizer::from_gguf(
            file,
        )?))),
        Some("llama") => Ok(CliTokenizer::Spm(GgufSpmTokenizer::from_gguf(file)?)),
        Some("t5") => Ok(CliTokenizer::Unigram(GgufUnigramTokenizer::from_gguf(
            file,
        )?)),
        // `bert` is NOT here because the tokenizer is missing -- ferrox
        // has WordPiece, and it is byte-exact against llama.cpp
        // (`ferrox parity`). It is here because this is the *generation*
        // path and a `bert` checkpoint is an encoder: no output head,
        // no logits, nothing to sample. The refusal names where it can
        // be used instead rather than repeating a claim that stopped
        // being true.
        Some("bert") => anyhow::bail!(
            "this checkpoint's tokenizer is `bert` (WordPiece), which means it is a BERT-family \
             ENCODER: it has no output head and cannot generate text, so there is nothing for \
             `ferrox run` to sample. Ferrox can embed with it: start ferrox-server with \
             FERROX_EMBEDDING_MODEL_PATH pointing at this file and POST /v1/embeddings."
        ),
        Some(known @ ("rwkv" | "none")) => anyhow::bail!(
            "this checkpoint's tokenizer is `{known}`, which ferrox cannot read yet. \
             Supported: `llama` (SentencePiece), `gpt2` and `gemma4` (BPE), `t5` (Unigram)."
        ),
        other => anyhow::bail!(
            "this checkpoint declares tokenizer.ggml.model = {other:?}, which ferrox does \
             not recognise. Supported: `llama`, `gpt2`, `gemma4`, `t5`. Serving it would \
             mean feeding the model ids from a vocabulary it was not trained on, which \
             produces fluent text that is wrong rather than an error."
        ),
    }
}

enum CliTokenizer {
    Bpe(Box<GgufBpeTokenizer>),
    Spm(GgufSpmTokenizer),
    Unigram(GgufUnigramTokenizer),
}

impl CliTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            CliTokenizer::Bpe(t) => t.encode(text).into_iter().map(|id| id as usize).collect(),
            CliTokenizer::Spm(t) => t.encode(text).into_iter().map(|id| id as usize).collect(),
            CliTokenizer::Unigram(t) => t.encode(text).into_iter().map(|id| id as usize).collect(),
        }
    }

    fn decode(&self, ids: &[usize]) -> String {
        let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        match self {
            CliTokenizer::Bpe(t) => t.decode(&ids32),
            CliTokenizer::Spm(t) => t.decode(&ids32),
            CliTokenizer::Unigram(t) => t.decode(&ids32),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            CliTokenizer::Bpe(_) => "gguf-bpe",
            CliTokenizer::Spm(_) => "gguf-spm",
            CliTokenizer::Unigram(_) => "gguf-unigram",
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
                ferrox_models::chat_template::ChatTemplate::vocab_has_chatml(file),
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
    if args.escape && !args.no_escape {
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

    eprintln!("{}", banner_line(args, device));
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
/// Loads a GGUF decoder, streaming experts when the weights will not
/// fit in memory.
///
/// The CLI could not stream AT ALL before this: `from_gguf_with_expert_cache`
/// was used only by `ferrox-server`, so `FERROX_SSD_STREAMING` and
/// `FERROX_EXPERT_CACHE_BYTES` were silently ignored by `ferrox -m`.
/// The CLI even printed advice to set the latter, for a feature it did
/// not implement. That matters because running a model too big for the
/// machine is the project's headline capability and the CLI is how
/// people run models.
///
/// Same decision as the server: explicit settings win in both
/// directions, an unknown amount of memory resolves to resident rather
/// than guessing, and enabling it says so, because streaming is slower
/// than resident and a slow run should never be a silent one.
pub(crate) fn load_decoder_streaming_if_needed(
    path: &std::path::Path,
    config: ferrox_models::config::ModelConfig,
) -> anyhow::Result<Decoder> {
    let explicit = std::env::var("FERROX_EXPERT_CACHE_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    let refused = matches!(
        std::env::var("FERROX_SSD_STREAMING").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    );
    let budget = if let Some(b) = explicit {
        Some(b)
    } else if refused {
        None
    } else {
        let weights = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let available = ferrox_core::host_memory::available_bytes();
        match ferrox_core::host_memory::plan_for(
            weights,
            available,
            /* headroom = */ 4 * 1024 * 1024 * 1024,
            /* floor = */ 2 * 1024 * 1024 * 1024,
        ) {
            ferrox_core::host_memory::FitPlan::Resident => None,
            ferrox_core::host_memory::FitPlan::Stream { cache_bytes } => {
                // REFUSE rather than stream. Expert streaming produces
                // WRONG OUTPUT on real checkpoints: OLMoE-1B-7B Q4_0
                // answers "Paris." resident and "amongst amongst, and
                // of" streamed, deterministically, at temperature 0.
                //
                // The fixture test
                // `store_backed_experts_produce_bit_identical_logits_to_resident`
                // passes, so whatever differs is not exercised by it.
                // Until that is understood, enabling this automatically
                // would turn "your model does not fit" into "your model
                // answers nonsense", which is far worse.
                let gib = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
                anyhow::bail!(
                    "this checkpoint is {:.1} GiB and only {:.1} GiB is available. Expert \
                     streaming would fit it in about {:.1} GiB, but it currently produces \
                     WRONG OUTPUT on real checkpoints and is not enabled automatically \
                     for that reason. Use a smaller quantization, or set \
                     FERROX_EXPERT_CACHE_BYTES explicitly to try streaming anyway and \
                     compare the output against llama.cpp yourself.",
                    gib(weights),
                    available.map(gib).unwrap_or(0.0),
                    gib(cache_bytes),
                );
            }
        }
    };
    Ok(Decoder::from_gguf_with_expert_cache(path, config, budget)?)
}

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
    // `glm4moe` is NOT here, and its absence is the fix. GLM-4.5,
    // GLM-4.5-Air and GLM-4.6 all tag `glm4moe`, and none of them is an
    // MLA model: `src/models/glm4-moe.cpp` reads no `q_lora_rank` and
    // builds plain Q/K/V. Sending them here made a real GLM-4.5-Air
    // download fail with "missing hparam glm4moe.attention.q_lora_rank",
    // a true statement about a key the architecture is not supposed to
    // have. It now reaches the generic path and refuses there, naming
    // the one thing that is actually missing -- the norm slot.
    // See `crates/ferrox-models/tests/glm4moe_refusal.rs`.
    if matches!(arch_early.as_str(), "glm-dsa" | "glm4") {
        return run_glm52_infer(args, path, &file);
    }

    let config = ModelConfig::from_gguf(&file)?;
    // Checked here rather than at parse time: the layer count is what
    // makes a given `--ngl N` exact or partial, and it is in the file.
    args.n_gpu_layers.check_supported(config.n_layers)?;
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

    let tokenizer = cli_tokenizer_from_gguf(&file)?;
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

    let chat = ChatKind::detect_for_gguf(&file, false);
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
    let decoder = load_decoder_streaming_if_needed(path, config)?;
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

    let sampling = args.sampling();
    let seed = seed_from_args(args.seed);
    let sampler = Sampler::new(seed);
    let mut step = token_step(
        &args,
        sampler,
        &tokenizer,
        &stop_tokens,
        decoder.config.vocab_size,
    )?;

    #[cfg(feature = "metal")]
    let _metal_greedy_guard = {
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                ferrox_models::set_metal_greedy_argmax(false);
            }
        }
        // NOT `temperature <= 0.0` alone. The fold makes the stack
        // return ONE element holding the chosen id, and a grammar needs
        // one logit per vocabulary entry to mask. Gating on temperature
        // only produced exactly that: `--json-schema` at `--temp 0`
        // failed with "was handed 1 for a vocabulary of 128256".
        //
        // Same defect `ferrox-server`'s `greedy_gpu_fold_allowed` fixed
        // for `json_object`, and the third instance of it. The rule is
        // the server's: the fold is sound only when NOTHING needs to
        // inspect the vocabulary before a token is chosen.
        if sampling.temperature <= 0.0 && !step.needs_vocab_logits() {
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
        let Some(next) = step.next(&logits, &sampling, &generated)? else {
            // The grammar is satisfied and permits nothing further: a
            // finished answer, not a failure.
            break;
        };
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
    let tokenizer = cli_tokenizer_from_gguf(file)?;
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

    let chat = ChatKind::detect_for_gguf(file, false);
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

    let sampling = args.sampling();
    let sampler = Sampler::new(seed_from_args(args.seed));
    let mut step = token_step(
        &args,
        sampler,
        &tokenizer,
        &stop_tokens,
        engine.vocab_size(),
    )?;
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
        let Some(next) = step.next(&logits, &sampling, &generated)? else {
            // The grammar is satisfied and permits nothing further: a
            // finished answer, not a failure.
            break;
        };
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
    let tokenizer = cli_tokenizer_from_gguf(file)?;
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

    let chat = ChatKind::detect_for_gguf(file, false);
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

    let sampling = args.sampling();
    let sampler = Sampler::new(seed_from_args(args.seed));
    let mut step = token_step(
        &args,
        sampler,
        &tokenizer,
        &stop_tokens,
        engine.vocab_size(),
    )?;
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
        let Some(next) = step.next(&logits, &sampling, &generated)? else {
            // The grammar is satisfied and permits nothing further: a
            // finished answer, not a failure.
            break;
        };
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
    let tokenizer = cli_tokenizer_from_gguf(file)?;
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

    let chat = ChatKind::detect_for_gguf(file, false);
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

    let sampling = args.sampling();
    let sampler = Sampler::new(seed_from_args(args.seed));
    let mut step = token_step(
        &args,
        sampler,
        &tokenizer,
        &stop_tokens,
        engine.vocab_size(),
    )?;
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
        let Some(next) = step.next(&logits, &sampling, &generated)? else {
            // The grammar is satisfied and permits nothing further: a
            // finished answer, not a failure.
            break;
        };
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

    use super::{banner_line, kv_elem_for, InferArgs, OffloadDevice};
    use clap::Parser;

    /// `InferArgs` is a `clap::Args` group, not a `Parser`, so the test
    /// gives it the top-level command it is normally flattened into.
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        infer: InferArgs,
    }

    fn args(argv: &[&str]) -> InferArgs {
        let mut full = vec!["ferrox"];
        full.extend_from_slice(argv);
        Cli::parse_from(full).infer
    }

    /// The banner may not promise a KV dtype the run will not use.
    ///
    /// `ferrox -m m.gguf -dev cpu -ngl all` printed `ctk=f16` because
    /// the banner echoed the flag's default, while the host `KvCache`
    /// is `Vec<f32>` and the budget priced it at f32. That made the
    /// memory warning look like a bug: it charged 229376 bytes/token
    /// where the banner implied 114688, so a 3B model at its 131072
    /// trained context read as 37.5 GB instead of 22.5 GB. The warning
    /// was right and the banner was wrong.
    ///
    /// The flag and the store are two things that must agree, so the
    /// banner now derives from `kv_elem_for`, the same function the
    /// budget prices with.
    #[test]
    fn the_banner_reports_the_kv_dtype_the_run_will_actually_keep() {
        let a = args(&["-m", "m.gguf", "--device", "cpu", "--ctk", "f16"]);
        assert_eq!(kv_elem_for(&a).as_str(), "f32", "the host KV cache is f32");

        let line = banner_line(&a, OffloadDevice::Cpu);
        assert!(line.contains("ctk=f32"), "{line}");
        assert!(
            !line.contains("ctk=f16"),
            "the banner echoed the flag: {line}"
        );
        assert!(
            line.contains("--ctk f16 ignored"),
            "a flag with no effect must say so: {line}"
        );
    }

    /// `-dev cpu` is not `-dev none`, and `-ngl all` under either does
    /// nothing. Both were printed as if honoured.
    #[test]
    fn the_banner_does_not_promise_gpu_layers_on_a_cpu_device() {
        let a = args(&["-m", "m.gguf", "--device", "cpu", "--ngl", "all"]);
        let line = banner_line(&a, OffloadDevice::Cpu);
        assert!(line.contains("device=cpu"), "{line}");
        assert!(line.contains("ignored, no GPU offload"), "{line}");
    }

    /// And a run that really does select the dtype keeps a clean line.
    #[test]
    fn a_metal_run_reports_the_requested_dtype_with_no_caveat() {
        let a = args(&[
            "-m", "m.gguf", "--device", "metal", "--ngl", "all", "--ctk", "f16",
        ]);
        let line = banner_line(&a, OffloadDevice::Metal);
        assert!(line.contains("ctk=f16"), "{line}");
        assert!(!line.contains("ignored"), "{line}");
    }

    /// `-ngl N` must not silently mean "all layers".
    ///
    /// llama.cpp's `-ngl N` puts exactly N layers in VRAM and runs the
    /// rest on the CPU, which is how people fit a model that otherwise
    /// does not fit. ferrox parsed the count and read only
    /// `offload_enabled()`, a bool, so `--ngl 10` on a 32-layer model
    /// offloaded all 32 -- same flag, same value, no error, and the
    /// failure arrives as an OOM on the machine the flag existed to
    /// accommodate.
    ///
    /// Partial offload is not implemented, so it refuses. The two exact
    /// cases still work.
    #[test]
    fn a_partial_gpu_layer_count_is_refused_rather_than_rounded_up() {
        let err = GpuLayers::Count(10)
            .check_supported(32)
            .expect_err("10 of 32 is partial");
        let msg = err.to_string();
        assert!(msg.contains("PARTIAL"), "{msg}");
        assert!(
            msg.contains("--ngl 0"),
            "the message must say what works: {msg}"
        );
        assert!(msg.contains("--ngl 32"), "{msg}");

        // The exact cases are not partial and must stay accepted.
        GpuLayers::Count(0)
            .check_supported(32)
            .expect("0 = CPU only");
        GpuLayers::Count(32).check_supported(32).expect("32 = all");
        GpuLayers::Count(99)
            .check_supported(32)
            .expect("clamps to all");
        GpuLayers::All.check_supported(32).expect("all");
        GpuLayers::Auto.check_supported(32).expect("auto");
    }

    /// `--escape` defaults TRUE, as llama.cpp does.
    ///
    /// ferrox defaulted false, so `-p "a\\nb"` reached the model as a
    /// literal backslash-n on ferrox and as a newline on llama.cpp: the
    /// same command, a different prompt, and no error on either side.
    #[test]
    fn escapes_are_processed_by_default_like_llama_cpp() {
        use clap::Parser;
        #[derive(Parser)]
        struct Probe {
            #[command(flatten)]
            args: super::InferArgs,
        }
        let parsed = Probe::try_parse_from(["ferrox", "-m", "x.gguf"]).expect("defaults parse");
        assert!(parsed.args.escape, "llama.cpp common/common.h:563 is true");
        assert!(!parsed.args.no_escape);

        let off = Probe::try_parse_from(["ferrox", "-m", "x.gguf", "--no-escape"])
            .expect("--no-escape parses");
        assert!(off.args.no_escape, "llama.cpp spells the negation this way");
    }

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
