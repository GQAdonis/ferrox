//! ferrox-server: OpenAI-compatible HTTP surface (`/health`,
//! `/v1/models`, `/v1/chat/completions`, `/v1/completions`,
//! `/v1/tokenize`, `/v1/detokenize`, `/v1/embeddings`) over the
//! ferrox-models decoder, plus a whole-response cache for exact-repeat
//! requests (see `cache` module). Loads a real GGUF checkpoint and its
//! own real tokenizer when `-m`/`--model` or `FERROX_MODEL_PATH` is set
//! (see `model` module). Supports sampling
//! (temperature/top_p/top_k/repetition_penalty), stop sequences, and SSE
//! streaming (see `generate` module).
//!
//! Concurrency: the loaded model
//! (`Model`) is immutable once loaded and shared via `Arc`, not locked
//! behind a `Mutex` -- there is no shared mutable decoder state for
//! concurrent requests to contend on or for one panicking request to
//! poison. The *pointer* to it is swappable (`AppState::active`, behind
//! an `RwLock` held only long enough to clone one `Arc`), which is what
//! `/admin/models/load` swaps; a request that has already cloned its
//! handle finishes against the exact weights it started on, and the old
//! model is freed when the last such request lets go.
//! Each request builds its own KV cache (see `generate::generate`)
//! and runs its decode loop on tokio's blocking-thread pool via
//! `spawn_blocking`, so CPU-bound generation no longer blocks the async
//! reactor threads -- multiple requests can decode genuinely
//! concurrently, bounded by that pool rather than serialized through one
//! lock. Only the small whole-response cache is still mutable shared
//! state, and it's locked only for the brief get/put around it, never
//! across a decode.
//!
//! Streaming scope: when `stream: true` and tools are inactive, each
//! decoded chunk is pushed through a bounded `mpsc` channel from the
//! blocking generate task into the SSE writer so time-to-first-byte
//! overlaps with ongoing decode. Tool-call requests still buffer the
//! full response first (detection needs the stop-bounded text).
//! Continuous-batching streaming also buffers (batcher returns one
//! string).

mod admin;
mod anthropic;
mod batch_scheduler;
mod cache;
mod cancel;
mod chat_template;
mod generate;
mod health;
mod hub;
mod journal;
mod json_mode;
mod limits;
mod mcp;
mod model;
mod openai_extra;
mod security;
mod session;
mod stats;
mod stop;
mod tasks;
mod ui;

use std::convert::Infallible;
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

use cache::{CacheKey, ResponseCache};
use ferrox_core::cache::KvBlockPool;
use ferrox_models::kimi_tokenizer::KimiTokenizer;
use ferrox_models::sampling::SamplingParams;
use ferrox_models::tokenizer::StopTokens;
use ferrox_models::{Decoder, Gemma4Engine, KimiEngine, MlaEngine, PrefixCache};
use generate::{FinishReason, GenerationParams};
use model::ServerTokenizer;

#[derive(Parser, Debug)]
#[command(
    name = "ferrox-server",
    version,
    about = "OpenAI-compatible Ferrox inference server"
)]
struct ServerArgs {
    /// Model path (GGUF file or Kimi checkpoint directory).
    #[arg(short = 'm', long = "model", value_name = "FILE")]
    model: Option<String>,

    /// IP address to listen on.
    #[arg(long, value_name = "HOST")]
    host: Option<IpAddr>,

    /// Port to listen on. `0` asks the kernel for a free one; the
    /// actually-bound address is then announced on stdout (see
    /// [`announce_ready`]), which is how a supervising process is meant
    /// to learn it.
    #[arg(long, value_name = "PORT")]
    port: Option<u16>,

    /// CPU threads (sets FERROX_CPU_THREADS and RAYON_NUM_THREADS).
    #[arg(short = 't', long = "threads", value_name = "N")]
    threads: Option<usize>,

    /// Device used for offloading (`none` disables GPU use).
    #[arg(
        long = "device",
        visible_alias = "dev",
        value_name = "DEVICE",
        ignore_case = true
    )]
    device: Option<OffloadDevice>,

    /// Print available offload devices and exit.
    #[arg(long = "list-devices", default_value_t = false)]
    list_devices: bool,

    /// GPU layers: `0`, a positive number, `auto`, or `all`.
    ///
    /// Partial placement is not implemented yet; any value above zero
    /// currently enables all supported operations on the selected backend.
    #[arg(
        long = "n-gpu-layers",
        visible_aliases = ["gpu-layers", "ngl"],
        value_name = "N"
    )]
    n_gpu_layers: Option<GpuLayers>,

    /// Serve the embedded Ferrox Studio UI at `/` and `/ui` (chat,
    /// models, activity and connect screens, all driven by the same
    /// public HTTP API any other client uses).
    #[arg(long = "ui-server", default_value_t = false)]
    ui_server: bool,

    /// MCP tool-server config JSON (stub: listed in `/v1/models` metadata).
    #[arg(long = "mcp-config", value_name = "PATH")]
    mcp_config: Option<PathBuf>,

    /// Exit when stdin reaches EOF (for a supervising parent process).
    ///
    /// Opt-in on purpose: a server started with stdin redirected from
    /// `/dev/null` -- systemd, cron, `nohup` -- sees EOF immediately,
    /// and making this the default would turn those into a server that
    /// exits the moment it starts. A parent that *wants* the guarantee
    /// (the desktop shell) passes the flag and keeps the pipe open.
    #[arg(long = "exit-on-stdin-close", default_value_t = false)]
    exit_on_stdin_close: bool,

    /// Start even though another ferrox process is already holding a
    /// model. Off by default: two models on one box do not share it,
    /// they thrash it, and both serve slower than either would alone.
    /// `FERROX_ALLOW_MULTIPLE_INSTANCES=1` does the same.
    #[arg(long = "allow-multiple-instances", default_value_t = false)]
    allow_multiple_instances: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OffloadDevice {
    Auto,
    None,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuLayers {
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

fn rewrite_llama_style_argv(args: Vec<String>) -> Vec<String> {
    args.into_iter()
        .map(|arg| match arg.as_str() {
            "-ngl" => "--n-gpu-layers".into(),
            "-dev" => "--device".into(),
            _ => arg,
        })
        .collect()
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

fn cli_bind_addr(args: &ServerArgs, env_addr: Option<&str>) -> Option<String> {
    if args.host.is_none() && args.port.is_none() {
        return None;
    }

    let existing = env_addr.and_then(|value| value.parse::<SocketAddr>().ok());
    let host = args
        .host
        .or_else(|| existing.map(|addr| addr.ip()))
        .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
    let port = args
        .port
        .or_else(|| existing.map(|addr| addr.port()))
        .unwrap_or(8383);
    Some(SocketAddr::new(host, port).to_string())
}

fn apply_cli_overrides(args: &ServerArgs) -> anyhow::Result<()> {
    if let Some(model) = &args.model {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FERROX_MODEL_PATH", model) };
    }

    if let Some(addr) = cli_bind_addr(args, std::env::var("FERROX_ADDR").ok().as_deref()) {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FERROX_ADDR", addr) };
    }

    if let Some(threads) = args.threads {
        if threads == 0 {
            anyhow::bail!("--threads must be greater than zero");
        }
        // SAFETY: called before the runtime starts worker threads.
        unsafe {
            std::env::set_var("FERROX_CPU_THREADS", threads.to_string());
            std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
        }
    }

    if args.device.is_none() && args.n_gpu_layers.is_none() {
        // device overrides skipped
    } else {
        let layers = args.n_gpu_layers.unwrap_or(GpuLayers::Auto);
        let device = if layers.offload_enabled() {
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
                if std::env::var_os("FERROX_METAL_ATTN").is_none() {
                    std::env::set_var("FERROX_METAL_ATTN", "1");
                }
            },
            OffloadDevice::Metal => {
                #[cfg(not(feature = "metal"))]
                {
                    anyhow::bail!(
                        "Metal requested but this binary was built without --features metal"
                    );
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
                    anyhow::bail!(
                        "CUDA requested but this binary was built without --features cuda"
                    );
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
    }

    if args.ui_server {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FERROX_UI", "1") };
    }

    Ok(())
}

/// The loaded model: immutable once built, so it needs no lock at all --
/// just cheap `Arc` sharing across concurrent request tasks. Two real
/// checkpoint shapes exist (see `model::LoadedModel`'s doc comment for
/// why `FERROX_MODEL_PATH` picks between them); everything that isn't
/// engine-specific (chat template, tokenizer kind reporting, whether
/// this is the synthetic demo) goes through the small inherent methods
/// below rather than being matched on ad hoc at every call site.
#[allow(clippy::large_enum_variant)] // KimiEngine/MlaEngine dwarf Arc<Decoder>; boxing would churn call sites
pub(crate) enum Model {
    Gguf(GgufModel),
    Kimi(KimiModel),
    Mla(MlaModel),
    Gemma4(Gemma4Model),
    Glm52(Glm52Model),
}

pub(crate) struct GgufModel {
    decoder: Arc<Decoder>,
    tokenizer: Arc<ServerTokenizer>,
    stop_tokens: StopTokens,
    bos_id: Option<usize>,
    is_synthetic: bool,
    chat_template: chat_template::ChatTemplate,
}

pub(crate) struct KimiModel {
    engine: KimiEngine,
    tokenizer: KimiTokenizer,
    stop_tokens: StopTokens,
    chat_template: chat_template::ChatTemplate,
}

pub(crate) struct MlaModel {
    engine: MlaEngine,
    tokenizer: ServerTokenizer,
    stop_tokens: StopTokens,
    bos_id: Option<usize>,
    name: String,
    chat_template: chat_template::ChatTemplate,
}

pub(crate) struct Gemma4Model {
    engine: Gemma4Engine,
    tokenizer: ServerTokenizer,
    stop_tokens: StopTokens,
    bos_id: Option<usize>,
    name: String,
    chat_template: chat_template::ChatTemplate,
}

pub(crate) struct Glm52Model {
    engine: ferrox_models::Glm52Engine,
    tokenizer: ServerTokenizer,
    stop_tokens: StopTokens,
    bos_id: Option<usize>,
    name: String,
    chat_template: chat_template::ChatTemplate,
}

impl Model {
    pub(crate) fn chat_template(&self) -> chat_template::ChatTemplate {
        match self {
            Model::Gguf(m) => m.chat_template,
            Model::Kimi(m) => m.chat_template,
            Model::Mla(m) => m.chat_template,
            Model::Gemma4(m) => m.chat_template,
            Model::Glm52(m) => m.chat_template,
        }
    }

    /// Kimi K3 / MLA / GLM-5.2 have no synthetic-weight demo path through this
    /// server (unlike GGUF, which falls back to one when
    /// `FERROX_MODEL_PATH` is unset) -- a loaded `Model::Kimi` /
    /// `Model::Mla` / `Model::Glm52` is always a real checkpoint.
    fn is_synthetic(&self) -> bool {
        match self {
            Model::Gguf(m) => m.is_synthetic,
            Model::Kimi(_) | Model::Mla(_) | Model::Gemma4(_) | Model::Glm52(_) => false,
        }
    }

    fn tokenizer_kind(&self) -> &'static str {
        match self {
            Model::Gguf(m) => m.tokenizer.kind(),
            Model::Kimi(_) => "kimi-tiktoken-bpe",
            Model::Mla(m) => m.tokenizer.kind(),
            Model::Gemma4(m) => m.tokenizer.kind(),
            Model::Glm52(m) => m.tokenizer.kind(),
        }
    }

    /// Live counters of the bounded expert cache, when the model
    /// streams routed experts (`FERROX_EXPERT_CACHE_BYTES`); `None`
    /// for fully resident models.
    fn expert_store_stats(&self) -> Option<ferrox_core::expert_store::ExpertStoreStats> {
        match self {
            Model::Gguf(m) => m.decoder.expert_store_stats(),
            Model::Kimi(m) => m.engine.weights.expert_store_stats(),
            Model::Mla(_) | Model::Gemma4(_) | Model::Glm52(_) => None,
        }
    }

    pub(crate) fn name(&self) -> &str {
        match self {
            Model::Gguf(m) => m.decoder.config.name,
            Model::Kimi(_) => "kimi-k3",
            Model::Mla(m) => m.name.as_str(),
            Model::Gemma4(m) => m.name.as_str(),
            Model::Glm52(m) => m.name.as_str(),
        }
    }

    pub(crate) fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            Model::Gguf(m) => m.tokenizer.encode(text),
            Model::Kimi(m) => m
                .tokenizer
                .encode(text)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
            Model::Mla(m) => m.tokenizer.encode(text),
            Model::Gemma4(m) => m.tokenizer.encode(text),
            Model::Glm52(m) => m.tokenizer.encode(text),
        }
    }

    pub(crate) fn decode(&self, ids: &[usize]) -> String {
        match self {
            Model::Gguf(m) => m.tokenizer.decode(ids),
            Model::Kimi(m) => {
                let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
                m.tokenizer.decode(&ids32)
            }
            Model::Mla(m) => m.tokenizer.decode(ids),
            Model::Gemma4(m) => m.tokenizer.decode(ids),
            Model::Glm52(m) => m.tokenizer.decode(ids),
        }
    }

    /// Final-normed last-layer hidden states for GGUF Decoder only.
    /// Returns `None` for engines without a hidden-state hook (e.g. Kimi/MLA/GLM).
    pub(crate) fn embed_tokens(&self, tokens: &[usize]) -> Option<Vec<Vec<f32>>> {
        match self {
            Model::Gguf(m) => {
                let mut caches: Vec<_> = (0..m.decoder.layers.len())
                    .map(|_| {
                        ferrox_core::cache::KvCache::new(
                            m.decoder.config.n_kv_heads,
                            m.decoder.config.head_dim,
                        )
                    })
                    .collect();
                Some(m.decoder.forward_hidden_batch(tokens, 0, &mut caches))
            }
            Model::Kimi(_) | Model::Mla(_) | Model::Gemma4(_) | Model::Glm52(_) => None,
        }
    }

    pub(crate) fn vocab_size(&self) -> Option<usize> {
        match self {
            Model::Gguf(m) => Some(m.decoder.config.vocab_size),
            Model::Kimi(m) => Some(m.tokenizer.vocab_size()),
            Model::Mla(m) => Some(ferrox_models::Engine::vocab_size(&m.engine)),
            Model::Gemma4(m) => Some(ferrox_models::Engine::vocab_size(&m.engine)),
            Model::Glm52(m) => Some(ferrox_models::Engine::vocab_size(&m.engine)),
        }
    }
}

/// The model the server is serving *right now*, together with the
/// pieces that are built from it and must be replaced with it.
///
/// The continuous batcher owns a worker thread holding an
/// `Arc<Decoder>`, so it belongs to one specific model: keeping it in a
/// separate field would let a swap leave a batcher decoding against the
/// old weights while `Model` named the new ones. Bundling them means
/// one `Arc` swap replaces a consistent pair.
pub(crate) struct ActiveModel {
    /// Admin-surface id (see `admin::discover`), or `None` for a model
    /// that was not discovered through it -- the synthetic fallback, or
    /// a `FERROX_MODEL_PATH` outside the scanned directory.
    pub(crate) id: Option<String>,
    pub(crate) model: Arc<Model>,
    /// Opt-in continuous-batching decode worker (`FERROX_CONTINUOUS_BATCHING=1`).
    /// Shares `forward_multi_seq` across concurrent GGUF requests. Disabled
    /// when a KV pool or prefix cache is configured (those keep the
    /// private-loop `generate` path).
    pub(crate) batcher: Option<batch_scheduler::ContinuousBatcher>,
}

pub(crate) struct AppState {
    /// The swappable active model.
    ///
    /// **A reader clones the `Arc` under the read lock and then runs;
    /// the lock is never held across a decode.** That is the whole
    /// design: `RwLock` guards the *pointer*, not the model, so
    /// `/admin/models/load` swapping in a new `Arc` cannot stall a
    /// request that is already generating, and a request that started
    /// against the old model keeps decoding against the exact weights
    /// it began with until it finishes -- the old `ActiveModel` (and
    /// its batcher thread) is dropped only when the last in-flight
    /// holder releases it, not when the swap happens. Requests that
    /// arrive after the swap see the new model. There is deliberately
    /// no attempt to migrate an in-flight request: half a completion
    /// from one checkpoint and half from another is worse than either.
    ///
    /// `None` means nothing is loaded (after `/admin/models/unload`, or
    /// a failed startup load): generation endpoints answer 503 rather
    /// than pretending, and `/health` reports `unavailable`.
    active: std::sync::RwLock<Option<Arc<ActiveModel>>>,
    /// Set while a load task is in flight, so a second load request is
    /// rejected instead of racing the first. A load is not cheap and
    /// two concurrent ones would fight for the same memory.
    pub(crate) load_in_progress: std::sync::atomic::AtomicBool,
    /// Long-running jobs (download, load) -- see the `tasks` module.
    pub(crate) tasks: Arc<tasks::TaskRegistry>,
    /// Generations that can currently be stopped by `POST /v1/cancel`
    /// -- see the `cancel` module for why a dropped socket alone is not
    /// enough.
    pub(crate) cancels: Arc<cancel::CancelRegistry>,
    /// Recent-request ring buffer and the counters behind
    /// `/admin/stats` -- see the `stats` module.
    pub(crate) stats: stats::Stats,
    /// The directory `/admin/models` scans, when one is configured.
    pub(crate) model_dir: Option<PathBuf>,
    /// The only shared *mutable* state in the server. Locked only for
    /// the brief get/put around a cache lookup, never held across a
    /// decode -- see the module doc comment.
    response_cache: Mutex<ResponseCache>,
    /// `Some` when `FERROX_KV_POOL_BLOCKS`/`FERROX_KV_POOL_BLOCK_SIZE`
    /// are set: every request's per-layer KV caches then draw from
    /// this one shared, bounded pool instead of each growing
    /// unboundedly. A request whose caches can't get their first block
    /// retries for up to `FERROX_KV_POOL_QUEUE_TIMEOUT_MS` (zero by
    /// default -- reject immediately) before being rejected with 503,
    /// rather than being admitted regardless of how many other
    /// requests are already decoding -- see
    /// `ferrox_core::cache::KvBlockPool` and `generate::KvPoolConfig`.
    /// `None` (the default) preserves the
    /// original unbounded-per-request behavior exactly.
    pub(crate) kv_pool: Option<generate::KvPoolConfig>,
    /// `Some` when `FERROX_PREFIX_CACHE_ENTRIES` is set: a shared,
    /// LRU-bounded store of previously processed prompt+KV-state
    /// snapshots (see `ferrox_models::PrefixCache`), consulted so a
    /// request that *extends* an earlier one -- the common multi-turn-
    /// chat case -- can skip recomputing the shared part. Mutually
    /// exclusive with `kv_pool` (see `generate::generate`'s doc
    /// comment for why); `None` (the default) means every request
    /// processes its full prompt from scratch, exactly as before this
    /// existed.
    pub(crate) prefix_cache: Option<Arc<Mutex<PrefixCache>>>,
    /// Server-side per-session conversation history -- see
    /// `session::SessionStore`'s doc comment.
    /// Always present (unlike `kv_pool`/`prefix_cache`, it's not
    /// opt-in): a request that never sends `session_id` simply never
    /// touches it, at negligible cost (one empty `HashMap`).
    sessions: session::SessionStore,
    requests_total: std::sync::atomic::AtomicU64,
    request_errors_total: std::sync::atomic::AtomicU64,
    started_at: std::time::Instant,
    /// Milliseconds after `started_at` at which the last request
    /// finished; 0 means none has. Reported by `/health` as an age, so a
    /// client that sees a slow health poll from a GPU-saturated server
    /// has positive evidence of liveness instead of declaring it dead.
    last_request_ms: std::sync::atomic::AtomicU64,
    /// Backend capability probe behind `/health` (see `health` module).
    detection: Arc<health::Detection>,
    /// Loaded MCP config (`--mcp-config`); tool invocation not wired yet.
    mcp: Option<mcp::LoadedMcpConfig>,
    /// Whether a swapped-in GGUF model should get a continuous-batching
    /// worker, decided once at startup from the same env var and
    /// exclusions as the initial load.
    pub(crate) continuous_batching_enabled: bool,
    /// The model id a load task is currently working on, so
    /// `/admin/models` can report `loading` for it. Separate from
    /// `load_in_progress` because that is a gate and this is a label.
    loading_model: Mutex<Option<String>>,
    /// The last failed load, as `(model id, message)`. Sticky until the
    /// next successful load so `/admin/models` can say *why* an entry
    /// is in `error` without the user retrying to find out.
    last_load_error: Mutex<Option<(String, String)>>,
}

impl AppState {
    /// Clones the active model's `Arc` and releases the lock before
    /// returning. Every caller then runs against its own handle, so no
    /// decode ever holds this lock -- see [`AppState::active`].
    pub(crate) fn active(&self) -> Option<Arc<ActiveModel>> {
        self.active
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// [`AppState::active`] for a request that cannot proceed without a
    /// model. 503 with a `Retry-After`-shaped explanation is the honest
    /// answer while nothing is loaded; the alternative -- keeping a
    /// stale model around so the endpoint never fails -- would serve
    /// tokens from a checkpoint the operator explicitly unloaded.
    pub(crate) fn require_active(&self) -> Result<Arc<ActiveModel>, ApiError> {
        self.active().ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": {
                    "message": "no model is loaded; POST /admin/models/load with an id from \
                                GET /admin/models",
                    "type": "model_not_loaded"
                }})),
            )
        })
    }

    /// [`AppState::active`]'s model only, for the many call sites that
    /// do not care about the batcher.
    pub(crate) fn require_model(&self) -> Result<Arc<Model>, ApiError> {
        Ok(Arc::clone(&self.require_active()?.model))
    }

    /// Publishes a new active model (or `None` to unload) and returns
    /// the previous one.
    ///
    /// The write lock is held only for the pointer swap. The returned
    /// value is the caller's to drop *outside* the lock: dropping a
    /// multi-gigabyte model can take a moment, and doing it under the
    /// lock would block every reader for exactly as long.
    pub(crate) fn swap_active(&self, next: Option<Arc<ActiveModel>>) -> Option<Arc<ActiveModel>> {
        let mut guard = self.active.write().unwrap_or_else(|p| p.into_inner());
        std::mem::replace(&mut *guard, next)
    }

    /// Stamps "a request just finished" for `/health`'s liveness
    /// vouching. Relaxed: this is a freshness hint, not a
    /// synchronization point.
    fn mark_request_finished(&self) {
        let ms = self.started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
        self.last_request_ms
            .store(ms, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub(crate) fn requests_total(&self) -> u64 {
        self.requests_total
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn errors_total(&self) -> u64 {
        self.request_errors_total
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn cache_stats(&self) -> cache::CacheStats {
        lock_cache(&self.response_cache).stats()
    }

    /// Seconds since the last request finished, or `None` when none
    /// has. Same derivation `/health` uses, so the two agree.
    pub(crate) fn last_request_age_seconds(&self) -> Option<f64> {
        let last = self
            .last_request_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        (last > 0)
            .then(|| self.uptime().as_secs_f64() - (last as f64 / 1000.0))
            .map(|age| age.max(0.0))
    }

    pub(crate) fn loading_model_id(&self) -> Option<String> {
        self.loading_model
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub(crate) fn set_loading_model(&self, id: Option<String>) {
        *self.loading_model.lock().unwrap_or_else(|p| p.into_inner()) = id;
    }

    pub(crate) fn last_load_error(&self) -> Option<(String, String)> {
        self.last_load_error
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub(crate) fn set_last_load_error(&self, error: Option<(String, String)>) {
        *self
            .last_load_error
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = error;
    }

    /// Records one finished request in the `/admin/stats` ring buffer.
    pub(crate) fn record_request(
        &self,
        request_id: &str,
        route: &str,
        status: u16,
        stream: bool,
        duration_ms: u64,
        usage: Option<&ferrox_api::Usage>,
    ) {
        self.stats.record(stats::entry(
            request_id,
            route,
            status,
            stream,
            duration_ms,
            usage,
        ));
    }
}

/// Defense in depth: if a panic ever happened while this lock was held
/// (none of the CPU-bound decode work runs under it, so this should be
/// very unlikely), recovering the inner state on poison rather than
/// `.unwrap()`ing keeps the cache from permanently bricking the server.
fn lock_cache(cache: &Mutex<ResponseCache>) -> MutexGuard<'_, ResponseCache> {
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image_url: Option<serde_json::Value>,
}

impl MessageContent {
    fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.text.as_deref())
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    fn has_image(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts
                .iter()
                .any(|p| p.kind == "image_url" || p.image_url.is_some()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    /// `None` for an assistant message that made tool calls instead of
    /// replying with text (the real OpenAI convention: `content` and
    /// `tool_calls` are mutually exclusive on an assistant message).
    #[serde(default)]
    pub(crate) content: Option<MessageContent>,
    /// Present on a replayed assistant message that previously made
    /// one or more tool calls (conversation history a client sends
    /// back on a follow-up request).
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<ToolCallIn>>,
    /// Present on a `"tool"`-role message carrying a call's result
    /// (unused by rendering today -- `role` alone already
    /// distinguishes it -- but accepted so real OpenAI-shaped tool-
    /// result messages deserialize without error).
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) tool_call_id: Option<String>,
}

impl ChatMessage {
    /// The text this message actually contributes to a rendered
    /// prompt: `content` verbatim for an ordinary message, or (for a
    /// replayed assistant message carrying `tool_calls`) each call
    /// re-rendered as the same `<tool_call>{...}</tool_call>` marker
    /// text a model is asked to produce for a *new* call -- see
    /// `chat_template`'s module doc comment for why.
    fn rendered_content(&self) -> String {
        let mut out = self
            .content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default();
        if let Some(calls) = &self.tool_calls {
            for call in calls {
                out.push_str(&format!(
                    "<tool_call>{{\"name\": \"{}\", \"arguments\": {}}}</tool_call>",
                    call.function.name, call.function.arguments
                ));
            }
        }
        out
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallIn {
    #[serde(default)]
    #[allow(dead_code)]
    id: String,
    #[serde(rename = "type", default)]
    #[allow(dead_code)]
    kind: String,
    function: ToolCallFunctionIn,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolCallFunctionIn {
    name: String,
    /// A JSON-encoded string (the real OpenAI convention for
    /// `tool_calls[].function.arguments`), not a nested object --
    /// spliced directly into the re-rendered `<tool_call>{...}` marker
    /// text since it's already valid JSON.
    arguments: String,
}

/// A tool definition in the real OpenAI request shape:
/// `{"type": "function", "function": {"name", "description", "parameters"}}`.
#[derive(Debug, Clone, Deserialize)]
struct ToolDef {
    #[serde(rename = "type", default)]
    #[allow(dead_code)]
    kind: String,
    function: ToolFunctionDef,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolFunctionDef {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
}

/// OpenAI's `tool_choice`: `"auto"`/`"none"`/`"required"`, or an object
/// pinning one specific function. Only whether it's literally
/// `"none"` is actually consulted (to suppress tool-calling prompting
/// entirely) -- forcing a *specific* named call isn't implementable
/// honestly without grammar-constrained decoding (which doesn't exist
/// in this server), so `"required"` and a
/// specific-function choice are both treated the same as `"auto"`:
/// offered, not forced. A real, disclosed simplification, not silently
/// wrong behavior.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ToolChoice {
    Mode(String),
    #[allow(dead_code)]
    Specific(serde_json::Value),
}

/// OpenAI's `stop` field accepts either a single string or an array of
/// strings.
#[derive(Deserialize)]
#[serde(untagged)]
enum StopParam {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    tools: Vec<ToolDef>,
    #[serde(default)]
    tool_choice: Option<ToolChoice>,
    /// Server-side conversation history key (see the `session`
    /// module): when set, `messages` is treated as
    /// *only the new turn(s)* to append to this session's stored
    /// history, not the whole conversation.
    #[serde(default)]
    session_id: Option<String>,
    /// OpenAI fields we explicitly reject rather than silently ignore.
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<u32>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
}

fn default_max_tokens() -> usize {
    16
}

impl ChatCompletionRequest {
    fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature.unwrap_or(0.0),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k.unwrap_or(0),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
        }
    }

    fn stop_sequences(&self) -> Vec<String> {
        self.stop
            .as_ref()
            .map(|s| match s {
                StopParam::One(v) => vec![v.clone()],
                StopParam::Many(v) => v.clone(),
            })
            .unwrap_or_default()
    }

    /// Real tool-calling is only offered when `tools` is non-empty AND
    /// the client hasn't explicitly disabled it via `tool_choice:
    /// "none"` -- see `ToolChoice`'s doc comment for what the other
    /// values do (nothing different from `"auto"`).
    fn tools_active(&self) -> bool {
        !self.tools.is_empty()
            && !matches!(&self.tool_choice, Some(ToolChoice::Mode(m)) if m == "none")
    }

    /// Reject OpenAI fields we do not implement, and `tool_choice`
    /// values that would silently lie (required / named function).
    fn validate_supported_fields(&self) -> Result<(), ApiError> {
        for msg in &self.messages {
            if msg.content.as_ref().is_some_and(MessageContent::has_image) {
                return Err(unsupported_feature(
                    "image_url content parts are not implemented (multimodal/VL deferred — see docs/API.md)",
                ));
            }
        }
        if self.logprobs == Some(true) || self.top_logprobs.is_some() {
            return Err(unsupported_feature(
                "logprobs / top_logprobs are not implemented yet (see docs/API.md)",
            ));
        }
        if self.n.is_some_and(|n| n > 1) {
            return Err(unsupported_feature(
                "n > 1 is not implemented (single completion only)",
            ));
        }
        if let Some(fmt) = &self.response_format {
            match fmt.get("type").and_then(|v| v.as_str()) {
                Some("json_object") => {}
                Some(other) => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": {
                                "message": format!(
                                    "response_format type {other:?} is not supported (only json_object)"
                                )
                            }
                        })),
                    ));
                }
                None => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": {
                                "message": "response_format must include \"type\" (only json_object is supported)"
                            }
                        })),
                    ));
                }
            }
        }
        match &self.tool_choice {
            Some(ToolChoice::Mode(m)) if m == "required" => {
                return Err(unsupported_feature(
                    "tool_choice=required needs constrained decoding (not implemented)",
                ));
            }
            Some(ToolChoice::Specific(_)) => {
                return Err(unsupported_feature(
                    "named tool_choice is not implemented (use auto/none)",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    /// `stop_sequences()` plus `</tool_call>` when tool-calling is
    /// active -- reusing the existing stop-sequence machinery
    /// (`generate::generate`'s `earliest_stop_match`) to end generation
    /// right after a tool call's JSON body, rather than adding any new
    /// decode-time logic. See `tool_preamble`'s doc comment for the
    /// full real, disclosed approach.
    fn effective_stop_sequences(&self) -> Vec<String> {
        let mut stop = self.stop_sequences();
        if self.tools_active() {
            stop.push("</tool_call>".to_string());
        }
        stop
    }

    fn json_object_mode(&self) -> bool {
        self.response_format
            .as_ref()
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str())
            == Some("json_object")
    }

    fn generation_params(&self) -> GenerationParams {
        GenerationParams {
            max_tokens: self.max_tokens,
            sampling: self.sampling_params(),
            seed: self.resolved_seed(),
            stop: self.effective_stop_sequences(),
            // Resolved by `run_generation_emit`, the layer that holds a
            // tokenizer: a request body names stop strings, and only
            // the model can say which of them are single tokens.
            stop_token_ids: Vec::new(),
            json_object: self.json_object_mode(),
            // Filled in by the handler that owns the request id --
            // the request body cannot name its own cancel token.
            cancel: None,
        }
    }

    /// Like [`Self::generation_params`], plus architecture-default stop
    /// strings (Gemma IT emits `<end_of_turn>` before `<eos>`).
    fn generation_params_for_template(
        &self,
        template: chat_template::ChatTemplate,
    ) -> GenerationParams {
        let mut params = self.generation_params();
        if matches!(
            template,
            chat_template::ChatTemplate::Gemma | chat_template::ChatTemplate::Gemma4
        ) {
            let stop = match template {
                chat_template::ChatTemplate::Gemma => "<end_of_turn>",
                chat_template::ChatTemplate::Gemma4 => "<turn|>",
                _ => unreachable!(),
            };
            if !params.stop.iter().any(|s| s == stop) {
                params.stop.push(stop.to_string());
            }
        }
        params
    }

    /// A request only has a deterministic outcome -- and therefore is
    /// only safe to serve from or populate into the whole-response
    /// cache -- when it's plain greedy decode (temperature <= 0) or an
    /// explicit seed was given. Anything else must always regenerate:
    /// a "cache hit" for an unseeded sampled request would silently
    /// replay one random draw forever, defeating the purpose of
    /// sampling and surprising any client expecting fresh output per
    /// call.
    fn is_cacheable(&self) -> bool {
        self.temperature.unwrap_or(0.0) <= 0.0 || self.seed.is_some()
    }

    fn cache_key(&self, prompt: &str) -> CacheKey {
        CacheKey {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            max_tokens: self.max_tokens,
            temperature_bits: self.temperature.unwrap_or(0.0).to_bits(),
            top_p_bits: self.top_p.unwrap_or(1.0).to_bits(),
            top_k: self.top_k.unwrap_or(0),
            repetition_penalty_bits: self.repetition_penalty.unwrap_or(1.0).to_bits(),
            presence_penalty_bits: self.presence_penalty.unwrap_or(0.0).to_bits(),
            frequency_penalty_bits: self.frequency_penalty.unwrap_or(0.0).to_bits(),
            seed: self.seed,
            stop: self.effective_stop_sequences(),
        }
    }

    fn resolved_seed(&self) -> u64 {
        self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xDEFA017)
        })
    }
}

#[derive(Serialize)]
struct ChatCompletionChoice {
    index: usize,
    message: ChatCompletionResponseMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionResponseMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallOut>>,
}

#[derive(Serialize, Clone)]
struct ToolCallOut {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: ToolCallFunctionOut,
}

#[derive(Serialize, Clone)]
struct ToolCallFunctionOut {
    name: String,
    /// A JSON-encoded string, matching the real OpenAI
    /// `tool_calls[].function.arguments` convention (see
    /// `ToolCallFunctionIn::arguments`'s doc comment).
    arguments: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    /// Non-standard extension: the same value as `id`, stated under the
    /// name the rest of ferrox keys by (metrics, logs, `POST /cancel`
    /// once it exists). `id` is OpenAI's completion id and a client has
    /// no way to know ferrox also uses it as the request key -- saying
    /// so costs one field and removes the guess.
    request_id: String,
    object: &'static str,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    /// OpenAI-convention token accounting (prompt/completion/total),
    /// counted from the exact ids the generation loop processed. On a
    /// whole-response cache hit, this is the original computation's
    /// accounting (same prompt, same deterministic outcome).
    usage: generate::Usage,
    /// Non-standard extension field (not part of the OpenAI API
    /// contract, but additive and harmless to OpenAI-compatible
    /// clients that ignore unknown fields): "hit" if this exact
    /// cacheable request was already computed, "miss" if this request
    /// just computed and cached a fresh completion, or "skip" if the
    /// request wasn't cacheable at all (sampling without a seed --
    /// see `ChatCompletionRequest::is_cacheable`).
    ferrox_cache: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallOut>>,
}

#[derive(Serialize)]
struct ChatCompletionChunkChoice {
    index: usize,
    delta: ChatCompletionChunkDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    /// Present on the **first** chunk of a stream (see
    /// `ChatCompletionResponse::request_id`). A client learns the key
    /// for this generation before any content arrives, so a live view
    /// can correlate metrics with the stream it is rendering instead of
    /// guessing which in-flight request is "probably mine" -- a guess
    /// that mis-attributes the moment two chats run at once.
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    object: &'static str,
    model: String,
    choices: Vec<ChatCompletionChunkChoice>,
    /// Present only on the final chunk (the one carrying
    /// `finish_reason`), mirroring OpenAI's stream `usage` shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<generate::Usage>,
}

/// Liveness, readiness and capabilities in one cheap answer (see the
/// `health` module for why detection is a visible state rather than a
/// gap). Never behind auth or rate limiting, and never blocking: this is
/// the endpoint a supervisor asks when it is deciding whether to kill
/// the process.
async fn health(State(state): State<Arc<AppState>>) -> Response {
    let snapshot = state.detection.snapshot();
    let mut capabilities = snapshot.capabilities;
    let active = state.active();

    // Model-derived capabilities need no probing, so they are answered
    // even while backend detection is still running.
    capabilities.push(match active.as_deref() {
        // `unavailable` was defined in Phase 1 but unreachable, because
        // the server only bound the port after a successful load. With
        // `/admin/models/unload` it is a state a client can actually
        // observe, and it must not read as "loaded but synthetic".
        None => ferrox_api::Capability::unavailable(
            ferrox_api::health::capability::REAL_WEIGHTS,
            ferrox_api::health::reason::MODEL_NOT_LOADED,
            "No model is loaded. POST /admin/models/load with an id from GET /admin/models.",
        ),
        Some(active) if active.model.is_synthetic() => ferrox_api::Capability::unavailable(
            ferrox_api::health::capability::REAL_WEIGHTS,
            ferrox_api::health::reason::MODEL_NOT_LOADED,
            "Serving synthetic random weights: set FERROX_MODEL_PATH (or -m) to a real \
             checkpoint. Output from this model is noise.",
        ),
        Some(active) => ferrox_api::Capability::available(
            ferrox_api::health::capability::REAL_WEIGHTS,
            format!("Serving the real checkpoint '{}'.", active.model.name()),
        ),
    });
    capabilities.push(if active.as_ref().is_some_and(|a| a.batcher.is_some()) {
        ferrox_api::Capability::available(
            ferrox_api::health::capability::CONTINUOUS_BATCHING,
            "Concurrent requests share one batched decode step.",
        )
    } else {
        ferrox_api::Capability::unavailable(
            ferrox_api::health::capability::CONTINUOUS_BATCHING,
            ferrox_api::health::reason::DISABLED,
            "Off; set FERROX_CONTINUOUS_BATCHING=1 (incompatible with a KV pool or prefix cache).",
        )
    });

    let last_request_ms = state
        .last_request_ms
        .load(std::sync::atomic::Ordering::Relaxed);
    let uptime = state.started_at.elapsed();
    // Readiness is "can this server generate", and with nothing loaded
    // it cannot -- so `unavailable` (503) wins over whatever the backend
    // probe concluded. Phase 1 defined this state but nothing could
    // reach it, because the process only bound the port after a
    // successful load; `/admin/models/unload` makes it reachable, and a
    // 200 `ready` here would tell a supervisor to send traffic that is
    // guaranteed to 503.
    let health_state = if active.is_none() {
        ferrox_api::HealthState::Unavailable
    } else {
        snapshot.state
    };
    let body = ferrox_api::HealthResponse {
        state: health_state,
        reason: match health_state {
            ferrox_api::HealthState::Ready => None,
            ferrox_api::HealthState::Unavailable => {
                Some(ferrox_api::health::reason::MODEL_NOT_LOADED.to_string())
            }
            ferrox_api::HealthState::Detecting => {
                Some(ferrox_api::health::reason::DETECTING.to_string())
            }
        },
        detail: match health_state {
            ferrox_api::HealthState::Ready => None,
            ferrox_api::HealthState::Unavailable => Some(
                "No model is loaded. POST /admin/models/load with an id from GET /admin/models."
                    .to_string(),
            ),
            ferrox_api::HealthState::Detecting => {
                Some("Probing available compute backends.".to_string())
            }
        },
        model: active
            .as_deref()
            .map(|active| ferrox_api::health::ModelSummary {
                id: active.model.name().to_string(),
                tokenizer: active.model.tokenizer_kind().to_string(),
                synthetic_weights: active.model.is_synthetic(),
            }),
        capabilities,
        version: env!("CARGO_PKG_VERSION").to_string(),
        pid: std::process::id(),
        uptime_seconds: uptime.as_secs_f64(),
        server_time_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis().min(u64::MAX as u128) as u64)
            .unwrap_or(0),
        last_request_age_seconds: (last_request_ms > 0)
            .then(|| uptime.as_secs_f64() - (last_request_ms as f64 / 1000.0))
            .map(|age| age.max(0.0)),
    };

    let status =
        StatusCode::from_u16(body.state.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(body)).into_response()
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // OpenAI's `/v1/models` lists what can be *used* right now, which
    // after an unload is nothing. The inventory of what is on disk is a
    // different question and lives at `/admin/models`.
    let Some(active) = state.active() else {
        return Json(serde_json::json!({ "object": "list", "data": [] }));
    };
    let mut model_entry = serde_json::json!({
        "id": active.model.name(),
        "object": "model",
        "ferrox_synthetic_weights": active.model.is_synthetic(),
        "ferrox_tokenizer": active.model.tokenizer_kind(),
    });
    if let Some(mcp) = &state.mcp {
        model_entry["ferrox_mcp"] = mcp.models_metadata();
    }
    Json(serde_json::json!({
        "object": "list",
        "data": [model_entry]
    }))
}

#[derive(Serialize)]
struct CombinedCacheStats {
    response_cache: cache::CacheStats,
    /// `None` when `FERROX_PREFIX_CACHE_ENTRIES` isn't set.
    prefix_cache: Option<ferrox_models::PrefixCacheStats>,
}

async fn cache_stats(State(state): State<Arc<AppState>>) -> Json<CombinedCacheStats> {
    Json(CombinedCacheStats {
        response_cache: lock_cache(&state.response_cache).stats(),
        prefix_cache: state
            .prefix_cache
            .as_ref()
            .map(|pc| pc.lock().unwrap_or_else(|p| p.into_inner()).stats()),
    })
}

/// Prometheus text-exposition format (`# HELP`/`# TYPE` plus
/// `name value` lines), so this endpoint can be scraped directly by a
/// Prometheus server or anything compatible with that format without
/// ferrox needing to speak any particular metrics client library.
async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    use std::sync::atomic::Ordering;

    let cache_stats = lock_cache(&state.response_cache).stats();
    let active = state.active();
    let requests_total = state.requests_total.load(Ordering::Relaxed);
    let errors_total = state.request_errors_total.load(Ordering::Relaxed);
    let uptime = state.started_at.elapsed().as_secs_f64();

    let body = format!(
        "# HELP ferrox_requests_total Total chat completion requests received.\n\
         # TYPE ferrox_requests_total counter\n\
         ferrox_requests_total {requests_total}\n\
         # HELP ferrox_request_errors_total Total chat completion requests that returned an error.\n\
         # TYPE ferrox_request_errors_total counter\n\
         ferrox_request_errors_total {errors_total}\n\
         # HELP ferrox_cache_hits_total Whole-response cache hits.\n\
         # TYPE ferrox_cache_hits_total counter\n\
         ferrox_cache_hits_total {}\n\
         # HELP ferrox_cache_misses_total Whole-response cache misses.\n\
         # TYPE ferrox_cache_misses_total counter\n\
         ferrox_cache_misses_total {}\n\
         # HELP ferrox_cache_entries Current whole-response cache entry count.\n\
         # TYPE ferrox_cache_entries gauge\n\
         ferrox_cache_entries {}\n\
         # HELP ferrox_synthetic_weights 1 if serving synthetic random weights instead of a real checkpoint.\n\
         # TYPE ferrox_synthetic_weights gauge\n\
         ferrox_synthetic_weights {}\n\
         # HELP ferrox_uptime_seconds Seconds since this server process started.\n\
         # TYPE ferrox_uptime_seconds gauge\n\
         ferrox_uptime_seconds {uptime}\n",
        cache_stats.hits,
        cache_stats.misses,
        cache_stats.entries,
        // With nothing loaded there are no weights at all, synthetic or
        // otherwise; 0 is the reading that keeps the gauge meaning
        // "serving noise" rather than "serving nothing".
        active
            .as_ref()
            .map(|a| a.model.is_synthetic() as u8)
            .unwrap_or(0),
    );

    // Expert-store counters, present only when the model streams
    // routed experts through the bounded cache
    // (FERROX_EXPERT_CACHE_BYTES).
    let body = match active
        .as_ref()
        .and_then(|a| a.model.expert_store_stats())
    {
        Some(es) => format!(
            "{body}\
             # HELP ferrox_expert_cache_hits_total Expert-store cache hits.\n\
             # TYPE ferrox_expert_cache_hits_total counter\n\
             ferrox_expert_cache_hits_total {}\n\
             # HELP ferrox_expert_cache_misses_total Expert-store cache misses (source reads).\n\
             # TYPE ferrox_expert_cache_misses_total counter\n\
             ferrox_expert_cache_misses_total {}\n\
             # HELP ferrox_expert_cache_evictions_total Expert-store LRU evictions.\n\
             # TYPE ferrox_expert_cache_evictions_total counter\n\
             ferrox_expert_cache_evictions_total {}\n\
             # HELP ferrox_expert_cache_pass_throughs_total Acquires served uncached (entry could not fit the budget).\n\
             # TYPE ferrox_expert_cache_pass_throughs_total counter\n\
             ferrox_expert_cache_pass_throughs_total {}\n\
             # HELP ferrox_expert_cache_bytes_read_total Bytes read from the checkpoint for expert misses.\n\
             # TYPE ferrox_expert_cache_bytes_read_total counter\n\
             ferrox_expert_cache_bytes_read_total {}\n\
             # HELP ferrox_expert_cache_resident_bytes Current expert-cache footprint in bytes.\n\
             # TYPE ferrox_expert_cache_resident_bytes gauge\n\
             ferrox_expert_cache_resident_bytes {}\n",
            es.hits, es.misses, es.evictions, es.pass_throughs, es.bytes_read, es.resident_bytes,
        ),
        None => body,
    };

    // Scheduler counters, present only under continuous batching
    // (FERROX_CONTINUOUS_BATCHING=1). `prefill_chunks` next to
    // `prefill_tokens` is what makes chunked prefill observable: their
    // ratio is the effective chunk size the worker actually ran.
    let body = match active.as_ref().and_then(|a| a.batcher.as_ref()) {
        Some(batcher) => {
            let sched = batcher.stats();
            format!(
                "{body}\
                 # HELP ferrox_prefill_chunks_total Bounded prefill chunks the batch scheduler has run.\n\
                 # TYPE ferrox_prefill_chunks_total counter\n\
                 ferrox_prefill_chunks_total {}\n\
                 # HELP ferrox_prefill_tokens_total Prompt tokens run through chunked prefill.\n\
                 # TYPE ferrox_prefill_tokens_total counter\n\
                 ferrox_prefill_tokens_total {}\n\
                 # HELP ferrox_decode_steps_total Batched decode steps the batch scheduler has run.\n\
                 # TYPE ferrox_decode_steps_total counter\n\
                 ferrox_decode_steps_total {}\n\
                 # HELP ferrox_scheduler_queue_depth Requests waiting for admission to the batch scheduler.\n\
                 # TYPE ferrox_scheduler_queue_depth gauge\n\
                 ferrox_scheduler_queue_depth {}\n\
                 # HELP ferrox_scheduler_queue_rejected_total Requests refused with 503 because the admission queue was full.\n\
                 # TYPE ferrox_scheduler_queue_rejected_total counter\n\
                 ferrox_scheduler_queue_rejected_total {}\n\
                 # HELP ferrox_kv_blocks_total KV blocks in the scheduler's admission budget (0 when unconfigured).\n\
                 # TYPE ferrox_kv_blocks_total gauge\n\
                 ferrox_kv_blocks_total {}\n\
                 # HELP ferrox_kv_blocks_free KV blocks not reserved by an in-flight request.\n\
                 # TYPE ferrox_kv_blocks_free gauge\n\
                 ferrox_kv_blocks_free {}\n\
                 # HELP ferrox_kv_block_size Token positions per KV block.\n\
                 # TYPE ferrox_kv_block_size gauge\n\
                 ferrox_kv_block_size {}\n\
                 # HELP ferrox_kv_rejected_too_large_total Requests refused with 400 because they exceed the whole KV block budget.\n\
                 # TYPE ferrox_kv_rejected_too_large_total counter\n\
                 ferrox_kv_rejected_too_large_total {}\n\
                 # HELP ferrox_kv_rejected_context_length_total Requests refused with 400 for exceeding the per-request context ceiling.\n\
                 # TYPE ferrox_kv_rejected_context_length_total counter\n\
                 ferrox_kv_rejected_context_length_total {}\n\
                 # HELP ferrox_scheduler_aborted_total Requests the batch scheduler stopped because they were cancelled.\n\
                 # TYPE ferrox_scheduler_aborted_total counter\n\
                 ferrox_scheduler_aborted_total {}\n",
                sched.prefill_chunks,
                sched.prefill_tokens,
                sched.decode_steps,
                sched.queue_depth,
                sched.queue_rejected,
                sched.kv_blocks_total,
                sched.kv_blocks_free,
                sched.kv_block_size,
                sched.kv_rejected_too_large,
                sched.kv_rejected_context_length,
                sched.aborted,
            )
        }
        None => body,
    };

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    )
        .into_response()
}

pub(crate) type ApiError = (StatusCode, Json<serde_json::Value>);

pub(crate) fn unsupported_feature(message: &str) -> ApiError {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({"error": {"message": message, "type": "unsupported"}})),
    )
}

pub(crate) fn decode_error_response(e: generate::DecodeError) -> ApiError {
    let status = match e {
        generate::DecodeError::TokenOutOfVocab { .. } => StatusCode::BAD_REQUEST,
        // The request is bigger than the server can ever serve. That
        // is a property of the request, so it is the client's 400 --
        // answering 503 would send it into a retry loop that cannot
        // succeed.
        generate::DecodeError::KvBudgetExceeded { .. } => StatusCode::BAD_REQUEST,
        // Not the client's fault, and true of the exact same request a
        // moment later once capacity frees up -- 503, not 400. The
        // `Retry-After` header these need is stamped centrally by
        // `limits::retry_after`; see that function for why it lives in a
        // layer rather than here.
        generate::DecodeError::KvPoolExhausted | generate::DecodeError::QueueFull { .. } => {
            StatusCode::SERVICE_UNAVAILABLE
        }
    };
    tracing::warn!("decode error: {e}");
    let mut body = serde_json::json!({"error": {"message": e.to_string()}});
    // A refusal against a ceiling names the ceiling and both sides of
    // the arithmetic. "Out of memory" (or a bare 400) tells a caller
    // that something did not fit; it does not tell them whether to
    // shorten the prompt or to run a bigger box, and those are the only
    // two actions available.
    if let generate::DecodeError::KvBudgetExceeded {
        binding,
        estimated_bytes,
        limit_bytes,
        positions,
        positions_limit,
        ..
    } = &e
    {
        body["error"]["type"] = serde_json::json!("invalid_request_error");
        body["error"]["code"] = serde_json::json!(binding);
        body["error"]["binding"] = serde_json::json!(binding);
        body["error"]["estimated_bytes"] = serde_json::json!(estimated_bytes);
        body["error"]["limit_bytes"] = serde_json::json!(limit_bytes);
        body["error"]["positions"] = serde_json::json!(positions);
        body["error"]["positions_limit"] = serde_json::json!(positions_limit);
    }
    // The header carries the same hint (stamped by `limits::retry_after`);
    // repeating it in the body is for clients that read JSON and never
    // look at headers, which is most of them.
    if let Some(secs) = e.retry_after_secs() {
        body["error"]["retry_after_seconds"] = serde_json::json!(secs);
    }
    (status, Json(body))
}

pub(crate) fn join_error_response(e: tokio::task::JoinError) -> ApiError {
    tracing::error!("generation task panicked: {e}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": {"message": "internal error during generation"}})),
    )
}

/// Runs generation for `params` against `model`, calling `emit` for each
/// decoded text chunk. Returns finish reason, usage, and the concatenated
/// text (for sessions / tool-call detection). Pure CPU-bound work with
/// no I/O and no shared lock: safe to run on `spawn_blocking`.
fn run_generation_emit(
    model: &Model,
    prompt: &str,
    params: &GenerationParams,
    kv_pool: Option<&generate::KvPoolConfig>,
    prefix_cache: Option<&Mutex<PrefixCache>>,
    continuous_batcher: Option<&batch_scheduler::ContinuousBatcher>,
    mut emit: impl FnMut(&str),
) -> Result<(FinishReason, generate::Usage, String), generate::DecodeError> {
    let synthetic = model.is_synthetic();
    let mut chunks = Vec::new();
    // Layer 1 of the stop machinery is resolved exactly here, because
    // this is the one place that has both the request's stop strings
    // and the model's tokenizer. Both the batched and the private
    // decode paths below read the result off the params, so there is
    // one answer rather than two that can drift.
    let params = &{
        let mut resolved = params.clone();
        resolved.stop_token_ids =
            crate::stop::resolve_stop_tokens(&resolved.stop, |text| model.encode(text));
        resolved
    };
    let used_batcher = matches!((model, continuous_batcher), (Model::Gguf(_), Some(_)));
    let (finish, usage) = match model {
        Model::Gguf(m) => {
            if let Some(batcher) = continuous_batcher {
                let mut tokens = m.tokenizer.encode(prompt);
                ferrox_models::tokenizer::prepend_bos(&mut tokens, m.bos_id);
                let (finish, _generated_ids, text, usage) =
                    batcher.generate(tokens, params.clone(), m.stop_tokens.clone())?;
                if !text.is_empty() {
                    chunks.push(text);
                }
                (finish, usage)
            } else {
                generate::generate(
                    &m.decoder,
                    m.tokenizer.as_ref(),
                    &m.stop_tokens,
                    m.bos_id,
                    prompt,
                    params,
                    kv_pool,
                    prefix_cache,
                    |chunk| {
                        chunks.push(chunk.to_string());
                        if !synthetic {
                            emit(chunk);
                        }
                    },
                )?
            }
        }
        Model::Kimi(m) => generate::generate_engine(
            &m.engine,
            &m.tokenizer,
            &m.stop_tokens,
            None,
            prompt,
            params,
            |chunk| {
                chunks.push(chunk.to_string());
                if !synthetic {
                    emit(chunk);
                }
            },
        )?,
        Model::Mla(m) => generate::generate_engine(
            &m.engine,
            &m.tokenizer,
            &m.stop_tokens,
            m.bos_id,
            prompt,
            params,
            |chunk| {
                chunks.push(chunk.to_string());
                if !synthetic {
                    emit(chunk);
                }
            },
        )?,
        Model::Gemma4(m) => generate::generate_engine(
            &m.engine,
            &m.tokenizer,
            &m.stop_tokens,
            m.bos_id,
            prompt,
            params,
            |chunk| {
                chunks.push(chunk.to_string());
                if !synthetic {
                    emit(chunk);
                }
            },
        )?,
        Model::Glm52(m) => generate::generate_engine(
            &m.engine,
            &m.tokenizer,
            &m.stop_tokens,
            m.bos_id,
            prompt,
            params,
            |chunk| {
                chunks.push(chunk.to_string());
                if !synthetic {
                    emit(chunk);
                }
            },
        )?,
    };

    let mut full = chunks.concat();
    if synthetic {
        full = format!(
            "[ferrox synthetic-weight demo: no real checkpoint loaded -- set FERROX_MODEL_PATH \
             to serve a real model. Decoded ids -> {full:?}]"
        );
        emit(&full);
    } else if used_batcher && !full.is_empty() {
        emit(&full);
    }

    Ok((finish, usage, full))
}

/// Collecting wrapper around [`run_generation_emit`] for non-streaming
/// paths and tests.
pub(crate) fn run_generation(
    model: &Model,
    prompt: &str,
    params: &GenerationParams,
    kv_pool: Option<&generate::KvPoolConfig>,
    prefix_cache: Option<&Mutex<PrefixCache>>,
    continuous_batcher: Option<&batch_scheduler::ContinuousBatcher>,
) -> Result<(Vec<String>, FinishReason, generate::Usage), generate::DecodeError> {
    let (finish, usage, full) = run_generation_emit(
        model,
        prompt,
        params,
        kv_pool,
        prefix_cache,
        continuous_batcher,
        |_| {},
    )?;
    Ok((
        if full.is_empty() {
            Vec::new()
        } else {
            vec![full]
        },
        finish,
        usage,
    ))
}

pub(crate) fn prompt_from_messages(
    messages: &[ChatMessage],
    template: chat_template::ChatTemplate,
    tools: &[ToolDef],
) -> String {
    if tools.is_empty() {
        template.render(messages)
    } else {
        let mut with_preamble = Vec::with_capacity(messages.len() + 1);
        with_preamble.push(ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text(tool_preamble(tools))),
            tool_calls: None,
            tool_call_id: None,
        });
        with_preamble.extend_from_slice(messages);
        template.render(&with_preamble)
    }
}

/// Real, disclosed approach for tool-calling without grammar-
/// constrained decoding (which doesn't exist in this server):
/// describe each tool in plain text and ask the
/// model to wrap a call in a literal `<tool_call>{...}</tool_call>`
/// marker, then reuse the existing stop-sequence machinery (see
/// `ChatCompletionRequest::effective_stop_sequences`) to end
/// generation right after it, and parse the captured text for that
/// marker afterward (`extract_tool_call`). This is stop-bounded,
/// prompt-engineered JSON extraction, not enforced-valid-JSON output --
/// a real limitation, not overclaimed.
fn tool_preamble(tools: &[ToolDef]) -> String {
    let mut out = String::from(
        "You can call tools to help answer the user. To call a tool, respond with \
         EXACTLY one line in this format and nothing else:\n\
         <tool_call>{\"name\": \"<tool name>\", \"arguments\": {<arguments as a JSON \
         object matching that tool's parameters>}}</tool_call>\n\n\
         Available tools:\n",
    );
    for t in tools {
        out.push_str(&format!(
            "- {}: {}\n  parameters (JSON schema): {}\n",
            t.function.name,
            t.function.description.as_deref().unwrap_or(""),
            t.function
                .parameters
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "{}".to_string()),
        ));
    }
    out
}

/// Looks for a `<tool_call>{...}</tool_call>` marker (see
/// `tool_preamble`) and parses its JSON body into a `(name,
/// arguments)` pair -- `arguments` kept as a JSON-encoded *string*,
/// matching OpenAI's real `tool_calls[].function.arguments`
/// convention (a string, even though the model itself writes it as
/// literal JSON).
fn extract_tool_call(text: &str) -> Option<(String, String)> {
    const START: &str = "<tool_call>";
    const END: &str = "</tool_call>";
    let start = text.find(START)? + START.len();
    let end = start + text[start..].find(END)?;
    let body = text[start..end].trim();
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    Some((name, arguments.to_string()))
}

/// Builds the final response message + finish reason from raw
/// generated text: promotes `base_finish` to `"tool_calls"` (moving
/// the text into a structured `tool_calls` entry instead of `content`)
/// when tool-calling was active and the text actually contains a real
/// `<tool_call>{...}</tool_call>` marker -- a model can still just
/// answer in plain text despite tools being offered, which must fall
/// through to an ordinary text response, not an error.
fn build_response_message(
    content: String,
    tools_active: bool,
    base_finish: &'static str,
) -> (ChatCompletionResponseMessage, &'static str) {
    if tools_active {
        if let Some((name, arguments)) = extract_tool_call(&content) {
            return (
                ChatCompletionResponseMessage {
                    role: "assistant",
                    content: None,
                    tool_calls: Some(vec![ToolCallOut {
                        id: "call_0".to_string(),
                        kind: "function",
                        function: ToolCallFunctionOut { name, arguments },
                    }]),
                },
                "tool_calls",
            );
        }
    }
    (
        ChatCompletionResponseMessage {
            role: "assistant",
            content: Some(content),
            tool_calls: None,
        },
        base_finish,
    )
}

/// Resolves the full message history a prompt should be rendered
/// from: `req.messages` verbatim when no session is in play, or (see
/// `session` module) `req.messages` appended to `session_id`'s stored
/// history, returning the accumulated whole.
fn resolve_history(state: &AppState, req: &ChatCompletionRequest) -> Vec<ChatMessage> {
    let mut history = match &req.session_id {
        Some(id) => state.sessions.extend_and_get(id, &req.messages),
        None => req.messages.clone(),
    };
    if req.json_object_mode() {
        inject_json_object_system_hint(&mut history);
    }
    history
}

fn inject_json_object_system_hint(messages: &mut Vec<ChatMessage>) {
    const HINT: &str =
        "You must respond with valid JSON only (a single JSON object, no markdown fences).";
    if let Some(sys) = messages.iter_mut().find(|m| m.role == "system") {
        match &mut sys.content {
            Some(MessageContent::Text(s)) if !s.contains("JSON") => {
                s.push_str("\n\n");
                s.push_str(HINT);
            }
            None => {
                sys.content = Some(MessageContent::Text(HINT.to_string()));
            }
            _ => {}
        }
    } else {
        messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: Some(MessageContent::Text(HINT.to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    state
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let started = std::time::Instant::now();

    // One id per request, assigned before any work starts -- including
    // before validation -- so the streaming and non-streaming paths
    // agree and a rejected request is still nameable in the monitor.
    let request_id = ferrox_api::next_request_id();
    let stream = req.stream.unwrap_or(false);

    if let Err(err) = req.validate_supported_fields() {
        state
            .request_errors_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let response = err.into_response();
        state.record_request(
            &request_id,
            ferrox_api::routes::V1_CHAT_COMPLETIONS,
            response.status().as_u16(),
            stream,
            started.elapsed().as_millis() as u64,
            None,
        );
        return response;
    }

    let response = if stream {
        chat_completions_stream(Arc::clone(&state), req, request_id.clone(), started)
            .await
            .into_response()
    } else {
        chat_completions_full(Arc::clone(&state), req, request_id.clone(), started)
            .await
            .into_response()
    };

    if response.status().is_client_error() || response.status().is_server_error() {
        state
            .request_errors_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Only failures are recorded here. A success has already
        // recorded itself from the path that knows the token counts --
        // and, for a stream, that has not even happened yet.
        state.record_request(
            &request_id,
            ferrox_api::routes::V1_CHAT_COMPLETIONS,
            response.status().as_u16(),
            stream,
            started.elapsed().as_millis() as u64,
            None,
        );
    }
    state.mark_request_finished();

    response
}

async fn chat_completions_full(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    request_id: String,
    started: std::time::Instant,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let tools_active = req.tools_active();
    // Cloned once, up front: this request decodes against exactly this
    // model even if `/admin/models/load` swaps a different one in
    // halfway through (see `AppState::active`).
    let active = state.require_active()?;
    let history = resolve_history(&state, &req);
    let prompt = prompt_from_messages(&history, active.model.chat_template(), &req.tools);
    let key = req.is_cacheable().then(|| req.cache_key(&prompt));

    let (completion, cache_status) = if let Some(cached) = key
        .as_ref()
        .and_then(|key| lock_cache(&state.response_cache).get(key))
    {
        tracing::debug!("cache hit for key {}", key.as_ref().unwrap().digest());
        (cached, "hit")
    } else {
        let model = Arc::clone(&active.model);
        let kv_pool = state.kv_pool.clone();
        let prefix_cache = state.prefix_cache.clone();
        let batcher = active.batcher.clone();
        let params = req.generation_params_for_template(active.model.chat_template());
        let prompt_for_task = prompt.clone();
        let (chunks, finish, usage) = tokio::task::spawn_blocking(move || {
            run_generation(
                &model,
                &prompt_for_task,
                &params,
                kv_pool.as_ref(),
                prefix_cache.as_deref(),
                batcher.as_ref(),
            )
        })
        .await
        .map_err(join_error_response)?
        .map_err(decode_error_response)?;

        let completion = cache::CachedCompletion {
            content: chunks.concat(),
            finish,
            usage,
        };
        let cache_status = if let Some(key) = key {
            tracing::debug!("cache miss for key {}", key.digest());
            lock_cache(&state.response_cache).put(key, completion.clone());
            "miss"
        } else {
            "skip"
        };
        (completion, cache_status)
    };
    let content = completion.content;

    if req.json_object_mode() {
        json_mode::validate_json_object_output(&content)?;
    }

    // Stored regardless of cache hit/miss, so a session's history is
    // always consistent with what a client would see, whether or not
    // this exact prompt happened to be served from cache.
    if let Some(id) = &req.session_id {
        state.sessions.store_reply(
            id,
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(MessageContent::Text(content.clone())),
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }

    let (message, finish_reason) =
        build_response_message(content, tools_active, completion.finish.as_str());

    state.record_request(
        &request_id,
        ferrox_api::routes::V1_CHAT_COMPLETIONS,
        200,
        false,
        started.elapsed().as_millis() as u64,
        Some(&completion.usage),
    );

    Ok(Json(ChatCompletionResponse {
        id: request_id.clone(),
        request_id,
        object: "chat.completion",
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message,
            finish_reason,
        }],
        usage: completion.usage,
        ferrox_cache: cache_status,
    }))
}

async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    request_id: String,
    started: std::time::Instant,
) -> Result<Response, ApiError> {
    // Streaming requests are never served from or written to the response cache.
    let tools_active = req.tools_active();
    // See `chat_completions_full`: the handle is taken once and the
    // whole stream runs against it, so a mid-stream model swap cannot
    // splice two checkpoints into one completion.
    let active = state.require_active()?;
    let history = resolve_history(&state, &req);
    let prompt = prompt_from_messages(&history, active.model.chat_template(), &req.tools);
    let model_name = req.model.clone();
    let session_id = req.session_id.clone();
    let sessions = state.sessions.clone();

    let model = Arc::clone(&active.model);
    let kv_pool = state.kv_pool.clone();
    let prefix_cache = state.prefix_cache.clone();
    let batcher = active.batcher.clone();
    let mut params = req.generation_params_for_template(active.model.chat_template());
    let stats_state = Arc::clone(&state);

    // Tier two of cancellation: the id is already on the wire, so the
    // client can name it. The guard rides with the generation task and
    // deregisters however that task ends, panic included -- see the
    // `cancel` module.
    let (cancel_token, cancel_guard) = state.cancels.register(&request_id);
    params.cancel = Some(cancel_token.clone());

    // Tool-call detection needs the full stop-bounded text; continuous
    // batching returns one string. Both stay buffered. Otherwise each
    // decoded chunk is pushed on a channel for overlapped SSE delivery.
    let overlap = !tools_active && batcher.is_none();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::task::spawn_blocking(move || {
        // Held for the whole generation; dropping it is what takes the
        // id back out of the cancel registry.
        let _cancel_guard = cancel_guard;
        let tx_chunks = tx.clone();
        let mut first = true;
        let head_request_id = request_id.clone();
        let result = run_generation_emit(
            &model,
            &prompt,
            &params,
            kv_pool.as_ref(),
            prefix_cache.as_deref(),
            batcher.as_ref(),
            |chunk| {
                if !overlap || chunk.is_empty() {
                    return;
                }
                let role = if first { Some("assistant") } else { None };
                let request_id = first.then(|| head_request_id.clone());
                first = false;
                let payload = ChatCompletionChunk {
                    id: head_request_id.clone(),
                    request_id,
                    object: "chat.completion.chunk",
                    model: model_name.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            role,
                            content: Some(chunk.to_string()),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                // Tier one of cancellation. A failed send means the SSE
                // receiver is gone -- the browser tab closed, the
                // client aborted, the connection dropped -- and until
                // this was checked the return value was discarded and
                // the decode loop happily generated the remaining
                // hundreds of tokens into nothing. Flipping the same
                // flag `/v1/cancel` sets means there is one stop path,
                // not two.
                if tx_chunks
                    .blocking_send(Ok(Event::default().json_data(payload).unwrap()))
                    .is_err()
                {
                    cancel_token.cancel();
                }
            },
        );

        // `first` is still true when nothing was streamed from the emit
        // closure (the buffered tool-call/batching path, or an empty
        // generation), so the id has not gone out yet. `take()` on the
        // way into each payload below guarantees it is announced
        // exactly once, on whichever chunk really is first.
        let mut pending_request_id = first.then(|| request_id.clone());

        match result {
            Ok((finish, usage, full_text)) => {
                if let Some(id) = &session_id {
                    sessions.store_reply(
                        id,
                        ChatMessage {
                            role: "assistant".to_string(),
                            content: Some(MessageContent::Text(full_text.clone())),
                            tool_calls: None,
                            tool_call_id: None,
                        },
                    );
                }
                let tool_call = if tools_active {
                    extract_tool_call(&full_text)
                } else {
                    None
                };
                if !overlap {
                    if let Some((name, arguments)) = &tool_call {
                        let payload = ChatCompletionChunk {
                            id: request_id.clone(),
                            request_id: pending_request_id.take(),
                            object: "chat.completion.chunk",
                            model: model_name.clone(),
                            choices: vec![ChatCompletionChunkChoice {
                                index: 0,
                                delta: ChatCompletionChunkDelta {
                                    role: Some("assistant"),
                                    content: None,
                                    tool_calls: Some(vec![ToolCallOut {
                                        id: "call_0".to_string(),
                                        kind: "function",
                                        function: ToolCallFunctionOut {
                                            name: name.clone(),
                                            arguments: arguments.clone(),
                                        },
                                    }]),
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        let _ = tx.blocking_send(Ok(Event::default().json_data(payload).unwrap()));
                    } else if !full_text.is_empty() {
                        let payload = ChatCompletionChunk {
                            id: request_id.clone(),
                            request_id: pending_request_id.take(),
                            object: "chat.completion.chunk",
                            model: model_name.clone(),
                            choices: vec![ChatCompletionChunkChoice {
                                index: 0,
                                delta: ChatCompletionChunkDelta {
                                    role: Some("assistant"),
                                    content: Some(full_text),
                                    tool_calls: None,
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        let _ = tx.blocking_send(Ok(Event::default().json_data(payload).unwrap()));
                    }
                }
                let final_finish_reason = if tool_call.is_some() {
                    "tool_calls"
                } else {
                    finish.as_str()
                };
                let final_payload = ChatCompletionChunk {
                    id: request_id.clone(),
                    request_id: pending_request_id.take(),
                    object: "chat.completion.chunk",
                    model: model_name,
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            role: None,
                            content: None,
                            tool_calls: None,
                        },
                        finish_reason: Some(final_finish_reason),
                    }],
                    usage: Some(usage.clone()),
                };
                let _ = tx.blocking_send(Ok(Event::default().json_data(final_payload).unwrap()));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                // Recorded here rather than where the handler returned:
                // the handler returns as soon as the SSE headers go out,
                // which is before a single token exists, so timing it
                // there would report every stream as instant.
                stats_state.record_request(
                    &request_id,
                    ferrox_api::routes::V1_CHAT_COMPLETIONS,
                    200,
                    true,
                    started.elapsed().as_millis() as u64,
                    Some(&usage),
                );
            }
            Err(e) => {
                tracing::warn!("decode error on streamed request {request_id}: {e}");
                // The socket carried 200 -- SSE headers precede the
                // first token -- but the request produced no completion.
                // The monitor records outcomes, and a 200 row with zero
                // tokens would read as a successful empty answer, so the
                // failure is stated as 500 here and only here.
                stats_state.record_request(
                    &request_id,
                    ferrox_api::routes::V1_CHAT_COMPLETIONS,
                    500,
                    true,
                    started.elapsed().as_millis() as u64,
                    None,
                );
                let payload = ChatCompletionChunk {
                    id: request_id.clone(),
                    request_id: pending_request_id.take(),
                    object: "chat.completion.chunk",
                    model: model_name,
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            role: Some("assistant"),
                            content: Some(format!("[error: {e}]")),
                            tool_calls: None,
                        },
                        finish_reason: Some("stop"),
                    }],
                    usage: None,
                };
                let _ = tx.blocking_send(Ok(Event::default().json_data(payload).unwrap()));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
        }
    });

    let stream =
        futures_util::stream::unfold(
            rx,
            |mut rx| async move { rx.recv().await.map(|ev| (ev, rx)) },
        );
    // `X-Accel-Buffering: no` is the one header that actually reaches
    // the problem the plan names: nginx (and the proxies that copied
    // its convention) buffer `text/event-stream` by default, which
    // turns a token-by-token stream into one silent wait followed by
    // the whole answer at once -- indistinguishable, from the browser,
    // from a hung backend. axum already sets `Cache-Control: no-cache`
    // on an `Sse` response, so that half is covered.
    //
    // The keep-alive comment every 15s is the other half: it gives an
    // idle-but-healthy stream something to send, so a client's stall
    // timeout measures the *connection* rather than the model's
    // time-to-first-token on a long prompt.
    Ok((
        [(
            axum::http::HeaderName::from_static("x-accel-buffering"),
            axum::http::HeaderValue::from_static("no"),
        )],
        Sse::new(stream).keep_alive(KeepAlive::default()),
    )
        .into_response())
}

/// `POST /v1/cancel` -- the explicit half of two-tier cancellation.
///
/// Answers `200` when a live generation was signalled and `404` when
/// the id names nothing that is running. That difference is the whole
/// point of the endpoint returning a body at all: "already finished"
/// and "stopped it" are both fine outcomes, but only one of them saved
/// any work, and a UI told `ok: true` for both will claim it stopped
/// something it did not.
async fn cancel_generation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ferrox_api::CancelGenerationRequest>,
) -> Response {
    let cancelled = state.cancels.cancel(&req.request_id);
    let status = if cancelled {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    };
    let detail = if cancelled {
        "the generation was asked to stop; it ends at its next token".to_string()
    } else {
        "no generation with that request_id is running -- it has already \
         finished, was never issued, or was served by a path that does \
         not register for cancellation"
            .to_string()
    };
    (
        status,
        Json(ferrox_api::CancelGenerationResponse {
            request_id: req.request_id,
            cancelled,
            detail,
        }),
    )
        .into_response()
}

/// Turns a freshly loaded checkpoint into the pair that gets published
/// as the active model.
///
/// Extracted from `build_app_state` so `/admin/models/load` builds its
/// replacement exactly the way startup builds the first one -- a second
/// copy of this match would be a second place for a new engine variant
/// to be forgotten, and the difference would only show up as a model
/// that silently loses continuous batching after a swap.
pub(crate) fn activate_loaded_model(
    loaded: model::LoadedModel,
    enable_continuous_batching: bool,
) -> (Model, Option<batch_scheduler::ContinuousBatcher>) {
    match loaded {
        model::LoadedModel::Gguf(g) => {
            let decoder = Arc::new(g.decoder);
            let tokenizer = Arc::new(g.tokenizer);
            let batcher = if enable_continuous_batching {
                tracing::info!(
                    "continuous batching enabled: decode steps share Decoder::forward_multi_seq \
                     (stop sequences use the same pending-buffer trim as the private generate loop)"
                );
                let tok = Arc::clone(&tokenizer);
                let decode = Arc::new(move |ids: &[usize]| tok.decode(ids));
                Some(batch_scheduler::ContinuousBatcher::spawn(
                    Arc::clone(&decoder),
                    decode,
                ))
            } else {
                None
            };
            (
                Model::Gguf(GgufModel {
                    decoder,
                    tokenizer,
                    stop_tokens: g.stop_tokens,
                    bos_id: g.bos_id,
                    is_synthetic: g.is_synthetic,
                    chat_template: g.chat_template,
                }),
                batcher,
            )
        }
        model::LoadedModel::Kimi(k) => (
            Model::Kimi(KimiModel {
                engine: k.engine,
                tokenizer: k.tokenizer,
                stop_tokens: k.stop_tokens,
                chat_template: k.chat_template,
            }),
            None,
        ),
        model::LoadedModel::Mla(m) => (
            Model::Mla(MlaModel {
                engine: m.engine,
                tokenizer: m.tokenizer,
                stop_tokens: m.stop_tokens,
                bos_id: m.bos_id,
                name: m.name,
                chat_template: m.chat_template,
            }),
            None,
        ),
        model::LoadedModel::Gemma4(m) => (
            Model::Gemma4(Gemma4Model {
                engine: m.engine,
                tokenizer: m.tokenizer,
                stop_tokens: m.stop_tokens,
                bos_id: m.bos_id,
                name: m.name,
                chat_template: m.chat_template,
            }),
            None,
        ),
        model::LoadedModel::Glm52(g) => (
            Model::Glm52(Glm52Model {
                engine: g.engine,
                tokenizer: g.tokenizer,
                stop_tokens: g.stop_tokens,
                bos_id: g.bos_id,
                name: g.name,
                chat_template: g.chat_template,
            }),
            None,
        ),
    }
}

fn build_app_state(
    loaded: model::LoadedModel,
    kv_pool: Option<generate::KvPoolConfig>,
    prefix_cache: Option<Arc<Mutex<PrefixCache>>>,
    enable_continuous_batching: bool,
    mcp: Option<mcp::LoadedMcpConfig>,
    detection: Arc<health::Detection>,
) -> AppState {
    let (model, batcher) = activate_loaded_model(loaded, enable_continuous_batching);
    // The startup model's admin id is whichever discovered entry sits
    // at the configured path; `None` when it was not discovered (the
    // synthetic fallback, or a path outside the scanned directories),
    // in which case `/admin/models` reports nothing as active rather
    // than inventing an id no `load` request could name.
    let id = startup_model_id();
    AppState {
        active: std::sync::RwLock::new(Some(Arc::new(ActiveModel {
            id,
            model: Arc::new(model),
            batcher,
        }))),
        load_in_progress: std::sync::atomic::AtomicBool::new(false),
        tasks: Arc::new(tasks::TaskRegistry::new()),
        cancels: Arc::new(cancel::CancelRegistry::new()),
        stats: stats::Stats::new(),
        model_dir: admin::model_dirs().into_iter().next(),
        response_cache: Mutex::new(ResponseCache::new(1000, Duration::from_secs(3600))),
        kv_pool,
        prefix_cache,
        sessions: session::SessionStore::new(),
        requests_total: std::sync::atomic::AtomicU64::new(0),
        request_errors_total: std::sync::atomic::AtomicU64::new(0),
        started_at: std::time::Instant::now(),
        last_request_ms: std::sync::atomic::AtomicU64::new(0),
        detection,
        mcp,
        continuous_batching_enabled: enable_continuous_batching,
        loading_model: Mutex::new(None),
        last_load_error: Mutex::new(None),
    }
}

/// The `/admin/models` id of the checkpoint `FERROX_MODEL_PATH` names,
/// when discovery finds it. Matching on the resolved path rather than
/// on the filename keeps two same-named files in different directories
/// from claiming each other's id.
fn startup_model_id() -> Option<String> {
    let configured = std::env::var("FERROX_MODEL_PATH").ok()?;
    let configured = std::fs::canonicalize(&configured).ok()?;
    admin::discover(&admin::model_dirs())
        .into_iter()
        .find(|d| {
            std::fs::canonicalize(&d.path)
                .map(|p| p == configured)
                .unwrap_or(false)
        })
        .map(|d| d.id)
}

/// Builds the global rayon pool up front, on the main thread, with an
/// explicit width and QoS (see [`ferrox_core::threads`]).
///
/// Doing this from `main` rather than letting rayon build lazily is the
/// point: the first rayon call inside this server happens on a Tokio
/// `spawn_blocking` thread, so the workers used to inherit that thread's
/// QoS class -- which on macOS decides whether they land on performance
/// or efficiency cores.
fn init_cpu_pool() {
    match ferrox_core::threads::init_cpu_pool() {
        Some(n) => eprintln!(
            "ferrox-server: rayon pool {n} threads (perf cores {}; override with FERROX_CPU_THREADS)",
            ferrox_core::threads::perf_core_count()
        ),
        None => eprintln!("ferrox-server: global rayon pool already built; leaving it alone"),
    }
}

/// Prints the machine-readable ready line (see `ferrox_api::lifecycle`)
/// on stdout and flushes it.
///
/// This one line is what makes `--port 0` usable, and it deletes a whole
/// feature from any supervising process: no "is the port free" probe, no
/// `lsof` to work out whether an existing listener is a stale copy of
/// ourselves or a stranger's server, no dialog to explain the result.
/// The kernel picks the port and the child says what it got.
///
/// Shares stdout with the tracing subscriber on purpose -- a parent
/// reads stdout line by line and ignores anything that is not the ready
/// event, which `ServerReady::from_line` does for it.
fn announce_ready(addr: SocketAddr, scheme: &str) {
    use std::io::Write;
    let ready =
        ferrox_api::ServerReady::new(addr, scheme, env!("CARGO_PKG_VERSION"), std::process::id());
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{}", ready.to_line());
    let _ = stdout.flush();
}

/// Resolves when the server should stop serving.
///
/// Stdin-close is the one orphan-prevention mechanism that behaves
/// identically on macOS, Windows and Linux and survives a parent that
/// dies rather than exiting cleanly: the kernel closes the pipe either
/// way. The POSIX alternative -- a signal handler plus an exit hook plus
/// a reaper -- has no Windows equivalent at all, since there is no
/// SIGTERM there.
///
/// When disabled this future never resolves, which is exactly the
/// previous behaviour: serve until the process is stopped externally.
async fn shutdown_signal(exit_on_stdin_close: bool) {
    if !exit_on_stdin_close {
        std::future::pending::<()>().await;
        return;
    }
    let _ = tokio::task::spawn_blocking(|| {
        use std::io::Read;
        let mut sink = [0u8; 256];
        let mut stdin = std::io::stdin().lock();
        loop {
            match stdin.read(&mut sink) {
                // EOF: the parent is gone, or closed the pipe.
                Ok(0) => break,
                // Input on stdin is not a protocol here; drain it.
                Ok(_) => continue,
                Err(e) => {
                    tracing::warn!("stdin read failed ({e}); treating it as closed");
                    break;
                }
            }
        }
    })
    .await;
    tracing::info!("stdin closed; shutting down");
}

/// Tokio worker threads. The default is one per logical core, which on a
/// 10-core M2 Pro means 10 async workers oversubscribing the same cores
/// the rayon decode pool needs. Serving work here is almost entirely I/O
/// plus `spawn_blocking` handoff, so a small fixed pool is enough.
fn tokio_worker_threads() -> usize {
    std::env::var("FERROX_TOKIO_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(2)
}

/// Parses llama-server-style options and applies their environment
/// overrides before creating Tokio or Rayon worker threads. It then
/// brackets the async server lifecycle with journal records.
/// Install rustls' `ring` crypto provider as the process default.
///
/// `axum-server` is built with `tls-rustls-no-provider`, which
/// deliberately does NOT pick a backend -- see the comment on the
/// dependency in `Cargo.toml`. rustls then has no default provider, and
/// building a `ServerConfig` without one fails at ACCEPT time rather
/// than at compile time, which is the worst place for it to surface: a
/// server that started cleanly and refuses every TLS connection.
///
/// So this runs unconditionally at startup, not lazily in the TLS arm.
/// `install_default` returns `Err` if a provider is already installed,
/// which is not a failure -- it means something else got there first
/// and the invariant we care about (there IS a provider) already holds.
fn install_ring_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse_from(rewrite_llama_style_argv(std::env::args().collect()));
    if args.list_devices {
        print_available_devices();
        return Ok(());
    }
    apply_cli_overrides(&args)?;

    // Before the model is loaded and before the port is bound: refuse
    // to be the second process holding weights on this host. Held for
    // the life of the process -- dropping it deregisters us.
    let _instance = {
        use ferrox_core::instance::{register, InstancePolicy};
        let policy = if args.allow_multiple_instances {
            InstancePolicy::Multi
        } else {
            InstancePolicy::from_env_or(InstancePolicy::Single)
        };
        let model = std::env::var("FERROX_MODEL_PATH").ok();
        register(
            "server",
            model.as_deref(),
            ferrox_core::instance::current_backend(),
            policy,
        )
        .map_err(|conflict| anyhow::anyhow!("{conflict}"))?
    };

    let journal = journal::Journal::from_env();
    eprintln!(
        "ferrox-server: process lifecycle journal at {:?} (override with FERROX_JOURNAL_PATH)",
        journal.path()
    );
    journal.append(&journal::Record::session_start(
        env!("CARGO_PKG_VERSION"),
        std::process::id(),
    ));
    journal::install_panic_hook(journal.clone());

    let mcp_config_path = args.mcp_config.clone();
    let ui_server = args.ui_server
        || std::env::var("FERROX_UI")
            .map(|v| v == "1")
            .unwrap_or(false);
    let exit_on_stdin_close = args.exit_on_stdin_close
        || std::env::var("FERROX_EXIT_ON_STDIN_CLOSE")
            .map(|v| v == "1")
            .unwrap_or(false);

    // Before Tokio exists, so the decode pool's threads are not spawned
    // from (and do not inherit the QoS of) a blocking-pool thread.
    // SAFETY: still single-threaded here.
    unsafe { ferrox_core::weight_matrix::default_cpu_int_dot_on() };
    init_cpu_pool();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(tokio_worker_threads())
        .enable_all()
        .build()?;
    let result = runtime.block_on(run(mcp_config_path, ui_server, exit_on_stdin_close));

    let reason = match &result {
        Ok(()) => "normal".to_string(),
        Err(e) => e.to_string(),
    };
    journal.append(&journal::Record::session_exit(reason));

    // Dropping the runtime instead would wait for blocking tasks, and
    // the stdin watcher parks in a blocking read that may never return
    // (a terminal keeps stdin open forever). The serving future has
    // already finished by here, so nothing useful is being abandoned.
    runtime.shutdown_background();

    result
}

async fn run(
    mcp_config_path: Option<PathBuf>,
    ui_server: bool,
    exit_on_stdin_close: bool,
) -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Fail-closed listener check, before anything else (including
    // loading the model, so a misconfigured bind fails fast rather than
    // after however long that takes): refuse to start bound to a
    // non-loopback address with no API key configured, unless the
    // operator has explicitly opted into that via
    // FERROX_ALLOW_UNAUTHENTICATED_REMOTE=1 -- see
    // `security::check_bind_authorization`'s doc comment for why an
    // address that doesn't even parse as loopback is treated the same
    // as a confirmed non-loopback one.
    let addr = std::env::var("FERROX_ADDR").unwrap_or_else(|_| "127.0.0.1:8383".to_string());
    let api_key_configured = std::env::var("FERROX_API_KEY").is_ok();
    let allow_unauthenticated_remote = std::env::var("FERROX_ALLOW_UNAUTHENTICATED_REMOTE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if let Err(msg) =
        security::check_bind_authorization(&addr, api_key_configured, allow_unauthenticated_remote)
    {
        anyhow::bail!(msg);
    }

    let mut loaded = model::load()?;
    match &loaded {
        model::LoadedModel::Gguf(g) => tracing::info!(
            "loaded GGUF model '{}' (synthetic={}, tokenizer={})",
            g.decoder.config.name,
            g.is_synthetic,
            g.tokenizer.kind()
        ),
        model::LoadedModel::Kimi(k) => tracing::info!(
            "loaded Kimi K3 checkpoint (tokenizer={} base tokens)",
            k.tokenizer.vocab_size()
        ),
        model::LoadedModel::Mla(m) => tracing::info!(
            "loaded MLA GGUF '{}' (tokenizer={})",
            m.name,
            m.tokenizer.kind()
        ),
        model::LoadedModel::Gemma4(m) => tracing::info!(
            "loaded Gemma4 GGUF '{}' (tokenizer={})",
            m.name,
            m.tokenizer.kind()
        ),
        model::LoadedModel::Glm52(g) => tracing::info!(
            "loaded GLM-5.2 GGUF '{}' (tokenizer={})",
            g.name,
            g.tokenizer.kind()
        ),
    }
    // Opt-in VRAM budget for GPU-resident MoE experts. When unset but
    // Metal is active, default to a large budget so routed experts that
    // have Metal-capable quants run via `run_expert_placed` (Metal
    // matvec) instead of staying on CPU after Metal attention. Explicit
    // `FERROX_GPU_VRAM_BUDGET_BYTES=0` keeps the historical all-CPU MoE
    // placement. CUDA builds still require an explicit budget (Vast /
    // multi-GPU hosts vary too much for a safe default).
    let metal_default_moe_budget = {
        #[cfg(feature = "metal")]
        {
            ferrox_core::metal_dense_enabled()
                && std::env::var("FERROX_GPU_VRAM_BUDGET_BYTES").is_err()
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    };
    if let Ok(budget_str) = std::env::var("FERROX_GPU_VRAM_BUDGET_BYTES") {
        let budget: u64 = budget_str
            .parse()
            .expect("FERROX_GPU_VRAM_BUDGET_BYTES must be a non-negative integer");
        match &mut loaded {
            model::LoadedModel::Gguf(g) => {
                tracing::info!(
                    "GPU expert placement enabled: {budget} byte VRAM budget for routed experts \
                     (CUDA and/or Metal matvecs when built with the matching feature)"
                );
                g.decoder.gpu_vram_budget_bytes = Some(budget);
            }
            model::LoadedModel::Kimi(_) => {
                tracing::warn!(
                    "FERROX_GPU_VRAM_BUDGET_BYTES is set but the loaded model is Kimi K3 -- not \
                     supported yet (its MoE stack isn't wired to PlacementPlan), ignoring"
                );
            }
            model::LoadedModel::Mla(_) => {
                tracing::warn!(
                    "FERROX_GPU_VRAM_BUDGET_BYTES is set but the loaded model is MLA -- dense \
                     FFN path only today; ignoring expert VRAM budget"
                );
            }
            model::LoadedModel::Gemma4(_) => {
                tracing::warn!(
                    "FERROX_GPU_VRAM_BUDGET_BYTES is set but the loaded model is Gemma4 -- \
                     ignoring expert VRAM budget"
                );
            }
            model::LoadedModel::Glm52(_) => {
                tracing::warn!(
                    "FERROX_GPU_VRAM_BUDGET_BYTES is set but the loaded model is GLM-5.2 DSA -- \
                     GPU expert placement not wired yet; ignoring"
                );
            }
        }
    } else if metal_default_moe_budget {
        // ~64 GiB sentinel: place as many experts as the planner allows;
        // Metal unified memory makes a hard VRAM split less meaningful
        // than on discrete CUDA cards.
        const METAL_DEFAULT_MOE_BUDGET: u64 = 64 * 1024 * 1024 * 1024;
        if let model::LoadedModel::Gguf(g) = &mut loaded {
            tracing::info!(
                "Metal MoE expert placement default-on ({METAL_DEFAULT_MOE_BUDGET} byte budget); \
                 set FERROX_GPU_VRAM_BUDGET_BYTES=0 to force CPU experts"
            );
            g.decoder.gpu_vram_budget_bytes = Some(METAL_DEFAULT_MOE_BUDGET);
        }
    }
    #[cfg(feature = "cuda")]
    {
        if ferrox_core::cuda_dense_enabled() {
            tracing::info!(
                "CUDA dense matvec enabled for WeightMatrix::apply \
                 (FERROX_CUDA=0|cpu forces CPU; weight buffers stay resident after first upload)"
            );
        } else {
            tracing::info!(
                "CUDA dense matvec disabled (FERROX_CUDA); dense decode uses CPU or Metal"
            );
        }
    }
    #[cfg(feature = "metal")]
    {
        if ferrox_core::metal_dense_enabled() {
            tracing::info!(
                "Metal dense matvec enabled for WeightMatrix::apply \
                 (FERROX_METAL=0|cpu forces CPU; weight buffers stay resident after first upload)"
            );
            match std::env::var("FERROX_METAL_ATTN").ok().as_deref() {
                Some("1") | Some("true") | Some("on") | Some("attn") => {
                    tracing::info!(
                        "Metal fused attention requested (FERROX_METAL_ATTN): \
                         QKV→RoPE→GQA→O on-GPU for Norm/NeoX decode without QKV bias/QK-norm"
                    );
                }
                _ => {}
            }
            match std::env::var("FERROX_METAL_LOGITS").ok().as_deref() {
                Some("1") | Some("true") | Some("on") | Some("logits") => {
                    tracing::info!(
                        "Metal logits-in-stack enabled (FERROX_METAL_LOGITS): \
                         final_norm+lm_head on-GPU (often slower than host lm_head)"
                    );
                }
                _ => {}
            }
            match std::env::var("FERROX_METAL_GREEDY_GPU").ok().as_deref() {
                Some("0") | Some("false") | Some("off") | Some("no") => {
                    tracing::info!(
                        "Metal greedy GPU argmax disabled (FERROX_METAL_GREEDY_GPU=0); \
                         temperature<=0 uses host lm_head"
                    );
                }
                _ => {
                    tracing::info!(
                        "Metal greedy GPU argmax on by default (opt out FERROX_METAL_GREEDY_GPU=0): \
                         temperature<=0 folds final_norm+lm_head+argmax into dense stack"
                    );
                }
            }
        } else {
            tracing::info!("Metal dense matvec disabled (FERROX_METAL); dense decode uses CPU");
        }
    }
    // Both env vars are required together to enable pooling; unset ->
    // caches keep their original unbounded-per-request growth. This
    // mirrors the FERROX_API_KEY / FERROX_RATE_LIMIT_PER_MINUTE
    // pattern below: opt-in, off by default.
    //
    // Block count can be set explicitly (`FERROX_KV_POOL_BLOCKS` +
    // `FERROX_KV_POOL_BLOCK_SIZE`) or derived from a byte budget
    // (`FERROX_KV_BYTE_BUDGET` + `FERROX_KV_POOL_BLOCK_SIZE`, GGUF
    // models only). `FERROX_KV_POOL_BLOCKS` and
    // `FERROX_KV_BYTE_BUDGET` are mutually exclusive.
    let blocks_env = std::env::var("FERROX_KV_POOL_BLOCKS");
    let block_size_env = std::env::var("FERROX_KV_POOL_BLOCK_SIZE");
    let byte_budget_env = std::env::var("FERROX_KV_BYTE_BUDGET");
    if blocks_env.is_ok() && byte_budget_env.is_ok() {
        panic!(
            "FERROX_KV_POOL_BLOCKS and FERROX_KV_BYTE_BUDGET are mutually exclusive \
             (set one block-count source plus FERROX_KV_POOL_BLOCK_SIZE, or neither to disable)"
        );
    }
    let kv_pool = match (blocks_env, block_size_env, byte_budget_env) {
        (Ok(blocks), Ok(block_size), Err(_)) => {
            let total_blocks: usize = blocks
                .parse()
                .expect("FERROX_KV_POOL_BLOCKS must be a positive integer");
            let block_size: usize = block_size
                .parse()
                .expect("FERROX_KV_POOL_BLOCK_SIZE must be a positive integer");
            // Optional and independent of the two above: how long a
            // request retries before giving up when the pool is
            // momentarily exhausted, instead of rejecting on the very
            // first failed attempt. Zero (the default if unset)
            // preserves the original reject-immediately behavior.
            let queue_wait_ms: u64 = std::env::var("FERROX_KV_POOL_QUEUE_TIMEOUT_MS")
                .ok()
                .map(|v| {
                    v.parse()
                        .expect("FERROX_KV_POOL_QUEUE_TIMEOUT_MS must be a non-negative integer")
                })
                .unwrap_or(0);
            tracing::info!(
                "KV cache block pool enabled: {total_blocks} blocks x {block_size} positions \
                 each, shared across all concurrent requests, {queue_wait_ms}ms admission queue wait"
            );
            Some(generate::KvPoolConfig {
                pool: Arc::new(Mutex::new(KvBlockPool::new(block_size, total_blocks))),
                queue_wait: Duration::from_millis(queue_wait_ms),
            })
        }
        (Err(_), Ok(block_size), Ok(byte_budget)) => {
            let block_size: usize = block_size
                .parse()
                .expect("FERROX_KV_POOL_BLOCK_SIZE must be a positive integer");
            let budget: u64 = byte_budget
                .parse()
                .expect("FERROX_KV_BYTE_BUDGET must be a positive integer");
            let cfg = match &loaded {
                model::LoadedModel::Gguf(g) => &g.decoder.config,
                model::LoadedModel::Kimi(_)
                | model::LoadedModel::Mla(_)
                | model::LoadedModel::Gemma4(_)
                | model::LoadedModel::Glm52(_) => {
                    panic!(
                        "FERROX_KV_BYTE_BUDGET requires a GGUF decoder model \
                         (set FERROX_MODEL_PATH to a generic-decoder .gguf file)"
                    );
                }
            };
            let bytes_per_block = block_size
                * cfg.n_layers
                * cfg.n_kv_heads
                * cfg.head_dim
                * 2
                * std::mem::size_of::<f32>();
            assert!(
                bytes_per_block > 0,
                "derived KV block byte size must be positive (check model config and block size)"
            );
            let total_blocks = (budget as usize / bytes_per_block).max(1);
            let queue_wait_ms: u64 = std::env::var("FERROX_KV_POOL_QUEUE_TIMEOUT_MS")
                .ok()
                .map(|v| {
                    v.parse()
                        .expect("FERROX_KV_POOL_QUEUE_TIMEOUT_MS must be a non-negative integer")
                })
                .unwrap_or(0);
            tracing::info!(
                "KV cache block pool enabled from byte budget: {budget} bytes / \
                 {bytes_per_block} bytes per block ({block_size} positions x {} layers) -> \
                 {total_blocks} blocks, {queue_wait_ms}ms admission queue wait",
                cfg.n_layers
            );
            Some(generate::KvPoolConfig {
                pool: Arc::new(Mutex::new(KvBlockPool::new(block_size, total_blocks))),
                queue_wait: Duration::from_millis(queue_wait_ms),
            })
        }
        (Err(_), Err(_), Err(_)) => None,
        (Err(_), Ok(_), Err(_)) => panic!(
            "FERROX_KV_POOL_BLOCK_SIZE requires FERROX_KV_POOL_BLOCKS or FERROX_KV_BYTE_BUDGET \
             (or unset all three to disable KV cache pooling)"
        ),
        (Ok(_), Ok(_), Ok(_)) => {
            unreachable!("FERROX_KV_POOL_BLOCKS and FERROX_KV_BYTE_BUDGET are mutually exclusive")
        }
        (Ok(_), Err(_), _) | (Err(_), Err(_), Ok(_)) => panic!(
            "FERROX_KV_POOL_BLOCKS/FERROX_KV_BYTE_BUDGET and FERROX_KV_POOL_BLOCK_SIZE must be \
             set together (or neither, to disable KV cache pooling)"
        ),
    };
    // Mutually exclusive with kv_pool (see generate::generate's doc
    // comment on why a pool-backed cache can't safely be restored from
    // a prefix-cache clone): if both are set, the KV pool wins and
    // prefix caching is simply never consulted -- generate() already
    // enforces this per-request, so this is a heads-up for the
    // operator, not a hard failure.
    let prefix_cache = std::env::var("FERROX_PREFIX_CACHE_ENTRIES").ok().map(|v| {
        let max_entries: usize = v
            .parse()
            .expect("FERROX_PREFIX_CACHE_ENTRIES must be a positive integer");
        if kv_pool.is_some() {
            tracing::warn!(
                "FERROX_PREFIX_CACHE_ENTRIES is set but so is the KV pool -- prefix \
                     caching will never be consulted while a KV pool is configured"
            );
        }
        tracing::info!(
            "KV-prefix cache enabled: up to {max_entries} stored prefixes, shared across \
                 all requests"
        );
        Arc::new(Mutex::new(PrefixCache::new(max_entries)))
    });
    if matches!(
        loaded,
        model::LoadedModel::Kimi(_) | model::LoadedModel::Mla(_) | model::LoadedModel::Glm52(_)
    ) && (kv_pool.is_some() || prefix_cache.is_some())
    {
        tracing::warn!(
            "KV pool / prefix cache are configured but the loaded model is Kimi, MLA, or GLM-5.2 -- \
             neither is consulted for those engines (state shapes differ from Decoder KV); see \
             ferrox_models::engine's module docs"
        );
    }
    let enable_cb = std::env::var("FERROX_CONTINUOUS_BATCHING")
        .map(|v| v == "1")
        .unwrap_or(false)
        && kv_pool.is_none()
        && prefix_cache.is_none()
        && matches!(loaded, model::LoadedModel::Gguf(_));
    if std::env::var("FERROX_CONTINUOUS_BATCHING")
        .map(|v| v == "1")
        .unwrap_or(false)
        && (kv_pool.is_some() || prefix_cache.is_some())
    {
        tracing::warn!(
            "FERROX_CONTINUOUS_BATCHING=1 ignored while KV pool or prefix cache is configured \
             (those modes keep the private generate path)"
        );
    }
    if let Ok(n) = std::env::var("FERROX_CHUNKED_PREFILL") {
        if let Ok(chunk) = n.parse::<usize>() {
            if chunk > 0 {
                tracing::info!("chunked prefill enabled: {chunk} tokens per forward_batch chunk");
            }
        }
    }
    if matches!(
        std::env::var("FERROX_CPU_KV_OFFLOAD").ok().as_deref(),
        Some("1")
    ) {
        tracing::warn!(
            "FERROX_CPU_KV_OFFLOAD=1: syncing Metal KV to host after each decode step \
             (minimal spill; full layer offload still planned)"
        );
    }

    let mcp = match mcp_config_path {
        Some(path) => {
            let loaded = mcp::load_mcp_config(&path)?;
            tracing::info!(
                "MCP config loaded from {} ({} server(s); invocation not wired yet)",
                loaded.path,
                loaded.servers.len()
            );
            Some(loaded)
        }
        None => None,
    };

    // Started before the router is built so the probe overlaps with
    // binding the port: by the time a client can ask, it has usually
    // already landed.
    let detection = health::Detection::spawn();

    let state = Arc::new(build_app_state(
        loaded,
        kv_pool,
        prefix_cache,
        enable_cb,
        mcp,
        detection,
    ));

    // Paths come from `ferrox_api::routes` rather than string literals
    // so the UI, `ferrox chat` and this router cannot disagree about
    // what the surface is.
    use ferrox_api::routes;

    let mut public = Router::new().route(routes::HEALTH, get(health));
    if ui_server {
        tracing::info!("web UI enabled at {} and {}", routes::ROOT, routes::UI);
        // The shell and its assets are static and carry no data, so
        // they sit on the unauthenticated side beside /health: the
        // screen a user needs in order to *enter* an API key cannot
        // itself be behind that key. Every call the frontend then makes
        // goes through the same gate as any other client's.
        public = ui::attach(public);
    }

    let mut protected = Router::new()
        .route(routes::V1_MODELS, get(list_models))
        .route(routes::V1_CHAT_COMPLETIONS, post(chat_completions))
        // Behind the same key as the endpoint that started the work:
        // an unauthenticated caller must not be able to stop someone
        // else's generation by guessing at request ids.
        .route(routes::V1_CANCEL, post(cancel_generation))
        .route(routes::V1_MESSAGES, post(anthropic::messages))
        .route(routes::V1_COMPLETIONS, post(openai_extra::completions))
        .route(routes::V1_TOKENIZE, post(openai_extra::tokenize))
        .route(routes::V1_DETOKENIZE, post(openai_extra::detokenize))
        .route(routes::V1_EMBEDDINGS, post(openai_extra::embeddings))
        .route(routes::CACHE_STATS, get(cache_stats))
        .route(routes::METRICS, get(metrics))
        // The control surface. Registered inside `protected` on
        // purpose: these routes change what the server serves and write
        // to disk, so they get the same FERROX_API_KEY gate as /v1/*
        // and never the unauthenticated treatment /health has.
        .route(routes::ADMIN_MODELS, get(admin::models))
        .route(routes::ADMIN_MODELS_LOAD, post(admin::load_model))
        .route(routes::ADMIN_MODELS_UNLOAD, post(admin::unload_model))
        .route(routes::ADMIN_DOWNLOAD, post(admin::download))
        .route(routes::ADMIN_TASKS, get(admin::tasks))
        .route(&admin::cancel_route(), post(admin::cancel_task))
        .route(routes::ADMIN_STATS, get(admin::stats));

    // Both off by default; set the corresponding env var to enable.
    // route_layer (not layer) so these apply only to the routes above,
    // never to /health, which stays reachable for liveness/readiness
    // probes regardless of auth or rate-limit configuration.
    if let Ok(key) = std::env::var("FERROX_API_KEY") {
        tracing::info!("API key auth enabled");
        let auth = limits::AuthConfig {
            api_key: Arc::new(key),
        };
        protected = protected.route_layer(axum::middleware::from_fn_with_state(
            auth,
            limits::require_api_key,
        ));
    }
    if let Ok(rpm) = std::env::var("FERROX_RATE_LIMIT_PER_MINUTE") {
        let rpm: u32 = rpm
            .parse()
            .expect("FERROX_RATE_LIMIT_PER_MINUTE must be a positive integer");
        tracing::info!("rate limiting enabled: {rpm} requests/minute (global)");
        let limiter = Arc::new(limits::RateLimiter::per_minute(rpm));
        protected = protected.route_layer(axum::middleware::from_fn_with_state(
            limiter,
            limits::rate_limit,
        ));
    }
    // Off by default; set FERROX_CORS_ORIGINS (comma-separated exact
    // origins) to enable. No wildcard support by design -- see
    // `security::parse_cors_origins`'s doc comment. Added last (so it's
    // the outermost route_layer, run before auth/rate-limiting): a CORS
    // preflight (OPTIONS) request carries no Authorization header and
    // is answered directly by `CorsLayer` itself, so it must not be
    // blocked by the auth/rate-limit layers underneath.
    if let Ok(spec) = std::env::var("FERROX_CORS_ORIGINS") {
        let origins = security::parse_cors_origins(&spec)
            .unwrap_or_else(|e| panic!("FERROX_CORS_ORIGINS: {e}"));
        tracing::info!(
            "CORS enabled: {} allow-listed origin(s) ({})",
            origins.len(),
            spec
        );
        let cors = tower_http::cors::CorsLayer::new()
            .allow_origin(tower_http::cors::AllowOrigin::list(origins))
            .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
            ]);
        protected = protected.route_layer(cors);
    }

    // Outermost on purpose: every 503 this server can emit -- from a
    // handler, from `require_active`, or from the batch scheduler's
    // queue cap -- leaves with a `Retry-After` a client can act on.
    let app = public
        .merge(protected)
        .layer(axum::middleware::from_fn(limits::retry_after))
        .with_state(state);

    // TLS is off by default -- set FERROX_TLS_CERT and FERROX_TLS_KEY
    // together to serve HTTPS instead of plain HTTP; unset (either or
    // both) preserves the original plain-HTTP behavior exactly. See
    // `security::tls_paths_from_env`'s doc comment for why this can't
    // be meaningfully unit-tested here.
    let tls_paths = security::tls_paths_from_env().unwrap_or_else(|e| panic!("{e}"));
    install_ring_crypto_provider();
    // Both arms bind first and read the address back off the socket
    // rather than trusting the requested one: with `--port 0` the
    // requested port is a lie by construction, and the ready line has
    // to carry what the kernel actually handed out.
    match tls_paths {
        Some(paths) => {
            let config =
                axum_server::tls_rustls::RustlsConfig::from_pem_file(&paths.cert, &paths.key)
                    .await
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "failed to load TLS cert/key ({:?}, {:?}): {e}",
                            paths.cert,
                            paths.key
                        )
                    })?;
            let socket_addr: std::net::SocketAddr = addr
                .parse()
                .map_err(|e| anyhow::anyhow!("invalid FERROX_ADDR {addr:?} for TLS: {e}"))?;
            let listener = std::net::TcpListener::bind(socket_addr)?;
            // Tokio panics outright when handed a BLOCKING socket
            // ("Registering a blocking socket with the tokio runtime is
            // unsupported"), and axum-server registers this one
            // internally. Without this the TLS arm binds, prints its
            // ready line, and then panics on the first accept -- so the
            // failure looks like a healthy start followed by a server
            // that answers nothing.
            listener.set_nonblocking(true)?;
            let bound = listener.local_addr()?;
            tracing::info!("TLS enabled: ferrox-server listening on https://{bound}");
            announce_ready(bound, "https");

            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();
            tokio::spawn(async move {
                shutdown_signal(exit_on_stdin_close).await;
                shutdown_handle.graceful_shutdown(Some(Duration::from_secs(5)));
            });
            axum_server::from_tcp_rustls(listener, config)?
                .handle(handle)
                .serve(app.into_make_service())
                .await?;
        }
        None => {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            let bound = listener.local_addr()?;
            tracing::info!("ferrox-server listening on {bound}");
            announce_ready(bound, "http");
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal(exit_on_stdin_close))
                .await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_models::config::test_dense_fixture;

    #[test]
    fn parses_llama_server_style_options() {
        let argv = [
            "ferrox-server",
            "-m",
            "model.gguf",
            "--host",
            "::1",
            "--port",
            "9000",
            "-t",
            "4",
            "-dev",
            "Metal",
            "-ngl",
            "all",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let args = ServerArgs::try_parse_from(rewrite_llama_style_argv(argv)).unwrap();

        assert_eq!(args.model.as_deref(), Some("model.gguf"));
        assert_eq!(args.host, Some(IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)));
        assert_eq!(args.port, Some(9000));
        assert_eq!(args.threads, Some(4));
        assert_eq!(args.device, Some(OffloadDevice::Metal));
        assert_eq!(args.n_gpu_layers, Some(GpuLayers::All));
        assert_eq!(
            cli_bind_addr(&args, Some("127.0.0.1:8383")).as_deref(),
            Some("[::1]:9000")
        );
    }

    #[test]
    fn port_zero_survives_argument_parsing_as_a_real_request() {
        // `--port 0` must reach the bind call intact: it is a request
        // for a kernel-assigned port, not a missing value to default to
        // 8383. The address it produces is deliberately provisional --
        // the ready line reports what was actually bound.
        let argv = ["ferrox-server", "--port", "0"]
            .into_iter()
            .map(String::from)
            .collect();
        let args = ServerArgs::try_parse_from(rewrite_llama_style_argv(argv)).unwrap();
        assert_eq!(args.port, Some(0));
        assert_eq!(
            cli_bind_addr(&args, Some("127.0.0.1:8383")).as_deref(),
            Some("127.0.0.1:0")
        );
    }

    #[test]
    fn stdin_close_exit_is_opt_in() {
        // Default off: a server whose stdin is /dev/null (systemd, cron,
        // nohup) would otherwise exit the instant it started.
        let args =
            ServerArgs::try_parse_from(["ferrox-server"].into_iter().map(String::from)).unwrap();
        assert!(!args.exit_on_stdin_close);
        let args = ServerArgs::try_parse_from(
            ["ferrox-server", "--exit-on-stdin-close"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(args.exit_on_stdin_close);
    }

    #[test]
    fn the_ready_line_round_trips_through_a_parent_reading_stdout() {
        let addr: SocketAddr = "127.0.0.1:51999".parse().unwrap();
        let ready = ferrox_api::ServerReady::new(addr, "http", "0.5.0", std::process::id());
        let parsed = ferrox_api::ServerReady::from_line(&ready.to_line()).unwrap();
        assert_eq!(parsed.port, 51999);
        assert_eq!(parsed.base_url(), "http://127.0.0.1:51999");
        // A parent reads stdout line by line; tracing shares the stream.
        assert!(ferrox_api::ServerReady::from_line("INFO ferrox-server listening").is_none());
    }

    fn test_model() -> Model {
        // Tiny vocab (32): raw byte ids ≥32 (e.g. ASCII "hello") are OOV.
        // HTTP/chat-template tests that need full ASCII use
        // `test_model_full_byte_vocab` instead.
        let cfg = test_dense_fixture();
        Model::Gguf(GgufModel {
            decoder: Arc::new(Decoder::new_random_small(cfg, 2, 32)),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: None,
            is_synthetic: true,
            chat_template: chat_template::ChatTemplate::Plain,
        })
    }

    fn greedy_params(max_tokens: usize) -> GenerationParams {
        GenerationParams {
            max_tokens,
            sampling: SamplingParams::default(),
            seed: 1,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            json_object: false,
            cancel: None,
        }
    }

    /// Declares a full 0..255 byte-compatible vocab so HTTP-level tests
    /// that render chat templates (ASCII role names) do not spuriously
    /// reject their own prompt prefixes.
    fn test_model_full_byte_vocab() -> Model {
        let mut cfg = test_dense_fixture();
        cfg.vocab_size = 256;
        Model::Gguf(GgufModel {
            decoder: Arc::new(Decoder::new_random_small(cfg, 2, 256)),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: None,
            is_synthetic: true,
            chat_template: chat_template::ChatTemplate::Plain,
        })
    }

    /// One `AppState` for the HTTP-level tests, so a new field on the
    /// struct is added in one place rather than in every test that
    /// builds one.
    fn test_state(model: Model, response_cache: ResponseCache) -> AppState {
        AppState {
            active: std::sync::RwLock::new(Some(Arc::new(ActiveModel {
                id: None,
                model: Arc::new(model),
                batcher: None,
            }))),
            load_in_progress: std::sync::atomic::AtomicBool::new(false),
            tasks: Arc::new(tasks::TaskRegistry::new()),
            cancels: Arc::new(cancel::CancelRegistry::new()),
            stats: stats::Stats::new(),
            model_dir: None,
            response_cache: Mutex::new(response_cache),
            kv_pool: None,
            prefix_cache: None,
            sessions: session::SessionStore::new(),
            requests_total: std::sync::atomic::AtomicU64::new(0),
            request_errors_total: std::sync::atomic::AtomicU64::new(0),
            started_at: std::time::Instant::now(),
            last_request_ms: std::sync::atomic::AtomicU64::new(0),
            detection: Arc::new(health::Detection::ready(health::probe_backends())),
            mcp: None,
            continuous_batching_enabled: false,
            loading_model: Mutex::new(None),
            last_load_error: Mutex::new(None),
        }
    }

    /// A real axum `Router` wired exactly like `main()`'s (minus auth/
    /// rate-limiting, which are orthogonal and already covered by
    /// `limits`'s own tests), backed by a fresh
    /// `test_model_full_byte_vocab()` -- so tool-calling/session tests
    /// exercise the real HTTP request/response path (JSON
    /// (de)serialization, routing, handler wiring, chat-template
    /// rendering) via `tower::ServiceExt::oneshot`, not just the inner
    /// functions directly.
    fn test_app() -> Router {
        test_app_with_state(Arc::new(test_state(
            test_model_full_byte_vocab(),
            ResponseCache::new(1000, Duration::from_secs(3600)),
        )))
    }

    /// [`test_app`] over a caller-owned state, so a test can reach in
    /// and swap or unload the model behind a live router.
    fn test_app_with_state(state: Arc<AppState>) -> Router {
        Router::new()
            .route(ferrox_api::routes::HEALTH, get(health))
            .route(ferrox_api::routes::V1_MODELS, get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/tokenize", post(openai_extra::tokenize))
            .route("/v1/detokenize", post(openai_extra::detokenize))
            .route("/v1/embeddings", post(openai_extra::embeddings))
            .route("/v1/completions", post(openai_extra::completions))
            .route(
                ferrox_api::routes::ADMIN_MODELS_UNLOAD,
                post(admin::unload_model),
            )
            .route(ferrox_api::routes::ADMIN_TASKS, get(admin::tasks))
            .route(ferrox_api::routes::ADMIN_STATS, get(admin::stats))
            .route(ferrox_api::routes::V1_CANCEL, post(cancel_generation))
            .with_state(state)
    }

    fn named_test_model(name: &'static str, vocab_size: usize) -> Model {
        let mut cfg = test_dense_fixture();
        cfg.name = name;
        cfg.vocab_size = vocab_size;
        Model::Gguf(GgufModel {
            decoder: Arc::new(Decoder::new_random_small(cfg, 2, 256)),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: None,
            is_synthetic: true,
            chat_template: chat_template::ChatTemplate::Plain,
        })
    }

    fn active_model(state: &AppState, name: &'static str) -> Arc<ActiveModel> {
        Arc::new(ActiveModel {
            id: Some(name.to_string()),
            model: Arc::new(named_test_model(name, 256)),
            batcher: None,
        })
        .tap_into(state)
    }

    /// Small helper so the swap tests read as "publish this model".
    trait TapInto {
        fn tap_into(self, state: &AppState) -> Self;
    }
    impl TapInto for Arc<ActiveModel> {
        fn tap_into(self, state: &AppState) -> Self {
            state.swap_active(Some(Arc::clone(&self)));
            self
        }
    }

    /// The load-order guarantee the whole swap design exists to make:
    /// a request that has already taken its handle finishes against the
    /// weights it started on, even though a different model has since
    /// been published. Anything else would splice two checkpoints into
    /// one completion.
    #[test]
    fn an_in_flight_request_keeps_the_model_it_started_on() {
        let state = test_state(
            named_test_model("model-a", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        );

        // A request that has begun: it has cloned the handle and is
        // about to decode against it.
        let in_flight = state.active().expect("a model is loaded");
        assert_eq!(in_flight.model.name(), "model-a");

        active_model(&state, "model-b");

        // The swap is visible to anything that asks *now*...
        assert_eq!(state.active().unwrap().model.name(), "model-b");
        // ...and completely invisible to the request already running.
        assert_eq!(in_flight.model.name(), "model-a");
        let (_chunks, finish, _usage) =
            run_generation(&in_flight.model, "hi", &greedy_params(3), None, None, None)
                .expect("the old model must still decode after being swapped out");
        assert!(matches!(finish, FinishReason::Length | FinishReason::Stop));
    }

    /// The other half of the same guarantee: the old model is not freed
    /// at swap time, it is freed when the last holder lets go. A design
    /// that dropped it eagerly would free weights out from under a
    /// decode loop.
    #[test]
    fn a_swapped_out_model_lives_until_its_last_holder_releases_it() {
        let state = test_state(
            named_test_model("model-a", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        );
        let in_flight = state.active().expect("a model is loaded");
        let weights = Arc::clone(&in_flight.model);
        assert!(Arc::strong_count(&weights) >= 2);

        let previous = state.swap_active(Some(Arc::new(ActiveModel {
            id: Some("model-b".to_string()),
            model: Arc::new(named_test_model("model-b", 256)),
            batcher: None,
        })));
        drop(previous);
        // The registry has let go; the in-flight request has not.
        assert!(Arc::strong_count(&weights) >= 2);
        drop(in_flight);
        assert_eq!(Arc::strong_count(&weights), 1);
    }

    /// Unload is not "keep serving the last thing loaded". A request
    /// that arrives afterwards must be told there is no model, not
    /// quietly served by a checkpoint the operator dropped.
    #[tokio::test]
    async fn unloading_answers_503_instead_of_serving_the_dropped_model() {
        let state = Arc::new(test_state(
            named_test_model("model-a", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        ));
        let app = test_app_with_state(Arc::clone(&state));

        let (status, body) = post_json_uri(
            &app,
            ferrox_api::routes::ADMIN_MODELS_UNLOAD,
            serde_json::json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["ok"], true);
        assert!(body["active"].is_null());
        assert!(state.active().is_none());

        let (status, _) = get_json(&app, ferrox_api::routes::V1_MODELS).await;
        assert_eq!(status, StatusCode::OK);
        let (_, models) = get_json(&app, ferrox_api::routes::V1_MODELS).await;
        assert_eq!(models["data"].as_array().unwrap().len(), 0);

        let (status, body) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "model_not_loaded");
    }

    /// `/health` must keep answering with nothing loaded -- a supervisor
    /// polls it to decide whether to kill the process, and "no model"
    /// is not "no server".
    #[tokio::test]
    async fn health_reports_the_unloaded_state_rather_than_going_silent() {
        let state = Arc::new(test_state(
            named_test_model("model-a", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        ));
        let app = test_app_with_state(Arc::clone(&state));
        state.swap_active(None);

        let (status, body) = get_json(&app, ferrox_api::routes::HEALTH).await;
        // Not `ready`: a supervisor reading 200 here would route traffic
        // that is guaranteed to 503 on arrival.
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["state"], "unavailable");
        assert_eq!(body["reason"], "model_not_loaded");
        assert!(body["model"].is_null());
        let real_weights = body["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == "real_weights")
            .cloned()
            .expect("real_weights is always reported");
        assert_eq!(real_weights["available"], false);
        assert_eq!(real_weights["reason"], "model_not_loaded");
    }

    /// The API-monitor contract: a finished request lands in the ring
    /// buffer keyed by the id the response carried, with the two
    /// durations reported separately.
    #[tokio::test]
    async fn a_finished_request_lands_in_the_stats_ring_with_both_durations() {
        let app = test_app();

        let (status, completion) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let request_id = completion["request_id"].as_str().unwrap().to_string();

        let (status, stats) = get_json(&app, ferrox_api::routes::ADMIN_STATS).await;
        assert_eq!(status, StatusCode::OK);
        let recent = stats["recent"].as_array().unwrap();
        assert_eq!(recent.len(), 1);
        let row = &recent[0];
        assert_eq!(row["request_id"], request_id);
        assert_eq!(row["route"], ferrox_api::routes::V1_CHAT_COMPLETIONS);
        assert_eq!(row["status"], 200);
        assert_eq!(row["stream"], false);
        // Separate fields, and the decode phase is a real measurement
        // rather than a copy of the total.
        assert!(row["duration_ms"].is_number());
        assert!(row["decode_ms"].is_number());
        assert!(stats["tokens_generated_total"].as_u64().unwrap() > 0);
        assert_eq!(
            stats["tokens_prompt_total"].as_u64().unwrap(),
            row["prompt_tokens"].as_u64().unwrap()
        );
    }

    /// A rejected request is still a request the monitor should show;
    /// otherwise the screen quietly omits exactly the traffic someone
    /// is debugging.
    #[tokio::test]
    async fn a_rejected_request_is_recorded_too() {
        let state = Arc::new(test_state(
            named_test_model("model-a", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        ));
        let app = test_app_with_state(Arc::clone(&state));
        state.swap_active(None);

        let (status, _) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

        let (_, stats) = get_json(&app, ferrox_api::routes::ADMIN_STATS).await;
        let recent = stats["recent"].as_array().unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0]["status"], 503);
        assert_eq!(recent[0]["completion_tokens"], 0);
        assert!(recent[0]["decode_ms"].is_null());
        assert_eq!(stats["errors_total"], 1);
    }

    /// An empty task list is a list, not a missing key -- the UI renders
    /// "no jobs" from it rather than from an error.
    #[tokio::test]
    async fn the_task_list_starts_empty_rather_than_absent() {
        let app = test_app();
        let (status, body) = get_json(&app, ferrox_api::routes::ADMIN_TASKS).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["tasks"].as_array().unwrap().len(), 0);
    }

    async fn post_json_uri(
        app: &Router,
        uri: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({}));
        (status, json)
    }

    async fn post_json(app: &Router, body: serde_json::Value) -> serde_json::Value {
        post_json_uri(app, "/v1/chat/completions", body).await.1
    }

    /// Cancelling an id that is not generating must not answer `200`.
    /// A UI told "ok" for an already-finished request would report that
    /// it stopped work it did not stop, and the two outcomes are the
    /// only thing this endpoint exists to distinguish.
    #[tokio::test]
    async fn cancelling_an_id_that_is_not_generating_is_a_404_that_says_so() {
        let app = test_app();
        let (status, body) = post_json_uri(
            &app,
            ferrox_api::routes::V1_CANCEL,
            serde_json::json!({ "request_id": "chatcmpl-never-issued" }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["cancelled"], serde_json::json!(false));
        assert_eq!(body["request_id"], "chatcmpl-never-issued");
        assert!(
            body["detail"].as_str().is_some_and(|d| !d.is_empty()),
            "the verdict must carry a human reason: {body}"
        );
    }

    /// The endpoint reaches the registry the streaming path registers
    /// into -- not a second, parallel one. Registered by hand here
    /// because a `oneshot` router cannot hold a stream open.
    #[tokio::test]
    async fn cancelling_a_live_generation_signals_its_token_and_answers_200() {
        let state = Arc::new(test_state(
            test_model_full_byte_vocab(),
            ResponseCache::new(1000, Duration::from_secs(3600)),
        ));
        let app = test_app_with_state(Arc::clone(&state));
        let (token, _guard) = state.cancels.register("chatcmpl-live");

        let (status, before) = get_json(&app, ferrox_api::routes::ADMIN_STATS).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(before["generating_now"], serde_json::json!(1));

        let (status, body) = post_json_uri(
            &app,
            ferrox_api::routes::V1_CANCEL,
            serde_json::json!({ "request_id": "chatcmpl-live" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["cancelled"], serde_json::json!(true));
        assert!(
            token.is_cancelled(),
            "the endpoint answered ok without setting the flag the decode loop reads"
        );
    }

    #[tokio::test]
    async fn tokenize_detokenize_roundtrip_and_embeddings_mean() {
        let app = test_app();
        let (status, tok) =
            post_json_uri(&app, "/v1/tokenize", serde_json::json!({ "prompt": "Hi" })).await;
        assert_eq!(status, StatusCode::OK);
        let tokens = tok["tokens"].as_array().unwrap();
        assert_eq!(tok["count"], tokens.len());
        assert!(!tokens.is_empty());

        let (status, detok) = post_json_uri(
            &app,
            "/v1/detokenize",
            serde_json::json!({ "tokens": tokens }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(detok["text"], "Hi");

        let (status, emb) = post_json_uri(
            &app,
            "/v1/embeddings",
            serde_json::json!({
                "input": "Hi",
                "embedding_type": "mean"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let vec = emb["data"][0]["embedding"].as_array().unwrap();
        assert!(!vec.is_empty());
        assert!(vec.iter().all(|v| v.as_f64().is_some()));
    }

    /// The /metrics endpoint must expose the bounded expert cache's
    /// counters when the model streams routed experts, and the
    /// counters must reflect real decode activity (a forward pass
    /// through store-backed MoE layers produces misses/hits).
    #[tokio::test]
    async fn metrics_exposes_expert_store_counters_when_streaming_is_active() {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let fixture = concat!(
            "../ferrox-models/tests/fixtures/",
            "ferrox_real_moe_test.gguf"
        );
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(fixture);
        let decoder = Decoder::from_gguf_with_expert_cache(
            &fixture,
            ferrox_models::config::test_moe_fixture(),
            Some(1024 * 1024),
        )
        .expect("MoE fixture must load store-backed");

        // Drive one real forward pass so the store sees decode
        // activity (the fixture's tiny vocab can't survive the HTTP
        // path's template text, so decode directly).
        let mut caches: Vec<ferrox_core::cache::KvCache> = decoder
            .layers
            .iter()
            .map(|_| {
                ferrox_core::cache::KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim)
            })
            .collect();
        decoder.forward_token(1, 0, &mut caches);

        let model = Model::Gguf(GgufModel {
            decoder: Arc::new(decoder),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: None,
            is_synthetic: false,
            chat_template: chat_template::ChatTemplate::Plain,
        });
        let state = Arc::new(test_state(
            model,
            ResponseCache::new(16, Duration::from_secs(60)),
        ));
        let app = Router::new()
            .route("/metrics", axum::routing::get(metrics))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);

        let fetch_metrics = |app: Router| async move {
            let resp = app
                .oneshot(
                    axum::http::Request::builder()
                        .method("GET")
                        .uri("/metrics")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            String::from_utf8(bytes.to_vec()).unwrap()
        };

        let after = fetch_metrics(app.clone()).await;
        assert!(
            after.contains("ferrox_expert_cache_misses_total"),
            "streaming model must expose expert-cache metrics: {after}"
        );
        let misses: u64 = after
            .lines()
            .find(|l| l.starts_with("ferrox_expert_cache_misses_total"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .expect("misses metric line must parse");
        assert!(
            misses > 0,
            "decode must have read experts through the store: {after}"
        );
    }

    fn weather_tool() -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        })
    }

    fn weather_tool_def() -> ToolDef {
        ToolDef {
            kind: "function".to_string(),
            function: ToolFunctionDef {
                name: "get_weather".to_string(),
                description: Some("Get the current weather for a location.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                })),
            },
        }
    }

    #[test]
    fn tool_preamble_mentions_every_tool_name_and_description() {
        let preamble = tool_preamble(&[weather_tool_def()]);
        assert!(preamble.contains("get_weather"));
        assert!(preamble.contains("Get the current weather for a location."));
        assert!(preamble.contains("<tool_call>"));
        assert!(preamble.contains("</tool_call>"));
    }

    #[test]
    fn extract_tool_call_parses_a_real_marker() {
        let text = "sure, let me check.<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}</tool_call>";
        let (name, arguments) = extract_tool_call(text).expect("must find the marker");
        assert_eq!(name, "get_weather");
        let parsed: serde_json::Value = serde_json::from_str(&arguments).unwrap();
        assert_eq!(parsed["location"], "Paris");
    }

    #[test]
    fn extract_tool_call_returns_none_when_no_marker_present() {
        assert_eq!(
            extract_tool_call("just a plain answer, no markers here"),
            None
        );
    }

    #[test]
    fn extract_tool_call_returns_none_on_malformed_json_inside_the_marker() {
        assert_eq!(
            extract_tool_call("<tool_call>not valid json at all</tool_call>"),
            None
        );
    }

    #[test]
    fn extract_tool_call_defaults_to_empty_arguments_when_the_field_is_absent() {
        let (name, arguments) =
            extract_tool_call("<tool_call>{\"name\": \"ping\"}</tool_call>").unwrap();
        assert_eq!(name, "ping");
        assert_eq!(arguments, "{}");
    }

    #[test]
    fn build_response_message_promotes_a_real_tool_call_to_tool_calls() {
        let (message, finish) = build_response_message(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Rome\"}}</tool_call>"
                .to_string(),
            true,
            "stop",
        );
        assert_eq!(finish, "tool_calls");
        assert!(message.content.is_none());
        let calls = message.tool_calls.expect("must carry a tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn build_response_message_falls_back_to_plain_text_when_tools_are_inactive() {
        let (message, finish) = build_response_message(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>".to_string(),
            false,
            "stop",
        );
        assert_eq!(finish, "stop");
        assert!(message.tool_calls.is_none());
        assert!(message.content.is_some());
    }

    #[test]
    fn build_response_message_falls_back_to_plain_text_when_no_marker_is_present() {
        let (message, finish) = build_response_message("just an answer".to_string(), true, "stop");
        assert_eq!(finish, "stop");
        assert!(message.tool_calls.is_none());
        assert_eq!(message.content.as_deref(), Some("just an answer"));
    }

    /// Zero-regression proof: an ordinary request with no `tools`/
    /// `session_id` produces the plain response shape -- `content` a
    /// string, no `tool_calls` field -- with an honest finish reason:
    /// this 4-token greedy request truncates at `max_tokens`, so
    /// `finish_reason` must be "length" (an earlier version hardcoded
    /// "stop" for every non-streaming response), and `usage` counts
    /// exactly the generated tokens.
    #[tokio::test]
    async fn a_request_with_no_tools_or_session_behaves_exactly_as_before() {
        let app = test_app();
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
            "max_tokens": 4,
            "temperature": 0,
        });
        let resp = post_json(&app, body).await;
        let message = &resp["choices"][0]["message"];
        assert!(message["content"].is_string());
        assert!(message.get("tool_calls").is_none());
        assert_eq!(resp["choices"][0]["finish_reason"], "length");
        assert_eq!(resp["usage"]["completion_tokens"], 4);
        assert_eq!(
            resp["usage"]["total_tokens"],
            resp["usage"]["prompt_tokens"].as_u64().unwrap() + 4
        );
    }

    async fn get_json(app: &Router, uri: &str) -> (StatusCode, serde_json::Value) {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn health_answers_a_capability_handshake_not_a_boolean() {
        let app = test_app();
        let (status, body) = get_json(&app, ferrox_api::routes::HEALTH).await;
        assert_eq!(status, StatusCode::OK);

        let health: ferrox_api::HealthResponse = serde_json::from_value(body).unwrap();
        assert_eq!(health.state, ferrox_api::HealthState::Ready);
        assert!(health.pid > 0);
        assert!(health.server_time_unix_ms > 0);
        // Nothing has been served yet: the field is absent rather than
        // claiming a request happened at time zero.
        assert_eq!(health.last_request_age_seconds, None);

        // Every control the UI might grey out has a code it can switch
        // on and a sentence it can show.
        for id in [
            ferrox_api::health::capability::CPU,
            ferrox_api::health::capability::METAL,
            ferrox_api::health::capability::CUDA,
            ferrox_api::health::capability::REAL_WEIGHTS,
            ferrox_api::health::capability::CONTINUOUS_BATCHING,
        ] {
            let cap = health
                .capability(id)
                .unwrap_or_else(|| panic!("{id} missing"));
            assert!(!cap.reason.is_empty(), "{cap:?}");
            assert!(!cap.detail.is_empty(), "{cap:?}");
        }
        // The test app serves synthetic random weights, and health must
        // say so: a UI that presents noise as a model invites a bug
        // report about "quality".
        let weights = health
            .capability(ferrox_api::health::capability::REAL_WEIGHTS)
            .unwrap();
        assert!(!weights.available);
        assert_eq!(weights.reason, ferrox_api::health::reason::MODEL_NOT_LOADED);
        assert!(health.model.as_ref().unwrap().synthetic_weights);
    }

    #[tokio::test]
    async fn health_vouches_for_liveness_after_a_request_has_been_served() {
        let app = test_app();
        let _ = post_json(
            &app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}"}],
                "max_tokens": 1,
                "temperature": 0,
            }),
        )
        .await;
        let (_status, body) = get_json(&app, ferrox_api::routes::HEALTH).await;
        let health: ferrox_api::HealthResponse = serde_json::from_value(body).unwrap();
        let age = health
            .last_request_age_seconds
            .expect("a served request is evidence of liveness");
        assert!((0.0..5.0).contains(&age), "implausible age {age}");
    }

    /// Every `data:` payload of an SSE response body, `[DONE]` excluded.
    async fn post_sse_chunks(app: &Router, body: serde_json::Value) -> Vec<serde_json::Value> {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|payload| *payload != "[DONE]")
            .map(|payload| serde_json::from_str(payload).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn a_stream_states_its_request_id_once_in_the_first_chunk() {
        let app = test_app();
        let chunks = post_sse_chunks(
            &app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": true,
            }),
        )
        .await;

        assert!(!chunks.is_empty());
        let request_id = chunks[0]["request_id"]
            .as_str()
            .expect("the first chunk names the request")
            .to_string();
        assert!(request_id.starts_with("chatcmpl-"), "{request_id}");
        // Once, and before any content: a client that reads the id from
        // chunk zero never has to correlate by heuristic.
        for (i, chunk) in chunks.iter().enumerate().skip(1) {
            assert!(
                chunk.get("request_id").is_none(),
                "chunk {i} repeats request_id"
            );
        }
        // Every chunk of one stream carries the same `id`, and it is
        // that request id -- not a shared constant.
        for chunk in &chunks {
            assert_eq!(chunk["id"], serde_json::json!(request_id));
        }

        let other = post_sse_chunks(
            &app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": true,
            }),
        )
        .await;
        assert_ne!(
            other[0]["request_id"].as_str().unwrap(),
            request_id,
            "two concurrent chats must not share an id"
        );
    }

    #[tokio::test]
    async fn a_non_streamed_response_names_the_same_request_id_as_its_completion_id() {
        let app = test_app();
        let resp = post_json(
            &app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "max_tokens": 2,
                "temperature": 0,
            }),
        )
        .await;
        assert_eq!(resp["id"], resp["request_id"]);
        assert!(resp["request_id"]
            .as_str()
            .unwrap()
            .starts_with("chatcmpl-"));
    }

    /// The whole point of server-reported timings: a client can tell
    /// prefill from decode without a stopwatch (see `ferrox_api::usage`).
    #[tokio::test]
    async fn usage_carries_separate_prefill_and_decode_timings() {
        let app = test_app();
        let resp = post_json(
            &app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "max_tokens": 4,
                "temperature": 0,
            }),
        )
        .await;
        let usage = &resp["usage"];
        assert!(usage["prompt_eval_duration_ms"].is_number(), "{usage}");
        assert!(usage["generation_duration_ms"].is_number(), "{usage}");
        assert!(usage["time_to_first_token_ms"].is_number(), "{usage}");
        assert!(usage["predicted_per_second"].is_number(), "{usage}");
        // No prefix cache in this app: the field must be absent, not 0.
        assert!(usage.get("cached_tokens").is_none(), "{usage}");
    }

    /// A real, deterministic small model with random weights will not
    /// spontaneously produce a `<tool_call>{...}</tool_call>` marker
    /// (whether a real deployed model does is a property of that
    /// model, not of ferrox's plumbing) -- so the real, testable
    /// end-to-end property here is that a `tools`-bearing request
    /// whose output does NOT contain the marker falls through cleanly
    /// to an ordinary text response instead of erroring or panicking.
    #[tokio::test]
    async fn a_tools_request_with_no_marker_in_the_output_falls_back_to_plain_content() {
        let app = test_app();
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
            "max_tokens": 4,
            "temperature": 0,
            "tools": [weather_tool()],
        });
        let resp = post_json(&app, body).await;
        let message = &resp["choices"][0]["message"];
        assert!(
            message["content"].is_string(),
            "must fall back to plain content when no real tool-call marker is present: {resp:?}"
        );
        assert!(message.get("tool_calls").is_none());
        // Truncated at max_tokens, so the honest finish reason is
        // "length" -- the point here is only that it is NOT
        // "tool_calls".
        assert_eq!(resp["choices"][0]["finish_reason"], "length");
    }

    /// A whole-response cache hit must be indistinguishable from
    /// recomputing: same content, same (honest) finish_reason, same
    /// usage counts -- only the `ferrox_cache` marker may differ.
    #[tokio::test]
    async fn a_cache_hit_reports_the_original_finish_reason_and_usage() {
        let app = test_app();
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "\u{1}\u{2}"}],
            "max_tokens": 3,
            "temperature": 0,
        });
        let first = post_json(&app, body.clone()).await;
        assert_eq!(first["ferrox_cache"], "miss");
        let second = post_json(&app, body).await;
        assert_eq!(second["ferrox_cache"], "hit");
        assert_eq!(
            first["choices"][0]["message"]["content"],
            second["choices"][0]["message"]["content"]
        );
        assert_eq!(
            first["choices"][0]["finish_reason"],
            second["choices"][0]["finish_reason"]
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(second["usage"]["completion_tokens"], 3);
    }

    /// The real proof for session reuse:
    /// a two-request session where the second request sends only its
    /// new message must produce exactly the same output as manually
    /// resending the full history (built from the *real* first reply,
    /// not an assumed one) with no `session_id` at all.
    #[tokio::test]
    async fn session_reuse_produces_the_same_output_as_manually_resending_full_history() {
        let session_app = test_app();
        let manual_app = test_app();

        // Turn 1, via session.
        let turn1 = post_json(
            &session_app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "session_id": "s1",
                "max_tokens": 5,
                "temperature": 0,
            }),
        )
        .await;
        let reply1 = turn1["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();

        // Turn 1, manually, for comparison -- must match exactly
        // (trivially, since it's the literal same single-turn
        // request), confirming the session path's first turn isn't
        // doing anything different from a plain request.
        let manual_turn1 = post_json(
            &manual_app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "max_tokens": 5,
                "temperature": 0,
            }),
        )
        .await;
        assert_eq!(
            manual_turn1["choices"][0]["message"]["content"]
                .as_str()
                .unwrap(),
            reply1
        );

        // Turn 2, via session: sends ONLY the new message.
        let turn2 = post_json(
            &session_app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{4}\u{5}"}],
                "session_id": "s1",
                "max_tokens": 5,
                "temperature": 0,
            }),
        )
        .await;
        let reply2 = turn2["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();

        // Turn 2, manually: the full three-message history
        // reconstructed using the REAL reply1 text, with no
        // session_id -- must produce byte-identical output.
        let manual_turn2 = post_json(
            &manual_app,
            serde_json::json!({
                "model": "m",
                "messages": [
                    {"role": "user", "content": "\u{1}\u{2}\u{3}"},
                    {"role": "assistant", "content": reply1},
                    {"role": "user", "content": "\u{4}\u{5}"},
                ],
                "max_tokens": 5,
                "temperature": 0,
            }),
        )
        .await;
        assert_eq!(
            manual_turn2["choices"][0]["message"]["content"]
                .as_str()
                .unwrap(),
            reply2,
            "resuming a session must produce identical output to manually resending the full history"
        );
    }

    /// `lock_cache` must return a usable guard even after the mutex was
    /// poisoned by a panic elsewhere.
    #[test]
    fn lock_cache_recovers_from_a_poisoned_mutex() {
        let cache = Arc::new(Mutex::new(ResponseCache::new(10, Duration::from_secs(60))));

        let poison_cache = Arc::clone(&cache);
        let _ = std::thread::spawn(move || {
            let _guard = poison_cache.lock().unwrap();
            panic!("simulated panic while holding the lock");
        })
        .join();

        // A plain `.lock().unwrap()` would panic here; lock_cache must not.
        let recovered = lock_cache(&cache);
        assert_eq!(recovered.stats().entries, 0);
    }

    #[test]
    fn is_cacheable_true_for_greedy_or_seeded_requests() {
        let mut req_body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let req: ChatCompletionRequest = serde_json::from_value(req_body.clone()).unwrap();
        assert!(
            req.is_cacheable(),
            "default (temperature 0) must be cacheable"
        );

        req_body["temperature"] = serde_json::json!(0.8);
        let req: ChatCompletionRequest = serde_json::from_value(req_body.clone()).unwrap();
        assert!(
            !req.is_cacheable(),
            "unseeded sampling must never be cacheable"
        );

        req_body["seed"] = serde_json::json!(42);
        let req: ChatCompletionRequest = serde_json::from_value(req_body).unwrap();
        assert!(
            req.is_cacheable(),
            "sampling with an explicit seed is deterministic and must be cacheable"
        );
    }

    #[test]
    fn stop_param_accepts_both_single_string_and_array() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "END",
        }))
        .unwrap();
        assert_eq!(req.stop_sequences(), vec!["END".to_string()]);

        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["A", "B"],
        }))
        .unwrap();
        assert_eq!(req.stop_sequences(), vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn run_generation_rejects_out_of_vocab_tokens_instead_of_panicking() {
        let model = test_model();
        let result = run_generation(&model, "hello", &greedy_params(4), None, None, None);
        assert!(matches!(
            result,
            Err(generate::DecodeError::TokenOutOfVocab { .. })
        ));
    }

    #[test]
    fn run_generation_honors_an_exhausted_kv_pool_and_maps_it_to_a_503() {
        let model = test_model(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(ferrox_core::cache::KvBlockPool::new(64, 1)));
        let config = generate::KvPoolConfig {
            pool,
            queue_wait: Duration::ZERO,
        };

        let result = run_generation(
            &model,
            &prompt,
            &greedy_params(4),
            Some(&config),
            None,
            None,
        );
        assert!(matches!(
            result,
            Err(generate::DecodeError::KvPoolExhausted)
        ));

        let (status, _body) = decode_error_response(result.unwrap_err());
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A full admission queue is the server being behind, not the
    /// client being wrong: 503, with the wait hint in the body (and the
    /// `Retry-After` header stamped by `limits::retry_after`) and the
    /// depth and cap named so an operator can tell a retry storm from a
    /// single oversized request.
    #[test]
    fn decode_error_response_maps_a_full_queue_to_a_retryable_503() {
        let (status, Json(body)) = decode_error_response(generate::DecodeError::QueueFull {
            queued: 512,
            cap: 512,
        });
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["retry_after_seconds"], 1);
        let message = body["error"]["message"].as_str().expect("message");
        assert!(message.contains("512"), "{message}");
    }

    #[test]
    fn decode_error_response_omits_a_retry_hint_for_an_unretryable_error() {
        let (_status, Json(body)) = decode_error_response(generate::DecodeError::TokenOutOfVocab {
            token: 99,
            vocab_size: 32,
        });
        assert!(
            body["error"]["retry_after_seconds"].is_null(),
            "retrying a prompt this model cannot tokenize never helps"
        );
    }

    #[test]
    fn decode_error_response_maps_token_out_of_vocab_to_bad_request() {
        let (status, _body) = decode_error_response(generate::DecodeError::TokenOutOfVocab {
            token: 99,
            vocab_size: 32,
        });
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn run_generation_succeeds_and_releases_blocks_when_the_pool_has_room() {
        let model = test_model(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(ferrox_core::cache::KvBlockPool::new(64, 2)));
        let config = generate::KvPoolConfig {
            pool: pool.clone(),
            queue_wait: Duration::ZERO,
        };

        let (_, finish, _usage) = run_generation(
            &model,
            &prompt,
            &greedy_params(4),
            Some(&config),
            None,
            None,
        )
        .unwrap();
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            2,
            "a completed request must return its blocks to the pool"
        );
    }

    /// The core concurrency claim: two requests using the *same* `Arc<Model>`
    /// must be able to run their (independent, per-call) KV caches
    /// concurrently without interfering with each other or needing any
    /// shared lock around the model itself.
    #[tokio::test]
    async fn concurrent_requests_against_the_same_model_do_not_interfere() {
        let model = Arc::new(test_model());
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let model = Arc::clone(&model);
            let prompt = prompt.clone();
            handles.push(tokio::task::spawn_blocking(move || {
                run_generation(&model, &prompt, &greedy_params(6), None, None, None).unwrap()
            }));
        }

        let mut results = Vec::new();
        for h in handles {
            results.push(h.await.unwrap());
        }
        // Same prompt, same seed, same (greedy) sampling, same
        // immutable model -> every concurrent run must produce
        // identical output, proving no request's KV cache leaked into
        // another's.
        for r in &results[1..] {
            assert_eq!(r.0, results[0].0, "decoded chunks must match");
            assert_eq!(r.1, results[0].1, "finish reason must match");
            assert_eq!(
                r.2.prompt_tokens, results[0].2.prompt_tokens,
                "prompt token count must match"
            );
            assert_eq!(
                r.2.completion_tokens, results[0].2.completion_tokens,
                "completion token count must match"
            );
        }
    }

    /// A real, minimal safetensors shard: JSON header (name -> real
    /// dtype/shape/`data_offsets`) followed by the concatenated raw
    /// F32 bytes -- exactly the format `ShardedSafetensors::open_index`
    /// parses, hand-built here rather than depending on
    /// `ferrox-models::kimi_loader`'s own private test helpers (not
    /// visible across the crate boundary).
    fn write_safetensors_shard(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header_entries = Vec::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            let end = data.len();
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header_entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_str}],\"data_offsets\":[{start},{end}]}}"
            ));
        }
        let header = format!("{{{}}}", header_entries.join(","));
        let header_bytes = header.as_bytes();
        let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(&data);
        out
    }

    /// Builds a small but completely real Kimi K3 checkpoint directory
    /// on disk (real `model.safetensors.index.json` + shard bytes +
    /// `tiktoken.model`, the exact file layout `ferrox-cli`'s
    /// `run-kimi` command expects) and loads it through
    /// `model::load_kimi_checkpoint_with_config` (the same real loading
    /// logic `model::load()` uses for `FERROX_MODEL_PATH` pointing at a
    /// directory, parametrized here only so the checkpoint can be small
    /// -- see that function's doc comment). Shared by every test that
    /// needs a real, loaded `KimiLoaded` rather than duplicating this
    /// setup per test.
    fn build_synthetic_kimi_loaded() -> model::KimiLoaded {
        use ferrox_models::config::{AttentionKind, KdaConfig, KimiHybridAttention, MlaConfig};
        use ferrox_models::kimi_loader::KimiRealHparams;
        use ferrox_moe::{GatingFunction, MoeLayerConfig};

        let hidden_dim = 8;
        let kda_num_heads = 2;
        let kda_head_dim = 3;
        let kda_proj = kda_num_heads * kda_head_dim;
        let conv_kernel = 4;
        let dense_intermediate = 5;
        // One token per byte value -- enough to round-trip a simple
        // ASCII prompt through the real tiktoken-format vocab below,
        // matching `kimi_generate`'s own test convention.
        let vocab_size = 256;
        let mla_num_heads = 1;
        let mla_q_lora_rank = 2;
        let mla_kv_lora_rank = 2;
        let mla_qk_nope_head_dim = 2;
        let mla_qk_rope_head_dim = 2;
        let mla_v_head_dim = 2;

        let model_cfg = ferrox_models::ModelConfig {
            name: "synthetic-kimi-server-test",
            n_layers: 1,
            hidden_dim,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
            vocab_size,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: None,
            moe: MoeLayerConfig {
                expert_weights_scale: 1.0,
                n_experts: 1,
                n_experts_active: 1,
                n_shared_experts: 0,
                hidden_dim,
                expert_ffn_dim: 4,
                gating: GatingFunction::Sigmoid,
                norm_topk_prob: true,
                expert_group_count: None,
                expert_group_used_count: None,
            },
            // Layer 0 is the sole dense leading layer, using KDA
            // attention (real Kimi K3's own layer-0 shape) -- the
            // 1-indexed `kda_layers`/`full_attn_layers` convention is
            // `ModelConfig::layer_attention_kind`'s, not this test's.
            n_dense_leading_layers: 1,
            attention: AttentionKind::KimiHybrid(KimiHybridAttention {
                kda_layers: vec![1],
                full_attn_layers: vec![],
                mla: MlaConfig {
                    num_heads: mla_num_heads,
                    q_lora_rank: mla_q_lora_rank,
                    kv_lora_rank: mla_kv_lora_rank,
                    qk_nope_head_dim: mla_qk_nope_head_dim,
                    qk_rope_head_dim: mla_qk_rope_head_dim,
                    v_head_dim: mla_v_head_dim,
                    use_output_gate: true,
                    rope: None,
                },
                kda: KdaConfig {
                    num_heads: kda_num_heads,
                    head_dim: kda_head_dim,
                    short_conv_kernel_size: conv_kernel,
                    gate_lower_bound: -5.0,
                    use_full_rank_gate: true,
                },
            }),
            rope_freqs: None,
            rope_attn_factor: 1.0,
            rope_dim: None,
            rope_freqs_long: None,
            rope_freqs_short: None,
            rope_orig_ctx: None,
            rope_layout: ferrox_models::config::RopeLayout::Neox,
            qk_norm_style: ferrox_models::capability::QkNormStyle::WholeVector,
            swa_pattern: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            embedding_scale: None,
            attention_scale: None,
            rope_theta_swa: None,
            ffn_activation: ferrox_models::config::FfnActivation::Swiglu,
            best_effort_fields: &["synthetic test config, not a real preset"],
        };
        let hp = KimiRealHparams {
            hidden_dim,
            kda_num_heads,
            kda_head_dim,
            mla_num_heads,
            mla_q_lora_rank,
            mla_kv_lora_rank,
            mla_qk_nope_head_dim,
            mla_qk_rope_head_dim,
            mla_v_head_dim,
            dense_intermediate_dim: dense_intermediate,
            moe_hidden_dim: hidden_dim,
            moe_intermediate_dim: 4,
            n_experts: 1,
            num_shared_experts: 0,
        };

        // Every real tensor name `kimi_loader::load_kimi_layer` (dense
        // FFN + KDA attention + block residual) and
        // `load_kimi_checkpoint` (top-level) actually read.
        let prefix = "language_model.model.layers.0";
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let push = |tensors: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
                    name: String,
                    shape: Vec<usize>,
                    n: usize| {
            tensors.push((name, shape, vec![0.01f32; n]));
        };
        push(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attention_res_norm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attention_res_proj.weight"),
            vec![1, hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp_res_norm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp_res_proj.weight"),
            vec![1, hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.q_conv1d.weight"),
            vec![kda_proj, 1, conv_kernel],
            kda_proj * conv_kernel,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.k_conv1d.weight"),
            vec![kda_proj, 1, conv_kernel],
            kda_proj * conv_kernel,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.v_conv1d.weight"),
            vec![kda_proj, 1, conv_kernel],
            kda_proj * conv_kernel,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.A_log"),
            vec![kda_num_heads],
            kda_num_heads,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.f_a_proj.weight"),
            vec![kda_head_dim, hidden_dim],
            kda_head_dim * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.f_b_proj.weight"),
            vec![kda_proj, kda_head_dim],
            kda_proj * kda_head_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.dt_bias"),
            vec![kda_proj],
            kda_proj,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.b_proj.weight"),
            vec![kda_num_heads, hidden_dim],
            kda_num_heads * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.g_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.o_norm.weight"),
            vec![kda_head_dim],
            kda_head_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![hidden_dim, kda_proj],
            hidden_dim * kda_proj,
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![dense_intermediate, hidden_dim],
            dense_intermediate * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.up_proj.weight"),
            vec![dense_intermediate, hidden_dim],
            dense_intermediate * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.down_proj.weight"),
            vec![hidden_dim, dense_intermediate],
            hidden_dim * dense_intermediate,
        );
        push(
            &mut tensors,
            "language_model.model.embed_tokens.weight".to_string(),
            vec![vocab_size, hidden_dim],
            vocab_size * hidden_dim,
        );
        push(
            &mut tensors,
            "language_model.lm_head.weight".to_string(),
            vec![vocab_size, hidden_dim],
            vocab_size * hidden_dim,
        );
        push(
            &mut tensors,
            "language_model.model.norm.weight".to_string(),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            "language_model.model.output_attn_res_norm.weight".to_string(),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            "language_model.model.output_attn_res_proj.weight".to_string(),
            vec![1, hidden_dim],
            hidden_dim,
        );

        let dir = std::env::temp_dir().join(format!(
            "ferrox_server_kimi_e2e_test_{}_{}",
            std::process::id(),
            vocab_size
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let shard_bytes = write_safetensors_shard(&tensors);
        std::fs::write(dir.join("shard0.safetensors"), &shard_bytes).unwrap();
        let map_entries: Vec<String> = tensors
            .iter()
            .map(|(name, ..)| format!("\"{name}\":\"shard0.safetensors\""))
            .collect();
        let index = format!("{{\"weight_map\":{{{}}}}}", map_entries.join(","));
        std::fs::write(dir.join("model.safetensors.index.json"), &index).unwrap();

        // A real tiktoken-format vocab file: one base64-encoded byte
        // plus its rank per line -- enough to round-trip an ASCII
        // prompt without needing the real 163584-entry Kimi K3 vocab.
        use base64::Engine;
        let vocab_lines: Vec<String> = (0..vocab_size as u32)
            .map(|b| {
                let b64 = base64::engine::general_purpose::STANDARD.encode([b as u8]);
                format!("{b64} {b}")
            })
            .collect();
        std::fs::write(dir.join("tiktoken.model"), vocab_lines.join("\n")).unwrap();

        let loaded = model::load_kimi_checkpoint_with_config(dir.to_str().unwrap(), model_cfg, hp)
            .expect("must load the synthetic Kimi checkpoint end to end");
        std::fs::remove_dir_all(&dir).ok();
        loaded
    }

    /// The real end-to-end proof for Kimi-through-the-server: a real
    /// synthetic Kimi K3 checkpoint served through the exact same
    /// `run_generation` entry point the HTTP handlers call for the
    /// GGUF path. Proves the whole new plumbing end to end: directory-
    /// shaped checkpoint loading, `KimiEngine`/`KimiTokenizer` wired
    /// through the `Model` enum, and `generate::generate_engine`
    /// producing real, bounded generated text.
    #[test]
    fn kimi_model_serves_real_text_end_to_end_via_run_generation() {
        let loaded = build_synthetic_kimi_loaded();
        let state = build_app_state(
            model::LoadedModel::Kimi(loaded),
            None,
            None,
            false,
            None,
            Arc::new(health::Detection::ready(health::probe_backends())),
        );
        let active = state.active().expect("a freshly built state has a model");
        assert_eq!(active.model.tokenizer_kind(), "kimi-tiktoken-bpe");
        assert!(!active.model.is_synthetic());

        let (_chunks, finish, _usage) =
            run_generation(&active.model, "hi", &greedy_params(5), None, None, None)
                .expect("a real Kimi checkpoint must generate without error");
        assert!(matches!(finish, FinishReason::Length | FinishReason::Stop));
    }

    /// Explicit proof of the "gate, don't paper over" design decision
    /// (see `ferrox_models::engine`'s module docs): even when an operator configures
    /// a KV block pool and/or prefix cache, a Kimi request must never
    /// consult either -- `generate_engine`'s signature has no
    /// parameter for them at all, so this isn't just an unexercised
    /// code path, it's structurally impossible for a Kimi request to
    /// touch them. Confirmed here by observing both are completely
    /// untouched (pool blocks unchanged, cache stats unchanged) after a
    /// real Kimi generation runs alongside both.
    #[test]
    fn kv_pool_and_prefix_cache_are_never_consulted_for_a_kimi_model() {
        let loaded = build_synthetic_kimi_loaded();
        let state = build_app_state(
            model::LoadedModel::Kimi(loaded),
            None,
            None,
            false,
            None,
            Arc::new(health::Detection::ready(health::probe_backends())),
        );

        let pool = Arc::new(Mutex::new(ferrox_core::cache::KvBlockPool::new(64, 4)));
        let kv_pool_config = generate::KvPoolConfig {
            pool: pool.clone(),
            queue_wait: Duration::ZERO,
        };
        let pc = Mutex::new(PrefixCache::new(4));

        run_generation(
            &state
                .active()
                .expect("a freshly built state has a model")
                .model,
            "hi",
            &greedy_params(5),
            Some(&kv_pool_config),
            Some(&pc),
            None,
        )
        .expect("a real Kimi checkpoint must generate without error");

        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            4,
            "the KV pool must be completely untouched by a Kimi request"
        );
        let stats = pc.lock().unwrap().stats();
        assert_eq!(
            stats.hits + stats.misses,
            0,
            "the prefix cache must never be consulted for a Kimi request"
        );
    }
}
