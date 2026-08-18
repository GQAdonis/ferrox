//! ferrox CLI — llama.cpp-style GGUF completion (`-m`/`-p`/`-n`/…) plus
//! inspect / presets / smoke / Kimi helpers. See `docs/CLI.md`.

mod bench_model;
mod bench_suite;
mod chat;
mod parity;
mod pull;
mod run;
mod verify;
mod verify_engine;

use clap::{Parser, Subcommand};

use ferrox_core::cache::KvCache;
use ferrox_gguf::ShardedGguf;
use ferrox_models::{
    config::test_dense_fixture, deepseek_v4_pro, glm_5_2, kimi_k3, Decoder, ModelConfig,
};

#[derive(Parser)]
#[command(name = "ferrox", version, about = "Pure-Rust MoE inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// GGUF completion (llama.cpp-style `-m`/`-p`/`-n`/…).
    ///
    /// Also accepts top-level flags: `ferrox -m model.gguf -p "Hi" -n 64`.
    Run(run::InferArgs),
    /// Multi-turn chat REPL against a running `ferrox-server` (HTTP).
    ///
    /// Reuses the server's chat-template + streaming path — start the
    /// server first (`FERROX_MODEL_PATH=… ferrox-server`).
    Chat(chat::ChatArgs),
    /// Download a GGUF from Hugging Face Hub (`hf download`).
    Pull(pull::PullArgs),
    /// Print GGUF header metadata and tensor list for a model file.
    Inspect { path: String },
    /// Dry-run residency plan for a GGUF checkpoint: what it would
    /// cost to run (dense weights, routed experts resident vs.
    /// streamed, KV caches) against the selected backend's memory
    /// budget -- computed from the header alone, nothing loaded. Always
    /// reports the largest context that fits and the arithmetic behind
    /// it. --strict exits non-zero when the plan overcommits.
    InspectPlan {
        path: String,
        /// Context length each request's KV cache is sized for.
        /// Omit to price the model's own `{arch}.context_length`.
        #[arg(long)]
        context: Option<usize>,
        /// Concurrent requests to budget KV caches for.
        #[arg(long, default_value_t = 1)]
        concurrency: usize,
        /// Stream routed experts through a bounded cache of this many
        /// bytes instead of counting them fully resident.
        #[arg(long)]
        expert_cache_bytes: Option<u64>,
        /// Which backend's memory budget to plan against.
        #[arg(long, default_value = "cpu")]
        backend: ferrox_models::BudgetBackend,
        /// KV cache dtype the plan should price (`--ctk` analogue).
        /// Only the Metal path has a device KV whose dtype this
        /// selects; on CPU the host cache is always f32.
        #[arg(long, default_value = "f32")]
        ctk: String,
        /// Refuse (exit 1) when the plan exceeds the usable budget.
        #[arg(long)]
        strict: bool,
    },
    /// List the built-in architecture presets and their headline stats.
    Presets,
    /// Print the GGUF architecture coverage manifest (llama.cpp inventory
    /// classified into Ferrox families / scope). Pass `--write PATH` to
    /// regenerate `docs/manifests/architecture_manifest.md`.
    Archs {
        #[arg(long)]
        write: Option<String>,
    },
    /// Probe and print this host's hardware capabilities (CPU cores,
    /// RAM, SIMD flags, and CUDA device info if built with --features
    /// cuda). See ferrox-cuda's docs for exactly what is and isn't
    /// verified about the CUDA fields.
    Caps,
    /// Run a tiny synthetic forward-pass smoke test for a preset
    /// (random weights, small dims) to prove the pipeline executes.
    Smoke {
        /// One of: glm-5.2, deepseek-v4-pro, kimi-k3
        #[arg(default_value = "glm-5.2")]
        preset: String,
        #[arg(long, default_value_t = 8)]
        steps: usize,
    },
    /// Load REAL weights from a GGUF file (test-dense or test-moe
    /// fixture) and print logits for a single decode step, for
    /// cross-validation against the Python reference implementation.
    RunReal {
        path: String,
        #[arg(long, default_value_t = 0)]
        token: usize,
        #[arg(long, default_value_t = 0)]
        pos: usize,
        /// "dense" (single-expert test fixture) or "moe" (4-expert
        /// test fixture)
        #[arg(long, default_value = "dense")]
        fixture: String,
    },
    /// Benchmark: fused quantized matvec vs. dequant-then-matmul, at a
    /// realistic single-expert-FFN matrix size, measuring both wall
    /// time and resident memory. Also benchmarks end-to-end decode
    /// throughput (tokens/sec) for each preset at synthetic-but-full-
    /// scale-shape dimensions.
    /// Check that a GPU backend agrees with the CPU reference.
    ///
    /// Greedy-decodes the same prompt on both and compares token ids.
    /// The benchmark harness measures throughput and never inspects
    /// output, which has let wrong-kernel bugs sit behind green rows.
    Verify {
        /// GGUF to check.
        #[arg(short = 'm', long)]
        model: String,
        /// Backend to compare against the CPU reference.
        #[arg(long, default_value = "metal")]
        backend: String,
        /// Internal: print token ids for one backend and exit.
        #[arg(long, hide = true)]
        emit: bool,
        /// Stretch the prompt to this many tokens (repeating it) before
        /// prefill. Under 8 tokens the batched-prefill attention kernels
        /// never run, so the default prompt checks decode only.
        #[arg(long)]
        prompt_tokens: Option<usize>,
        /// Prompt to compare on. Defaults to a fixed short one so runs
        /// are comparable across models.
        #[arg(short = 'p', long)]
        prompt: Option<String>,
    },
    /// Check that ferrox agrees with llama.cpp on the first-token
    /// distribution.
    ///
    /// `verify` compares ferrox-CPU against ferrox-Metal, so both can be
    /// wrong together. This feeds the same token ids to llama.cpp's own
    /// library and compares the logit distributions — not greedy text,
    /// which cannot separate a wrong graph from two near-tied logits
    /// swapping.
    Parity {
        /// GGUF to check.
        #[arg(short = 'm', long)]
        model: String,
        /// Prompt to compare on. Defaults to a fixed short one so runs
        /// are comparable across models and sessions.
        #[arg(short = 'p', long)]
        prompt: Option<String>,
        /// Stretch the prompt to this many tokens before prefill, so the
        /// batched-prefill kernels are actually reached.
        #[arg(long)]
        prompt_tokens: Option<usize>,
        /// How many top tokens to intersect between the two engines.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// Compiled reference dumper (see .local-scripts/llama_logits.c).
        #[arg(long)]
        dumper: Option<String>,
    },
    Bench {
        /// Real GGUF to benchmark. With this set, `bench` becomes a
        /// `llama-bench` work-alike (pp/tg on real weights) and the
        /// synthetic matvec microbenchmark below is skipped.
        #[arg(short = 'm', long)]
        model: Option<String>,
        /// Prompt tokens for the `pp<N>` batched-prefill row (`0` skips).
        #[arg(short = 'p', long = "n-prompt", default_value_t = 512)]
        n_prompt: usize,
        /// Decode steps for the `tg<N>` row (`0` skips).
        #[arg(short = 'n', long = "n-gen", default_value_t = 128)]
        n_gen: usize,
        /// Timed repetitions per row; one extra warmup is discarded.
        #[arg(short = 'r', long = "repetitions", default_value_t = 3)]
        reps: usize,
        /// CPU threads (`0` = performance-core default)
        #[arg(short = 't', long, default_value_t = 0)]
        threads: usize,
        /// GPU layers: `0` forces CPU, anything else offloads
        #[arg(long = "n-gpu-layers", default_value_t = 0)]
        n_gpu_layers: usize,
        /// Context size (`0` = GGUF default)
        #[arg(short = 'c', long, default_value_t = 0)]
        ctx_size: usize,
        #[arg(long, default_value_t = 4096)]
        hidden: usize,
        #[arg(long, default_value_t = 14336)]
        ffn_dim: usize,
        #[arg(long, default_value_t = 20)]
        iters: usize,
        /// Also run `llama-bench` on the same GGUF and print the gap.
        #[arg(long)]
        compare: bool,
        /// Run every entry in `benchmarks/suite.json` (each in a fresh
        /// child process), write engine receipts, then re-render the
        /// engine table in `benchmarks/RESULTS.md`.
        #[arg(long)]
        suite: bool,
        /// Re-render the engine table from existing receipts only.
        #[arg(long)]
        render: bool,
        /// Restrict `--suite` to one suite id.
        #[arg(long)]
        id: Option<String>,
        /// Restrict `--suite` to one backend.
        #[arg(long)]
        backend: Option<String>,
        /// Skip suite entries whose estimated RAM exceeds ~75% of host RAM.
        #[arg(long)]
        fit_host: bool,
        /// Skip suite entries whose GGUF is not present.
        #[arg(long)]
        skip_missing: bool,
        /// benchmarks/ directory.
        #[arg(long, default_value = "benchmarks")]
        bench_dir: String,
        /// Internal (`--suite` children): suite id to record in the receipt.
        #[arg(long)]
        suite_id: Option<String>,
        /// Internal (`--suite` children): backend label for the receipt.
        #[arg(long, default_value = "cpu")]
        backend_label: String,
        /// Write a JSON receipt to this path.
        #[arg(long)]
        receipt: Option<String>,
    },
    /// Demonstrate prompt-lookup speculative decoding on a repetitive
    /// prompt, reporting forward_batch call count vs. tokens produced.
    /// Uses synthetic (random) weights -- see the printed caveat about
    /// why hit rate is not representative of a real trained model.
    Speculative {
        #[arg(default_value = "glm-5.2")]
        preset: String,
        #[arg(
            long,
            default_value = "the cat sat on the mat. the cat sat on the mat. the cat sat on the"
        )]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 4)]
        ngram_size: usize,
        #[arg(long, default_value_t = 8)]
        max_draft_len: usize,
    },
    /// Load a real Kimi K3 checkpoint directory (a real
    /// `model.safetensors.index.json` + shards, plus a real
    /// `tiktoken.model` and optionally `tokenizer_config.json`) and
    /// generate real text from a prompt. This is the only command that
    /// runs Kimi K3's real, complete inference path end to end --
    /// loading (`kimi_loader::load_kimi_checkpoint`), the dedicated
    /// forward pass (`kimi_decoder`), the real tokenizer
    /// (`kimi_tokenizer`), and sampling all wired together. Not
    /// runnable against the actual published checkpoint in this
    /// project's development environment (96 shards, 1.56TB even
    /// MXFP4-compressed; see docs/MODELS.md) -- but real
    /// code, tested end to end against small synthetic checkpoints, for
    /// anyone with the real hardware/storage to point it at real
    /// weights.
    RunKimi {
        /// Directory containing model.safetensors.index.json, its
        /// shard files, tiktoken.model, and (optionally)
        /// tokenizer_config.json.
        checkpoint_dir: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        #[arg(long, default_value_t = 1.0)]
        repetition_penalty: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

fn preset_by_name(name: &str) -> anyhow::Result<ModelConfig> {
    match name {
        "glm-5.2" | "glm5.2" | "glm" => Ok(glm_5_2()),
        "deepseek-v4-pro" | "deepseek" | "dsv4" => Ok(deepseek_v4_pro()),
        "kimi-k3" | "kimi" => Ok(kimi_k3()),
        other => {
            anyhow::bail!("unknown preset '{other}'; expected glm-5.2, deepseek-v4-pro, or kimi-k3")
        }
    }
}

/// Seeds `RAYON_NUM_THREADS` for subcommands that never reach
/// `apply_backend_env` / `bench_model::apply_env` (both of which build
/// the pool explicitly via [`ferrox_core::threads::init_cpu_pool`] once
/// their `-t` is known). Subcommand flags parsed later still win, since
/// those paths overwrite the variable before the pool is built.
///
/// This used to be `available_parallelism() / 2`, which on a 6P+4E M2
/// Pro guessed 5 -- close enough to look right, wrong enough to make
/// every default-config CPU measurement run one core short.
fn init_rayon_threads() {
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        let n = ferrox_core::threads::resolve_cpu_threads();
        // SAFETY: single-threaded init before worker threads spawn.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", n.to_string()) };
    }
}

/// Rewrite `ferrox -m …` into `ferrox run -m …` so llama.cpp-style
/// top-level flags work without typing the `run` subcommand.
fn rewrite_llama_style_argv(args: Vec<String>) -> Vec<String> {
    let args: Vec<String> = args
        .into_iter()
        .map(|arg| match arg.as_str() {
            "-ngl" => "--n-gpu-layers".into(),
            "-dev" => "--device".into(),
            _ => arg,
        })
        .collect();
    if args.len() < 2 {
        return args;
    }
    const SUBCOMMANDS: &[&str] = &[
        "run",
        "pull",
        "chat",
        "inspect",
        "inspect-plan",
        "presets",
        "archs",
        "caps",
        "smoke",
        "run-real",
        "bench",
        "verify",
        "parity",
        "speculative",
        "run-kimi",
        "help",
    ];
    let first = args[1].as_str();
    if SUBCOMMANDS.contains(&first)
        || first == "-h"
        || first == "--help"
        || first == "-V"
        || first == "--version"
    {
        return args;
    }
    let mut out = Vec::with_capacity(args.len() + 1);
    out.push(args[0].clone());
    out.push("run".into());
    out.extend(args.into_iter().skip(1));
    out
}

fn main() -> anyhow::Result<()> {
    init_rayon_threads();
    tracing_subscriber::fmt::init();
    let cli = Cli::parse_from(rewrite_llama_style_argv(std::env::args().collect()));

    match cli.command {
        Commands::Run(args) => run::run_infer(args)?,
        Commands::Chat(args) => chat::run_chat(args)?,
        Commands::Pull(args) => pull::run_pull(args)?,
        Commands::Inspect { path } => {
            let file = ShardedGguf::open(&path)?;
            if file.shard_count() > 1 {
                println!("Split GGUF: {} shards", file.shard_count());
                for p in file.shard_paths() {
                    println!("  {}", p.display());
                }
            }
            if let Some(name) = file.metadata_str("general.name") {
                println!("Model name: {name}");
            }
            println!("Tensor count: {}", file.tensor_count());
            for (i, (_, t)) in file.tensors().enumerate() {
                if i >= 20 {
                    println!("  ... and {} more", file.tensor_count() - 20);
                    break;
                }
                println!("  {:<40} {:?} {:?}", t.name, t.shape, t.dtype);
            }
        }
        Commands::InspectPlan {
            path,
            context,
            concurrency,
            expert_cache_bytes,
            backend,
            ctk,
            strict,
        } => {
            let budget = ferrox_models::DeviceBudget::detect(backend);
            let ctx_cap = ferrox_gguf::ShardedGguf::open(&path)
                .ok()
                .and_then(|f| {
                    let arch = f.metadata_str("general.architecture")?.to_string();
                    f.metadata_u64(&format!("{arch}.context_length"))
                })
                .unwrap_or(4096) as usize;
            let assumptions = ferrox_models::residency_report::ResidencyAssumptions {
                // `auto` still needs *a* context to price the plan's KV
                // line; the model's own is the honest placeholder, and
                // the chosen number is reported separately below.
                context_tokens: context.unwrap_or(ctx_cap),
                concurrent_requests: concurrency,
                expert_cache_bytes,
                kv_elem: ferrox_models::KvElem::from_ctk(&ctk),
                ..Default::default()
            };
            let report = ferrox_models::residency_report::ResidencyReport::from_gguf(
                &path,
                assumptions,
                budget.usable_bytes,
            )?;
            println!("{budget}");
            println!("{report}");
            let fit = report.auto_context(ctx_cap);
            println!("  {fit}");
            println!("  {}", budget.caveat());
            if strict {
                if let Err(e) = report.check_strict() {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Presets => {
            for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
                println!("{}", cfg.name);
                println!(
                    "  layers={} hidden={} heads={}/{} (q/kv)",
                    cfg.n_layers, cfg.hidden_dim, cfg.n_heads, cfg.n_kv_heads
                );
                println!(
                    "  experts: {} total, {} active/token, {} shared",
                    cfg.moe.n_experts, cfg.moe.n_experts_active, cfg.moe.n_shared_experts
                );
                println!(
                    "  best-effort / unconfirmed fields: {:?}",
                    cfg.best_effort_fields
                );
                println!();
            }
        }
        Commands::Archs { write } => {
            let report = ferrox_models::coverage_report_markdown();
            if let Some(path) = write {
                std::fs::write(&path, &report)?;
                println!("wrote architecture coverage manifest to {path}");
            } else {
                print!("{report}");
            }
        }
        Commands::Caps => {
            let profile = ferrox_cuda::HardwareProfile::detect();
            println!("Hardware capabilities (detected, not assumed):");
            println!("  CPU logical cores : {}", profile.cpu_logical_cores);
            if profile.host_ram_total_bytes > 0 {
                println!(
                    "  Host RAM total    : {:.2} GiB",
                    profile.host_ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            } else {
                println!("  Host RAM total    : could not detect (non-Linux host?)");
            }
            println!("  SIMD              : {}", profile.simd.label());
            println!(
                "    avx2={} avx512f={} fma={} neon={}",
                profile.simd.avx2, profile.simd.avx512f, profile.simd.fma, profile.simd.neon
            );
            if profile.cuda_available {
                println!(
                    "  CUDA              : {} device(s), first = {:?}, {:.2} GiB VRAM",
                    profile.cuda_device_count,
                    profile.cuda_device_name,
                    profile.cuda_vram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            } else {
                println!("  CUDA              : not available (no device found, or built without --features cuda)");
            }
            let metal = ferrox_metal::MetalProfile::detect();
            if metal.available {
                println!("  Metal             : {:?}", metal.device_name);
            } else {
                println!("  Metal             : not available (no device found, or built without --features metal)");
            }
        }
        Commands::Smoke { preset, steps } => {
            let base_cfg = preset_by_name(&preset)?;
            println!(
                "Running smoke test for '{}' (small synthetic weights, {} steps)",
                base_cfg.name, steps
            );

            let mut cfg = base_cfg;
            cfg.hidden_dim = 32;
            cfg.n_heads = 4;
            cfg.n_kv_heads = 2;
            cfg.head_dim = 8;
            cfg.moe.hidden_dim = 32;
            cfg.moe.n_experts = cfg.moe.n_experts.min(16);
            cfg.moe.expert_ffn_dim = 16;

            let vocab = 32;
            let decoder = Decoder::new_random_small(cfg.clone(), 2, vocab);
            let mut caches: Vec<KvCache> = (0..2)
                .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
                .collect();

            for pos in 0..steps {
                let token = pos % vocab;
                let logits = decoder.forward_token(token, pos, &mut caches);
                let argmax = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                let all_finite = logits.iter().all(|v| v.is_finite());
                println!("  step {pos}: token_in={token} argmax_out={argmax} finite={all_finite}");
                if !all_finite {
                    anyhow::bail!("non-finite logits at step {pos}");
                }
            }
            println!("Smoke test passed: {steps} decode steps, all logits finite.");

            // Real expert-placement plan, driven by each expert's actual
            // resident byte size and the per-expert activation counts
            // just observed above -- not a synthetic example. Falls back
            // to a documented placeholder budget when no CUDA VRAM was
            // detected (e.g. this host, or built without --features cuda),
            // so the feature is still visible without a GPU present.
            let profile = ferrox_cuda::HardwareProfile::detect();
            let (budget_bytes, budget_note) = if profile.cuda_vram_total_bytes > 0 {
                (
                    profile.cuda_vram_total_bytes,
                    "detected CUDA VRAM".to_string(),
                )
            } else {
                let placeholder = 512 * 1024 * 1024;
                (
                    placeholder,
                    "no CUDA VRAM detected; using a 512 MiB placeholder budget".to_string(),
                )
            };
            let plan = decoder.layers[0].moe.placement_plan(budget_bytes);
            let on_gpu = plan.overrides.len();
            println!(
                "Layer 0 expert placement plan ({budget_note}, {:.2} MiB budget): {on_gpu}/{} experts fit on GPU",
                budget_bytes as f64 / (1024.0 * 1024.0),
                decoder.layers[0].moe.n_experts()
            );
        }
        Commands::RunReal {
            path,
            token,
            pos,
            fixture,
        } => {
            let cfg = match fixture.as_str() {
                "dense" => test_dense_fixture(),
                "moe" => ferrox_models::config::test_moe_fixture(),
                "mixed" => ferrox_models::config::test_mixed_fixture(),
                other => {
                    anyhow::bail!("unknown fixture '{other}'; expected 'dense', 'moe', or 'mixed'")
                }
            };
            let decoder = Decoder::from_gguf(&path, cfg)?;
            let mut caches: Vec<ferrox_core::cache::KvCache> = (0..decoder.layers.len())
                .map(|_| {
                    ferrox_core::cache::KvCache::new(
                        decoder.config.n_kv_heads,
                        decoder.config.head_dim,
                    )
                })
                .collect();
            let logits = decoder.forward_token(token, pos, &mut caches);
            println!("logits ({} values):", logits.len());
            for (i, v) in logits.iter().enumerate() {
                println!("  [{i}] {v:.6}");
            }
        }
        Commands::Verify {
            model,
            backend,
            emit,
            prompt_tokens,
            prompt,
        } => {
            return verify::run(verify::VerifyArgs {
                model,
                backend,
                emit,
                prompt_tokens,
                prompt,
            });
        }
        Commands::Parity {
            model,
            prompt,
            prompt_tokens,
            top_k,
            dumper,
        } => {
            return parity::run(parity::ParityArgs {
                model,
                prompt,
                prompt_tokens,
                top_k,
                dumper,
            });
        }
        Commands::Bench {
            model,
            n_prompt,
            n_gen,
            reps,
            threads,
            n_gpu_layers,
            ctx_size,
            hidden,
            ffn_dim,
            iters,
            compare,
            suite,
            render,
            id,
            backend,
            fit_host,
            skip_missing,
            bench_dir,
            suite_id,
            backend_label,
            receipt,
        } => {
            if render {
                return bench_suite::render(std::path::Path::new(&bench_dir));
            }
            if suite {
                return bench_suite::run_suite(bench_suite::SuiteArgs {
                    bench_dir: bench_dir.into(),
                    n_prompt,
                    n_gen,
                    reps,
                    only_id: id,
                    only_backend: backend,
                    fit_host,
                    skip_missing,
                });
            }
            if let Some(model) = model {
                bench_model::apply_env(threads, n_gpu_layers);
                return bench_model::run(bench_model::BenchArgs {
                    model,
                    n_prompt,
                    n_gen,
                    reps,
                    ctx_size,
                    compare,
                    backend: backend_label,
                    receipt: receipt.map(Into::into),
                    id: suite_id,
                });
            }
            use ferrox_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
            use std::time::Instant;

            println!(
                "=== matvec microbenchmark: [{ffn_dim} x {hidden}] weight, {iters} iterations ==="
            );

            let mut rng_state: u64 = 12345;
            let mut next = || {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                ((rng_state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            };
            let weights: Vec<f32> = (0..ffn_dim * hidden).map(|_| next() * 0.1).collect();
            let x: Vec<f32> = (0..hidden).map(|_| next() * 0.1).collect();

            let f32_matrix = WeightMatrix::F32(ferrox_core::tensor::Tensor::new(
                weights.clone(),
                vec![ffn_dim, hidden],
            ));
            let mut packed = Vec::new();
            for row in weights.chunks(hidden) {
                packed.extend(ferrox_quant::quantize_q8_0(row));
            }
            let packed_for_scalar_bench = packed.clone();
            let quant_matrix = WeightMatrix::Quantized {
                data: WeightBytes::Owned(packed),
                rows: ffn_dim,
                cols: hidden,
                kind: QuantKind::Q8_0,
            };

            // warm-up
            let _ = f32_matrix.apply(&x);
            let _ = quant_matrix.apply(&x);

            let t0 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(f32_matrix.apply(&x));
            }
            let f32_elapsed = t0.elapsed();

            let t1 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(quant_matrix.apply(&x));
            }
            let quant_elapsed = t1.elapsed();

            // This isolates the *scalar* kernel cost as a single-
            // threaded direct loop, to compare against "dispatched"
            // below (which goes through WeightMatrix::apply, i.e. both
            // rayon row-parallelism AND AVX2/FMA SIMD dispatch if the
            // host supports it). On an AVX2-capable x86_64 host, most
            // of that gap is genuinely SIMD; on an ARM host without a
            // NEON kernel for this format, the "dispatched" path
            // silently falls back to the same scalar kernel per row,
            // so the entire gap here is multi-core parallelism, not
            // SIMD -- confirmed by comparing this against the explicit
            // 1-thread-vs-all-cores section below, which isolates
            // parallelism on its own. Don't read this ratio as "SIMD
            // speedup" without checking which case you're in.
            let row_bytes =
                (hidden / ferrox_quant::Q8_0_BLOCK_ELEMS) * ferrox_quant::Q8_0_BLOCK_BYTES;
            let scalar_rows: Vec<&[u8]> = packed_for_scalar_bench.chunks_exact(row_bytes).collect();
            let t2 = Instant::now();
            for _ in 0..iters {
                let mut acc = 0f32;
                for row in &scalar_rows {
                    acc += ferrox_quant::dot_q8_0_f32_scalar(row, &x);
                }
                std::hint::black_box(acc);
            }
            let scalar_elapsed = t2.elapsed();

            let f32_bytes = f32_matrix.resident_bytes();
            let quant_bytes = quant_matrix.resident_bytes();

            println!(
                "  f32 dequant-resident matmul : {:>8.3} ms/call  ({} bytes resident)",
                f32_elapsed.as_secs_f64() * 1000.0 / iters as f64,
                f32_bytes
            );
            println!(
                "  fused Q8_0 (scalar, no SIMD, single-threaded) : {:>8.3} ms/call",
                scalar_elapsed.as_secs_f64() * 1000.0 / iters as f64
            );
            println!(
                "  fused Q8_0 (dispatched, row-parallel)          : {:>8.3} ms/call  ({} bytes resident)  <- rayon row parallelism + AVX2/FMA (x86_64) or NEON (aarch64) if host supports it",
                quant_elapsed.as_secs_f64() * 1000.0 / iters as f64,
                quant_bytes
            );
            println!(
                "  dispatched is {:.2}x the single-threaded scalar path's time (see \"multi-core scaling\" below to separate the SIMD and parallelism contributions to this number)",
                quant_elapsed.as_secs_f64() / scalar_elapsed.as_secs_f64()
            );
            println!(
                "  memory reduction: {:.2}x smaller resident",
                f32_bytes as f64 / quant_bytes as f64
            );
            println!(
                "  speed ratio: fused Q8_0 (dispatched) is {:.2}x the f32 path's time (<1.0 = fused is faster; with AVX2+FMA this is usually both faster AND smaller, not a memory-for-speed tradeoff)",
                quant_elapsed.as_secs_f64() / f32_elapsed.as_secs_f64()
            );

            println!("\n=== Q4_0 matvec microbenchmark: [{ffn_dim} x {hidden}] weight, {iters} iterations ===");
            let q4_row_bytes =
                (hidden / ferrox_quant::Q4_0_BLOCK_ELEMS) * ferrox_quant::Q4_0_BLOCK_BYTES;
            let mut q4_packed = Vec::with_capacity(ffn_dim * q4_row_bytes);
            for row in weights.chunks(hidden) {
                for block in row.chunks(ferrox_quant::Q4_0_BLOCK_ELEMS) {
                    let amax = block.iter().fold(0f32, |a, &b| a.max(b.abs()));
                    let scale = if amax == 0.0 { 1.0 } else { amax / 7.0 };
                    q4_packed.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                    for i in 0..16 {
                        let lo = ((block.get(i).copied().unwrap_or(0.0) / scale)
                            .round()
                            .clamp(-8.0, 7.0) as i32
                            + 8) as u8;
                        let hi = ((block.get(i + 16).copied().unwrap_or(0.0) / scale)
                            .round()
                            .clamp(-8.0, 7.0) as i32
                            + 8) as u8;
                        q4_packed.push(lo | (hi << 4));
                    }
                }
            }
            let q4_rows: Vec<&[u8]> = q4_packed.chunks_exact(q4_row_bytes).collect();

            let t3 = Instant::now();
            for _ in 0..iters {
                let mut acc = 0f32;
                for row in &q4_rows {
                    acc += ferrox_quant::dot_q4_0_f32_scalar(row, &x);
                }
                std::hint::black_box(acc);
            }
            let q4_scalar_elapsed = t3.elapsed();

            // Route through WeightMatrix::apply, not a direct
            // dot_q4_0_f32 loop -- same rayon row-parallel dispatch the
            // Q8_0 "dispatched" number above uses. An earlier version of
            // this benchmark called dot_q4_0_f32 directly in a plain
            // sequential loop here, which meant it never exercised
            // multi-core parallelism at all and wasn't measuring the
            // same thing as the Q8_0 number above -- caught by actually
            // comparing RAYON_NUM_THREADS=1 vs default on real multi-
            // core hardware and finding the two numbers didn't move
            // together the way they should have.
            let q4_quant_matrix = WeightMatrix::Quantized {
                data: WeightBytes::Owned(q4_packed.clone()),
                rows: ffn_dim,
                cols: hidden,
                kind: QuantKind::Q4_0,
            };
            let _ = q4_quant_matrix.apply(&x); // warm-up
            let t4 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(q4_quant_matrix.apply(&x));
            }
            let q4_dispatched_elapsed = t4.elapsed();

            println!(
                "  Q4_0 (scalar, no SIMD, single-threaded) : {:>8.3} ms/call  ({} bytes resident)",
                q4_scalar_elapsed.as_secs_f64() * 1000.0 / iters as f64,
                q4_packed.len()
            );
            println!(
                "  Q4_0 (dispatched, row-parallel)          : {:>8.3} ms/call  <- rayon row parallelism + AVX2/FMA (x86_64) or NEON (aarch64) if host supports it",
                q4_dispatched_elapsed.as_secs_f64() * 1000.0 / iters as f64
            );
            println!(
                "  Q4_0 dispatched is {:.2}x the single-threaded scalar path's time",
                q4_dispatched_elapsed.as_secs_f64() / q4_scalar_elapsed.as_secs_f64()
            );
            println!(
                "  Q4_0 memory reduction vs f32: {:.2}x smaller resident",
                f32_bytes as f64 / q4_packed.len() as f64
            );

            println!("\n=== multi-core scaling: fused Q8_0 matvec, [{ffn_dim} x {hidden}], {iters} iterations ===");
            let available_cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let single_thread_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("building a 1-thread rayon pool must not fail");
            let t5 = Instant::now();
            single_thread_pool.install(|| {
                for _ in 0..iters {
                    std::hint::black_box(quant_matrix.apply(&x));
                }
            });
            let single_thread_elapsed = t5.elapsed();
            let single_thread_ms = single_thread_elapsed.as_secs_f64() * 1000.0 / iters as f64;
            let all_cores_ms = quant_elapsed.as_secs_f64() * 1000.0 / iters as f64;
            println!("  available cores (std::thread::available_parallelism): {available_cores}");
            println!("  1 thread            : {single_thread_ms:>8.3} ms/call");
            println!("  {available_cores} threads (default) : {all_cores_ms:>8.3} ms/call");
            println!(
                "  measured speedup: {:.2}x (ideal linear speedup would be {:.2}x for {} cores)",
                single_thread_ms / all_cores_ms,
                available_cores,
                available_cores
            );

            println!("\n=== end-to-end decode throughput (synthetic weights, capped layer count for sandbox runtime) ===");
            for (name, preset) in [
                ("glm-5.2", glm_5_2()),
                ("deepseek-v4-pro", deepseek_v4_pro()),
                ("kimi-k3", kimi_k3()),
            ] {
                let mut cfg = preset;
                cfg.hidden_dim = 512;
                cfg.n_heads = 8;
                cfg.n_kv_heads = 2;
                cfg.head_dim = 64;
                cfg.moe.hidden_dim = 512;
                cfg.moe.n_experts = cfg.moe.n_experts.min(32);
                cfg.moe.expert_ffn_dim = 512;
                let n_layers = 4;
                let vocab = 256;

                let decoder = Decoder::new_random_small(cfg, n_layers, vocab);
                let mut caches: Vec<ferrox_core::cache::KvCache> = (0..n_layers)
                    .map(|_| {
                        ferrox_core::cache::KvCache::new(
                            decoder.config.n_kv_heads,
                            decoder.config.head_dim,
                        )
                    })
                    .collect();

                let n_tokens = 32;
                let t0 = Instant::now();
                for pos in 0..n_tokens {
                    std::hint::black_box(decoder.forward_token(pos % vocab, pos, &mut caches));
                }
                let elapsed = t0.elapsed();
                let toks_per_sec = n_tokens as f64 / elapsed.as_secs_f64();
                println!(
                    "  {name:<16} {n_layers} layers, hidden=512, {} experts ({} active): {:>7.1} tok/s ({:.2} ms/token, CPU reference, single thread pool)",
                    decoder.config.moe.n_experts,
                    decoder.config.moe.n_experts_active,
                    toks_per_sec,
                    elapsed.as_secs_f64() * 1000.0 / n_tokens as f64
                );
            }
            println!(
                "\nNote: these are CPU-reference numbers on this sandbox's vCPUs, not GPU numbers."
            );
            println!("They demonstrate the pipeline's relative cost structure (attention vs MoE FFN, memory");
            println!("footprint of quantized vs f32 weights), not absolute performance on target hardware");
            println!("like a DGX Spark. See benchmarks/RESULTS.md for methodology and caveats.");
        }
        Commands::Speculative {
            preset,
            prompt,
            max_new_tokens,
            ngram_size,
            max_draft_len,
        } => {
            let base_cfg = preset_by_name(&preset)?;
            let mut cfg = base_cfg;
            cfg.hidden_dim = 64;
            cfg.n_heads = 8;
            cfg.n_kv_heads = 2;
            cfg.head_dim = 8;
            cfg.moe.hidden_dim = 64;
            cfg.moe.n_experts = cfg.moe.n_experts.min(16);
            cfg.moe.expert_ffn_dim = 32;

            let decoder = Decoder::new_random_small(cfg, 3, 256);
            let mut caches: Vec<ferrox_core::cache::KvCache> = (0..3)
                .map(|_| {
                    ferrox_core::cache::KvCache::new(
                        decoder.config.n_kv_heads,
                        decoder.config.head_dim,
                    )
                })
                .collect();

            let prompt_tokens: Vec<usize> = ferrox_models::ByteTokenizer::encode(&prompt)
                .into_iter()
                .map(|b| b as usize)
                .collect();

            println!("Prompt: {prompt:?} ({} tokens)", prompt_tokens.len());
            println!("Speculator: ngram_size={ngram_size}, max_draft_len={max_draft_len}");
            println!(
                "NOTE: this decoder uses random synthetic weights, not a trained model. A real",
            );
            println!(
                "trained model asked to continue an obviously repeating pattern like this prompt"
            );
            println!(
                "would very likely predict the repeat correctly (that's exactly the case prompt-"
            );
            println!(
                "lookup decoding targets) -- random weights will not reliably reproduce that, so a"
            );
            println!("low or zero accept rate below is expected and does not indicate a bug.\n");

            let speculator = ferrox_models::PromptLookupSpeculator::new(ngram_size, max_draft_len);
            let result = ferrox_models::speculative_decode(
                &decoder,
                &prompt_tokens,
                max_new_tokens,
                &mut caches,
                &speculator,
            );

            println!("Tokens generated : {}", result.tokens_generated);
            println!("forward_batch calls: {}", result.forward_calls);
            println!(
                "Tokens per call   : {:.2} (1.00 = no speedup; higher = drafts were accepted)",
                result.tokens_per_call()
            );
            println!(
                "Calls saved vs. sequential decode: {} (sequential would need exactly {} calls)",
                max_new_tokens as i64 - result.forward_calls as i64 + 1,
                max_new_tokens + 1
            );
        }
        Commands::RunKimi {
            checkpoint_dir,
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            seed,
        } => {
            use std::path::Path;

            let dir = Path::new(&checkpoint_dir);
            let index_path = dir.join("model.safetensors.index.json");
            println!(
                "Opening real Kimi K3 checkpoint index: {}",
                index_path.display()
            );
            let shard = ferrox_safetensors::ShardedSafetensors::open_index(&index_path)?;

            let model_cfg = kimi_k3();
            let hp = ferrox_models::kimi_loader::KimiRealHparams::real();
            println!(
                "Loading all {} real layers (this eagerly touches every routed expert's mmap \
                 range, but never materializes a dequantized f32 copy -- see \
                 kimi_loader's module docs)...",
                model_cfg.n_layers
            );
            let weights =
                ferrox_models::kimi_loader::load_kimi_checkpoint(&shard, &model_cfg, &hp)?;
            println!("Loaded. Vocab size: {}", weights.output_head.rows());

            let vocab_path = dir.join("tiktoken.model");
            let vocab_text = std::fs::read_to_string(&vocab_path)?;
            let ranks = ferrox_models::kimi_tokenizer::parse_tiktoken_vocab(&vocab_text)?;
            let tokenizer_config_path = dir.join("tokenizer_config.json");
            let special_tokens = if tokenizer_config_path.exists() {
                let text = std::fs::read_to_string(&tokenizer_config_path)?;
                ferrox_models::kimi_tokenizer::parse_special_tokens(&text)?
            } else {
                std::collections::HashMap::new()
            };
            let eos_id = special_tokens.get("[EOS]").copied();
            let tokenizer =
                ferrox_models::kimi_tokenizer::KimiTokenizer::new(ranks, special_tokens)?;
            println!(
                "Loaded real tokenizer: {} base tokens, eos_id={eos_id:?}",
                tokenizer.vocab_size()
            );

            let ferrox_models::config::AttentionKind::KimiHybrid(hybrid) = &model_cfg.attention
            else {
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

            let sampling = ferrox_models::sampling::SamplingParams {
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
            };

            println!("Generating (max {max_new_tokens} new tokens)...");
            let (text, ids) = ferrox_models::kimi_generate::kimi_generate(
                &weights,
                &decoder_cfg,
                &hybrid.mla,
                &hybrid.kda,
                &tokenizer,
                &prompt,
                max_new_tokens,
                &sampling,
                eos_id,
                seed,
            );
            println!("Generated {} tokens.", ids.len());
            println!("---");
            println!("{text}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::rewrite_llama_style_argv;

    #[test]
    fn rewrites_llama_multi_character_short_options() {
        let args = ["ferrox", "-m", "model.gguf", "-dev", "none", "-ngl", "0"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            rewrite_llama_style_argv(args),
            [
                "ferrox",
                "run",
                "-m",
                "model.gguf",
                "--device",
                "none",
                "--n-gpu-layers",
                "0"
            ]
        );
    }
}
