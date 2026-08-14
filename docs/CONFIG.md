# Configuration

Prefer CLI flags ([CLI.md](CLI.md)). Environment variables are for server
deployments and advanced tuning. Flags override env when both are set.

## Server

| Variable | Purpose |
|---|---|
| `FERROX_MODEL_PATH` | GGUF path (or Kimi dir); same as `-m` |
| `FERROX_MODEL_DIR` | Extra directory `GET /admin/models` scans, and the one `POST /admin/download` writes into. Without it, the directory holding `FERROX_MODEL_PATH` is used; with neither, downloads are refused (`412`) rather than guessing a location |
| `FERROX_ADDR` | Bind address, e.g. `127.0.0.1:8383` |
| `FERROX_API_KEY` | Require `Authorization: Bearer <key>`. Also gates the whole `/admin` control surface, which can swap models and write files |

## Hugging Face

Read only by `POST /admin/download` (see [API.md](API.md)).

| Variable | Purpose |
|---|---|
| `HF_TOKEN` · `HUGGING_FACE_HUB_TOKEN` | Bearer token for gated or private repos. First one set wins |
| `HF_ENDPOINT` | Hub base URL for a mirror or an air-gapped cache. Default `https://huggingface.co` |

## Backends

Usually set by `-dev` / `-ngl`. Set these only when embedding Ferrox as a
library or overriding the CLI.

| Variable | Purpose |
|---|---|
| `FERROX_METAL` | `1` / `0` / `auto` — Metal offload |
| `FERROX_METAL_ATTN` | `1` / `0` — fused Metal attention + resident KV |
| `FERROX_CTK` | KV dtype: `f16` (default), `q8_0` / `turbo8` / `fp8` / `turbo4` (Metal); `turbo3` falls back to F16. Same as `--ctk` |
| `FERROX_CUDA` | `1` / `0` / `auto` (build with `--features cuda`) |
| `FERROX_CPU_THREADS` | Worker threads; same as `-t`. Default: **performance cores** (`hw.perflevel0.physicalcpu` on macOS), matching llama.cpp — not logical cores |
| `FERROX_CPU_INT_DOT` | int8×int8 matvec + repacked GEMV. **On by default** in `ferrox` / `ferrox-server`; `0` opts out. Off in the library so golden cross-validation stays reference-exact |

## Tuning

| Variable | Purpose |
|---|---|
| `FERROX_CONTINUOUS_BATCHING` | `1` — share decode across concurrent requests |
| `FERROX_CHUNKED_PREFILL` | Split long prefills into N-token chunks |
| `FERROX_CPU_KV_OFFLOAD` | `1` — sync Metal KV to host after each decode step |
| `FERROX_TOKIO_WORKERS` | `ferrox-server` async worker threads (default `2`); keeps the HTTP runtime from oversubscribing the decode pool |
| `FERROX_QOS_LOG` | `1` — log each rayon worker's macOS QoS class at pool start |
| `FERROX_UI` | `1` — serve chat UI at `/` and `/ui` |
| `FERROX_EXIT_ON_STDIN_CLOSE` | `1` — exit on stdin EOF (same as `--exit-on-stdin-close`); off by default so a `/dev/null` stdin does not stop the server |
| `FERROX_KV_POOL_BLOCKS` | Paged-KV pool size (blocks) |
| `FERROX_EXPERT_CACHE_BYTES` | MoE expert-streaming cache budget |
| `FERROX_SSD_STREAMING` | `1` — stream MoE experts from disk |
| `FERROX_GPU_VRAM_BUDGET_BYTES` | Cap GPU-resident MoE experts (`0` = CPU experts on Metal) |

## Kernel-lookup registry

Every kernel lookup the model will make is resolved once at load and
recorded; the registry is then sealed, and a later lookup that misses and
takes a fallback warns once with its call site. See
`ferrox_core::kernel_registry`.

| Variable | Purpose |
|---|---|
| `FERROX_KERNEL_REGISTRY` | `1` — print the whole load-time kernel table (one line per backend × op × quant kind, with the tensor role and the fallback). `0` — record nothing. Default: record, print only the misses that will run off the selected accelerator |
| `FERROX_STRICT_KERNELS` | `1` — refuse to load a model whose weights have no kernel on the selected accelerator, instead of running it on a slower path. Set this in CI and in benchmark harnesses so a number cannot be published for a backend it was not taken on |

Debug switches (`FERROX_METAL_FA_VEC`, `FERROX_METAL_LOGITS`, …) live next
to the code and may change without notice.
