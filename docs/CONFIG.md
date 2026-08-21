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
| `FERROX_CB_MAX_SEQS` | Continuous batching: cap on in-flight sequences, counting prompts still prefilling (default: unlimited) |
| `FERROX_CB_PREFILL_CHUNK` | Continuous batching: prompt tokens per prefill chunk (default `128`). The scheduler runs one chunk plus one batched decode step per tick, so this is the granularity at which a long prompt yields to in-flight decodes |
| `FERROX_CB_MAX_QUEUE` | Continuous batching: requests allowed to wait for admission (default `512`). Past it, new requests get `503` + `Retry-After` instead of queueing without bound |
| `FERROX_CB_KV_BLOCKS` | Continuous batching: total KV blocks the scheduler may hand out. Unset means no block budget (sequence count alone). Admission is `blocks_needed <= blocks_free`, where a request needs `ceil((prompt + max_tokens) / block_size)` blocks reserved for its whole lifetime |
| `FERROX_CB_KV_BLOCK_SIZE` | Continuous batching: token positions per KV block (default `256`). The admission quantum. A request bigger than the whole budget gets `400` (`kv_budget_exceeded`), not `503` -- retrying cannot fix it |
| `FERROX_CHUNKED_PREFILL` | Private (non-batched) generate path only: split a long prefill into N-token `forward_batch` chunks. Unrelated to `FERROX_CB_PREFILL_CHUNK`, which is a *scheduling* quantum, not a batch-shape one |
| `FERROX_CPU_KV_OFFLOAD` | `1` — sync Metal KV to host after each decode step |
| `FERROX_TOKIO_WORKERS` | `ferrox-server` async worker threads (default `2`); keeps the HTTP runtime from oversubscribing the decode pool |
| `FERROX_QOS_LOG` | `1` — log each rayon worker's macOS QoS class at pool start |
| `FERROX_UI` | `1` — serve the embedded Ferrox Studio at `/` and `/ui` (same as `--ui-server`). **Off by default: without it `/` and every deep link return 404**, not an empty page |
| `FERROX_EXIT_ON_STDIN_CLOSE` | `1` — exit on stdin EOF (same as `--exit-on-stdin-close`); off by default so a `/dev/null` stdin does not stop the server |
| `FERROX_KV_POOL_BLOCKS` | Paged-KV pool size (blocks) |
| `FERROX_KV_POOL_BLOCK_SIZE` | Tokens per paged-KV block |
| `FERROX_KV_POOL_QUEUE_TIMEOUT_MS` | How long a request waits for a free block before it is rejected |
| `FERROX_KV_BYTE_BUDGET` | Byte ceiling for the paged-KV pool, independent of block count |
| `FERROX_PREFIX_CACHE_ENTRIES` | Prefix-cache capacity for the private generate path. Mutually exclusive with continuous batching |
| `FERROX_EXPERT_CACHE_BYTES` | MoE expert-streaming cache budget |
| `FERROX_SSD_STREAMING` | `1` — stream MoE experts from disk |
| `FERROX_GPU_VRAM_BUDGET_BYTES` | Cap GPU-resident MoE experts (`0` = CPU experts on Metal) |
| `FERROX_DEVICE_BUDGET_BYTES` | Override the probed memory budget the pre-load KV check plans against (Metal `recommendedMaxWorkingSetSize` / free VRAM / host RAM minus a reserve). For container limits and shared GPUs |

## Security and transport

Everything except `/health` sits behind `FERROX_API_KEY` when it is
set — including `/admin/*`, `/metrics` and `/cache/stats`. A Prometheus
scraper pointed at a keyed server gets a `401`.

| Var | Effect |
|---|---|
| `FERROX_API_KEY` | Bearer token required by every route except `/health` |
| `FERROX_ALLOW_UNAUTHENTICATED_REMOTE` | `1` — permit binding a non-loopback address with no API key. Without it the server **refuses to start** in that configuration, which is deliberate: an unauthenticated model endpoint on a LAN address is an open proxy |
| `FERROX_TLS_CERT` / `FERROX_TLS_KEY` | PEM cert and key. **Both or neither** — setting one alone is a startup error. Unset means plain HTTP |
| `FERROX_CORS_ORIGINS` | Comma-separated **exact** origins. `*` is rejected on purpose, because a wildcard plus a bearer token is a credential-leak shape |
| `FERROX_RATE_LIMIT_PER_MINUTE` | Global request cap. A non-integer value is a startup error, not a silent default |
| `FERROX_JOURNAL_PATH` | Where the process-lifecycle journal is written |

## Kernel-lookup registry

Every kernel lookup the model will make is resolved once at load and
recorded; the registry is then sealed, and a later lookup that misses and
takes a fallback warns once with its call site. See
`ferrox_core::kernel_registry`.

| Variable | Purpose |
|---|---|
| `FERROX_KERNEL_REGISTRY` | `1` — print the whole load-time kernel table (one line per backend × op × quant kind, with the tensor role and the fallback). `0` — record nothing. Default: record, print only the misses that will run off the selected accelerator |
| `FERROX_ALLOW_UNKNOWN_TENSORS` | `1` — load a checkpoint that carries tensors this build never reads, with a warning, instead of refusing. An unread tensor is a missing term of the graph (gpt-oss attention sinks, `ffn_exp_probs_b`), so the default is refusal: a wrong answer is worse than no answer |
| `FERROX_ALLOW_MULTIPLE_INSTANCES` | `1` — start even though another ferrox process is already holding a model. Default is refusal: two models on one box do not share it, they thrash it, and every timing either reports becomes noise. `--allow-multiple-instances` on `ferrox` / `ferrox-server` does the same for one run |
| `FERROX_INSTANCE_DIR` | Where the running-instance registry lives (default `$XDG_CACHE_HOME/ferrox/instances`, else `~/.cache/ferrox/instances`). One small file per live process, pruned when its pid is gone |
| `FERROX_STRICT_KERNELS` | `1` — refuse to load a model whose weights have no kernel on the selected accelerator, instead of running it on a slower path. Set this in CI and in benchmark harnesses so a number cannot be published for a backend it was not taken on |

Debug switches (`FERROX_METAL_FA_VEC`, `FERROX_METAL_LOGITS`, …) live next
to the code and may change without notice.
