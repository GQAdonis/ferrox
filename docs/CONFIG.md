# Configuration

Prefer CLI flags ([CLI.md](CLI.md)). Environment variables are for server
deployments and advanced tuning. Flags override env when both are set.

## Server

| Variable | Purpose |
|---|---|
| `FERROX_MODEL_PATH` | GGUF path (or Kimi dir). Same as `-m` |
| `FERROX_MODEL_DIR` | Extra directory `GET /admin/models` scans, and the one `POST /admin/download` writes into. Without it, the directory holding `FERROX_MODEL_PATH` is used. With neither set, a download returns `412` instead of guessing a location |
| `FERROX_ADDR` | Bind address, e.g. `127.0.0.1:8383` |
| `FERROX_API_KEY` | Require `Authorization: Bearer <key>`. It also covers the whole `/admin` control surface, which swaps models and writes files |

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
| `FERROX_METAL` | `1` / `0` / `auto`, Metal offload |
| `FERROX_METAL_ATTN` | `1` / `0`, fused Metal attention + resident KV |
| `FERROX_CTK` | KV dtype: `f16` (default), `q8_0` / `turbo8` / `fp8` / `turbo4` (Metal). `turbo3` falls back to F16. Same as `--ctk` |
| `FERROX_CUDA` | `1` / `0` / `auto` (build with `--features cuda`) |
| `FERROX_CPU_THREADS` | Worker threads, same as `-t`. Default: **performance cores** (`hw.perflevel0.physicalcpu` on macOS), matching llama.cpp, not logical cores |
| `FERROX_CPU_INT_DOT` | int8×int8 matvec + repacked GEMV. **On by default** in `ferrox` / `ferrox-server`, and `0` opts out. Off in the library so golden cross-validation stays reference-exact |

## Tuning

| Variable | Purpose |
|---|---|
| `FERROX_CONTINUOUS_BATCHING` | `1`, share decode across concurrent requests |
| `FERROX_CB_MAX_SEQS` | Continuous batching: cap on in-flight sequences, counting prompts still prefilling (default: unlimited) |
| `FERROX_CB_PREFILL_CHUNK` | Continuous batching: prompt tokens per prefill chunk (default `128`). The scheduler runs one chunk plus one batched decode step per tick, so this is the granularity at which a long prompt yields to in-flight decodes |
| `FERROX_CB_MAX_QUEUE` | Continuous batching: requests allowed to wait for admission (default `512`). Past it, new requests get `503` + `Retry-After` instead of queueing without bound |
| `FERROX_CB_KV_BLOCKS` | Continuous batching: total KV blocks the scheduler hands out. Unset means no block budget (sequence count alone). Admission is `blocks_needed <= blocks_free`, where a request needs `ceil((prompt + max_tokens) / block_size)` blocks reserved for its whole lifetime |
| `FERROX_CB_KV_BLOCK_SIZE` | Continuous batching: token positions per KV block (default `256`). The admission quantum. A request bigger than the whole budget gets `400` (`code: device_memory_budget_exceeded`), not `503`, because retrying will not fix it |
| `FERROX_CB_MAX_CONTEXT` | Continuous batching: token positions (prompt + `max_tokens`) allowed in a single request. Over it, `400` with `code: context_length_exceeded`, `estimated_bytes` and `limit_bytes`. Unset means no per-request ceiling |
| `FERROX_CHUNKED_PREFILL` | Private (non-batched) generate path only: split a long prefill into N-token `forward_batch` chunks. Unrelated to `FERROX_CB_PREFILL_CHUNK`, which is a *scheduling* quantum, not a batch-shape one |
| `FERROX_CPU_KV_OFFLOAD` | `1`, sync Metal KV to host after each decode step |
| `FERROX_TOKIO_WORKERS` | `ferrox-server` async worker threads (default `2`). Keeps the HTTP runtime from oversubscribing the decode pool |
| `FERROX_QOS_LOG` | `1`, log each rayon worker's macOS QoS class at pool start |
| `FERROX_EXIT_ON_STDIN_CLOSE` | `1`, exit on stdin EOF (same as `--exit-on-stdin-close`). Off by default, so a `/dev/null` stdin does not stop the server |
| `FERROX_KV_POOL_BLOCKS` | Paged-KV pool size (blocks) |
| `FERROX_KV_POOL_BLOCK_SIZE` | Tokens per paged-KV block |
| `FERROX_KV_POOL_QUEUE_TIMEOUT_MS` | How long a request waits for a free block before it is rejected |
| `FERROX_KV_BYTE_BUDGET` | Byte ceiling for the paged-KV pool, independent of block count |
| `FERROX_PREFIX_CACHE_ENTRIES` | Prefix-cache capacity for the private generate path. Mutually exclusive with continuous batching |
| `FERROX_EXPERT_CACHE_BYTES` | MoE expert-streaming cache budget |
| `FERROX_SSD_STREAMING` | `1`, stream MoE experts from disk |
| `FERROX_GPU_VRAM_BUDGET_BYTES` | Cap GPU-resident MoE experts (`0` = CPU experts on Metal) |
| `FERROX_DEVICE_BUDGET_BYTES` | Override the probed memory budget the pre-load KV check plans against (Metal `recommendedMaxWorkingSetSize` / free VRAM / host RAM minus a reserve). For container limits and shared GPUs |

## Security and transport

Everything except `/health` sits behind `FERROX_API_KEY` when it is
set, including `/admin/*`, `/metrics` and `/cache/stats`. A Prometheus
scraper pointed at a keyed server gets a `401`.

| Var | Effect |
|---|---|
| `FERROX_API_KEY` | Bearer token required by every route except `/health` |
| `FERROX_ALLOW_UNAUTHENTICATED_REMOTE` | `1`, permit binding a non-loopback address with no API key. Without it the server **exits at startup** in that configuration, on purpose: an unauthenticated model endpoint on a LAN address is an open proxy |
| `FERROX_TLS_CERT` / `FERROX_TLS_KEY` | PEM cert and key. **Both or neither**, setting one alone is a startup error. Unset means plain HTTP |
| `FERROX_CORS_ORIGINS` | Comma-separated **exact** origins. `*` is rejected on purpose, because a wildcard plus a bearer token is a credential-leak shape. **Required to serve Ferrox Studio (`ui/`) from another origin**, set it to that origin exactly. `ui/`'s dev server proxies the API instead, so `npm run dev` needs none of this |
| `FERROX_RATE_LIMIT_PER_MINUTE` | Global request cap. A non-integer value is a startup error, not a silent default |
| `FERROX_JOURNAL_PATH` | Where the process-lifecycle journal is written |

## Kernel-lookup registry

Every kernel lookup the model will make is resolved once at load and
recorded. The registry is then sealed. A later lookup that misses and
takes a fallback warns once, with its call site. See
`ferrox_core::kernel_registry`.

| Variable | Purpose |
|---|---|
| `FERROX_KERNEL_REGISTRY` | `1`, print the whole load-time kernel table (one line per backend × op × quant kind, with the tensor role and the fallback). `0`, record nothing. Default: record, print only the misses that will run off the selected accelerator |
| `FERROX_ALLOW_UNKNOWN_TENSORS` | `1`, load a checkpoint carrying tensors this build never reads, with a warning. By default that load stops with an error. An unread tensor is a missing term of the graph (gpt-oss attention sinks, `exp_probs_b`), and a wrong answer is worse than no answer |
| `FERROX_ALLOW_MULTIPLE_INSTANCES` | `1`, start even though another ferrox process is already holding a model. By default the second one stops with an error. Two models on one box do not share it, they fight over it, and every timing either reports turns into noise. `--allow-multiple-instances` on `ferrox` / `ferrox-server` does the same for one run |
| `FERROX_INSTANCE_DIR` | Where the running-instance registry lives (default `$XDG_CACHE_HOME/ferrox/instances`, else `~/.cache/ferrox/instances`). One small file per live process, pruned when its pid is gone |
| `FERROX_STRICT_KERNELS` | `1`, stop with an error when a model's weights have no kernel on the selected accelerator, instead of running them on a slower path. Set it in CI and in benchmark harnesses, so nobody publishes a number for a backend it was not taken on |

Debug switches (`FERROX_METAL_FA_VEC`, `FERROX_METAL_LOGITS`, …) live next
to the code and change without notice.
