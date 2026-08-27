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
| `FERROX_METAL` | `1` / `0` / `auto`, Metal offload |
| `FERROX_METAL_ATTN` | `1` / `0`, fused Metal attention + resident KV |
| `FERROX_CTK` | KV dtype: `f16` (default), `q8_0` / `turbo8` / `fp8` / `turbo4` (Metal); `turbo3` falls back to F16. Same as `--ctk` |
| `FERROX_CUDA` | `1` / `0` / `auto` (build with `--features cuda`) |
| `FERROX_CPU_THREADS` | Worker threads; same as `-t`. Default: **performance cores** (`hw.perflevel0.physicalcpu` on macOS), matching llama.cpp, not logical cores |
| `FERROX_CPU_INT_DOT` | int8×int8 matvec + repacked GEMV. **On by default** in `ferrox` / `ferrox-server`; `0` opts out. Off in the library so golden cross-validation stays reference-exact |

## Tuning

| Variable | Purpose |
|---|---|
| `FERROX_CONTINUOUS_BATCHING` | `1`, share decode across concurrent requests |
| `FERROX_CB_MAX_SEQS` | Continuous batching: cap on in-flight sequences, counting prompts still prefilling (default: unlimited) |
| `FERROX_CB_PREFILL_CHUNK` | Continuous batching: prompt tokens per prefill chunk (default `128`). The scheduler runs one chunk plus one batched decode step per tick, so this is the granularity at which a long prompt yields to in-flight decodes |
| `FERROX_CB_MAX_QUEUE` | Continuous batching: requests allowed to wait for admission (default `512`). Past it, new requests get `503` + `Retry-After` instead of queueing without bound |
| `FERROX_CB_KV_BLOCKS` | Continuous batching: total KV blocks the scheduler may hand out. Unset means it is *derived* at load alongside `FERROX_CB_MAX_CONTEXT`, or absent when the model cannot be priced. Admission is `blocks_needed <= blocks_free`, where a request needs `ceil((prompt + max_tokens) / block_size)` blocks reserved for its whole lifetime |
| `FERROX_CB_KV_BLOCK_SIZE` | Continuous batching: token positions per KV block (default `256`). The admission quantum. A request bigger than the whole budget gets `400` (`code: device_memory_budget_exceeded`), not `503` -- retrying cannot fix it |
| `FERROX_CB_MAX_CONTEXT` | Token positions (prompt + `max_tokens`) any one request may ask for, on **both** decode paths. Over it, `400` with `code: context_length_exceeded`, `estimated_bytes` and `limit_bytes`. Unset means the ceiling is *derived* at load from weights + per-token KV against the device budget, capped at the model's trained context; unset **and** unpriceable (no device-budget probe, or a header the planner cannot read) means no ceiling |
| `FERROX_SSE_ORPHAN_TIMEOUT_MS` | How long a streaming send waits for a client to make room before the stream is declared abandoned and the generation cancelled (default `30000`). `0` disables the deadline. Guards against a client that is neither reading nor disconnected pinning a blocking thread and the model handle it holds |
| `FERROX_CHUNKED_PREFILL` | Private (non-batched) generate path only: split a long prefill into N-token `forward_batch` chunks. Unrelated to `FERROX_CB_PREFILL_CHUNK`, which is a *scheduling* quantum, not a batch-shape one |
| `FERROX_CPU_KV_OFFLOAD` | `1`, sync Metal KV to host after each decode step |
| `FERROX_TOKIO_WORKERS` | `ferrox-server` async worker threads (default `2`); keeps the HTTP runtime from oversubscribing the decode pool |
| `FERROX_QOS_LOG` | `1`, log each rayon worker's macOS QoS class at pool start |
| `FERROX_EXIT_ON_STDIN_CLOSE` | `1`, exit on stdin EOF (same as `--exit-on-stdin-close`); off by default so a `/dev/null` stdin does not stop the server |
| `FERROX_KV_POOL_BLOCKS` | KV block-pool size (blocks). Bounds how much KV all requests may hold; each request still owns a private contiguous buffer |
| `FERROX_KV_POOL_BLOCK_SIZE` | Tokens per KV pool block |
| `FERROX_KV_POOL_QUEUE_TIMEOUT_MS` | How long a request waits for free KV before it is rejected. Applies to both the pool and paged KV |
| `FERROX_KV_BYTE_BUDGET` | Byte ceiling for the KV block pool, independent of block count |
| `FERROX_PAGED_KV_BLOCKS` | Blocks per layer of real paged KV: shared page storage many requests read through a block table, rather than a private buffer each. `ferrox-server` only. Mutually exclusive with `FERROX_KV_POOL_BLOCKS`/`FERROX_KV_BYTE_BUDGET` and with `FERROX_PREFIX_CACHE_ENTRIES`; setting an excluded pair stops the server with an error naming both. Read the paragraph under this table before using it |
| `FERROX_PAGED_KV_BLOCK_SIZE` | Positions per paged-KV block. Must be set together with `FERROX_PAGED_KV_BLOCKS`, or the server stops |
| `FERROX_PREFIX_CACHE_ENTRIES` | Prefix-cache capacity for the private generate path: whole KV snapshots in an LRU list, reported under `GET /cache/stats`. Mutually exclusive with continuous batching and with paged KV. Paged KV carries no such exclusion: it composes with continuous batching and shares prefixes through the radix tree instead |
| `FERROX_EXPERT_CACHE_BYTES` | MoE expert-streaming cache budget |
| `FERROX_SSD_STREAMING` | `1`, stream MoE experts from disk |
| `FERROX_GPU_VRAM_BUDGET_BYTES` | Cap GPU-resident MoE experts (`0` = CPU experts on Metal) |
| `FERROX_DEVICE_BUDGET_BYTES` | Override the probed memory budget the pre-load KV check plans against (Metal `recommendedMaxWorkingSetSize` / free VRAM / host RAM minus a reserve). For container limits and shared GPUs |
| `FERROX_PIN_BUDGET_GB` | Override the page-locking budget expert-bank placement plans against. Unset means no cap on plain Linux; on WSL, where WDDM-backed CUDA caps pinning near half of RAM *shared across processes*, it defaults to 40% of physical RAM |
| `FERROX_BENCHBW_PATH` | Explicit path to this host's measured bandwidth profile. Otherwise one file per GPU uuid under `$XDG_CACHE_HOME/ferrox/benchbw/`, then the legacy `benchbw.json`. A file that exists but does not parse yields **no** profile rather than falling through, so a corrupt per-card profile never silently borrows another card's numbers |
| `FERROX_DECODE_LOG_INTERVAL` | Decode forwards between two batch status lines (default 40). `0` logs every forward. An unparseable value takes the default rather than failing a server to start over a log setting |
| `FERROX_ACCOUNTING_OUTBOX` | Directory accounting receipts are written to, atomically and idempotently by receipt id, before `POST /v1/admin/prepare-stop` answers. Unset means no outbox and no persistence step |
| `FERROX_INSTANCE_ID` | Names this engine generation for the receipt id. Unset falls back to the pid plus the process's wall-clock start, which is stable for the life of the process and different in the next one |

### Reading KV back after a Metal prefill

Paged KV used to be refused on any GPU backend, because it returned
fluent wrong tokens there. Measured on an M2 Pro with Llama-3.2-3B
Q4_K_M, same prompt and same seed: CPU paged and CPU contiguous both
answered "Blue and Red.", and Metal paged answered "Blue ( question
mark;>a> is a> is".

The cause was not the page indirection. A Metal prefill leaves K/V on
the device and fills the host cache with placeholder rows, which the
contiguous decode path knows and reads around; the paged prefill copied
those placeholders into the page store, and decode then attended over a
prompt the model never saw. The prefill now downloads the real rows for
the caller that reads them.

Held by `cargo test -p ferrox-models --features metal --test
paged_metal_parity -- --ignored`, which greedy-decodes the same prompt
twice in one process, once through each cache, on a dense model, an MoE
model and a sliding-window model. It runs one model per process on
purpose: two checkpoints loaded into a single process do not answer the
same as either alone on Metal, which is a separate bug and not one this
check should be at the mercy of.

`FERROX_PREFIX_CACHE_ENTRIES` had the same bug and no refusal in front
of it. A stored snapshot is the host rows, so on Metal it was all zeros,
and the next request restoring it answered nonsense at full speed --
"Blue and red." became " question mark of the day. The question of the
day is a question of the day." `sync_metal_attn_kv_to_host` could not
repair it: that function appends past `seq_len`, and the placeholder
fill has already advanced `seq_len` past the region needing filled. The
prefill now downloads the real rows when a prefix cache is configured,
and only then, since nothing else on that path reads them back.

### Sharing pages between prompts

Turning paged KV on also turns on a radix prefix cache over the same
pages. There is no separate switch, because sharing means two sequences
pointing at one page rather than one of them holding a copy, and only
the paged store can do that.

Each new prompt is matched against a tree of already-computed prefixes,
keyed by page. The matched prefix is locked for the request's life and
its page groups have their reference counts raised, so a thousand
conversations off one system prompt hold one copy of its KV rather than
a thousand. What the request reused shows up as `usage.cached_tokens`.
There is no aggregate hit rate on `/v1/stats` or `/metrics`, and no
eviction knob: back pressure comes from the page store running out and
`FERROX_KV_POOL_QUEUE_TIMEOUT_MS` turning waiting requests away.

This is a different mechanism from `FERROX_PREFIX_CACHE_ENTRIES`, which
stores whole contiguous KV snapshots and copies them. The two cannot be
on at once.

## Security and transport

Everything except `/health` sits behind `FERROX_API_KEY` when it is
set, including `/admin/*`, `/metrics` and `/cache/stats`. A Prometheus
scraper pointed at a keyed server gets a `401`.

| Var | Effect |
|---|---|
| `FERROX_API_KEY` | Bearer token required by every route except `/health` |
| `FERROX_ALLOW_UNAUTHENTICATED_REMOTE` | `1`, permit binding a non-loopback address with no API key. Without it the server **refuses to start** in that configuration, which is deliberate: an unauthenticated model endpoint on a LAN address is an open proxy |
| `FERROX_TLS_CERT` / `FERROX_TLS_KEY` | PEM cert and key. **Both or neither**, setting one alone is a startup error. Unset means plain HTTP |
| `FERROX_CORS_ORIGINS` | Comma-separated **exact** origins. `*` is rejected on purpose, because a wildcard plus a bearer token is a credential-leak shape. **Required to serve Ferrox Studio (`ui/`) from another origin**, set it to that origin exactly. `ui/`'s dev server proxies the API instead, so `npm run dev` needs none of this |
| `FERROX_RATE_LIMIT_PER_MINUTE` | Global request cap. A non-integer value is a startup error, not a silent default |
| `FERROX_JOURNAL_PATH` | Where the process-lifecycle journal is written |
| `FERROX_CONVERSATIONS_DIR` | Where server-side conversations are stored, one JSON file each (default `./ferrox-conversations`). Created on first write. Nothing is evicted: the caps refuse with a reason rather than dropping an older conversation |

## Kernel-lookup registry

Every kernel lookup the model will make is resolved once at load and
recorded; the registry is then sealed, and a later lookup that misses and
takes a fallback warns once with its call site. See
`ferrox_core::kernel_registry`.

| Variable | Purpose |
|---|---|
| `FERROX_KERNEL_REGISTRY` | `1`, print the whole load-time kernel table (one line per backend × op × quant kind, with the tensor role and the fallback). `0`, record nothing. Default: record, print only the misses that will run off the selected accelerator |
| `FERROX_ALLOW_UNKNOWN_TENSORS` | `1`, load a checkpoint that carries tensors this build never reads, with a warning, instead of refusing. An unread tensor is a missing term of the graph (gpt-oss attention sinks, `exp_probs_b`), so the default is refusal: a wrong answer is worse than no answer |
| `FERROX_ALLOW_MULTIPLE_INSTANCES` | `1`, start even though another ferrox process is already holding a model. Default is refusal: two models on one box do not share it, they thrash it, and every timing either reports becomes noise. `--allow-multiple-instances` on `ferrox` / `ferrox-server` does the same for one run |
| `FERROX_INSTANCE_DIR` | Where the running-instance registry lives (default `$XDG_CACHE_HOME/ferrox/instances`, else `~/.cache/ferrox/instances`). One small file per live process, pruned when its pid is gone |
| `FERROX_STRICT_KERNELS` | `1`, refuse to load a model whose weights have no kernel on the selected accelerator, instead of running it on a slower path. Set this in CI and in benchmark harnesses so a number cannot be published for a backend it was not taken on |

Debug switches (`FERROX_METAL_FA_VEC`, `FERROX_METAL_LOGITS`, …) live next
to the code and may change without notice.
