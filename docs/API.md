# API

`ferrox-server` exposes an OpenAI-compatible HTTP API for chat serving.

Fields marked **Reject** return HTTP 400/501 with a clear error.
Unsupported multimodal input is rejected the same way.

## Endpoints

| Endpoint | Status |
|---|---|
| `GET /health` | Supported (capability handshake, see below) |
| `GET /` · `GET /ui` | Static chat UI (`--ui-server` or `FERROX_UI=1`) |
| `GET /v1/models` | Supported |
| `POST /v1/chat/completions` | Supported (JSON + SSE) |
| `POST /v1/completions` | Supported (`prompt`, `max_tokens`, sampling subset) |
| `POST /v1/tokenize` | Supported |
| `POST /v1/detokenize` | Supported |
| `POST /v1/embeddings` | Supported for GGUF Decoder (mean/last pool of hidden states) |
| `POST /v1/messages` | Anthropic-shaped; non-stream text |
| `POST /v1/cancel` | Stop a streamed generation by `request_id` (see below) |
| `GET /cache/stats` · `GET /metrics` | Ferrox extensions |
| `/admin/*` | Control surface (see below) |
| Audio / images | Not supported |

## Chat completions fields

| Field | Status |
|---|---|
| `model`, `messages`, `max_tokens` | Supported |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop` | Supported |
| `presence_penalty`, `frequency_penalty` | Supported |
| `stream` | Supported (overlapped SSE when tools off and CB off) |
| `tools` / `tool_choice: none\|auto` | Supported (prompt-engineered) |
| `tool_choice: required` / named function | **Reject** |
| `logprobs` / `top_logprobs` / `n` (>1) | **Reject** |
| `response_format: json_object` | Supported (best-effort mask + validate) |
| Other `response_format` types | **Reject** |
| `session_id` | Ferrox extension (server-side history) |

## Health

`GET /health` returns JSON (it used to return the string `ok`) and is
never behind auth or rate limiting.

```jsonc
{
  "state": "ready",              // ready | detecting | unavailable
  "model": { "id": "…", "tokenizer": "…", "synthetic_weights": false },
  "capabilities": [
    { "id": "metal", "available": false,
      "reason": "metal_not_built",       // stable code, safe to switch on
      "detail": "Apple M2 Pro is present but this binary was built without --features metal; rebuild to use it." }
  ],
  "version": "0.8.0", "pid": 4242, "uptime_seconds": 12.5,
  "server_time_unix_ms": 1786000000000,
  "last_request_age_seconds": 0.4    // absent until one is served
}
```

Three states, not two. `detecting` means backend probing has not
finished; capabilities in that response are not measurements and a
client should hold rather than render a verdict. Detection gets a hard
1s budget, after which the answer is filled in provisionally with
`reason: "detection_timed_out"`. The handler itself never blocks.

Status is 200 for `ready` and `detecting`, 503 for `unavailable`.
`unavailable` is reachable: `POST /admin/models/unload` produces it,
with `reason: "model_not_loaded"` and a null `model`. A supervisor that
read 200 there would route traffic guaranteed to 503 on arrival.

Every capability carries both a machine `reason` and a human `detail`,
so a UI can grey a control and show the sentence without re-deriving
why. `metal_unavailable` (no device), `metal_not_built` (this binary),
and `disabled` (a flag turned it off) are three different problems with
three different fixes.

`last_request_age_seconds` exists so a busy server is not read as a dead
one: a saturated GPU can starve the health handler, and recent request
activity is positive evidence of liveness.

## Request ids

Every `/v1/chat/completions` response carries a server-assigned
`chatcmpl-…` id. It is the `id` field, and it is repeated once under
`request_id` — in the JSON body for a non-streamed call, and in the
**first** SSE chunk for a streamed one, before any content. That is the
key ferrox logs and (in future) cancels by, so a client never has to
guess which in-flight request is its own.

## Usage timings

`usage` extends the OpenAI shape with llama.cpp-style timings. Prefill
and decode are reported separately, deliberately: dividing total tokens
by total wall time reads a 50 tok/s model as 5 tok/s on a long prompt.

| Field | Meaning |
|---|---|
| `prompt_tokens`, `completion_tokens`, `total_tokens` | OpenAI convention |
| `prompt_eval_duration_ms` | wall time spent on the prompt |
| `generation_duration_ms` | wall time spent in the decode loop |
| `time_to_first_token_ms` | prefill start → first token produced |
| `prompt_per_second`, `predicted_per_second` | per-phase throughput |
| `cached_tokens` | prompt tokens reused from the KV prefix cache |

Every timing is optional and **omitted** rather than nulled or zeroed
when it was not measured (a cached response, a batched decode). Note
`cached_tokens`: absent means no prefix cache is configured, `0` means
the cache was consulted and missed.

Streamed requests get the same `usage` object on the final chunk.

## Admin / control surface

Everything under `/admin` either changes what the server serves or
writes to disk, so it sits behind the same `FERROX_API_KEY` gate as
`/v1/*` — never on the unauthenticated `/health` side. Paths and
payload shapes are defined once in the `ferrox-api` crate, so the UI and
the server cannot disagree about them.

| Endpoint | Answer |
|---|---|
| `GET /admin/models` | inventory (below) |
| `POST /admin/models/load` `{"id":"…"}` | `202 {"task_id":"…"}` |
| `POST /admin/models/unload` | `200 {"ok":true,"active":null}` |
| `POST /admin/download` `{"repo":"…","file":"*.gguf"}` | `202 {"task_id":"…"}` |
| `GET /admin/tasks` | every job, newest last |
| `POST /admin/tasks/{task_id}/cancel` | `200 {"ok":true}` |
| `GET /admin/stats` | counters + recent-request ring |

### Models

```jsonc
{
  "model_dir": "/models",          // null when none is configured
  "active": "Qwen3-0.6B-Q4_K_M",   // null when nothing is loaded
  "models": [{
    "id": "Qwen3-0.6B-Q4_K_M",     // file stem; the only way to name a model
    "path": "/models/Qwen3-0.6B-Q4_K_M.gguf",
    "size_bytes": 396705472,
    "arch": "qwen3", "quant": "Q4_K_M",
    "context_length": 40960, "param_count": 596049920,
    "state": "available",          // loaded | loading | available | error
    "error": null,                 // why the last load attempt failed
    "resident_bytes": null
  }]
}
```

Read from **GGUF headers only** — metadata and tensor descriptors, never
a weight — so listing a directory of checkpoints costs header parses,
not loads. A field that cannot be established cheaply is `null` rather
than guessed, and the key is always present: an unknown context length
and a zero one are different facts. `quant` comes from
`general.file_type` when that maps to a known name (the only place the
`_M` in `Q4_K_M` is stated), otherwise from the measured dominant tensor
dtype — coarser, never invented. `resident_bytes` is always `null`:
ferrox keeps checkpoints mmap-resident, so the true figure is a
page-cache property this process cannot read.

Discovery scans `FERROX_MODEL_DIR` and the directory holding
`FERROX_MODEL_PATH`, non-recursively, for `*.gguf` plus any
safetensors-index checkpoint directory. Split checkpoints fold into one
entry named for the shard prefix.

### Model swap

`POST /admin/models/load` answers immediately with a task id; the load
runs on a blocking thread. Only ids from `GET /admin/models` are
accepted — there is deliberately **no load-by-path endpoint**, so no
request can make the server open an arbitrary file. A second load while
one is running is rejected with `409`.

**In-flight requests keep the model they started on.** A request clones
its handle once, up front, and decodes against exactly those weights
even if a different checkpoint is published mid-generation; the old
model is freed when the last such request finishes. There is no attempt
to migrate a running request — half a completion from one checkpoint and
half from another is worse than either.

After `POST /admin/models/unload`, generation endpoints answer `503`
with `"type": "model_not_loaded"`, `GET /v1/models` returns an empty
list, and `/health` reports `unavailable` (503).

### Tasks

One shape for every long-running job.

```jsonc
{"tasks": [{
  "task_id": "dl-2", "kind": "download",      // download | load
  "label": "Downloading *Q4_K_M.gguf from unsloth/Qwen3-0.6B-GGUF",
  "status": "running",            // queued | running | done | error | cancelled
  "error": null,
  "started_at_ms": 1786717917646, "updated_at_ms": 1786717925031,
  "progress": {
    "fraction": 0.126, "bytes_done": 50198634, "bytes_total": 396705472,
    "rate_bytes_per_s": 9802506.2, "eta_seconds": 35.3,
    "state": "stable"             // warming | stable
  }
}]}
```

Timestamps are the server's Unix epoch milliseconds — the browser clock
is not trusted for ordering. `done` / `error` / `cancelled` are terminal
and never change, so a client can stop polling the moment it reads one.

`rate_bytes_per_s` and `eta_seconds` are `null` while `state` is
`warming`, which lasts until the rate window holds at least 3 samples
spanning at least 3 s. Show "measuring", not a number: the first tick of
any transfer divides out to gigabytes per second. `fraction` is
available immediately — it is a ratio of two counters, not a derivative.

Cancellation is cooperative. `POST …/cancel` raises a flag and returns;
the task reaches `cancelled` only once a worker acknowledges it. A
download stops within one chunk and keeps its `.part` file for a resume.
A model load **cannot be interrupted** mid-mmap, so a cancel arriving
during one discards the finished result rather than pretending the work
stopped early. Cancelling a finished task is a `409`.

### Download

Fetches one `.gguf` from the Hugging Face Hub into the model directory.
`file` may be a literal name or a `*` glob resolved against the repo's
file list (sorted, first match, root-level names only). Resumable: bytes
land in `<target>.part` and a restart sends `Range`, falling back to a
clean restart if the server answers `200` instead of `206`.

Security is a whitelist, not a filter: only bare `*.gguf` filenames, no
separators of either kind, no leading dot, no `..`, no `:`; the joined
path is re-checked to be a direct child of a model directory **fixed at
startup**, never one named by the request. Two downloads of the same
target are rejected with `409` — interleaved writes into one `.part`
would produce a corrupt checkpoint of exactly the right size.

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated repos, and
`HF_ENDPOINT` for a mirror.

### Stats

```jsonc
{
  "uptime_seconds": 43, "requests_total": 1, "errors_total": 0,
  "cache_hits": 0, "cache_misses": 1,
  "tokens_prompt_total": 17, "tokens_generated_total": 24,
  "last_request_age_seconds": 0.02,
  "generating_now": 0,               // decoding right now; NOT a queue depth
  "recent": [{
    "request_id": "chatcmpl-b28a1aeab8f8000000",
    "at_ms": 1786718007013, "route": "/v1/chat/completions",
    "status": 200, "prompt_tokens": 17, "completion_tokens": 24,
    "ttft_ms": 6586.2, "duration_ms": 23603, "decode_ms": 17004.8,
    "stream": false
  }]
}
```

`recent` is a 200-entry ring keyed by the same `request_id` the response
carried, so a log row joins its message by equality rather than by a
claiming heuristic. **`duration_ms` and `decode_ms` are separate and
must stay separate**: the former carries queue wait plus prefill plus
decode, and dividing completion tokens by it reads a 50 tok/s model as
5. `decode_ms` and `ttft_ms` are `null` when the engine did not time
itself or the answer came from cache.

Recorded today for `/v1/chat/completions` (streamed and not, including
rejections), `/v1/completions` and `/v1/messages`. Not yet recorded for
`/v1/embeddings`, `/v1/tokenize` or `/v1/detokenize`.

## Cancellation

Two tiers, both ending at the same server-side flag.

1. **Drop the connection.** The streaming path now notices that its SSE
   receiver is gone and stops at the next token. Before this it ignored
   the failed send and generated the rest of the answer into nothing.
2. **`POST /v1/cancel`** with the `request_id` the first SSE chunk
   states. Behind `FERROX_API_KEY` like the endpoint that started the
   work. A browser should send it with `keepalive: true` so it survives
   the page unload that killed the stream — the case tier 1 is worst at.

```bash
curl -s http://127.0.0.1:8383/v1/cancel \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"chatcmpl-…"}'
```

`200 {"request_id":"…","cancelled":true,"detail":"…"}` when a live
generation was signalled, `404` with `"cancelled": false` when the id
names nothing that is running. The two are different facts: only one of
them saved any work, and a client told `ok` for both would claim it
stopped something it did not.

A cancelled stream ends with `finish_reason: "cancelled"` — not an
OpenAI-defined value, because OpenAI has no cancel endpoint, but *a*
finish reason, so the stream is not read as truncation. Tokens already
decoded are kept and reported in `usage`.

Cooperative and honest about its edges: the flag is read between
decoded tokens, so a prefill already inside a forward pass still
completes, and a continuous-batching request decodes on the shared
batcher thread rather than through this loop and is not covered.
`/v1/completions` and `/v1/messages` are buffered and register no id.

## Streaming behind a proxy

Streamed completions carry `X-Accel-Buffering: no` beside axum's
`Cache-Control: no-cache`. nginx — and the proxies that copied its
convention — buffer `text/event-stream` by default, which turns a
token-by-token stream into one silent wait followed by the whole answer,
and that is indistinguishable at the client from a hung backend.

A keep-alive comment goes out every 15 s, so an idle-but-healthy stream
still puts bytes on the wire. A client's stall timeout should therefore
be measured against bytes received, not tokens: that way a long prefill
never trips it and a swallowed connection does.

Not implemented: `id:` / `retry:` / `Last-Event-ID` replay, and a
polling fallback. `id:` without a server-side replay buffer would be a
promise the server cannot keep.

## Continuous batching

Set `FERROX_CONTINUOUS_BATCHING=1`. Mutually exclusive with
`FERROX_KV_POOL_BLOCKS` and `FERROX_PREFIX_CACHE_ENTRIES`.

Prefill is chunked: the scheduler runs one bounded prefill chunk
(`FERROX_CB_PREFILL_CHUNK`, default 128 tokens, round-robin across
waiting prompts) plus one batched decode step per tick, so a long prompt
joining the batch costs an in-flight decode one chunk rather than the
whole prompt. `FERROX_CB_MAX_SEQS` caps in-flight sequences, counting
prompts still prefilling. `FERROX_CB_MAX_QUEUE` (default 512) caps how many requests may wait for
admission; past it a request is refused with `503` and a `Retry-After`
header rather than queued without bound, and the JSON body names the
queue depth, the cap and `retry_after_seconds`. `GET /metrics` reports
`ferrox_prefill_chunks_total`, `ferrox_prefill_tokens_total`,
`ferrox_decode_steps_total`, `ferrox_scheduler_queue_depth` and
`ferrox_scheduler_queue_rejected_total` while a batcher is active.

Every `503` this server returns carries `Retry-After` (1 second), a
fixed hint rather than a computed one -- an honest estimate would need a
caller's throughput, which the server does not know.

## MCP

`--mcp-config PATH` loads server metadata under `ferrox_mcp` in
`GET /v1/models`. Tool invocation is not wired yet.

## Not yet

Anthropic streaming/tools/images · full JSON schema / grammar ·
`tool_choice=required` · dedicated embedding models · multi-GPU / TP / PD.

See [`ROADMAP.md`](ROADMAP.md).
