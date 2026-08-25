# API

`ferrox-server` exposes an OpenAI-compatible HTTP API for chat serving.

Fields marked **Reject** return HTTP 400 or 501 with an error message
that names the problem. Multimodal input the server does not handle
comes back the same way.

## Endpoints

| Endpoint | Status |
|---|---|
| `GET /health` | Supported (capability handshake, see below) |
| `GET /v1/models` | Supported |
| `POST /v1/chat/completions` | Supported (JSON + SSE) |
| `POST /v1/completions` | Supported (`prompt`, `max_tokens`, sampling subset) |
| `POST /v1/tokenize` | Supported |
| `POST /v1/detokenize` | Supported |
| `POST /v1/embeddings` | Supported for GGUF Decoder (mean/last pool of hidden states) |
| `POST /v1/messages` | Anthropic-shaped, non-stream text |
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
| `tools` / `tool_choice: none\|auto` | Supported (prompt-engineered, parsed in nine wire formats) |
| `tool_choice: required` / named function | **Reject** |
| `logprobs` / `top_logprobs` / `n` (>1) | **Reject** |
| `response_format: json_object` | Supported (best-effort mask + validate) |
| Other `response_format` types | **Reject** |
| `session_id` | Ferrox extension (server-side history) |
| `reasoning_content` (response) | Ferrox extension: a reasoning model's chain of thought, split out of `content` |

### Where a completion stops

Besides `max_tokens` and the request's own `stop` strings, generation ends
on any token in the checkpoint's **end-of-generation set**: the
`eos`/`eot`/`eom` metadata ids plus every vocabulary entry whose text is on
llama.cpp's literal EOG list (`<|eot_id|>`, `<end_of_turn>`, `<|im_end|>`,
`<turn|>`, …). Stop on `tokenizer.ggml.eos_token_id` alone, as this
server did until `StopTokens` landed, and a Llama-3 or gemma checkpoint
runs past the end of its own turn, because neither one's turn ender *is*
the metadata EOS. Both decode paths (`generate` and the continuous
batcher) use the same set. A Kimi K3 checkpoint keeps its vocabulary in
`tokenizer_config.json` rather than GGUF metadata, so the set is derived
from its special-token names and `[EOT]` ends a turn there too.

BOS follows the rule in [CLI.md](CLI.md#who-adds-bos): the chat template
owns it when it prints one, the loader otherwise, added idempotently so a
template that already emitted BOS never gets a second one.

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
  "version": "0.9.1", "pid": 4242, "uptime_seconds": 12.5,
  "server_time_unix_ms": 1786000000000,
  "last_request_age_seconds": 0.4    // absent until one is served
}
```

There are three states, not two. `detecting` means backend probing has
not finished. The capabilities in that response are guesses, not
measurements, so a client should wait rather than draw a conclusion.
Detection gets a hard 1s budget. Past that the answer is filled in
provisionally with `reason: "detection_timed_out"`. The handler itself
never blocks.

Status is 200 for `ready` and `detecting`, 503 for `unavailable`. You
reach `unavailable` by calling `POST /admin/models/unload`, which
answers with `reason: "model_not_loaded"` and a null `model`. A
supervisor that read 200 there would route traffic straight into a 503.

Every capability carries a machine `reason` and a human `detail`, so a
UI greys the control and shows the sentence without working out why
itself. `metal_unavailable` (no device), `metal_not_built` (this
binary), and `disabled` (a flag turned it off) are three different
problems with three different fixes.

`last_request_age_seconds` exists so nobody reads a busy server as a
dead one. A saturated GPU starves the health handler, and recent
request activity is positive evidence that the process is alive.

## Request ids

Every `/v1/chat/completions` response carries a server-assigned
`chatcmpl-…` id. It is the `id` field, and it is repeated once under
`request_id`. For a non-streamed call it sits in the JSON body. For a
streamed one it arrives in the **first** SSE chunk, before any content.
That is the key ferrox logs and cancels by, so a client never has to
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
| `acceptance_length` | completion tokens per speculative verification step |
| `draft_tokens`, `accepted_draft_tokens` | draft tokens evaluated / kept |
| `draft_accept_rate_per_position` | accept rate at each position in the draft block |

Every timing is optional and **omitted** rather than nulled or zeroed
when it was not measured (a cached response, a batched decode). Note
`cached_tokens`: absent means no prefix cache is configured, `0` means
the cache was consulted and missed.

The four speculation fields are absent unless speculative decoding ran,
which is not the same statement as `acceptance_length: 1.0` ("it ran and
never helped"). `draft_accept_rate_per_position` is reported beside the
mean rather than folded into it: a drafter that is right at the first
drafted position and useless at the last has the same mean as a
uniformly mediocre one, and the two call for opposite block sizes.
`/admin/stats` rows carry `acceptance_length` and
`draft_accept_rate_per_position` for the same reason.

**Today these fields are always absent**: `ferrox-server` has no
speculative decode path yet (`ferrox speculative` is a CLI-only demo),
so nothing populates them. They are the wire contract the engine's
metrics land on, not evidence that the server speculates.

Streamed requests get the same `usage` object on the final chunk.

## Admin / control surface

Everything under `/admin` either changes what the server serves or
writes to disk, so it needs the same `FERROX_API_KEY` as `/v1/*`. None
of it sits on the unauthenticated `/health` side. Paths and payload
shapes are defined once in the `ferrox-api` crate, so the UI and the
server cannot disagree about them.

| Endpoint | Answer |
|---|---|
| `GET /admin/models` | inventory (below) |
| `POST /admin/models/load` `{"id":"…"}` | `202 {"task_id":"…"}` |
| `POST /admin/models/unload` | `200 {"ok":true,"active":null}` |
| `POST /admin/download` `{"repo":"…","file":"*.gguf"}` | `202 {"task_id":"…"}` |
| `GET /admin/tasks` | every job, newest last |
| `POST /admin/tasks/{task_id}/cancel` | `200 {"ok":true}` |
| `GET /admin/stats` | counters + recent-request ring |

| Stream recovery | |
|---|---|
| `GET /v1/stream/{request_id}` | reconnect into a resumable stream (SSE) |
| `GET /v1/stream/{request_id}/poll` | the same replay buffer over JSON |

### Models

```jsonc
{
  "model_dir": "/models",          // null when none is configured
  "active": "Qwen3-0.6B-Q4_K_M",   // null when nothing is loaded
  "models": [{
    "id": "Qwen3-0.6B-Q4_K_M",     // file stem, the only way to name a model
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

This comes from **GGUF headers only**: metadata and tensor descriptors,
never a weight. Listing a directory of checkpoints costs header parses,
not loads. A field that is expensive to establish comes back `null`
rather than guessed, and the key is always present, because an unknown
context length and a zero one are different facts. `quant` comes from
`general.file_type` when that maps to a known name, which is the only
place the `_M` in `Q4_K_M` is stated. Failing that it comes from the
dominant tensor dtype, which is coarser and still measured rather than
invented. `resident_bytes` is always `null`, because ferrox keeps
checkpoints mmap-resident and the true figure is a page-cache property
this process has no way to read.

Discovery scans `FERROX_MODEL_DIR` and the directory holding
`FERROX_MODEL_PATH`, non-recursively, for `*.gguf` plus any
safetensors-index checkpoint directory. Split checkpoints fold into one
entry named for the shard prefix.

### Model swap

`POST /admin/models/load` answers immediately with a task id, and the
load runs on a blocking thread. It accepts only ids from
`GET /admin/models`. There is deliberately **no load-by-path
endpoint**, so no request talks the server into opening an arbitrary
file. A second load while one is running comes back `409`.

**In-flight requests keep the model they started on.** A request clones
its handle once, up front, and decodes against exactly those weights
even if a different checkpoint is published mid-generation. The old
model is freed when the last such request finishes. Nothing tries to
migrate a running request. Half a completion from one checkpoint and
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

Timestamps are the server's Unix epoch milliseconds. The browser clock
is not trusted for ordering. `done`, `error` and `cancelled` are
terminal and never change, so a client stops polling the moment it
reads one.

`rate_bytes_per_s` and `eta_seconds` are `null` while `state` is
`warming`, which lasts until the rate window holds at least 3 samples
spanning at least 3 s. Show "measuring", not a number. The first tick of
any transfer divides out to gigabytes per second. `fraction` is
available immediately, because it is a ratio of two counters rather
than a derivative.

Cancellation is cooperative. `POST …/cancel` raises a flag and returns.
The task reaches `cancelled` only once a worker acknowledges it. A
download stops within one chunk and keeps its `.part` file for a resume.
A model load **cannot be interrupted** mid-mmap, so a cancel that
arrives during one discards the finished result rather than pretending
the work stopped early. Cancelling a finished task is a `409`.

### Download

Fetches one `.gguf` from the Hugging Face Hub into the model directory.
Give `file` a literal name or a `*` glob, which is resolved against the
repo's file list (sorted, first match, root-level names only).
Downloads resume: bytes land in `<target>.part` and a restart sends
`Range`, falling back to a clean restart when the server answers `200`
instead of `206`.

Security here is a whitelist, not a filter. Bare `*.gguf` filenames
only: no separators of either kind, no leading dot, no `..`, no `:`.
The joined path is then re-checked to be a direct child of a model
directory **fixed at startup**, never one named by the request. Two
downloads of the same target come back `409`, because interleaved
writes into one `.part` file produce a corrupt checkpoint of exactly
the right size.

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated repos, and
`HF_ENDPOINT` for a mirror.

### Stats

```jsonc
{
  "uptime_seconds": 43, "requests_total": 1, "errors_total": 0,
  "cache_hits": 0, "cache_misses": 1,
  "tokens_prompt_total": 17, "tokens_generated_total": 24,
  "last_request_age_seconds": 0.02,
  "generating_now": 0,               // decoding right now, NOT a queue depth
  "generating_now": 0,               // decoding right now; NOT a queue depth
  "queue_depth": null,               // null unless continuous batching is on
  "queue_rejected_total": null,      // turned away by the queue cap, since start
  "recent": [{
    "request_id": "chatcmpl-b28a1aeab8f8000000",
    "at_ms": 1786718007013, "route": "/v1/chat/completions",
    "model": "Qwen3-0.6B-Q4_K_M",   // what SERVED it, not what it asked for
    "status": 200, "prompt_tokens": 17, "completion_tokens": 24,
    "ttft_ms": 6586.2, "duration_ms": 23603, "decode_ms": 17004.8,
    "stream": false,
    "via_api_key": "key-4f21a0c3",   // fingerprint, never the key; null if none
    "client": "ferrox-studio"        // SELF-DECLARED (X-Ferrox-Client); null if absent
  }]
}
```

`recent` is a 200-entry ring keyed by the same `request_id` the response
carried, so a log row joins its message on equality instead of on a
guess. **`duration_ms` and `decode_ms` are separate and must stay
separate.** The first carries queue wait plus prefill plus decode, so
dividing completion tokens by it reads a 50 tok/s model as 5.
`decode_ms` and `ttft_ms` are `null` when the engine did not time itself
or the answer came from cache.

Recorded today for `/v1/chat/completions` (streamed and not, including
rejections), `/v1/completions`, `/v1/messages`, `/v1/embeddings`,
`/v1/tokenize` and `/v1/detokenize`, the last three with their status,
success or failure. `/v1/tokenize` and `/v1/detokenize` carry no token
counts on purpose: they run the tokenizer and not the model, and
`prompt_tokens` here feeds `tokens_prompt_total`, which means "tokens
this server put through a forward pass". `/v1/embeddings` does run one,
so its prompt tokens are counted and its `decode_ms` is `null`, there
is no decode loop to time.

`queue_depth` and `queue_rejected_total` come from the continuous-batching
scheduler's queue and are `null`, not `0`, when batching is off, because
then nothing queues at all: every request gets its own blocking thread. A
gauge reading `0` claims an empty queue was measured.

`model` names the model that answered, as `/v1/models` reports it, and
never the `model` field the request carried, this server decodes
against whatever is loaded and ignores that string, so echoing it back
would make the log agree with the caller's belief rather than with what
happened. `null` means nothing was loaded, which is what a 503 row is.

**Attribution.** `via_api_key` is a short fingerprint of the bearer key
that served the request, never the key and never anything the key can be
recovered from; it is salted per process, so it is stable within one
server run, meaningless across a restart, and useless for testing key
guesses offline. `null` means no `Authorization: Bearer` header was
presented, which on a server started without `FERROX_API_KEY` is every
request. `client` is the caller's own `X-Ferrox-Client` header, kept to
32 label characters, **a claim, not proof**: Ferrox Studio sends
`ferrox-studio` and so could anything else. Nothing authenticates it, and
a UI that shows it must say so.

## Errors that retrying will not fix

Two ceilings do not move with load. A request that hits one gets a `400`
and never a `503`, because an idle server would answer the same way, and
a `Retry-After` here would be a lie that turns into a retry loop.

```json
{"error": {
  "message": "context_length_exceeded: request asks for 9000 token positions …",
  "type": "invalid_request_error",
  "code": "context_length_exceeded",
  "estimated_bytes": 4718592000,
  "limit_bytes": 4194304000,
  "positions": 9000,
  "positions_limit": 8000
}}
```

`code` names **which** ceiling is binding. The two have different fixes,
and only one of them is in the client's hands:

| `code` | Meaning | Fix |
|---|---|---|
| `context_length_exceeded` | Longer than any single request is allowed to be (`FERROX_CB_MAX_CONTEXT`) | Shorter prompt, lower `max_tokens` |
| `device_memory_budget_exceeded` | Bigger than the server's whole KV budget (`FERROX_CB_KV_BLOCKS`) | More KV blocks, or a smaller model |

`estimated_bytes` and `limit_bytes` are the KV cost of the request and
of the ceiling, priced from the model's own layout. It is the same
arithmetic `ferrox inspect-plan` prints, so check it yourself instead of
taking it on trust. When a request exceeds both ceilings, the
per-request one is reported, because that is the one the caller acts on.

Momentary pressure gets a different answer: `503` with `Retry-After`,
counted separately, so an operator reading `/metrics` tells "one request
too big" apart from "too many requests".

## Stop sequences

Two layers, because a stop sequence promises something about what
reaches the client, and streaming makes that promise hard to keep.

1. **Token level.** A stop string that is exactly one token in this
   model's vocabulary, the usual case for `<|im_end|>`,
   `<end_of_turn>` and `<|eot_id|>`, is matched on the token id before
   the token is detokenized. That catches control tokens whose rendered
   form is not the string the client asked for, which the text scan
   never sees, and the token never becomes part of the answer. This is
   the same treatment the EOS token gets, and it is left out of `usage`
   too.
2. **Text level.** Everything else is matched on the emitted text, with
   the output buffered so that no trailing text still capable of
   becoming a stop string is ever sent. Only the real partial match is
   withheld, not a
   fixed number of bytes, so ordinary text goes out the moment it is
   provably safe rather than permanently trailing the model by the
   length of the longest stop string.

A partial match therefore never appears on the wire and is never taken
back. SSE has no mechanism for taking anything back. Text withheld
against a match that never arrives is released when the answer ends, so
nothing is lost.

## Cancellation

Two tiers, both ending at the same server-side flag.

1. **Drop the connection.** The streaming path now notices that its SSE
   receiver is gone and stops at the next token. Before this it ignored
   the failed send and generated the rest of the answer into nothing.
2. **`POST /v1/cancel`** with the `request_id` the first SSE chunk
   states. Behind `FERROX_API_KEY` like the endpoint that started the
   work. A browser should send it with `keepalive: true` so it survives
   the page unload that killed the stream, which is the case tier 1
   handles worst.

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

A cancelled stream ends with `finish_reason: "cancelled"`. OpenAI does
not define that value, because OpenAI has no cancel endpoint, but it is
*a* finish reason, so no client reads the stream as truncation. Tokens
already decoded are kept and reported in `usage`.

Continuous-batching requests are covered too, by a different route. The
cancel enqueues an id, then the scheduler thread drains that queue
between steps and drops the sequence itself. Removing a sequence means
touching KV buffers a forward pass might be reading, so the mutation
happens on the thread that owns the step rather than wherever the cancel
arrived. A request stops wherever it has got to, whether still queued,
mid-prefill, or decoding, and keeps whatever it produced.

The mechanism has one honest edge: the flag is read between decoded
tokens, so a prefill already inside a forward pass runs to completion.
`/v1/completions` and `/v1/messages` are buffered and register no id.

## Streaming behind a proxy

Streamed completions carry `X-Accel-Buffering: no` beside axum's
`Cache-Control: no-cache`. nginx, and the proxies that copied its
convention, buffer `text/event-stream` by default. That turns a
token-by-token stream into one silent wait followed by the whole answer,
which looks exactly like a hung backend from the client side.

A keep-alive comment goes out every 15 s, so an idle but healthy stream
still puts bytes on the wire. Measure your stall timeout against bytes
received rather than tokens. A long prefill then never trips it, and a
swallowed connection does.

Not implemented: `id:`, `retry:` and `Last-Event-ID` replay, plus a
polling fallback. Sending `id:` without a server-side replay buffer
would be a promise this server cannot keep.
### Resumable streams

`id:` is a promise that a reconnect can pick up where the connection
stopped, so it is emitted **only** where a replay buffer exists. Ask for
one per request:

```jsonc
{"model": "...", "messages": [...], "stream": true, "stream_resumable": true}
```

Then every event carries `id: {request_id}:{n}` and the first one also
carries `retry: 1500`. Two ways back in, both behind `FERROX_API_KEY`
like the request that filled the buffer:

```bash
# Reconnect over SSE, continuing after the last event seen
curl -N http://127.0.0.1:8383/v1/stream/chatcmpl-… \
  -H 'Last-Event-ID: chatcmpl-…:41'

# Or drain the same buffer over plain JSON
curl "http://127.0.0.1:8383/v1/stream/chatcmpl-…/poll?from=42"
# {"request_id":"…","events":[{"index":42,"data":"{…}"}],
#  "next_index":43,"done":false}
```

`data` is the exact payload the stream sent, `[DONE]` included, so a
client feeds replayed events to the same parser as live ones. `done` is
`true` only once the generation has ended *and* the buffer is drained,
so a client that stops on it never discards events it was not given.
A poll parks for up to 10 s waiting for the next event, which keeps it
reading like a stream while staying a short JSON response, the thing a
buffering proxy cannot hold back, which is the whole point of the
fallback.

**`stream_resumable` also changes what a dropped socket means.** Without
it, the connection closing cancels the generation (tier 1 above). With
it, the generation keeps running into the replay buffer, that is the
point of asking for one, and `POST /v1/cancel` is its stop path. The
caller decides because neither answer suits both: a tab that navigated
away wants the CPU back, and a tab whose proxy dropped a 90-second
answer wants the answer.

Both bounds fail closed rather than quietly. The replay window is 1 MiB
per stream; past it the oldest events are dropped and asking for an
evicted position answers `410 replay_window_lost` rather than a stream
with a hole in it. A finished stream stays reconnectable for 120 s and
at most 64 streams are remembered, oldest finished evicted first, a
live stream is never evicted out from under its reader. A `Last-Event-ID`
naming a different request is `400 bad_last_event_id`, not silently
rounded down to a full replay of the wrong answer. An unknown or
forgotten id is `404 stream_not_found`.

## Continuous batching

Set `FERROX_CONTINUOUS_BATCHING=1`. Mutually exclusive with
`FERROX_KV_POOL_BLOCKS` and `FERROX_PREFIX_CACHE_ENTRIES`.

Prefill is chunked. Per tick the scheduler runs one bounded prefill
chunk (`FERROX_CB_PREFILL_CHUNK`, default 128 tokens, round-robin
across waiting prompts) plus one batched decode step. A long prompt
joining the batch therefore costs an in-flight decode one chunk instead
of the whole prompt.

`FERROX_CB_MAX_SEQS` caps in-flight sequences, counting prompts still
prefilling. `FERROX_CB_MAX_QUEUE` (default 512) caps how many requests
wait for admission. Past that cap a request gets a `503` with a
`Retry-After` header instead of joining an unbounded queue, and the JSON
body names the queue depth, the cap and `retry_after_seconds`.

While a batcher is active, `GET /metrics` reports
`ferrox_prefill_chunks_total`, `ferrox_prefill_tokens_total`,
`ferrox_decode_steps_total`, `ferrox_scheduler_queue_depth` and
`ferrox_scheduler_queue_rejected_total`.

Every `503` this server returns carries `Retry-After: 1`. It is a fixed
hint, not a computed one. An honest estimate would need the caller's
throughput, and the server does not know it.

## MCP

`--mcp-config PATH` loads server metadata under `ferrox_mcp` in
`GET /v1/models`. Tool invocation is not wired yet.

## Reasoning content

A checkpoint whose family reasons inside markers (`<think>`, DeepSeek's
DSML variant, Gemma's channels, MiniMax-M3's namespaced pair, the
gpt-oss harmony `analysis` channel) has that block split out of
`content` and returned as `reasoning_content`, on the message and on
each streamed delta. The split runs *as tokens arrive*, so an
overlapped SSE stream and a buffered one report the same fields for the
same request rather than differing by transport.

Which family applies is inferred from the served model's name, which is
all there is: ferrox carries no per-checkpoint parser declaration. A
name that implies nothing gets no reasoning parser at all — the right
answer for a model that does not reason, since an unconditional
splitter would eat a literal `<think>` written in a code block.

## Tool-call formats

`tools` is still prompt-engineered (there is no grammar-constrained
decoding here — see the `tool_choice` rejections above), and the
preamble asks for a Hermes-style
`<tool_call>{"name": …, "arguments": {…}}</tool_call>`. But a model
trained on a different format frequently answers in its own, correctly
and in its own terms. Parsing tries the format the served checkpoint's
family implies, then the one the preamble asked for:

| Family | Shape |
|---|---|
| Hermes / Qwen2.5 | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` |
| Llama 3 | `<\|python_tag\|>{"name": …, "parameters": {…}}` |
| Mistral | `[TOOL_CALLS] [{…}]` |
| Qwen3-Coder | `<function=name><parameter=key>value</parameter></function>` |
| GLM-4.7 | `<arg_key>k</arg_key><arg_value>v</arg_value>` |
| DeepSeek | `<｜DSML｜invoke name="…"><｜DSML｜parameter name="…">…` |
| MiniMax | `<minimax:tool_call><invoke name="…"><parameter name="…">…` |
| gpt-oss | `<\|channel\|>commentary to=functions.name<\|message\|>{…}<\|call\|>` |
| Gemma 4 | `<\|tool_call>call:name{k: v}<tool_call\|>` |

Every call in a response is returned, not just the first, with ids
`call_0`, `call_1`, … — nothing in these formats carries an id, and a
client correlates by index. A call naming a tool the request never
offered is dropped rather than forwarded; a namespaced name
(`skills:read`) is forwarded, because the client is what resolves the
namespace.

The XML-ish formats state no types, so a parameter's value is typed
from the tool's own `parameters` schema: a declared `string` is handed
over verbatim, which is what keeps a zero-padded id like `"018956"`
from arriving as a number.

## Not yet

Anthropic streaming/tools/images · full JSON schema / grammar ·
`tool_choice=required` · dedicated embedding models · multi-GPU / TP / PD ·
incremental `tool_calls[].index` argument deltas (calls are emitted
whole, on completion).

See [`ROADMAP.md`](ROADMAP.md).
