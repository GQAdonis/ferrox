# API

`ferrox-server` exposes an OpenAI-compatible HTTP API for chat serving.

Fields marked **Reject** return HTTP 400/501 with a clear error.
Unsupported multimodal input is rejected the same way.

## Endpoints

| Endpoint | Status |
|---|---|
| `GET /health` | Supported |
| `GET /` · `GET /ui` | Static chat UI (`--ui-server` or `FERROX_UI=1`) |
| `GET /v1/models` | Supported |
| `POST /v1/chat/completions` | Supported (JSON + SSE) |
| `POST /v1/completions` | Supported (`prompt`, `max_tokens`, sampling subset) |
| `POST /v1/tokenize` | Supported |
| `POST /v1/detokenize` | Supported |
| `POST /v1/embeddings` | Supported for GGUF Decoder (mean/last pool of hidden states) |
| `POST /v1/messages` | Anthropic-shaped; non-stream text |
| `GET /cache/stats` · `GET /metrics` | Ferrox extensions |
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

## Continuous batching

Set `FERROX_CONTINUOUS_BATCHING=1`. Mutually exclusive with
`FERROX_KV_POOL_BLOCKS` and `FERROX_PREFIX_CACHE_ENTRIES`.

## MCP

`--mcp-config PATH` loads server metadata under `ferrox_mcp` in
`GET /v1/models`. Tool invocation is not wired yet.

## Not yet

Anthropic streaming/tools/images · full JSON schema / grammar ·
`tool_choice=required` · dedicated embedding models · multi-GPU / TP / PD.

See [`ROADMAP.md`](ROADMAP.md).
