# OpenAI API compatibility

Ferrox-server exposes a practical OpenAI-compatible subset for chat
serving. This matrix is the contract: fields marked **reject** return
HTTP 400 with a clear message; **ignore** are accepted but have no
effect (documented here so clients are not surprised).

## Endpoints

| Endpoint | Status |
|---|---|
| `GET /health` | Supported |
| `GET /v1/models` | Supported |
| `POST /v1/chat/completions` | Supported (non-stream + SSE) |
| `GET /cache/stats`, `GET /metrics` | Ferrox extensions |
| `POST /v1/completions` | Not implemented |
| `POST /v1/embeddings` | Not implemented (no embedding engine) |
| Audio / images / multimodal | Not implemented |

## `chat/completions` request fields

| Field | Status |
|---|---|
| `model`, `messages`, `max_tokens` | Supported |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop` | Supported |
| `stream` | Supported — overlapped SSE when tools inactive and continuous batching off |
| `tools` / tool-call stop markers | Supported (prompt-engineered; not grammar-constrained) |
| `tool_choice: "none"` / `"auto"` | Supported |
| `tool_choice: "required"` / named function | **Reject** (501) — needs constrained decoding |
| `logprobs` / `top_logprobs` | **Reject** until implemented |
| `n` (>1) | **Reject** |
| `presence_penalty` / `frequency_penalty` | **Reject** (use `repetition_penalty`) |
| `response_format` / JSON mode | **Reject** |
| `session_id` | Ferrox extension (server-side history) |

## Continuous batching

Opt-in via `FERROX_CONTINUOUS_BATCHING=1`. Mutually exclusive with
`FERROX_KV_POOL_BLOCKS` and `FERROX_PREFIX_CACHE_ENTRIES`. Throughput
receipt helper: [`benchmarks/cb_throughput.py`](../benchmarks/cb_throughput.py).
