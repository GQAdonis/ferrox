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

## Planned (roadmap P8–P9)

Not implemented. Listed so clients and contributors know the intended
surface; rows stay **Not implemented** / **Reject** until code + tests
land. Do not treat this section as Supported.

| Item | Phase | Notes |
|---|---|---|
| `POST /v1/embeddings` (mean / last pool) | P8 → P9 | Needs embeddings engine first |
| Anthropic Messages API | P9 | Rewrite in-tree |
| `POST /v1/tokenize`, `/detokenize` | P9 | Thin wrappers over existing tokenizer |
| `POST /v1/completions` | P9 | Legacy completions |
| Guided decode / JSON schema / grammar | P9 | Unlocks `response_format` + `tool_choice=required` |
| MCP tool servers (`--mcp-config`) | P9 | External tool attach |
| Built-in web UI (`--ui-server`) | P9 | Static chat UI; `ferrox chat` remains first-class |
| `presence_penalty` / `frequency_penalty` | P9 | Fill current Rejects |
| Continuous-batching default + pin | P10 | See CB section above |