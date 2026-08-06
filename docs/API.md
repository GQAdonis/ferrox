# OpenAI API compatibility

Ferrox-server exposes a practical OpenAI-compatible subset for chat
serving. This matrix is the contract: fields marked **reject** return
HTTP 400 with a clear message; **ignore** are accepted but have no
effect (documented here so clients are not surprised).

## Endpoints

| Endpoint | Status |
|---|---|
| `GET /health` | Supported |
| `GET /` / `GET /ui` | Supported when `--ui-server` or `FERROX_UI=1` (static chat UI) |
| `GET /v1/models` | Supported |
| `POST /v1/chat/completions` | Supported (non-stream + SSE) |
| `POST /v1/completions` | Supported (minimal: `prompt`, `max_tokens`, sampling subset) |
| `POST /v1/tokenize` | Supported (`prompt` → `tokens` + `count`) |
| `POST /v1/detokenize` | Supported (`tokens` → `text`) |
| `POST /v1/embeddings` | Supported for GGUF `Decoder` only — mean/last pool of final-normed hidden states (pre-`lm_head`). Not a dedicated embedding model; Kimi / other engines return 501 |
| `POST /v1/messages` | Supported (Anthropic-shaped; non-stream text only) |
| `GET /cache/stats`, `GET /metrics` | Ferrox extensions |
| Audio / images / multimodal | Not implemented |

## `chat/completions` request fields

| Field | Status |
|---|---|
| `model`, `messages`, `max_tokens` | Supported |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop` | Supported |
| `presence_penalty`, `frequency_penalty` | Supported (OpenAI-style logit penalties on generated history) |
| `stream` | Supported — overlapped SSE when tools inactive and continuous batching off |
| `tools` / tool-call stop markers | Supported (prompt-engineered; not grammar-constrained) |
| `tool_choice: "none"` / `"auto"` | Supported |
| `tool_choice: "required"` / named function | **Reject** (501) — needs constrained decoding |
| `logprobs` / `top_logprobs` | **Reject** until implemented |
| `n` (>1) | **Reject** |
| `response_format: { "type": "json_object" }` | Supported (best-effort: JSON-safe token mask + post-validate; not full grammar) |
| `response_format` (other types) | **Reject** |
| `session_id` | Ferrox extension (server-side history) |

## Continuous batching

Opt-in via `FERROX_CONTINUOUS_BATCHING=1`. Mutually exclusive with
`FERROX_KV_POOL_BLOCKS` and `FERROX_PREFIX_CACHE_ENTRIES`. Throughput
receipt helper: [`benchmarks/cb_throughput.py`](../benchmarks/cb_throughput.py).

## MCP tool servers

`--mcp-config PATH` loads a JSON list of MCP servers and exposes stub
metadata under `ferrox_mcp` in `GET /v1/models`. Tool invocation is
**not implemented** yet (planned P9+).

## Planned (roadmap)

Not implemented. Listed so clients and contributors know the intended
surface; rows stay **Not implemented** / **Reject** until code + tests
land.

| Item | Phase | Notes |
|---|---|---|
| Anthropic Messages streaming / tools / images | P9+ | Non-stream text `/v1/messages` shipped |
| JSON schema / full grammar constrained decode | P9+ | `json_object` best-effort shipped |
| MCP tool invocation | P9+ | Config + models metadata stub shipped |
| `tool_choice=required` | P9+ | Needs grammar |
| Continuous-batching default + pin | P10 | See CB section above |
| Dedicated embedding models / base64 encoding | later | `/v1/embeddings` today pools decoder hiddens only |
| Multi-GPU / tensor parallel / prefill-decode disaggregation | P10+ | Documented in ROADMAP only |
