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
