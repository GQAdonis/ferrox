# Agent / IDE cookbooks

Point coding agents at a running `ferrox-server` OpenAI-compatible base URL.

```bash
cargo build -p ferrox-server --release --features metal   # or without metal
FERROX_MODEL_PATH=/path/to/model.gguf FERROX_ADDR=127.0.0.1:8383 \
  ./target/release/ferrox-server
```

Base URL for tools: `http://127.0.0.1:8383/v1`

| Client | Setting |
|---|---|
| Cursor / OpenAI SDK | `baseURL` / `base_url` → `http://127.0.0.1:8383/v1` |
| OpenCode / Cline / similar | OpenAI-compatible provider → same base URL + any model id from `GET /v1/models` |
| curl smoke | `POST /v1/chat/completions` with `messages` (see [`CLI.md`](CLI.md)) |

Optional: `FERROX_API_KEY` → send `Authorization: Bearer …`.

Tokenizer helpers: `POST /v1/tokenize`, `POST /v1/detokenize`.  
Decoder embeddings (mean/last pool): `POST /v1/embeddings` — not a dedicated embed model.

API matrix: [`API.md`](API.md).
