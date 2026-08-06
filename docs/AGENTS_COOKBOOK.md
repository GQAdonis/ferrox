# Agent / IDE cookbooks

Point coding agents at a running `ferrox-server` OpenAI-compatible base URL.

```bash
cargo build -p ferrox-server --release --features metal   # or without metal
FERROX_MODEL_PATH=/path/to/model.gguf FERROX_ADDR=127.0.0.1:8383 \
  ./target/release/ferrox-server
```

Base URL for tools: `http://127.0.0.1:8383/v1`

Optional static browser UI (same host):

```bash
./target/release/ferrox-server -m model.gguf --ui-server
# or FERROX_UI=1 — open http://127.0.0.1:8383/ or /ui
```

| Client | Setting |
|---|---|
| Cursor / OpenAI SDK | `baseURL` / `base_url` → `http://127.0.0.1:8383/v1` |
| OpenCode / Cline / similar | OpenAI-compatible provider → same base URL + any model id from `GET /v1/models` |
| Anthropic SDK / agents | `baseURL` → `http://127.0.0.1:8383` and use `POST /v1/messages` (non-stream text today) |
| curl smoke | `POST /v1/chat/completions` with `messages` (see [`CLI.md`](CLI.md)) |

Optional: `FERROX_API_KEY` → send `Authorization: Bearer …`.

Tokenizer helpers: `POST /v1/tokenize`, `POST /v1/detokenize`.  
Decoder embeddings (mean/last pool): `POST /v1/embeddings` — not a dedicated embed model.

### Continuous batching caveats

- Opt-in: `FERROX_CONTINUOUS_BATCHING=1` on the server process.
- **Mutually exclusive** with `FERROX_KV_POOL_BLOCKS` and `FERROX_PREFIX_CACHE_ENTRIES` (server logs a warning and keeps the private generate path).
- GGUF `Decoder` only; Kimi / MLA ignore CB.
- Throughput smoke: `python3 benchmarks/cb_throughput.py --url http://127.0.0.1:8383` (needs CB-enabled server; writes receipt under `benchmarks/receipts/` when both modes succeed).

API matrix: [`API.md`](API.md).
