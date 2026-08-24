# Agents & IDEs

Point coding agents at a running `ferrox-server`.

```bash
cargo build -p ferrox-server --release --features metal
./target/release/ferrox-server -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8383
```

**OpenAI-compatible base URL:** `http://127.0.0.1:8383/v1`

| Client | Setting |
|---|---|
| Cursor / OpenAI SDK | `baseURL` → `http://127.0.0.1:8383/v1` |
| OpenCode / Cline / similar | OpenAI-compatible provider → same URL |
| Anthropic SDK | `baseURL` → `http://127.0.0.1:8383`, use `POST /v1/messages` |
| curl | `POST /v1/chat/completions` (see [CLI.md](CLI.md)) |

If `FERROX_API_KEY` is set, send `Authorization: Bearer …`.

Also available: `POST /v1/tokenize`, `/v1/detokenize`, `/v1/embeddings`
(Decoder pool), `/v1/messages` (non-stream text).

For a browser client, Ferrox Studio lives in [`ui/`](../ui) as a
separate app. This server does not serve it, and `GET /` here is a 404.
Run `npm run dev` from that directory.

## Continuous batching

```bash
FERROX_CONTINUOUS_BATCHING=1 ./target/release/ferrox-server -m model.gguf …
```

- Mutually exclusive with `FERROX_KV_POOL_BLOCKS` and `FERROX_PREFIX_CACHE_ENTRIES`
- GGUF Decoder only (Kimi / MLA ignore CB)

Full API matrix: [API.md](API.md).
