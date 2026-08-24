# ferrox-server

OpenAI-compatible HTTP server for Ferrox (`ferrox-server` binary).

Serves chat/completions against a GGUF (or Kimi safetensors dir) via
`FERROX_MODEL_PATH`. See [`docs/API.md`](../../docs/API.md).

## The web UI is a separate app

This binary serves the HTTP API and nothing else. `GET /` is a 404 like
any other unknown path. **Ferrox Studio**, the chat / models / activity /
connect frontend, lives at [`ui/`](../../ui) in the repository root and
reaches this server over the same public API an editor would use.

```bash
cargo run -p ferrox-server -- -m model.gguf     # terminal 1
cd ui && npm install && npm run dev             # terminal 2 -> :5173
```

`npm run dev` proxies `/v1`, `/admin`, `/health`, `/metrics` and
`/cache` to `127.0.0.1:8383`, so the browser sees one origin and CORS
never applies. Serving the built `ui/dist/` from a **different** origin
does bring CORS into it. Start this server with
`FERROX_CORS_ORIGINS=<that exact origin>`. A `*` wildcard there is a
startup error, on purpose: a wildcard beside a bearer token is a
credential-leak shape. See [`docs/CONFIG.md`](../../docs/CONFIG.md).
