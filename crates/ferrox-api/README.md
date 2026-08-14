# ferrox-api

The wire contract between `ferrox-server` and its clients.

Route constants and serde DTOs shared by the server, the UI it serves at
`/`, the desktop shell that spawns it, and `ferrox chat` — so a path or
a payload shape has exactly one definition instead of one per end.

Contents:

- `routes` — every path the server serves, named once.
- `health` — the three-state `/health` handshake (`ready` /
  `detecting` / `unavailable`) with per-capability machine reason codes
  and human sentences.
- `lifecycle` — the `ferrox.server.ready` stdout line carrying the
  actually-bound address and pid, which is what makes `--port 0` usable
  by a supervising process.
- `usage` — token accounting plus llama.cpp-style timings, with prefill
  and decode kept separate.
- `request_id` — server-assigned ids, stated in the first streamed chunk.
- `progress` — rolling-window transfer rate/ETA that reports nothing
  until it can report something true.

Depends on serde only: a CLI, a desktop shell and a browser-targeted
frontend may all link it.
