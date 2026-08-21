# Ferrox Studio — frontend source

This directory is the **source** of the web UI that `ferrox-server`
serves at `/`. Edit the frontend here, not in
`crates/ferrox-server/static/`.

```
ui/                              ← you are here (source)
crates/ferrox-server/static/     ← build output, committed to git
crates/ferrox-server/src/ui.rs   ← rust-embed + the SPA fallback
```

## Working on it

```bash
cd ui
npm install

# Terminal 1 — the real backend
cargo run -p ferrox-server -- -m models/some-model.gguf --ui-server

# Terminal 2 — Vite on :5173, proxying /v1 /admin /health /metrics /cache to :8383
npm run dev
```

`npm run dev` talks to a real `ferrox-server`, not a mock — every screen
here goes through the public HTTP API, which is the rule that keeps the
API from rotting without the UI breaking first.

```bash
npm run typecheck   # tsc, no emit
npm run lint        # eslint
npm run licenses    # refuses any non-permissive dependency
npm run build       # writes ../crates/ferrox-server/static/
```

## Why the build output is committed

`npm run build` writes into `crates/ferrox-server/static/`, and **that
output is committed to git on purpose**: `cargo install ferrox-server`
must work on a machine with no Node and no network beyond crates.io, so
there is deliberately no `build.rs` that shells out to npm. The output
also has to live *inside* the crate directory, because `cargo publish`
only packages files under the crate — a top-level `ui/dist/` would be
absent from the published crate and the installed binary would carry no
UI at all.

The consequence: **if you change anything in `ui/src`, run
`npm run build` and commit the regenerated `static/` alongside it.** CI
fails a pull request whose `static/` does not match its source (see
`.github/workflows/ci.yml`, job `ui`).

Filenames are fixed (`app.js`, `app.css`) rather than content-hashed.
The server serves every asset `Cache-Control: no-cache`, so hashing buys
nothing and would rewrite every filename in the committed folder on each
build.

## Stack

React 19 · Vite · Tailwind v4 · Radix UI primitives (the shadcn/ui
foundation) · `@assistant-ui/react` for the chat transcript ·
TanStack Table for the Activity log · lucide-react icons. Every runtime
dependency is MIT / Apache-2.0 / ISC / BSD — see
`docs/THIRD_PARTY_NOTICES.md`, and `npm run licenses` enforces it.

## Layout

```
src/
  main.tsx              router; / and /ui/<screen> both resolve here
  index.css             design tokens + Tailwind theme (one light block,
                        one prefers-color-scheme block, nothing else)
  lib/api.ts            the ONLY place that talks HTTP
  lib/format.ts         "unknown" is an em dash, never a zero
  components/           app shell, health pill, shadcn-style primitives
  screens/chat/         assistant-ui runtime, markdown, thread
  screens/{models,activity,connect}.tsx
```
