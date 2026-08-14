---
name: ferrox UI — web, desktop, one frontend
overview: "GOAL: one web frontend, embedded in ferrox-server and served at `/`, plus a Tauri 2 shell that wraps the same frontend into a native app for macOS/Windows/Linux. First cut: chat + model manager + live metrics. Every UI feature must be reachable through the public HTTP API — the UI calls `/v1/chat/completions` like any other client, so the API cannot rot without the UI breaking first. Sourced from a read-only study of oMLX (Python/MLX, Swift menubar) and Unsloth Studio (React + FastAPI + Tauri 2) under `.scratch/`; both were read for contracts only. LICENCE RULE: Unsloth `studio/` is AGPL-3.0 and ferrox is Apache-2.0 — read the shapes, copy no code."
todos:
  - id: ui-api-contract-crate
    content: "Shared route-constant + DTO crate so app and server cannot drift (Unsloth has this implicitly; oMLX's Endpoints.swift is 106 lines and is the whole app-server contract). LANDED 2e8c214: `crates/ferrox-api` (serde-only) with routes / health / lifecycle / usage / request_id / progress. ferrox-server's router and Usage now come from it. Only implemented routes get a constant — a constant for a missing endpoint reads as a promise. The OpenAI request/response bodies deliberately stay in ferrox-server where they are validated"
    status: completed
  - id: ui-health-capability
    content: "GET /health with THREE states: ready, unavailable+reason, and `detecting` while backends are probed. A guessed 'CPU only' on first paint is visually identical to a measured one. Hard 1s budget, then answer provisionally. LANDED 21a175c: `/health` returns the handshake instead of the string 'ok'. Probe runs once in the background, handler never blocks; before the 1s budget no GPU verdict is offered at all, after it the answer is filled in with reason `detection_timed_out`, and a late probe replaces that. Capabilities cpu/metal/cuda/real_weights/continuous_batching each carry a machine reason + a human sentence, so metal_unavailable / metal_not_built / disabled stay three different answers. `last_request_age_seconds` lets a saturated backend vouch for its own liveness. NOW REACHABLE (1a59c15): POST /admin/models/unload leaves nothing loaded, and /health answers 503 `unavailable` with reason `model_not_loaded` and a null model, instead of the 200 `ready` a supervisor would have acted on"
    status: completed
  - id: ui-server-lifecycle
    content: "`--port 0` + a structured stdout line carrying the bound addr/pid, and stdin-close => exit. Deletes the entire port-conflict UI and all OS-specific signal reaping. LANDED ff8f10d: both serve paths (plain + TLS) bind first and read the address back off the socket, then print the `ferrox.server.ready` JSON line (addr/port/scheme/pid/version). `--exit-on-stdin-close` / FERROX_EXIT_ON_STDIN_CLOSE=1 is OPT-IN — a /dev/null stdin (systemd, cron, nohup) is EOF at startup, so defaulting it on would make the server exit as it starts. Graceful shutdown added to both paths; runtime is shut down in the background since the stdin watcher parks in a blocking read. Verified end to end. NOT DONE: the preflight disposition (managed/owned/attached/external) is supervisor-side and belongs with ui-tauri-shell"
    status: completed
  - id: ui-request-id-stream
    content: "Emit `request_id` in the first SSE chunk and key metrics by the same id. oMLX's chat page reverse-engineers this with a claiming heuristic; ferrox can just say it. LANDED 233328f: every /v1/chat/completions request gets a `chatcmpl-…` id (was the constant 'ferrox-demo-0' for everyone). Stated as `id` on every chunk and once as `request_id` on the first one, before any content. PARTIAL: metrics are not yet keyed by it — that needs ui-api-monitor's ring buffer"
    status: completed
  - id: ui-usage-timings
    content: "Extend `usage` with llama.cpp-style timings (ttft, prompt_eval_duration, generation_duration, tok/s both phases, cached_tokens) in both non-streaming and the include_usage final chunk. Server-reported beats client wall-clock. LANDED 233328f: prompt_eval_duration_ms / generation_duration_ms / time_to_first_token_ms / cached_tokens beside the existing per-phase rates, on the Decoder path AND the Kimi/MLA engine path (which reported no timings at all). TTFT is stamped inside the decode step closure — the real moment a token existed — not approximated as prefill time. cached_tokens distinguishes a prefix-cache miss (Some(0)) from no prefix cache (absent). Untimed usage still serializes to exactly the three OpenAI fields"
    status: completed
  - id: ui-cancel-two-tier
    content: "Cancellation: AbortSignal on the socket PLUS an explicit POST /cancel with keepalive. A dropped TCP connection does not reliably stop a decode loop"
    status: pending
  - id: ui-sse-hardening
    content: "SSE with id:/retry:/Last-Event-ID replay, a stall timeout, and a polling fallback beside every stream — reverse proxies buffer text/event-stream. Both reference projects hit this in production"
    status: pending
  - id: ui-admin-models
    content: "GET /admin/models: per model id, path, loaded/loading, on-disk + resident size, backend, context length, quant, and capability flags each PAIRED WITH A HUMAN-READABLE REASON so the UI never re-derives why a control is disabled. LANDED 1a59c15: header-only discovery over FERROX_MODEL_DIR + the FERROX_MODEL_PATH directory (GgufFile::open parses metadata and tensor descriptors, never a weight), non-recursive, split checkpoints folded into one entry, safetensors-index directories included. Every optional field is null-not-absent, because an unknown context length and a zero one are different facts. quant prefers general.file_type (the only place the _M in Q4_K_M is stated) and falls back to the measured dominant tensor dtype rather than guessing from the filename. NOT DONE: resident_bytes is always null and says so — checkpoints are mmap-resident, so the true figure is a page-cache property this process cannot read and the file size would be a lie in both directions. NOT DONE: per-model capability flags; the reason-carrying capability list is on /health, which is where the UI already reads it"
    status: completed
  - id: ui-model-swap
    content: "Model load/unload/swap. ferrox-server's model.rs::load() is single-shot from env today; this is the real backend work behind the model manager. LANDED 1a59c15: AppState.active is an RwLock<Option<Arc<ActiveModel>>> and the rule is in its doc comment — a reader clones the Arc under the read lock and then runs, so the lock guards the POINTER and is never held across a decode. An in-flight request finishes against the weights it started on and the old model is freed when the last holder lets go; no attempt is made to migrate a running request, because half a completion from each of two checkpoints is worse than either. Tested: the old handle still decodes after a swap, and strong_count shows the swapped-out weights outliving the registry reference. ActiveModel bundles the continuous batcher with the model because the batcher thread holds one specific Arc<Decoder>. model.rs::load_from_path() is the env-free entry point; activate_loaded_model() is shared by startup and the load task so an engine variant cannot be wired into one and forgotten in the other. Load is by discovered id only — there is deliberately NO load-by-path endpoint. Unload makes /health reach the `unavailable` state Phase 1 defined but could not exercise. NOT DONE: the KV pool is still sized from the startup model, and no warmup prefill runs after a swap (ui-warmup is unclaimed)"
    status: completed
  - id: ui-task-contract
    content: "ONE long-running-task contract reused by download/convert/bench: POST start -> {task_id}, GET tasks, POST cancel/{id}, fields {status, progress, bytes_done, bytes_total, error, timestamps, retry_count}, plus a per-task event stream. LANDED 1a59c15: one TaskView shape for both job kinds (download, load). Terminal is terminal — a late update from a worker that has not noticed its cancellation is dropped, so a UI that stopped polling is never wrong. Progress comes from ferrox_api::progress::RateEstimator and nowhere else, so a warming window reports null rate and null ETA by construction (ui-transfer-stats finally has its consumer). Cancellation is cooperative and honest: a download stops within a chunk and keeps its .part file, a model load cannot be interrupted mid-mmap and so discards its finished result rather than pretending it stopped early, and a task only reaches `cancelled` when a worker acknowledges it. A TaskGuard rides with every worker so a panic cannot leave a task at `running` forever — nothing awaits a spawn_blocking handle. NOT DONE: no per-task event stream (polling only) and no retry_count — nothing retries yet, and a field that is always 0 is a promise, not a contract. convert/bench are not task kinds because neither job exists"
    status: completed
  - id: ui-transfer-stats
    content: "Rolling-window rate/ETA smoothing: >=3 samples over >=3s before reporting `stable`, reset on counter regression, ETA clamped. Prevents the '123 GB/s' flash on the first tick. Reimplement from the description — the reference is AGPL. LANDED 2e8c214: `ferrox_api::progress::RateEstimator`, written from this description only. A warming report carries NO number at all rather than an untrusted one, which makes the flash structurally impossible; window trims by age and drops from the middle on overflow so a fast-ticking job still reaches `stable`; ETA needs a known total and a positive rate. CONSUMER LANDED 1a59c15: ferrox_api::TaskProgress can only be built from a RateReport, so every /admin/tasks progress block inherits the refusal by construction — verified end to end on a real 396 MB Hub download, which reported no rate for the first four seconds and then a rate and an ETA"
    status: completed
  - id: ui-frontend-chat
    content: "Chat: streaming, markdown+code, reasoning blocks, sampling controls, model switch mid-conversation, conversation tree persisted SERVER-side (SQLite), not localStorage"
    status: pending
  - id: ui-api-monitor
    content: "API Monitor screen: ring-buffered live request log. Keep duration_ms and decode_ms SEPARATE (duration carries queue wait + prefill; conflating them reads a 50 tok/s model as 5) and flag external-vs-UI traffic. BACKEND LANDED 1a59c15: GET /admin/stats serves a 200-entry ring keyed by the request_id the response already carried, so a row joins its message by equality instead of the claiming heuristic oMLX needs. duration_ms and decode_ms are separate fields all the way to the wire, and decode_ms/ttft_ms are null rather than a copy of the total when the engine did not time itself. Recorded for /v1/chat/completions (streamed and not, including rejections), /v1/completions and /v1/messages. NOT DONE: the screen itself; via_api_key / external-vs-UI attribution (nothing records which key served a request); a queue gauge; /v1/embeddings, /v1/tokenize and /v1/detokenize are not recorded"
    status: pending
  - id: ui-embed-server
    content: "Embed the built frontend into the ferrox-server binary (rust-embed) and serve at `/` with an SPA fallback; replaces the 72-line static/ui.html stub"
    status: pending
  - id: ui-tauri-shell
    content: "Tauri 2 shell: bundles the same frontend, spawns ferrox-server as a child, targets app/dmg/deb/nsis. Backend state machine incl. `unresponsive` as distinct from `failed` — a hung server must not trigger a restart"
    status: pending
  - id: ui-settings-snippets
    content: "Settings > Connect: copy-pasteable curl / Python-SDK / IDE snippets prefilled with the live base URL, key and loaded model. Highest-leverage screen for a server whose job is being someone's OpenAI endpoint"
    status: pending
  - id: ui-cross-platform-truth
    content: "BLOCKING HONESTY: Windows/Linux GPU means CUDA, which the parity plan keeps only at 'must compile' — no receipts, no host pin. A desktop app shipping to Windows today is CPU-only in practice. State it, do not imply otherwise. LANDED: stated in docs/FEATURES.md beside the backend table, and /health now says the same per capability with a reason string instead of silently greying a control. Still to state on a download page if one ever ships (ui-tauri-shell)"
    status: completed
isProject: false
---

# ferrox UI — web, desktop, one frontend

> One frontend. Served by `ferrox-server` at `/` (that is the web UI) and
> bundled by a Tauri 2 shell into a native app for macOS, Windows and
> Linux. Written **2026-08-13** from a read-only study of two shipped
> products under `.scratch/`: **oMLX** (Python/MLX, Swift menubar app)
> and **Unsloth Studio** (React 19 + FastAPI + Tauri 2).
>
> **Licence rule, non-negotiable.** Unsloth's `studio/` is **AGPL-3.0**;
> ferrox is Apache-2.0. Everything below is a description of a *contract*
> or a *behaviour*, arrived at by reading. No code is copied, and none may
> be. oMLX is Apache-2.0 but is MLX-bound, so nothing of it is liftable
> either — same rule applies by accident rather than by licence.

## Why this shape

Both reference products converged on it from opposite directions:

- **oMLX's native Swift app has no chat screen at all.** 37k lines of
  Swift across 122 files, and chat + dashboard are ordinary HTML pages
  served by the backend and opened in the user's *default browser* —
  there is not even a WKWebView. The app is a supervisor and a settings
  panel. Its own portable UI turned out to be the browser layer.
- **Unsloth Studio built exactly the target architecture**: one React SPA
  served by FastAPI for web, and a Tauri 2 shell that bundles the same
  SPA and spawns the backend as a child process. 215k LOC of frontend
  with five TODO markers in the whole tree — a shipped product, not a
  demo.

ferrox starts from a better position than either: **one static Rust
binary**. oMLX's packaging chapter — venvstacks, `PYTHONHOME` relocation,
`install_name_tool` surgery on broken wheels — and Unsloth's 285 KB
`install.sh` that provisions a Python interpreter at install time both
exist to solve a problem ferrox does not have. Nothing in this plan may
reintroduce it.

## Landed so far (2026-08-14)

Phase 1a, 1b and 1c are in, plus the contract crate they all write to.

- **`crates/ferrox-api`** — routes, health/capability DTOs, the ready
  line, `Usage`, request ids, the rate estimator. serde-only, so a CLI,
  a desktop shell and a browser frontend can all link it. The server's
  router and `Usage` come from it, so a path or payload has one
  definition rather than one per end.
- **`GET /health`** is the three-state handshake. Probing happens once
  in the background under a 1s budget and the handler never blocks;
  before the budget it offers *no* GPU verdict, after it answers
  provisionally with `detection_timed_out`. `unavailable` is defined but
  unreachable until model swap exists — the port only binds after a
  model is loaded.
- **`--port 0` + the `ferrox.server.ready` stdout line**, and opt-in
  `--exit-on-stdin-close`. Verified end to end (kernel-assigned port,
  requests served on it, exit 0 when the parent closes the pipe).
- **`request_id`** and **per-phase `usage` timings** on chat
  completions, including the Kimi/MLA engine path.
- The **honesty clause** is stated in `docs/FEATURES.md`.

Deliberately not started: cancellation, SSE hardening, the admin
surface, model swap, the task contract, and everything in Phases 2 and
3. Model swap is the gate on the whole model-manager screen and is the
next real backend work.

## What exists today

- `ferrox-server` already serves `/` and `/ui` from
  `crates/ferrox-server/static/ui.html` — **72 lines**, one input, one
  log, posts to `/v1/chat/completions`. That is the thing being replaced.
- Already present and reusable: `/v1/chat/completions` (SSE),
  `/v1/models`, `/v1/completions`, `/v1/embeddings`, `/v1/tokenize`,
  `/v1/detokenize`, Anthropic `/v1/messages`, `/metrics` (Prometheus —
  which **oMLX has no equivalent of**), `/cache/stats`, `/health`,
  server-side `session.rs`, `batch_scheduler.rs` (continuous batching,
  opt-in), prefix cache, response cache.
- ~~Missing and load-bearing for the first cut: **model swap**.~~
  **Landed 1a59c15.** `model.rs::load()` still reads env at startup, but
  `load_from_path()` beside it takes a path, and `AppState.active` is a
  swappable `Arc` behind an `RwLock` held only long enough to clone a
  handle. `/admin/models`, `/admin/models/load|unload`,
  `/admin/download`, `/admin/tasks`, `/admin/tasks/{id}/cancel` and
  `/admin/stats` exist and are documented in `docs/API.md`. Phase 2 is
  unblocked.

## Architecture

```
ferrox-server (one Rust binary)
  ├─ /v1/*        OpenAI + Anthropic API   (exists)
  ├─ /admin/*     control API              (new)
  ├─ /metrics     Prometheus               (exists)
  └─ /            embedded SPA, rust-embed (new)

Web     = open http://127.0.0.1:8383
Desktop = Tauri 2 shell → bundled SPA → spawns ferrox-server as a child
Both    = one frontend codebase, one streaming path
```

### The rule that makes it work

**The UI is just another API client.** It calls `/v1/chat/completions`
with the same token an external client would use — never a private
`/api/chat`. Unsloth mounts one router at two prefixes for exactly this
reason, and it is the single highest-value decision in their codebase:
every UI feature is automatically an API feature, the public contract
cannot rot without the UI breaking first, and there is one streaming code
path to debug instead of two.

## Phase 1 — backend contracts (before any frontend work)

### 1a. Health as a capability handshake, with three states

Not two. `ready` / `unavailable + reason` / **`detecting`**. The third is
the one both products learned the hard way: while backends are being
probed, rendering the *guess* blacks out GPU-dependent controls on first
paint in a way that is visually identical to a measured "unsupported".
The UI must be able to *hold*.

- Liveness (port bound) and readiness (model loaded) are separate answers.
  oMLX returns 503 `"loading"` while preloading, because the port binds
  before the engine is ready.
- Hard **1s budget** on detection, then answer provisionally — the
  desktop shell probes with a short timeout and does not retry.
- Capability bits carry a machine reason *and* a human sentence:
  `metal_unavailable`, `cuda_not_built`, `cpu_only`. The UI greys the
  control **with the reason in a tooltip**; it never silently hides it and
  never re-derives the explanation.
- **Auxiliary health.** A saturated GPU can starve `/health`. Let the
  cheap status endpoint the UI already polls *vouch* for liveness within a
  freshness window, so the app does not declare a busy backend dead. This
  is a real bug class in oMLX's own history, not polish.

### 1b. Process lifecycle that needs no OS-specific code

- **`--port 0`** plus a structured line on stdout carrying the actually
  bound address and pid. This deletes oMLX's entire port-conflict feature:
  the `lsof` shell-out, the owner identification, the alert dialog, and a
  `killExternal()` helper that was written and never wired to anything.
- **Stdin close ⇒ exit.** The one orphan-prevention mechanism that behaves
  identically on macOS, Windows and Linux, and survives a parent crash.
  oMLX needs POSIX signal handlers plus an `atexit` trampoline plus a
  synchronous reaper; Windows has no SIGTERM at all.
- Preflight disposition when a server is already on the port:
  `managed_ready | owned_ready | attached_ready | external_conflict`,
  distinguished by an opaque per-install id so "mine" and "a stranger's"
  are different answers.

### 1c. Streaming contract

- **`request_id` in the first chunk**, and metrics keyed by the same id.
  oMLX's chat page has to claim request ids out of a stats snapshot with a
  heuristic so concurrent chats do not steal each other's numbers. ferrox
  can simply state it.
- **Two-tier cancellation**: `AbortSignal`/socket close **plus** an
  explicit `POST /cancel` with `keepalive`. A dropped connection does not
  reliably stop a decode loop, and a proxy that swallows the abort leaves
  the backend generating into nothing.
- **Truncation is an error, not a completion.** EOF without `[DONE]` and
  without a `finish_reason` must surface as a retryable failure, never as
  a finished message.
- **SSE hardening**: `id:` + `retry:` + `Last-Event-ID` replay so a
  reconnect resumes rather than restarts; a stall timeout; and a polling
  fallback beside every stream, because reverse proxies buffer
  `text/event-stream`. Both reference projects shipped the fallback after
  hitting this in production.
- **`usage` timings**, both non-streaming and in the `include_usage` final
  chunk: `time_to_first_token`, `prompt_eval_duration`,
  `generation_duration`, prompt and generation tok/s, `cached_tokens`.
  ferrox's kernels already know these; emitting them means the UI never
  wall-clock-guesses. Keep **queue wait + prefill (`duration_ms`) separate
  from decode (`decode_ms`)** — conflating them reads a 50 tok/s model as
  5, and every tok/s number downstream becomes a lie.

### 1d. Admin surface

- `GET /admin/models` — per model: id, path, loaded/loading, on-disk size,
  resident size, backend, context length, quant, plus capability flags
  **each paired with a reason string**.
- `POST /admin/models/{id}/load|unload`, `PUT /admin/models/{id}/settings`.
  This requires making `model.rs::load()` swappable, which is the real
  work of the first cut.
- **One task contract** for every long job (download, convert, bench):
  `POST …/start → {task_id}`, `GET …/tasks`, `POST …/cancel/{id}`, with
  `{status, progress, bytes_done, bytes_total, error, timestamps,
  retry_count}` and a per-task event stream. Uniformity is what makes the
  UI cheap; both products reuse one shape across four job types.
- A **non-terminal `finalizing` phase** in every job state machine. For
  model load that is `mmap → dequant → warmup → ready`. A bar pinned at
  100% while work continues is indistinguishable from a hang.
- Rate/ETA smoothing over a rolling window: at least 3 samples spanning at
  least 3 s before a rate is reported `stable`, buffer reset when the byte
  counter goes backwards, ETA clamped at zero. Reimplemented from this
  description — the reference implementation is AGPL.

### 1e. Warmup

Neither product warms up after load, and both pay Metal shader/JIT
compilation on the first real request; both of their *benchmark* paths
warm up explicitly because of it. ferrox has the same exposure and should
do a warmup prefill as part of `finalizing`, not as a user's first token.

## Phase 2 — the frontend

Stack decision deferred to implementation, but constrained: it must build
to static assets embeddable via `rust-embed`, and it must run under a
CSP with no inline script (both reference apps hit this — one loads a
theme bootstrap as an external file purely to satisfy `script-src 'self'`).

Screens, first cut:

1. **Chat** — streaming, markdown + code + math, collapsible reasoning
   blocks, sampling controls, model switch mid-conversation, per-model
   settings that survive the switch. **Conversation state server-side in
   SQLite**, with a `parent_id` per message so edit/regenerate branches
   work. Not `localStorage`: oMLX's chat stores the entire history as one
   JSON blob and *pops conversations off the end* when the quota is
   exceeded — silent data loss by design.
2. **Models** — list with size/quant/backend/context, load/unload,
   download with progress, "will it fit?" **computed exactly from the
   GGUF header** rather than heuristically (ferrox knows `n_layers`,
   `n_kv_heads`, `head_dim`, KV dtype; both reference products guess).
3. **Metrics / API Monitor** — live request log, ring-buffered:
   `{id, endpoint, model, status, started_at, duration_ms, decode_ms,
   ttft_ms, tok_per_sec, prompt_tokens, completion_tokens, context_usage,
   stop_reason, via_api_key}` plus a queue gauge and a server timestamp so
   the browser clock is never trusted. `via_api_key` distinguishes an
   external caller (someone's editor) from the UI's own traffic — the most
   directly useful screen for a server that is already pointed at IDEs.
4. **Settings > Connect** — copy-pasteable curl / Python-SDK / IDE config
   snippets prefilled with the live base URL, key and loaded model.
   `docs/AGENTS_COOKBOOK.md` describes this today; a screen that *emits*
   it is strictly better, and a `ferrox launch <tool>` that writes the
   config file is better still.

## Phase 3 — the desktop shell

Tauri 2, bundling the same frontend, spawning `ferrox-server` as a child.
Targets `app` / `dmg` / `deb` / `nsis`. Single-instance, window-state,
CSP `connect-src` allowing `http://127.0.0.1:*`.

Backend state machine — **`unresponsive` must be a state distinct from
`failed`**, because a hung backend must not trigger a restart loop:

```
checking → starting → running(pid) → unresponsive(pid)
                   ↘ failed(msg)   ↘ stopping → stopped
```

Restart backoff 5/10/20 s, cap 3, counter reset after 60 s stable, and an
`expectingExit` flag so a user-initiated stop does not read as a crash.
A startup screen driven off the backend's own stderr lines
("Loading model… / Building Metal pipelines… / Warming up…").

**Do not inherit:** unix-domain control sockets (no clean Windows
equivalent), `lsof` for port ownership, `hdiutil`-based DMG self-update
with an unsigned swap, or an API key passed in a URL query string —
oMLX does the last one for its browser handoff, and also returns the
server's plaintext key in a stats payload.

## The honesty clause

**Windows and Linux GPU means CUDA.** The parity plan keeps CUDA at
"must compile" — no receipts, no host pin, and `ferrox-cli --features
cuda` did not compile at all at one point this cycle. A desktop app
shipped to Windows today is **CPU-only in practice**. That is a scope
decision to state on the download page, not something to imply otherwise
by shipping a Windows installer next to a Metal benchmark table.

Unsloth handles the mirror image of this honestly and it is worth copying:
their macOS build defaults to chat-only, and the UI greys the unavailable
features **with a reason string** rather than hiding them — while their
README says only "works on Windows, Linux, WSL and macOS".

## Sequencing

Phase 1 is entirely backend and is independently useful — `request_id`,
`usage` timings, two-tier cancel, task contract and `--port 0` improve
`ferrox-server` for IDE users whether or not a frontend ever ships.
Phase 2 depends on 1d (model swap) only — **1d has landed**, so Phase 2
is unblocked. Phase 3 depends on 1b.

Do Phase 1 first. It is the part that cannot be replaced later without
breaking clients.
