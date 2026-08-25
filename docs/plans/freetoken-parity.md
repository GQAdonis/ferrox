---
name: FreeToken feature parity — what the port left, and in what order
overview: "GOAL: close the distance between `ferrox-edge` (the FreeToken policy port that landed on branch `claude/freetoken-rust-conversion-23u6ng`) and FreeToken's actual feature set. WHAT ALREADY LANDED: ~7,700 lines of Rust, 193 tests, covering the host-side DECISION logic -- radix prefix caches (plain/SWA/hybrid), the q* bandwidth split, the global expert LRU, pool budgets, admission arithmetic, and the reasoning/tool-call parsers. Only the two parsers and the stop-string withhold rule are wired into anything. THE BLOCKER THAT RANKS EVERYTHING BELOW: `ferrox-moe/src/lib.rs:878` says in its own doc comment that `run_expert_placed` 're-uploads each weight matrix to the device from scratch' on every call. There is no persistent GPU expert residency, and there is no CPU MoE path that can run concurrently with a device copy. Without BOTH, the q* premise -- that a step costs max(fetch, cpu) rather than fetch + cpu -- is arithmetically false on this engine, and `ferrox-edge::qstar` is a correct policy with nothing to decide. So Phase C is not 'the hard part to do later', it is the part that makes Phase C's own policy mean anything, and everything in Phase A is ranked ahead of it precisely because Phase A pays off on CPU and Metal TODAY. HONESTY NOTE ON SCOPE: FreeToken is CUDA-only (RTX 30/40/50). Ferrox holds CUDA to 'must compile' with no pinned benchmark host and no published timings (docs/FEATURES.md). Full parity therefore means building a CUDA path ferrox has never had receipts for, on hardware nobody has run this engine on. Phase C should not start until parity-scope-decision is answered in writing. NOT A GOAL: porting FreeToken's Triton/CUDA kernels, its C++ CPU-MoE extension, or its torch model stacks -- ferrox has its own, and a line-by-line port of those is a rewrite of ferrox, not a port into it."
todos:
  - id: parity-scope-decision
    content: "BLOCKING PHASE C, AND ONLY PHASE C. Decide in writing whether CUDA parity with FreeToken is a goal at all. The facts on both sides: FreeToken is CUDA-only and its whole q* thesis is about a PCIe link, which a unified-memory Mac does not have; ferrox's published ledger is CPU and Metal only, and docs/FEATURES.md says CUDA's bar is 'must compile'. If the answer is no, Phase C's items get closed as won't-do and this plan is a Phase A + Phase B plan about prefix reuse and the agent-facing API, which is where its cheap wins are anyway. If the answer is yes, name the benchmark host in the same document, because every Phase C item needs one. Acceptance: a written decision in this file citing which hardware, closed by re-decision and never by preference"
    status: pending
  - id: anchor-checkpoints
    content: "DONE as policy, NOT YET WIRED. `crates/ferrox-edge/src/anchor.rs`, 15 tests. All four pieces: `resolve_anchor_token` (opener -> single token id, None otherwise -- and the single-token restriction is the POINT, it makes detection an integer compare on the sampled id with no detokenization on the hot path); `AnchorState::observe` (first call of a turn only, never a terminal token); `decode_slide` (cadence + first-step overlap guard, the anchor cap, and the drop rule) with `prefill_slide` as its every-batch sibling; and `snapshot_at_anchor` for the recurrent half, whose four gates are each a real failure -- no anchor, a pending snapshot that would be overwritten, an anchor that is not the cached length (unreachable, no node ends there), an anchor that is not page-aligned (belongs to no node). SIGNED ARITHMETIC THROUGHOUT: an early request's threshold is legitimately negative and clamping it to zero before the anchor comparison makes the drop rule fire on requests that have barely started -- found by writing the short-generation test first. TWO PROPERTY TESTS carry the semantics rather than examples: an anchor may only ever hold MORE state than a plain slide (swept over 60 x 28 position/anchor pairs), and held state stays bounded by `2*(window+gap) + interval + 2*page` over a 60k-token generation, which is exactly what the `anchor_checkpoints` term in pool.rs:372 is sized for -- so that term is no longer paying for a fiction. STILL OWED: nothing calls any of it. `observe` needs the sampling loop, `decode_slide` needs a window pool to free INTO (swa-paged-allocator) and a page table to free FROM (cache-manager), and `snapshot_at_anchor` needs a recurrent-state pool ferrox does not have. That is the ordering this plan already has, not a surprise"
    status: completed
  - id: cache-manager
    content: "The consumer contract that makes radix usable at all, and the reason `wire-radix-prefix-cache` cannot just call `match_prefix`. FreeToken's `scheduler/cache.py::CacheManager` owns: the page-base free list; `_allocate` (evict on shortage, then assert the pages arrived); `allocate_paged` (whole pages per request, charged to both pools for a window model); and `cache_req`, which is where the leaks live -- the caller must free `page_indices[old.cached_len .. cached_len]` (pages the tree already had), REPOINT the request's page-table row at the tree's canonical pages for that span, and only then re-lock. Two supporting pieces: `lazy_free_region`, which snapshots the rows it is handed because a later repoint rewrites them, and `check_integrity`, whose exact equality for the window pool catches a leak as well as a double free. `ferrox-edge`'s caches already return exactly the numbers this needs (`InsertResult::cached_len` is documented as 'how much you must free', not 'how much I stored'). Acceptance: a conservation test over a mixed workload -- every page handed out is in the tree or handed back, never both and never neither -- and the integrity assertion enabled in tests"
    status: pending
  - id: swa-paged-allocator
    content: "DONE as policy, NOT YET WIRED. `crates/ferrox-edge/src/window_pool.rs`, 11 tests. `WindowSlotPool` is the full-position -> window-slot map plus its free list: slot 0 reserved as the sentinel (so an unmapped position reads as 'no state' rather than as slot 0's bytes, and a double free is a no-op), FIFO head-take rather than LIFO (a just-freed slot is the one a sliding request is most likely to free again immediately, so LIFO would concentrate churn on a few slots), all-or-nothing allocation with the capacity check BEFORE the first slot moves, and the map over-allocated by one page because an allocation addresses whole pages and the last one can name a position a page past the final token. TWO DEPARTURES FROM UPSTREAM, both deliberate: `translate` handles a negative location explicitly instead of relying on a trailing sentinel row plus negative tensor indexing (same rule, written where a reader can see it), and `alloc` carries a `debug_assert` that the position is currently unmapped, which turns allocating over a live slot -- a leak upstream tolerates silently -- into a test failure. The conservation invariant is an EQUALITY (`available + live == capacity`) and its test proves it catches a LEAK, not only a double free: a `<=` would tolerate exactly the failure that surfaces as 'the pool is full' an hour into a long run with nothing to point at. A 2048-step sliding workload conserves after every single move. STILL OWED: the wiring, which is cache-manager's job"
    status: completed
  - id: wire-radix-prefix-cache
    content: "Replace `ferrox-models::prefix_cache` with `ferrox_edge::radix`. What is there today, by its own admission: 'not a trie/radix-tree structure' (crates/ferrox-models/src/prefix_cache.rs:16), `find_longest_prefix` is an O(n) linear scan over whole-conversation snapshots (:76), and `store` evicts index 0 -- FIFO, despite the module calling it LRU (:131). It also CLONES a `Vec<KvCache>` per entry, so a thousand requests off one system prompt hold a thousand copies of its KV. The radix cache shares the nodes instead and reference-counts the pages under them. Gated on cache-manager. Acceptance: two prompts sharing a 2000-token system prompt hold ONE copy of it, shown as a byte figure in `/cache/stats`, and `ferrox verify` greedy-id parity is unchanged on at least 3 models"
    status: pending
  - id: radix-on-the-batcher
    content: "Remove the exclusivity. Continuous batching is refused today when a KV pool or prefix cache is configured -- 'incompatible with a KV pool or prefix cache' (crates/ferrox-server/src/lib.rs:1376), and the overlap path is disabled for the same reason (:572). That is the current design's honest answer to a real hazard, but it means the two features a serving deployment most wants are mutually exclusive: concurrency OR prefix reuse, never both. Page-granular block sharing is what removes the hazard, which is why this is gated on wire-radix-prefix-cache rather than being its own design. Acceptance: `FERROX_CONTINUOUS_BATCHING=1` with a prefix cache configured starts, serves, and passes the conservation test under concurrency; the refusal string and the two doc comments that explain it are deleted rather than reworded"
    status: pending
  - id: chat-template-kwargs-plumbing
    content: "The template layer already accepts it: `extra` is documented as 'the OpenAI-extension chat_template_kwargs passthrough' (crates/ferrox-models/src/chat_template.rs:142) with a test that it reaches the template (:1008). The SERVER never sends it -- `crates/ferrox-server/src/output.rs:91` notes the consequence, that no template can open a reasoning block in the prompt, so the reasoning parser can only ever see a block the model opened itself. Plumb `chat_template_kwargs` and `reasoning_effort` from the request, then use the three ferrox-edge pieces that exist for it: `resolve_thinking_mode` (tools force thinking), `sanitize_effort` (quantize onto what the checkpoint accepts), `broadcast_effort_spellings`. Acceptance: a request with `reasoning_effort: minimal` against a checkpoint that grades only the OpenAI triple renders with `low` and does not fail, and `force_reasoning` is derived from the resolved mode rather than hardcoded false"
    status: pending
  - id: think-gears-on-models
    content: "`derive_think_gears` is not ported: `/v1/models` says nothing about whether a checkpoint reasons, at what gears, or which is its default, so a client has to guess. The build order is significant and test-pinned upstream: off (if toggleable), then adaptive (if it has one), then either the effort ladder ascending by scale or a bare 'on'; the default is off/adaptive when the bare render matches one, else the probed effort default, else medium, else the last gear. `ferrox_edge::effort::probe_thinking_profile` already produces the profile this reads. Acceptance: `/v1/models` advertises `supported_reasoning_efforts` and `default_reasoning_effort`, and a checkpoint that grades nothing advertises neither rather than an empty list"
    status: pending
  - id: streaming-tool-call-deltas
    content: "`ferrox_edge::ToolCallParser` streams prefix-stable JSON fragments for the four invoke/parameter families and the server throws them away: the chat path calls `parse_output` on the FULL text at the end (crates/ferrox-server/src/lib.rs, the `Ok((finish, usage, full_text))` arm) and emits each call whole. For a coding agent whose argument is a file, that is the difference between watching the write and waiting for it. Needs OpenAI's incremental shape -- `tool_calls[].index`, `function.name` on the first delta, `function.arguments` as fragments after it. Acceptance: a 4 KB argument arrives in >= 10 deltas whose concatenation parses as the final arguments, and the non-streaming body for the same generation is byte-identical to today's"
    status: pending
  - id: cache-status-and-rebuild
    content: "`ferrox-edge::pool` is a complete elastic-memory policy with no endpoint and no caller. There is no `/v1/cache/status` and no `/v1/cache/rebuild` in crates/ferrox-api/src/routes.rs, and nothing sizes the pools with `plan_cache_budget` at load. Wire all three: size at load (MoE-first, KV floor), report the geometry (`ferrox_edge::cache_report` renders it already, including the rule that a column nothing can be said about is DROPPED rather than zero-filled), and accept a live re-split. The failure ordering is the whole point and is already encoded in `validate_rebuild`: every check is arithmetic, up front, so a refused rebuild leaves the engine serving exactly what it was serving. Acceptance: `curl -X POST /v1/cache/rebuild -d '{\"kv\": 65536}'` moves VRAM from the expert cache to KV without a restart, an impossible target is refused with the engine untouched, and a rebuild while busy is refused rather than queued into a crash"
    status: pending
  - id: stats-and-request-ring
    content: "Three small pure modules FreeToken has and ferrox does not: the request ring (a bounded deque with an all-time cursor, p95 over the window, TTFT averaged only over rows that HAVE one), the sliding-window throughput tracker (a window, not a cumulative average, so a quiet server reports 0 rather than its lifetime mean), and the prepare-stop accounting barrier -- close admission, drain, abort what will not drain, seal the totals, and NEVER reopen admission on a timeout. Acceptance: `/v1/stats` and `/v1/requests` serve the documented shapes; a stop with an in-flight request that never terminates leaves the server closed and unsealed rather than silently reopened"
    status: pending
  - id: wire-edge-scheduler
    content: "`ferrox-edge::scheduler` and `ferrox-server::batch_scheduler` currently hold the same policy twice. They already AGREE on the important part -- both are strict FIFO with head-of-line blocking, and both wrote down the same reason (crates/ferrox-server/src/batch_scheduler.rs:96, 'FIFO cannot starve, so FIFO it is'). They differ on what a chunk reserves: ferrox-edge charges the whole remaining prompt plus the whole output budget at admission, the batcher charges `ceil((prompt + max_tokens)/block_size)` blocks for the request's lifetime. Decide which is authoritative and delete the other; do NOT leave two. The window-model rules (a continuation chunk must end on a page boundary, a request reclaims its own slid-out window) exist only in ferrox-edge and are needed the moment a window model is served. Acceptance: one implementation, the batcher's tests still pass against it, and the deleted one is gone rather than deprecated"
    status: pending
  - id: window-slide-during-decode
    content: "Not ported: `maybe_free_swa_out_of_window` and its prefill sibling. A window model's decode must periodically give the window pool back what has slid out -- on a cadence (every N steps, N=128 by default) rather than every step, skipping the first decode step of each request (an overlap guard), never freeing below the tree-owned prefix, and taking `min` with the anchor cap from anchor-checkpoints. Without it a long generation holds window state for its whole history and the pool floor is a fiction. Gated on cache-manager and swa-paged-allocator. Acceptance: a 100k-token generation on a 4k-window model holds bounded window state, asserted as a ceiling rather than a spot check"
    status: pending
  - id: persistent-gpu-expert-cache
    content: "THE PHASE C BLOCKER, AND THE REASON q* IS INERT. `run_expert_placed` 're-uploads each weight matrix to the device from scratch' on every call -- ferrox's own words, crates/ferrox-moe/src/lib.rs:878, disclosed there as 'not yet the persistent-GPU-residency throughput win real expert offload needs'. What is needed: one device-side slot pool per weight bank sized `cache_size x row_shape`, a copy path that executes the `(dst_slot, src_row)` pairs `ferrox_edge::ExpertCache::ensure` already returns, and layer-local host banks so residency can differ per layer. `ferrox-core::expert_store` is the CPU-side half of this and already has the lease/refcount discipline (crates/ferrox-core/src/expert_store.rs:183) -- it hands back `&[u8]`, so what is missing is the device side, not the eviction policy. Gated on parity-scope-decision. Acceptance: a decode step that hits the expert cache issues ZERO weight uploads, shown as a counter, and `ExpertCacheStats::miss_rate` on a real MoE model is reported by `/v1/stats`"
    status: pending
  - id: concurrent-cpu-moe-executor
    content: "THE OTHER HALF OF THE SAME BLOCKER. q* assumes a step costs max(fetch, cpu). Ferrox's CPU MoE runs synchronously inside the forward, so it costs fetch + cpu and the optimal split is trivially 'fetch everything' -- i.e. `qstar` cannot be wrong because it cannot matter. FreeToken's executor is a pinned-buffer worker pool with a doorbell handshake, one thread per PHYSICAL core (SMT siblings share load ports and buy nothing), and it reserves the last core for the coordinator. It also documents why the obvious implementation is wrong: a spin-wait kernel pinned reported GPU utilisation at 99% and laptop power schedulers responded by clamping CPU frequency, a NET DECODE REGRESSION. Gated on persistent-gpu-expert-cache. Acceptance: a measured A/B on one MoE model showing hybrid decode beating pure offload on a host where `ferrox bench bw` recommends hybrid, and matching it where it does not"
    status: pending
  - id: bench-bw
    content: "`ferrox_edge::qstar::BandwidthProfile` reads a profile no tool writes, so every deployment gets the unbenchmarked default of one fetch per layer per step. The measurement is three numbers per (format, GPU): CPU-MoE bandwidth, PCIe expert-gather bandwidth, and CRUCIALLY the contended pair -- both measured while the other runs, which is the form the fraction actually prefers (`pcie_ov / (pcie_ov + cpu_ov)`), because standalone numbers assume each side owns the machine and neither does. Profile is keyed to the GPU uuid and REFUSED if the recorded name disagrees, which `BandwidthProfile::matches_gpu` already enforces. Acceptance: `ferrox bench bw` writes a profile that `policy_for` turns into a non-default `QStarPolicy`, and a profile from another card is refused with a warning rather than applied"
    status: pending
  - id: double-buffered-prefill
    content: "Not ported and not portable as policy alone: it is a copy stream, two events per buffer, and a fence. A prefill touches every expert, so streaming misses one at a time is pointless -- the layer is copied whole into a borrowed two-layer slot region while the previous layer computes, with position == expert id so routing ids index the buffer directly. The invalidation rule is the subtle part: the borrowed slots' previous owners must lose their residency, or the next decode hits on a slot holding someone else's bytes. `ferrox_edge::ExpertCache::materialize_layer` already implements exactly that bookkeeping and is tested for it; what is missing is the stream. Gated on persistent-gpu-expert-cache. Acceptance: prefill on a MoE model overlaps its expert copies with compute, shown as a wall-clock A/B against the serialized path"
    status: pending
  - id: expert-bank-quant-formats
    content: "FreeToken's expert banks are MXFP4, NVFP4, FP8-block and ds_fp4 with per-format bank schemas (the bank ORDER is an ABI -- every copy path and kernel dispatch iterates it). Ferrox has MXFP4 in ferrox-quant as a GGUF-style fused dequant+dot (crates/ferrox-quant/src/lib.rs, AVX2 path at :1812) and none of the others. Decide honestly whether this is parity or scope creep: ferrox is a GGUF engine and these are safetensors-era formats. If the answer is that ferrox stays GGUF-native, say so here and close this, because it also closes ftw-format. Acceptance: a written decision, not an implementation"
    status: pending
  - id: ftw-format
    content: "FreeToken's fast-load weight format: one logical byte region sliced into shards, every tensor 4096-aligned and padded so any tensor or shard-local slice is O_DIRECT-legal, with an index JSON and per-layer expert-bank entries. Its real win is not the format but the loading discipline around it -- pin-after-fill (registering a lazy mmap first faults and zero-fills every page, ~47 s on a 137 GiB checkpoint, and that zero-fill is immediately overwritten by the read), plus mlock with a sticky downgrade to pageable on failure. Ferrox mmaps GGUF and keeps weights quantized, which is a different and arguably better answer to the same problem. Gated on expert-bank-quant-formats: if ferrox stays GGUF-native this is won't-do. Acceptance: a decision; if yes, a load-time A/B on a checkpoint large enough for the pin cost to be visible"
    status: pending
  - id: dsv4-cost-model
    content: "366 lines of pure integer arithmetic with zero torch in it, skipped by the port purely on scope: per-page cache cost for a compressed/window/indexer tiered pool, the ceil-to-bytes-per-token unit costs the cache-status sliders read, and a binary search for the largest page count that fits a budget. Only worth doing when a real DeepSeek-V4 checkpoint runs -- CLAUDE.md:89 says the `deepseek_v4_pro` preset is a sketch. Two traps if it is done: the sizing uses Python's BANKER'S rounding while the unit costs use ceil, and the reserved-window-pages docstring contradicts its own code (port the code). Acceptance: gated on a real checkpoint; until then this stays open as a marker, not a task"
    status: pending
  - id: remaining-parser-families
    content: "Two decision-side grammars the port left, already recorded in docs/THIRD_PARTY_NOTICES.md: MiniMax-M3's namespaced recursively-nested tool-argument grammar (its reasoning markers and adaptive leading-closer ARE ported; only `ToolCallFormat` lacks an M3 arm), and muse-glimmer's ATEM channel format on both sides. Both slot into the existing `ReasoningFormat` / `ToolCallFormat` tables. Do them when a checkpoint from either family is in docs/MODELS.md, not before. Acceptance: the cross-family streaming matrix test covers the new arm at the same standard as the nine that are there"
    status: pending
  - id: real-moe-checkpoints
    content: "FreeToken serves 20+ MoE checkpoints for real. CLAUDE.md:89: 'Presets glm_5_2 / deepseek_v4_pro / kimi_k3 are sketches — not proof of real-checkpoint support.' No amount of policy parity substitutes for this, and several items above (dsv4-cost-model, window-slide-during-decode, persistent-gpu-expert-cache) cannot be validated without one. Acceptance: at least one frontier MoE checkpoint runs end to end with a verified greedy-id parity receipt, which is a prerequisite for calling ANY of Phase C done"
    status: pending
isProject: false
---

# FreeToken feature parity — what the port left, and in what order

> Written **2026-08-25** on branch
> `claude/freetoken-rust-conversion-23u6ng`, from the FreeToken source
> at `FlashML-org/FreeToken@main` and a read of ferrox's own tree. It is
> the follow-on to the port that landed on that branch: four commits
> adding `crates/ferrox-edge` (193 tests) and wiring its two output
> parsers into `ferrox-server`.
>
> Every ferrox claim below carries a `file:line`. Every FreeToken claim
> comes from its Python source, not its README.
>
> **The paper was not readable from the environment this was written
> in** — `arxiv.org` and every mirror tried are blocked by the egress
> proxy. The abstract and the `q*` framing come from search results; the
> algorithms come from the source, which is the ground truth for a port
> anyway. Nothing here is sourced from the PDF.

## What is already done

`crates/ferrox-edge` is the host-side **decision** logic: the arithmetic
that decides what to compute where, and the state machines that turn a
token stream back into an agent-shaped response. It is deliberately
tensor-free, so all 193 of its tests run on any host with no GPU and no
model.

| Module | State |
|---|---|
| `parser::{reasoning,tool_call}` | **wired** — `reasoning_content` and nine tool-call formats on `/v1/chat/completions`, streaming and buffered |
| `detokenize::stop_prefix_holdback` | **wired** — `ferrox-server::stop` delegates to it, so the workspace has one implementation of the withhold rule |
| `radix::{plain,swa,hybrid}` | tested, no caller |
| `qstar`, `expert_cache` | tested, no caller |
| `pool`, `placement` | tested, no caller |
| `scheduler` | tested, duplicated by `ferrox-server::batch_scheduler` |
| `effort`, `cache_report` | tested, no caller |

## The blocker that ranks everything

FreeToken's central claim is that a decode step's expert-cache misses
can go two ways at once — fetched over PCIe *while* the CPU computes
others — so the step costs `max(fetch, cpu)` and the split that
minimises that maximum is a property of the machine.

Ferrox cannot make that true yet, for two independent reasons:

1. **There is no persistent GPU expert residency.**
   `crates/ferrox-moe/src/lib.rs:878`, in its own doc comment: *"Every
   call re-uploads each weight matrix to the device from scratch […] not
   yet the persistent-GPU-residency throughput win real expert offload
   needs."* An expert cache with nowhere to cache an expert is a
   bookkeeping exercise.
2. **The CPU MoE path is synchronous.** It runs inside the forward, so
   the two sides serialise and the step costs `fetch + cpu`. Under that
   cost model the optimal split is always "fetch everything", which
   means `qstar` is not merely unused — it *cannot matter*.

Both are in Phase C. Neither is a policy problem. This is why Phase A
comes first: **Phase A pays off on CPU and Metal today**, on the
hardware ferrox actually has receipts for, and needs no CUDA at all.

## Phase A — prefix reuse and agentic caching (no GPU)

The cheap half of FreeToken, and the half ferrox is furthest behind on
in a way users would feel.

What exists today is not a radix cache and says so:
`crates/ferrox-models/src/prefix_cache.rs:16` — *"not a trie/radix-tree
structure"*. `find_longest_prefix` is an O(n) linear scan (`:76`),
`store` evicts index 0 (`:131`) despite the module calling it LRU, and
each entry clones a whole `Vec<KvCache>`. A thousand requests off one
system prompt hold a thousand copies of its KV.

Order: `anchor-checkpoints` → `cache-manager` → `swa-paged-allocator` →
`wire-radix-prefix-cache` → `radix-on-the-batcher` →
`window-slide-during-decode`.

`anchor-checkpoints` is first because it is the largest thing the port
left, it is pure logic, and `ferrox-edge` already *pays* for it: the
window pool floor has an `anchor_checkpoints` term
(`crates/ferrox-edge/src/pool.rs:368`) sizing a pool for a feature that
does not exist.

## Phase B — the agent-facing API (no GPU)

Smaller, independent, and each item is worth doing alone. The one with
the widest blast radius is `chat-template-kwargs-plumbing`: the template
layer already accepts the passthrough
(`crates/ferrox-models/src/chat_template.rs:142`, tested at `:1008`) and
the server never sends it, which is why
`crates/ferrox-server/src/output.rs:91` has to note that the reasoning
parser can only ever see a block the model opened itself.

`radix-on-the-batcher` sits in Phase A but is felt here: continuous
batching is refused whenever a prefix cache is configured
(`crates/ferrox-server/src/lib.rs:1376`), so a deployment picks
concurrency *or* prefix reuse. That exclusivity is the current design's
honest answer to a real hazard, and page-granular sharing is what
removes it.

## Phase C — the compute half (CUDA, gated)

Do not start before `parity-scope-decision`. FreeToken is CUDA-only;
ferrox's CUDA bar is "must compile" with no benchmark host. Every item
here needs hardware nobody has run this engine on.

Order within the phase is forced: `persistent-gpu-expert-cache` →
`concurrent-cpu-moe-executor` → `bench-bw` → `double-buffered-prefill`.
The first two are what make `qstar` a real decision; the third is what
gives it real numbers; the fourth is a separate throughput win on the
prefill path.

CUDA graph capture is **not** a gap — `CudaDecodeGraph` exists at
`crates/ferrox-cuda/src/graph.rs:31`.

## Explicitly not ported, and not planned

- **FreeToken's Triton and CUDA kernels, its C++ CPU-MoE extension, its
  torch model stacks.** Ferrox has its own. A line-by-line port is a
  rewrite of ferrox, not a port into it.
- **Tensor parallel / multi-GPU.** FreeToken carries TP plumbing
  throughout its scheduler I/O; ferrox has none anywhere in `crates/`.
  It is on ferrox's own roadmap independently and does not belong to
  this plan.
- **The daemon, supervisor, proxy, `ft launch`, `ft shell` TUI.**
  Product surface around the engine, not the engine. `ft launch`'s one
  genuinely good idea — clearing cloud API keys from the child
  environment so an agent cannot silently fall back to a paid endpoint —
  is worth stealing on its own merits, and is not parity work.
- **`/v1/responses`.** Neither engine's users have asked for it here.

## Definition of done

There is no single "parity" line to cross, and pretending otherwise
would be the wrong shape for this plan. Instead:

- **Phase A done** = a second request against a shared system prompt
  reuses it without recomputation, on CPU and Metal, with a conservation
  test proving no page is leaked or double-freed, and an agentic turn
  that edits its context at a tool-call boundary keeps the prefix before
  the edit.
- **Phase B done** = a coding agent pointed at `ferrox-server` gets
  streamed tool-call arguments, `reasoning_content`, advertised thinking
  gears, and can resize the pools without a restart.
- **Phase C done** = on a named CUDA host, a decode step that hits the
  expert cache issues zero weight uploads, and hybrid beats pure offload
  on a machine where `ferrox bench bw` says it should. Not before
  `real-moe-checkpoints`.
