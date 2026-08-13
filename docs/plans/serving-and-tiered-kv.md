---
name: serving — tiered KV, prefill/decode fairness, admission
overview: "GOAL: make ferrox-server behave well under real concurrent load and across restarts — a disk-backed prefix cache that survives a process restart, chunked prefill that does not stall in-flight decodes, and an admission gate that answers 'will this fit' before accepting rather than OOMing later. Sourced from a read-only study of oMLX (.scratch/omlx) whose paged/SSD KV cache and time-debt scheduler are its two genuinely mature subsystems. KEY CORRECTION: oMLX's 'paged KV cache' is NOT paged attention — CacheBlock holds no tensor data — so all of it sits on top of a contiguous per-sequence KV, which is exactly what ferrox already has."
todos:
  - id: kv-block-hashing
    content: "Parent-chained SHA-256 block hashing: sha256(model || parent_hash || token_ids || extra_keys), root-seeded, with extra_keys as the salt slot for LoRA/multimodal identity. ~40 lines, the foundation for everything else"
    status: pending
  - id: kv-cache-signature
    content: "cache_signature stamped FROM THE BLOCK'S OWN PAYLOAD, never from the manager's expectation; reject unmarked blocks rather than trusting them. This is the line between a persistent cache and silent corruption after a config change"
    status: pending
  - id: kv-ssd-tier
    content: "Disk tier for the prefix cache: one file per block, sharded by hash prefix, per-layer flattened, dtype passthrough, format-versioned, temp-file+rename publish with a post-rename eviction re-check"
    status: pending
  - id: kv-ssd-async-read
    content: "Make the disk READ async/prefetched from the start. oMLX's is synchronous on the request path with no prefetch because a Metal deadlock blocked the fix — ferrox has no such constraint and should not inherit the retrofit"
    status: pending
  - id: kv-write-ordering
    content: "Write-path invariant: buffer -> index -> queue, so a concurrent reader never sees an index hit for a block with no file and no buffer. Bounded queue that falls back to an INLINE write rather than dropping"
    status: pending
  - id: kv-disk-budget
    content: "Effective size clamped against real free disk with a TTL'd stat, invalidated on ENOSPC, so eviction fires before the filesystem does"
    status: pending
  - id: kv-swa-block-alignment
    content: "Block size must be a multiple of the sliding-window size for SWA models. Non-obvious, and it will bite silently whenever ferrox runs a Gemma/gpt-oss-shaped model through the block cache"
    status: pending
  - id: sched-chunked-prefill
    content: "Resumable chunked prefill as a state machine (cache, tokens_remaining, tokens_processed) with `fn step_chunk(&mut self) -> bool done`. PREREQUISITE for everything else: it converts an unbounded prefill into a bounded unit of work. batch_scheduler.rs prefill is sequential forward_token today"
    status: pending
  - id: sched-time-debt
    content: "Time-debt prefill/decode interleaving: GPUs cannot preempt a running kernel, so chunk DURATION is the scheduling quantum. Cap contended chunks in ms converted to tokens via measured prefill tok/s; each chunk accrues duration*share debt; decode wall-time repays it; the gate blocks the next chunk until debt clears"
    status: pending
  - id: sched-keyed-row-state
    content: "Per-row state keyed by uid (HashMap<Uid, RowState>), never a Vec parallel to the batch. oMLX had to monkey-patch its way out of exactly this: a plain request joining a json_schema batch collapsed all processor slots and silently applied the wrong constraints to the wrong row"
    status: pending
  - id: sched-deferred-abort
    content: "Cancellation enqueues an id; the inference thread drains it at a step boundary and does the batch mutation, syncing the GPU before touching in-flight buffers. ferrox-metal has the same command-buffer lifetime hazard"
    status: pending
  - id: sched-block-admission
    content: "Admission on an INTEGER block budget (blocks_needed <= blocks_free), not byte watermarks. ferrox knows its KV layout exactly; oMLX's byte-watermark model is a workaround for MLX's opaque allocator and should not be ported"
    status: pending
  - id: sched-output-mailbox
    content: "Single-slot coalescing output mailbox per request (watch/Notify + explicit merge) rather than an unbounded channel, so a slow or disconnected client cannot grow memory. Plus an orphan reap for abandoned streams"
    status: pending
  - id: sched-stop-buffering
    content: "Two-layer stop sequences: a token-level matcher PLUS output-suffix buffering that withholds any tail which is a prefix of a stop string, so a partial match never reaches the wire. ferrox likely has only the text-level scan"
    status: pending
  - id: sched-queue-cap
    content: "Queue depth cap -> 503 + Retry-After, so client retry storms cannot grow memory without bound"
    status: pending
  - id: mem-preload-kv-budget
    content: "Compute the KV budget BEFORE load, not after: weights + n_ctx * per_token_kv + headroom <= device budget. oMLX admits models on weights-only cost and spends ~4300 lines compensating downstream"
    status: pending
  - id: mem-ctx-auto
    content: "`--ctx auto`: closed-form (budget - weights) // per_token, or bisect the real admission predicate and verify with an actual prefill. The honest meaning of 'set context limits'"
    status: pending
  - id: mem-typed-rejection
    content: "Typed 400 naming estimated_bytes / limit_bytes and WHICH ceiling is binding, instead of an OOM. Split the rejection counters (too-big vs under-pressure) so an operator can tell the two apart"
    status: pending
isProject: false
---

# serving — tiered KV, prefill/decode fairness, admission

> Written **2026-08-13** from a read-only study of **oMLX** under
> `.scratch/omlx` (Apache-2.0, Python + MLX). Its paged/SSD prefix cache
> (158 + 122 tests) and its time-debt scheduler are the two subsystems
> worth learning from; almost everything else in it is compensating for
> MLX's allocator.
>
> Scope note: this is *serving* behaviour. It does not close a single
> `benchmarks/RESULTS.md` row — those belong to
> [`llama-cpp-parity-push.md`](llama-cpp-parity-push.md). Keep the two
> tracks separate so a serving change is never credited with an engine
> speedup.

## The correction that makes this cheap

**oMLX's "paged KV cache" is not paged attention.** `CacheBlock` holds no
tensor data at all — it is `block_id`, `ref_count`, `block_hash`,
free-list pointers, `token_count`. Attention still runs on a stock
*contiguous* KV cache, rebuilt by concatenating per-block slices on
restore.

So blocks are a **storage, dedup and eviction granularity for a prefix
cache**, not a memory layout for the attention kernel. That is exactly why
it ports: it sits on top of a contiguous per-sequence KV, which is what
ferrox has today. **No paged-attention kernel work is required for any of
this.**

## What ferrox already has

- `batch_scheduler.rs` — continuous batching of *decode*, opt-in via
  `FERROX_CONTINUOUS_BATCHING=1`, with prefill priority and a
  `FERROX_CB_MAX_SEQS` cap. **Prefill is still sequential
  `forward_token` per sequence**, which is the gap `sched-chunked-prefill`
  closes.
- An in-memory `PrefixCache`, a response cache, `session.rs`, a KV pool.
- Exact knowledge of its own KV layout from the GGUF header — the thing
  oMLX lacks and works around everywhere.

Worth stating plainly: on the batching axis ferrox is **not behind**
oMLX. oMLX's continuous batching is decode-only too (`prefill_batch_size`
is hardcoded to 1), it has **no preemption** (`RequestStatus.PREEMPTED` is
defined and never assigned), **no priority policy** (the enum exists and
is never read), and no per-step token budget (`max_num_batched_tokens` is
unreferenced dead config). What it has that ferrox does not is chunked
prefill with a fairness mechanism, and a durable cache tier.

## Phase 1 — chunked prefill and fairness

### 1a. Resumable prefill (prerequisite)

A per-request state machine holding `cache`, `tokens_remaining`,
`tokens_processed`, plus boundary bookkeeping, with
`fn step_chunk(&mut self) -> Result<bool /* done */>`. This converts an
unbounded prefill into a bounded unit of work, which is what makes a
single-threaded step loop possible at all.

### 1b. Time-debt interleaving

The load-bearing insight, and it is backend-agnostic: **a GPU cannot
preempt a running kernel, so chunk duration *is* the scheduling
quantum.** Not token count — duration.

1. Contended chunk size is derived in **milliseconds**, converted to
   tokens via measured prefill tok/s, floored and capped.
2. Each chunk accrues `chunk_seconds × share` of debt.
3. Decode wall-time repays it.
4. The gate blocks the next chunk until the debt clears.

oMLX's constants (0.5 fair share, 500 ms stall target) were measured on an
M3 Ultra and are a reasonable starting point, not a derivation. One flaw
not to inherit: it advances *all* pending chunked prefills before decode
runs, so N concurrent long prompts stall decode for N chunks per step.
Cap the per-step prefill work.

### 1c. Keyed row state

Per-row state must be a `HashMap<Uid, RowState>`, never a `Vec` parallel
to the batch. This is a correctness bug class, not a style preference:
oMLX ships a monkey-patched batch step that rebuilds positional arrays
from a registry on *every* step, because a plain chat request joining a
batch that served a `json_schema` request collapsed every processor slot
to `None` and silently applied the wrong constraints to the wrong row.
ferrox owns its whole stack and can simply build it right.

Related: per-request RNG state lives in the row struct. oMLX seeds a
*global* RNG per request and its own comments admit this breaks under
concurrency.

### 1d. Deferred abort

Cancellation enqueues an id into a shared set; the inference thread drains
it at a step boundary and performs the batch mutation, **syncing the GPU
first** because slicing KV against in-flight command buffers corrupts
them. ferrox-metal has the identical hazard. In Rust this is an mpsc
drain at the top of the step — more natural than the Python original.

### 1e. Output mailbox and stop sequences

- **Single-slot coalescing mailbox** per request with an explicit merge,
  not an unbounded channel: a slow or disconnected SSE consumer must not
  grow memory. Pair it with an orphan reap for streams abandoned rather
  than closed.
- **Stop sequences need two layers**, and ferrox likely has only the
  second: a token-level matcher, *plus* output buffering that withholds
  any output suffix which is a prefix of a stop string so a partial match
  never reaches the wire. Without the buffering layer, streaming stop
  sequences leak fragments.
- **Queue cap → 503 + Retry-After.** Trivial; prevents retry storms from
  growing memory without bound.

## Phase 2 — admission that answers before it accepts

oMLX's structural weakness, stated plainly so ferrox does not copy it: it
admits a model on **weights-only** cost (`sum(*.safetensors) × 1.05`, no
KV, no activations) and then discovers the KV cost per request forever
after. Its prefill-eviction callback, background pressure enforcer,
adaptive chunk throttle, abort ladder and stall-timeout killer are all
compensating for that one under-charge.

ferrox does not need any of it:

```
weights + n_ctx × per_token_kv + activation_headroom  ≤  device_budget
per_token_kv = n_layers × n_kv_heads × head_dim × bytes × 2
```

All four terms are exact from the GGUF header at `inspect-plan` time.
Variants worth carrying: the sliding-window cap
(`min(tokens, window + chunk − 1)`) and the MLA form
(`kv_lora_rank + rope_dim` per layer).

From that follow, cheaply:

- **`--ctx auto`** — closed-form `(budget − weights) // per_token`, or
  bisect the real admission predicate and verify with an actual prefill.
  The bisect is the more honest of the two and oMLX implements it, but
  only as an opt-in admin job that unloads every model first.
- **Admission on an integer block count** once the block cache exists:
  `blocks_needed ≤ blocks_free`. Strictly better than a byte watermark,
  and available to ferrox precisely because its allocator is not opaque.
- **Typed rejection**: a 400 naming `estimated_bytes`, `limit_bytes` and
  *which* ceiling binds — with split counters for "prompt too big" versus
  "system under pressure", so an operator is not sent to the wrong knob.

## Phase 3 — the disk tier

Design, with the pieces that matter:

- **Block hashing**: parent-chained SHA-256 over
  `(model, parent_hash, token_ids, extra_keys)`, root-seeded.
  `extra_keys` is the salt slot for LoRA/multimodal identity. Sampling
  params are correctly *not* part of the key — KV is sampling-independent.
- **`cache_signature`**, and this is the one to get right: stamp
  compatibility **from the block's own payload, never from the manager's
  expectation**, and reject blocks with no recorded depth rather than
  trusting them. oMLX states the rule as "a signature must never vouch for
  a width the payload does not have". It is the difference between a
  persistent cache and silent corruption after a config change.
- **On disk**: one file per block, sharded into subdirectories by hash
  prefix, per-layer flattened keys, dtype passed through unchanged, no
  compression, an explicit format version with a readable-set, and
  metadata carrying block hash, token count, layer count, block size and
  the signature.
- **Publish atomically** with temp-file + rename, then re-check the block
  was not evicted mid-write.
- **Write ordering invariant**: buffer → index → queue. A concurrent
  reader must never see an index hit for a block with no file and no
  buffer.
- **Backpressure**: bounded queue; on full, write **inline on the calling
  thread** rather than drop. Count the fallbacks.
- **Disk budget** clamped against real free space with a TTL'd stat,
  invalidated on ENOSPC, so eviction fires before the filesystem does.
- **Block size must be a multiple of the sliding-window size** for SWA
  models. Non-obvious and silent when wrong.

Two things oMLX got wrong that ferrox should not inherit:

1. **The read is synchronous on the request path with no prefetch** —
   they were blocked by a Metal deadlock that ferrox does not have. Build
   it async and prefetched from the start; retrofitting it there was
   impossible.
2. **Its RAM tier is off by default**, so the advertised "hot tier fills,
   then spills to SSD" never happens out of the box. If ferrox ships a RAM
   tier, ship it on, or do not claim it.

Also: measure the block read and block write. oMLX's cache stats contain
**zero time-valued fields** — hit rate is observable, "SSD hit versus
recompute" latency is not, which makes the tier impossible to tune.

## Explicitly not ported

- **TurboQuant KV quantization.** Metal kernels generated as Python
  f-strings and JIT-compiled per `(key_bits, val_bits, dim)`; NumPy QR and
  Lloyd–Max at codec-build time. Only the *format* idea is portable: store
  an fp16 norm plus rotated packed indices and rebuild the codec
  deterministically from `(dim, bits, seed)` rather than serializing it.
  Note its asymmetric K/V precision is narrower than advertised — K and V
  differ only at fractional bit depths; at integer bits they are identical.
- **Byte-watermark memory accounting**, `phys_footprint` sampling, wired
  limits and jetsam avoidance, the macOS `free + inactive + active × ratio`
  dynamic ceiling, allocator-cache hygiene, lazy-array eval ceremony. All
  MLX/Apple-UMA artifacts.
- **The unload settle barrier**, at least unchanged: it polls the
  allocator until freed bytes come back. Against ferrox's **mmap'd**
  quantized weights that would report a false timeout — freeing address
  space is not freeing RSS. This is the single biggest conceptual mismatch
  between the two engines.
- Monkey-patching as an integration mechanism. ferrox needs trait dispatch.

## Sequencing

1. **1a + 1b** (chunked prefill + time-debt). Largest behavioural win,
   and the prerequisite for the rest.
2. **1c–1e** (keyed rows, deferred abort, mailbox, stop buffering, queue
   cap). Correctness under concurrency; cheap.
3. **Phase 2** (pre-load KV budget, `--ctx auto`, typed rejection).
   Independent of 1 and 3, and the highest user-visible value per line.
4. **Phase 3** (disk tier). Largest, and worth doing only after the block
   hashing and signature discipline from Phase 2 exist.
