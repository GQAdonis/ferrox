---
name: serving — tiered KV, prefill/decode fairness, admission
overview: "GOAL: make ferrox-server behave well under real concurrent load and across restarts — a disk-backed prefix cache that survives a process restart, chunked prefill that does not stall in-flight decodes, and an admission gate that answers 'will this fit' before accepting rather than OOMing later. Sourced from a read-only study of oMLX (.scratch/omlx) whose paged/SSD KV cache and time-debt scheduler are its two genuinely mature subsystems. KEY CORRECTION: oMLX's 'paged KV cache' is NOT paged attention — CacheBlock holds no tensor data — so all of it sits on top of a contiguous per-sequence KV, which is exactly what ferrox already has."
todos:
  - id: kv-block-hashing
    content: "LANDED as `ferrox-core::kv_block` (`BlockHasher`, `BlockHash`). Parent-chained SHA-256, root-seeded at H(domain,\"root\",model,extra_keys); each block is H(domain,\"block\",model,extra_keys,parent,token_ids). Every field is length-prefixed, so `[\"ab\",\"c\"]` and `[\"a\",\"bc\"]` cannot collide; the domain tag is versioned so an encoding change orphans old blocks instead of misreading them. `chain()` hashes WHOLE blocks only -- a growing tail has no stable identity. Sampling params are excluded on purpose (KV is sampling-independent). Golden digests cross-validated against an independent Python hashlib reference, per this repo's fixture convention. NOT DONE: nothing consumes it yet -- `PrefixCache` still does a linear longest-common-prefix scan over whole sequences, and no block STORE exists (that is `kv-ssd-tier`); `extra_keys` is a slot with no LoRA/multimodal producer wired to it."
    status: completed
  - id: kv-cache-signature
    content: "LANDED as `ferrox-core::kv_signature`. `CacheSignature::from_payload` MEASURES the tensors (layers, kv heads, head dim, dtype, token depth) and takes no expected-shape argument to fill a gap from; `seq_len` is treated as a claim and checked against `k.len()`. `UnverifiedBlock::verify` runs three ordered checks: unmarked -> refuse (absence is never agreement); recorded stamp vs measured payload -> `PayloadMismatch` (a stamp may not vouch for a width the payload lacks); only then measured vs reader expectation -> `Incompatible`, naming the field that changed. Format version has an explicit readable-set. Confirmed the central test FAILS when `verify` is patched to trust the stamp instead of the payload. NOT DONE: nothing stores or reads blocks yet, so no signature is written to disk (that is `kv-ssd-tier`); `KvDtype` has one variant (F32) because `KvCache` is `Vec<f32>` -- the enum exists so an f16/quantized KV tier invalidates old blocks instead of reinterpreting their bytes; the signature is not yet bound to a `BlockHash`, since nothing yet holds the two together."
    status: completed
  - id: kv-ssd-tier
    content: "LANDED and merged. crates/ferrox-core/src/kv_disk.rs (~3.5k lines): one file per block, sharded by hash prefix, per-layer flattened, dtype passthrough, format-versioned, temp-file-plus-rename publish with a post-rename eviction re-check. Format bumped to v2 by kv-swa-block-alignment and v1 DROPPED from the readable set, because a v1 file cannot say which sliding window it was cut under and absence is never agreement. That is one cold start for anyone already running the disk tier, not a reinterpretation of their data"
    status: completed
  - id: kv-ssd-async-read
    content: "LANDED and merged, async and prefetched from the start rather than retrofitted. The plan's reasoning held: oMLX's read is synchronous on the request path only because a Metal deadlock blocked the fix, and ferrox had no such constraint to inherit"
    status: completed
  - id: kv-write-ordering
    content: "LANDED and merged. Write path is buffer, then index, then queue, so a concurrent reader never sees an index hit for a block with no file and no buffer. The bounded queue falls back to an INLINE write rather than dropping. Crash safety is tested by writing a truncated file and asserting it is rejected on read rather than deserialized into garbage"
    status: completed
  - id: kv-disk-budget
    content: "LANDED and merged. Effective size clamped against real free disk via a TTL'd statvfs probe, invalidated on ENOSPC, so eviction fires before the filesystem does. libc is pulled in for statvfs alone; the probe is injectable, so nothing else depends on it"
    status: completed
  - id: kv-swa-block-alignment
    content: "LANDED as `ferrox-core::kv_swa` (`BlockLayout`, `aligned_block_size`) plus the layout being carried in `CacheSignature`. CORRECTION TO THE PLAN'S WORDING, stated in the module doc: the implementable relation is `window % block_size == 0` (the block size divides the window), not the operands the other way round -- forcing block size up to a multiple of a 128-token gpt-oss window would make every block a whole window. Same constraint vLLM asserts. `BlockLayout` is only constructible through `new`, so holding one is the proof the rule holds; `aligned_block_size` rounds a requested size DOWN to a divisor so a config layer never hands the cache a size the cache refuses. `ModelConfig::kv_block_window` / `kv_block_layout` bridge it to real models: an alternating-SWA model is constrained as soon as ANY layer slides (5/6 full-attention is not 5/6 exempt). The durable half: `CacheSignature` gained `layout`, the block file format went to v2 with block-size and window fields, and v1 was DROPPED from the readable set rather than read with the window assumed absent -- a v1 file cannot say what window it was cut under, and absence is never agreement, so a restart onto this build starts cold once. `block_size` is payload-checked (a stored block is exactly one whole block, so a stamp claiming 64 over a 48-token payload is `BlockSizeMismatch`); the window is not tensor-provable, so it is carried like `model` and settled against the reader's expectation. Tests confirmed to FAIL when the checks are removed: `a_block_written_under_a_different_window_is_refused_not_reused`, `a_full_causal_reader_will_not_take_a_sliding_window_block`, `a_block_cut_at_a_different_block_size_is_incompatible`, `a_stamp_may_not_claim_a_block_size_the_payload_lacks`, plus the disk-tier `a_block_written_under_one_window_is_not_served_to_a_reader_expecting_another` (write, drop the store, reopen, reindex, ask with a changed window -> miss + `incompatible` counter, and an unchanged window still hits). NOT DONE, deliberately: nothing yet CUTS a sequence into blocks -- `PrefixCache` is still a whole-sequence scan -- so no production call site passes a `BlockLayout` today; the guard is in place before the producer, which is the order that keeps a mis-aligned block from ever becoming durable. And the KV is still stored whole with masking rather than truncated to the window, so the alignment rule is currently prophylactic for ferrox's own kernels and load-bearing only for the durable format."
    status: completed
  - id: sched-chunked-prefill
    content: "LANDED. `PrefillState` in batch_scheduler.rs is the resumable state machine (caches, tokens_processed, tokens_remaining) with `step_chunk(&mut self) -> bool /* done */`; the worker runs ONE chunk (round-robin over waiting prompts) plus ONE batched decode step per tick, so a long prompt costs an in-flight decode one chunk, not its whole length, and N long prompts still cost one chunk per tick (the oMLX flaw the plan says not to inherit). Chunk size from `FERROX_CB_PREFILL_CHUNK` (default 128) or `BatcherConfig` for tests; `FERROX_CB_MAX_SEQS` now counts prefilling prompts too, since each holds a full KV set. Counters (prefill_chunks / prefill_tokens / decode_steps) on `/metrics`. NOT DONE, deliberately: (a) a chunk is still a per-token `forward_token` loop, not `forward_batch` -- chunking here is a scheduling boundary and is asserted bit-identical to the sequential prefill it replaced, so no throughput claim is made and no benchmark row moves; (b) no time-debt gate, that is `sched-time-debt`; (c) the batcher still cannot use the prefix cache or KV pool, so a chunk boundary is not yet a cache-block boundary."
    status: completed
  - id: sched-time-debt
    content: "Time-debt prefill/decode interleaving: GPUs cannot preempt a running kernel, so chunk DURATION is the scheduling quantum. Cap contended chunks in ms converted to tokens via measured prefill tok/s; each chunk accrues duration*share debt; decode wall-time repays it; the gate blocks the next chunk until debt clears"
    status: pending
  - id: sched-keyed-row-state
    content: "LANDED. `batch_scheduler.rs` keeps in-flight rows in `Rows { state: HashMap<Uid, Slot>, order: Vec<Uid> }` -- keyed state plus an explicit admission order (HashMap iteration order is not deterministic, batch composition must be). `swap_remove` on a `Vec<Slot>` is gone: a stale `Uid` resolves to `None`, never to whichever row moved into the vacated position. The per-step batch is still a slice for the kernel, but `active[j]` holds a `Uid`, so the scatter after `forward_multi_seq` cannot land on the wrong request. Per-request sampler state already lived in the row (not a global RNG) and is documented as deliberate. Tests: `removing_a_row_never_reassigns_another_rows_state` (with the positional `swap_remove` failure spelled out beside it), `flush_replies_on_each_rows_own_channel`, and `a_row_leaving_mid_batch_does_not_shift_its_neighbours_output` -- three concurrent rows where one trips a stop mid-batch, so the batch is narrower than the row table on that tick; confirmed to FAIL when the scatter is patched to address rows by batch position. NOT DONE: no per-row logits processor / constrained-decoding state exists yet (json_object is a whole-request flag), so the specific oMLX json_schema collapse has no ferrox analogue to regress -- the table is simply built so it cannot happen."
    status: completed
  - id: sched-deferred-abort
    content: "LANDED, and it closes a gap the `cancel` module previously documented as unfixable ('a continuous-batching request ... is not covered'). `CancelToken::on_cancel` (new) hands the scheduler a callback that does exactly one thing -- push an `AbortId` into `AbortInbox` -- and `apply_aborts` drains that set at the TOP of a worker tick, before any forward pass, and does the batch mutation there. The hook fires immediately if the token is already cancelled, so a cancel racing its own submission is not lost; ids that name a job not yet through the channel are CARRIED across ticks rather than dropped, for the same reason. A request is stopped wherever it is: still queued (queue slot released, nothing prefilled), mid-prefill (blocks released, remaining chunks never run), or decoding (marked `FinishReason::Cancelled` and left to leave through `flush_finished`, so there is exactly ONE path that releases blocks and replies -- a second removal path is a second place to forget one of those). Partial output survives, since a cancelled generation has tokens and discarding them serves nobody. ON THE GPU SYNC: ferrox-metal already calls `waitUntilCompleted` per dispatch, so no command buffer outlives a `forward_multi_seq` call and the step boundary really is a point where nothing is in flight -- there is no separate sync to insert, and none was faked. What the deferral buys is that the mutation happens on the thread that owns the step at all, rather than from whichever HTTP handler received the cancel while `std::mem::take` has a row's caches lifted out into the batch. Tests confirmed to FAIL when broken: `a_decoding_request_stops_when_it_is_cancelled` (finishes with Length after all 4000 tokens without the drain), `a_cancel_that_arrives_before_its_job_is_not_lost` (fails when unmatched ids are dropped), `a_cancelled_row_leaves_through_the_one_exit_every_row_uses` (fails when `mark_cancelled` removes instead of marking -- catches the double-reply), `a_cancelled_prefill_is_abandoned_and_gives_its_blocks_back` (fails when the release is dropped), plus `cancelling_one_request_leaves_its_neighbours_running`. NOT DONE: no cancellation inside a prefill CHUNK -- the check is between chunks, so a request is charged at most `FERROX_CB_PREFILL_CHUNK` tokens after the cancel, which is a bounded cost and the reason chunking landed first; and `AbortInbox` carries an unmatched id indefinitely (bounded by the number of cancels for jobs whose send failed, which only happens when the worker is already dead)."
    status: completed
  - id: sched-block-admission
    content: "LANDED. `BlockBudget` in batch_scheduler.rs is an integer ledger of KV blocks (`FERROX_CB_KV_BLOCKS` total, `FERROX_CB_KV_BLOCK_SIZE` positions per block, default 256); a request needs `ceil((prompt + max_tokens) / block_size)` blocks, reserved at admission for its whole lifetime and released in `Rows::flush_finished`. No byte watermark, no hysteresis band, no pressure enforcer -- ferrox reads its KV layout from the GGUF header, so 'will this fit' has an exact integer answer. The worker now keeps its own `waiting: VecDeque<Job>` and `QueueGate` releases a slot at ADMISSION rather than when the job leaves the channel, so the queue cap still counts a job parked for capacity. Two rejections, deliberately different: `blocks_needed > blocks_total` is `DecodeError::KvBudgetExceeded` -> 400 with `retry_after_secs() == None`, refused before a queue slot is reserved (an empty server would refuse it too, so 503 would be a lie); momentary pressure just waits. Counters split accordingly -- `kv_rejected_too_large` vs `queue_rejected`, plus `kv_blocks_total/free/peak/size` on `/metrics`. Admission is strict FIFO: a head job that does not fit stops the line rather than being skipped, because skip-ahead starves large requests indefinitely behind a stream of small ones. Tests confirmed to FAIL against the broken version: `concurrent_requests_never_hold_more_blocks_than_the_budget` (6x2 blocks against a 4-block budget; peak is measured from the ROWS, not from the ledger, because a ledger-derived peak cannot exceed the budget however broken admission is and would report the invariant instead of checking it), `every_admitted_request_gives_its_blocks_back`, `a_job_rejected_at_validation_gives_its_blocks_back`, `a_head_job_that_does_not_fit_holds_the_line` (fails under both a no-op reserve and a skip-ahead policy). DELIBERATELY NOT DONE: the budget is in blocks of POSITIONS, not bytes -- it is not yet derived from `ferrox-models::kv_budget`'s per-token KV bytes and a device budget, so an operator sets `FERROX_CB_KV_BLOCKS` by hand rather than the server deriving it from the model (that join is `mem-preload-kv-budget`); no preemption or eviction, so an admitted request keeps its blocks until it finishes; and the reservation is worst-case (`prompt + max_tokens`) rather than growing with the sequence, which over-charges a request that stops early -- correct, but conservative."
    status: completed
  - id: sched-output-mailbox
    content: "NOT DONE, and the plan's premise is WRONG, which is why it stays open rather than being implemented as written. The SSE channel is NOT unbounded: it is a tokio::mpsc::channel(64) with cancel-on-send-failure, so a slow consumer already gets backpressure rather than unbounded memory growth. The real hazards, and what a future attempt should actually fix, are (a) blocking_send stalling the generation thread and (b) no orphan deadline for an abandoned stream. Note also that under continuous batching the emit closure is not used at all (overlap = !tools_active && batcher.is_none()), so a mailbox would not help the batched path. Restructuring the SSE handler on a false premise, with no HTTP test harness in place, risks a silent streaming break worth more than the fix"
    status: pending
  - id: sched-stop-buffering
    content: "LANDED as `ferrox-server::stop` (`StopMatcher`, `StopStep`, `resolve_stop_tokens`), used by BOTH `generate::sample_until_stop` and the batch scheduler, so a row in a batch and a row on its own cannot disagree about where an answer ends. CORRECTION TO THE PLAN'S PREMISE: ferrox did NOT have only a text-level scan -- it already withheld `longest_stop - 1` bytes on both paths, which is safe. What it lacked was (a) the token-level layer entirely and (b) precision in the text layer. Layer 1: a stop string that encodes to exactly one token is matched on the ID, before detokenization, and is treated exactly like EOS -- not emitted, not counted in usage. That answers a question the text layer provably cannot: `a_stop_token_that_renders_as_nothing_is_still_a_stop` shows a control token rendering to \"\" that the text scan can never see. Multi-token stop strings are deliberately NOT given a token-level form: a multi-token encoding is not a statement about how the model will emit that text. Layer 2: `partial_suffix_len` withholds the longest suffix that is a PROPER prefix of some stop, instead of a fixed byte count, so with `stop: [\"<|im_end|>\"]` ordinary text is no longer permanently 9 bytes behind the model; byte-wise comparison with a `floor_char_boundary` guard, so a split can never land mid-character. Stop-token ids are resolved in exactly one place (`run_generation_emit`, the only layer holding both the request's stop strings and the model's tokenizer) and ride on `GenerationParams`, so the batched and private paths cannot drift. Tests confirmed to FAIL when broken: `a_token_level_stop_ends_generation_and_never_reaches_the_output` and `a_token_level_stop_ends_a_batched_row` (both run to the token limit without the id check), `a_disproved_partial_is_released_by_the_token_that_disproves_it` plus six `stop::` unit tests (all fail under the fixed hold-back). NOT DONE: no incremental/streaming matcher state across the disk-cached prefix, no regex or token-sequence stops, and empty stop strings are dropped rather than rejected at the API edge -- an empty stop would otherwise match at position 0 and end every answer at its first token."
    status: completed
  - id: sched-queue-cap
    content: "LANDED. `QueueGate` in batch_scheduler.rs bounds jobs WAITING for admission (in-flight sequences remain `FERROX_CB_MAX_SEQS`'s business); default 512 via `FERROX_CB_MAX_QUEUE`. Over the cap, `generate` returns `DecodeError::QueueFull { queued, cap }` -> 503, with `retry_after_seconds` in the body and a `Retry-After` header stamped by `limits::retry_after`, a layer that marks any 503 lacking one (a 503 is by definition temporary, so this also fixes the pre-existing KV-pool and no-model-loaded 503s). Reservation is a CAS loop, not check-then-act; `queue_gate_never_exceeds_its_cap_under_concurrent_submitters` (32 threads x 64 rounds) was confirmed to FAIL with a load-then-fetch_add gate. `queue_depth` / `queue_rejected` on `/metrics`. DELIBERATELY NOT DONE: the Retry-After value is a fixed 1s, not queue depth divided by measured throughput -- the honest computation needs a drain-rate estimate this scheduler does not keep yet; the cap only guards the continuous-batching path, since the private `generate` path has no queue to cap; and the cap counts jobs, not tokens or bytes, so one huge prompt still counts as one."
    status: completed
  - id: mem-preload-kv-budget
    content: "MOSTLY LANDED ELSEWHERE, verified rather than assumed. ferrox-models::kv_budget plus ferrox-cli/src/run.rs already do the pre-load arithmetic: --ctx-size auto, a pre-load KvBudget::check, and a Ceiling::DeviceMemory refusal, so weights plus n_ctx * per_token_kv plus headroom is checked against the device budget BEFORE the load rather than compensated for afterwards. OPEN HALF: ferrox-server does not price its model at load, so the server still admits on configured ceilings instead of measured ones. Recorded here so the next run does not reimplement the part that exists"
    status: pending
  - id: mem-ctx-auto
    content: "`--ctx auto`: closed-form (budget - weights) // per_token, or bisect the real admission predicate and verify with an actual prefill. The honest meaning of 'set context limits'"
    status: pending
  - id: mem-typed-rejection
    content: "HALF LANDED (3412df9, merged). Typed 400 naming `binding` (a ferrox_models::Ceiling code), estimated_bytes and limit_bytes priced from the model's real KvShape, plus positions and positions_limit. The context ceiling is reported before the device ceiling. Rejection counters are split three ways on /metrics, and retry_after_secs() is None for this class, since a request too big to ever fit is not a request to retry. OPEN HALF: it covers the continuous-batching path only, so the private generate path still answers 503 with no context ceiling; and the ceilings are CONFIGURED (FERROX_CB_MAX_CONTEXT, FERROX_CB_KV_BLOCKS) rather than derived from weights plus per-token KV against a device budget, which is what mem-preload-kv-budget asks for"
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
  `FERROX_CONTINUOUS_BATCHING=1`, with a `FERROX_CB_MAX_SEQS` cap.
  Prefill **is** chunked now (`sched-chunked-prefill`, landed): one
  bounded `PrefillState::step_chunk` plus one batched decode step per
  tick. Each chunk is still a per-token `forward_token` loop, so this
  bought fairness, not throughput.
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
