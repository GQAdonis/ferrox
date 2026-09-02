# Metal parallel decode concurrency

Status: **design / in progress** (branch `fix/metal-parallel-concurrency`)

Related defect: GitHub issue *(linked after creation)*

## Problem

With Metal enabled and continuous batching **off** (the default), ferrox-server accepts multiple concurrent streaming requests but only reliably serves **one at a time**. Two or more parallel private-loop decodes against the same loaded GGUF model produce:

- Truncated SSE streams (one token, then silence; HTTP 200, no `[DONE]`)
- Journal panics: `rms_norm` length mismatch (`0` vs `3072`) in `ferrox-core/src/matmul.rs`
- Cascading `PoisonError` on `Decoder::metal_attn_kv` (`decoder.rs:1810`, `3749`)
- Subsequent requests fail with `internal error during generation` until restart

Reproduction: run two concurrent `POST /v1/chat/completions` with `stream: true` against a real GGUF on Metal (`ferrox serve -dev metal -ngl all`). A Python parallel bench lives in `pi-agent-tests/ferrox_parallel_bench.py`.

## Root cause

The server module docs in `ferrox-server/src/lib.rs` state that the loaded model is immutable and that each request builds its own host `KvCache`, so multiple requests can decode concurrently via `spawn_blocking`. That is **true for weights** and **true for host KV**, but **false for Metal-resident KV**.

`Decoder` carries one shared arena:

```rust
// ferrox-models/src/decoder.rs
pub(crate) metal_attn_kv: Mutex<Option<Vec<MetalKvBuffers>>>,
```

Every concurrent private-loop request locks this mutex and runs `forward_token` against the **same** Metal KV buffers. After request A releases the lock, request B may reset or reshape those buffers for its sequence. When A resumes:

1. `metal_kvs.iter().all(|m| m.seq_len == pos)` is false → fused Metal dense stack is **skipped**
2. GPU embedding gather leaves `hidden == Vec::new()` (see `forward_token` ~1768)
3. CPU fallback calls `rms_norm(&hidden, …)` with empty `hidden` → panic
4. `metal_attn_kv.lock().unwrap()` poisons the mutex; all later Metal decodes fail

Continuous batching avoids this by design: one `ferrox-continuous-batch` worker calls `forward_multi_seq` per tick — no concurrent `forward_token` on shared Metal KV. Trade-off: streaming is buffered, not token-overlapped.

The existing concurrency integration test (`concurrent_requests_against_the_same_model_do_not_interfere`) uses `Decoder::new_random_small` (synthetic, no Metal fused path) and does not cover this failure mode.

## Architecture options

Ranked by effort vs correctness for multi-client Metal serving.

### A. Short-term mitigation — single-flight gate (recommended first land)

When `metal_attn_enabled()` and continuous batching is off, serialize decode through a process-wide semaphore (count = 1) around `run_generation_emit` / `forward_token` for GGUF models.

- **Pros:** Small diff, stops panics and truncated streams immediately, honest behavior
- **Cons:** No parallel Metal throughput on private path; document in `/health` capabilities
- **Files:** `ferrox-server/src/lib.rs`, possibly `loaded.rs`

### B. Proper fix — per-request Metal KV residency

Move `metal_attn_kv` off `Decoder` into the generation context (alongside per-request host `KvCache`). Each request owns its Metal buffers for the lifetime of the decode loop.

- **Pros:** Matches the documented concurrency model; true parallel private-loop decode on Metal
- **Cons:** Higher GPU memory use (N × KV); allocation/teardown per request; largest change
- **Files:** `ferrox-models/src/decoder.rs`, `ferrox-metal/src/attn/`, `generate` module

### C. Middle ground — sync host KV before lock release

Before releasing `metal_attn_kv`, always `sync_metal_attn_kv_to_host` for the active request and rebuild Metal state on next lock acquisition from that request's host caches only.

- **Pros:** Keeps one Metal arena; may allow limited parallelism if sync is cheap
- **Cons:** Easy to get wrong; still serializes on the mutex in practice; fragile under overlap

### D. Require continuous batching for parallel Metal serving

Treat `FERROX_CONTINUOUS_BATCHING=1` as the supported multi-request path on Metal; reject or queue additional private-loop decodes with 503 + `Retry-After`.

- **Pros:** Uses existing safe scheduler
- **Cons:** Buffered streaming; CB disabled when prefix cache or non-paged KV pool is configured

## Additional hardening (any path)

1. **Defensive `hidden` fill:** If `hidden.is_empty()` before CPU `rms_norm`, call `embed_token` / `dequant_row` (same as Metal `Err` path ~2193).
2. **Poison-tolerant locks:** Replace `lock().unwrap()` with `lock().unwrap_or_else(|p| p.into_inner())` on `metal_attn_kv` (partial recovery; does not fix logic bug).
3. **Streaming error surfacing:** Await `spawn_blocking` in the streaming chat path and map panics to SSE error + `[DONE]` (today fire-and-forget → truncated 200).
4. **Integration test:** `#[cfg(feature = "metal")]` test — two parallel `run_generation_emit` on same real-ish GGUF decoder; assert no panic, full token counts.

## Success criteria

- [ ] Two concurrent streaming requests on Metal (CB off) complete with full output and `[DONE]`/`usage`
- [ ] No panics in `ferrox-journal.log` under parallel load
- [ ] `/health` accurately reports parallel decode capability
- [ ] Regression test fails on `main`, passes on fix branch

## References

| Location | Role |
|----------|------|
| `ferrox-server/src/lib.rs:11-27` | Concurrency documentation (host KV only) |
| `ferrox-server/src/lib.rs:2397-2428` | Private loop vs continuous batcher routing |
| `ferrox-server/src/lib.rs:3038` | Streaming `spawn_blocking` (not joined) |
| `ferrox-models/src/decoder.rs:422-426` | Shared `metal_attn_kv` |
| `ferrox-models/src/decoder.rs:1807-1843` | Lock, reset, stale Metal detection |
| `ferrox-models/src/decoder.rs:2057-2248` | Dense stack skip + empty `hidden` + `rms_norm` |
| `ferrox-core/src/matmul.rs:63` | Panic site |
| `ferrox-journal.log` | Observed panic chain from parallel bench |
