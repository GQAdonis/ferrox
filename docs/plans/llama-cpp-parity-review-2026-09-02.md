# llama.cpp parity review (2026-09-02)

Companion to [`llama-cpp-gap-inventory.md`](llama-cpp-gap-inventory.md) (evidence-backed differential). This document ranks **what to do next** for parity with llama.cpp as a serving + CLI stack.

**North star:** same GGUF, same command shapes, same or better performance on hardware people own ([`north-star.md`](north-star.md)).

---

## Executive summary

| Area | Parity level | Priority |
|------|--------------|----------|
| Architecture coverage | ~11 audited + 4 dedicated engines vs 140 llama.cpp graphs | P0 — expand audited set |
| CLI flag semantics | Several **same flag, different meaning** (`-ngl`, `-e`, repeat penalty) | P0 — refuse or match |
| Server flags | ~10 argv flags vs llama-server’s dozens; env-heavy | P1 — `-cb`, `-np`, `-c`, `--api-key` |
| Sampling / grammar | Large sampler + grammar surface missing on API/CLI | P1 |
| Tools (quantize, perplexity) | No in-tree quantize or corpus eval | P1 |
| Serving / batching | Continuous batching now **auto on Metal**; streaming still buffered under CB | P1 — incremental CB streams |
| Metal concurrency | Mitigated (CB default + private-path gate); per-request Metal KV is proper fix | P0 — track #46 |

---

## P0 — Correctness and trust

### 1. Flag semantics that silently diverge (fix or refuse)

From gap inventory §4.1:

- **`-ngl`**: llama.cpp partial layer offload vs ferrox all-or-nothing — **refuse with clear error** until partial placement exists, or implement layer counting.
- **`-e` / escape**: default off in ferrox vs on in llama.cpp — default-on or `--no-escape` alias parity.
- **`--repeat-penalty`**: windowed vs full-history — add `--repeat-last-n` and match default 64.

### 2. Metal parallel decode (#46)

**Done on branch `fix/metal-parallel-concurrency` (partial):**

- Auto-enable continuous batching on Metal (safe multi-request path)
- Single-flight gate for private-loop Metal when CB disabled
- Poison-tolerant `metal_attn_kv` locks + empty-`hidden` CPU fallback
- CLI `--cont-batching` / `-cb`, `--no-cont-batching`

**Still open:**

- Per-request Metal KV residency (true parallel private path)
- Incremental streaming under continuous batching
- Metal concurrency integration test on real GGUF

### 3. Architecture refusals vs silent wrong

The unaudited-arch refusal gate is **good** (better than silent wrong). Next:

- Split generic `UnauditedArchitecture` messages by triage verdict (fixture-away vs new-code) — gap inventory §1.3
- Close **fixture-away** rows first: `bailingmoe2`, `minimax-m2`, `olmo2`/`seed_oss` post-norms wiring

---

## P1 — Serving parity

### Continuous batching & parallelism

| llama.cpp | ferrox (after this branch) | Gap |
|-----------|----------------------------|-----|
| `-cb` / `--cont-batching` | `--cont-batching`, `-cb`, `FERROX_CONTINUOUS_BATCHING` | **CLI added**; document auto-default on Metal |
| `-np` / `--parallel` | `FERROX_CB_MAX_SEQS` only | Add `-np` → env |
| Slot save/load | None | Missing |
| Streamed CB output | Token stream | ferrox buffers full completion under CB |

### Server flags → env today

High-value CLI additions (map to existing `FERROX_*`):

- `-c` / `--ctx-size` → context ceiling
- `--api-key` → `FERROX_API_KEY`
- `-fa` → `FERROX_METAL_ATTN` / flash-attn toggles
- `-b` / `-ub` → prefill/decode batch envs

### API surface

Strong: OpenAI chat/completions, SSE, embeddings, health handshake, admin models.

Gaps (inventory §3): `logit_bias` dropped on chat, JSON schema under CB, Anthropic/OpenAI responses parity edges, multimodal.

---

## P1 — CLI & tools

### Missing llama.cpp tools

| Tool | ferrox | Action |
|------|--------|--------|
| `quantize` | None | **Critical** — users need llama.cpp side-by-side |
| `perplexity` / benchmarks | `bench`, `parity`, no corpus ppl | Add perplexity or document `ferrox bench` scope |
| `gguf-split` | read shards only | Merge/split utility |
| `imatrix` | None | Lower priority |

### `ferrox run` vs `llama-cli`

Align: `-n` default (-1 = EOS), `-cnv`/`-no-cnv`, `-sys` spelling, `-hf` on run path, grammar/json flags.

---

## P2 — Performance parity

- **Engine bench:** `ferrox bench` vs `llama-bench` — keep receipt discipline ([`benchmarks/README.md`](../../benchmarks/README.md))
- **HTTP bench:** `ferrox serve-bench` vs `llama-server` load — extend parallel/concurrency scenarios
- **Kernel parity:** Metal/CUDA gap inventory in [`llama-cpp-gap-inventory.md`](llama-cpp-gap-inventory.md) §2

---

## P2 — Architecture scale

- **Model layer reorg** ([`model-layer-reorg.md`](model-layer-reorg.md)) — prerequisite for 140-arch maintenance
- **Out-of-core MoE** — separate track
- **Vulkan** — verdict GO; backend seam refactor pending

---

## Recommended roadmap slices

1. **Merge `fix/metal-parallel-concurrency`** — Metal serving safe by default
2. **Flag semantics sprint** — `-ngl`, escape, repeat-last-n (refuse > wrong)
3. **Server CLI parity pack** — `-np`, `-c`, `--api-key`, document env mapping table
4. **Fixture-away architectures** — 3–5 new audited rows with tiny GGUF fixtures
5. **CB streaming** — incremental SSE from batch worker (closes UX gap vs llama-server)
6. **Quantize tool** — or official doc that points to llama.cpp quantize with ferrox-compatible outputs

---

## Verification discipline

Every parity claim needs:

1. Side-by-side command on same GGUF
2. Test that fails when reverted
3. Receipt or HTTP bench JSON checked in

Use [`llama-cpp-gap-inventory.md`](llama-cpp-gap-inventory.md) as the evidence ledger; update counts when code changes, not docs alone.

---

## References

- Issue [#46](https://github.com/antonellof/ferrox/issues/46) — Metal parallel decode
- [`docs/plans/metal-parallel-concurrency.md`](metal-parallel-concurrency.md) — design
- [`docs/CONFIG.md`](../../CONFIG.md) — env vars
- [`docs/API.md`](../../API.md) — HTTP surface
