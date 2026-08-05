# Roadmap

What works today: [`MODELS.md`](MODELS.md) · CLI: [`CLI.md`](CLI.md) ·
speed: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) ·
API: [`API.md`](API.md) · agents: [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).

**Goal:** ≥ [llama.cpp](https://github.com/ggerganov/llama.cpp) tok/s on the same host / backend / GGUF.
Evidence-first: no “supported” or “fast” without a receipt. No Candle / Crane /
ds4 deps — rewrite in-tree.

## Now (foundation P1–P5)

| Phase | Focus | Notes |
|---|---|---|
| **P1** | Metal prefill | FA-vec prefill d=128 landed; Llama-3.1-8B Metal fair-chat ~parity (~0.97×) — re-check `prompt_per_second` under quiet host |
| **P2** | MLA GGUF → `MlaEngine` | CLI + **ferrox-server** dense-lead deepseek2/mistral4; MoE-after-dense fail-closed; GLM-4.7 / DS V3 MoE still open |
| **P3** | Hybrid GDN | `gdn.rs` + loader scaffold; `HybridEngine` assemble + Qwen3.5 smoke still open |
| **P4** | Gemma-4 text | E2B fail-closed (**DedicatedOnly**: per-layer emb + shared KV + SWA/full head-dim); suite refuse pin; MoE-A4B / VL → P7 |
| **P5** | KV quant | Metal `q8_0` store + shared f16 dequant scratch landed; `fp8`/`turbo*` still warn→F16; CPU turbo4 sketch |

Also on this horizon (unchanged intent):

- **Receipts** — real GGUF oracles for Gemma-2, Qwen2-MoE, Mistral, Mixtral (suite entries exist; pins pending checkpoints).
- **Metal MoE** — keep OLMoE expert placement; fuse only if profiling proves it.
- **Frontier** — Kimi multi-layer → full e2e; GLM-5.2 / DeepSeek V4 real quants (engines exist; fail-closed until receipts).
- **CUDA** — fair-chat pins via `run_suite.py --backend cuda --host-label …`; staged ≥0.5× then parity (no invented numbers).

## Next (models P6–P8)

| Phase | Focus | Notes |
|---|---|---|
| **P6** | Text / MoE matrix | Phi-4-mini Metal pin landed; Qwen3-MoE / GLM4 / Llama4 / MiniMax / MiroThinker still open |
| **P7** | VL | Qwen3-VL, Gemma4-VL, Mistral3-VL + mmproj GGUF; server `image_url` later |
| **P8** | MTP / embed / GLM-5.2 | Speculative MTP heads, embeddings engine, GLM-5.2 / DSA e2e on real GGUF |

Also: recurrent / hybrid / T5 stubs already exist — implement after receipts, not before.

## Serving (P9–P11)

| Phase | Focus | Notes |
|---|---|---|
| **P9** | Server API | Tokenize / detokenize / completions / decoder embeddings + **Anthropic `/v1/messages`** (non-stream text); guided decode, MCP, web UI still Planned in [`API.md`](API.md) |
| **P10** | Runtime scale | CB opt-in exists (`FERROX_CONTINUOUS_BATCHING`); throughput pin + TP/PD/HF hub still open |
| **P11** | Polish | Agent cookbook ([`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md)); Metal/CUDA fair-chat pins + optional ISQ still open |

Also: continuous-batching throughput pin (`benchmarks/cb_throughput.py` against a live server) under **P10**.

## Shipped (do not re-list as open)

- Evidence baseline + RESULTS validation (`render_results.py`)
- Metal FA-vec d=96/256; softcap CPU + Metal legacy GQA
- Overlapped SSE; API compatibility matrix; `ferrox chat` REPL
- CUDA suite backend + host-label pins workflow
- Fail-closed architecture registry (`ferrox archs`)

## Rules

Evidence-first. Prefer honest scaffolding and clear `DedicatedOnly` /
`Deferred` reasons over silent wrong graphs.
No Candle / Crane / ds4 as dependencies — rewrite in-tree.
