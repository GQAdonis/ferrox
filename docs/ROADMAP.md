# Roadmap

What works today: [`MODELS.md`](MODELS.md) · CLI: [`CLI.md`](CLI.md) ·
speed: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) ·
API: [`API.md`](API.md).

**Goal:** ≥ [llama.cpp](https://github.com/ggerganov/llama.cpp) tok/s on the same host / backend / GGUF.
Evidence-first: no “supported” or “fast” without a receipt. No Candle / Crane /
ds4 deps — rewrite in-tree.

## Now (foundation P1–P5)

| Phase | Focus | Notes |
|---|---|---|
| **P1** | Metal prefill | FA-vec prefill; pin `prompt_per_second` on Llama-3.1-8B Metal |
| **P2** | MLA GGUF → `MlaEngine` | deepseek2 → mistral4 → GLM-4.7 / DS V3; fail-closed until wired |
| **P3** | Hybrid GDN | `HybridEngine` + GDN; smoke Qwen3.5 GGUF, then qwen3next / 35moe |
| **P4** | Gemma-4 text | Admit graph (+ MoE-A4B); VL deferred to P7 |
| **P5** | KV quant | P5a Metal `q8_0` / `-ctk`; P5b turbo4 / fp8 / turbo{8,3} family |

Also on this horizon (unchanged intent):

- **Receipts** — real GGUF oracles for Gemma-2, Qwen2-MoE, Mistral, Mixtral (suite entries exist; pins pending checkpoints).
- **Metal MoE** — keep OLMoE expert placement; fuse only if profiling proves it.
- **Frontier** — Kimi multi-layer → full e2e; GLM-5.2 / DeepSeek V4 real quants (engines exist; fail-closed until receipts).
- **CUDA** — fair-chat pins via `run_suite.py --backend cuda --host-label …`; staged ≥0.5× then parity (no invented numbers).

## Next (models P6–P8)

| Phase | Focus | Notes |
|---|---|---|
| **P6** | Text / MoE matrix | Qwen3-MoE, Phi-4, GLM4/4.7, Llama4, MiniMax (fail-closed until sigmoid MoE+MTP), MiroThinker; multi-shard GGUF clarity |
| **P7** | VL | Qwen3-VL, Gemma4-VL, Mistral3-VL + mmproj GGUF; server `image_url` later |
| **P8** | MTP / embed / GLM-5.2 | Speculative MTP heads, embeddings engine, GLM-5.2 / DSA e2e on real GGUF |

Also: recurrent / hybrid / T5 stubs already exist — implement after receipts, not before.

## Serving (P9–P11)

| Phase | Focus | Notes |
|---|---|---|
| **P9** | Server API | Anthropic Messages, embeddings, tokenize, guided decode, MCP, web UI — see [`API.md`](API.md) Planned |
| **P10** | Runtime scale | CB throughput pin, multi-GPU/TP, PD disagg, chunked prefill, CPU KV offload, HF hub GGUF fetch |
| **P11** | Polish | Metal fair-chat ≥ llama, CUDA pins, optional ISQ (no Candle), agent docs; keep Gap methodology |

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
