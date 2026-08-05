# Roadmap

What works today: [`MODELS.md`](MODELS.md) · CLI: [`CLI.md`](CLI.md) ·
speed: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) ·
API: [`API.md`](API.md).

**Goal:** ≥ [llama.cpp](https://github.com/ggerganov/llama.cpp) tok/s on the same host / backend / GGUF.

## Now (evidence gates)

1. **Receipts** — real GGUF oracles for Gemma-2, Qwen2-MoE, Mistral, Mixtral (suite entries exist; pins pending checkpoints).
2. **Metal MoE** — keep OLMoE expert placement; fuse only if profiling proves it.
3. **Frontier** — Kimi multi-layer → full e2e; GLM-5.2 / DeepSeek V4 real quants (engines exist; fail-closed until receipts).
4. **CUDA** — fair-chat pins via `run_suite.py --backend cuda --host-label …`; staged ≥0.5× then parity (no invented numbers).

## Next

1. **MLA GGUF load** — wire DeepSeek-2 / Mistral-4 weights into `MlaEngine`.
2. **Recurrent / hybrid / T5 / VL** — stubs in `ferrox-models`; implement after receipts.
3. Continuous-batching throughput pin (`benchmarks/cb_throughput.py` against a live server).

## Shipped (do not re-list as open)

- Evidence baseline + RESULTS validation (`render_results.py`)
- Metal FA-vec d=96/256; softcap CPU + Metal legacy GQA
- Overlapped SSE; API compatibility matrix; `ferrox chat` REPL
- CUDA suite backend + host-label pins workflow

## Rules

Evidence-first. No “supported” or “fast” without a receipt.
No Candle / Crane / ds4 as dependencies — rewrite in-tree.
