# Roadmap

What works today: [`MODELS.md`](MODELS.md) · CLI: [`CLI.md`](CLI.md) ·
speed: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

**Goal:** ≥ [llama.cpp](https://github.com/ggerganov/llama.cpp) tok/s on the same host / backend / GGUF.

## Now

1. **Metal** — keep Llama-3.1-8B decode ≥1×; close prefill gap; don’t regress TinyLlama / 1B / 3B pins by >2%.
2. **Receipts** — promote Qwen3 / Gemma-3 / Phi-3 / Mistral from “works” → verified (token parity + Metal pin where relevant).
3. **Qwen2-MoE** — re-run oracle after QKV-bias fix.

## Next

1. **MLA / DSA** — DeepSeek-2-style and GLM DSA serve paths (fail-closed until then).
2. **Recurrent / hybrid / T5** — dedicated engines when needed.
3. **Kimi / DeepSeek V4** — real-checkpoint e2e when storage / shapes allow.
4. **CUDA** — reopen after Metal program; fair-chat still far behind.

## Later

- Continuous-batching throughput receipt
- Interactive `ferrox chat` TUI / optional UI apps
  (`ferrox -m/-p` completion already ships — see [`CLI.md`](CLI.md))

## Rules

Evidence-first. No “supported” or “fast” without a receipt.
No Candle / Crane / ds4 as dependencies — rewrite in-tree.
