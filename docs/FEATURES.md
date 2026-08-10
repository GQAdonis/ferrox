# Features

Ferrox is a pure-Rust GGUF inference engine for dense and MoE models.
Weights stay quantized on mmap; dequantization is fused into matvecs.
Backends: CPU, Apple Metal, and CUDA.

## Models

Verified against llama.cpp on the same host and GGUF (Gap =
`llama_pred / ferrox_pred`; &lt;1 means Ferrox is faster):

- **Dense GQA** — TinyLlama, Llama 3.2, Mistral-7B, SmolLM2,
  Qwen2.5/Qwen3, Gemma-3/4, Phi-4. Metal decode leads on the small and
  mid models (Qwen2.5-0.5B **~0.63×**, SmolLM2 **~0.76×**, Gemma-3-1B
  **~0.74×**) and is near parity on Llama-3.2-3B / Mistral-7B. **CPU decode
  is behind everywhere (1.3-2.6×), and prefill is behind on both
  backends.**
- **MoE** — OLMoE-1B-7B; still behind on Metal decode (~1.3-1.6×). (Metal Concurrent + fused encode groups +
  `MoeMemRanges` + `mul_mv_id` / prefill `mul_mm_id` + fused attn+O
  residual CB (`FERROX_METAL_PREFILL_FUSE_O=1`); CPU int-dot +
  interleaved Q4_K).
- **MLA** — dense-lead and MoE-after-dense `deepseek2` / `mistral4`.
- **Gemma-4** — dedicated engine (per-layer emb + shared KV + SWA/full)
  + SPM-style `gemma4` BPE + `<|turn>` chat wrap.
- **Also loadable** — yi, qwen2moe / qwen3moe (e.g. MiroThinker GGUFs),
  Gemma-2, Phi-3, Llama-3.1, GLM4 when tensors are present (not in the
  published suite).

Full matrix and pins: [`MODELS.md`](MODELS.md) ·
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

## Backends

| Backend | Capabilities |
|---|---|
| **CPU** | Dense + MoE; int8×int8 matvec on by default (`FERROX_CPU_INT_DOT=0` opts out) + interleaved Q4_Kx8 / Q8_0x4 GEMV, Q8_0x4 batch GEMM for prefill, Q5/Q6 int-dot; pool sized to performance cores |
| **Metal** | FA-vec attention (decode d=64/96/128/256, prefill d=128/256), concurrent FFN/QKV encode, MoE Concurrent + fused groups + `MoeMemRanges` + `mul_mm_id` prefill, quantized KV (`q8_0` / `turbo8` / `fp8` / `turbo4`) |
| **CUDA** | Matvec, resident weights, FFN fuse (`--features cuda`) |

## CLI

llama.cpp-style completion flags (`-m`, `-p`, `-n`, `-ngl`, `--ctk`, …),
plus `ferrox chat`, `ferrox pull` (Hugging Face Hub), `inspect`, `archs`,
and `presets`. See [`CLI.md`](CLI.md).

`ferrox bench -m model.gguf` is a `llama-bench` work-alike (`pp512` /
`tg128`, median ± stddev); `--compare` runs `llama-bench` alongside it
and prints the gap, and `--suite` drives every entry in
[`benchmarks/suite.json`](../benchmarks/suite.json) and regenerates the
engine table in [`RESULTS.md`](../benchmarks/RESULTS.md). That is the
*engine* number — the *serving* number comes from
[`benchmarks/run_suite.py`](../benchmarks/README.md).

## Server

OpenAI-compatible HTTP API:

- Chat completions (SSE), completions, tokenize / detokenize
- Decoder embeddings (mean/last pool)
- Anthropic-shaped `POST /v1/messages` (non-stream text)
- Presence / frequency penalties, best-effort `json_object`
- Optional web UI (`--ui-server`), continuous batching, chunked prefill

See [`API.md`](API.md) and [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).
