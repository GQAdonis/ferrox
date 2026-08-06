# Features

Ferrox is a pure-Rust GGUF inference engine for dense and MoE models.
Weights stay quantized on mmap; dequantization is fused into matvecs.
Backends: CPU, Apple Metal, and CUDA.

## Models

Verified against llama.cpp on the same host and GGUF (Gap =
`llama_pred / ferrox_pred`; &lt;1 means Ferrox is faster):

- **Dense GQA** — TinyLlama, Llama 3.1/3.2, Mistral-7B, SmolLM2,
  Qwen2.5/Qwen3, Gemma-2/3, Phi-3/4. Llama-8B Metal fair-chat Gap
  **~0.92×**; Llama-3.2-3B **~0.97×**.
- **MoE** — OLMoE-1B-7B (Metal Concurrent + fused encode groups +
  `MoeMemRanges` + `mul_mv_id` / prefill `mul_mm_id` + fused attn+O
  residual CB (`FERROX_METAL_PREFILL_FUSE_O=1`); CPU int-dot +
  interleaved Q4_K).
- **MLA** — dense-lead and MoE-after-dense `deepseek2` / `mistral4`.
- **Gemma-4** — dedicated engine (per-layer emb + shared KV + SWA/full);
  tokenizer pin pending.
- **Also loadable** — yi, qwen3moe (e.g. MiroThinker GGUFs), GLM4 when
  tensors are present.

Full matrix and pins: [`MODELS.md`](MODELS.md) ·
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

## Backends

| Backend | Capabilities |
|---|---|
| **CPU** | Dense + MoE; `FERROX_CPU_INT_DOT` int8×int8 + interleaved Q4_Kx8 / Q8_0x4 GEMV + Q5/Q6 int-dot |
| **Metal** | FA-vec attention (decode d=64/96/128/256, prefill d=128/256), concurrent FFN/QKV encode, MoE Concurrent + fused groups + `MoeMemRanges` + `mul_mm_id` prefill, quantized KV (`q8_0` / `turbo8` / `fp8` / `turbo4`) |
| **CUDA** | Matvec, resident weights, FFN fuse (`--features cuda`) |

## CLI

llama.cpp-style completion flags (`-m`, `-p`, `-n`, `-ngl`, `--ctk`, …),
plus `ferrox chat`, `ferrox pull` (Hugging Face Hub), `inspect`, `archs`,
and `presets`. See [`CLI.md`](CLI.md).

## Server

OpenAI-compatible HTTP API:

- Chat completions (SSE), completions, tokenize / detokenize
- Decoder embeddings (mean/last pool)
- Anthropic-shaped `POST /v1/messages` (non-stream text)
- Presence / frequency penalties, best-effort `json_object`
- Optional web UI (`--ui-server`), continuous batching, chunked prefill

See [`API.md`](API.md) and [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).
