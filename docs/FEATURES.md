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
- **MoE** — OLMoE-1B-7B (Metal Concurrent encode + `MoeMemRanges` +
  `mul_mv_id`; CPU int-dot path).
- **MLA** — dense-lead `deepseek2` / `mistral4` on CLI and server.
- **Also loadable** — yi, qwen3moe (e.g. MiroThinker GGUFs), GLM4 when
  tensors are present.

Full matrix and pins: [`MODELS.md`](MODELS.md) ·
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

## Backends

| Backend | Capabilities |
|---|---|
| **CPU** | Dense + MoE; optional int8×int8 matvec (`FERROX_CPU_INT_DOT`) |
| **Metal** | FA-vec attention (decode d=64/96/128/256, prefill d=128/256), concurrent FFN/QKV encode, MoE Concurrent + `MoeMemRanges`, quantized KV (`q8_0` / `turbo8` / `fp8` / `turbo4`) |
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
