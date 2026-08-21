# Features

Ferrox is a pure-Rust GGUF inference engine for dense and MoE models.
Weights stay quantized on mmap; dequantization is fused into matvecs.
Backends: CPU, Apple Metal, and CUDA.

## Models

Verified against llama.cpp on the same host and GGUF via `ferrox bench`
(Gap = `llama / ferrox`; &lt;1 means Ferrox is faster):

- **Dense GQA** — TinyLlama, Llama 3.2, Mistral-7B, SmolLM2,
  Qwen2.5/Qwen3, Gemma-3/4. Metal decode leads on several small models
  (SmolLM2 **~0.67×**, Qwen2.5-0.5B **~0.70×**, Qwen3-0.6B **~0.71×**,
  Gemma-3-1B **~0.88×**), and Llama-3.2-3B is **~0.96×** — ahead, not
  behind. Mistral-7B and Llama-3.2-1B Q4_K_M sit at 1.00×. Dense Metal
  prefill is closed (every dense `pp512` row is 1.02–1.08×).
  **CPU decode is still behind everywhere (~1.17–2.44×), and CPU
  prefill is behind on 6 of 8 rows.**
- **Phi-4** — CPU **and** Metal. Partial rotary (`n_rot < head_dim`) and
  LongRoPE `attn_factor` ride the Metal RoPE kernels as `rot_dim` /
  `mscale` uniforms, matching ggml's `rope_yarn` (the magnitude scale
  reaches the rotated channels only). The published Metal speed rows
  predate this and are owed a re-measurement.
- **MoE** — OLMoE-1B-7B; still behind on Metal decode (~1.41×). (Metal Concurrent + fused encode groups +
  `MoeMemRanges` + `mul_mv_id` / prefill `mul_mm_id` + fused attn+O
  residual CB (`FERROX_METAL_PREFILL_FUSE_O=1`); CPU int-dot +
  interleaved Q4_K).
- **MLA** — dense-lead and MoE-after-dense `deepseek2` / `mistral4`.
- **Gemma-4** — dedicated engine (per-layer emb + shared KV + SWA/full)
  + SPM-style `gemma4` BPE + `<|turn>` chat wrap.
- **Also loadable** — yi, qwen2moe / qwen3moe (e.g. MiroThinker GGUFs),
  Gemma-2, Phi-3, Llama-3.1, GLM4 when tensors are present (not in the
  published suite).
- **MoE routing bias** (`exp_probs_b`, DeepSeek-V3's aux-loss-free
  selection bias) plus `expert_weights_scale` / `expert_weights_norm`, on
  the generic path — the tensor is what the fail-closed gate used to
  refuse on dots1, ernie4_5-moe, bailingmoe2, exaone-moe, hunyuan-moe and
  afmoe. Checked against llama.cpp's own dots1 implementation reading the
  same synthetic checkpoint; not validated on a published one.
- **Granite / MiniCPM / Command-R scalar multipliers are refused, not
  implemented.** `logit_scale`, `residual_scale`, `embedding_scale` and
  `attention.scale` are hparams rather than tensors, so the
  tensor-consumption gate cannot see them; a checkpoint declaring a
  non-no-op value is refused by name. `minicpm` is refused outright,
  since llama.cpp applies its three multipliers even when the file
  carries no key at all.
- **Parallel-residual architectures are refused, not implemented** —
  `command-r`, `cohere2`, `cohere2moe`, `falcon`, `gptneox`, `phi2`,
  `plamo`. They sum `inpL + attn_out + ffn_out` once instead of taking
  two sequential residuals, which is a different graph and is invisible
  in the tensor list.

Full matrix: [`MODELS.md`](MODELS.md) ·
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) ·
[`benchmarks/suite.json`](../benchmarks/suite.json).

## Backends

| Backend | Capabilities |
|---|---|
| **CPU** | Dense + MoE; int8×int8 matvec on by default (`FERROX_CPU_INT_DOT=0` opts out) + interleaved Q4_Kx8 / Q8_0x4 GEMV, Q8_0x4 batch GEMM for prefill, Q5/Q6 int-dot; pool sized to performance cores |
| **Metal** | FA-vec attention (decode d=64/96/128/256, prefill d=128/256), concurrent FFN/QKV encode, MoE Concurrent + fused groups + `MoeMemRanges` + `mul_mm_id` prefill, quantized KV (`q8_0` / `turbo8` / `fp8` / `turbo4`) |
| **CUDA** | Matvec, resident weights, FFN fuse (`--features cuda`) |

**Platform honesty.** Every benchmarked number in this repo is CPU or
Apple Metal. GPU acceleration on Windows and Linux means CUDA, and CUDA
is held to "must compile" — there is no pinned benchmark host and no
published receipts for it. Treat a Windows or Linux install as
**CPU-only in practice** until that changes. `/health` says the same
thing per capability, with a reason string, rather than silently
greying a control out.

## CLI

llama.cpp-style completion flags (`-m`, `-p`, `-n`, `-ngl`, `--ctk`, …),
plus `ferrox chat`, `ferrox pull` (Hugging Face Hub), `inspect`, `archs`,
and `presets`. See [`CLI.md`](CLI.md).

`ferrox bench -m model.gguf` is a `llama-bench` work-alike (`pp512` /
`tg128`, median ± stddev); `--compare` runs `llama-bench` alongside it
and prints the gap, and `--suite` drives every entry in
[`benchmarks/suite.json`](../benchmarks/suite.json) and regenerates
[`RESULTS.md`](../benchmarks/RESULTS.md). See
[`benchmarks/README.md`](../benchmarks/README.md).

## Server

OpenAI-compatible HTTP API:

- Chat completions (SSE), completions, tokenize / detokenize
- Decoder embeddings (mean/last pool)
- Anthropic-shaped `POST /v1/messages` (non-stream text)
- Presence / frequency penalties, best-effort `json_object`
- Web UI (Ferrox Studio, `ui/`, a separate app on the same HTTP API),
  continuous batching, chunked prefill

See [`API.md`](API.md) and [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).
