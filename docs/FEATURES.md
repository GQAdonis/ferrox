# Features

Ferrox is a pure-Rust GGUF inference engine for dense and MoE models.
Weights stay quantized when the file is mmapped, and dequantization
happens inside the matvec. Backends: CPU, Apple Metal, and CUDA.

## Models

Measured against llama.cpp on the same host and the same GGUF with
`ferrox bench`. Gap = `llama / ferrox`, so anything under 1 means Ferrox
is faster.

- **Dense GQA**: TinyLlama, Llama 3.2, Mistral-7B, SmolLM2,
  Qwen2.5/Qwen3, Gemma-3/4. Metal decode leads on several small models
  (SmolLM2 **~0.67×**, Qwen2.5-0.5B **~0.70×**, Qwen3-0.6B **~0.71×**,
  Gemma-3-1B **~0.88×**), and Llama-3.2-3B sits at **~0.96×**, ahead of
  llama.cpp rather than behind it. Mistral-7B and Llama-3.2-1B Q4_K_M
  land on 1.00×. Dense Metal prefill is closed: every dense `pp512` row
  is 1.02–1.08×. **CPU decode is still behind everywhere (~1.17–2.44×),
  and CPU prefill is behind on 6 of 8 rows.**
- **Phi-4**: CPU **and** Metal. Partial rotary (`n_rot < head_dim`) and
  LongRoPE `attn_factor` ride the Metal RoPE kernels as `rot_dim` and
  `mscale` uniforms, matching ggml's `rope_yarn` (the magnitude scale
  reaches the rotated channels only). The published Metal speed rows
  predate that fix and need measuring again before you quote them.
- **MoE**: OLMoE-1B-7B, still behind on Metal decode (~1.41×). Metal
  Concurrent plus fused encode groups, `MemRanges`, `mul_mv_id` and
  prefill `mul_mm_id`, and a fused attn+O residual command buffer
  (`FERROX_METAL_PREFILL_FUSE_O=1`). On CPU: int-dot and interleaved
  Q4_K.
- **MLA**: dense-lead and MoE-after-dense `deepseek2` / `mistral4`.
- **Gemma-4**: dedicated engine (per-layer embeddings, shared KV,
  SWA/full), an SPM-style `gemma4` BPE tokenizer, and the `<|turn>` chat
  wrap.
- **Also loadable**: yi, qwen2moe / qwen3moe (MiroThinker GGUFs, for
  example), Gemma-2, Phi-3, Llama-3.1, and GLM4 when the tensors are
  there. None of these are in the published suite.
- **MoE routing bias** (`exp_probs_b`, DeepSeek-V3's aux-loss-free
  selection bias) plus `expert_weights_scale` and
  `expert_weights_norm`, on the generic path. That one tensor is what
  used to stop dots1, ernie4_5-moe, bailingmoe2, exaone-moe,
  hunyuan-moe and afmoe from loading at all. It is checked against
  llama.cpp's own dots1 implementation reading the same synthetic
  checkpoint, and has not been validated on a published one.
- **Granite, MiniCPM and Command-R scalar multipliers stop the load.**
  `logit_scale`, `residual_scale`, `embedding_scale` and
  `attention.scale` are hyperparameters, not tensors, so the check for
  unread tensors never sees them. A checkpoint that declares one of them
  with a value that changes the maths stops with an error naming the
  key. `minicpm` stops outright, because llama.cpp applies its three
  multipliers even when the file carries no key at all.
- **Parallel-residual architectures do not load either**: `command-r`,
  `cohere2`, `cohere2moe`, `falcon`, `gptneox`, `phi2`, `plamo`. They
  sum `inpL + attn_out + ffn_out` once instead of taking two sequential
  residuals. That is a different graph, and the tensor list looks
  identical either way, so these are listed by name.

Full matrix: [`MODELS.md`](MODELS.md) ·
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) ·
[`benchmarks/suite.json`](../benchmarks/suite.json).

## Backends

| Backend | Capabilities |
|---|---|
| **CPU** | Dense and MoE. int8×int8 matvec on by default (`FERROX_CPU_INT_DOT=0` opts out), interleaved Q4_Kx8 / Q8_0x4 GEMV, Q8_0x4 batch GEMM for prefill, Q5/Q6 int-dot, pool sized to performance cores |
| **Metal** | FA-vec attention (decode d=64/96/128/256, prefill d=128/256), concurrent FFN/QKV encode, MoE Concurrent with fused groups, `MemRanges`, `mul_mm_id` prefill, quantized KV (`q8_0` / `turbo8` / `fp8` / `turbo4`) |
| **CUDA** | Matvec, resident weights, FFN fuse (`--features cuda`) |

**CUDA compiles and runs, and nobody has benchmarked it.** Every number
in this repo was taken on CPU or Apple Metal. GPU acceleration on
Windows and Linux means CUDA, and the bar CUDA is held to is "must
compile". There is no pinned benchmark host for it and no published
timings. Treat a Windows or Linux install as **CPU-only in practice**
until that changes. `/health` reports the same thing per capability,
with a reason string, instead of quietly greying a control out.

## CLI

llama.cpp-style completion flags (`-m`, `-p`, `-n`, `-ngl`, `--ctk`, …),
plus `ferrox chat`, `ferrox pull` (Hugging Face Hub), `inspect`, `archs`,
and `presets`. See [`CLI.md`](CLI.md).

`ferrox bench -m model.gguf` works like `llama-bench`: the same `pp512`
and `tg128` workloads, reported as a median with a population stddev.
Add `--compare` to run `llama-bench` alongside it and print the gap.
`--suite` drives every entry in
[`benchmarks/suite.json`](../benchmarks/suite.json) and regenerates
[`RESULTS.md`](../benchmarks/RESULTS.md). See
[`benchmarks/README.md`](../benchmarks/README.md).

## Server

OpenAI-compatible HTTP API:

- Chat completions (SSE, optionally resumable: `id:` + `retry:` +
  `Last-Event-ID` replay and a JSON polling fallback), completions,
  tokenize / detokenize
- Decoder embeddings (mean/last pool)
- Anthropic-shaped `POST /v1/messages` (non-stream text)
- Presence and frequency penalties, best-effort `json_object`
- Continuous batching and chunked prefill

Ferrox Studio, the web UI in [`ui/`](../ui), is a separate app that
talks to this API over HTTP. `ferrox-server` does not serve it.

See [`API.md`](API.md) and [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).
