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
  wrap. Dense only — the MoE router is ported and tested
  (`ferrox_moe::route_gemma4_moe`) but the loader still expects
  `ffn_gate.weight`, so a MoE Gemma-4 GGUF does not load yet.
- **MiniMax-M3**: the block-sparse block selection is ported and tested
  (`ferrox_core::block_sparse`); the engine itself is still fail-closed
  — no loader, no 256-expert sigmoid MoE, no MTP draft heads.
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
- Anthropic Messages: `POST /v1/messages` streaming and buffered
  (thinking and tool blocks, protocol-native `ping` keepalive) plus
  `POST /v1/messages/count_tokens`
- Presence and frequency penalties, best-effort `json_object`
- Continuous batching and chunked prefill
- `ferrox serve-bench`: concurrency, TTFT, TPOT and queueing numbers
  for a live server, with the methodology (positional split, pooled
  nearest-rank percentiles, whole-run throughput) tested socket-free
- Live serving telemetry (`GET /v1/stats`, `GET /v1/requests`) and an
  elastic KV/expert split that can be reported and re-sized without a
  restart (`GET /v1/cache/status`, `POST /v1/cache/rebuild`), behind a
  maintenance gate that refuses rather than queues while it happens
- `reasoning_content`: a reasoning model's chain of thought is split
  out of `content`, streamed as it arrives rather than at the end
- Tool calls in nine wire formats, not one — the format the served
  checkpoint's family emits, then the prompt-engineered one — and every
  call in a response, not the first
- Prompts rendered by *evaluating* the checkpoint's own
  `tokenizer.chat_template`, with `chat_template_kwargs` and
  `reasoning_effort` passed through (the effort quantized onto what that
  checkpoint's template really grades)

## Serving policy (`ferrox-edge`)

A Rust port of the host-side decision logic in
[FreeToken](https://github.com/FlashML-org/FreeToken)
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)): the parts of an
edge-native MoE engine that *decide* rather than compute. Tensor-free
and testable without a GPU — each module takes measured numbers and
returns a decision.

| Module | Decides |
|---|---|
| `qstar` | how many of a step's expert-cache misses to fetch over PCIe vs. run on the CPU, from measured bandwidths |
| `expert_cache` | which experts stay resident, as one global LRU over a flat `(layer, expert)` id space |
| `expert_slots` | executes those residency plans against a bounded slot pool, and counts what crossed the link |
| `dsv4` | per-layer KV tier sizing, and which compressor each layer runs (none / CSA / HCA) |
| `radix` | which prefix of a prompt is already computed — page-keyed, node-sharing, with sliding-window and recurrent-state variants |
| `pool` | how VRAM splits between the expert cache and KV, and how it is re-split live |
| `placement` | which layers decode on the CPU when the expert banks exceed the host's page-locking budget |
| `scheduler` | admission, chunked-prefill sizing, and what a chunk reserves |
| `parser` | where reasoning ends and the answer begins; which tool was called, in which format |
| `effort` | which reasoning-effort dialect a checkpoint speaks, probed from its own template |
| `stats` | what a server may honestly claim about its own throughput and latency |
| `maintenance` | whether a request, a cache rebuild or a stop may proceed right now |
| `residency` | which class a layer's expert bank settles into, and what that then forbids |
| `supervisor` | whether a start spawns, no-ops or conflicts; whether a death was asked for; which process the OOM killer should take |
| `state_pool` | which recurrent-state slot a request holds, and where a prefill freezes one |
| `bench_profile` | where this machine's measured bandwidth profile lives, and when it may be trusted |

Wired in today: the two parsers (chat completions, streaming and
buffered), the stop-string withhold rule, which `ferrox-server`'s
`StopMatcher` now delegates to so there is one implementation of it, and
`effort` — probed once per checkpoint at load, then applied to every
request's `chat_template_kwargs`.
The cache and placement policies are complete and tested but are not
yet driving the decoder — see [`ROADMAP.md`](ROADMAP.md).
`expert_slots` closes the gap between planning residency and performing
it: a warm decode step provably copies zero bytes, on a host pool. No
GPU backend implements its `SlotDevice` trait yet, so on a real card
that property is written down and not yet measured.

`supervisor` is the same shape for processes: the lifecycle rules are
here and testable, and spawning sits behind a `ProcessHost` the caller
supplies. Ferrox ships one binary, so nothing implements that trait
yet. The rules are the reason it exists at all, since each is a race
you would otherwise only meet in production: a retried start must not
become a second engine, one party reaps and everyone else waits on
what it publishes (a poll transiently lies), the stop-requested latch
is set before the signal so a crash mid-stop is not read as an
unplanned death, shutdown is permanent so a start queued behind the
final stop is rejected, and a recorded child is re-adopted only on
`(pid, start_time, argv, port)` -- `start_time` being what stops a
recycled PID from being adopted as the engine.

Ferrox Studio, the web UI in [`ui/`](../ui), is a separate app that
talks to this API over HTTP. `ferrox-server` does not serve it.

See [`API.md`](API.md) and [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).
