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
  wrap. Dense only: the MoE router is ported and tested
  (`ferrox_moe::route_gemma4_moe`) but the loader still expects
  `ffn_gate.weight`, so a MoE Gemma-4 GGUF does not load yet.
- **MiniMax-M3**: the block-sparse block selection is ported and tested
  (`ferrox_core::block_sparse`). The engine itself stops with an error:
  no loader, no 256-expert sigmoid MoE, no MTP draft heads.
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

Two ways to start the same server. `ferrox serve` is a subcommand of the
main binary behind an optional `serve` feature, off by default for
`cargo install` because it pulls in 98 crates a completion-only user
does not need. `ferrox-server` is that same server as its own
executable, and both parse identical arguments through identical code.
The prebuilt release binary is built with `serve`, so the downloaded
`ferrox` does both.

OpenAI-compatible HTTP API:

- Chat completions (SSE, optionally resumable: `id:` + `retry:` +
  `Last-Event-ID` replay and a JSON polling fallback), completions,
  tokenize / detokenize
- `POST /v1/cancel` stops a running generation by request id, which is
  the stop path a resumable stream needs, since closing its socket no
  longer ends it
- Decoder embeddings (mean/last pool)
- Anthropic Messages: `POST /v1/messages` streaming and buffered
  (thinking and tool blocks, protocol-native `ping` keepalive) plus
  `POST /v1/messages/count_tokens`
- `POST /v1/responses`, the surface `codex` speaks, streaming and
  buffered. This server keeps no responses, so the two lookups by
  response id answer 404
- Presence and frequency penalties, best-effort `json_object`
- Continuous batching and chunked prefill
- Paged KV: shared page storage many requests read through a block
  table, with a radix tree over reference-counted page groups so
  conversations off one system prompt share its KV rather than each
  holding a copy. **CPU only**, since the paged attention path returns
  wrong tokens on a GPU backend. Off unless
  `FERROX_PAGED_KV_BLOCKS` is set, and the server stops at startup when
  it is set beside `-dev metal` or `-dev cuda`. See
  [`CONFIG.md`](CONFIG.md)
- On a model whose layers *all* slide by the same window, that
  window slides during decode, so a request holds its prompt and a
  window rather than its whole context — and admission prices it
  that way, so a store too small for the whole context still serves
  it. A tool call anchors the slide at the position the next agentic
  turn will rejoin at, and the anchor is dropped once the cursor
  drifts a window past it. An alternating-SWA model (gpt-oss,
  Gemma-3) does not slide: a page group holds one block in every
  layer, and the full-attention layers still read position 0
- `ferrox serve-bench`: concurrency, TTFT, TPOT and queueing numbers
  for a live server, with the methodology (positional split, pooled
  nearest-rank percentiles, whole-run throughput) tested socket-free
- Live serving telemetry (`GET /v1/stats`, `GET /v1/requests`) and an
  elastic KV/expert split that can be reported and re-sized without a
  restart (`GET /v1/cache/status`, `POST /v1/cache/rebuild`). A request
  that arrives mid-rebuild is turned away with an error rather than
  parked in a queue behind it
- `reasoning_content`: a reasoning model's chain of thought is split
  out of `content`, streamed as it arrives rather than at the end
- Tool calls in eleven wire formats, not one: the format the served
  checkpoint's family emits, then the prompt-engineered one, and every
  call in a response rather than the first. Five of the eleven stream
  their arguments as deltas
- Prompts rendered by *evaluating* the checkpoint's own
  `tokenizer.chat_template`, with `chat_template_kwargs` and
  `reasoning_effort` passed through (the effort quantized onto what that
  checkpoint's template really grades)

## Serving policy (`ferrox-edge`)

A Rust port of the host-side decision logic in
[FreeToken](https://github.com/FlashML-org/FreeToken)
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)): the parts of an
edge-native MoE engine that *decide* rather than compute. Tensor-free
and testable without a GPU. Each module takes measured numbers and
returns a decision.

### Driving something today

| Module | Decides | Where it runs |
|---|---|---|
| `parser` | where reasoning ends and the answer begins, and which tool was called in which format | `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, streaming and buffered |
| `detokenize` | what text is safe to stream after one more token | the stop-string withhold rule, which `ferrox-server`'s `StopMatcher` delegates to so there is one implementation |
| `radix` | which prefix of a new prompt is already computed, page-keyed and node-sharing | the paged-KV serving path, where it shares KV pages between prompts by reference count |
| `anchor` | how far a window may slide, and where a tool call pins it so the next agentic turn rejoins rather than recomputes | the paged-KV serving path, on both the private generate loop and the continuous batcher |
| `scheduler` | admission, chunked-prefill sizing, and what a chunk reserves | the continuous batcher's status and pool accounting |
| `effort` | which reasoning-effort dialect a checkpoint speaks | probed once per checkpoint at load, then applied to every request's `chat_template_kwargs`, and advertised on `/v1/models` |
| `stats` | what a server may honestly claim about its own throughput and latency | `/v1/stats`, `/v1/requests`, `/admin/stats` |
| `maintenance` | whether a request, a cache rebuild or a stop may proceed right now | `POST /v1/cache/rebuild` and `POST /v1/admin/prepare-stop` |
| `pool` | how VRAM splits between the expert cache and KV, and how it is re-split live | the target geometry `POST /v1/cache/rebuild` validates against |
| `rebuild` · `outbox` · `footprint` | whether a re-split rolls back, what a stop receipt is worth, what this process really occupies | the same two admin endpoints |
| `dsv4` | per-layer KV tier sizing, and which compressor each layer runs (none / CSA / HCA) | the DeepSeek-V4 decoder |
| `bench_profile` · `bench_client` | when a measured bandwidth profile may be trusted, and what a serving benchmark may report | `ferrox bench-bw` and `ferrox serve-bench` |

### Complete, tested, and waiting for a consumer

`qstar` (the `q*` bandwidth split), `expert_cache`, `placement`,
`residency`, `cache_manager`, `cache_report`, `window_pool`,
`state_pool` and `supervisor`. Each is covered by unit tests and none of
them is on a serving path. Do not read a benchmark as evidence for any
of them. [`ROADMAP.md`](ROADMAP.md) says what each is waiting on.

`expert_slots` sits between the two: it executes the expert cache's copy
plans against a bounded slot pool, and a warm decode step copies zero
bytes on a host pool. `ferrox-cuda`'s `CudaExpertPool` implements its
`SlotDevice` trait under `--features cuda`, and that pool is
compile-verified with its hardware test left `#[ignore]`d, so on a real
card the property is written down and not yet measured.

`supervisor` is the same shape for processes: the lifecycle rules are
here and testable, and spawning sits behind a `ProcessHost` the caller
supplies. No ferrox binary spawns another one, so nothing implements
that trait yet. The rules are the reason it exists at all, since each is a race
you would otherwise only meet in production: a retried start must not
become a second engine, one party reaps and everyone else waits on
what it publishes (a poll transiently lies), the stop-requested latch
is set before the signal so a crash mid-stop is not read as an
unplanned death, shutdown is permanent so a start queued behind the
final stop is rejected, and a recorded child is re-adopted only on
`(pid, start_time, argv, port)` -- `start_time` being what stops a
recycled PID from being adopted as the engine.

Ferrox Studio, the web UI in [`ui/`](../ui), is a separate app that
talks to this API over HTTP. `ferrox-server` does not serve it, and
`GET /` on it is a 404.

See [`API.md`](API.md) and [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).
