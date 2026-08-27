# Roadmap

The goal is to match or beat [llama.cpp](https://github.com/ggerganov/llama.cpp)
tok/s on the same host, the same backend and the same GGUF. Current
numbers: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

What ships today: [`FEATURES.md`](FEATURES.md) ·
[`MODELS.md`](MODELS.md).

## Speed gaps against llama.cpp

Closed since the last pass over this list: bench last-token `lm_head`
only, CPU Q4_K batch GEMM (`gemm_q4_kx8_group` in `weight_matrix`),
SmolLM2 Metal greedy lm_head, OLMoE Metal gather plus `mul_mm_sg` and
Q4_0 `mul_mv_id`, the Qwen shared-expert loader fallback, CPU MoE
token-to-expert bucketing (`moe_ffn_batch`), and dense Metal prefill
stack QKV bias with QK-norm (Qwen2.5 / Qwen3 / Gemma-3).

Still open (see the Open section of
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)):

- **Metal prefill.** The Qwen2.5 / Qwen3 / Gemma-3 Q8_0 dense stack now
  includes QKV bias and QK-norm, which took it from roughly 18–21×
  hybrid CPU projection down to 1.2–2.1×. What remains: OLMoE
  gather → `mul_mm_sg` → scatter against a fused `kernel_mul_mm_id`,
  dense 1–3B (~1.5–3×), and a compiled graph or pre-encoded command
  buffer replay to get sub-1.5B models to 1× or better.
- **CPU prefill and decode.** Phi-4 and Mistral Q4_K `pp512` need
  measuring again after the GEMM change. i8mm SMMLA if the gap is still
  above 1×. A persistent decode threadpool.
- **Correctness.** Gemma-4 end-to-end chat smoke test. The older
  Gemma-2 Metal greedy check stays in the tree for regressions only,
  and is not part of the published suite.

## Where the project goes next

Beyond closing the measured gaps.

1. **Run bigger models on the same hardware.** Make Qwen3 35B-A3B Q5
   usable on a box that today handles Q4, or an 8B. Most of what
   follows serves this.
2. **RAM and VRAM optimization.** Residency planning already exists
   (`ferrox inspect-plan`). What is missing is acting on it hard enough
   to change which models fit: tighter KV (`turbo3`, quantized CTK),
   streaming expert residency, and not materializing activations
   nothing reads.
3. **Hybrid CPU/GPU, especially for MoE.** Routed experts are the
   natural split. Hot experts stay resident on the GPU, cold ones get
   streamed or run on CPU. `PlacementPlan` and `ExpertStore` are the
   groundwork.
4. **CUDA performance.** The kernels build and run on real hardware.
   Nobody has tuned them.
5. **Tool calling and full OpenAI API compatibility.** See
   [`API.md`](API.md). Grammar and JSON-schema constrained decoding,
   plus MCP invocation, are the gaps.
6. **Docker images**, so evaluating any of this stops requiring a Rust
   toolchain.

**Models**

- Gemma-4 tokenizer (`gemma4`) plus an end-to-end chat smoke test (the
  engine loads today)
- HybridEngine and Qwen3.5
- Llama 4 and MiniMax engines
- Vision (projector plus generate)
- Real GLM-5.2, DeepSeek V4 and full Kimi, run end to end
- MTP draft heads
- Qwen2-MoE and Mixtral pins, once the GGUF fits Host B

**Serving**

- Tool calling: the OpenAI `tools` / `tool_choice` request and response
  shape. *Eleven wire formats now parse* (`ferrox-edge::parser::tool_call`),
  every call in a response rather than the first, and a reasoning
  model's chain of thought comes back as `reasoning_content`. Five of
  the eleven stream `tool_calls[].index` argument deltas. What is left
  here is `tool_choice: required`/named, which needs constrained
  decoding, argument deltas for the six JSON-payload formats, and
  streamed tool calls on the continuous-batching path
- Full grammar and JSON-schema constrained decoding
- MCP tool invocation. Anthropic streaming and tools now ship
- The rest of the OpenAI API surface (see [`API.md`](API.md))
- Docker images (CPU, Metal and CUDA variants)
- Throughput measurement for concurrent continuous-batching requests
- Full KV layer offload, multi-GPU, tensor parallel, PD disaggregation

**KV cache and memory**

- The `turbo3` dtype, and Metal WHT on the CTK path
- Act on the residency plan `inspect-plan` produces: stream cold
  experts, bound the KV budget, report what a host really fits
- Hybrid CPU/GPU expert placement for MoE, the main lever for running a
  larger model or a higher quant on unchanged hardware

**Wiring `ferrox-edge`**

The ported FreeToken serving policy (`crates/ferrox-edge`, see
[`FEATURES.md`](FEATURES.md)) is complete and tested. Roughly half of it
now drives something: the two output parsers, the withhold rule, the
effort/thinking probe, the request ring, the batcher's status and pool
accounting, the two maintenance endpoints, the DeepSeek-V4 KV tier
sizing, and `radix`, which shares KV pages between prompts on the
paged-KV path. `FEATURES.md` has the per-module split.

Closed since the last pass: `ferrox bench-bw` measures this host's
CPU-MoE bandwidth and writes the profile `qstar` reads, so a deployment
no longer has to take the unbenchmarked one-fetch-per-step default
(the PCIe half still needs a CUDA benchmark host). `POST
/v1/cache/rebuild` moves VRAM between the expert cache and KV without a
restart, validated by `ferrox-edge::pool` and rolled back by
`ferrox-edge::rebuild`.

Still waiting on a consumer:

- Drive expert residency from `ferrox-edge::expert_cache` and the `q*`
  split, which needs the other half FreeToken has and ferrox does not:
  a *persistent* GPU expert cache (`ferrox-moe::run_expert_placed`
  re-uploads every weight matrix per call) and a CPU MoE path that can
  run concurrently with a device copy
- Make paged KV, and the radix cache riding on it, correct on Metal and
  CUDA. Today the paged attention path returns wrong tokens on a GPU
  backend, so the server stops rather than serving them. Lift that once
  the paged path passes `ferrox verify --backend metal` with prefill
  covered
- Give the radix cache an aggregate hit rate on `/v1/stats` and
  `/metrics`, and an eviction budget. `RadixCache::evict` is written and
  tested with nothing calling it, so back pressure today is the page
  store running out
- Make prefix reuse work on the continuous-batching path. Paged KV and
  continuous batching are separate switches and the sharing only
  happens under the first
- Size the pools with `ferrox-edge::pool` at load, not only when a
  rebuild asks for a new geometry
- Find consumers for `placement`, `residency`, `cache_manager`,
  `anchor`, `window_pool`, `state_pool` and `supervisor`

A full recursive re-read of the reference (six readers over its 435
files, checked against every ferrox crate rather than against the port's
own scope) found 34 further items, now tracked individually in the plan
below. One of them is a correctness bug in shipped code rather than an
omission: `route_top_k_grouped` implements "k from every group" where
the DeepSeek-V3/GLM rule scores each group by the sum of its top-2
biased scores and then runs one global top-k. That one is first.

Staged plan, including what the port left behind entirely (semantic
anchor checkpoints, the cache manager, the window slide) and what a
CUDA-side parity would actually cost:
[`plans/freetoken-parity.md`](plans/freetoken-parity.md).

**Engineering practice worth taking from llama.cpp**

- Something equivalent to `test-backend-ops`: every kernel checked
  against a CPU reference across shapes and quant kinds, so no backend
  gets merged on the strength of running fast alone.
