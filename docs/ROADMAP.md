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
  shape. *Nine wire formats now parse* (`ferrox-edge::parser::tool_call`),
  every call in a response rather than the first, and a reasoning
  model's chain of thought comes back as `reasoning_content`. What is
  left here is `tool_choice: required`/named, which needs constrained
  decoding, and incremental `tool_calls[].index` argument deltas on the
  streaming path
- Full grammar and JSON-schema constrained decoding
- MCP tool invocation, plus Anthropic streaming and tools
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
[`FEATURES.md`](FEATURES.md)) is complete and tested but only its two
output parsers and its withhold rule are driving anything yet. The rest
is groundwork waiting on a consumer:

- Replace `ferrox-models::prefix_cache`'s flat snapshot list with
  `ferrox-edge::radix`, so a thousand requests off one system prompt
  share its nodes instead of cloning its KV, and so prefix reuse works
  on the continuous-batching path (today the two are mutually
  exclusive)
- Drive expert residency from `ferrox-edge::expert_cache` and the `q*`
  split, which needs the other half FreeToken has and ferrox does not:
  a *persistent* GPU expert cache (`ferrox-moe::run_expert_placed`
  re-uploads every weight matrix per call) and a CPU MoE path that can
  run concurrently with a device copy
- `ferrox bench bw` — measure this host's CPU-MoE and PCIe bandwidths so
  `qstar::BandwidthProfile` has real numbers to read, instead of the
  unbenchmarked one-fetch-per-step default
- Size the pools with `ferrox-edge::pool` at load, and expose the live
  re-split (`POST /v1/cache/rebuild`) so VRAM can move between the
  expert cache and KV without a restart

Staged plan, including what the port left behind entirely (semantic
anchor checkpoints, the cache manager, the window slide) and what a
CUDA-side parity would actually cost:
[`plans/freetoken-parity.md`](plans/freetoken-parity.md).

**Engineering practice worth taking from llama.cpp**

- Something equivalent to `test-backend-ops`: every kernel checked
  against a CPU reference across shapes and quant kinds, so no backend
  gets merged on the strength of running fast alone.
- Build the CUDA feature combination in CI, even with no GPU to run it
  on. The last break there was a compile error, not a runtime one.
