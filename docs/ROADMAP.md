# Roadmap

Goal: match or beat [llama.cpp](https://github.com/ggerganov/llama.cpp)
tok/s on the same host, backend, and GGUF. Current speed numbers:
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

What ships today: [`FEATURES.md`](FEATURES.md) · [`MODELS.md`](MODELS.md).

## Planned

**Performance**

- Close the Metal MoE gap (`mul_mm_id` prefill, tighter `mul_mv_id`, multi-CB)
- Improve Metal prefill `prompt_per_second` vs llama.cpp
- CUDA fair-chat pins on a GPU host

**Models**

- Gemma-4 dedicated engine
- MLA MoE-after-dense; HybridEngine + Qwen3.5
- Llama 4 / MiniMax engines
- Vision (projector + generate)
- Real GLM-5.2 / DeepSeek V4 / full Kimi e2e
- MTP draft heads

**Serving**

- Full grammar / JSON schema constrained decoding
- MCP tool invocation; Anthropic streaming + tools
- Continuous-batching multi-request throughput pin
- Full KV layer offload; multi-GPU / tensor parallel / PD disaggregation

**KV cache**

- `turbo3` dtype; Metal WHT on the CTK path
