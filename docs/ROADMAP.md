# Roadmap

Goal: match or beat [llama.cpp](https://github.com/ggerganov/llama.cpp)
tok/s on the same host, backend, and GGUF. Current speed numbers:
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

What ships today: [`FEATURES.md`](FEATURES.md) · [`MODELS.md`](MODELS.md).

## Planned

**Performance**

- Re-pin OLMoE Metal; chase Gap ≤ ~1.05× (full multi-layer prefill stack CB)
- Deferred MoE residual (dense-style) — correct tokens then land
- Improve Metal dense prefill `prompt_per_second` (true simdgroup `mul_mm`)
- **Benchmark harness: stop forcing `-t 10`.** It handicaps llama.cpp 2–4×
  on Host B, so every CPU row in `RESULTS.md` overstates ferrox. Report
  each engine at its own default plus a thread sweep, and split the
  engine number (`llama-bench`-style, no HTTP) from the serving number.
  See [`.scratch/NOTES_LLAMA_CPU.md`](../.scratch/NOTES_LLAMA_CPU.md)
- CPU: persistent spin-barrier worker pool (llama `ggml_barrier`) to
  replace per-matvec rayon fork-join — ferrox's parallel scaling ceilings
  at ~1.9× where llama keeps going
- CPU: per-thread GEMV throughput is ~2× off llama at equal thread count
- CPU: close remaining SmolLM2/Qwen3/Phi-3 gaps (Q5_Kx8 / deeper Q8)
- CUDA fair-chat pins on a GPU host

**Models**

- Gemma-4 tokenizer (`gemma4`) + fair-chat pin (engine loads today)
- HybridEngine + Qwen3.5
- Llama 4 / MiniMax engines
- Vision (projector + generate)
- Real GLM-5.2 / DeepSeek V4 / full Kimi e2e
- MTP draft heads
- Qwen2-MoE / Mixtral pins when GGUF fits Host B

**Serving**

- Full grammar / JSON schema constrained decoding
- MCP tool invocation; Anthropic streaming + tools
- Continuous-batching multi-request throughput pin
- Full KV layer offload; multi-GPU / tensor parallel / PD disaggregation

**KV cache**

- `turbo3` dtype; Metal WHT on the CTK path
