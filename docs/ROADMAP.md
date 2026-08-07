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
- **Batched prefill is the biggest gap in the project: ~4–5× on CPU
  after batching the dense FFN, 15–90× on Metal** (`ferrox bench` vs
  `llama-bench`, both at their own default threads). CPU next step is a
  real GEMM with register blocking over the batch dimension
  (llama's `ggml_gemm_q4_K_8x8_q8_K` / `ggml_gemm_q8_0_4x4_q8_0`);
  `apply_batch` still calls a GEMV per position, so today's only win is
  weight-cache reuse. Metal needs the simdgroup `mul_mm`. The fair-chat
  suite could not see any of this: its prompt is ~30 tokens
- MoE prefill: group positions by expert (llama `mul_mat_id` map0) so
  MoE layers can batch too — they still run one position at a time
- **Benchmark harness: stop forcing `-t 10`.** It handicaps llama.cpp 2–4×
  on Host B, so every CPU row in `RESULTS.md` overstates ferrox. Report
  each engine at its own default plus a thread sweep, and split the
  engine number (`ferrox bench`, landed) from the serving number.
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
