# Roadmap

Goal: match or beat [llama.cpp](https://github.com/ggerganov/llama.cpp)
tok/s on the same host, backend, and GGUF. Current speed numbers:
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

What ships today: [`FEATURES.md`](FEATURES.md) · [`MODELS.md`](MODELS.md).

## Planned

**Performance**

All figures below are engine numbers (`ferrox bench` vs `llama-bench`,
each engine at its own default thread count), from
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md). Gap = llama/ferrox.

*Ordered by size of gap × breadth of impact.*

1. **Prefill — the largest gap in the project.** `pp512`: CPU 3.0–8.2×,
   Metal 13.7–**98.6×** (SmolLM2 metal, 123 vs 12134 tok/s).
   - Metal: a real simdgroup `mul_mm`. Today's Q4_K one is a stub that
     returns `Err`, so every dense prefill falls back to N× matvec. This
     single kernel is worth more than everything else on this list.
   - CPU: a Q4_K batch GEMM (`ggml_gemm_q4_K_8x8_q8_K`). Q8_0 has one
     (`gemm_q8_0x4_group`, +8–14%); Q4_K still runs a GEMV per position,
     which is why Phi-3 and Gemma-2 barely moved when Q8_0 did.
   - MoE prefill: group positions by expert (llama `mul_mat_id` map0) so
     MoE layers batch at all — they still run one position at a time.
2. **Metal decode: three shapes lose, the rest lead.** ferrox is ahead or
   tied on 12 of 15 rows. Remaining: **Qwen1.5-MoE 2.75×** (worst row in
   the suite, and only visible since the engine suite started covering
   it), OLMoE 1.29×, Gemma-2-2B 1.14×. MoE decode is the common thread —
   `mul_mv_id` occupancy and the router/expert dispatch, not the dense
   stack.
3. **CPU decode: behind everywhere**, 1.33× (TinyLlama) to 2.60×
   (SmolLM2). Two independent causes, both measured:
   - Parallel scaling ceilings at ~1.9× total (43.8 → 83.3 → 83.5 tok/s
     at 1/4/8 threads) where llama keeps scaling. Fix is llama's shape:
     a persistent spin-barrier pool (`ggml_barrier`) instead of a rayon
     fork-join per matvec (~200/token on SmolLM2).
   - Per-thread GEMV throughput is ~2× off llama at equal thread count.
4. Deferred MoE residual (dense-style) — correct tokens then land.
5. Serving pins: re-measure everything now that the harness no longer
   forces `-t`. `RESULTS.md` warns per-pin until this is done.
6. CUDA pins on a GPU host — none in-tree, skipped on darwin.

**Measurement rules** (learned the hard way, see
[`.scratch/NOTES_LLAMA_CPU.md`](../.scratch/NOTES_LLAMA_CPU.md)):

- Never force a thread count on either engine. llama.cpp defaults to
  performance cores and loses 2–4× above them; pinning both to one count
  flatters ferrox rather than making it fair.
- Sequential runs spread ±20% on Host B. Anything claiming less must be
  measured by interleaved A/B, as `gemm_q8_0x4_group` was.
- The serving suite's prompt is ~30 tokens and cannot see prefill. Use
  the engine table for it.

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
