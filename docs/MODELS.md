# Models

What Ferrox can run, and how it compares to llama.cpp on the same host.
Speed ledger: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)
(`ferrox bench` vs `llama-bench`). Suite list:
[`benchmarks/suite.json`](../benchmarks/suite.json). Architecture list:
`ferrox archs` →
[`manifests/architecture_manifest.md`](manifests/architecture_manifest.md).

**Gap** = `llama / ferrox`. Values below 1.0 mean Ferrox is faster.

Suite policy: keep the **current** generation per family (e.g. Llama-3.2, not
3.1; Gemma-3/4, not Gemma-2; Phi-4, not Phi-3). Older GGUFs still load when
the architecture is supported — they are just not in the published bench
ledger. Add a suite entry (and a GGUF under `models/`) to measure a new model.

## Recommended starters

| Model | Notes |
|---|---|
| SmolLM2-135M-Instruct Q8_0 | Tiny; Metal ahead of llama, CPU well behind |
| TinyLlama-1.1B-Chat Q8_0 | Smallest verified smoke |
| Phi-4-mini-Instruct Q4_K_M | **CPU only** since v0.5.0 — partial rotary (`n_rot` 96 of head_dim 128) and LongRoPE `attn_factor` are implemented on CPU but not in the Metal RoPE kernels, so Metal is refused rather than allowed to compute different attention. Its published Metal row predates the fix and was taken on the wrong graph |
| Llama-3.2-3B-Instruct Q4_K_M | Metal flagship in the suite |
| Gemma-4-E2B-IT Q4_K_M | Dedicated engine + `gemma4` BPE |

```bash
./target/release/ferrox -m /path/to/model.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

./target/release/ferrox-server -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8383

./target/release/ferrox chat --url http://127.0.0.1:8383
```

## Verified (Host B)

Gap = `llama / ferrox` from `ferrox bench` vs `llama-bench` (tg128 unless
noted). **Bold** = ferrox faster. Neither engine's thread count is forced.

| Model | Metal decode | CPU decode |
|---|---|---|
| SmolLM2-135M Q8_0 | **0.67×** | 2.44× |
| Qwen2.5-0.5B Q8_0 | **0.70×** | 1.66× |
| Qwen3-0.6B Q8_0 | **0.71×** | 1.63× |
| Gemma-3-1B-IT Q8_0 | **0.88×** | 1.31× |
| Llama-3.2-1B IQ4_XS | **0.94×** | — |
| Llama-3.2-1B Q4_K_M | 1.00× | — |
| TinyLlama-1.1B Q8_0 | **0.85×** | 1.49× |
| Llama-3.2-3B Q4_K_M | **0.96×** | — |
| Phi-4-mini Q4_K_M | — (refused, see above) | 1.22× |
| Mistral-7B-v0.2 Q4_K_M | 1.00× | 1.17× |
| OLMoE-1B-7B Q4_0 | 1.41× | 1.50× |

Numbers drift as receipts refresh — prefer
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md), which is generated
from the receipts, over this hand-written summary.

Prefill is **closed on Metal for dense models** (every dense `pp512` row
is 1.02–1.08×). What is left on `pp512` is CPU across the board, plus
OLMoE (1.11×) and Gemma-3-1B (1.18×) on Metal.

## Other support

| Model / family | Status |
|---|---|
| Yi (text) | Works (GenericGqa, Neox RoPE) — not in suite yet |
| MiroThinker | Works via `qwen3moe` |
| Qwen2-MoE / Qwen1.5-MoE | Loads; not in current suite (OLMoE is the MoE entry) |
| Mixtral | Suite entry; skipped on 32 GiB Host B (`--fit-host`) |
| MLA (`deepseek2` / `mistral4`) | Dense-lead + MoE-after-dense via `MlaEngine` |
| GLM4 / glm4moe | Loads via GLM-5.2 path when tensors present; no suite receipt |
| Gemma-4-E2B | Dedicated `Gemma4Engine` + SPM-style `gemma4` BPE tokenizer + `<|turn>` chat wrap. GGUF: `models/gemma-4-E2B-it-Q4_K_M.gguf` (`unsloth/gemma-4-E2B-it-GGUF`). Suite id `gemma4_e2b_q4km` — Homebrew llama may still lack `gemma4` arch. |
| gpt-oss | **CPU only.** Attention sinks, alternating sliding-window attention, biased router and the `swiglu_oai` clamp, checked against llama.cpp's own reference logits. Metal and the paged-KV decode path both refuse rather than compute different attention |
| Llama 4 / MiniMax | **Refused**, with the reason stated at load: `llama4 MoE + non-GQA attn` and `MiniMax 256-expert sigmoid MoE + MTP` |
| Hybrid GDN / Qwen3.5 | Scaffold only |
| Kimi K3 / GLM-5.2 / DeepSeek V4 | Loaders/primitives; no frontier e2e receipt |
| Vision | mmproj discover + warn; `image_url` rejected |
| MTP / speculative | `--mtp` errors by design. `ferrox speculative` is prompt-lookup only (an n-gram match over the history, no draft model) and runs on **synthetic random weights**, so the hit rate it prints is not representative of a real drafter. Plan for a real one: [`docs/plans/dflash-speculative-decoding.md`](plans/dflash-speculative-decoding.md) |
| Embeddings | `/v1/embeddings` for GGUF Decoder (mean/last pool) |

## Refused checkpoints

Ferrox fails closed. A checkpoint it cannot compute correctly is
refused at load rather than admitted to a path that computes something
else and returns confident, wrong tokens. Three ways that happens:

1. **Unregistered architecture** — not in the capability registry.
2. **Registered but explicitly unsupported** — `llama4`, `minimax-m2`,
   `minimax-m3` refuse with a reason naming the missing graph feature.
   So do the architectures whose *residual topology* differs from the
   `x + attn(norm(x))` then `y + ffn(norm(y))` shape the generic decoder
   computes: `command-r`, `cohere2`, `cohere2moe`, `falcon`, `gptneox`,
   `phi2` and `plamo` feed both branches the same normed input and sum
   once, and `minicpm` scales its embeddings, residuals and logits by
   multipliers llama.cpp applies *even when the GGUF omits every key*.
   None of that is visible in a tensor or (for MiniCPM) in metadata, so
   these are refused by name rather than caught by a gate.
3. **Any checkpoint carrying a tensor this build never reads.** The
   GGUF reader records every tensor name a loader looks up, and the
   load fails on leftovers:
   *"checkpoint carries N tensor(s) this build never reads, so its
   graph is not the one this build computes."*
   This is what catches a missing graph term by construction rather
   than by enumeration — `attn_sinks` and `exp_probs_b` were both
   found this way. Tensor prefixes for parts ferrox does not claim to
   run (`mm.`, `v.`, `mmproj.`, `resampler.`, `audio.`) are ignored.
4. **Any checkpoint declaring a scalar multiplier this build does not
   apply.** The tensor gate above cannot see these — they are hparams,
   not weights — so a Granite / MiniCPM / Command-R checkpoint would
   otherwise load cleanly and compute a differently-scaled graph than it
   was trained as. `{arch}.logit_scale`, `{arch}.residual_scale`,
   `{arch}.embedding_scale` and `{arch}.attention.scale` are refused by
   name unless they hold their no-op value. **This is a refusal, not an
   implementation**: `residual_scale` multiplies both branch outputs
   before every residual add, which in ferrox means every CPU path plus
   the fused Metal kernels that fold the residual in, and half of that
   would be exactly the silent divergence the gate exists to stop.

`FERROX_ALLOW_UNKNOWN_TENSORS=1` loads anyway and accepts wrong output.
It exists for debugging; it is not a workaround.

## Quantization support

Parsed and executable on CPU: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`,
`Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`,
`IQ4_NL`, `IQ4_XS`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S`, `MXFP4`.

Two caveats that matter in practice:

- **The IQ codebook formats and MXFP4 are CPU-only** (scalar + AVX2, no
  NEON and no GPU kernels). They load and produce correct output; they
  do not go fast, and they do not run on Metal or CUDA.
- **`I32` is recognized and sized but has no execution path.** A
  checkpoint that needs it is refused, not silently skipped.

`IQ2_XS`, `IQ2_S`, `IQ3_S` and `IQ1_M` were validated bit-exact against
llama.cpp's own `dequantize_row_*` by linking `ggml-quants.c`, not by
re-reading the spec. They have not been validated end to end on a
published `UD-*` checkpoint.

## Backends

| Backend | What it covers |
|---|---|
| CPU | Dense + MoE; `FERROX_CPU_INT_DOT=1` (Q4_Kx8 / Q8_0x4 / Q5·Q6 int-dot) on suite runs |
| Metal | Dense + MoE + FA-vec; fused MoE encode groups; `mul_mm_id` prefill; quantized KV |
| CUDA | Matvec + resident weights + FFN fuse |

Capabilities overview: [`FEATURES.md`](FEATURES.md).
Planned work: [`ROADMAP.md`](ROADMAP.md).
