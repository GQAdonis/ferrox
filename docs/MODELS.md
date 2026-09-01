# Models

What Ferrox runs, and how it compares to llama.cpp on the same host.
Speed table: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)
(`ferrox bench` vs `llama-bench`). Suite list:
[`benchmarks/suite.json`](../benchmarks/suite.json). Architecture list:
`ferrox archs` →
[`manifests/architecture_manifest.md`](manifests/architecture_manifest.md).

**Gap** = `llama / ferrox`. Values below 1.0 mean Ferrox is faster.

Suite policy: keep the **current** generation per family. Llama-3.2, not
3.1. Gemma-3/4, not Gemma-2. Phi-4, not Phi-3. Older GGUFs still load
when the architecture is supported, they are simply not measured in the
published table. To measure a new model, add a suite entry and put the
GGUF under `models/`.

## Recommended starters

| Model | Notes |
|---|---|
| SmolLM2-135M-Instruct Q8_0 | Tiny. Metal ahead of llama, CPU well behind |
| TinyLlama-1.1B-Chat Q8_0 | Smallest verified smoke |
| Phi-4-mini-Instruct Q4_K_M | Metal works again. The RoPE kernels now carry `n_rot` (96 of head_dim 128) and LongRoPE's `attn_factor`, and `verify --backend metal` returns identical CPU and Metal token ids with prefill covered. The Metal rows in `benchmarks/RESULTS.md` predate that fix and were taken on the wrong graph. **Do not quote them until Phi-4 is measured again.** |
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
| Llama-3.2-1B IQ4_XS | **0.94×** |, |
| Llama-3.2-1B Q4_K_M | 1.00× |, |
| TinyLlama-1.1B Q8_0 | **0.85×** | 1.49× |
| Llama-3.2-3B Q4_K_M | **0.96×** |, |
| Phi-4-mini Q4_K_M |, (owed, see above) | 1.22× |
| Mistral-7B-v0.2 Q4_K_M | 1.00× | 1.17× |
| OLMoE-1B-7B Q4_0 | 1.41× | 1.50× |

These numbers drift as runs are refreshed.
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) is generated
straight from the raw timing files, so trust it over this hand-written
summary.

Prefill is **closed on Metal for dense models** (every dense `pp512` row
is 1.02–1.08×). What is left on `pp512` is CPU across the board, plus
OLMoE (1.11×) and Gemma-3-1B (1.18×) on Metal.

## Other support

| Model / family | Status |
|---|---|
| Yi (text) | Works (GenericGqa, Neox RoPE), not in suite yet |
| MiroThinker | Works via `qwen3moe` |
| Qwen2-MoE / Qwen1.5-MoE | Loads. Not in the current suite (OLMoE is the MoE entry) |
| Mixtral | In the suite, skipped on 32 GiB Host B (`--fit-host`) |
| MLA (`deepseek2` / `mistral4`) | Dense-lead + MoE-after-dense via `MlaEngine` |
| GLM4 / glm4moe | Loads via the GLM-5.2 path when the tensors are there. Never measured in the suite |
| Gemma-4-E2B | Dedicated `Gemma4Engine` + SPM-style `gemma4` BPE tokenizer + `<|turn>` chat wrap. GGUF: `models/gemma-4-E2B-it-Q4_K_M.gguf` (`unsloth/gemma-4-E2B-it-GGUF`). Suite id `gemma4_e2b_q4km`, Homebrew llama may still lack `gemma4` arch. |
| gpt-oss | **CPU only.** Attention sinks, alternating sliding-window attention, biased router and the `swiglu_oai` clamp, checked against llama.cpp's own reference logits. Metal stops with an error, because no Metal kernel implements attention sinks. The paged-KV decode path runs it: all three attention arms are bit-identical to their contiguous twins |
| Llama 4 / MiniMax | **Will not load**, with the reason stated: `llama4 MoE + non-GQA attn` and `MiniMax 256-expert sigmoid MoE + MTP` |
| Hybrid GDN / Qwen3.5 | Scaffold only |
| Kimi K3 / GLM-5.2 / DeepSeek V4 | Loaders and primitives only. Nothing has been run end to end on a real checkpoint |
| Vision | Finds an mmproj file and warns about it. An `image_url` in a request returns an error |
| MTP / speculative | `--mtp` errors by design. `ferrox speculative` is prompt-lookup only (an n-gram match over the history, no draft model) and runs on **synthetic random weights**, so the hit rate it prints is not representative of a real drafter. Plan for a real one: [`docs/plans/dflash-speculative-decoding.md`](plans/dflash-speculative-decoding.md) |
| Embeddings | `/v1/embeddings` for GGUF Decoder (mean/last pool) |

## When a model will not load

Some checkpoints stop with an error instead of running. That is
deliberate. The alternative is worse: a model whose graph Ferrox only
partly implements will load, run fast, and return fluent text computed
by the wrong maths, and nothing in the output tells you. An error you
can read beats output you cannot trust.

The error always names the reason. Six things cause it:

1. **Ferrox does not know the architecture.** It is not in the
   capability registry.

2. **Ferrox knows it and has not implemented it.** `llama4`,
   `minimax-m2` and `minimax-m3` stop with the missing feature named.
   So do architectures whose residual wiring differs from the
   `x + attn(norm(x))` then `y + ffn(norm(y))` shape the generic decoder
   computes: `command-r`, `cohere2`, `cohere2moe`, `falcon`, `gptneox`,
   `phi2` and `plamo` feed both branches the same normed input and sum
   once. `minicpm` scales its embeddings, residuals and logits by
   multipliers llama.cpp applies even when the GGUF omits every key.
   None of that shows up in a tensor or, for MiniCPM, in metadata, so
   these are listed by name rather than detected.

3. **The file contains weights Ferrox never reads.** The GGUF reader
   records every tensor name a loader looks up and stops if any are left
   over: *"checkpoint carries N tensor(s) this build never reads, so its
   graph is not the one this build computes."* Unread weights mean the
   file describes a model Ferrox is not computing. This catches missing
   graph features automatically rather than one at a time, which is how
   `attn_sinks` and `exp_probs_b` were both found. Tensors for parts
   Ferrox does not claim to run (`mm.`, `v.`, `mmproj.`, `resampler.`,
   `audio.`) are ignored.

4. **The file declares a scale factor Ferrox does not apply.** These are
   hyperparameters rather than weights, so the check above cannot see
   them, and a Granite, MiniCPM or Command-R checkpoint would otherwise
   load cleanly while computing a differently-scaled graph than it was
   trained as. `{arch}.logit_scale`, `{arch}.residual_scale`,
   `{arch}.embedding_scale` and `{arch}.attention.scale` stop the load
   unless they hold a value that changes nothing. Implementing
   `residual_scale` properly means touching every CPU residual add plus
   the fused Metal kernels that fold the residual in, and getting half
   of that right gives you a model that loads, runs, and returns wrong
   answers with nothing in the output to say so. That is the outcome
   this check exists to prevent.

5. **The architecture encodes position some other way than RoPE.** The
   generic decoder rotates every Q and K head of every layer. `gpt2`
   uses a learned absolute position table instead; `mpt`, `refact`,
   `bloom` and `jais` use ALiBi. All five stop with the reason named.
   This is the least visible failure of the five: `bloom` and `refact`
   hardcode their ALiBi slope in llama.cpp's own loader and carry no
   GGUF key at all, and `mpt` leaves no unread tensor behind, so neither
   check 3 nor check 4 could ever see them. `baichuan` is the same
   problem conditionally: the 7B rotates, the 13B uses ALiBi, and
   llama.cpp tells them apart by layer count alone, so a 40-layer
   Baichuan is refused and a 32-layer one is not.

6. **Nobody has ever verified this architecture against llama.cpp.**
   The shared generic-GQA decoder is a *guess*: it assumes plain GQA
   because nothing said otherwise, and that guess was already wrong for
   the five architectures in cause 5. So the generic path is opt-in.
   An architecture reaches it only if there is a benchmark row, a pinned
   logit comparison against real `libllama`, or a fixture; 11 do today
   (`llama`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `olmoe`,
   `gemma2`, `gemma3`, `phi3`, `gpt-oss`, `dots1`). The other **47**
   stop with `UnauditedArchitecture`. `FERROX_ALLOW_UNAUDITED_ARCH=1`
   runs one anyway; compare the output against llama.cpp yourself
   before you trust it.

### What "unaudited" costs you, per architecture

"Unaudited" is not one thing. Some of the 47 are one fixture away from
running and some need an attention implementation, so the refusal says
which, with the `llama.cpp/src/models/*.cpp` line that decides it:

| Class | Means |
|---|---|
| `FIXTURE-AWAY` | Ferrox already computes this graph. What is missing is evidence. |
| `ONE MATCH ARM` | One small, named piece: an activation, a norm slot, a routing flag, an ordering. |
| `NEW CODE` | A different attention or residual structure. Not close. |
| `UNKNOWN` | Reading both trees did not settle it. The message says what would. |

Read on both sides so far, 15 of 47 (`ferrox_models::capability`, pinned
by `crates/ferrox-models/tests/unaudited_triage.rs`):

| Arch | Class | What decides it |
|---|---|---|
| `gemma` | fixture-away | `gemma.cpp:16-33` creates only tensors the generic decoder loads; the embedding scale, GeGLU and attention scale are all implemented for the Gemma family |
| `internlm2` | fixture-away | `internlm2.cpp:25-33` is the plain Llama tensor set; sequential residual, SiLU SwiGLU |
| `exaone` | fixture-away | `exaone.cpp:29-38` likewise, plus the global `rope_freqs.weight` the loader already reads |
| `ernie4_5` | fixture-away | `ernie4-5.cpp:39-67`; its one unslotted tensor, an optional `attn_output.bias` at `:45`, is caught by name by the unread-tensor gate |
| `bailingmoe2` | fixture-away | `bailingmoe2.cpp:47-73`: fused QKV Ferrox splits, per-head QK norm before RoPE, `exp_probs_b`, shared experts, gating read from metadata |
| `seed_oss` | one match arm | `seed-oss.cpp:36-37,113-115`: `post_attention_norm` **is** the pre-FFN norm and there is no `ffn_norm`. gpt-oss's slot, behind an `is_gpt_oss` flag that would be widened |
| `deepseek` | one match arm | `deepseek.cpp:145` passes `norm_w=false` and the converter never writes `expert_weights_norm`, so Ferrox renormalises top-k weights where llama.cpp does not |
| `ernie4_5-moe` | one match arm | `ernie4-5-moe.cpp:64` interleaves MoE layers by `n_moe_layer_step`; `layer_is_dense` implements only a leading-dense prefix |
| `hunyuan-moe` | one match arm | `hunyuan-moe.cpp:93-118` applies QK norm **after** RoPE; Ferrox norms then rotates |
| `olmo2` | new code | `olmo2.cpp:47,52,92,169`: no `attn_norm` and no `ffn_norm` at all. Q/K/V come off the raw residual. The generic decoder requires and applies both |
| `exaone4` | new code | `exaone4.cpp:60-67,118,159`: the same post-norm-only topology |
| `granite`, `granitemoe`, `granite-moe` | new code | `granite.cpp:7-10,188,241-242,301-302`: four multipliers, including a residual scale on every branch output. Cause 4 above already refuses these by name first |
| `phi4` | unknown | `phi4` is not in llama.cpp's `LLM_ARCH_NAMES` at all, so there is no reference graph to diff. A real GGUF spelling it would settle whether it is phi3's fused-QKV graph |

The remaining 32 say so explicitly rather than guessing a class.

`FERROX_ALLOW_UNKNOWN_TENSORS=1` loads the checkpoint anyway and accepts
whatever comes out. Use it while you debug, not to get past the error
and carry on.

## Quantization support

Parsed and executable on CPU: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`,
`Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`,
`IQ4_NL`, `IQ4_XS`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S`, `MXFP4`.

"Executable" is not one speed. What a format actually gets, read off
the kernel tables (`ferrox_quant`'s dispatch functions,
`metal_matvec_kind_name` / `metal_mul_mm_kind_supported` and
`cuda_matvec_kind_supported` in `ferrox-core`):

| Tier | Formats | CPU SIMD | GPU |
|---|---|---|---|
| Full | `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` | AVX2 + NEON, plus the int-dot path (`FERROX_CPU_INT_DOT=1`) | Metal matvec + simdgroup GEMM, CUDA matvec |
| Metal only | `IQ4_XS` | AVX2 + NEON | Metal matvec + simdgroup GEMM |
| CPU-vectorized | `Q4_1`, `Q5_1`, `Q8_1`, `Q2_K`, `Q3_K`, `IQ4_NL`, safetensors two-buffer `MXFP4` | AVX2 + NEON | none |
| Prefill only on GPU | `Q5_0` | AVX2 + NEON | a Metal simdgroup GEMM, but **no matvec**: prefill runs on the GPU and decode falls back to the CPU |
| AVX2 only | `IQ1_S`, `IQ2_XXS`, `IQ3_XXS` | AVX2; **scalar on ARM** | none |
| Scalar only | `IQ2_XS`, `IQ2_S`, `IQ3_S`, `IQ1_M`, GGUF-block `MXFP4` | none | none |

Metal's MoE indexed GEMM (`mul_mm_id`) is narrower still: `Q4_0`,
`Q8_0` and `Q4_K` only.

Three caveats that matter in practice:

- **The IQ tiers split, and the split matters if you are choosing a
  quant.** The bottom two rows load and produce correct output, and they
  are slow. That was deliberate. They were added for coverage, because
  before them those tags could not be decoded at all, which ruled out 5
  of the 16 published Unsloth `UD-*` variants. A vectorized path was
  left out rather than written without a golden vector that could tell
  it apart from the scalar one.
- **On an Apple machine the "AVX2 only" row is the scalar row.**
  `IQ1_S`, `IQ2_XXS` and `IQ3_XXS` have x86 kernels and no NEON ones, so
  on ARM they run at the same speed as the scalar tier below them.
- **`I32`, `TQ1_0`, `TQ2_0`, `NVFP4`, `Q1_0` and `Q2_0` are recognized
  and sized, but nothing executes them.** They parse, `ferrox inspect`
  reports their real footprint, and a checkpoint that needs one stops
  with an error naming the format rather than being quietly skipped or
  silently mis-measured. Ternary (`TQ*`) and the two newest `Q*_0`
  formats are a real gap, not a claim of support.

`IQ2_XS`, `IQ2_S`, `IQ3_S` and `IQ1_M` were validated bit-exact against
llama.cpp's own `dequantize_row_*` by linking `ggml-quants.c`, not by
re-reading the spec. They have not been validated end to end on a
published `UD-*` checkpoint.

## Backends

| Backend | What it covers |
|---|---|
| CPU | Dense and MoE. `FERROX_CPU_INT_DOT=1` (Q4_Kx8 / Q8_0x4 / Q5·Q6 int-dot) on suite runs |
| Metal | Dense, MoE, FA-vec, fused MoE encode groups, `mul_mm_id` prefill, quantized KV |
| CUDA | Matvec + resident weights + FFN fuse |

Every number on this page was taken on CPU or Apple Metal. CUDA compiles
and runs, has no pinned benchmark host, and has no published timings, so
treat a Windows or Linux install as CPU-only in practice.

Paged KV is CPU only as well. On Metal or CUDA the paged attention path
returns wrong tokens rather than failing, so `ferrox-server` stops at
startup when `FERROX_PAGED_KV_BLOCKS` is set beside `-dev metal` or
`-dev cuda`. See [`CONFIG.md`](CONFIG.md).

Capabilities overview: [`FEATURES.md`](FEATURES.md).
Planned work: [`ROADMAP.md`](ROADMAP.md).
