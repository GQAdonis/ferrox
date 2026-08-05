# Results vs llama.cpp

Host B = Apple M2 Pro (10 cores). Greedy chat, warm, then `max_tokens=512` × 3 reps (median ± stddev) unless noted. Prefer **predicted** tok/s.

Suite: [`suite.json`](suite.json). Runner: [`run_suite.py`](run_suite.py). Pins: [`receipts/pins/`](receipts/pins/). **This file is generated** by [`render_results.py`](render_results.py) — do not hand-edit headlines.

**Gap** = `llama_pred / ferrox_pred` (&lt;1 ferrox faster; &gt;1 ferrox slower). **Winner** = faster engine on predicted tok/s (near-parity within ~5%).

**North star:** ≥ llama.cpp same host/GGUF/backend.
**8B Metal pin:** **27.98** vs llama **28.73** pred (⚪ **~1.03×**) — [`pins/llama31_8b_q4km_metal.json`](receipts/pins/llama31_8b_q4km_metal.json).

One pin per `(model_id, backend)`. Re-run overwrites the pin. Gap only when both engines succeed.

**Gap colors (GitHub-safe):** 🟢 ferrox better; ⚪ near-parity (within ~5%); 🔴 ferrox meaningfully slower.

Keep off (regressions): legacy GQA NSG=4, sequential GREEDY argmax, float4 elem, early Multi-CB. `FERROX_METAL_FA_VEC=0` → ~25.5 pred.

## Headlines

| Model | Backend | ferrox pred (tok/s) | llama pred (tok/s) | Gap | Winner | Status | Pin |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | metal | **116.69** ±1.4 | **109.85** ±3.0 | 🟢 **~0.94×** | 🟢 **ferrox** | ok | [`tinyllama_q8_metal`](receipts/pins/tinyllama_q8_metal.json) |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | cpu | **44.56** | **38.16** (−ngl 0) | 🟢 **~0.86×** | 🟢 **ferrox** | ok | [`tinyllama_q8_cpu`](receipts/pins/tinyllama_q8_cpu.json) |
| Llama-3.2-1B-Instruct Q4_K_M | metal | **139.39** ±1.6 | **139.94** ±1.5 | ⚪ **1.00×** | ⚪ parity | ok | [`llama32_1b_q4km_metal`](receipts/pins/llama32_1b_q4km_metal.json) |
| OLMoE-1B-7B-0924 Q4_0 | cpu | **20.52** ±0.6 | **13.20** ±4.2 (−ngl 0) | 🟢 **~0.64×** | 🟢 **ferrox** | ok | [`olmoe_q4_cpu`](receipts/pins/olmoe_q4_cpu.json) |
| OLMoE-1B-7B-0924 Q4_0 | metal | **21.65** ±3.5 | **144.43** ±1.7 | 🔴 **~6.67×** | 🔴 **llama** | ok | [`olmoe_q4_metal`](receipts/pins/olmoe_q4_metal.json) |
| OLMoE-1B-7B-0924 Q4_0 | cuda | — | — | — | — | no pin | — |
| Llama-3.1-8B-Instruct Q4_K_M | metal | **27.98** ±1.0 | **28.73** ±1.0 | ⚪ **~1.03×** | ⚪ parity | ok | [`llama31_8b_q4km_metal`](receipts/pins/llama31_8b_q4km_metal.json) |
| Llama-3.1-8B-Instruct Q4_K_M | cuda | — | — | — | — | no pin | — |
| Llama-3.2-3B-Instruct Q4_K_M | metal | **57.02** ±0.2 | **59.43** ±1.5 | ⚪ **~1.04×** | ⚪ parity | ok | [`llama32_3b_q4km_metal`](receipts/pins/llama32_3b_q4km_metal.json) |
| Qwen1.5-MoE-A2.7B Q4_K_M | cpu | — | — | — | — | no pin | — |
| Qwen1.5-MoE-A2.7B Q4_K_M | metal | — | — | — | — | no pin | — |
| Mistral-7B-Instruct-v0.2 Q4_K_M | cpu | **3.10** ±0.5 | **3.19** ±1.2 (−ngl 0) | ⚪ **~1.03×** | ⚪ parity | ok | [`mistral_7b_q4km_cpu`](receipts/pins/mistral_7b_q4km_cpu.json) |
| Mistral-7B-Instruct-v0.2 Q4_K_M | metal | **29.44** ±1.5 | **27.84** ±1.0 | 🟢 **~0.95×** | 🟢 **ferrox** | ok | [`mistral_7b_q4km_metal`](receipts/pins/mistral_7b_q4km_metal.json) |
| Mixtral-8x7B-Instruct Q4_K_M | cpu | — | — | — | — | no pin | — |
| Llama-3.2-1B-Instruct IQ4_XS | metal | **148.43** ±0.3 | **138.25** ±0.8 | 🟢 **~0.93×** | 🟢 **ferrox** | ok | [`iq4_xs_metal`](receipts/pins/iq4_xs_metal.json) |
| Gemma-2-2B-IT Q4_K_M | metal | **58.08** ±0.6 | **58.85** ±4.4 | ⚪ **1.00×** | ⚪ parity | ok | [`gemma2_2b_q4km_metal`](receipts/pins/gemma2_2b_q4km_metal.json) |
| Gemma-2-2B-IT Q4_K_M | cpu | **8.40** ±0.4 | **12.94** ±1.2 (−ngl 0) | 🔴 **~1.54×** | 🔴 **llama** | ok | [`gemma2_2b_q4km_cpu`](receipts/pins/gemma2_2b_q4km_cpu.json) |
| SmolLM2-135M-Instruct Q8_0 | metal | **283.66** ±9.7 | **193.48** ±4.9 | 🟢 **~0.68×** | 🟢 **ferrox** | ok | [`smollm2_135m_q8_metal`](receipts/pins/smollm2_135m_q8_metal.json) |
| SmolLM2-135M-Instruct Q8_0 | cpu | **55.84** ±0.7 | **43.48** ±5.9 (−ngl 0) | 🟢 **~0.78×** | 🟢 **ferrox** | ok | [`smollm2_135m_q8_cpu`](receipts/pins/smollm2_135m_q8_cpu.json) |
| Qwen2.5-0.5B-Instruct Q8_0 | metal | **192.09** ±7.0 | **123.00** ±5.6 | 🟢 **~0.64×** | 🟢 **ferrox** | ok | [`qwen25_05b_q8_metal`](receipts/pins/qwen25_05b_q8_metal.json) |
| Qwen2.5-0.5B-Instruct Q8_0 | cpu | **50.97** ±0.9 | **44.86** ±6.4 (−ngl 0) | 🟢 **~0.88×** | 🟢 **ferrox** | ok | [`qwen25_05b_q8_cpu`](receipts/pins/qwen25_05b_q8_cpu.json) |
| Qwen3-0.6B Q8_0 | metal | **126.41** ±5.4 | **114.48** ±1.2 | 🟢 **~0.91×** | 🟢 **ferrox** | ok | [`qwen3_06b_q8_metal`](receipts/pins/qwen3_06b_q8_metal.json) |
| Qwen3-0.6B Q8_0 | cpu | **33.38** ±0.9 | **29.82** ±5.7 (−ngl 0) | 🟢 **~0.89×** | 🟢 **ferrox** | ok | [`qwen3_06b_q8_cpu`](receipts/pins/qwen3_06b_q8_cpu.json) |
| Gemma-3-1B-IT Q8_0 | metal | **87.92** ±1.4 | **74.72** ±0.3 | 🟢 **~0.85×** | 🟢 **ferrox** | ok | [`gemma3_1b_q8_metal`](receipts/pins/gemma3_1b_q8_metal.json) |
| Gemma-3-1B-IT Q8_0 | cpu | **31.67** ±3.7 | **19.72** ±1.9 (−ngl 0) | 🟢 **~0.62×** | 🟢 **ferrox** | ok | [`gemma3_1b_q8_cpu`](receipts/pins/gemma3_1b_q8_cpu.json) |
| Phi-3-mini-4k-Instruct Q4 | metal | **46.98** ±2.4 | **48.43** ±0.4 | ⚪ **~1.03×** | ⚪ parity | ok | [`phi3_mini_q4_metal`](receipts/pins/phi3_mini_q4_metal.json) |
| Phi-3-mini-4k-Instruct Q4 | cpu | **5.78** ±0.1 | **9.48** ±1.0 (−ngl 0) | 🔴 **~1.64×** | 🔴 **llama** | ok | [`phi3_mini_q4_cpu`](receipts/pins/phi3_mini_q4_cpu.json) |


## CLI completion (`llama-completion` vs `ferrox run`)

One-shot `-p … -n N --ignore-eos -c 4096`, fresh process per rep, interleaved (llama then ferrox each rep). Requires `llama-completion` (not `llama-cli`). Engines' own stderr timings; **pred** tok/s excludes model load. **startup** = wall − decode (comparable process overhead); falls back to engine-reported load if startup missing. **Startup gap** = `ferrox / llama` (&lt;1 ferrox better).
Pins that used `llama-cli` or rejected options are omitted.

| Model | Backend | ferrox pred | llama pred | Gap | ferrox startup (s) | llama startup (s) | Startup gap | Pin |
|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | metal | **112.78** ±3.5 | **113.36** ±4.3 | ⚪ **1.00×** | **0.050** ±0.006 | **0.142** ±0.289 | 🟢 **~0.35×** | [`tinyllama_q8_metal_cli`](receipts/pins/tinyllama_q8_metal_cli.json) |
| OLMoE-1B-7B-0924 Q4_0 | metal | **38.28** ±12.6 | **153.69** ±0.9 | 🔴 **~4.01×** | **0.568** ±0.022 | **0.807** ±0.190 | 🟢 **~0.70×** | [`olmoe_q4_metal_cli`](receipts/pins/olmoe_q4_metal_cli.json) |
| Llama-3.1-8B-Instruct Q4_K_M | metal | **24.99** ±0.1 | **24.69** ±0.5 | ⚪ **1.00×** | **0.120** ±0.040 | **4.718** ±5.418 | 🟢 **~0.03×** | [`llama31_8b_q4km_metal_cli`](receipts/pins/llama31_8b_q4km_metal_cli.json) |
| Mistral-7B-Instruct-v0.2 Q4_K_M | metal | **26.78** ±0.1 | **29.90** ±1.3 | 🔴 **~1.12×** | **0.080** ±0.038 | **1.115** ±1.242 | 🟢 **~0.07×** | [`mistral_7b_q4km_metal_cli`](receipts/pins/mistral_7b_q4km_metal_cli.json) |
| Gemma-2-2B-IT Q4_K_M | metal | **35.71** ±0.1 | **57.86** ±0.5 | 🔴 **~1.62×** | **0.100** ±0.010 | **0.661** ±0.617 | 🟢 **~0.15×** | [`gemma2_2b_q4km_metal_cli`](receipts/pins/gemma2_2b_q4km_metal_cli.json) |
| Gemma-2-2B-IT Q4_K_M | cpu | **8.86** ±0.2 | **10.95** ±3.9 | 🔴 **~1.24×** | **0.060** ±0.000 | **0.929** ±0.562 | 🟢 **~0.06×** | [`gemma2_2b_q4km_cpu_cli`](receipts/pins/gemma2_2b_q4km_cpu_cli.json) |

## Open

1. Metal prefill ≪ llama on large models; FA-vec covers d=64/96/128/256 (Phi-3 / Gemma-3 decode path).
2. CUDA — re-measure on comparable CUDA hardware (no in-tree CUDA pin).
3. Gemma-2 arch support (attn softcap + pin; suite currently refuse).
4. CB multi-request tok/s receipt.
5. DS4 / GLM real e2e when feasible.
6. Qwen2-MoE / Mistral / Mixtral oracle receipts.

Do not invent numbers without a pin.
