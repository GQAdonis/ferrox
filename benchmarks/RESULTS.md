# Results vs llama.cpp

Host B = Apple M2 Pro (10 cores). Greedy chat, warm, then `max_tokens=512` × 3 reps (median ± stddev) unless noted. Prefer **predicted** tok/s.

Suite: [`suite.json`](suite.json). Runner: [`run_suite.py`](run_suite.py). Pins: [`receipts/pins/`](receipts/pins/). **This file is generated** by [`render_results.py`](render_results.py) — do not hand-edit headlines.

**Gap** = `llama_pred / ferrox_pred` (&lt;1 ferrox faster; &gt;1 ferrox slower). **Winner** = faster engine on predicted tok/s (near-parity within ~5%).

**North star:** ≥ llama.cpp same host/GGUF/backend.
**8B Metal pin:** **28.17** vs llama **25.86** pred (🟢 **~0.92×**) — [`pins/llama31_8b_q4km_metal.json`](receipts/pins/llama31_8b_q4km_metal.json).

One pin per `(model_id, backend)`. Re-run overwrites the pin. Gap only when both engines succeed.

**Gap colors (GitHub-safe):** 🟢 ferrox better; ⚪ near-parity (within ~5%); 🔴 ferrox meaningfully slower.

Keep off (regressions): legacy GQA NSG=4, sequential GREEDY argmax, float4 elem, early Multi-CB. `FERROX_METAL_FA_VEC=0` → ~25.5 pred.

## Headlines

| Model | Backend | ferrox pred (tok/s) | llama pred (tok/s) | Gap | Winner | Status | Pin |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | metal | **114.63** ±0.5 | **110.26** ±0.5 | ⚪ **~0.96×** | ⚪ parity | ok | [`tinyllama_q8_metal`](receipts/pins/tinyllama_q8_metal.json) |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | cpu | **40.83** ±0.3 | **46.93** ±0.7 (−ngl 0) | 🔴 **~1.15×** | 🔴 **llama** | ok | [`tinyllama_q8_cpu`](receipts/pins/tinyllama_q8_cpu.json) |
| Llama-3.2-1B-Instruct Q4_K_M | metal | **137.88** ±5.0 | **136.99** ±0.4 | ⚪ **1.00×** | ⚪ parity | ok | [`llama32_1b_q4km_metal`](receipts/pins/llama32_1b_q4km_metal.json) |
| OLMoE-1B-7B-0924 Q4_0 | cpu | **23.44** ±0.1 | **32.89** ±5.0 (−ngl 0) | 🔴 **~1.40×** | 🔴 **llama** | ok | [`olmoe_q4_cpu`](receipts/pins/olmoe_q4_cpu.json) |
| OLMoE-1B-7B-0924 Q4_0 | metal | **89.92** ±1.6 | **143.41** ±1.1 | 🔴 **~1.59×** | 🔴 **llama** | ok | [`olmoe_q4_metal`](receipts/pins/olmoe_q4_metal.json) |
| OLMoE-1B-7B-0924 Q4_0 | cuda | — | — | — | — | no pin | — |
| Llama-3.1-8B-Instruct Q4_K_M | metal | **28.17** ±0.8 | **25.86** ±1.2 | 🟢 **~0.92×** | 🟢 **ferrox** | ok | [`llama31_8b_q4km_metal`](receipts/pins/llama31_8b_q4km_metal.json) |
| Llama-3.1-8B-Instruct Q4_K_M | cuda | — | — | — | — | no pin | — |
| Llama-3.2-3B-Instruct Q4_K_M | metal | **62.74** ±1.5 | **61.03** ±0.9 | ⚪ **~0.97×** | ⚪ parity | ok | [`llama32_3b_q4km_metal`](receipts/pins/llama32_3b_q4km_metal.json) |
| Qwen1.5-MoE-A2.7B Q4_K_M | cpu | — | — | — | — | no pin | — |
| Qwen1.5-MoE-A2.7B Q4_K_M | metal | — | — | — | — | no pin | — |
| Mistral-7B-Instruct-v0.2 Q4_K_M | cpu | **8.18** ±0.1 | **9.35** ±0.4 (−ngl 0) | 🔴 **~1.14×** | 🔴 **llama** | ok | [`mistral_7b_q4km_cpu`](receipts/pins/mistral_7b_q4km_cpu.json) |
| Mistral-7B-Instruct-v0.2 Q4_K_M | metal | **27.76** ±1.4 | **26.41** ±1.3 | ⚪ **~0.95×** | ⚪ parity | ok | [`mistral_7b_q4km_metal`](receipts/pins/mistral_7b_q4km_metal.json) |
| Mixtral-8x7B-Instruct Q4_K_M | cpu | — | — | — | — | no pin | — |
| Llama-3.2-1B-Instruct IQ4_XS | metal | **146.29** ±2.4 | **138.21** ±3.3 | 🟢 **~0.94×** | 🟢 **ferrox** | ok | [`iq4_xs_metal`](receipts/pins/iq4_xs_metal.json) |
| Gemma-2-2B-IT Q4_K_M | metal | **57.75** ±1.0 | **63.59** ±0.2 | 🔴 **~1.10×** | 🔴 **llama** | ok | [`gemma2_2b_q4km_metal`](receipts/pins/gemma2_2b_q4km_metal.json) |
| Gemma-2-2B-IT Q4_K_M | cpu | **15.54** ±0.0 | **20.70** ±0.2 (−ngl 0) | 🔴 **~1.33×** | 🔴 **llama** | ok | [`gemma2_2b_q4km_cpu`](receipts/pins/gemma2_2b_q4km_cpu.json) |
| SmolLM2-135M-Instruct Q8_0 | metal | **282.29** ±0.8 | **217.91** ±4.4 | 🟢 **~0.77×** | 🟢 **ferrox** | ok | [`smollm2_135m_q8_metal`](receipts/pins/smollm2_135m_q8_metal.json) |
| SmolLM2-135M-Instruct Q8_0 | cpu | **70.17** ±1.0 | **146.99** ±11.3 (−ngl 0) | 🔴 **~2.09×** | 🔴 **llama** | ok | [`smollm2_135m_q8_cpu`](receipts/pins/smollm2_135m_q8_cpu.json) |
| Qwen2.5-0.5B-Instruct Q8_0 | metal | **185.66** ±1.1 | **116.03** ±0.4 | 🟢 **~0.62×** | 🟢 **ferrox** | ok | [`qwen25_05b_q8_metal`](receipts/pins/qwen25_05b_q8_metal.json) |
| Qwen2.5-0.5B-Instruct Q8_0 | cpu | **64.06** ±0.6 | **94.04** ±6.4 (−ngl 0) | 🔴 **~1.47×** | 🔴 **llama** | ok | [`qwen25_05b_q8_cpu`](receipts/pins/qwen25_05b_q8_cpu.json) |
| Qwen3-0.6B Q8_0 | metal | **131.50** ±0.9 | **107.06** ±1.9 | 🟢 **~0.81×** | 🟢 **ferrox** | ok | [`qwen3_06b_q8_metal`](receipts/pins/qwen3_06b_q8_metal.json) |
| Qwen3-0.6B Q8_0 | cpu | **34.35** ±0.1 | **61.64** ±3.2 (−ngl 0) | 🔴 **~1.79×** | 🔴 **llama** | ok | [`qwen3_06b_q8_cpu`](receipts/pins/qwen3_06b_q8_cpu.json) |
| Gemma-3-1B-IT Q8_0 | metal | **91.22** ±1.0 | **74.43** ±0.6 | 🟢 **~0.82×** | 🟢 **ferrox** | ok | [`gemma3_1b_q8_metal`](receipts/pins/gemma3_1b_q8_metal.json) |
| Gemma-3-1B-IT Q8_0 | cpu | **40.26** ±0.3 | **49.15** ±0.6 (−ngl 0) | 🔴 **~1.22×** | 🔴 **llama** | ok | [`gemma3_1b_q8_cpu`](receipts/pins/gemma3_1b_q8_cpu.json) |
| Phi-3-mini-4k-Instruct Q4 | metal | **48.19** ±3.8 | **44.56** ±1.4 | 🟢 **~0.92×** | 🟢 **ferrox** | ok | [`phi3_mini_q4_metal`](receipts/pins/phi3_mini_q4_metal.json) |
| Phi-3-mini-4k-Instruct Q4 | cpu | **8.91** ±0.2 | **17.89** ±0.2 (−ngl 0) | 🔴 **~2.01×** | 🔴 **llama** | ok | [`phi3_mini_q4_cpu`](receipts/pins/phi3_mini_q4_cpu.json) |
| Gemma-4-E2B-IT Q4_K_M | metal | refused | — | — | — | refuse | [`gemma4_e2b_q4km_metal`](receipts/pins/gemma4_e2b_q4km_metal.json) |
| Gemma-4-E2B-IT Q4_K_M | cpu | refused | — | — | — | refuse | [`gemma4_e2b_q4km_cpu`](receipts/pins/gemma4_e2b_q4km_cpu.json) |
| Phi-4-mini-Instruct Q4_K_M | metal | **49.54** ±0.3 | **49.96** ±0.2 | ⚪ **1.00×** | ⚪ parity | ok | [`phi4_mini_q4km_metal`](receipts/pins/phi4_mini_q4km_metal.json) |
| Phi-4-mini-Instruct Q4_K_M | cpu | **11.98** ±0.6 | **19.64** ±1.4 (−ngl 0) | 🔴 **~1.64×** | 🔴 **llama** | ok | [`phi4_mini_q4km_cpu`](receipts/pins/phi4_mini_q4km_cpu.json) |


## CLI completion (`llama-completion` vs `ferrox run`)

One-shot `-p … -n N --ignore-eos -c 4096` with the **same capitals prompt + chat template** as fair-chat server (ferrox wraps via GGUF template; llama `-cnv --jinja`). Fresh process per rep, interleaved (llama then ferrox each rep). Requires `llama-completion` (not `llama-cli`). Engines' own stderr timings; **pred** tok/s excludes model load. **startup** = wall − decode (comparable process overhead); falls back to engine-reported load if startup missing. **Startup gap** = `ferrox / llama` (&lt;1 ferrox better).
Pins that used `llama-cli` or rejected options are omitted.

| Model | Backend | ferrox pred | llama pred | Gap | ferrox startup (s) | llama startup (s) | Startup gap | Pin |
|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | metal | **117.91** ±1.7 | **109.39** ±0.8 | 🟢 **~0.93×** | **0.827** ±0.020 | **0.448** ±0.048 | 🔴 **~1.85×** | [`tinyllama_q8_metal_cli`](receipts/pins/tinyllama_q8_metal_cli.json) |
| Llama-3.2-1B-Instruct Q4_K_M | metal | **142.16** ±2.0 | **122.83** ±5.8 | 🟢 **~0.86×** | **0.980** ±0.112 | **1.361** ±0.127 | 🟢 **~0.72×** | [`llama32_1b_q4km_metal_cli`](receipts/pins/llama32_1b_q4km_metal_cli.json) |
| OLMoE-1B-7B-0924 Q4_0 | metal | **84.20** ±0.8 | **140.48** ±2.0 | 🔴 **~1.67×** | **0.619** ±0.026 | **1.069** ±0.554 | 🟢 **~0.58×** | [`olmoe_q4_metal_cli`](receipts/pins/olmoe_q4_metal_cli.json) |
| Llama-3.1-8B-Instruct Q4_K_M | metal | **28.85** ±0.1 | **28.64** ±0.4 | ⚪ **1.00×** | **2.849** ±0.105 | **1.892** ±1.421 | 🔴 **~1.51×** | [`llama31_8b_q4km_metal_cli`](receipts/pins/llama31_8b_q4km_metal_cli.json) |
| Llama-3.2-3B-Instruct Q4_K_M | metal | **58.82** ±0.8 | **59.08** ±2.9 | ⚪ **1.00×** | **1.544** ±0.102 | **1.354** ±0.467 | 🔴 **~1.14×** | [`llama32_3b_q4km_metal_cli`](receipts/pins/llama32_3b_q4km_metal_cli.json) |
| Mistral-7B-Instruct-v0.2 Q4_K_M | metal | **29.92** ±0.7 | **29.01** ±0.7 | ⚪ **~0.97×** | **2.546** ±0.067 | **1.131** ±1.254 | 🔴 **~2.25×** | [`mistral_7b_q4km_metal_cli`](receipts/pins/mistral_7b_q4km_metal_cli.json) |
| Llama-3.2-1B-Instruct IQ4_XS | metal | **149.75** ±42.1 | **142.42** ±4.2 | ⚪ **~0.95×** | **1.078** ±0.015 | **1.098** ±0.189 | ⚪ **~0.98×** | [`iq4_xs_metal_cli`](receipts/pins/iq4_xs_metal_cli.json) |
| Gemma-2-2B-IT Q4_K_M | metal | **57.17** ±0.2 | **60.38** ±0.7 | 🔴 **~1.06×** | **2.069** ±0.500 | **1.194** ±0.087 | 🔴 **~1.73×** | [`gemma2_2b_q4km_metal_cli`](receipts/pins/gemma2_2b_q4km_metal_cli.json) |
| Gemma-2-2B-IT Q4_K_M | cpu | **8.86** ±0.2 | **10.95** ±3.9 | 🔴 **~1.24×** | **0.060** ±0.000 | **0.929** ±0.562 | 🟢 **~0.06×** | [`gemma2_2b_q4km_cpu_cli`](receipts/pins/gemma2_2b_q4km_cpu_cli.json) |
| SmolLM2-135M-Instruct Q8_0 | metal | **80.71** ±1.3 | **103.96** ±5.6 | 🔴 **~1.29×** | **0.748** ±0.022 | **0.505** ±0.130 | 🔴 **~1.48×** | [`smollm2_135m_q8_metal_cli`](receipts/pins/smollm2_135m_q8_metal_cli.json) |
| Qwen2.5-0.5B-Instruct Q8_0 | metal | **184.21** ±1.1 | **117.42** ±35.1 | 🟢 **~0.64×** | **0.909** ±0.036 | **0.796** ±0.228 | 🔴 **~1.14×** | [`qwen25_05b_q8_metal_cli`](receipts/pins/qwen25_05b_q8_metal_cli.json) |
| Qwen3-0.6B Q8_0 | metal | **131.50** ±0.3 | **107.37** ±2.0 | 🟢 **~0.82×** | **0.948** ±0.090 | **1.029** ±0.273 | 🟢 **~0.92×** | [`qwen3_06b_q8_metal_cli`](receipts/pins/qwen3_06b_q8_metal_cli.json) |
| Gemma-3-1B-IT Q8_0 | metal | **89.14** ±0.3 | **72.35** ±0.4 | 🟢 **~0.81×** | **1.828** ±0.064 | **1.125** ±0.263 | 🔴 **~1.62×** | [`gemma3_1b_q8_metal_cli`](receipts/pins/gemma3_1b_q8_metal_cli.json) |
| Phi-3-mini-4k-Instruct Q4 | metal | **47.91** ±0.2 | **46.27** ±1.0 | ⚪ **~0.97×** | **1.740** ±0.052 | **0.933** ±0.437 | 🔴 **~1.86×** | [`phi3_mini_q4_metal_cli`](receipts/pins/phi3_mini_q4_metal_cli.json) |
| Phi-4-mini-Instruct Q4_K_M | metal | **48.68** ±0.5 | **50.72** ±1.7 | ⚪ **~1.04×** | **1.564** ±0.345 | **1.543** ±0.238 | ⚪ **1.00×** | [`phi4_mini_q4km_metal_cli`](receipts/pins/phi4_mini_q4km_metal_cli.json) |

## Open

1. Metal fair-chat 8B is ~parity (~0.97× quiet pin); keep watching `prompt_per_second` vs llama after FA-vec prefill.
2. CUDA — re-measure on comparable CUDA hardware (no in-tree CUDA pin; skipped on darwin via `--fit-host`).
3. Gemma-4-E2B: both ferrox and Homebrew llama refuse (`gemma4` arch / per-layer+shared-KV+SWA split); suite `expect=refuse`. Phi-4-mini metal pin landed.
4. CB multi-request tok/s receipt.
5. DS4 / GLM / MLA MoE real-checkpoint e2e when feasible.
6. Qwen2-MoE / Mixtral: missing GGUF or `--fit-host` RAM skip on Host B.
7. Suite contention: re-pin outliers alone if full-suite medians disagree with CLI.

Do not invent numbers without a pin.
