# Results vs llama.cpp

Host B = Apple M2 Pro (10 cores). Greedy chat, warm, then `max_tokens=512` × 3 reps (median ± stddev) unless noted. Prefer **predicted** tok/s.

Suite: [`suite.json`](suite.json). Runner: [`run_suite.py`](run_suite.py). Pins: [`receipts/pins/`](receipts/pins/). **This file is generated** by [`render_results.py`](render_results.py) — do not hand-edit headlines.

**Gap** = `llama_pred / ferrox_pred` (&lt;1 ferrox faster; &gt;1 ferrox slower). **Winner** = faster engine on predicted tok/s (tie within ~1.5%).

**North star:** ≥ llama.cpp same host/GGUF/backend.
**8B Metal pin:** **26.91** vs llama **27.83** pred (~1.03×) — [`pins/llama31_8b_q4km_metal.json`](receipts/pins/llama31_8b_q4km_metal.json).

One pin per `(model_id, backend)`. Re-run overwrites the pin. Gap only when both engines succeed.

Keep off (regressions): legacy GQA NSG=4, sequential GREEDY argmax, float4 elem, early Multi-CB. `FERROX_METAL_FA_VEC=0` → ~25.5 pred.

## Headlines

| Model | Backend | ferrox pred (tok/s) | llama pred (tok/s) | Gap | Winner | Status | Pin |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | metal | **116.69** ±1.4 | **109.85** ±3.0 | ~0.94× | ferrox | ok | [`tinyllama_q8_metal`](receipts/pins/tinyllama_q8_metal.json) |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | cpu | **44.56** | **38.16** (−ngl 0) | ~0.86× | ferrox | ok | [`tinyllama_q8_cpu`](receipts/pins/tinyllama_q8_cpu.json) |
| Llama-3.2-1B-Instruct Q4_K_M | metal | **141.41** ±3.2 | **136.23** ±2.1 | ~0.96× | ferrox | ok | [`llama32_1b_q4km_metal`](receipts/pins/llama32_1b_q4km_metal.json) |
| OLMoE-1B-7B-0924 Q4_0 | cpu | **17.89** | **19.74** (−ngl 0) | ~1.10× | llama | ok | [`olmoe_q4_cpu`](receipts/pins/olmoe_q4_cpu.json) |
| OLMoE-1B-7B-0924 Q4_0 | metal | **10.26** ±0.3 | **152.79** ±1.2 | ~14.89× | llama | ok | [`olmoe_q4_metal`](receipts/pins/olmoe_q4_metal.json) |
| OLMoE-1B-7B-0924 Q4_0 | cuda | — | — | — | — | no pin | — |
| Llama-3.1-8B-Instruct Q4_K_M | metal | **26.91** ±0.8 | **27.83** ±1.0 | ~1.03× | llama | ok | [`llama31_8b_q4km_metal`](receipts/pins/llama31_8b_q4km_metal.json) |
| Llama-3.1-8B-Instruct Q4_K_M | cuda | — | — | — | — | no pin | — |
| Llama-3.2-3B-Instruct Q4_K_M | metal | **57.02** ±0.2 | **59.43** ±1.5 | ~1.04× | llama | ok | [`llama32_3b_q4km_metal`](receipts/pins/llama32_3b_q4km_metal.json) |
| Qwen1.5-MoE-A2.7B Q4_K_M | cpu | — | — | — | — | no pin | — |
| Qwen1.5-MoE-A2.7B Q4_K_M | metal | — | — | — | — | no pin | — |
| Mistral-7B-Instruct-v0.2 Q4_K_M | cpu | — | — | — | — | no pin | — |
| Mistral-7B-Instruct-v0.2 Q4_K_M | metal | **30.72** ±0.2 | **32.05** ±0.2 | ~1.04× | llama | ok | [`mistral_7b_q4km_metal`](receipts/pins/mistral_7b_q4km_metal.json) |
| Mixtral-8x7B-Instruct Q4_K_M | cpu | — | — | — | — | no pin | — |
| Llama-3.2-1B-Instruct IQ4_XS | metal | **148.43** ±0.3 | **138.25** ±0.8 | ~0.93× | ferrox | ok | [`iq4_xs_metal`](receipts/pins/iq4_xs_metal.json) |
| Gemma-2-2B-IT Q4_K_M | metal | **36.72** ±1.3 | **53.37** ±4.4 | ~1.45× | llama | ok | [`gemma2_2b_q4km_metal`](receipts/pins/gemma2_2b_q4km_metal.json) |
| Gemma-2-2B-IT Q4_K_M | cpu | **8.65** ±0.8 | **19.31** ±6.8 (−ngl 0) | ~2.23× | llama | ok | [`gemma2_2b_q4km_cpu`](receipts/pins/gemma2_2b_q4km_cpu.json) |
| SmolLM2-135M-Instruct Q8_0 | metal | **283.66** ±9.7 | **193.48** ±4.9 | ~0.68× | ferrox | ok | [`smollm2_135m_q8_metal`](receipts/pins/smollm2_135m_q8_metal.json) |
| SmolLM2-135M-Instruct Q8_0 | cpu | **55.84** ±0.7 | **43.48** ±5.9 (−ngl 0) | ~0.78× | ferrox | ok | [`smollm2_135m_q8_cpu`](receipts/pins/smollm2_135m_q8_cpu.json) |
| Qwen2.5-0.5B-Instruct Q8_0 | metal | **192.09** ±7.0 | **123.00** ±5.6 | ~0.64× | ferrox | ok | [`qwen25_05b_q8_metal`](receipts/pins/qwen25_05b_q8_metal.json) |
| Qwen2.5-0.5B-Instruct Q8_0 | cpu | **50.97** ±0.9 | **44.86** ±6.4 (−ngl 0) | ~0.88× | ferrox | ok | [`qwen25_05b_q8_cpu`](receipts/pins/qwen25_05b_q8_cpu.json) |
| Qwen3-0.6B Q8_0 | metal | **126.41** ±5.4 | **114.48** ±1.2 | ~0.91× | ferrox | ok | [`qwen3_06b_q8_metal`](receipts/pins/qwen3_06b_q8_metal.json) |
| Qwen3-0.6B Q8_0 | cpu | **26.02** ±1.4 | **36.69** ±3.8 (−ngl 0) | ~1.41× | llama | ok | [`qwen3_06b_q8_cpu`](receipts/pins/qwen3_06b_q8_cpu.json) |
| Gemma-3-1B-IT Q8_0 | metal | **50.20** ±0.5 | **74.48** ±1.1 | ~1.48× | llama | ok | [`gemma3_1b_q8_metal`](receipts/pins/gemma3_1b_q8_metal.json) |
| Gemma-3-1B-IT Q8_0 | cpu | **31.67** ±3.7 | **19.72** ±1.9 (−ngl 0) | ~0.62× | ferrox | ok | [`gemma3_1b_q8_cpu`](receipts/pins/gemma3_1b_q8_cpu.json) |
| Phi-3-mini-4k-Instruct Q4 | metal | **39.80** ±0.2 | **50.86** ±0.2 | ~1.28× | llama | ok | [`phi3_mini_q4_metal`](receipts/pins/phi3_mini_q4_metal.json) |
| Phi-3-mini-4k-Instruct Q4 | cpu | **5.10** ±0.7 | **8.80** ±4.4 (−ngl 0) | ~1.73× | llama | ok | [`phi3_mini_q4_cpu`](receipts/pins/phi3_mini_q4_cpu.json) |


## CLI completion (llama-cli vs `ferrox run`)

One-shot `-p … -n N --no-cnv --ignore-eos`, fresh process per rep, strictly sequential (llama exits before ferrox starts). Engines' own stderr timings; decode excludes model load.

| Model | Backend | ferrox pred (tok/s) | llama pred (tok/s) | Gap | Winner | Pin |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | metal | **116.84** ±1.2 | **111.10** ±3.8 | ~0.95× | ferrox | [`tinyllama_q8_metal_cli`](receipts/pins/tinyllama_q8_metal_cli.json) |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | cpu | **33.87** ±1.8 | **19.60** ±4.1 | ~0.58× | ferrox | [`tinyllama_q8_cpu_cli`](receipts/pins/tinyllama_q8_cpu_cli.json) |
| Llama-3.2-1B-Instruct Q4_K_M | metal | **143.96** ±1.8 | **140.20** ±1.2 | ~0.97× | ferrox | [`llama32_1b_q4km_metal_cli`](receipts/pins/llama32_1b_q4km_metal_cli.json) |
| OLMoE-1B-7B-0924 Q4_0 | cpu | **13.99** ±0.3 | **15.30** ±7.8 | ~1.09× | llama | [`olmoe_q4_cpu_cli`](receipts/pins/olmoe_q4_cpu_cli.json) |
| Llama-3.1-8B-Instruct Q4_K_M | metal | **28.64** ±0.2 | **28.60** ±0.6 | **1.00×** | tie | [`llama31_8b_q4km_metal_cli`](receipts/pins/llama31_8b_q4km_metal_cli.json) |
| Llama-3.2-3B-Instruct Q4_K_M | metal | **57.97** ±0.2 | **61.90** ±0.8 | ~1.07× | llama | [`llama32_3b_q4km_metal_cli`](receipts/pins/llama32_3b_q4km_metal_cli.json) |
| Llama-3.2-1B-Instruct IQ4_XS | metal | **146.54** ±2.7 | **135.20** ±3.6 | ~0.92× | ferrox | [`iq4_xs_metal_cli`](receipts/pins/iq4_xs_metal_cli.json) |
| SmolLM2-135M-Instruct Q8_0 | metal | **281.65** ±8.9 | **225.70** ±8.2 | ~0.80× | ferrox | [`smollm2_135m_q8_metal_cli`](receipts/pins/smollm2_135m_q8_metal_cli.json) |
| SmolLM2-135M-Instruct Q8_0 | cpu | **56.50** ±0.1 | **61.90** ±11.9 | ~1.10× | llama | [`smollm2_135m_q8_cpu_cli`](receipts/pins/smollm2_135m_q8_cpu_cli.json) |
| Qwen2.5-0.5B-Instruct Q8_0 | metal | **200.61** ±4.1 | **129.40** ±0.6 | ~0.65× | ferrox | [`qwen25_05b_q8_metal_cli`](receipts/pins/qwen25_05b_q8_metal_cli.json) |
| Qwen2.5-0.5B-Instruct Q8_0 | cpu | **51.42** ±0.4 | **41.70** ±4.4 | ~0.81× | ferrox | [`qwen25_05b_q8_cpu_cli`](receipts/pins/qwen25_05b_q8_cpu_cli.json) |
| Qwen3-0.6B Q8_0 | metal | **134.21** ±0.9 | **115.40** ±4.0 | ~0.86× | ferrox | [`qwen3_06b_q8_metal_cli`](receipts/pins/qwen3_06b_q8_metal_cli.json) |
| Qwen3-0.6B Q8_0 | cpu | **27.42** ±0.8 | **33.00** ±3.2 | ~1.20× | llama | [`qwen3_06b_q8_cpu_cli`](receipts/pins/qwen3_06b_q8_cpu_cli.json) |
| Gemma-3-1B-IT Q8_0 | metal | **49.79** ±0.7 | **78.30** ±1.2 | ~1.57× | llama | [`gemma3_1b_q8_metal_cli`](receipts/pins/gemma3_1b_q8_metal_cli.json) |
| Gemma-3-1B-IT Q8_0 | cpu | **37.81** ±1.8 | **31.10** ±11.3 | ~0.82× | ferrox | [`gemma3_1b_q8_cpu_cli`](receipts/pins/gemma3_1b_q8_cpu_cli.json) |
| Phi-3-mini-4k-Instruct Q4 | metal | **40.52** ±0.3 | **53.00** ±0.1 | ~1.31× | llama | [`phi3_mini_q4_metal_cli`](receipts/pins/phi3_mini_q4_metal_cli.json) |

## Open

1. Metal prefill ≪ llama on large models; FA-vec covers d=64/96/128/256 (Phi-3 / Gemma-3 decode path).
2. CUDA — re-measure on comparable CUDA hardware (no in-tree CUDA pin).
3. Gemma-2 arch support (attn softcap + pin; suite currently refuse).
4. CB multi-request tok/s receipt.
5. DS4 / GLM real e2e when feasible.
6. Qwen2-MoE / Mistral / Mixtral oracle receipts.

Do not invent numbers without a pin.
