# Models

What Ferrox can run today. Speed ratios use llama.cpp as the baseline; receipts:
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).
Architecture registry: `ferrox archs` → [`manifests/architecture_manifest.md`](manifests/architecture_manifest.md).

**Rule:** “Works” means a real GGUF loaded and answered correctly. “Fast” needs a pin in RESULTS. No pin → no speed claim.

## Try these

| Model | Why |
|---|---|
| SmolLM2-135M-Instruct Q8_0 | Tiny; CPU ~1.3× |
| TinyLlama-1.1B-Chat Q8_0 | Smallest verified smoke; CPU ~1.2× |
| OLMoE-1B-7B-0924 Q4_0 | MoE; CPU parity (+ CUDA historically) |
| Llama-3.1-8B-Instruct Q4_K_M | Mainstream dense model; Metal ~1× decode |

```bash
# CLI (see docs/CLI.md)
./target/release/ferrox -m /path/to/model.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

# HTTP server
./target/release/ferrox-server -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8383
```

## Status

| Status | Meaning |
|---|---|
| **Verified** | Real checkpoint + oracle / receipt |
| **Works** | Loads and chats coherently (local smoke); not yet pinned |
| **Partial** | Primitives or slices only — not end-to-end |
| **No** | Fail-closed or out of scope |

### Verified

| Model | Backend |
|---|---|
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | CPU (~1.2×), Metal |
| OLMoE-1B-7B-0924 Q4_0 | CPU (~parity), CUDA |
| Llama-3.1-8B-Instruct Q4_K_M | CPU, Metal (~1× decode) |
| SmolLM2-135M-Instruct Q8_0 | CPU (~1.3×), Metal |
| Qwen2.5-0.5B-Instruct Q8_0 | CPU (~1.1×); Metal ~1.56× **faster** (bias on GPU) |
| Qwen3-0.6B Q8_0 | CPU ~0.7×; Metal ~1.1× (per-head QK-norm on GPU) |
| Gemma-3-1B-IT Q8_0 | CPU (~1.6×); Metal ~0.67× — full Metal stack (GeGLU, SWA, sandwich norms); d=256 legacy attn is the gap |
| Phi-3-mini-4k-Instruct Q4 | CPU ~0.6×; Metal ~0.78× — fused QKV/FFN split as quantized slices; d=96 legacy attn |

Also Metal-pinned (see RESULTS): Llama-3.2-1B / 3B Q4_K_M,
Llama-3.2-1B IQ4_XS.

### Works (no pin yet)

| Model | Notes |
|---|---|
| — | (everything smoked so far is pinned) |

### Partial / not yet

| | |
|---|---|
| Qwen2-MoE | Loads; oracle re-verify pending |
| Mistral / Mixtral | Profile + SWA / grouped MoE wired; need receipt |
| Kimi K3 | Real slices only (~1.56 TB full run not done) |
| GLM-5.2 / DeepSeek V4 | Synthetic / dedicated stacks — no real GGUF e2e |
| MLA / Mamba / T5 / VL | Fail-closed at load |
| CUDA speed | Deferred (compile OK; fair-chat remains far behind) |

## Backends

| Backend | Today |
|---|---|
| CPU | Primary correctness path |
| Metal | Dense + attn incl. QKV bias, per-head QK-norm, GeGLU, SWA, sandwich norms; protect Llama 8B pin in RESULTS |
| CUDA | Compiles; speed work paused until Metal parity program settles |

Unknown `general.architecture` values fail closed with a clear error (`ferrox archs` for the list).
