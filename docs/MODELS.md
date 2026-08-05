# Models

What Ferrox can run today. Speed ratios use llama.cpp as the baseline; receipts:
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).
Architecture registry: `ferrox archs` → [`manifests/architecture_manifest.md`](manifests/architecture_manifest.md).

**Rule:** “Works” means a real GGUF loaded and answered correctly. “Fast” needs a pin in RESULTS. No pin → no speed claim.

**Ratio convention:** `benchmarks/RESULTS.md` Gap = `llama_pred / ferrox_pred`.
Gap &lt; 1.0 means ferrox is faster; gap &gt; 1.0 means ferrox is slower.
Human prose below (e.g. “~1.56× faster”) is the inverse when ferrox wins —
always prefer the pin / RESULTS Gap for comparisons.

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

# Interactive chat (server must already be running)
./target/release/ferrox chat --url http://127.0.0.1:8383
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
| OLMoE-1B-7B-0924 Q4_0 | CPU (~parity); CUDA historically (no current Host B pin) |
| Llama-3.1-8B-Instruct Q4_K_M | CPU; Metal fair-chat Gap ~1.03× (CLI pin **1.00×**) |
| SmolLM2-135M-Instruct Q8_0 | CPU (~1.3×), Metal |
| Qwen2.5-0.5B-Instruct Q8_0 | CPU (~1.1×); Metal ~1.56× **faster** (bias on GPU) |
| Qwen3-0.6B Q8_0 | CPU ~0.7×; Metal ~1.1× (per-head QK-norm on GPU) |
| Gemma-3-1B-IT Q8_0 | CPU (~1.6×); Metal — full Metal stack (GeGLU, SWA, sandwich norms); FA-vec covers d=256 |
| Phi-3-mini-4k-Instruct Q4 | CPU ~0.6×; Metal — fused QKV/FFN split as quantized slices; FA-vec covers d=96 |

Also Metal-pinned (see RESULTS): Llama-3.2-1B / 3B Q4_K_M,
Llama-3.2-1B IQ4_XS.

### Partial / not yet

| | |
|---|---|
| Qwen2-MoE | Loads (QKV bias + shared_expert_gate); suite entry added; oracle receipt needs real GGUF |
| Mistral / Mixtral | Profile + SWA / grouped MoE wired; suite entries added; need receipt |
| Gemma-2 | Attn softcap + SWA wired (CPU + Metal legacy GQA); needs real GGUF pin |
| Gemma-4 | Admitted to GemmaFamily (`gemma4` / `gemma4-assistant` GenericGqa NeoX); Works pending receipt — MoE-A4B / VL later |
| Phi-4 | Arch `phi4` admitted as PhiFamily (same fused path as phi3); no suite pin yet (**P6**) |
| MiniMax M2/M3 | DedicatedOnly — 256-expert sigmoid MoE + MTP not implemented (was wrongly generic) |
| MLA | `mla_gguf_loader` + `load_mla_engine_from_path` + CLI `ferrox run` for dense-lead deepseek2/mistral4; MoE fail-closed; server path still Decoder-only |
| Hybrid GDN | `gdn.rs` primitive + `HybridEngine` stub (**P3**); hybrid arches remain DedicatedOnly until GGUF load lands |
| Kimi K3 | Real slice loaders + `kimi_validate` index check; full ~1.56 TB e2e gated on storage |
| GLM-5.2 / DeepSeek V4 | Dedicated engines (`Glm52Engine`, `DeepseekV4Engine`) + synthetic stacks — no real GGUF e2e (**P8**) |
| VL / MTP | Deferred (**P7** / **P8**); `vl_engine` stub; no multimodal or MTP serve path |
| Mamba / T5 | Fail-closed; engine stubs (`recurrent_engine` / `t5_engine`) |
| CUDA speed | Suite supports `--backend cuda`; need GPU host pins (staged ≥0.5× then parity) |

## Backends

| Backend | Today |
|---|---|
| CPU | Primary correctness path |
| Metal | Dense + attn + MoE expert placement (default VRAM budget); FA-vec decode d=64/96/128/256; FA-vec prefill d=128; softcap via legacy GQA; `FERROX_CTK` q8_0 scaffolded (f16 default) |
| CUDA | Compiles; `run_suite.py --backend cuda --host-label …` for pins; explicit `FERROX_GPU_VRAM_BUDGET_BYTES` for MoE |

Unknown `general.architecture` values fail closed with a clear error (`ferrox archs` for the list).
