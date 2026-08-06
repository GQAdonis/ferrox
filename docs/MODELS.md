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
| SmolLM2-135M-Instruct Q8_0 | Tiny; Metal ~1.2× faster |
| TinyLlama-1.1B-Chat Q8_0 | Smallest verified smoke; Metal ~parity |
| Phi-4-mini-Instruct Q4_K_M | New PhiFamily pin; Metal ~1.06× |
| Llama-3.1-8B-Instruct Q4_K_M | Mainstream dense; Metal fair-chat ~parity |

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

| Model | Backend (Gap = llama/ferrox; Host B pins) |
|---|---|
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | Metal ~0.96× (quiet re-pin); CPU pin exists |
| Llama-3.2-1B / 3B Q4_K_M | Metal ~1.00× / ~1.04× |
| Llama-3.2-1B IQ4_XS | Metal ~1.00× |
| Llama-3.1-8B-Instruct Q4_K_M | Metal fair-chat **~0.97×**; CLI **~1.00×** |
| Mistral-7B-Instruct-v0.2 Q4_K_M | Metal ~1.06×; CPU pin exists |
| OLMoE-1B-7B-0924 Q4_0 | Metal ~1.66× (experts; unfused matvec_id); CPU pin exists |
| SmolLM2-135M / Qwen2.5-0.5B / Qwen3-0.6B | Metal faster than llama on these small Q8 pins |
| Gemma-2-2B-IT Q4_K_M | Metal ~1.11×; softcap on FA-vec decode + prefill d=256 |
| Gemma-3-1B-IT Q8_0 | Metal ~0.87×; GeGLU + SWA + sandwich; FA-vec d=256 |
| Phi-3-mini-4k-Instruct Q4 | Metal ~1.09×; fused QKV; FA-vec d=96 |
| Phi-4-mini-Instruct Q4_K_M | Metal ~1.06×; CLI ~1.04× — [`phi4_mini_q4km_metal`](../benchmarks/receipts/pins/phi4_mini_q4km_metal.json) |

### Partial / not yet

| | |
|---|---|
| Qwen2-MoE | Loads (QKV bias + shared_expert_gate); suite entry; GGUF not on Host B |
| Mixtral | Suite entry; `estimated_ram_gb=48` → skipped on 32 GiB Host B (`--fit-host`) |
| Yi (text) | **Works** via GenericGqa (`yi`, Neox RoPE) — no speed pin yet |
| Gemma-4-E2B | **DedicatedOnly** — per-layer emb + shared KV + SWA/full head-dim split; suite `expect=refuse` (ferrox + current Homebrew llama both refuse `gemma4`) |
| Llama 4 | **DedicatedOnly** — MoE + non-GQA attn; `llama4_engine.rs` stub |
| GLM4 / glm4moe | **Partial** — CLI + **ferrox-server** via `load_glm52_engine_from_path` when tensors present; no real-checkpoint e2e pin |
| MiroThinker | Published GGUFs use `qwen3moe` (**Works** via GenericGqa) |
| MiniMax M2/M3 | **DedicatedOnly** — sigmoid MoE + MTP; `minimax_engine.rs` stub |
| MLA | CLI + **ferrox-server** dense-lead deepseek2/mistral4; MoE-after-dense fail-closed |
| Hybrid GDN | Primitive + `hybrid_gguf_loader` scaffold; assemble / serve fail-closed |
| Kimi K3 | Slice loaders + validate; full ~1.56 TB e2e gated on storage |
| GLM-5.2 / DeepSeek V4 | GLM-5.2 loader + synthetic stacks; DS V4 dedicated engine stub — no real GGUF e2e pin |
| VL | `mmproj::find_mmproj_beside` + warn on load; `vl_engine` stub; server **rejects** `image_url` (400) |
| MTP | `--mtp` fail-closed; prompt-lookup via `ferrox speculative` only |
| Embeddings (`/v1/embeddings`) | **Supported** for GGUF `Decoder` (mean/last pool); BERT-family **DeferredEncoder** |
| Mamba / T5 | Fail-closed stubs |
| CUDA speed | Suite `--backend cuda`; need GPU host pins |
| Metal KV quant | `FERROX_CTK=q8_0` / `--ctk q8_0`: ggml Q8_0 resident store + process-wide f16 dequant for FA (needs `n_kv*head_dim` % 32 == 0); `fp8`/`turbo*` still F16 |

## Backends

| Backend | Status |
|---|---|
| CPU | Dense + MoE; `FERROX_CPU_INT_DOT=1` on suite runs |
| Metal | Dense + attn + MoE expert placement; FA-vec decode d=64/96/128/256; FA-vec prefill d=128/256 + softcap |
| CUDA | Matvec + resident weights + FFN fuse; fair-chat pins need a CUDA host |
