# Models

What Ferrox can run, and how it compares to llama.cpp on the same host.
Pins: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).
Architecture list: `ferrox archs` →
[`manifests/architecture_manifest.md`](manifests/architecture_manifest.md).

**Gap** = `llama_pred / ferrox_pred`. Values below 1.0 mean Ferrox is faster.

## Recommended starters

| Model | Notes |
|---|---|
| SmolLM2-135M-Instruct Q8_0 | Tiny; Metal ahead of llama |
| TinyLlama-1.1B-Chat Q8_0 | Smallest verified smoke |
| Phi-4-mini-Instruct Q4_K_M | Metal ~parity |
| Llama-3.2-3B-Instruct Q4_K_M | Metal ~parity (~0.97×) |
| Llama-3.1-8B-Instruct Q4_K_M | Metal fair-chat ahead (~0.92×) |

```bash
./target/release/ferrox -m /path/to/model.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

./target/release/ferrox-server -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8383

./target/release/ferrox chat --url http://127.0.0.1:8383
```

## Verified (Host B pins)

| Model | Metal Gap | Notes |
|---|---|---|
| TinyLlama-1.1B-Chat Q8_0 | ~0.96× | CPU pin also exists |
| Llama-3.2-1B Q4_K_M | ~1.00× | |
| Llama-3.2-3B Q4_K_M | **~0.97×** | Concurrent FFN encode |
| Llama-3.2-1B IQ4_XS | ~0.94× | |
| Llama-3.1-8B-Instruct Q4_K_M | **~0.92×** | CLI ~1.00× |
| Mistral-7B-Instruct-v0.2 Q4_K_M | ~0.95× | CPU pin exists |
| OLMoE-1B-7B-0924 Q4_0 | **~1.41×** | CPU **~0.96×** (ahead); Metal fused encode + `mul_mm_id` (gated) |
| SmolLM2-135M / Qwen2.5-0.5B / Qwen3-0.6B | ahead | Small Q8 pins |
| Gemma-2-2B-IT Q4_K_M | ~1.10× | Softcap + FA-vec |
| Gemma-3-1B-IT Q8_0 | ~0.82× | GeGLU + SWA |
| Phi-3-mini-4k Q4 | ~0.92× | Fused QKV; FA-vec d=96 |
| Phi-4-mini-Instruct Q4_K_M | ~1.00× | CLI ~1.04× |

## Other support

| Model / family | Status |
|---|---|
| Yi (text) | Works (GenericGqa, Neox RoPE) — no speed pin yet |
| MiroThinker | Works via `qwen3moe` |
| Qwen2-MoE | Loads; suite entry; GGUF not on Host B |
| Mixtral | Suite entry; skipped on 32 GiB Host B (`--fit-host`) |
| MLA (`deepseek2` / `mistral4`) | Dense-lead + MoE-after-dense via `MlaEngine` |
| GLM4 / glm4moe | Loads via GLM-5.2 path when tensors present; no e2e pin |
| Gemma-4-E2B | Dedicated `Gemma4Engine` loads (per-layer emb + shared KV + SWA/full head-dim). Tokenizer `gemma4` still falls back to byte; suite `expect=refuse` until fair-chat pin. GGUF: `models/gemma-4-E2B-it-Q4_K_M.gguf` |
| Llama 4 / MiniMax | Refused (stubs) |
| Hybrid GDN / Qwen3.5 | Scaffold only |
| Kimi K3 / GLM-5.2 / DeepSeek V4 | Loaders/primitives; no frontier e2e pin |
| Vision | mmproj discover + warn; `image_url` rejected |
| MTP | `--mtp` errors; prompt-lookup via `ferrox speculative` |
| Embeddings | `/v1/embeddings` for GGUF Decoder (mean/last pool) |

## Backends

| Backend | What it covers |
|---|---|
| CPU | Dense + MoE; `FERROX_CPU_INT_DOT=1` (+ interleaved Q4_K) on suite runs |
| Metal | Dense + MoE + FA-vec; fused MoE encode groups; `mul_mm_id` prefill; quantized KV |
| CUDA | Matvec + resident weights + FFN fuse |

Capabilities overview: [`FEATURES.md`](FEATURES.md).
Planned work: [`ROADMAP.md`](ROADMAP.md).
