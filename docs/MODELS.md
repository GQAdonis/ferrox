# Models

What Ferrox can run, and how it compares to llama.cpp on the same host.
Pins: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).
Architecture list: `ferrox archs` →
[`manifests/architecture_manifest.md`](manifests/architecture_manifest.md).

**Gap** = `llama_pred / ferrox_pred`. Values below 1.0 mean Ferrox is faster.

Suite policy: keep the **current** generation per family (e.g. Llama-3.2, not
3.1; Gemma-3/4, not Gemma-2; Phi-4, not Phi-3). Older GGUFs still load when
the architecture is supported — they are just not in the published bench
ledger.

## Recommended starters

| Model | Notes |
|---|---|
| SmolLM2-135M-Instruct Q8_0 | Tiny; Metal ahead of llama, CPU well behind |
| TinyLlama-1.1B-Chat Q8_0 | Smallest verified smoke |
| Phi-4-mini-Instruct Q4_K_M | Metal ~parity |
| Llama-3.2-3B-Instruct Q4_K_M | Metal flagship in the suite |
| Gemma-4-E2B-IT Q4_K_M | Dedicated engine + `gemma4` BPE |

```bash
./target/release/ferrox -m /path/to/model.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

./target/release/ferrox-server -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8383

./target/release/ferrox chat --url http://127.0.0.1:8383
```

## Verified (Host B pins)

| Model | Metal serving | Metal decode (engine) | CPU decode (engine) |
|---|---|---|---|
| SmolLM2-135M Q8_0 | **0.76×** | **0.64×** | 2.60× |
| Qwen2.5-0.5B Q8_0 | **0.63×** | **0.60×** | 1.84× |
| Qwen3-0.6B Q8_0 | **0.81×** | **0.71×** | 1.78× |
| Gemma-3-1B-IT Q8_0 | **0.74×** | **0.79×** | 1.92× |
| Llama-3.2-1B IQ4_XS | **0.83×** | **0.94×** | — |
| Llama-3.2-1B Q4_K_M | **0.88×** | **0.95×** | — |
| TinyLlama-1.1B Q8_0 | **0.89×** | **0.89×** | 1.33× |
| Llama-3.2-3B Q4_K_M | **0.94×** | **0.84×** | — |
| Phi-4-mini Q4_K_M | 1.00× | **0.93×** | 1.92× |
| Mistral-7B-v0.2 Q4_K_M | 1.04× | 0.99× | 1.62× |
| OLMoE-1B-7B Q4_0 | 1.59× | 1.29× | 1.71× |

Gap = `llama / ferrox`; **bold** = ferrox faster. *Serving* is over HTTP
with template and sampler in the loop; *engine* is `ferrox bench` vs
`llama-bench`, no HTTP. Neither engine's thread count is forced.

Numbers drift as receipts refresh — prefer
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) for the latest table.
Prefill (`pp512`) is still the main engine gap; read it off RESULTS, not
this summary.

## Other support

| Model / family | Status |
|---|---|
| Yi (text) | Works (GenericGqa, Neox RoPE) — no speed pin yet |
| MiroThinker | Works via `qwen3moe` |
| Qwen2-MoE / Qwen1.5-MoE | Loads; not in current suite (OLMoE is the MoE pin) |
| Mixtral | Suite entry; skipped on 32 GiB Host B (`--fit-host`) |
| MLA (`deepseek2` / `mistral4`) | Dense-lead + MoE-after-dense via `MlaEngine` |
| GLM4 / glm4moe | Loads via GLM-5.2 path when tensors present; no e2e pin |
| Gemma-4-E2B | Dedicated `Gemma4Engine` + SPM-style `gemma4` BPE tokenizer + `<|turn>` chat wrap. GGUF: `models/gemma-4-E2B-it-Q4_K_M.gguf` (`unsloth/gemma-4-E2B-it-GGUF`). Fair-chat pin still open. |
| Llama 4 / MiniMax | Refused (stubs) |
| Hybrid GDN / Qwen3.5 | Scaffold only |
| Kimi K3 / GLM-5.2 / DeepSeek V4 | Loaders/primitives; no frontier e2e pin |
| Vision | mmproj discover + warn; `image_url` rejected |
| MTP | `--mtp` errors; prompt-lookup via `ferrox speculative` |
| Embeddings | `/v1/embeddings` for GGUF Decoder (mean/last pool) |

## Backends

| Backend | What it covers |
|---|---|
| CPU | Dense + MoE; `FERROX_CPU_INT_DOT=1` (Q4_Kx8 / Q8_0x4 / Q5·Q6 int-dot) on suite runs |
| Metal | Dense + MoE + FA-vec; fused MoE encode groups; `mul_mm_id` prefill; quantized KV |
| CUDA | Matvec + resident weights + FFN fuse |

Capabilities overview: [`FEATURES.md`](FEATURES.md).
Planned work: [`ROADMAP.md`](ROADMAP.md).
