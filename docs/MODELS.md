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
| SmolLM2-135M Q8_0 | **0.73×** | 3.32× |
| Qwen2.5-0.5B Q8_0 | **0.75×** | 2.10× |
| Qwen3-0.6B Q8_0 | **0.88×** | 2.07× |
| Gemma-3-1B-IT Q8_0 | **0.95×** | 1.86× |
| Llama-3.2-1B IQ4_XS | 1.05× | — |
| Llama-3.2-1B Q4_K_M | **0.94×** | — |
| TinyLlama-1.1B Q8_0 | **0.93×** | 1.59× |
| Llama-3.2-3B Q4_K_M | 1.11× | — |
| Phi-4-mini Q4_K_M | 1.04× | 2.69× |
| Mistral-7B-v0.2 Q4_K_M | 1.04× | 2.94× |
| OLMoE-1B-7B Q4_0 | 1.54× | 1.71× |

Numbers drift as receipts refresh — prefer
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) for the latest table.
Prefill (`pp512`) is still the main gap; read it off RESULTS, not
this summary.

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
| Llama 4 / MiniMax | Refused (stubs) |
| Hybrid GDN / Qwen3.5 | Scaffold only |
| Kimi K3 / GLM-5.2 / DeepSeek V4 | Loaders/primitives; no frontier e2e receipt |
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
