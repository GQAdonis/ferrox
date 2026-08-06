# Roadmap

What works today: [`MODELS.md`](MODELS.md) · CLI: [`CLI.md`](CLI.md) ·
speed: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) ·
API: [`API.md`](API.md) · agents: [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).

**Goal:** ≥ [llama.cpp](https://github.com/ggerganov/llama.cpp) tok/s on the same host / backend / GGUF.
Evidence-first: no “supported” or “fast” without a receipt. No Candle / Crane /
ds4 deps — rewrite in-tree.

## Now (foundation P1–P5)

| Phase | Focus | Notes |
|---|---|---|
| **P1** | Metal prefill | FA-vec prefill d=128 landed; Llama-3.1-8B Metal fair-chat ~parity (~0.97×) — re-check `prompt_per_second` under quiet host |
| **P2** | MLA GGUF → `MlaEngine` | CLI + **ferrox-server** dense-lead deepseek2/mistral4; MoE-after-dense fail-closed; GLM-4.7 / DS V3 MoE still open |
| **P3** | Hybrid GDN | `gdn.rs` + loader scaffold; `HybridEngine` assemble + Qwen3.5 smoke still open |
| **P4** | Gemma-4 text | E2B fail-closed (**DedicatedOnly**: per-layer emb + shared KV + SWA/full head-dim); suite refuse pin; MoE-A4B / VL → P7 |
| **P5** | KV quant | Metal `q8_0` store + shared f16 dequant scratch landed; `fp8`/`turbo*` still warn→F16; CPU turbo4 sketch |

Also on this horizon (unchanged intent):

- **Receipts** — real GGUF oracles for Gemma-2, Qwen2-MoE, Mistral, Mixtral (suite entries exist; pins pending checkpoints).
- **Metal MoE** — unfused `matvec_id` (llama-style occupancy); still ~1.66× on OLMoE — next: simdgroup `mul_mm` prefill + expert residency hoist.
- **Frontier** — Kimi multi-layer → full e2e; GLM-5.2 / DeepSeek V4 real quants (engines exist; fail-closed until receipts).
- **CUDA** — fair-chat pins via `run_suite.py --backend cuda --host-label …`; staged ≥0.5× then parity (no invented numbers).

## Next (models P6–P8)

| Phase | Focus | Notes |
|---|---|---|
| **P6** | Text / MoE matrix | Phi-4-mini Metal pin landed; **yi** GenericGqa; MiroThinker→`qwen3moe`; llama4/minimax stubs; GLM4 serve path wired (loader-level, no e2e pin) |
| **P7** | VL | `mmproj::find_mmproj_beside` + load warnings; `vl_engine` projector stub; server **rejects** `image_url` (400) |
| **P8** | MTP / embed / GLM-5.2 | `--mtp` fail-closed; `/v1/embeddings` documented Supported (Decoder); GLM-5.2 CLI+server via `load_glm52_engine_from_path` |

Also: recurrent / hybrid / T5 stubs already exist — implement after receipts, not before.

## Serving (P9–P11)

| Phase | Focus | Notes |
|---|---|---|
| **P9** | Server API | Tokenize / detokenize / completions / decoder embeddings + **Anthropic `/v1/messages`**; **presence/frequency penalties**, best-effort **`json_object`**, **web UI** (`--ui-server`), **MCP config stub** — shipped; full grammar / MCP invoke still open |
| **P10** | Runtime scale | CB opt-in + `cb_throughput.py`; **chunked prefill** (`FERROX_CHUNKED_PREFILL`), **HF hub pull** (`ferrox pull`), **CPU KV offload stub** (`FERROX_CPU_KV_OFFLOAD`); multi-GPU / TP / PD **planned** |
| **P11** | Polish | Agent cookbook (Anthropic route, CB, UI); CUDA fair-chat pins still need GPU host; optional ISQ deferred |

Also: continuous-batching throughput pin (`benchmarks/cb_throughput.py` against a live server) under **P10**. Multi-GPU, tensor parallel, and prefill/decode disaggregation remain **planned** (no fake implementation).

Also under **P10**: `FERROX_CHUNKED_PREFILL=N` splits long prompt prefill into incremental `forward_batch` chunks; `ferrox pull org/model` (or `FERROX_MODEL_PATH=org/model` when `hf` CLI is installed) fetches GGUF from Hugging Face Hub; `FERROX_CPU_KV_OFFLOAD=1` syncs Metal KV to host each decode step (minimal spill — full layer offload still open).

## Shipped (do not re-list as open)

- Evidence baseline + RESULTS validation (`render_results.py`)
- Metal FA-vec d=96/256; softcap CPU + Metal legacy GQA
- Overlapped SSE; API compatibility matrix; `ferrox chat` REPL
- CUDA suite backend + host-label pins workflow
- Fail-closed architecture registry (`ferrox archs`)

## Rules

Evidence-first. Prefer honest scaffolding and clear `DedicatedOnly` /
`Deferred` reasons over silent wrong graphs.
No Candle / Crane / ds4 as dependencies — rewrite in-tree.
