# Configuration

Ferrox is configured through CLI flags first (see [CLI.md](CLI.md));
environment variables cover server deployment and advanced tuning.
Flags win over environment variables where both exist.

## Server basics

| Variable | Purpose |
|---|---|
| `FERROX_MODEL_PATH` | GGUF file (or Kimi dir) to serve; same as `-m/--model` |
| `FERROX_ADDR` | Bind address, e.g. `127.0.0.1:8383`; same as `--host/--port` |
| `FERROX_API_KEY` | When set, requests must send `Authorization: Bearer <key>` |

## Backends

Normally set for you by `-dev` / `-ngl` — only set these manually when
running without those flags (e.g. embedding ferrox as a library).

| Variable | Purpose |
|---|---|
| `FERROX_METAL` | `1`/`0`/`auto` — Metal matvec/dense offload |
| `FERROX_METAL_ATTN` | `1`/`0` — fused Metal attention + resident KV cache (decode/MoE stacks use concurrent encode like llama.cpp: gate∥up, Q∥K∥V) |
| `FERROX_CTK` | KV cache dtype: `f16` (default); Metal quantized: `q8_0`/`turbo8`/`fp8` (34 B/32 elems), `turbo4` (18 B/32 elems) — dequant to shared f16 scratch for FA. `turbo3` still warn→F16. Same as CLI `--ctk`. |
| `FERROX_CUDA` | `1`/`0`/`auto` — CUDA offload (build with `--features cuda`) |
| `FERROX_CPU_THREADS` | CPU worker threads; same as `-t/--threads` |
| `FERROX_CPU_INT_DOT` | `1` — int8×int8 matvec (Q8_0/Q4_0/Q4_K); suite CPU runs set this. MoE reuses one Q8 act quant across top-k experts |

## Advanced tuning

Defaults are sensible; measure before changing these.

| Variable | Purpose |
|---|---|
| `FERROX_CONTINUOUS_BATCHING` | `1` — share decode steps across concurrent server requests |
| `FERROX_CHUNKED_PREFILL` | Positive integer — split long prompt prefill into N-token `forward_batch` chunks |
| `FERROX_CPU_KV_OFFLOAD` | `1` — after each Metal decode step, sync GPU KV into host caches (minimal spill stub) |
| `FERROX_UI` | `1` — serve static chat UI at `/` and `/ui` (same as `--ui-server`) |
| `FERROX_KV_POOL_BLOCKS` | Paged-KV pool size (blocks) for the server |
| `FERROX_EXPERT_CACHE_BYTES` | MoE expert-streaming cache budget (bytes) |
| `FERROX_SSD_STREAMING` | `1` — stream MoE experts from disk (defaults cache to 2 GiB) |
| `FERROX_GPU_VRAM_BUDGET_BYTES` | Cap for GPU-resident MoE experts. On Metal, unset defaults to a large budget (Metal matvec experts); set `0` to force CPU experts. CUDA still requires an explicit value. |

Debug/experiment switches (`FERROX_METAL_FA_VEC`, `FERROX_METAL_LOGITS`,
`FERROX_METAL_MOE_RESIDENT`, …) are documented next to their code and may
change without notice.
