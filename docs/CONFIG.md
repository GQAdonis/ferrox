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
| `FERROX_METAL_ATTN` | `1`/`0` — fused Metal attention + resident KV cache |
| `FERROX_CUDA` | `1`/`0`/`auto` — CUDA offload (build with `--features cuda`) |
| `FERROX_CPU_THREADS` | CPU worker threads; same as `-t/--threads` |

## Advanced tuning

Defaults are sensible; measure before changing these.

| Variable | Purpose |
|---|---|
| `FERROX_CONTINUOUS_BATCHING` | `1` — share decode steps across concurrent server requests |
| `FERROX_KV_POOL_BLOCKS` | Paged-KV pool size (blocks) for the server |
| `FERROX_EXPERT_CACHE_BYTES` | MoE expert-streaming cache budget (bytes) |
| `FERROX_SSD_STREAMING` | `1` — stream MoE experts from disk (defaults cache to 2 GiB) |
| `FERROX_GPU_VRAM_BUDGET_BYTES` | Cap for GPU-resident weights |

Debug/experiment switches (`FERROX_METAL_FA_VEC`, `FERROX_METAL_LOGITS`,
`FERROX_CPU_INT_DOT`, …) are documented next to their code and may change
without notice.
