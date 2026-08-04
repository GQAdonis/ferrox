# Benchmarks

Ledger: **[`RESULTS.md`](RESULTS.md)** (generated — do not hand-edit).  
Suite: [`suite.json`](suite.json). Runner: [`run_suite.py`](run_suite.py).  
Stable pins: [`receipts/pins/`](receipts/pins/).

## Workload

- Host B: Apple M2 Pro, 10 logical cores
- Greedy chat (`temperature: 0`), warm request, then `max_tokens=256`
- Prompt: numbered list of 80 European capitals + unique suffix
- Prefer **predicted** tok/s; gap only when both ferrox and llama succeed

## Run suite (Metal)

```bash
cargo build -p ferrox-server --release --features metal

# list models + whether GGUF is present under models/
python3 benchmarks/run_suite.py --list

# one model
python3 benchmarks/run_suite.py --id llama31_8b_q4km --backend metal

# all metal entries that resolve (or write status=missing pins)
python3 benchmarks/run_suite.py --backend metal
```

Place GGUFs in `models/` at the paths configured by each suite `gguf` field.
Each run overwrites `receipts/pins/{id}_{backend}.json` and regenerates
[`RESULTS.md`](RESULTS.md).

## Run suite (CPU)

```bash
cargo build -p ferrox-server --release   # no metal feature required for CPU path

python3 benchmarks/run_suite.py --id tinyllama_q8 --backend cpu
# or: python3 benchmarks/run_suite.py --backend cpu
```

CPU path sets `FERROX_METAL=0`, `FERROX_CPU_INT_DOT=1`,
`RAYON_NUM_THREADS=10`; llama uses `-ngl 0 -t 10`.

## Regenerate ledger only

```bash
python3 benchmarks/render_results.py
```

## Legacy shim

[`fair_chat_256.py`](fair_chat_256.py) forwards to `run_suite.py`. Prefer
the suite CLI above.

## Env (ferrox)

| Var | Typical |
|---|---|
| `FERROX_METAL` | `1` (Metal), `0` for CPU runs |
| `FERROX_METAL_ATTN` | `1` |
| `FERROX_METAL_LOGITS` | unset (host lm_head; `1` = slower vocab-in-stack) |
| `FERROX_METAL_GREEDY_GPU` | default **on** for `temperature<=0`; `0` = host lm_head |
| `FERROX_METAL_FA_VEC` | default **on** for `head_dim==128`; `0` = legacy GQA |
| `FERROX_METAL_MUL_MM` | default **on** for prefill batch ≥ 8; `0` = N× matvec |
| `FERROX_METAL_WEIGHT_COPY` | unset (`BytesNoCopy`); `1` forces copy upload |
| `FERROX_CPU_INT_DOT` | `1` on CPU suite runs |
| `FERROX_CUDA_GQA` / `FERROX_CUDA_GRAPH` | CUDA path (Vast study, not Host B suite) |
| `RAYON_NUM_THREADS` | `10` on Host B |

## CUDA

Host B suite does not run CUDA. The prior Vast study is preserved in
[`receipts/pins/cuda_vast_llama31_8b_q4km.json`](receipts/pins/cuda_vast_llama31_8b_q4km.json).
