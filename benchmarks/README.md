# Benchmarks

Ledger: **[`RESULTS.md`](RESULTS.md)** (generated — do not hand-edit).  
Suite: [`suite.json`](suite.json). Runner: [`run_suite.py`](run_suite.py).  
Stable pins: [`receipts/pins/`](receipts/pins/).

## Workload

- Host B: Apple M2 Pro, 10 logical cores
- Greedy chat (`temperature: 0`), warm request, then `max_tokens=512` (suite default; older pins may still record 256)
- Prompt: numbered list of 80 European capitals + unique suffix
- Prefer **predicted** tok/s; gap only when both ferrox and llama succeed

## Gap ratio convention

`RESULTS.md` **Gap** = `llama_predicted / ferrox_predicted` (from
[`render_results.py`](render_results.py)).

| Gap | Meaning |
|---|---|
| &lt; 1.0 | ferrox faster than llama.cpp |
| **1.00×** | within 1.5% of parity |
| &gt; 1.0 | ferrox slower than llama.cpp |

Human prose in docs (e.g. “1.56× faster”) is the inverse when ferrox wins.
Never invent ratios without a pin under [`receipts/pins/`](receipts/pins/).

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

## CLI completion (+ load / startup)

One-shot `llama-completion` vs `ferrox run` (fresh process per rep):

```bash
cargo build -p ferrox-cli --release --features metal
cp target/release/ferrox target/bench/ferrox-cli-metal

python3 benchmarks/run_suite.py --id tinyllama_q8 --backend metal --mode cli
```

Pins record **predicted** tok/s (decode, excludes load) and **load_s**
(engine-reported: `ferrox: loaded in …s` vs llama
`common_perf_print: load time = … ms`). Load gap in RESULTS =
`ferrox_load / llama_load` (same convention as pred Gap: &lt;1 ferrox better).
Default llama binary is `llama-completion`
when on `PATH` (Homebrew llama.cpp ≥b76xx).

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
| `FERROX_METAL_FA_VEC` | default **on** for `head_dim` in {64,96,128,256}; `0` = legacy GQA |
| `FERROX_METAL_MUL_MM` | default **on** for prefill batch ≥ 4; `0` = N× matvec |
| `FERROX_METAL_WEIGHT_COPY` | unset (`BytesNoCopy`); `1` forces copy upload |
| `FERROX_CPU_INT_DOT` | `1` on CPU suite runs |
| `FERROX_CUDA_GQA` / `FERROX_CUDA_GRAPH` | CUDA path (Vast study, not Host B suite) |
| `RAYON_NUM_THREADS` | `10` on Host B |

## CUDA

Host B suite is Metal/CPU. CUDA fair-chat is available via:

```bash
cargo build -p ferrox-server --release --features cuda
python3 benchmarks/run_suite.py --id llama31_8b_q4km --backend cuda \
  --host-label "Vast RTX4090 / driver XXX" \
  --ferrox-bin ./target/release/ferrox-server
```

Staged goals: first ≥0.5× llama.cpp predicted tok/s on Llama-8B Q4_K_M,
then parity. Always record GPU/driver in `--host-label`. Env:
`FERROX_CUDA=1`, `FERROX_CUDA_GQA=1`, optional `FERROX_CUDA_GRAPH=1`.
See [`docs/ROADMAP.md`](../docs/ROADMAP.md).
