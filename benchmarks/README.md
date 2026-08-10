# Benchmarks

Ledger: **[`RESULTS.md`](RESULTS.md)** (generated — do not hand-edit).

There are **two** numbers here and they answer different questions. Keep
them apart: a change that moves one need not move the other, and neither
may be quoted as the other.

| | Engine | Serving |
|---|---|---|
| Measures | kernels, alone | what a `ferrox-server` user gets |
| Driver | `ferrox bench` (Rust) | [`run_suite.py`](run_suite.py) |
| Compared against | `llama-bench` | `llama-server` |
| Also in the loop | nothing | HTTP, chat template, tokenizer, sampler, SSE |
| Receipts | [`receipts/engine/`](receipts/engine/) | [`receipts/pins/`](receipts/pins/) |
| Workload | `pp512` / `tg128`, synthetic tokens | 80-capitals chat prompt, 512 max tokens |

Host B: Apple M2 Pro, 6 performance + 4 efficiency cores, 32 GiB unified.

Suite definition for both: [`suite.json`](suite.json).

## Engine (`ferrox bench`)

```bash
cargo build -p ferrox-cli --release --features metal

# one model, with the llama.cpp comparison run for you
./target/release/ferrox bench -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p 512 -n 128 -r 3 --compare

# every suite.json entry, both backends, then re-render RESULTS.md
./target/release/ferrox bench --suite --fit-host --skip-missing

# re-render the engine table from existing receipts, measuring nothing
./target/release/ferrox bench --render
```

`--suite` reads [`suite.json`](suite.json) — the same file the serving
runner uses — and runs each entry in a **fresh child process**. That is
required, not tidiness: backend selection reads process-global
environment and the rayon pool is built once, so benchmarking several
backends inside one process would silently measure the first one's
configuration for all of them.

`--fit-host` skips entries whose `estimated_ram_gb` exceeds ~75% of
physical RAM, and skips `cuda` on darwin. `--skip-missing` skips GGUFs
absent from `models/`.

Receipts land in [`receipts/engine/`](receipts/engine/) as
`{id}_{backend}.json`, and `--render` splices the engine table into
`RESULTS.md` between HTML markers, leaving the serving tables alone.

### Thread counts are not forced, on purpose

Neither engine is pinned to a thread count. This is a correction, not an
oversight: llama.cpp defaults to `hw.perflevel0.physicalcpu` (6 here) and
degrades sharply above it — it splits each graph node into equal static
slices, so the 4 efficiency cores stall every barrier. On Host B,
`llama-bench` on SmolLM2-135M Q8_0 measures 346 tok/s at `-t 4` and 176
at `-t 10`.

This suite used to force `-t 10` on both engines, which handicapped
llama.cpp by 2–4× and flattered ferrox. **CPU comparisons predating that
fix are not usable.** Each engine choosing its own default is the
comparison that means something.

### Run-to-run variance is ±20%

Sequential runs of the *same* binary spread ±20% on this host — SmolLM2
`pp512` measured 205, 284 and 306 on three consecutive runs. A single
before/after pair cannot resolve a change smaller than that.

For anything under ~20%, **interleave the two binaries** round by round
in one session and count rounds won, instead of comparing two batches:

```bash
for round in 1 2 3 4; do
  for bin in ./ferrox-base ./ferrox-new; do
    $bin bench -m model.gguf -p 512 -n 0 -r 3
  done
done
```

## Serving (`run_suite.py`)

Drives `ferrox-server` and `llama-server` over HTTP with the same chat
prompt and template. Still Python because it is process orchestration
and HTTP plumbing, not measurement.

```bash
cargo build -p ferrox-server -p ferrox-cli --release --features metal
mkdir -p target/bench
cp target/release/ferrox-server target/bench/ferrox-server-metal
cp target/release/ferrox target/bench/ferrox-cli-metal

python3 benchmarks/run_suite.py --list
python3 benchmarks/run_suite.py --backend metal --skip-missing --fit-host
python3 benchmarks/run_suite.py --id tinyllama_q8 --backend cpu

# regenerate the serving tables only
python3 benchmarks/render_results.py
```

CPU runs interleave llama→ferrox each rep with both servers warm. Metal
stays sequential so two GPU-resident servers do not contend.

**Its prompt is ~30 tokens**, so its `prompt_per_second` is noise and it
cannot see prefill at a size where a GEMM matters. Read prefill off the
engine table instead.

There is also [`cb_throughput.py`](cb_throughput.py), a continuous-
batching throughput smoke (concurrent vs sequential requests against a
server started with `FERROX_CONTINUOUS_BATCHING=1`).

## Gap convention (both tables)

`Gap` = `llama / ferrox`.

| Gap | Meaning |
|---|---|
| &lt; 1.0 | ferrox faster |
| ~1.00× | parity (within ~5%) |
| &gt; 1.0 | ferrox slower |

Prose elsewhere ("1.56× faster") is the inverse when ferrox wins. Never
quote a ratio without a receipt under [`receipts/`](receipts/).

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
| `FERROX_CPU_INT_DOT` | **on by default** in both binaries; `0` opts out |
| `FERROX_CPU_THREADS` | unset = performance cores (6 on Host B) |
| `FERROX_TOKIO_WORKERS` | server async workers (default 2) |
| `FERROX_CUDA_GQA` / `FERROX_CUDA_GRAPH` | CUDA path (not Host B) |

Full list: [`docs/CONFIG.md`](../docs/CONFIG.md).

## CUDA

Host B is Metal/CPU only. On a CUDA host:

```bash
cargo build -p ferrox-server -p ferrox-cli --release --features cuda
./target/release/ferrox bench -m model.gguf --n-gpu-layers 99 --compare
python3 benchmarks/run_suite.py --id llama32_3b_q4km --backend cuda \
  --host-label "Vast RTX4090 / driver XXX"
```

Always record GPU/driver in `--host-label`. See
[`docs/ROADMAP.md`](../docs/ROADMAP.md).
