# Benchmarks

The published table is **[`RESULTS.md`](RESULTS.md)**. It is generated,
so never edit it by hand.

Every timed run writes one small JSON file of raw numbers into
[`receipts/engine/`](receipts/engine/), named `{id}_{backend}.json`.
Those files are checked in, and `RESULTS.md` is rendered from them. The
rest of this repo calls them receipts.

The setup follows llama.cpp's [`llama-bench`](https://github.com/ggerganov/llama.cpp/tree/master/tools/llama-bench):
a native CLI, the `pp512` and `tg128` workloads, compared against
`llama-bench` on the same GGUF. No HTTP, no chat template, no tokenizer
and no sampler in the loop.

| | |
|---|---|
| Measures | kernels alone |
| Driver | `ferrox bench` (Rust) |
| Compared against | `llama-bench` |
| Raw numbers | [`receipts/engine/`](receipts/engine/) |
| Workload | `pp512` / `tg128`, synthetic tokens |

Host B: Apple M2 Pro, 6 performance and 4 efficiency cores, 32 GiB
unified memory.

Suite definition: [`suite.json`](suite.json).

## Run

```bash
cargo build -p ferrox-cli --release --features metal

# one model, with the llama.cpp comparison run for you
./target/release/ferrox bench -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p 512 -n 128 -r 3 --compare

# every suite.json entry × backends, then re-render RESULTS.md
./target/release/ferrox bench --suite --fit-host --skip-missing

# one suite id (or one backend across the suite)
./target/release/ferrox bench --suite --id llama32_3b_q4km --backend metal
./target/release/ferrox bench --suite --backend cpu --fit-host --skip-missing

# re-render RESULTS.md from existing receipts, measuring nothing
./target/release/ferrox bench --render
```

`--suite` reads [`suite.json`](suite.json) and runs each entry in a
**fresh child process**. That is required rather than tidy. Backend
selection reads process-global environment, and the rayon pool is built
once, so benchmarking several backends inside one process would measure
the first one's configuration for all of them and say nothing about
it.

`--fit-host` skips entries whose `estimated_ram_gb` exceeds ~75% of
physical RAM, and skips `cuda` on darwin. `--skip-missing` skips GGUFs
absent from `models/`.

Each run drops its numbers in [`receipts/engine/`](receipts/engine/)
as `{id}_{backend}.json`. `--render` rewrites the engine table in
`RESULTS.md` between HTML markers and leaves the Open notes alone.

### Adding a model

Append an object to `models[]` in [`suite.json`](suite.json):

```json
{
  "id": "my_model_q4km",
  "name": "My-Model Q4_K_M",
  "gguf": "models/My-Model-Q4_K_M.gguf",
  "backends": ["metal", "cpu"],
  "estimated_ram_gb": 4,
  "expect": "ok",
  "notes": "optional"
}
```

Put the GGUF under `models/` (path is repo-relative), then:

```bash
./target/release/ferrox bench --suite --id my_model_q4km --fit-host --skip-missing
```

### Thread counts are not forced, on purpose

Neither engine is pinned to a thread count. That is a correction, not
an oversight. llama.cpp defaults to `hw.perflevel0.physicalcpu` (6 on
this host) and degrades sharply above it, because it splits each graph
node into equal static slices and the 4 efficiency cores then stall
every barrier. On Host B, `llama-bench` on SmolLM2-135M Q8_0 measures
346 tok/s at `-t 4` and 176 at `-t 10`.

This suite used to force `-t 10` on both engines, which handicapped
llama.cpp by 2–4× and flattered ferrox. **CPU comparisons taken before
that fix are not usable.** Letting each engine choose its own default is
the comparison that means something.

### Run-to-run variance is ±20%

Sequential runs of the *same* binary spread ±20% on this host.
SmolLM2 `pp512` measured 205, 284 and 306 on three consecutive runs.
One before/after pair will not resolve a change smaller than that.

For anything under ~20%, **interleave the two binaries** round by round
in one session and count rounds won, instead of comparing two batches:

```bash
for round in 1 2 3 4; do
  for bin in ./ferrox-base ./ferrox-new; do
    $bin bench -m model.gguf -p 512 -n 0 -r 3
  done
done
```

## Gap convention

`Gap` = `llama / ferrox`.

| Gap | Meaning |
|---|---|
| &lt; 1.0 | ferrox faster |
| ~1.00× | parity (within ~5%) |
| &gt; 1.0 | ferrox slower |

Prose elsewhere ("1.56× faster") states the inverse when ferrox wins.
Never quote a ratio that has no matching file under
[`receipts/engine/`](receipts/engine/).

## Env (ferrox)

| Var | Typical |
|---|---|
| `FERROX_METAL` | `1` (Metal), `0` for CPU runs |
| `FERROX_METAL_ATTN` | `1` |
| `FERROX_METAL_LOGITS` | unset (host lm_head, `1` = slower vocab-in-stack) |
| `FERROX_METAL_GREEDY_GPU` | default **on** for `temperature<=0`, `0` = host lm_head |
| `FERROX_METAL_FA_VEC` | default **on** for `head_dim` in {64,96,128,256}, `0` = legacy GQA |
| `FERROX_METAL_MUL_MM` | default **on** for prefill batch ≥ 4, `0` = N× matvec |
| `FERROX_METAL_WEIGHT_COPY` | unset (`BytesNoCopy`), `1` forces copy upload |
| `FERROX_CPU_INT_DOT` | **on by default** in both binaries, `0` opts out |
| `FERROX_CPU_THREADS` | unset = performance cores (6 on Host B) |
| `FERROX_CUDA_GQA` / `FERROX_CUDA_GRAPH` | CUDA path (not Host B) |

Full list: [`docs/CONFIG.md`](../docs/CONFIG.md).

## CUDA

Host B is Metal/CPU only. On a CUDA host:

```bash
cargo build -p ferrox-cli --release --features cuda
./target/release/ferrox bench -m model.gguf --n-gpu-layers 99 --compare
./target/release/ferrox bench --suite --fit-host --skip-missing --backend cuda
```

Record the GPU and driver whenever you quote a CUDA number. There is
no pinned CUDA host here and no CUDA files under
[`receipts/engine/`](receipts/engine/), so nothing in `RESULTS.md`
covers it. See [`docs/ROADMAP.md`](../docs/ROADMAP.md).
