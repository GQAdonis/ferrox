# CLAUDE.md

Guidance for agents working in this repo.

## What this is

Pure-Rust GGUF / MoE inference engine: mmap loaders, quantized CPU +
Metal + CUDA kernels, OpenAI-compatible `ferrox-server`.

Capabilities: `docs/FEATURES.md`. Models & pins: `docs/MODELS.md`,
`benchmarks/RESULTS.md`. Planned: `docs/ROADMAP.md`.

| Doc | Role |
|---|---|
| `docs/FEATURES.md` | capabilities overview |
| `docs/CLI.md` | `ferrox` flags + `ferrox chat` |
| `docs/MODELS.md` | what runs / what doesn’t |
| `docs/API.md` | OpenAI compatibility matrix |
| `docs/AGENTS_COOKBOOK.md` | point IDEs at `ferrox-server` |
| `docs/CONFIG.md` | env vars |
| `benchmarks/RESULTS.md` | tok/s vs llama.cpp (Gap = llama/ferrox); engine + serving tables |
| `benchmarks/README.md` | how the two benchmark tracks are measured |
| `docs/ROADMAP.md` | planned work |

## Commands

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

cargo build --workspace --features cuda
cargo test -p ferrox-cuda --features cuda -- --ignored   # needs GPU

cargo build -p ferrox-cli -p ferrox-server --features metal
cargo test -p ferrox-metal --features metal -- --ignored   # needs Metal

# Completion (also: ferrox run -m …)
./target/debug/ferrox -m model.gguf -p "Hi" -n 64 --temp 0 --no-cnv
./target/debug/ferrox -m model.gguf -p "Hi" -n 64 --ngl 99   # Metal

./target/debug/ferrox presets | archs | caps | inspect <gguf> | inspect-plan <gguf>
./target/debug/ferrox smoke <preset> | run-kimi <dir>
./target/debug/ferrox chat --url http://127.0.0.1:8383   # needs ferrox-server

FERROX_MODEL_PATH=model.gguf FERROX_ADDR=127.0.0.1:8383 ./target/debug/ferrox-server

# Engine bench vs llama-bench (no HTTP). See benchmarks/README.md
./target/release/ferrox bench -m model.gguf -p 512 -n 128 --compare
./target/release/ferrox bench --suite --fit-host --skip-missing
./target/release/ferrox bench --render        # re-render engine table only

# Serving bench vs llama-server (HTTP, chat template, sampler)
python3 benchmarks/run_suite.py --list
python3 benchmarks/run_suite.py --id llama31_8b_q4km --backend metal
# CUDA host (requires --features cuda binary + GPU):
python3 benchmarks/run_suite.py --id llama31_8b_q4km --backend cuda \
  --host-label "host / GPU / driver"
```

Fixtures and golden values were generated and cross-validated with
independent NumPy references.

Tests are mostly `#[cfg(test)]` next to the code. Integration:
`crates/ferrox-models/tests/gguf_roundtrip.rs`. Never un-ignore CUDA /
Metal hardware tests without a real GPU.

## Architecture

```
ferrox-gguf + ferrox-quant
        → ferrox-core (WeightMatrix, RoPE, GQA, KV; optional cuda/metal)
        → ferrox-moe
        → ferrox-models (loader, Decoder, Kimi/GLM/DS4 stacks)
        → ferrox-cli / ferrox-server
```

Load path: GGUF mmap → keep quantized → fused dequant+dot →
RMSNorm → GQA(+RoPE) → MoE/dense FFN. Serving: `FERROX_MODEL_PATH`
GGUF or Kimi dir; generation on `spawn_blocking`.

Presets `glm_5_2` / `deepseek_v4_pro` / `kimi_k3` are sketches —
not proof of real-checkpoint support. `test_*_fixture` presets match
Python test GGUFs only.
