# CLAUDE.md

Guidance for agents working in this repo.

## What this is

Pure-Rust GGUF / MoE inference engine: mmap loaders, quantized CPU +
Metal + CUDA kernels, OpenAI-compatible `ferrox-server`.

**The goal is to be the Rust alternative to llama.cpp**: same models,
same command shapes, same or better performance, on the hardware people
actually own. `docs/plans/north-star.md` is the ranking every other plan
is read through, and `docs/plans/README.md` is the index.

Honest position, audited 2026-08-27: of 150 catalog rows, about **12**
run with evidence, ~90 refuse while naming what is missing, ~40 are
unproven, and a handful load and are WRONG. llama.cpp hand-writes 140
per-architecture graphs; `decoder.rs` is 6438 lines and that is why the
counts differ. Do not read the architecture catalog as a support
matrix.

Capabilities: `docs/FEATURES.md`. Models & speed ledger: `docs/MODELS.md`,
`benchmarks/RESULTS.md`. Planned: `docs/ROADMAP.md`.

| Doc | Role |
|---|---|
| `docs/FEATURES.md` | capabilities overview |
| `docs/CLI.md` | `ferrox` flags + `ferrox chat` |
| `docs/MODELS.md` | what runs / what doesn’t |
| `docs/API.md` | OpenAI compatibility matrix |
| `docs/AGENTS_COOKBOOK.md` | point IDEs at `ferrox-server` |
| `docs/CONFIG.md` | env vars |
| `benchmarks/RESULTS.md` | tok/s vs llama.cpp (Gap = llama/ferrox); `ferrox bench` ledger |
| `benchmarks/README.md` | how `ferrox bench` / `llama-bench` is measured |
| `docs/ROADMAP.md` | planned work |
| `docs/plans/README.md` | **the plan index and priority order** |
| `docs/plans/north-star.md` | the goal, and how plans are ranked against it |

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

# Bench vs llama-bench (no HTTP). Models: benchmarks/suite.json
./target/release/ferrox bench -m model.gguf -p 512 -n 128 --compare
./target/release/ferrox bench --suite --fit-host --skip-missing
./target/release/ferrox bench --suite --id llama32_3b_q4km --backend metal
./target/release/ferrox bench --render
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

ferrox-api  (routes + wire DTOs, serde-only) → ferrox-server + clients
ferrox-edge (serving policy, tensor-free)    → ferrox-server
                                             → ferrox-cuda (`cuda` only)
```

`ferrox-edge` is a Rust port of FreeToken's host-side decision logic
(Apache-2.0; see `docs/THIRD_PARTY_NOTICES.md`): the `q*` bandwidth
split, the global expert cache, page-keyed radix prefix caches
(plain / sliding-window / recurrent), pool budgets, admission and
chunked prefill, and the reasoning / tool-call output parsers. It has
no tensors and no device memory, so every policy in it is testable on
any host. Wired in today: the two parsers, the stop-string withhold rule, the
radix prefix cache over paged KV, the scheduler, stats, maintenance,
pool, rebuild, outbox, footprint and effort probing. STILL GROUNDWORK,
and this is the gap that matters most: `expert_cache`, `expert_slots`,
`placement` and `residency` hold the policy for running a model larger
than memory, and nothing executes it except a compile-only CUDA pool.

`ferrox-edge::expert_slots` is where that policy meets real memory: it
executes the expert cache's copy plans against a bounded slot pool
behind a `SlotDevice` trait, which is why `ferrox-cuda` depends on it
under `--features cuda` (`expert_pool::CudaExpertPool`). The trait
keeps the device memory out of ferrox-edge; the CUDA pool is
compile-verified only, and its hardware test is `#[ignore]`d.

Load path: GGUF mmap → keep quantized → fused dequant+dot →
RMSNorm → GQA(+RoPE) → MoE/dense FFN. Serving: `FERROX_MODEL_PATH`
GGUF or Kimi dir; generation on `spawn_blocking`.

Presets `glm_5_2` / `deepseek_v4_pro` / `kimi_k3` are sketches,
not proof of real-checkpoint support. `test_*_fixture` presets match
Python test GGUFs only.
