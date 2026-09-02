# CLAUDE.md

Guidance for agents working in this repo.

## What this is

Pure-Rust GGUF / MoE inference engine: mmap loaders, quantized CPU +
Metal + CUDA kernels, OpenAI-compatible `ferrox-server`.

**The goal is to be the Rust alternative to llama.cpp**: same models,
same command shapes, same or better performance, on the hardware people
actually own. `docs/plans/north-star.md` is the ranking every other plan
is read through, and `docs/plans/README.md` is the index.

Honest position, re-audited 2026-09-02. **16** architectures run with
evidence (`capability::AUDITED_GENERIC_GQA`), 4 more have dedicated
engines, and everything else REFUSES. The "loads and is WRONG" class is
closed: the generic path is opt-in, so an unaudited architecture stops
instead of guessing.

The 41 unaudited refusals are now TRIAGED, and the refusal says which of
three things is missing: 9 are a fixture away (implemented, unevidenced),
2 need one named match arm, 26 need new code, 4 are unknown with the
question stated. Five one-match-arm rows were closed on 2026-09-02, each
with a libllama-golden fixture, which is what moved 46 to 41.
`unaudited_triage` carries the verdict and the llama.cpp line that
decides it. llama.cpp hand-writes 140
per-architecture graphs; `decoder.rs` is 6702 lines and that is why the
counts differ.

Do not read the architecture catalog as a support matrix. `ferrox
parity` is the oracle: its tokenizer half matches llama.cpp on every
local checkpoint libllama can load, and its logit half MATCHES on
Q8_0/IQ4_NL while DRIFTING on K-quants — for a known reason that is not
a ferrox bug (`docs/plans/llama-cpp-gap-inventory.md` §10).

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

## How to write code here

**Keep files small, modules narrow, and the binary light.** This is not
style preference, it is the repo's most expensive lesson. Measured
2026-09-01, and every one of these GREW since the last measurement:

| File | Lines |
|---|---|
| `ferrox-server/src/lib.rs` | 9580 |
| `ferrox-metal/src/attn.rs` | 8860 |
| `ferrox-metal/src/gpu.rs` | 8700 |
| `ferrox-quant/src/lib.rs` | 8230 |
| `ferrox-models/src/decoder.rs` | 6702 |

Re-measured 2026-09-02. `ferrox-server/src/lib.rs` took the top spot by
GROWING 1839 lines in a day, which is the rule being broken while the
rule is written down two paragraphs below. `decoder.rs` shrank for the
first time (6875 to 6702), because the Gemma RoPE work went into
`decoder/rope.rs` instead of a sixth branch.

Those files are why llama.cpp has 140 architectures and ferrox has 16
proven. Adding a model means editing a 6700-line file, so nobody adds
one. The same decode layer used to be written out about ELEVEN times
across `decoder.rs` and `attn.rs`, which has already lost EIGHT model
features one at a time, each silently:
`attention_scale`, `post_attn_norm`, `post_ffn_norm`, gpt-oss `o_bias`,
`gpt_oss_ffn`, the four the Metal MoE decode stack ignored, and the
four-way drift of the GPU-router eligibility check, where the prefill
sites tested three conditions, the fused decode two and the whole-stack
decode NONE. A copy diverges from its original and nothing notices.

**THIS IS THE DOMINANT BUG SHAPE IN THIS REPO, and 2026-09-01 found a
dozen more instances of it in one day** — not all of them copied code.
The general form is TWO STRUCTURES THAT MUST AGREE ABOUT ONE THING,
WITH NOTHING ENFORCING IT: two spellings of a GGUF key (`unsupported_
feature_keys` gated on one no converter writes, so it never fired); two
copies of a default (the response cache restated every `unwrap_or`
independently of the sampler); four hand-written `SamplingParams`
literals in one file; three tables that had to agree about a Metal
kernel's threadgroup geometry (a correct kernel returned zeros for half
its rows); a wire struct and a sampler that silently disagreed about
which fields exist (SIX parameters accepted and ignored).

The durable fixes are never the individual patches. They are the places
where disagreement now fails to COMPILE or turns a test red: an
exhaustive destructure with no `..`, one predicate the four call sites
share, a derived table instead of a restated one, and a test asserting
every refused key is one a converter actually writes.

Rules that follow from that:

- **A new file beats a new section.** If a change would push a file past
  roughly 1000 lines, split it first, then make the change.
- **One concept per module.** A module named after a noun that holds
  three unrelated things is three modules.
- **Never copy a code path to vary it.** Parameterise the original. The
  precedent that works: `forward_multi_seq` takes a `MultiSeqKv`
  parameter rather than having a paged twin. The precedent that failed:
  `forward_token_paged` was a copy, and lost five features -- it is now
  collapsed onto one `attn_block` taking a `KvStep`.
- **Dead code is a liability, not an asset.** Delete on sight unless it
  serves a named roadmap theme; if it does, wire it or say where it is
  going. `ferrox-edge` was 5,400 uncalled lines and is now dissolved.
- **A gate that cannot fire is worse than no gate**, because it reads as
  coverage. Check that a refusal's condition is reachable at all: this
  repo shipped one keyed on a GGUF spelling nothing writes.
- **No new crate for something one crate uses.** `ferrox-edge` became a
  crate instead of an integration and half of it was never called.

Rust specifics this repo holds to:

- `cargo clippy --workspace --all-targets -- -D warnings` is a gate, not
  advice. Also run it `--release`: `debug_assert!` type-checks its
  argument in release, and a `#[cfg(debug_assertions)]` method called
  from one broke every release build while all of CI stayed green.
- Prefer borrowing to cloning on any path that runs per token. Hoist
  feature probes out of loops: `is_aarch64_feature_detected!` ran 131k
  times in one Mistral-7B projection before it was hoisted.
- `unsafe` needs a `// SAFETY:` comment stating the invariant, and a
  scalar twin it is checked against. Every SIMD arm here has one.
- Return `Result` and name what is missing. A model this engine only
  partly implements must STOP, never compute something else. A refusal
  is coverage, not a defect.
- Tests live in `#[cfg(test)]` beside the code. A test that cannot fail
  is not a test: sabotage it once and confirm it goes red.

## Architecture

```
ferrox-gguf + ferrox-quant
        → ferrox-core (WeightMatrix, RoPE, GQA, KV; optional cuda/metal)
        → ferrox-moe
        → ferrox-models (loader, Decoder, Kimi/GLM/DS4 stacks)
        → ferrox-cli / ferrox-server

ferrox-api  (routes + wire DTOs, serde-only) → ferrox-server + clients
```

**The FreeToken port** (Apache-2.0; see `docs/THIRD_PARTY_NOTICES.md`,
which is a licence obligation and must stay accurate) is a Rust port of
FreeToken's host-side decision logic. It used to be a crate of its own,
`ferrox-edge`; it now lives in the crates that use it, and the parts
nothing would ever use are deleted.

- **`ferrox-core`** holds the MoE expert-residency half, beside
  `expert_store`: `expert_cache`, `expert_slots` (the `SlotDevice`
  seam), `expert_pool` (its CUDA implementation), `expert_budget`,
  `qstar`, `bench_profile`, `residency`, `placement`. `expert_store` is
  the SINGLE holder of the expert byte budget -- on unified memory two
  budgets are the same RAM counted twice.
- **`ferrox-server::policy`** holds the serving half: the two parsers,
  the radix prefix cache, anchor/window slide, scheduler, serving stats,
  maintenance, pool, rebuild, outbox, footprint, effort probing.

Wired today: the parsers, the stop-string withhold rule, the radix
prefix cache over paged KV, effort probing, stats, maintenance, outbox,
footprint, and the scheduler's status reporting. STILL GROUNDWORK, and
this is the gap that matters most: the whole `ferrox-core` expert
residency stack holds the policy for running a model larger than memory
(`docs/plans/out-of-core-moe.md`) and nothing executes it except a
compile-only CUDA pool whose hardware test is `#[ignore]`d.

Anything in `policy` with an unwired half names the roadmap item that
would close it, at its declaration in `policy/mod.rs`. That
`allow(dead_code)` list is meant to be read as a to-do, not as cover.

Load path: GGUF mmap → keep quantized → fused dequant+dot →
RMSNorm → GQA(+RoPE) → MoE/dense FFN. Serving: `FERROX_MODEL_PATH`
GGUF or Kimi dir; generation on `spawn_blocking`.

Presets `glm_5_2` / `deepseek_v4_pro` / `kimi_k3` are sketches,
not proof of real-checkpoint support. `test_*_fixture` presets match
Python test GGUFs only.
