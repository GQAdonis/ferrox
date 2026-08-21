<div align="center">

<img src="docs/assets/ferrox-logo.webp" alt="Ferrox" width="70%" />


**Ferrox: a pure-Rust GGUF inference engine — dense and MoE, on CPU, Apple Metal, or CUDA.**



[![CI][ci-badge]][ci-workflow]
[![Latest release][release-badge]][latest-release]
[![crates.io][crates-badge]][crates-url]
[![docs.rs][docs-badge]][docs-url]
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/built_with-Rust-dea584.svg)](https://www.rust-lang.org/)
[![Backends](https://img.shields.io/badge/backends-CPU%20%7C%20Metal%20%7C%20CUDA-64748b.svg)](docs/FEATURES.md)

[Install](#install) · [Web UI](#web-ui) · [Models](docs/MODELS.md) · [Benchmarks](benchmarks/RESULTS.md) · [API](docs/API.md) · [Contributing](CONTRIBUTING.md)

</div>

It ships a llama.cpp-style CLI, an OpenAI-compatible HTTP server, and a
web UI. Weights stay quantized on mmap and dequantization happens inside
the matmul, so an 8B model fits on a laptop.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/antonellof/ferrox/main/scripts/install.sh | bash
```

Installs `ferrox` and `ferrox-server` into `~/.local/bin` (override with
`FERROX_INSTALL_DIR`, pin with `FERROX_VERSION=v0.8.0`). Prebuilts:
macOS arm64 (Metal) and Linux x86_64 (CPU).

Or from crates.io / source:

```bash
cargo install ferrox-cli --features metal      # `ferrox`; --features cuda on Linux+NVIDIA
cargo install ferrox-server --features metal   # the HTTP server

cargo build --release -p ferrox-cli -p ferrox-server --features metal
```

Neither install pulls in a GPU backend unless you ask for it. After a
source build the binaries are under `./target/release/`.

## Get a model

```bash
mkdir -p models
hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q8_0.gguf --local-dir models
```

Needs the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
(`pip install -U "huggingface_hub[cli]"`). Prefer `Q4_K_M` for everyday
use, `Q8_0` for small smokes. Good starters: `unsloth/gemma-4-E2B-it-GGUF`,
`bartowski/Llama-3.2-{1B,3B}-Instruct-GGUF`,
`bartowski/SmolLM2-135M-Instruct-GGUF`. What actually works today:
[docs/MODELS.md](docs/MODELS.md).

## Web UI

```bash
ferrox-server -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --ui-server
```

Then open **<http://127.0.0.1:8383/>**.

The UI is compiled into the binary — no Node, no build step, no separate
process. It is **off by default**: without `--ui-server` (or
`FERROX_UI=1`) `/` returns 404 rather than an empty page.

| Screen | What it does |
|---|---|
| **Chat** | Streaming chat with markdown, and the server's own `usage` under each answer: TTFT, prefill tok/s, decode tok/s — measured server-side, not with a client stopwatch |
| **Models** | Everything in your model directory: load, unload, swap the active model, download from Hugging Face with live progress |
| **Activity** | Live request log, keyed by `request_id`. `duration_ms` and `decode_ms` stay separate columns, because conflating them reads a 50 tok/s model as 5 |
| **Connect** | Copy-pasteable curl / Python-SDK / IDE snippets, filled from the live base URL and the model id `/v1/models` currently reports |

Every screen goes through the public HTTP API, so the API cannot rot
without the UI breaking first.

## CLI and server

```bash
# Chat model: omit --no-cnv so the GGUF chat template is applied.
ferrox -m models/gemma-4-E2B-it-Q4_K_M.gguf \
  -p "How are you?" -n 64 --temp 0 -dev metal -ngl all

# Raw completion, no chat wrap.
ferrox -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

# OpenAI-compatible server (default 127.0.0.1:8383).
ferrox-server -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf -dev metal -ngl all &
curl -s -X POST http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'
```

Flags: [docs/CLI.md](docs/CLI.md). API surface: [docs/API.md](docs/API.md).

## Benchmark vs llama.cpp

Same shape as `llama-bench` — `pp512` prefill, `tg128` decode, no HTTP.
Every published number has a receipt, and a run **refuses to start** on a
host too busy for the number to mean anything.

```bash
ferrox bench -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf -p 512 -n 128 -r 3 --compare
ferrox bench --suite --fit-host --skip-missing
```

Ledger: [benchmarks/RESULTS.md](benchmarks/RESULTS.md) ·
method: [benchmarks/README.md](benchmarks/README.md).

## Use it as a library

Published as [`ferrox-inference`](https://crates.io/crates/ferrox-inference)
(the name `ferrox` belongs to an unrelated crate) — a facade that
re-exports the whole workspace:

```toml
[dependencies]
ferrox-inference = "0.8"
```

```rust
use ferrox_inference::gguf::ShardedGguf;
use ferrox_inference::models::{Decoder, ModelConfig};

let path = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

// Read metadata without loading a single weight.
let file = ShardedGguf::open(path)?;
println!("{} tensors", file.tensor_count());

// Hyperparameters come from the file. Anything that had to be guessed
// is listed in `config.best_effort_fields`.
let config = ModelConfig::from_gguf(&file)?;

// Weights stay quantized and mmap-resident; dequant happens inside the
// matmul. This REFUSES a checkpoint carrying tensors this build never
// reads, rather than quietly computing a different graph.
let decoder = Decoder::from_gguf(path, config)?;
```

Features, none on by default: `metal`, `cuda`, `api` (route constants +
wire DTOs, for a client that should not depend on the server).

The individual layers are published too, if you want one rather than the
stack: [`ferrox-gguf`](https://crates.io/crates/ferrox-gguf),
[`ferrox-quant`](https://crates.io/crates/ferrox-quant),
[`ferrox-safetensors`](https://crates.io/crates/ferrox-safetensors),
[`ferrox-core`](https://crates.io/crates/ferrox-core),
[`ferrox-moe`](https://crates.io/crates/ferrox-moe),
[`ferrox-models`](https://crates.io/crates/ferrox-models),
[`ferrox-api`](https://crates.io/crates/ferrox-api),
[`ferrox-metal`](https://crates.io/crates/ferrox-metal),
[`ferrox-cuda`](https://crates.io/crates/ferrox-cuda). All share one
version number.

## Documentation

| Doc | Description |
| --- | --- |
| [docs/FEATURES.md](docs/FEATURES.md) | Capabilities overview |
| [docs/MODELS.md](docs/MODELS.md) | Supported models, quant coverage, what gets refused |
| [docs/CLI.md](docs/CLI.md) | CLI flags and examples |
| [docs/API.md](docs/API.md) | OpenAI-compatible API |
| [docs/CONFIG.md](docs/CONFIG.md) | Environment variables |
| [docs/AGENTS_COOKBOOK.md](docs/AGENTS_COOKBOOK.md) | Point IDEs / agents at the server |
| [benchmarks/RESULTS.md](benchmarks/RESULTS.md) | Speed vs llama.cpp |
| [benchmarks/README.md](benchmarks/README.md) | How it is measured |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Planned work |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

## AI full disclosure

This software is developed with strong assistance from Cursor, Grok 4.5, GPT 5.6, and Claude Fable 5, with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without [llama.cpp](https://github.com/ggerganov/llama.cpp) and
GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

Ferrox does not link against GGML, but it exists thanks to the path
opened by the llama.cpp project and the kernels, quantization formats,
GGUF ecosystem, and hard-won engineering knowledge developed there. We
are thankful and indebted to llama.cpp and its contributors. Their
implementation, kernels, tests, and design choices were an essential
reference while building this pure-Rust GGUF / MoE inference path. Some
source-level pieces are retained or adapted here under the MIT license —
notably IQ quantization codebook tables — and many other pieces (GGUF
layouts, quant/dot semantics, CLI and server conventions) were written
independently against that public design. For this reason, and because
we are genuinely grateful, we keep the GGML authors' copyright notice in
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

## License

Apache-2.0 — see [LICENSE](LICENSE) and
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

[ci-badge]: https://github.com/antonellof/ferrox/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/antonellof/ferrox/actions/workflows/ci.yml
[release-badge]: https://img.shields.io/github/v/release/antonellof/ferrox?display_name=tag
[latest-release]: https://github.com/antonellof/ferrox/releases/latest
[crates-badge]: https://img.shields.io/crates/v/ferrox-inference.svg
[crates-url]: https://crates.io/crates/ferrox-inference
[docs-badge]: https://docs.rs/ferrox-inference/badge.svg
[docs-url]: https://docs.rs/ferrox-inference
