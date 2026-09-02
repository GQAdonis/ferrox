<div align="center">

<img src="docs/assets/ferrox-logo.webp" alt="Ferrox" width="70%" />

**A pure-Rust GGUF inference engine. Dense and MoE, on CPU, Apple Metal, or CUDA.**

[![CI][ci-badge]][ci-workflow]
[![Latest release][release-badge]][latest-release]
[![crates.io][crates-badge]][crates-url]
[![docs.rs][docs-badge]][docs-url]
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/built_with-Rust-dea584.svg)](https://www.rust-lang.org/)
[![Backends](https://img.shields.io/badge/backends-CPU%20%7C%20Metal%20%7C%20CUDA-64748b.svg)](docs/FEATURES.md)

**[Features](docs/FEATURES.md)** ·
**[Models](docs/MODELS.md)** ·
**[CLI](docs/CLI.md)** ·
**[API](docs/API.md)** ·
**[Config](docs/CONFIG.md)** ·
**[Benchmarks](benchmarks/RESULTS.md)** ·
**[Studio UI](ui/)** ·
**[Agents](docs/AGENTS_COOKBOOK.md)** ·
**[Roadmap](docs/ROADMAP.md)** ·
**[Contributing](CONTRIBUTING.md)**

</div>

---

Ferrox loads GGUF checkpoints and runs inference on your hardware. No
bindings to llama.cpp, no ggml wrapper. The loader, quantized kernels,
attention and expert routing are all implemented here.

- **One binary, no runtime.** 19 MB with Metal and the server, 14 MB
  stripped. It runs completions, the API server, `download`, `bench` and
  `verify`. Nothing to activate, no wheels, no CUDA userspace to match
  against a driver. PyTorch alone is 402 MB before vLLM sits on top.
- **Quantized end to end.** Weights stay quantized on mmap and
  dequantize inside the matmul, so an 8B model fits on a laptop.
  K-quants, the IQ tiers, MXFP4, F16 and BF16.
- **Mixture-of-experts has its own path.** GPU routing, indexed expert
  GEMMs, residency planning. When the weights do not fit, experts stream
  from the checkpoint through a bounded cache. That is slower than
  keeping them resident, so it never switches on for a model that fits.
- **OpenAI-compatible server.** Continuous batching, paged KV that
  shares a system prompt's pages across conversations, runtime model
  swap, resumable streams, Anthropic and Responses endpoints, and an
  `/admin` surface. Point your existing client at it.
- **Structured output, guaranteed.** Ask for a JSON Schema, a GBNF
  grammar, or a specific tool call, and the model cannot emit anything
  else. No retry loop, no validating after the fact, no repair pass.
- **Same flags, same output as llama.cpp.** The sampler runs llama.cpp's
  chain in llama.cpp's order, `min_p` included. Tokenization is verified
  byte-for-byte against it on ten checkpoints.
- **Embeddings from real encoder models.** Point `-m` at a BGE, E5 or GTE
  checkpoint and `/v1/embeddings` serves it, pooled the way the file says
  to. Not a decoder's hidden states borrowed for the job.
- **Prompts framed by the checkpoint's own template.** The GGUF's real
  `tokenizer.chat_template` is compiled and evaluated, never sniffed. A
  family nobody hand-wrote a renderer for is still framed the way it was
  trained.
- **Agent-ready output.** Chain of thought splits into
  `reasoning_content` as tokens arrive, and tool calls are parsed in the
  eleven formats real checkpoints emit, streaming as argument deltas.
- **Speculative decoding stays lossless** at any temperature, not just
  `--temp 0`, and reports per-position accept rate rather than one
  average.
- **Every speed number is measured**, against llama.cpp on the same host
  and file. Runs write JSON receipts that generate
  [the table](benchmarks/RESULTS.md); a busy or thermally limited host
  stops the benchmark instead of publishing a bad number.
- **It refuses rather than guessing.** A model whose maths Ferrox only
  partly implements stops and names what is missing, instead of loading
  and returning fluent text computed the wrong way. The GGUF parser is
  bounded against a hostile file too: every length it reads is checked
  against what the file can actually contain.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/antonellof/ferrox/main/scripts/install.sh | bash
```

Installs `ferrox` and `ferrox-server` into `~/.local/bin` (override with
`FERROX_INSTALL_DIR`, pin with `FERROX_VERSION=v0.14.0`). The downloaded
`ferrox` is built with `serve`, so one binary runs completions and
serves the API. `ferrox-server` ships alongside it so an existing one on
your PATH keeps working. Prebuilts are macOS arm64 with Metal and Linux
x86_64 with CPU.

From crates.io or source instead:

```bash
# One binary, everything: completions, `ferrox serve`, `ferrox download`,
# bench, verify. Use --features cuda on Linux+NVIDIA.
cargo install ferrox-cli --features metal

# Or the server on its own, if you prefer two binaries.
cargo install ferrox-server --features metal

# From source.
cargo build --release -p ferrox-cli --features metal
```

**The only flag you need is your GPU.** `--features metal` on Apple
silicon, `--features cuda` on Linux with an NVIDIA card, nothing on a
CPU-only machine. Everything else is already in: `ferrox serve`,
`ferrox download`, `ferrox bench`, `ferrox verify` and completions all
work out of the box.

Using ferrox as a Rust library rather than a command? Depend on
[`ferrox-inference`](https://crates.io/crates/ferrox-inference), or on
`ferrox-models` / `ferrox-core` for just the engine. None of them pull
in the CLI or the server.

## Quick start

```bash
# 1. Get a model. No Python, no huggingface_hub: same syntax as `hf download`.
ferrox download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models

# 2. Run it. Ferrox evaluates the GGUF's own chat template and wraps your
#    prompt in it. Add --no-cnv for a raw completion.
ferrox -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -p "Explain quantization in two sentences" -n 128 -dev metal -ngl all

# 3. Or serve it on 127.0.0.1:8383 and point any OpenAI client at /v1.
#    `ferrox-server` is the same server standalone, if you prefer two binaries.
ferrox serve -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -dev metal -ngl all &
curl -s -X POST http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'

# 4. Check it against llama.cpp on your own machine.
ferrox bench -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 --compare
```

Prefer `Q4_K_M` day to day and `Q8_0` for small smoke tests.
[docs/MODELS.md](docs/MODELS.md) lists what runs today, and which
checkpoints stop with an error instead.

## Use it as a library

Published as [`ferrox-inference`](https://crates.io/crates/ferrox-inference),
a facade re-exporting the workspace. The name `ferrox` on crates.io
belongs to an unrelated crate.

```toml
[dependencies]
ferrox-inference = "0.14"
```

```rust
use ferrox_inference::gguf::ShardedGguf;
use ferrox_inference::models::{Decoder, ModelConfig};

let path = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

// Read metadata without loading a single weight.
let file = ShardedGguf::open(path)?;
println!("{} tensors", file.tensor_count());

// Hyperparameters come from the file. Anything guessed is listed in
// `config.best_effort_fields`.
let config = ModelConfig::from_gguf(&file)?;
let decoder = Decoder::from_gguf(path, config)?;
```

Features, none on by default: `metal`, `cuda`, `api`. Every layer is
published separately if you want one rather than the stack:
[gguf](https://crates.io/crates/ferrox-gguf),
[quant](https://crates.io/crates/ferrox-quant),
[safetensors](https://crates.io/crates/ferrox-safetensors),
[core](https://crates.io/crates/ferrox-core),
[moe](https://crates.io/crates/ferrox-moe),
[models](https://crates.io/crates/ferrox-models),
[api](https://crates.io/crates/ferrox-api),
[metal](https://crates.io/crates/ferrox-metal),
[cuda](https://crates.io/crates/ferrox-cuda). All share one version.

## AI full disclosure

This software is developed with strong assistance from Cursor, Grok 4.5,
GPT 5.6, and Claude Fable 5. Humans lead the ideas, the testing, and the
debugging. We say this openly because it shaped how the project was
built. If you are not happy with AI-developed code, this software is not
for you. The acknowledgement below matters as much: none of this would
exist without [llama.cpp](https://github.com/ggerganov/llama.cpp) and
GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

Ferrox does not link against GGML. It exists because llama.cpp opened
the path: the kernels, the quantization formats, the GGUF ecosystem, and
years of engineering knowledge worked out there in the open. We are
thankful and indebted to llama.cpp and its contributors. Their
implementation, kernels, tests, and design choices were an essential
reference while this pure-Rust GGUF / MoE inference path was built. Some
source-level pieces are retained or adapted here under the MIT license,
notably the IQ quantization codebook tables. Many other pieces (GGUF
layouts, quant/dot semantics, CLI and server conventions) were written
independently against that public design. For that reason, and because
we are genuinely grateful, we keep the GGML authors' copyright notice in
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

[ci-badge]: https://github.com/antonellof/ferrox/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/antonellof/ferrox/actions/workflows/ci.yml
[release-badge]: https://img.shields.io/github/v/release/antonellof/ferrox?display_name=tag
[latest-release]: https://github.com/antonellof/ferrox/releases/latest
[crates-badge]: https://img.shields.io/crates/v/ferrox-inference.svg
[crates-url]: https://crates.io/crates/ferrox-inference
[docs-badge]: https://docs.rs/ferrox-inference/badge.svg
[docs-url]: https://docs.rs/ferrox-inference
