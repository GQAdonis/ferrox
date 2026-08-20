

**Ferrox** is a pure-Rust inference engine for GGUF models. It runs dense
and MoE checkpoints on CPU, Apple Metal, or CUDA, with a llama.cpp-style
CLI and an OpenAI-compatible HTTP server.

CPU and Metal are the measured backends — every speed claim is pinned
against llama.cpp on the same host and GGUF, in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). CUDA builds and runs
but is held to "must compile": no pinned benchmark host, no published
receipts. Treat a Windows or Linux install as CPU-only in practice
([`docs/FEATURES.md`](docs/FEATURES.md)).

## Quick start

### Install

```bash
curl -fsSL https://raw.githubusercontent.com/antonellof/ferrox/main/scripts/install.sh | bash
```

Installs `ferrox` and `ferrox-server` into `~/.local/bin` (override with
`FERROX_INSTALL_DIR`). Pins a release with `FERROX_VERSION=v0.8.0`.
Prebuilts: macOS arm64 (Metal) and Linux x86_64 (CPU). CUDA / other
hosts: build from source.

### Install from crates.io

```bash
cargo install ferrox-cli      # the `ferrox` binary
cargo install ferrox-server   # the OpenAI-compatible server
```

The library is published as
[`ferrox-inference`](https://crates.io/crates/ferrox-inference) — the
name `ferrox` belongs to an unrelated crate. It is a facade that
re-exports the workspace, so a dependent writes one line:

```toml
[dependencies]
ferrox-inference = "0.8"
```

It deliberately ships no binary: `ferrox-cli` already installs one
called `ferrox`, and two crates writing the same path in
`~/.cargo/bin` would fight over it. GPU features are opt-in
(`metal`, `cuda`) because neither is safe to assume.

### Build from source

```bash
# macOS (Metal + CPU). Drop `--features metal` for CPU-only; use `--features cuda` on Linux+NVIDIA.
cargo build --release -p ferrox-cli -p ferrox-server --features metal
```

After a source build, binaries are `./target/release/ferrox` and
`./target/release/ferrox-server`.

### 1. Download a model

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
(`pip install -U "huggingface_hub[cli]"`), then pick a GGUF:

**Tiny smoke (~1 GB)** — TinyLlama Chat Q8_0:

```bash
mkdir -p models
hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q8_0.gguf --local-dir models
```

**Gemma-4 E2B Instruct (~3 GB)** — dedicated Ferrox engine + `gemma4` BPE tokenizer:

```bash
mkdir -p models
hf download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_M.gguf --local-dir models
```

Other useful GGUFs:


| Model                        | Repo                                                                            | File                                     |
| ---------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------- |
| TinyLlama 1.1B Chat Q8_0     | [TheBloke/…](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)     | `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`     |
| Gemma-4 E2B Instruct Q4_K_M  | [unsloth/…](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)                 | `gemma-4-E2B-it-Q4_K_M.gguf`             |
| Llama 3.2 1B Instruct Q4_K_M | [bartowski/…](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)      | `Llama-3.2-1B-Instruct-Q4_K_M.gguf`      |
| Llama 3.2 3B Instruct Q4_K_M | [bartowski/…](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)      | `Llama-3.2-3B-Instruct-Q4_K_M.gguf`      |
| SmolLM2 135M Instruct Q8_0   | [bartowski/…](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF)      | `SmolLM2-135M-Instruct-Q8_0.gguf`        |


Browse [llama.cpp-compatible models](https://huggingface.co/models?apps=llama.cpp&sort=trending)
on Hugging Face. Prefer `Q4_K_M` for everyday use; `Q8_0` for tiny smokes.
See [docs/MODELS.md](docs/MODELS.md) for what Ferrox supports today.

### 2. Run the CLI

Instruct / chat models (Gemma-4, Llama Instruct, …): **omit** `--no-cnv` so Ferrox
applies the GGUF chat template. Use `--no-cnv` only for raw completion prompts.

```bash
# Gemma-4 chat (Metal). Use -dev none -ngl 0 for CPU.
./ferrox -m models/gemma-4-E2B-it-Q4_K_M.gguf \
  -p "How are you?" -n 64 --temp 0 -dev metal -ngl all

# TinyLlama / raw completion (no chat wrap)
./ferrox -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

# Llama 3.2 Instruct + Metal
./ferrox -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p "What is 2+2?" -n 64 --temp 0 -dev metal -ngl all
```

If you built from source, use `./target/release/ferrox` instead of `./ferrox`.

### 3. Start the server

```bash
./ferrox-server \
  -m models/gemma-4-E2B-it-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8383 -dev metal -ngl all &

curl -s -X POST http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":64,"temperature":0}'
```

Swap `-m` for TinyLlama or another GGUF if you prefer a smaller first server smoke.
Use `./target/release/ferrox-server` after a source build.

### 4. Benchmark vs llama.cpp

Same shape as [`llama-bench`](https://github.com/ggerganov/llama.cpp/tree/master/tools/llama-bench):
`pp512` prefill / `tg128` decode, no HTTP. Models live in
[`benchmarks/suite.json`](benchmarks/suite.json).

```bash
# one GGUF (+ optional side-by-side llama-bench)
./target/release/ferrox bench -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p 512 -n 128 -r 3 --compare

# every suite entry that fits this host and has a GGUF on disk
./target/release/ferrox bench --suite --fit-host --skip-missing

# one suite id / backend
./target/release/ferrox bench --suite --id llama32_3b_q4km --backend metal

# rewrite RESULTS.md from existing receipts
./target/release/ferrox bench --render
```

Ledger: [benchmarks/RESULTS.md](benchmarks/RESULTS.md). Details:
[benchmarks/README.md](benchmarks/README.md) · [docs/CLI.md](docs/CLI.md).

## Documentation


| Doc                                                | Description                               |
| -------------------------------------------------- | ----------------------------------------- |
| [docs/FEATURES.md](docs/FEATURES.md)               | Capabilities overview                     |
| [docs/MODELS.md](docs/MODELS.md)                   | Supported models and speed summary        |
| [docs/CLI.md](docs/CLI.md)                         | CLI flags and examples                    |
| [docs/API.md](docs/API.md)                         | OpenAI-compatible API                     |
| [docs/CONFIG.md](docs/CONFIG.md)                   | Environment variables                     |
| [docs/AGENTS_COOKBOOK.md](docs/AGENTS_COOKBOOK.md) | Point IDEs / agents at the server         |
| [benchmarks/RESULTS.md](benchmarks/RESULTS.md)     | Speed vs llama.cpp (`ferrox bench`)       |
| [benchmarks/README.md](benchmarks/README.md)       | How `ferrox bench` / `llama-bench` is run |
| [docs/ROADMAP.md](docs/ROADMAP.md)                 | Planned work                              |
| [CONTRIBUTING.md](CONTRIBUTING.md)                 | How to contribute                         |




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