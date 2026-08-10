<p align="center">
  <img src="docs/assets/ferrox-logo.png" alt="Ferrox — pure-Rust GGUF / MoE inference" width="520">
</p>


**Ferrox** is a pure-Rust inference engine for GGUF models. It runs dense
and MoE checkpoints on CPU, Apple Metal, or CUDA, with a llama.cpp-style
CLI and an OpenAI-compatible HTTP server.

## Quick start

```bash
cargo build --release -p ferrox-cli -p ferrox-server --features metal
```

### 1. Download a model

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
(`pip install -U huggingface_hub`), then:

```bash
mkdir -p models
hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q8_0.gguf --local-dir models
```

Other useful GGUFs:


| Model                        | Repo                                                                            | File                                     |
| ---------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------- |
| TinyLlama 1.1B Chat Q8_0     | [TheBloke/…](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)     | `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`     |
| Llama 3.2 1B Instruct Q4_K_M | [bartowski/…](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)      | `Llama-3.2-1B-Instruct-Q4_K_M.gguf`      |
| Llama 3.1 8B Instruct Q4_K_M | [bartowski/…](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` |
| SmolLM2 135M Instruct Q8_0   | [bartowski/…](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF)      | `SmolLM2-135M-Instruct-Q8_0.gguf`        |


Browse [llama.cpp-compatible models](https://huggingface.co/models?apps=llama.cpp&sort=trending)
on Hugging Face. Prefer `Q4_K_M` for everyday use; `Q8_0` for tiny smokes.

### 2. Run the CLI

```bash
./target/release/ferrox -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

# Chat template + Metal
./target/release/ferrox -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p "What is 2+2?" -n 64 --temp 0 -dev metal -ngl all
```



### 3. Start the server

```bash
./target/release/ferrox-server \
  -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --host 127.0.0.1 --port 8383 -dev metal -ngl all &

curl -s -X POST http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"temperature":0}'
```



## Documentation


| Doc                                                | Description                               |
| -------------------------------------------------- | ----------------------------------------- |
| [docs/FEATURES.md](docs/FEATURES.md)               | Capabilities overview                     |
| [docs/MODELS.md](docs/MODELS.md)                   | Supported models and benchmarks           |
| [docs/CLI.md](docs/CLI.md)                         | CLI flags and examples                    |
| [docs/API.md](docs/API.md)                         | OpenAI-compatible API                     |
| [docs/CONFIG.md](docs/CONFIG.md)                   | Environment variables                     |
| [docs/AGENTS_COOKBOOK.md](docs/AGENTS_COOKBOOK.md) | Point IDEs / agents at the server         |
| [benchmarks/RESULTS.md](benchmarks/RESULTS.md)     | Speed vs llama.cpp (engine + serving)     |
| [benchmarks/README.md](benchmarks/README.md)       | How the two benchmark tracks are measured |
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