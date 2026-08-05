<p align="center">
  <img src="docs/assets/ferrox-logo.png" alt="Ferrox — pure-Rust GGUF / MoE inference" width="520">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
</p>

A pure-Rust inference engine for GGUF models — dense and Mixture-of-Experts —
with quantized CPU, Apple Metal, and CUDA backends, a llama.cpp-style CLI,
and an OpenAI-compatible HTTP server.

- **Zero-copy loading** — GGUF weights are mmapped and stay quantized;
  dequantization is fused into the dot products.
- **Fast where it counts** — benchmarked against llama.cpp on the same
  host, backend, and GGUF. CPU decode meets or beats llama.cpp on most
  tested models; Metal is at parity for the Llama family and ahead for
  Qwen. Every claim has a pinned receipt in
  [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
- **Broad architecture support** — Llama 3.x, TinyLlama, SmolLM2,
  Qwen2.5/Qwen3 (QKV bias, per-head QK-norm), Gemma-3 (GeGLU, sliding-window
  attention, sandwich norms), Phi-3 (fused QKV/FFN), OLMoE (MoE). See
  [docs/MODELS.md](docs/MODELS.md).

## Quick start

```bash
cargo build --release -p ferrox-cli -p ferrox-server --features metal
```

### CLI

```bash
# One-shot completion
./target/release/ferrox -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

# Chat (default when the GGUF ships a chat template), on Metal
./target/release/ferrox -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p "What is 2+2?" -n 64 --temp 0 -dev metal -ngl all
```

Flags mirror llama.cpp (`-m`, `-p`, `-n`, `-t`, `--temp`, `-ngl`, …) —
full reference in [docs/CLI.md](docs/CLI.md).

### Server (OpenAI-compatible API)

```bash
./target/release/ferrox-server \
  -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --host 127.0.0.1 --port 8383 -dev metal -ngl all &

curl -s -X POST http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"temperature":0}'
```

Good first models: TinyLlama Q8_0, SmolLM2-135M Q8_0, Llama-3.1-8B Q4_K_M.

## Status

| Area | State |
|---|---|
| Dense GQA (CPU / Metal) | Verified — TinyLlama, Llama 3.1/3.2; Llama-8B Metal fair-chat Gap ~1.03× (CLI 1.00×) |
| Qwen2.5 / Qwen3 / SmolLM2 | Verified — CPU strong; Metal ahead of llama.cpp |
| Gemma-3 / Phi-3-mini | Verified — full Metal stack; ~0.7–0.8× llama.cpp decode (legacy attn dims) |
| MoE (CPU) | Verified — OLMoE matches llama.cpp; CUDA historically, no current pin |
| Kimi / GLM / DeepSeek | Partial — primitives and synthetic stacks, no frontier checkpoint end-to-end |
| CUDA performance | Deferred — compiles and runs; fair-chat tuning paused |

Benchmark methodology and receipts: [benchmarks/RESULTS.md](benchmarks/RESULTS.md)
(`python3 benchmarks/run_suite.py`).

## Documentation

| Doc | Contents |
|---|---|
| [docs/CLI.md](docs/CLI.md) | CLI flags and examples |
| [docs/MODELS.md](docs/MODELS.md) | Supported models and verification status |
| [docs/API.md](docs/API.md) | OpenAI-compatible API matrix |
| [docs/CONFIG.md](docs/CONFIG.md) | Environment variables and tuning |
| [benchmarks/RESULTS.md](benchmarks/RESULTS.md) | Pinned tok/s vs llama.cpp |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Planned work |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

## Project layout

```
crates/      ferrox-{gguf,quant,core,moe,models,cli,server,cuda,metal,…}
docs/        CLI, MODELS, CONFIG, ROADMAP, architecture manifest
benchmarks/  suite runner + pinned results
```

## License

Apache-2.0 — see [LICENSE](LICENSE) and
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).
