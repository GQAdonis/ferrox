# CLI

`ferrox` accepts common [llama.cpp](https://github.com/ggerganov/llama.cpp)
completion flags. Top-level `-m` / `-p` work without typing `run`
(rewritten to `ferrox run …`).

```bash
cargo build --release -p ferrox-cli --features metal   # macOS Metal + CPU
cargo build --release -p ferrox-cli                    # CPU only
```

Binary: `./target/release/ferrox`. One executable covers every backend
compiled in; pick at runtime with `-dev` / `-ngl`.

## Completion (`run`)

### Quick examples

```bash
# Greedy completion (raw prompt)
./target/release/ferrox -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

# Chat-tuned wrap (default when GGUF has tokenizer.chat_template)
./target/release/ferrox -m models/hf_test/SmolLM2-135M-Instruct-Q8_0.gguf \
  -p "What is 2+2?" -n 64 --temp 0 \
  --system "Answer briefly."

# Prompt from file + escapes
./target/release/ferrox -m model.gguf -f prompt.txt -e -n 128

# Sampling
./target/release/ferrox -m model.gguf -p "Once upon a time" \
  -n 256 --temp 0.8 --top-k 40 --top-p 0.95 --repeat-penalty 1.1 -s 42

# Threads + context
./target/release/ferrox -m model.gguf -p "Hi" -n 64 -t 8 -c 4096

# List devices, then select Metal (requires a --features metal build)
./target/release/ferrox --list-devices
./target/release/ferrox -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p "Hello" -n 64 --temp 0 -dev metal -ngl all

# Force CPU with the same Metal-capable executable
./target/release/ferrox -m model.gguf -p "Hello" -n 64 -dev none -ngl 0
```

Same via explicit subcommand: `ferrox run -m …`.

### Completion flags

| Flag | Notes |
|---|---|
| `-m` / `--model` | GGUF path |
| `-p` / `--prompt` | Prompt string |
| `-f` / `--file` | Prompt from file |
| `-n` / `--n-predict` | `-1` = fill remaining context |
| `-c` / `--ctx-size` | `0` = GGUF `{arch}.context_length` (else 4096) |
| `-t` / `--threads` | Sets `RAYON_NUM_THREADS` |
| `--temp` | `0` = greedy |
| `--top-k` | `0` = off |
| `--top-p` | Nucleus sampling |
| `--repeat-penalty` | `1.0` = off |
| `-s` / `--seed` | `-1` = time-based |
| `-dev` / `--device` | `auto`, `none`, `cpu`, `metal`, or `cuda` |
| `--list-devices` | Print compiled, detected devices and exit |
| `-ngl` / `--gpu-layers` / `--n-gpu-layers` | `0`, a number, `auto`, or `all` |
| `--ctk` | KV dtype: `f16` (default), `q8_0`/`turbo8`/`fp8`/`turbo4` (Metal), `turbo3` (falls back) — sets `FERROX_CTK` |
| `--system` | Chat mode only |
| `--no-cnv` | Skip chat-template wrap |
| `-e` / `--escape` | `\n` `\t` `\r` `\\` in `-p` |
| `--ignore-eos` | Always emit up to `-n` |
| `--verbose-prompt` | Print final prompt to stderr |
| `--mtp` | Errors: MTP draft heads not loaded from GGUF yet |

Stderr prints load and throughput timings. Generated text goes to stdout.

**Speculative decoding:** prompt-lookup demo via `ferrox speculative` (no draft model). `--mtp` is reserved for future MiniMax/GLM MTP draft heads (`num_nextn_predict_layers`) and currently errors.

`--device none` (or `cpu`) and `-ngl 0` force CPU. Default is
`--device auto -ngl auto`. Any positive `-ngl`, `auto`, or `all` enables
all supported ops on the selected backend (partial layer placement is
not available yet).

### Chat vs completion

- **Default:** if the GGUF has a recognized `tokenizer.chat_template`, the user
  prompt is wrapped (ChatML / Llama 3 / Gemma / TinyLlama-style markers).
- **`--no-cnv`:** raw prompt (classic completion), BOS still prepended when the
  GGUF defines `bos_token_id`.

## Other commands

```bash
./target/release/ferrox inspect models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
./target/release/ferrox inspect-plan models/olmoe-1b-7b-0924-q4_0.gguf --strict
./target/release/ferrox caps
./target/release/ferrox archs
./target/release/ferrox presets
./target/release/ferrox smoke glm-5.2

# Kimi K3 safetensors directory (large checkpoint; see MODELS.md)
./target/release/ferrox run-kimi /path/to/kimi --prompt "Hi" --max-new-tokens 32
```

## Hugging Face Hub (`pull`)

Download a GGUF via the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (install: `pip install huggingface_hub`):

```bash
./target/release/ferrox pull org/model --file '*.gguf'
# Prints local path; also works as: ferrox -m org/model (auto-download when path missing)
```

Cache default: `~/.cache/ferrox/hf/<org--model>/`.

## Interactive chat (`chat`)

Multi-turn REPL against a running `ferrox-server` (reuses chat-template + SSE):

```bash
FERROX_MODEL_PATH=model.gguf FERROX_ADDR=127.0.0.1:8383 ./target/release/ferrox-server
./target/release/ferrox chat --url http://127.0.0.1:8383 --system "Be brief."
# Commands: /quit  /clear
```

| Flag | Notes |
|---|---|
| `--url` | Server base URL (default `http://127.0.0.1:8383`) |
| `--system` | Optional system message |
| `--max-tokens` / `--temperature` / `--top-p` | Sampling |
| `--no-stream` | Wait for full JSON instead of SSE |

## Server

OpenAI-compatible HTTP API:

```bash
./target/release/ferrox-server \
  -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --host 127.0.0.1 --port 8383 -dev metal -ngl all

# Optional browser UI at / and /ui
./target/release/ferrox-server -m model.gguf --ui-server

# MCP config (metadata in GET /v1/models; invoke not wired)
./target/release/ferrox-server -m model.gguf --mcp-config mcp.json

curl -s -X POST http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"temperature":0}'
```

The server accepts `-m/--model`, `--host`, `--port`, `-t/--threads`,
`-dev/--device`, `-ngl/--n-gpu-layers`, and `--list-devices`. Existing
`FERROX_MODEL_PATH`, `FERROX_ADDR`, and backend environment variables remain
supported; command-line values take precedence. Keep secrets such as
`FERROX_API_KEY` in the environment.

## Benchmark (`ferrox bench`)

With `-m`, `bench` is a [`llama-bench`](https://github.com/ggerganov/llama.cpp/tree/master/tools/llama-bench)
work-alike: same workload names (`pp<N>` batched prefill, `tg<N>` decode),
same reporting (median ± population stddev over `-r` reps, one warmup
discarded), same flag names — so the two outputs can be read side by side.

```bash
# CPU (default). Prints the exact llama-bench command to compare against.
./target/release/ferrox bench -m model.gguf -p 512 -n 128 -r 3

# Metal
./target/release/ferrox bench -m model.gguf --n-gpu-layers 99 -p 512 -n 128
```

| Flag | Meaning |
|---|---|
| `-m/--model` | GGUF to benchmark; without it, `bench` runs the synthetic matvec microbenchmark instead |
| `-p/--n-prompt` | Prefill tokens (default 512; `0` skips the `pp` row) |
| `-n/--n-gen` | Decode steps (default 128; `0` skips the `tg` row) |
| `-r/--repetitions` | Timed reps (default 3), plus one discarded warmup |
| `-t/--threads` | CPU threads (`0` = performance-core default) |
| `--n-gpu-layers` | `0` forces CPU; anything else offloads |

This is the **engine** number: no HTTP, no chat template, no tokenizer, no
sampling. The **serving** number — what a `ferrox-server` user actually
gets — is measured separately by
[`benchmarks/run_suite.py`](../benchmarks/README.md). Do not mix them:
a regression in one is not a regression in the other.

See also: [`FEATURES.md`](FEATURES.md) · [`MODELS.md`](MODELS.md) · [`API.md`](API.md).
