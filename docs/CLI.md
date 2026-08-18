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

# Largest context that fits, chosen before the weights load; the
# arithmetic behind the number is printed to stderr
./target/release/ferrox -m model.gguf -p "Hi" -n 64 -c auto

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
| `-c` / `--ctx-size` | `auto` = largest that fits the device memory budget, `0` = GGUF `{arch}.context_length` (else 4096), or a token count |
| `--strict-budget` | Refuse to load when the pre-load budget says the context will not fit (default: warn and continue) |
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
# Plan against a backend's real memory budget (Metal
# recommendedMaxWorkingSetSize / free VRAM / host RAM minus a reserve).
# Always reports the largest context that fits and the arithmetic:
./target/release/ferrox inspect-plan model.gguf --backend metal --ctk f16
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

### Running under a supervisor

```bash
# Kernel picks the port; the bound address is announced on stdout
./target/release/ferrox-server -m model.gguf --port 0 --exit-on-stdin-close
{"event":"ferrox.server.ready","addr":"127.0.0.1:52091","port":52091,"scheme":"http","pid":4242,"version":"0.5.0"}
```

`--port 0` plus that one line removes the need for a parent process to
probe whether a port is free, or to work out whether an existing
listener is a stale copy of itself or a stranger's server. Read stdout
line by line and ignore anything that is not the ready event — the
tracing subscriber shares the stream.

`--exit-on-stdin-close` (or `FERROX_EXIT_ON_STDIN_CLOSE=1`) exits when
stdin reaches EOF, which is the one orphan-prevention mechanism that
behaves identically on macOS, Windows and Linux and survives a parent
that dies rather than exiting cleanly. It is **opt-in**: a server
started with stdin redirected from `/dev/null` (systemd, cron, `nohup`)
sees EOF immediately, so the parent that wants the guarantee is the one
that asks for it and keeps the pipe open.

The server accepts `-m/--model`, `--host`, `--port`, `-t/--threads`,
`-dev/--device`, `-ngl/--n-gpu-layers`, `--exit-on-stdin-close`, and
`--list-devices`. Existing
`FERROX_MODEL_PATH`, `FERROX_ADDR`, and backend environment variables remain
supported; command-line values take precedence. Keep secrets such as
`FERROX_API_KEY` in the environment.

## Benchmark (`ferrox bench`)

With `-m`, `bench` is a [`llama-bench`](https://github.com/ggerganov/llama.cpp/tree/master/tools/llama-bench)
work-alike: same workload names (`pp<N>` batched prefill, `tg<N>` decode),
same reporting (median ± population stddev over `-r` reps, one warmup
discarded), same flag names — so the two outputs can be read side by side.

```bash
# one GGUF (CPU). Prints the exact llama-bench command to compare against.
./target/release/ferrox bench -m model.gguf -p 512 -n 128 -r 3 --compare

# Metal
./target/release/ferrox bench -m model.gguf --n-gpu-layers 99 -p 512 -n 128 --compare

# multi-model suite (same models list as benchmarks/suite.json)
./target/release/ferrox bench --suite --fit-host --skip-missing
./target/release/ferrox bench --suite --id tinyllama_q8 --backend metal
./target/release/ferrox bench --render
```

| Flag | Meaning |
|---|---|
| `-m/--model` | GGUF to benchmark; without it, `bench` runs the synthetic matvec microbenchmark instead |
| `-p/--n-prompt` | Prefill tokens (default 512; `0` skips the `pp` row) |
| `-n/--n-gen` | Decode steps (default 128; `0` skips the `tg` row) |
| `-r/--repetitions` | Timed reps (default 3), plus one discarded warmup |
| `-t/--threads` | CPU threads (`0` = performance-core default) |
| `--n-gpu-layers` | `0` forces CPU; anything else offloads |
| `--compare` | Also run `llama-bench` on the same GGUF and print the gap |
| `--suite` | Run every [`benchmarks/suite.json`](../benchmarks/suite.json) entry (fresh process each), write receipts, re-render [`RESULTS.md`](../benchmarks/RESULTS.md) |
| `--render` | Re-render the RESULTS table from existing receipts, measuring nothing |
| `--id` / `--backend` | Restrict `--suite` to one entry / backend |
| `--fit-host` / `--skip-missing` | Skip entries too large for the host / with no GGUF present |
| `--max-load` | Refuse to time a run when the host's 1-minute load average is at or above this (default `2.0`; `0` disables). `--suite` checks once up front and forwards the bar to every child |

### One model at a time

Every command that loads weights — `run`, `bench`, `verify`, `smoke`,
`run-kimi`, and `ferrox-server` — registers itself and **refuses to start
while another ferrox process is already holding a model**:

```
$ ferrox -m model.gguf -p "hi"
Error: 1 ferrox instance(s) are already running a model on this host:
  - server pid 59667, metal, models/SmolLM2-135M-Instruct-Q8_0.gguf
Running several models at once does not share the machine -- it thrashes
it, and any timing either process reports is noise. Stop the other
instance, or pass --allow-multiple-instances (or set
FERROX_ALLOW_MULTIPLE_INSTANCES=1) to start anyway.
```

Prefill is a dense GEMM across every core and the decode pool spins, so
two instances do not run at half speed each — they contend. Pass
`--allow-multiple-instances` (or set `FERROX_ALLOW_MULTIPLE_INSTANCES=1`)
when you want them anyway.

Header-only commands (`inspect`, `inspect-plan`, `presets`, `archs`,
`caps`), the HTTP client (`chat`), the downloader (`pull`) and
`bench --suite` / `--render` are exempt: none of them puts weights in
memory, and `--suite` is a supervisor whose children each register on
their own.

The registry is a directory of one small file per live process
(`$FERROX_INSTANCE_DIR`, default `~/.cache/ferrox/instances`). An entry
whose process is gone — a `kill -9`, a crash — is pruned by the next run
rather than blocking it. It is **advisory, not a lock**: two processes
starting in the same instant can each see the other and both refuse,
which is the safe direction, but nothing here stops a determined caller.

Add or change models in [`benchmarks/suite.json`](../benchmarks/suite.json)
(`id`, `name`, `gguf`, `backends`, `estimated_ram_gb`). No HTTP, chat
template, tokenizer, or sampling — same line llama.cpp draws with
`llama-bench` next to `llama-server`. Details:
[`benchmarks/README.md`](../benchmarks/README.md).

See also: [`FEATURES.md`](FEATURES.md) · [`MODELS.md`](MODELS.md) · [`API.md`](API.md).
