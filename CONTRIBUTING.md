# Contributing

## Workflow

Run before opening a PR (CI enforces all of these with `-D warnings`):

```
cargo fmt --all
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The `cuda` feature must keep compiling without a GPU or CUDA toolkit
present: `cargo clippy -p ferrox-cli -p ferrox-server --features cuda`.
CI also builds that CLI/server chain and, on `macos-latest`, compiles
`cargo clippy -p ferrox-metal -p ferrox-cli --features metal`. Hardware
kernel tests stay `#[ignore]`d on hosted CI.

## Benchmarks & presets

- Preset fields: confirm against a primary source, or list in
  `best_effort_fields`.
- New quant kernels need independent goldens (not only self-parity).
- Hardware claims state the machine, or say compile-tested only.
- Two tracks, both driven from `benchmarks/suite.json`, both generated
  into `benchmarks/RESULTS.md` (never hand-edited):
  - **Engine** — `ferrox bench --suite` vs `llama-bench`, no HTTP.
  - **Serving** — `run_suite.py` vs `llama-server`, over HTTP.
  Do not quote one as the other.
- Never force a thread count on either engine. llama.cpp defaults to
  performance cores and loses 2-4x above them, so pinning both to the
  same count flatters ferrox rather than making it fair.
- Run-to-run spread on Apple Silicon is ~20%. Any claim under that needs
  interleaved A/B (alternate the two binaries round by round in one
  session and count rounds won), not two batches of runs.
- Negative results get committed too. `.scratch/NOTES_LLAMA_*.md` is the
  record of what was tried and did not work; it is as useful as the wins.

## Documentation

| Doc | Role |
|---|---|
| `docs/FEATURES.md` | Capabilities overview |
| `docs/CLI.md` | CLI flags and examples |
| `docs/MODELS.md` | Supported models |
| `docs/API.md` | HTTP API matrix |
| `benchmarks/RESULTS.md` | Speed vs llama.cpp |
| `docs/ROADMAP.md` | Planned work |

Don’t duplicate those. Plans belong in `ROADMAP.md`; git holds history.
Don’t commit fast-staling counts (test totals, etc.) into prose.
