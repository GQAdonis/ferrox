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

- Preset fields: confirm each one against a primary source, or list it
  in `best_effort_fields`.
- New quant kernels need independent goldens, not only self-parity.
- A hardware claim names the machine it was measured on, or says
  compile-tested only.
- Speed numbers come from `ferrox bench --suite` against `llama-bench`,
  with no HTTP in the loop. `benchmarks/suite.json` drives the runs and
  `benchmarks/RESULTS.md` is generated from them, so never edit that
  table by hand. To measure a new model, add an entry to `suite.json`,
  put the GGUF under `models/`, then run
  `ferrox bench --suite --id <id> --fit-host --skip-missing`.
- Never force a thread count on either engine. llama.cpp defaults to
  performance cores and loses 2-4x above them, so pinning both to the
  same count flatters ferrox instead of making the comparison fair.
- Run-to-run spread on Apple Silicon is around 20%. A claim under that
  needs an interleaved A/B: alternate the two binaries round by round in
  one session and count rounds won. Two batches of runs will not do it.
- Commit the negative results too. `.scratch/NOTES_LLAMA_*.md` records
  what was tried and did not work, which is as useful as the wins.

## Documentation

| Doc | Role |
|---|---|
| `docs/FEATURES.md` | Capabilities overview |
| `docs/CLI.md` | CLI flags and examples |
| `docs/MODELS.md` | Supported models |
| `docs/API.md` | HTTP API matrix |
| `docs/CONFIG.md` | Environment variables |
| `benchmarks/RESULTS.md` | Speed vs llama.cpp (generated) |
| `benchmarks/README.md` | How those numbers are measured |
| `docs/ROADMAP.md` | Planned work |

Do not duplicate those. Plans belong in `ROADMAP.md`, and git holds the
history. Keep counts that go stale fast (test totals and the like) out
of prose.
