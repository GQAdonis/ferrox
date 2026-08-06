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
- Suite: `benchmarks/suite.json` → `run_suite.py` → pins → generated
  `benchmarks/RESULTS.md` (do not hand-edit headlines).

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
