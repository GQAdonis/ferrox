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

## Evidence requirements

Nothing is described as working until it has actually been run and
checked:

- Every `ModelConfig` preset field is either confirmed against a primary
  source or listed in that preset's `best_effort_fields`. Cite sources in
  the preset / commit message; keep `docs/MODELS.md` status rows honest.
- New quantization kernels need goldens from an independent reference,
  not only self-parity.
- Hardware claims state the exact machine, or say compile-tested only.
- Benchmarks: `benchmarks/suite.json` → `run_suite.py` → pins in
  `benchmarks/receipts/pins/` → generated `benchmarks/RESULTS.md`
  (do not hand-edit headlines).

## Documentation

| Doc | Role |
|---|---|
| `docs/CLI.md` | `ferrox` flags + examples (llama.cpp-style) |
| `docs/MODELS.md` | What runs / what doesn’t (one page) |
| `benchmarks/RESULTS.md` | Speed vs llama.cpp |
| `docs/ROADMAP.md` | Future work only |

Don’t duplicate those. Plans belong only in `ROADMAP.md`; git holds history.
Don’t commit fast-staling counts (test totals, etc.) into prose.
