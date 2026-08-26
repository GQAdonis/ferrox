# ferrox-inference

Facade crate for [Ferrox](https://github.com/antonellof/ferrox), a
pure-Rust GGUF / MoE inference engine: mmap loaders, quantized CPU +
Apple Metal + CUDA kernels, and an OpenAI-compatible server.

It contains no logic. It re-exports the workspace under one name, so a
dependent writes one line instead of six, and so the project is
findable on crates.io, since the name `ferrox` belongs to an unrelated
crate.

```toml
[dependencies]
ferrox-inference = "0.12"
```

```rust
use ferrox_inference::gguf::ShardedGguf;

let file = ShardedGguf::open("model.gguf")?;
```

## The binaries are elsewhere

```bash
cargo install ferrox-cli      # installs the `ferrox` binary
cargo install ferrox-server   # OpenAI-compatible HTTP server
```

They are not shipped from this crate on purpose: two crates installing
a binary called `ferrox` would fight over the same path in
`~/.cargo/bin`.

## Features

| Feature | Effect |
|---|---|
| `metal` | Apple Metal kernels. Apple Silicon only. |
| `cuda` | CUDA/NVRTC kernels. Needs a CUDA toolkit at build time. |
| `api` | Re-export `ferrox-api` (route constants + wire DTOs). |

Neither GPU feature is on by default. `metal` does not build off Apple
Silicon, and `cuda` needs a toolkit most machines do not have.

The bar CUDA is held to is "must compile". There is no pinned benchmark
host for it and no published timings, so treat a Windows or Linux
install as CPU-only in practice. See
[`docs/FEATURES.md`](https://github.com/antonellof/ferrox/blob/main/docs/FEATURES.md).

## The rest of the workspace

| Crate | What it is |
|---|---|
| [`ferrox-gguf`](https://crates.io/crates/ferrox-gguf) | GGUF mmap reader, sharded checkpoints |
| [`ferrox-quant`](https://crates.io/crates/ferrox-quant) | Block layouts, fused dequant+dot |
| [`ferrox-safetensors`](https://crates.io/crates/ferrox-safetensors) | SafeTensors mmap reader |
| [`ferrox-core`](https://crates.io/crates/ferrox-core) | Tensor ops, RoPE, GQA, KV cache |
| [`ferrox-moe`](https://crates.io/crates/ferrox-moe) | Expert routing and dispatch |
| [`ferrox-models`](https://crates.io/crates/ferrox-models) | Loaders and decoder stacks |
| [`ferrox-api`](https://crates.io/crates/ferrox-api) | Route constants + wire DTOs |
| [`ferrox-metal`](https://crates.io/crates/ferrox-metal) | Apple Metal kernels |
| [`ferrox-cuda`](https://crates.io/crates/ferrox-cuda) | CUDA/NVRTC kernels |

Speed claims live in
[`benchmarks/RESULTS.md`](https://github.com/antonellof/ferrox/blob/main/benchmarks/RESULTS.md),
measured against llama.cpp on the same host and the same GGUF. That
table is generated from the raw timing files each run writes, and it
says so wherever nothing has been measured.

Apache-2.0.
