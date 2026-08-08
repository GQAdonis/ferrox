# Roadmap

Goal: match or beat [llama.cpp](https://github.com/ggerganov/llama.cpp)
tok/s on the same host, backend, and GGUF. Current speed numbers:
`[benchmarks/RESULTS.md](../benchmarks/RESULTS.md)`.

What ships today: `[FEATURES.md](FEATURES.md)` · `[MODELS.md](MODELS.md)`.



Where the project should go beyond closing the measured gaps.

1. **Run bigger models on the same hardware.** Make Qwen3 35B-A3B Q5
  usable on a box that today only sensibly runs Q4, or an 8B. Most items
   below serve this.
2. **RAM / VRAM optimization.** Residency planning exists
  (`ferrox inspect-plan`); what is missing is acting on it hard enough to
   change which models fit — tighter KV (`turbo3`, quantized CTK),
   streaming expert residency, not materializing activations we do not
   need.
3. **Hybrid CPU/GPU, especially MoE.** Routed experts are the natural
  split: hot experts resident on GPU, cold ones streamed or run on CPU.
   `PlacementPlan` and `ExpertStore` are the groundwork.
4. **CUDA performance.** The kernels now build and run on real hardware
  but have had no tuning pass.
5. **Tool calling** and **full OpenAI API compatibility** — see
  `[API.md](API.md)`. Grammar/JSON-schema constrained decoding and MCP
   invocation are the gaps.
6. **Docker images**, so none of the above requires a Rust toolchain to
  evaluate.

**Models**

- Gemma-4 tokenizer (`gemma4`) + fair-chat pin (engine loads today)
- HybridEngine + Qwen3.5
- Llama 4 / MiniMax engines
- Vision (projector + generate)
- Real GLM-5.2 / DeepSeek V4 / full Kimi e2e
- MTP draft heads
- Qwen2-MoE / Mixtral pins when GGUF fits Host B

**Serving**

- Tool calling: OpenAI `tools` / `tool_choice` request + response shape
- Full grammar / JSON schema constrained decoding
- MCP tool invocation; Anthropic streaming + tools
- Full OpenAI API surface (see `[API.md](API.md)`)
- Docker images (CPU / Metal / CUDA variants)
- Continuous-batching multi-request throughput pin
- Full KV layer offload; multi-GPU / tensor parallel / PD disaggregation
- Serving pins: re-measure now that the harness no longer forces `-t`

**KV cache / memory**

- `turbo3` dtype; Metal WHT on the CTK path
- Act on `inspect-plan`'s residency plan: stream cold experts, bound the
KV budget, report what a host can actually fit
- Hybrid CPU/GPU expert placement for MoE — the main lever for running a
larger model or a higher quant on unchanged hardware

**Engineering practice worth taking from llama.cpp**

- A `test-backend-ops` equivalent: every kernel checked against a CPU
reference across shapes and quant kinds, so a backend can never be
merged that merely runs fast.
- Build the CUDA feature combination in CI, even without a GPU to run it
on. The break above was a compile error, not a runtime one.

