# Plans

The goal is in [`north-star.md`](north-star.md): **be the Rust
alternative to llama.cpp.** Same models, same command shapes, same or
better performance, on the hardware people actually own.

Every plan below is ranked against that. Read the north star first; it
is the argument, this file is the index.

## Priority order

### 1. Decoder scalability, the gate on everything else

[`model-layer-reorg.md`](model-layer-reorg.md)

`crates/ferrox-models/src/decoder.rs` is **6438 lines**. Adding a model
means editing it. llama.cpp adds a model as one small file, which is
why it has **140** architectures and ferrox has **47 paths of which
about 12 are proven**.

That is not a tidiness complaint. It already costs correctness: the
paged decode loop had **lost five model features** by being a copy of
the contiguous one, three of them wrong on CPU too. Copy-per-path loses
features one at a time and nothing notices.

Nothing else in this list scales until this is fixed, which is why it
is first.

### 2. Correctness of what already loads

[`llama-cpp-parity-push.md`](llama-cpp-parity-push.md)

A coverage audit on 2026-08-27 classified all 150 catalog rows by what
happens when a real checkpoint loads:

| Outcome | Count |
|---|---|
| Runs, with evidence | **~12** |
| Refuses, naming what is missing | ~90 |
| **Loads and is WRONG** | 5 arch strings + 3 cross-cutting axes |
| Unknown, unproven | ~40 |

The wrong ones are live and on popular checkpoints. They outrank every
performance item.

### 3. Running models that do not fit

[`freetoken-parity.md`](freetoken-parity.md),
[`serving-and-tiered-kv.md`](serving-and-tiered-kv.md)

The strongest reason to choose ferrox over llama.cpp rather than tie
with it. `ferrox-edge` already holds the policy (`expert_cache`,
`expert_slots`, `placement`, `residency`, `footprint`) and **nothing
executes it** outside a compile-only CUDA pool.

### 4. Hardware reach

[`amd-strix-halo.md`](amd-strix-halo.md)

Ferrox has CPU, Metal and CUDA. That is Apple and NVIDIA and nothing
else. Every AMD and Intel GPU has no path, while llama.cpp reaches all
of them through Vulkan.

### 5. Performance where it is behind

[`llama-cpp-parity-push.md`](llama-cpp-parity-push.md) (CPU half)

Metal is done: every dense `pp512` row is 0.98x-1.10x and **8 of 12
`tg128` rows are faster than llama.cpp**. CPU is not: all 16 comparable
rows are red, 1.41x to 5.06x.

Ranked below correctness because it makes no new model runnable, and
because the measured evidence says the remaining lever is thread
scaling rather than kernels.

### 6. Below the line

[`dflash-speculative-decoding.md`](dflash-speculative-decoding.md) makes
an already-running model faster, never makes a new one runnable, and
all nine items sit behind a GO/NO-GO that needs a drafter checkpoint
which may not exist in loadable form.

[`ferrox-ui.md`](ferrox-ui.md) has one item left, the Tauri shell. The
web UI works and is standalone.

## Housekeeping

[`completion-push.md`](completion-push.md) is a scheduling document, not
a feature plan: it says which items may run in parallel and which
collide on the same file. Its merge discipline still applies and is
worth reading before dispatching work.

[`one-binary-serve.md`](one-binary-serve.md) is effectively done, one
item left.

[`done/`](done/) holds plans whose todos are all completed.

## The rules that keep being re-learned

**A plan's own status field is a claim, not evidence.** Verify against
the code. A merged PR once marked `paged-decode-path` complete while it
returned wrong tokens on Metal.

**One agent owns a file.** Two branches editing the same file produce a
merge nobody can review, and this project has already had one branch
silently revert three others.

**No agent runs benchmarks.** Measurement needs a quiet host, and a
loaded run reads 25-45% low.

**Refusing is not a defect.** llama.cpp will often run something
approximately; this project stops and names what is missing. A refusal
is a gap in coverage, not a bug.
