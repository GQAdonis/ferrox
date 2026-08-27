# Plans

Two files hold the plan:

- **[`north-star.md`](north-star.md)** is the goal and the ranking rule.
  Be the Rust alternative to llama.cpp: same models, same command
  shapes, same or better performance, on the hardware people actually
  own.
- **[`roadmap.md`](roadmap.md)** is every open item, merged by theme.

Two items are large enough to carry their own design document:

- **[`model-layer-reorg.md`](model-layer-reorg.md)**, splitting the
  6438-line decoder so architectures scale.
- **[`out-of-core-moe.md`](out-of-core-moe.md)**, running a 155 GB model
  on a 32 GB machine.

Everything else is history: [`archive/`](archive/) holds the five plans
whose items were merged into the roadmap, [`on-hold/`](on-hold/) holds
work ranked below the goal with the condition that brings it back, and
[`done/`](done/) holds plans whose todos are all completed.

## Where the project stands

Audited 2026-08-27, by what happens when a real checkpoint loads rather
than by whether the architecture name is known:

| Outcome | Count |
|---|---|
| Runs, **with evidence** | **~12** |
| Refuses, naming what is missing | ~90 |
| **Loads and is WRONG** | 5 arch strings + 3 cross-cutting axes |
| Unknown, unproven | ~40 |

| | llama.cpp | ferrox |
|---|---|---|
| Per-architecture graphs | 140 hand-written | 47 paths, **~12 proven** |
| Metal `pp512` | baseline | 0.98x-1.10x, at parity |
| Metal `tg128` | baseline | **8 of 12 rows faster** |
| CPU, all rows | baseline | **1.41x-5.06x slower** |
| GPU backends | CUDA, Metal, Vulkan, SYCL, HIP | CUDA, Metal |

Do not read the architecture catalog as a support matrix.

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
