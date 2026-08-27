---
name: "north star: the Rust alternative to llama.cpp"
overview: "THE GOAL, stated once so every other plan can be ranked against it: Ferrox should be what somebody reaches for INSTEAD OF llama.cpp. Same models, same command shapes, same or better performance, on the hardware people actually own. That is a bigger claim than 'a fast Rust inference engine' and it sets a bar that is checkable rather than aspirational: for any GGUF a user can run under llama.cpp, ferrox should run it, produce the same tokens, and not be slower. WHERE THE PROJECT ACTUALLY IS, counted rather than estimated: llama.cpp hand-writes 140 per-architecture graphs in `src/models/*.cpp`. Ferrox has 47 dedicated paths, 32 architectures it honestly refuses, and 68 mapped onto ONE shared generic GQA path. On 2026-08-27 five of those 68 (`gpt2`, `mpt`, `refact`, `bloom`, `jais`) were found computing ALiBi and learned absolute position embeddings as though they were NEOX RoPE, so 63 remain unaudited. On speed: Metal is at or past parity (every dense pp512 row 0.98x-1.10x, 8 of 12 tg128 rows FASTER than llama.cpp), and CPU is not (all 16 comparable rows red, 1.41x to 5.06x). THE SEQUENCING ARGUMENT: 'same models as llama.cpp' is 140 architectures and years of work. The wedge that gets there while being useful the whole way is RECENT BIG MODELS ON CONSUMER HARDWARE, including models too big for the machine's memory, because that is where llama.cpp is weakest and where a new engine can be chosen on merit rather than on being a rewrite. This plan ranks the other plans. It contains no engineering."
todos:
  - id: the-bar
    content: "NOT A TASK, the definition of done for the whole project, written so it can be argued with. Ferrox reaches parity when, for a GGUF a user can run under llama.cpp: (1) it LOADS, or refuses with a sentence naming exactly what is missing, and never loads-and-computes-something-else; (2) it produces the SAME TOKENS at temperature 0, checked against llama.cpp's own logits rather than against a golden file this project wrote; (3) it is NOT SLOWER on the same host, file and backend; (4) the COMMAND a user already knows works, or the difference is documented. Item (2) is the one that cannot be faked and the one this project already has the tool for: `ferrox parity` with `tools/llama_logits.c`. Item (1) is where ferrox currently differs most from llama.cpp, and deliberately so: llama.cpp will often run something approximately, this project's standing rule is to refuse. That rule stays. A refusal is a gap, not a defect."
    status: pending
  - id: t1-scale-the-model-layer
    content: "TIER 1, and it gates every other model item, which is why it is first. `crates/ferrox-models/src/decoder.rs` is 6438 LINES and holds the generic path plus per-architecture branching. Adding a model means editing it, and reviewing a model change means reading it. llama.cpp adds a model as ONE SMALL FILE in `src/models/`, which is why it has 140 of them and ferrox has 47. This is not a style preference, it is the reason the counts differ. Evidence it already costs correctness: wave 0 found FIVE model features (`attention_scale`, `post_attn_norm`, `post_ffn_norm`, gpt-oss `o_bias`, `gpt_oss_ffn`) that the paged decode loop had LOST by being a copy of the contiguous one, three of them wrong on CPU too. A copy-per-path structure loses features one at a time and nothing notices. See `model-registry-reorg.md` for the design. DONE MEANS: adding an architecture is a new file, a trait impl, a registry line, a fixture and a parity test, with no edit to a shared multi-thousand-line file."
    status: pending
  - id: t1-close-the-68
    content: "TIER 1. 68 architectures share one generic GQA path against llama.cpp's 68 bespoke graphs, and five were already found silently wrong. THE CHEAP FIX FIRST, because it is worth more than the audit and takes a fraction of the time: make the generic path OPT-IN. Today an unrecognised architecture FALLS ONTO generic GQA and runs; it should have to ASSERT that it really is plain GQA. That inversion converts 63 unaudited architectures from 'silently wrong' into 'honestly refused', which is this project's stated rule everywhere else, and it makes the remaining work visible instead of hypothetical. THEN audit outward from what people run: `qwen3moe`, `glm4moe`, `deepseek2` (MLA), `minimax`, `gemma3`/`gemma4`, `dots1`, `olmoe`, before the 2023 tail. Note that the tail IS in scope under this goal, unlike under a purely edge-focused one: 'same models as llama.cpp' includes `bloom` and `gpt2`. It is last, not excluded."
    status: pending
  - id: t1-out-of-core-execution
    content: "TIER 1, and the strongest single reason to choose ferrox over llama.cpp rather than merely tie with it. Make a model larger than memory run. A 200B MoE at Q4 is roughly 110 GiB; consumer machines have 16 to 128. An MoE touches only top-k of N experts per token, so the working set is a fraction of the weights, which is what makes this reachable. ferrox-edge ALREADY HOLDS THE POLICY: `expert_cache` (LRU with copy plans), `expert_slots` (bounded pool behind a `SlotDevice` trait), `placement`, `residency`, `footprint`, `pool`. NOTHING EXECUTES IT except `expert_pool::CudaExpertPool`, which is compile-verified only with its hardware test `#[ignore]`d. So the capability exists on paper and not in the engine. NEEDS: a `SlotDevice` for Metal and one for host RAM. START WITH MoE, not dense weight streaming, which is a far worse trade and is not in scope."
    status: pending
  - id: t1-moe-execution-quality
    content: "TIER 1, because the recent models worth running are overwhelmingly MoE and MoE is where this engine is weakest. Three defects, all measured. (1) The Metal MoE path NEVER records `activation_counts`: `crates/ferrox-metal/src/attn.rs:5620` returns `vec![Vec::new(); layers.len()]`, commented as avoiding a sync tax, and `MoeDecodeScratch.ids` is a single top_k buffer reused per layer so it must be widened to `top_k * n_layers` or replaced by an atomic histogram. `placement_plan` has been reading an all-zero hotness signal on Metal, which matters DIRECTLY for out-of-core since eviction is only as good as its hotness input. (2) OLMoE CPU is 2.19x prefill / 2.46x decode while its Metal rows are at parity. (3) `concurrent-cpu-moe-executor` and `double-buffered-prefill` are unstarted. MoE bench coverage rested on ONE model until Qwen1.5-MoE (shared experts, which OLMoE lacks) was added."
    status: pending
  - id: t2-same-commands
    content: "TIER 2, cheap, and it is most of what 'alternative to llama.cpp' means in practice to a user. Somebody switching should not have to relearn flags. `ferrox bench` already mirrors `llama-bench`'s `-p`/`-n`/`-r` shape and `ferrox` mirrors `-m`/`-p`/`-n`/`-ngl`. AUDIT the rest against `llama-cli`, `llama-server` and `llama-bench` and close the gaps that are one-line aliases: `-c`/`--ctx-size`, `-t`/`--threads`, `-b`/`--batch-size`, `--temp`, `--top-k`, `--top-p`, `--repeat-penalty`, `-s`/`--seed`, `-ngl`, `--split-mode`, `--main-gpu`, `-fa`/`--flash-attn`. Where ferrox deliberately differs, DOCUMENT the difference rather than silently accepting a flag that means something else, which is worse than rejecting it. `ferrox download` landed with `hf download`'s exact syntax for the same reason: a command copied off a model card should just work."
    status: pending
  - id: t2-vulkan-is-most-consumer-gpus
    content: "TIER 2, and it is the largest single hardware gap. Ferrox has CPU, Metal and CUDA. That is Apple and NVIDIA and NOTHING ELSE: every AMD and Intel GPU has no path. llama.cpp ships Vulkan and reaches all of them with one backend, so 'same hardware as llama.cpp' is currently false by a wide margin. The four Vulkan items in `amd-strix-halo` are filed as blocked on owning a Strix Halo box, which is the wrong framing: Vulkan is testable on ANY machine with a Vulkan driver including an Intel iGPU, and Strix Halo is a tuning target rather than a prerequisite. Re-scope them off that hardware."
    status: pending
  - id: t2-cpu-performance
    content: "TIER 2, and the demotion needs saying plainly because the work is good and recent. Every red row in the ledger is CPU: all 16 comparable rows, prefill 1.41x to 5.06x, decode 1.68x to 3.55x. Under the bar in `the-bar` this is a genuine parity failure, so it cannot be dropped. It sits at TIER 2 only because it makes no new model RUNNABLE. Measured evidence that redirects the approach: the gap scales INVERSELY with model size, 5.06x at 135M down to 1.41x at 7B, the signature of fixed per-matmul overhead rather than kernel throughput, and every i8mm kernel already exists and is on the hot path. `cpu-decode-scaling` (fork-join, llama's persistent spin-barrier pool) is the remaining lever; the kernel items are largely spent. First evidence the recent work paid: TinyLlama CPU decode 48.11 to 54.41 tok/s, gap 2.15x to 2.00x."
    status: pending
  - id: t3-ranked-below
    content: "NOT ABANDONED, ranked below, recorded so nobody re-raises them without an argument. (1) `dflash-speculative-decoding`, all 9 items: it makes an already-running model faster, never makes a new one runnable, and sits behind `dflash-checkpoint-reality`, a GO/NO-GO needing a published drafter checkpoint that may not exist in loadable form. Ferrox does not train, and a drafter without weights proposes noise, which is slower than none. (2) `ui-tauri-shell`: the web UI works and is standalone. (3) The 2023 architecture tail: in scope under this goal, but after the recent families, and handled safely in the meantime by the opt-in inversion in `t1-close-the-68`."
    status: pending
  - id: verification-without-downloading-everything
    content: "CROSS-CUTTING, the answer to 'do we have to test every model'. No. 140 real checkpoints is tens of terabytes and would still only prove the ones downloaded. THREE MECHANISMS ALREADY EXIST HERE, each applied to far less than it could be. (1) DIFFERENTIAL, the decisive one: `ferrox parity` plus `tools/llama_logits.c` compares first-token logit distributions against llama.cpp on the same file, needs no knowledge of the architecture, and is not run across the suite. (2) SYNTHETIC FIXTURES: `scripts/make_dots1_fixture.py` and `make_gptoss_fixture.py`, two architectures out of 139. A fixture is kilobytes, needs no download, and catches wrong tensor wiring and wrong refusals. One per supported architecture is the best coverage-per-byte available and should be a requirement of the registry in `t1-scale-the-model-layer`. (3) PINNED PROPERTY TABLES transcribed from llama.cpp's loader: `LLAMA_NO_ROPE` does this for one property and is exactly what caught the gpt2/bloom class. Extend to ALiBi slopes, norm placement, QKV bias, MoE routing bias."
    status: pending
  - id: measurement-needs-more-than-one-laptop
    content: "CROSS-CUTTING, and it now blocks the claim rather than merely annoying us. Every number comes from ONE 32 GiB M2 Pro that is also the development machine. Already hit: two Metal rows skipped mid-suite when macOS CoreSuggestions spiked the load, a whole suite lost to an orphaned process holding the instance lock, Mixtral permanently unmeasurable, CUDA never measured, x86 never executed. 'Same or better performance than llama.cpp' cannot be claimed on hardware nobody has run. LANDED 2026-08-27: a receipt `host_spec` field recording CPU model, core split, RAM and OS but NOT hostname or user, and a render that REFUSES to merge receipts from different hosts rather than silently mixing them. STILL NEEDED: a bare-metal rental for CPU and x86, NOT a shared-vCPU instance, because a noisy neighbour's steal time does not appear in the guest's load average and would defeat the quiet-host guard invisibly; and a decision on whether CUDA earns a dedicated box or whether 'must compile' stays the honest claim."
    status: pending
isProject: false
---

# North star: the Rust alternative to llama.cpp

> Same models. Same command shapes. Same or better performance. On the
> hardware people actually own.
>
> This plan contains no engineering. It is the ranking the other plans
> are read through.

## The bar, so it can be argued with

For any GGUF a user can run under llama.cpp:

1. It **loads**, or refuses naming exactly what is missing. Never
   loads-and-computes-something-else.
2. It produces the **same tokens** at temperature 0, checked against
   llama.cpp's own logits.
3. It is **not slower** on the same host, file and backend.
4. The **command** a user already knows works, or the difference is
   documented.

Point 1 is where ferrox deliberately differs: llama.cpp will often run
something approximately, and this project refuses instead. That rule
stays. A refusal is a gap, not a defect.

## Counted, not estimated

A coverage audit on 2026-08-27 classified all 150 catalog rows by what
actually happens when a real checkpoint loads, rather than by whether
the name is known. The result is harsher than the path count suggests:

| Outcome | Count |
|---|---|
| **A. Runs, with evidence** (bench row, or pinned against llama.cpp logits) | **~12** |
| **B. Refuses, naming what is missing** | ~90 |
| **C. Loads and is WRONG** | 5 arch strings + 3 cross-cutting axes |
| **D. Unknown: nothing refuses it, nothing proves it** | ~40 |

Category A in full: `llama` (covering TinyLlama, Mistral, Mixtral,
SmolLM2), `qwen2`, `qwen2moe`, `qwen3`, `olmoe`, `gemma2`, `gemma3`,
`gemma4`, `phi3`/`phi4`, plus `gpt-oss` and `dots1` pinned against real
`libllama` logits. That is the honest extent of proven coverage.

| | llama.cpp | ferrox |
|---|---|---|
| Per-architecture graphs | 140 hand-written | 47 paths, **~12 proven** |
| Shared generic path | none | **68 architectures** |
| Metal `pp512` | baseline | 0.98x-1.10x, at parity |
| Metal `tg128` | baseline | **8 of 12 rows faster** |
| CPU, all rows | baseline | **1.41x-5.06x slower** |
| GPU backends | CUDA, Metal, Vulkan, SYCL, HIP | CUDA, Metal |

Two numbers do most of the work. **6438** is the scaling problem: the
line count of `decoder.rs`, the file you must edit to add a model, and
the reason the left column says 140 and the right says 47. **~12** is
how much of that 47 anyone has evidence for.

## Why the model layer is first

llama.cpp has 140 architectures because adding one is a small new file.
Ferrox has 47 because adding one means editing a 6438-line file. The
counts differ for a structural reason, not an effort reason.

It already costs correctness: the paged decode loop had **lost five
model features** by being a copy of the contiguous one, three of them
wrong on CPU too. Copy-per-path loses features one at a time and
nothing notices.

## The cheap fix that beats the expensive audit

Auditing 63 architectures is expensive. Inverting the default is not.

Today an unrecognised architecture **falls onto** the generic path and
runs. It should have to **assert** it is plain GQA. That one inversion
turns 63 silently-wrong risks into 63 honest refusals, and makes the
remaining work visible instead of hypothetical.

## Ranking

**Shipped and working beats big and undone.** That rule outranks the
tiers below. An item that cannot be cut into steps which each land is
filed wrong, and the fix is to re-cut it, not to start it.

The executable order lives in [`roadmap.md`](roadmap.md). It differs
from a pure tier sort in one deliberate way: correctness and the
verification oracle come FIRST, ahead of the model layer refactor,
because some checkpoints are wrong right now and their fixes are small,
and because refactoring 15464 lines across two crates without a working
cross-engine oracle means refactoring known-wrong code with no way to
tell if you broke it.

```
TIER 1  scale the model layer     6438 lines is why we have 47 not 140
        close the 68              invert the default, then audit outward
        out-of-core execution     the reason to choose ferrox, not tie
        MoE execution quality     hotness signal, CPU executor, residency

TIER 2  same commands             cheap, and most of what "alternative" means
        vulkan                    every AMD and Intel GPU, currently none
        cpu performance           every red row, but unlocks no new model

TIER 3  dflash speculative        faster, not newly-possible, and blocked
        tauri shell               the web UI already works
        the 2023 tail             in scope, but last
```
