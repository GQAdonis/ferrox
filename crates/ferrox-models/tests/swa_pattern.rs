//! ferrox's per-architecture default sliding-window pattern, pinned
//! against llama.cpp's own `load_arch_hparams`.
//!
//! Same mechanism as `rope_layout.rs`, aimed at the same class of bug.
//! `{arch}.attention.sliding_window_pattern` is *optional* in a GGUF:
//! llama.cpp seeds the period from a per-architecture literal and only
//! lets the key override it. ferrox reads the key and, when it is
//! absent, falls back to `capability::default_swa_layout`.
//!
//! A missing entry in that table is not neutral. With no period,
//! `ModelConfig::layer_sliding_window` returns the window for *every*
//! layer, so the layers that should attend over the whole context see
//! only a window instead. The file loads, the model runs at full speed,
//! and it answers from a truncated history. That is a different model,
//! and nothing says so.
//!
//! `LLAMA_SWA_DEFAULTS` below is a mechanical transcription of every
//! `src/models/*.cpp` that seeds `swa_period` before
//! `ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,
//! swa_period, false)`. Three llama.cpp spellings are deliberately
//! absent because none of them is a per-arch period:
//!
//! - `set_swa_pattern(0)` (`deepseek4.cpp:68`, `dflash.cpp:54`) — every
//!   layer sliding, which is already what ferrox does with no period.
//! - `set_swa_pattern(1)` (`phi3.cpp:23`) — no layer sliding, on a
//!   branch that has already zeroed `n_swa` and set `swa_type = NONE`.
//! - `get_key_or_arr(..., hparams.is_swa_impl, n_layer)` with no scalar
//!   seed (`gemma4`, `gemma4-assistant`, `step35`, `mimo2`, `dflash`) —
//!   a per-layer array in the file, so there is no default to pin.
//!
//! Regenerate by re-reading the reference; do not edit an entry to make
//! a failing test pass.

use ferrox_models::capability::{default_swa_layout, resolve_profile, SwaPattern};
use ferrox_models::config::test_dense_fixture;

/// `(gguf arch, period, dense_first, citation)`.
///
/// `dense_first` is llama.cpp's second argument to
/// `llama_hparams::set_swa_pattern` (`src/llama-hparams.cpp:8-22`):
///
/// - `false` → `is_swa[il] = il % p < (p - 1)`, full attention on the
///   **last** layer of each period;
/// - `true`  → `is_swa[il] = il % p != 0`, full attention on the
///   **first**.
///
/// It is transcribed because it is load-bearing, not decorative: for a
/// 32-layer period-4 model the two placements disagree on 16 layers.
const LLAMA_SWA_DEFAULTS: &[(&str, usize, bool, &str)] = &[
    ("gpt-oss", 2, false, "src/models/openai-moe.cpp:9"),
    ("gemma2", 2, false, "src/models/gemma2.cpp:6"),
    ("gemma3", 6, false, "src/models/gemma3.cpp:7"),
    ("gemma3n", 5, false, "src/models/gemma3n.cpp:4"),
    (
        "gemma-embedding",
        6,
        false,
        "src/models/gemma-embedding.cpp:5",
    ),
    ("cohere2", 4, false, "src/models/cohere2.cpp:5"),
    ("exaone4", 4, false, "src/models/exaone4.cpp:7"),
    ("olmo2", 4, false, "src/models/olmo2.cpp:9"),
    ("mellum", 4, false, "src/models/mellum.cpp:11"),
    ("exaone-moe", 4, false, "src/models/exaone-moe.cpp:6"),
    ("afmoe", 4, false, "src/models/afmoe.cpp:17"),
    ("plamo3", 8, false, "src/models/plamo3.cpp:9"),
    ("llama4", 4, false, "src/models/llama4.cpp:19"),
    ("smallthinker", 4, true, "src/models/smallthinker.cpp:9-11"),
    ("laguna", 4, true, "src/models/laguna.cpp:39-41"),
    ("cohere2moe", 4, true, "src/models/cohere2moe.cpp:31-33"),
    ("modern-bert", 3, true, "src/models/modern-bert.cpp:8-10"),
];

/// llama.cpp's `llama_hparams::set_swa_pattern` (`src/llama-hparams.cpp:8-22`).
///
/// The C reads `n_pattern == 0 || (il % n_pattern != 0)` for
/// `dense_first` and `n_pattern == 0 || (il % n_pattern < n_pattern - 1)`
/// otherwise; `is_multiple_of` is clippy's spelling of the first.
fn llama_layer_is_swa(il: usize, period: usize, dense_first: bool) -> bool {
    if period == 0 {
        true
    } else if dense_first {
        !il.is_multiple_of(period)
    } else {
        il % period < period - 1
    }
}

/// Every transcribed period and phase must be exactly what ferrox holds.
///
/// This is the test that goes red if a value drifts. It has caught one
/// already: `gemma3n` was transcribed as 6 (copied from `gemma3`) where
/// `gemma3n.cpp:4` says 5.
#[test]
fn swa_defaults_match_llama_cpp() {
    let mut wrong = Vec::new();
    for &(arch, period, dense_first, cite) in LLAMA_SWA_DEFAULTS {
        let want = SwaPattern {
            period,
            dense_first,
        };
        match default_swa_layout(arch) {
            Some(got) if got == want => {}
            other => wrong.push(format!(
                "{arch} ({cite}): ferrox {other:?}, llama.cpp {want:?}"
            )),
        }
    }
    assert!(
        wrong.is_empty(),
        "default SWA layout disagrees with llama.cpp:\n  {}",
        wrong.join("\n  ")
    );
}

/// The table may not grow an entry the reference does not have either.
///
/// The two assertions above and below are one-directional: they would
/// pass with an invented `("qwen3", 4)` sitting in `capability.rs`,
/// which would window three of every four Qwen3 layers for a model that
/// windows none.
#[test]
fn ferrox_invents_no_swa_period_llama_cpp_does_not_have() {
    let pinned: std::collections::HashSet<&str> =
        LLAMA_SWA_DEFAULTS.iter().map(|&(a, ..)| a).collect();
    let mut extra = Vec::new();
    for p in ferrox_models::capability::architecture_catalog() {
        if default_swa_layout(p.gguf_name).is_some() && !pinned.contains(p.gguf_name) {
            extra.push(p.gguf_name);
        }
    }
    assert!(
        extra.is_empty(),
        "ferrox seeds a SWA period llama.cpp does not: {extra:?}"
    );
}

/// The transcription itself must not silently shrink.
///
/// 17 is what `src/models/*.cpp` holds today. A later reference sync may
/// only raise this number.
#[test]
fn the_swa_transcription_is_complete_enough_to_be_worth_pinning() {
    assert!(
        LLAMA_SWA_DEFAULTS.len() >= 17,
        "transcribed {} architectures that seed a swa_period; llama.cpp has 17",
        LLAMA_SWA_DEFAULTS.len()
    );
    let mut unknown = Vec::new();
    for &(arch, ..) in LLAMA_SWA_DEFAULTS {
        if resolve_profile(arch).is_none() {
            unknown.push(arch);
        }
    }
    assert!(
        unknown.is_empty(),
        "in llama.cpp's inventory but not in ferrox's capability registry, so a \
         checkpoint tagged with it is refused for the wrong reason: {unknown:?}"
    );
}

/// For EVERY architecture in the table, feeding `default_swa_layout`'s
/// answer into `ModelConfig` has to put the full-attention layers on
/// exactly the indices llama.cpp does.
///
/// Both phases, in one loop. `dense_first` used to be excluded here
/// because `layer_sliding_window` could not express it; it can now, and
/// running the two phases through the same assertion is the point --
/// a phase that is only checked against itself is not checked.
///
/// Pinning the number alone would not catch an off-by-one in
/// `layer_sliding_window`; this walks 64 layers of each architecture and
/// compares layer by layer against `set_swa_pattern` itself.
#[test]
fn the_period_lands_on_the_layers_llama_cpp_windows() {
    const WINDOW: usize = 128;
    let mut wrong = Vec::new();
    for &(arch, period, dense_first, cite) in LLAMA_SWA_DEFAULTS {
        let mut cfg = test_dense_fixture();
        cfg.n_layers = 64;
        cfg.sliding_window = Some(WINDOW);
        cfg.swa_pattern = default_swa_layout(arch).map(|p| p.period);
        cfg.swa_dense_first = default_swa_layout(arch).is_some_and(|p| p.dense_first);
        for il in 0..cfg.n_layers {
            let want = llama_layer_is_swa(il, period, dense_first);
            let got = cfg.layer_sliding_window(il).is_some();
            if want != got {
                wrong.push(format!(
                    "{arch} ({cite}) layer {il}: ferrox windowed={got}, llama.cpp windowed={want}"
                ));
                break;
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "sliding window lands on the wrong layers:\n  {}",
        wrong.join("\n  ")
    );
}

/// The two phases must DISAGREE, or the loop above proves nothing.
///
/// `layer_sliding_window` used to implement only `dense_first = false`,
/// which put `smallthinker` and `laguna` -- both live on the generic GQA
/// path -- on every-layer-windowed instead. This pins the cost of that:
/// on a 32-layer period-4 model the two phases disagree about SIXTEEN
/// layers, so a single shared implementation cannot be right for both.
#[test]
fn the_two_phases_are_not_the_same_answer() {
    const WINDOW: usize = 128;
    let mut dense_first_seen = 0usize;
    for &(arch, period, dense_first, cite) in LLAMA_SWA_DEFAULTS {
        assert_eq!(
            default_swa_layout(arch).map(|p| (p.period, p.dense_first)),
            Some((period, dense_first)),
            "{arch} ({cite}) lost its transcribed layout"
        );
        if !dense_first {
            continue;
        }
        dense_first_seen += 1;

        let mut cfg = test_dense_fixture();
        cfg.n_layers = 32;
        cfg.sliding_window = Some(WINDOW);
        cfg.swa_pattern = Some(period);
        cfg.swa_dense_first = true;
        let differs = (0..cfg.n_layers)
            .filter(|&il| {
                cfg.layer_sliding_window(il).is_some() != llama_layer_is_swa(il, period, false)
            })
            .count();
        assert!(
            differs > 0,
            "{arch} ({cite}): if the two phases agreed there would be nothing to carry \
             -- re-derive this test"
        );
    }
    assert!(
        dense_first_seen >= 4,
        "only {dense_first_seen} dense_first architectures checked; llama.cpp passes \
         dense_first=true for smallthinker, laguna, cohere2moe and modern-bert"
    );
}

/// llama.cpp's two DEGENERATE periods, which are not the same as having
/// no period at all.
///
/// `set_swa_pattern(0)` windows every layer; `set_swa_pattern(1)` windows
/// none, under either phase. ferrox filtered `pattern == 1` out of the
/// metadata before it reached the config and fell back to the per-arch
/// default, i.e. to alternating or to every-layer -- exactly inverted
/// from "no layer slides". Latent, since no checkpoint here writes 1,
/// but inverted is the wrong direction to be latent in.
#[test]
fn period_one_windows_nothing_and_period_zero_windows_everything() {
    const WINDOW: usize = 128;
    for dense_first in [false, true] {
        let mut cfg = test_dense_fixture();
        cfg.n_layers = 8;
        cfg.sliding_window = Some(WINDOW);
        cfg.swa_dense_first = dense_first;

        cfg.swa_pattern = Some(1);
        for il in 0..cfg.n_layers {
            assert_eq!(
                cfg.layer_sliding_window(il),
                None,
                "period 1 (dense_first={dense_first}) must window no layer, llama-hparams.cpp:8-22"
            );
        }

        cfg.swa_pattern = Some(0);
        for il in 0..cfg.n_layers {
            assert_eq!(
                cfg.layer_sliding_window(il),
                Some(WINDOW),
                "period 0 (dense_first={dense_first}) must window every layer"
            );
        }
    }
}
