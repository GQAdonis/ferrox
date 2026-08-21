//! ferrox's per-architecture RoPE layout, pinned against llama.cpp's
//! `llama_model_rope_type`.
//!
//! Why a whole test for one enum: getting this wrong is silent. A NEOX
//! model rotated as NORM (or the reverse) still loads, still runs at
//! full speed, and still emits fluent text — it just rotates the wrong
//! pairs of every Q and K head, so the answers are wrong and nothing
//! says so. It has already happened twice in this repo: gpt-oss was on
//! the interleaved list until the gpt-oss coverage work, and an audit of
//! the whole table against the reference then found **24 more** archs on
//! the generic GQA path with the same defect (afmoe, apertus,
//! bailingmoe2, codeshell, dots1, exaone, exaone-moe, grovemoe,
//! hunyuan-dense, hunyuan-moe, laguna, mellum, mimo2, minicpm3,
//! nemotron, openelm, orion, plamo, plamo3, seed_oss, smallthinker,
//! starcoder2, step35, talkie).
//!
//! `LLAMA_ROPE_TYPES` below is a mechanical transcription of the
//! `LLAMA_ROPE_TYPE_NORM` and `LLAMA_ROPE_TYPE_NEOX` groups of that
//! switch (`src/llama-model.cpp`), keyed by the GGUF architecture string
//! from `LLM_ARCH_NAMES` (`src/llama-arch.cpp`). Architectures llama.cpp
//! returns `NONE` / `MROPE` / `IMROPE` for, or decides per-checkpoint
//! (`glm4`, `dflash`), are deliberately absent: ferrox does not
//! implement those layouts, and a placeholder here would assert
//! agreement that does not exist.
//!
//! Regenerate by re-reading the reference; do not edit an entry to make
//! a failing test pass.

use ferrox_models::capability::{architecture_catalog, ArchPath};
use ferrox_models::config::RopeLayout;

const LLAMA_ROPE_TYPES: &[(&str, RopeLayout)] = &[
    ("afmoe", RopeLayout::Neox),
    ("apertus", RopeLayout::Neox),
    ("arcee", RopeLayout::Norm),
    ("arctic", RopeLayout::Norm),
    ("baichuan", RopeLayout::Norm),
    ("bailingmoe", RopeLayout::Norm),
    ("bailingmoe2", RopeLayout::Neox),
    ("bert", RopeLayout::Neox),
    ("bitnet", RopeLayout::Neox),
    ("chameleon", RopeLayout::Norm),
    ("chatglm", RopeLayout::Norm),
    ("codeshell", RopeLayout::Neox),
    ("cogvlm", RopeLayout::Neox),
    ("cohere2", RopeLayout::Norm),
    ("cohere2moe", RopeLayout::Norm),
    ("command-r", RopeLayout::Norm),
    ("dbrx", RopeLayout::Neox),
    ("deci", RopeLayout::Norm),
    ("deepseek", RopeLayout::Norm),
    ("deepseek2", RopeLayout::Norm),
    ("deepseek2-ocr", RopeLayout::Norm),
    ("deepseek32", RopeLayout::Norm),
    ("deepseek4", RopeLayout::Norm),
    ("dots1", RopeLayout::Neox),
    ("dream", RopeLayout::Neox),
    ("eagle3", RopeLayout::Norm),
    ("ernie4_5", RopeLayout::Norm),
    ("ernie4_5-moe", RopeLayout::Norm),
    ("eurobert", RopeLayout::Neox),
    ("exaone", RopeLayout::Neox),
    ("exaone-moe", RopeLayout::Neox),
    ("exaone4", RopeLayout::Neox),
    ("falcon", RopeLayout::Neox),
    ("falcon-h1", RopeLayout::Neox),
    ("gemma", RopeLayout::Neox),
    ("gemma-embedding", RopeLayout::Neox),
    ("gemma2", RopeLayout::Neox),
    ("gemma3", RopeLayout::Neox),
    ("gemma3n", RopeLayout::Neox),
    ("gemma4", RopeLayout::Neox),
    ("gemma4-assistant", RopeLayout::Neox),
    ("glm-dsa", RopeLayout::Norm),
    ("gpt-oss", RopeLayout::Neox),
    ("gptneox", RopeLayout::Neox),
    ("granite", RopeLayout::Norm),
    ("granitehybrid", RopeLayout::Norm),
    ("granitemoe", RopeLayout::Norm),
    ("grok", RopeLayout::Neox),
    ("grovemoe", RopeLayout::Neox),
    ("hunyuan-dense", RopeLayout::Neox),
    ("hunyuan-moe", RopeLayout::Neox),
    ("hy_v3", RopeLayout::Neox),
    ("internlm2", RopeLayout::Norm),
    ("jais2", RopeLayout::Neox),
    ("jina-bert-v3", RopeLayout::Neox),
    ("laguna", RopeLayout::Neox),
    ("lfm2", RopeLayout::Neox),
    ("lfm2moe", RopeLayout::Neox),
    ("llada", RopeLayout::Norm),
    ("llada-moe", RopeLayout::Neox),
    ("llama", RopeLayout::Norm),
    ("llama-embed", RopeLayout::Norm),
    ("llama4", RopeLayout::Norm),
    ("maincoder", RopeLayout::Norm),
    ("mellum", RopeLayout::Neox),
    ("mimo2", RopeLayout::Neox),
    ("minicpm", RopeLayout::Norm),
    ("minicpm3", RopeLayout::Neox),
    ("minimax-m2", RopeLayout::Neox),
    ("minimax-m3", RopeLayout::Neox),
    ("mistral3", RopeLayout::Norm),
    ("mistral4", RopeLayout::Norm),
    ("modern-bert", RopeLayout::Neox),
    ("nanbeige", RopeLayout::Norm),
    ("nemotron", RopeLayout::Neox),
    ("neo-bert", RopeLayout::Norm),
    ("nomic-bert", RopeLayout::Neox),
    ("nomic-bert-moe", RopeLayout::Neox),
    ("olmo", RopeLayout::Norm),
    ("olmo2", RopeLayout::Neox),
    ("olmoe", RopeLayout::Neox),
    ("openelm", RopeLayout::Neox),
    ("orion", RopeLayout::Neox),
    ("pangu-embedded", RopeLayout::Neox),
    ("phi2", RopeLayout::Neox),
    ("phi3", RopeLayout::Neox),
    ("phimoe", RopeLayout::Neox),
    ("plamo", RopeLayout::Neox),
    ("plamo2", RopeLayout::Neox),
    ("plamo3", RopeLayout::Neox),
    ("plm", RopeLayout::Norm),
    ("qwen", RopeLayout::Neox),
    ("qwen2", RopeLayout::Neox),
    ("qwen2moe", RopeLayout::Neox),
    ("qwen3", RopeLayout::Neox),
    ("qwen3moe", RopeLayout::Neox),
    ("qwen3next", RopeLayout::Neox),
    ("rnd1", RopeLayout::Neox),
    ("seed_oss", RopeLayout::Neox),
    ("smallthinker", RopeLayout::Neox),
    ("smollm3", RopeLayout::Norm),
    ("stablelm", RopeLayout::Neox),
    ("starcoder", RopeLayout::Norm),
    ("starcoder2", RopeLayout::Neox),
    ("step35", RopeLayout::Neox),
    ("talkie", RopeLayout::Neox),
    ("xverse", RopeLayout::Norm),
];

/// Every architecture ferrox actually routes through a RoPE kernel must
/// agree with the reference.
///
/// `ArchPath::Deferred` entries are exempt: `capability::deferred_scope`
/// gives all of them the same placeholder layout, and no graph ever
/// reads it because the load refuses first. Making them agree would be
/// asserting something about code that does not run.
#[test]
fn rope_layout_matches_llama_cpp() {
    let expected: std::collections::HashMap<&str, RopeLayout> =
        LLAMA_ROPE_TYPES.iter().copied().collect();

    let mut checked = 0usize;
    let mut wrong = Vec::new();
    for p in architecture_catalog() {
        if matches!(
            p.path,
            ArchPath::Deferred { .. } | ArchPath::TestFixture { .. }
        ) {
            continue;
        }
        let Some(&want) = expected.get(p.gguf_name) else {
            // A ferrox-only alias (`mistral`, `mixtral`, `yi`, `phi4`,
            // `kimi_k3`, the `granite-*` spellings) has no entry in
            // llama.cpp's own name table, so there is nothing to pin it
            // against.
            continue;
        };
        checked += 1;
        if p.rope != want {
            wrong.push(format!(
                "{}: ferrox {:?}, llama.cpp {:?}",
                p.gguf_name, p.rope, want
            ));
        }
    }

    assert!(
        checked >= 85,
        "only {checked} architectures were actually compared -- the pin has gone vacuous"
    );
    assert!(
        wrong.is_empty(),
        "RoPE layout disagrees with llama.cpp:\n  {}",
        wrong.join("\n  ")
    );
}

/// The transcription itself must not silently shrink.
#[test]
fn the_reference_table_is_complete_enough_to_be_worth_pinning() {
    assert!(
        LLAMA_ROPE_TYPES.len() >= 100,
        "transcribed {} entries from llama_model_rope_type",
        LLAMA_ROPE_TYPES.len()
    );
}
