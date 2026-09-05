//! Helpers shared by the integration tests.
//!
//! Each of these had grown two copies. That matters more in a test than
//! in library code: a tolerance or a comparison that drifts between two
//! copies makes one suite quietly weaker than the other, and a test
//! that has stopped checking what it claims to check looks exactly like
//! a test that passes.
//!
//! Rust compiles every file under `tests/` as its own crate, so a
//! helper only one suite uses is dead code in the others. `#[allow]` on
//! the module rather than on each item, since which suite uses what is
//! not a property worth maintaining.

#![allow(dead_code)]

use ferrox_core::cache::KvCache;
use ferrox_models::{Decoder, ModelConfig};
use std::path::{Path, PathBuf};

/// Fails with the worst absolute difference and the first few values of
/// each side.
///
/// The worst difference rather than the first: a reference mismatch is
/// almost never at index 0, and reporting the first divergence hides
/// how bad it gets. The sample of both sides is what turns "0.03 > 0.01"
/// into a diagnosis.
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: logit count");
    let worst = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst <= tol,
        "{what}: max |ferrox - llama.cpp| = {worst} > {tol}\n  ferrox: {:?}\n  llama:  {:?}",
        &got[..8.min(got.len())],
        &want[..8.min(want.len())]
    );
}

/// Every `.gguf` under `dir`, recursively.
///
/// A missing or unreadable directory yields nothing rather than
/// failing: these suites scan an optional local model directory, and a
/// machine that has none should skip rather than error.
pub fn collect_gguf(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_gguf(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("gguf") {
            out.push(p);
        }
    }
}

// --- llama.cpp-golden graph fixtures --------------------------------
//
// The harness behind `one_match_arm_graphs.rs` and
// `fixture_away_graphs.rs`. Both suites admit architectures to
// `capability::AUDITED_GENERIC_GQA` on the same evidence -- a tiny
// synthetic GGUF whose golden values come from llama.cpp's own graph via
// libllama -- so they must compare on the same forward paths, at the
// same tolerance, from the same prompt. Two copies of that would be two
// standards, and the weaker one would be invisible.

/// The token ids every graph fixture is driven with. Explicit ids, never
/// tokenized text: the fixtures' vocabularies are placeholders.
pub const GRAPH_PROMPT: [usize; 6] = [3, 7, 11, 19, 23, 5];

/// Float32 accumulation order differs between the two engines (ggml
/// blocks its matmuls; ferrox does not), so this is a numeric-agreement
/// tolerance, not a bit-exactness claim. The measured worst case across
/// the fixtures is ~1e-6; every sabotage test moves the outputs by orders
/// of magnitude more.
pub const GRAPH_TOL: f32 = 1e-5;

pub fn graph_fixture_path(name: &str) -> String {
    format!(
        "{}/tests/fixtures/{name}_tiny.gguf",
        env!("CARGO_MANIFEST_DIR")
    )
}

pub fn load_graph_fixture(name: &str) -> Decoder {
    let path = graph_fixture_path(name);
    let file = ferrox_gguf::GgufFile::open(&path).expect("fixture opens");
    let config = ModelConfig::from_gguf(&file).expect("fixture config parses");
    Decoder::from_gguf(&path, config).expect("fixture loads")
}

pub fn graph_caches(decoder: &Decoder) -> Vec<KvCache> {
    decoder
        .layers
        .iter()
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect()
}

/// Prefill, decode and continuous batching are three separate bodies in
/// this repo, and the reason this helper runs all three on every
/// architecture is that they have diverged before: five model features
/// went missing from one of them while the others kept working.
pub fn assert_all_three_paths_match(name: &str, golden: &[f32]) {
    let decoder = load_graph_fixture(name);

    let mut kv = graph_caches(&decoder);
    assert_close(
        &decoder.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        golden,
        GRAPH_TOL,
        &format!("{name}: prefill (forward_batch_last)"),
    );

    let mut kv = graph_caches(&decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    assert_close(&out, golden, GRAPH_TOL, &format!("{name}: decode"));

    let mut kv = vec![graph_caches(&decoder)];
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        let batch = decoder.forward_multi_seq(&[tok], &[pos], &mut kv);
        out = batch.into_iter().next().unwrap();
    }
    assert_close(&out, golden, GRAPH_TOL, &format!("{name}: multi-seq"));
}

/// The worst absolute difference from the golden values, for the
/// sabotage tests: a mutation that does not move this is a mutation the
/// suite cannot see.
pub fn worst_vs(got: &[f32], golden: &[f32]) -> f32 {
    got.iter()
        .zip(golden.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}
