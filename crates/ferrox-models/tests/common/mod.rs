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
