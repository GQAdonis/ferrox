//! Writing the two logit vectors a parity run compared, so the
//! comparison can be redone without either engine.
//!
//! Why this exists: `parity` prints ONE number per run, KL(reference ||
//! ferrox), and that number is a distance between two points without
//! saying where either point is. When the same checkpoint scored DRIFT
//! against one libllama and WRONG against another
//! ([#102](https://github.com/antonellof/ferrox/issues/102)) there was
//! no way, from parity's output alone, to tell "ferrox moved" from "the
//! reference moved" — the two hypotheses produce the same printed line.
//!
//! Dumping the vectors makes the third comparison possible, and it is
//! the one that settles it: reference-A against reference-B, with ferrox
//! out of the experiment entirely. On Qwen2.5-1.5B Q4_K_M that
//! comparison read KL 2.73e-2 between two llama.cpp builds, larger than
//! either build's disagreement with ferrox — which is a fact about
//! llama.cpp and could not have been discovered by running `parity`
//! more times.
//!
//! The files are raw little-endian f32, the same wire
//! `tools/llama_logits.c` writes, so a dumped ferrox vector and a
//! dumped reference vector are interchangeable inputs to anything that
//! reads one.

use anyhow::Context;
use std::path::{Path, PathBuf};

/// Suffixes appended to the caller's prefix. Kept in one place because
/// the point of the dump is that a later run can find the earlier run's
/// files, and a suffix spelled twice is a suffix that drifts.
const REFERENCE_SUFFIX: &str = ".llama.f32";
const FERROX_SUFFIX: &str = ".ferrox.f32";
const TOKENS_SUFFIX: &str = ".tokens.txt";

/// Writes both logit vectors and the token ids they were computed from.
///
/// The token ids travel with the vectors because they are the only part
/// of the experiment that is not in the file names: two dumps of the
/// same checkpoint taken with different prompts are not comparable, and
/// nothing else would say so.
pub fn write(
    prefix: &str,
    tokens: &[u32],
    reference: &[f32],
    ferrox: &[f32],
) -> anyhow::Result<[PathBuf; 3]> {
    let ref_path = PathBuf::from(format!("{prefix}{REFERENCE_SUFFIX}"));
    let fx_path = PathBuf::from(format!("{prefix}{FERROX_SUFFIX}"));
    let tok_path = PathBuf::from(format!("{prefix}{TOKENS_SUFFIX}"));

    write_f32(&ref_path, reference)?;
    write_f32(&fx_path, ferrox)?;

    let ids = tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    std::fs::write(&tok_path, format!("{ids}\n"))
        .with_context(|| format!("writing {}", tok_path.display()))?;

    Ok([ref_path, fx_path, tok_path])
}

fn write_f32(path: &Path, values: &[f32]) -> anyhow::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &bytes).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dumped vector must read back bit-for-bit, because the whole
    /// use of the dump is comparing it against a vector produced by a
    /// DIFFERENT build months later. A dump that rounds is a dump that
    /// invents the divergence it is meant to measure.
    #[test]
    fn a_dumped_vector_reads_back_bit_for_bit() {
        let dir = std::env::temp_dir().join(format!("ferrox-dump-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("case").to_string_lossy().into_owned();

        // Values chosen to break a naive round-trip: a denormal, a
        // negative zero, and a value whose f32 -> f64 -> f32 trip is
        // only exact if nothing in between is text.
        let reference = vec![
            0.1f32,
            -0.0,
            f32::MIN_POSITIVE,
            3.402_823_5e38,
            -1.234_567_8e-9,
        ];
        let ferrox = vec![0.2f32, 1.0, -5.5, 0.0, 7.7];
        let paths = write(&prefix, &[1, 2, 3], &reference, &ferrox).unwrap();

        let back = std::fs::read(&paths[0]).unwrap();
        let read: Vec<f32> = back
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(read.len(), reference.len());
        for (a, b) in read.iter().zip(&reference) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "dumped {b} came back as {a}: the dump is not bit-exact"
            );
        }

        let ids = std::fs::read_to_string(&paths[2]).unwrap();
        assert_eq!(ids.trim(), "1 2 3");

        for p in &paths {
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_dir(&dir);
    }

    /// The reference and ferrox dumps must land in different files.
    ///
    /// They are the same length and the same wire format, so a shared
    /// suffix would silently leave one overwriting the other and the
    /// resulting "reference vs reference" KL would be exactly zero —
    /// which reads as "the two builds agree" and is the one answer this
    /// tool must never fabricate.
    #[test]
    fn the_two_vectors_do_not_share_a_file_name() {
        assert_ne!(REFERENCE_SUFFIX, FERROX_SUFFIX);
        assert_ne!(REFERENCE_SUFFIX, TOKENS_SUFFIX);
        assert_ne!(FERROX_SUFFIX, TOKENS_SUFFIX);
    }
}
