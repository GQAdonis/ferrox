//! Detect `mmproj-*.gguf` vision towers next to a main GGUF (P7).
//!
//! Ferrox does not yet run VL e2e — this helper only discovers companion
//! projector files the way OpenAI-style multimodal loaders expect, so
//! serve/CLI can fail closed with a clear path once `vl_engine` lands.

use std::path::{Path, PathBuf};

/// Look for `mmproj*.gguf` (case-insensitive) in the same directory as
/// `main_gguf`. Returns the first match by name, if any.
pub fn find_mmproj_beside(main_gguf: &Path) -> Option<PathBuf> {
    let dir = main_gguf.parent()?;
    let mut matches: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| {
                        let lower = n.to_ascii_lowercase();
                        lower.starts_with("mmproj")
                    })
                    .unwrap_or(false)
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn finds_mmproj_next_to_main() {
        let dir = std::env::temp_dir().join(format!("ferrox_mmproj_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let main = dir.join("model-Q4_K_M.gguf");
        let mm = dir.join("mmproj-f16.gguf");
        fs::File::create(&main).unwrap().write_all(b"x").unwrap();
        fs::File::create(&mm).unwrap().write_all(b"y").unwrap();
        let found = find_mmproj_beside(&main).expect("mmproj");
        assert_eq!(found, mm);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn none_when_no_mmproj() {
        let dir = std::env::temp_dir().join(format!("ferrox_nomm_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let main = dir.join("only.gguf");
        fs::File::create(&main).unwrap().write_all(b"x").unwrap();
        assert!(find_mmproj_beside(&main).is_none());
        let _ = fs::remove_dir_all(&dir);
    }
}
