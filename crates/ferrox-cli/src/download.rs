//! `ferrox download <repo> [file] [--local-dir DIR]`
//!
//! Deliberately the same shape as `hf download`, so a command copied
//! off a model card runs unchanged. The point is that it needs no
//! Python: `ferrox pull` used to shell out to the `hf` CLI, so a Rust
//! engine could not fetch its own weights without `pip install
//! huggingface_hub`.
//!
//! The transport is [`ferrox_models::hub`], which `ferrox-server`
//! already used for `POST /admin/download`: IPv4 resolved first because
//! Hugging Face publishes AAAA records that black-hole on some
//! networks, `HF_TOKEN` for gated repos, `HF_ENDPOINT` for mirrors, and
//! a byte range so an interrupted download resumes.
use std::io::Write;
use std::path::PathBuf;

#[derive(clap::Args, Debug)]
pub struct DownloadArgs {
    /// Hugging Face repo id, for example `bartowski/Llama-3.2-3B-Instruct-GGUF`.
    pub repo: String,

    /// File to fetch, or a glob. Defaults to the repo's single GGUF,
    /// and says so rather than choosing when several match.
    #[arg(default_value = "*.gguf")]
    pub file: String,

    /// Where to put it. `hf download` spells this the same way.
    #[arg(long = "local-dir", default_value = "models")]
    pub local_dir: PathBuf,
}

pub fn run(args: DownloadArgs) -> anyhow::Result<()> {
    let mut last = std::time::Instant::now();
    // Throttled: a fast link should spend its time on bytes, not on
    // formatting a line nobody can read at 300 Hz.
    let mut draw = move |done: u64, total: Option<u64>| {
        if last.elapsed() < std::time::Duration::from_millis(200) {
            return;
        }
        last = std::time::Instant::now();
        let mib = done as f64 / 1024.0 / 1024.0;
        match total {
            Some(t) if t > 0 => {
                print!(
                    "\r  {mib:>9.1} MiB  {:5.1}%",
                    (done as f64 / t as f64) * 100.0
                )
            }
            _ => print!("\r  {mib:>9.1} MiB"),
        }
        let _ = std::io::stdout().flush();
    };

    let path = ferrox_models::hub::fetch_to_dir_with_progress(
        &args.repo,
        &args.file,
        &args.local_dir,
        &mut draw,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!();
    let gb = std::fs::metadata(&path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    println!("saved {} ({gb:.2} GiB)", path.display());
    Ok(())
}
