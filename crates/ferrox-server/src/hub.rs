//! The one outbound HTTP client in `ferrox-server`, used only by
//! `/admin/download`.
//!
//! ## Why a client dependency at all
//!
//! The workspace had no HTTP *client* before this: `axum`/`hyper` are
//! the server side, and `ferrox_models::hf_pull` shells out to the
//! `huggingface_hub` CLI, which is not installed on most machines, is
//! not resumable under our control, and reports progress only to its
//! own terminal. A download the UI can show a real progress bar for
//! needs the byte counter in-process, so `ureq` (blocking, rustls) is
//! added -- deliberately with `default-features = false` so it brings
//! rustls and webpki-roots, most of which the TLS-serving side already
//! pulled in, and not a second async runtime.
//!
//! Blocking on purpose: every call here runs on `spawn_blocking`
//! alongside the file writes it feeds, so the Tokio reactor is never
//! parked on a socket read.
//!
//! ## Redirects and resume
//!
//! `…/resolve/main/<file>` redirects to a CDN host. Whether a `Range`
//! header survives that hop is not something this code assumes: it asks
//! for a range and then believes the *status*. Only a `206` counts as a
//! resume; a `200` means the server is sending the whole file from byte
//! zero, and the caller truncates rather than appending a second copy of
//! the prefix onto the first.

use std::io::Read;
use std::time::Duration;

/// Hub base URL. `HF_ENDPOINT` is the same variable the official client
/// reads, for mirrors and air-gapped caches. It is operator
/// configuration, never taken from a request body.
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|e| e.trim_end_matches('/').to_string())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

fn token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .iter()
        .find_map(|k| std::env::var(k).ok())
        .filter(|t| !t.trim().is_empty())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        // Generous: a cold CDN edge can take a while to answer, but a
        // dead host must not hold a blocking thread forever.
        .timeout_connect(Duration::from_secs(20))
        .build()
}

fn with_auth(req: ureq::Request) -> ureq::Request {
    match token() {
        Some(t) => req.set("Authorization", &format!("Bearer {t}")),
        None => req,
    }
}

/// Turns a `*` glob into a concrete filename by asking the Hub what the
/// repo contains.
///
/// Sorted, first match wins, so the choice is deterministic for the
/// same repo rather than depending on the order the API happens to
/// return. A repo with several matching quantizations is ambiguous by
/// construction; the caller can name one exactly to be sure.
pub(crate) fn resolve_glob(repo: &str, pattern: &str) -> Result<String, String> {
    let url = format!("{}/api/models/{repo}", endpoint());
    let response = with_auth(agent().get(&url))
        .call()
        .map_err(|e| format!("listing {repo} on the Hub failed: {e}"))?;
    let body = response
        .into_string()
        .map_err(|e| format!("reading the file list for {repo}: {e}"))?;
    let value: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("the Hub's file list for {repo} was not JSON: {e}"))?;
    let mut names: Vec<String> = value
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|e| e.get("rfilename")?.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
        .into_iter()
        // A repo may hold shards in subdirectories; a target this
        // server can safely write is a bare name in the repo root.
        .filter(|n| !n.contains('/'))
        .find(|n| crate::admin::glob_matches(pattern, n))
        .ok_or_else(|| format!("no file in {repo} matches '{pattern}'"))
}

/// An open response body plus what the caller needs to write it.
pub(crate) struct HubFile {
    pub(crate) body: Box<dyn Read + Send>,
    /// Total size of the *whole* file when the server states one, not
    /// of the remaining range: the caller's progress counter is
    /// cumulative. `None` when there is no usable `Content-Length`,
    /// which is a real case and means an indeterminate progress bar
    /// rather than a fabricated total.
    pub(crate) total_bytes: Option<u64>,
    /// True only on a `206`. See the module docs.
    pub(crate) resumed: bool,
}

/// Opens `repo`/`filename` for reading, asking to resume at
/// `resume_from` when that is non-zero.
pub(crate) fn open_file(repo: &str, filename: &str, resume_from: u64) -> Result<HubFile, String> {
    let url = format!("{}/{repo}/resolve/main/{filename}", endpoint());
    let mut req = with_auth(agent().get(&url));
    if resume_from > 0 {
        req = req.set("Range", &format!("bytes={resume_from}-"));
    }
    let response = req
        .call()
        .map_err(|e| format!("downloading {filename} from {repo} failed: {e}"))?;

    let status = response.status();
    let resumed = status == 206 && resume_from > 0;
    let total_bytes = if resumed {
        content_range_total(response.header("content-range"))
    } else {
        response
            .header("content-length")
            .and_then(|v| v.trim().parse::<u64>().ok())
    };
    Ok(HubFile {
        body: Box::new(response.into_reader()),
        total_bytes,
        resumed,
    })
}

/// Total size out of `Content-Range: bytes 100-999/1000`. `None` for
/// the `*/unknown` form, which is legal and means exactly that.
fn content_range_total(header: Option<&str>) -> Option<u64> {
    let total = header?.rsplit('/').next()?.trim();
    total.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_content_range_yields_the_whole_file_size_not_the_slice() {
        assert_eq!(content_range_total(Some("bytes 100-999/1000")), Some(1000));
    }

    #[test]
    fn an_unknown_content_range_total_is_not_invented() {
        assert_eq!(content_range_total(Some("bytes 0-99/*")), None);
        assert_eq!(content_range_total(None), None);
        assert_eq!(content_range_total(Some("garbage")), None);
    }

    #[test]
    fn the_endpoint_has_no_trailing_slash_to_double_up() {
        // Default, and any override, must join cleanly with "/api/...".
        assert!(!endpoint().ends_with('/'));
    }
}
