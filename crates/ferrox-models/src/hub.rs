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
use std::net::{SocketAddr, ToSocketAddrs};
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

/// Resolves a host and returns its addresses **IPv4 first**.
///
/// Not cosmetic. `huggingface.co` publishes eight AAAA records and four
/// A records, and the client tries them in order with a *halving* share
/// of the connect deadline each time. On a host with no IPv6 route --
/// which is most laptops, most CI, and every container run without
/// `--ipv6` -- every AAAA attempt burns until it times out, and the
/// budget is exhausted before the first working A record is reached.
/// The observable symptom is a download that sits at zero bytes and
/// then fails with "connection timed out" on a host where `curl` to the
/// same URL succeeds instantly, because curl does Happy Eyeballs and
/// this client does not.
///
/// Ordering rather than filtering: an IPv6-only host still gets its
/// AAAA records, just after the A records, and a connect to an
/// unreachable IPv4 address fails immediately (`ENETUNREACH`) rather
/// than hanging, so the cost of the wrong guess is bounded in the
/// direction that matters.
fn resolve_ipv4_first(netloc: &str) -> std::io::Result<Vec<SocketAddr>> {
    let mut addrs: Vec<SocketAddr> = netloc.to_socket_addrs()?.collect();
    // Stable sort: the resolver's own ordering is preserved inside each
    // family, so DNS round-robin still spreads load.
    addrs.sort_by_key(|addr| !addr.is_ipv4());
    Ok(addrs)
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        // Enough for a cold CDN edge to answer, short enough that a
        // black-holed address does not hold a blocking thread for
        // anything like a user's patience.
        .timeout_connect(Duration::from_secs(10))
        .resolver(resolve_ipv4_first)
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
pub fn resolve_glob(repo: &str, pattern: &str) -> Result<String, String> {
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
        .find(|n| glob_matches(pattern, n))
        .ok_or_else(|| format!("no file in {repo} matches '{pattern}'"))
}

/// An open response body plus what the caller needs to write it.
pub struct HubFile {
    pub body: Box<dyn Read + Send>,
    /// Total size of the *whole* file when the server states one, not
    /// of the remaining range: the caller's progress counter is
    /// cumulative. `None` when there is no usable `Content-Length`,
    /// which is a real case and means an indeterminate progress bar
    /// rather than a fabricated total.
    pub total_bytes: Option<u64>,
    /// True only on a `206`. See the module docs.
    pub resumed: bool,
}

/// Opens `repo`/`filename` for reading, asking to resume at
/// `resume_from` when that is non-zero.
pub fn open_file(repo: &str, filename: &str, resume_from: u64) -> Result<HubFile, String> {
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

/// Matches a `*`-glob against a filename. `*` matches any run of
/// characters including none; every other character is literal.
///
/// Enough for the Hub filename patterns people actually type
/// (`*.gguf`, `*Q4_K_M*.gguf`) and small enough to read in one go.
pub fn glob_matches(pattern: &str, name: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return pattern == name;
    }
    let mut rest = name;
    if let Some(first) = segments.first() {
        let Some(stripped) = rest.strip_prefix(first) else {
            return false;
        };
        rest = stripped;
    }
    if let Some(last) = segments.last() {
        let Some(stripped) = rest.strip_suffix(last) else {
            return false;
        };
        // A pattern like "a*b" over "ab" leaves nothing between: fine.
        rest = stripped;
    }
    for middle in segments
        .iter()
        .skip(1)
        .take(segments.len().saturating_sub(2))
    {
        match rest.find(*middle) {
            Some(at) => rest = &rest[at + middle.len()..],
            None => return false,
        }
    }
    true
}

/// Resolve `pattern` in `repo` and write the matching file into `dir`,
/// resuming an interrupted download rather than starting over.
///
/// Returns the path written. Progress goes to `on_progress` so a CLI
/// can draw a bar and a server can stay silent.
pub fn fetch_to_dir_with_progress(
    repo: &str,
    pattern: &str,
    dir: &std::path::Path,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<std::path::PathBuf, String> {
    use std::io::{Read, Write};

    let filename = resolve_glob(repo, pattern)?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let final_path = dir.join(&filename);

    // Resume needs the partial bytes to survive an interruption, but a
    // partial file under the final name would later be opened as if it
    // were a whole GGUF. So it lands beside the real name and is
    // renamed only once the last byte is in.
    let partial = dir.join(format!("{filename}.partial"));
    if final_path.exists() {
        return Ok(final_path);
    }
    let resume_from = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

    let mut hub = open_file(repo, &filename, resume_from)?;
    // A server that ignores `Range` answers 200 with the whole file.
    // Appending in that case would corrupt it, so what actually
    // happened is read off the response, not off what was asked for.
    let mut done = if hub.resumed { resume_from } else { 0 };
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(hub.resumed)
        .truncate(!hub.resumed)
        .open(&partial)
        .map_err(|e| format!("{}: {e}", partial.display()))?;

    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = hub
            .body
            .read(&mut buf)
            .map_err(|e| format!("reading {filename} from {repo} failed: {e}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .map_err(|e| format!("{}: {e}", partial.display()))?;
        done += n as u64;
        on_progress(done, hub.total_bytes);
    }
    out.flush().map_err(|e| e.to_string())?;
    // The rename must not outrun the bytes it publishes.
    out.sync_all().map_err(|e| e.to_string())?;
    drop(out);

    // A short file is a truncated download. Renaming it would hand the
    // loader a GGUF that ends in the middle of a tensor, so it stays
    // under the partial name and the next run resumes it.
    if let Some(total) = hub.total_bytes {
        if done != total {
            return Err(format!(
                "{filename} stopped at {done} of {total} bytes. The partial file is kept \
                 at {}, so running this again resumes rather than restarting.",
                partial.display()
            ));
        }
    }

    std::fs::rename(&partial, &final_path).map_err(|e| e.to_string())?;
    Ok(final_path)
}

/// [`fetch_to_dir_with_progress`] without the progress callback.
pub fn fetch_to_dir(
    repo: &str,
    pattern: &str,
    dir: &std::path::Path,
) -> Result<std::path::PathBuf, String> {
    fetch_to_dir_with_progress(repo, pattern, dir, |_, _| {})
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
    fn resolution_puts_every_ipv4_address_ahead_of_every_ipv6_one() {
        // localhost resolves to both families on most hosts; the shape
        // of the answer is what matters, not which addresses appear.
        let addrs = resolve_ipv4_first("localhost:443").expect("localhost must resolve");
        assert!(!addrs.is_empty());
        let first_v6 = addrs.iter().position(|a| !a.is_ipv4());
        let last_v4 = addrs.iter().rposition(|a| a.is_ipv4());
        if let (Some(first_v6), Some(last_v4)) = (first_v6, last_v4) {
            assert!(
                last_v4 < first_v6,
                "an IPv6 address was ordered ahead of an IPv4 one: {addrs:?}"
            );
        }
    }

    #[test]
    fn the_endpoint_has_no_trailing_slash_to_double_up() {
        // Default, and any override, must join cleanly with "/api/...".
        assert!(!endpoint().ends_with('/'));
    }
}
