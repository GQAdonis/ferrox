//! Server-backed multi-turn chat REPL (`ferrox chat`).
//!
//! Talks to a running `ferrox-server` over HTTP so chat-template wrapping
//! and streaming stay on the validated serve path (no duplicated decode).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde_json::{json, Value};

#[derive(Parser, Debug)]
pub struct ChatArgs {
    /// Base URL of ferrox-server (scheme + host:port).
    #[arg(long, default_value = "http://127.0.0.1:8383")]
    pub url: String,

    /// Model id echoed in requests (server uses the loaded GGUF).
    #[arg(long, default_value = "default")]
    pub model: String,

    /// Optional system prompt prepended once at the start of the session.
    #[arg(long)]
    pub system: Option<String>,

    #[arg(long, default_value_t = 256)]
    pub max_tokens: u32,

    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    /// Use SSE streaming (`stream: true`) and print tokens as they arrive.
    #[arg(long, default_value_t = true)]
    pub stream: bool,

    /// Disable SSE; wait for the full non-streaming JSON response.
    #[arg(long, default_value_t = false)]
    pub no_stream: bool,
}

pub fn run_chat(args: ChatArgs) -> Result<()> {
    let stream = args.stream && !args.no_stream;
    let base = args.url.trim_end_matches('/');
    let health = http_get(&format!("{base}/health"))
        .with_context(|| format!("health check failed for {base}"))?;
    if !(200..300).contains(&health.status) {
        anyhow::bail!("server /health returned HTTP {}", health.status);
    }
    eprintln!("ferrox chat → {base}  (quit with /quit or Ctrl-D)");
    if let Some(sys) = &args.system {
        eprintln!("system: {sys}");
    }

    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = &args.system {
        messages.push(json!({"role": "system", "content": sys}));
    }

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    loop {
        eprint!("> ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        let n = stdin.lock().read_line(&mut line)?;
        if n == 0 {
            eprintln!();
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "/quit" | "/exit" | "/q") {
            break;
        }
        if line == "/clear" {
            messages.clear();
            if let Some(sys) = &args.system {
                messages.push(json!({"role": "system", "content": sys}));
            }
            eprintln!("(history cleared)");
            continue;
        }

        messages.push(json!({"role": "user", "content": line}));
        let body = json!({
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stream": stream,
        });

        let reply = if stream {
            print_streamed_completion(base, &body, &mut stdout)?
        } else {
            let text = post_completion(base, &body)?;
            print!("{text}");
            stdout.flush()?;
            println!();
            text
        };
        messages.push(json!({"role": "assistant", "content": reply}));
    }
    Ok(())
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}

fn parse_url(url: &str) -> Result<(String, u16, String)> {
    let url = url.strip_prefix("http://").ok_or_else(|| {
        anyhow!("only http:// URLs are supported (got {url}); https not implemented in chat client")
    })?;
    let (hostport, path) = match url.split_once('/') {
        Some((hp, p)) => (hp, format!("/{p}")),
        None => (url, "/".to_string()),
    };
    let (host, port) = match hostport.split_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().context("bad port")?),
        None => (hostport.to_string(), 80u16),
    };
    Ok((host, port, path))
}

fn http_exchange(method: &str, url: &str, body: Option<&[u8]>) -> Result<HttpResponse> {
    let (host, port, path) = parse_url(url)?;
    let mut stream = TcpStream::connect((host.as_str(), port))
        .with_context(|| format!("connect {host}:{port}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(600)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;

    let mut req =
        format!("{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n");
    if let Some(b) = body {
        req.push_str("Content-Type: application/json\r\n");
        req.push_str(&format!("Content-Length: {}\r\n", b.len()));
        req.push_str("\r\n");
        stream.write_all(req.as_bytes())?;
        stream.write_all(b)?;
    } else {
        req.push_str("\r\n");
        stream.write_all(req.as_bytes())?;
    }

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Skip headers
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
    }

    let mut body = Vec::new();
    reader.read_to_end(&mut body)?;
    Ok(HttpResponse { status, body })
}

fn http_get(url: &str) -> Result<HttpResponse> {
    http_exchange("GET", url, None)
}

fn post_completion(base: &str, body: &Value) -> Result<String> {
    let bytes = serde_json::to_vec(body)?;
    let resp = http_exchange("POST", &format!("{base}/v1/chat/completions"), Some(&bytes))?;
    if !(200..300).contains(&resp.status) {
        let msg = String::from_utf8_lossy(&resp.body);
        anyhow::bail!("HTTP {}: {msg}", resp.status);
    }
    let v: Value = serde_json::from_slice(&resp.body).context("parse chat response")?;
    extract_message_content(&v)
}

fn print_streamed_completion(base: &str, body: &Value, out: &mut impl Write) -> Result<String> {
    let (host, port, path) = parse_url(&format!("{base}/v1/chat/completions"))?;
    let bytes = serde_json::to_vec(body)?;
    let mut stream = TcpStream::connect((host.as_str(), port))?;
    stream.set_read_timeout(Some(Duration::from_secs(600)))?;
    let header = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        bytes.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(&bytes)?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
    }
    if !(200..300).contains(&status) {
        let mut rest = String::new();
        reader.read_to_string(&mut rest)?;
        anyhow::bail!("HTTP {status}: {rest}");
    }

    let mut render = StreamRender::default();
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let trimmed = line.trim_end();
        if let Some(data) = trimmed.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                render.push_chunk(&v, out)?;
            }
        }
        line.clear();
    }
    render.finish(out)?;
    Ok(render.assembled)
}

/// Turns one SSE chunk into terminal output, and keeps the answer.
///
/// Split out of the socket loop so its three rules can be tested
/// without one.
#[derive(Default)]
struct StreamRender {
    /// The ANSWER only. Deliberately not the chain of thought: this
    /// becomes the assistant turn in the next request's history, and a
    /// replayed chain of thought is both wrong to re-send and rejected
    /// outright by some templates.
    assembled: String,
    /// Whether the dim reasoning run is open, so the escape is written
    /// once per run rather than per delta -- and, more importantly, is
    /// always closed before the answer starts.
    thinking: bool,
}

impl StreamRender {
    fn push_chunk(&mut self, v: &Value, out: &mut impl Write) -> Result<()> {
        // Reading only `delta.content` -- which this did -- makes a
        // reasoning model look like it answered nothing for several
        // seconds and then blurted the answer, because its whole chain
        // of thought arrives on the other field.
        if let Some(thought) = v
            .pointer("/choices/0/delta/reasoning_content")
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
        {
            if !self.thinking {
                self.thinking = true;
                write!(out, "{DIM}")?;
            }
            write!(out, "{thought}")?;
            out.flush()?;
        }
        if let Some(delta) = v
            .pointer("/choices/0/delta/content")
            .and_then(|c| c.as_str())
        {
            // The first content delta closes the thought.
            if self.thinking && !delta.is_empty() {
                self.thinking = false;
                write!(out, "{RESET}")?;
            }
            write!(out, "{delta}")?;
            out.flush()?;
            self.assembled.push_str(delta);
        }
        // Non-streaming-shaped chunk with a full message.
        if self.assembled.is_empty() {
            if let Ok(full) = extract_message_content(v) {
                if self.thinking {
                    self.thinking = false;
                    write!(out, "{RESET}")?;
                }
                write!(out, "{full}")?;
                out.flush()?;
                self.assembled = full;
            }
        }
        Ok(())
    }

    /// A stream that ended mid-thought -- a cancel, a dropped
    /// connection -- must not leave the terminal dim.
    fn finish(&mut self, out: &mut impl Write) -> Result<()> {
        if self.thinking {
            self.thinking = false;
            write!(out, "{RESET}")?;
        }
        writeln!(out)?;
        Ok(())
    }
}

/// Dim, then back to normal. Written around the reasoning run only, so
/// a terminal that ignores them shows the thought as ordinary text
/// rather than as escape codes.
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

fn extract_message_content(v: &Value) -> Result<String> {
    if let Some(s) = v
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
    {
        return Ok(s.to_string());
    }
    if let Some(s) = v
        .pointer("/choices/0/delta/content")
        .and_then(|c| c.as_str())
    {
        return Ok(s.to_string());
    }
    anyhow::bail!("no choices[0].message.content in response: {v}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn thought(text: &str) -> Value {
        json!({"choices": [{"delta": {"reasoning_content": text}}]})
    }

    fn content(text: &str) -> Value {
        json!({"choices": [{"delta": {"content": text}}]})
    }

    fn render(chunks: &[Value]) -> (String, String) {
        let mut out: Vec<u8> = Vec::new();
        let mut r = StreamRender::default();
        for c in chunks {
            r.push_chunk(c, &mut out).expect("render");
        }
        r.finish(&mut out).expect("finish");
        (String::from_utf8(out).expect("utf-8"), r.assembled)
    }

    /// The bug this exists to fix: reading only `delta.content` shows
    /// nothing at all while a reasoning model thinks, so its answer
    /// looks like it arrived out of a long silence.
    #[test]
    fn a_reasoning_models_thoughts_are_shown_rather_than_dropped() {
        let (printed, _) = render(&[thought("weighing "), thought("it up"), content("Paris.")]);
        assert!(printed.contains("weighing it up"), "{printed:?}");
        assert!(printed.contains("Paris."));
    }

    /// The chain of thought is shown and NOT kept. `assembled` becomes
    /// the assistant turn in the next request's history, and replaying
    /// a chain of thought is both wrong to re-send and rejected by some
    /// templates.
    #[test]
    fn the_thought_is_never_part_of_the_answer_that_is_replayed() {
        let (_, assembled) = render(&[thought("weighing it up"), content("Paris.")]);
        assert_eq!(assembled, "Paris.");
    }

    /// One escape per run, not per delta, and the run is closed by the
    /// first content delta -- otherwise the answer itself renders dim.
    #[test]
    fn the_dim_run_is_opened_once_and_closed_before_the_answer() {
        let (printed, _) = render(&[thought("a"), thought("b"), content("X"), content("Y")]);
        assert_eq!(printed.matches(DIM).count(), 1, "{printed:?}");
        let dim_at = printed.find(DIM).expect("opened");
        let reset_at = printed.find(RESET).expect("closed");
        let answer_at = printed.find('X').expect("answered");
        assert!(dim_at < reset_at, "the run opens before it closes");
        assert!(
            reset_at < answer_at,
            "the answer must not be inside the dim run: {printed:?}"
        );
    }

    /// A stream cut off mid-thought -- a cancel, a dropped connection --
    /// must not leave the terminal dim for everything the user types
    /// afterwards.
    #[test]
    fn a_stream_that_ends_mid_thought_still_resets_the_terminal() {
        let (printed, assembled) = render(&[thought("weighing it up")]);
        assert!(printed.ends_with("\x1b[0m\n"), "{printed:?}");
        assert!(assembled.is_empty());
    }

    /// A server with no reasoning to report is unchanged: no escapes at
    /// all, so a terminal that does not understand them sees nothing new.
    #[test]
    fn a_plain_answer_is_printed_without_any_escapes() {
        let (printed, assembled) = render(&[content("hello "), content("world")]);
        assert_eq!(printed, "hello world\n");
        assert_eq!(assembled, "hello world");
    }
}
