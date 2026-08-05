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

    let mut assembled = String::new();
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let trimmed = line.trim_end();
        if let Some(data) = trimmed.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                if let Some(delta) = v
                    .pointer("/choices/0/delta/content")
                    .and_then(|c| c.as_str())
                {
                    write!(out, "{delta}")?;
                    out.flush()?;
                    assembled.push_str(delta);
                }
                // Non-streaming-shaped chunk with full message
                if assembled.is_empty() {
                    if let Ok(full) = extract_message_content(&v) {
                        write!(out, "{full}")?;
                        out.flush()?;
                        assembled = full;
                    }
                }
            }
        }
        line.clear();
    }
    writeln!(out)?;
    Ok(assembled)
}

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
