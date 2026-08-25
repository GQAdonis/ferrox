//! `ferrox serve-bench`: what `ferrox-server` does under concurrency.
//!
//! `ferrox bench` measures kernels against `llama-bench` and is
//! deliberately single-stream and HTTP-free. That is the right tool for
//! "how fast is this matvec" and no tool at all for "what is the p99
//! time-to-first-token at sixteen concurrent clients", which is the
//! question an operator sizing a deployment is actually asking.
//!
//! Every rule that decides whether the numbers mean anything lives in
//! [`ferrox_edge::bench_client`], with no socket in it, so each one is
//! asserted on any host rather than inferred from a live server that
//! might have been slow for an unrelated reason. This module is the
//! socket: it opens connections, dispatches on a schedule, tics per
//! streamed chunk, and hands the tics to that arithmetic.
//!
//! The HTTP client is hand-rolled on `TcpStream` for the same reason
//! `chat` is: the CLI links no HTTP stack, and a benchmark that pulled
//! in a TLS-capable one would measure the client's connection handling
//! as much as the server's.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use ferrox_edge::{is_token_chunk, BenchReport, BenchSampling, Latency, RequestTiming};
use serde_json::{json, Value};

#[derive(Parser, Debug)]
pub struct ServeBenchArgs {
    /// Base URL of a running `ferrox-server`.
    #[arg(long, default_value = "http://127.0.0.1:8383")]
    pub url: String,
    /// Requests to send in total.
    #[arg(long, default_value_t = 32)]
    pub requests: usize,
    /// Requests in flight at once.
    #[arg(long, default_value_t = 8)]
    pub concurrency: usize,
    /// Tokens every request must produce, exactly.
    #[arg(long, default_value_t = 128)]
    pub output_len: usize,
    /// Prompt length, in characters of the generated filler.
    ///
    /// Characters and not tokens: an exact TOKEN length needs the
    /// server's own tokenizer in a fixed-point loop (see
    /// `ferrox_edge::prompt_of_exact_length`), which is a round trip per
    /// step against a server this command is about to benchmark. The
    /// reported prompt-token count is the server's, so a run says what
    /// it really sent rather than what it meant to.
    #[arg(long, default_value_t = 512)]
    pub prompt_chars: usize,
    /// Model name to send. Optional: the server serves whatever is
    /// loaded regardless.
    #[arg(long)]
    pub model: Option<String>,
    /// Emit the report as JSON instead of a table.
    #[arg(long)]
    pub json: bool,
}

pub fn run_serve_bench(args: ServeBenchArgs) -> Result<()> {
    if args.requests == 0 {
        anyhow::bail!("--requests must be at least 1");
    }
    if args.concurrency == 0 {
        anyhow::bail!("--concurrency must be at least 1");
    }
    if args.output_len == 0 {
        anyhow::bail!("--output-len must be at least 1: a request that generates nothing has no latency to measure");
    }
    let base = args.url.trim_end_matches('/').to_string();
    // Fail before the run rather than N times inside it: a connection
    // refused per worker buries the one useful line under noise.
    parse_url(&format!("{base}/v1/chat/completions"))?;

    let sampling = BenchSampling::new(args.output_len);
    let prompt = filler_prompt(args.prompt_chars);
    let started = Instant::now();

    // A shared work counter rather than a slice per worker: requests
    // are not equal-cost, so a static split leaves the fastest worker
    // idle while the slowest still has a queue -- which shows up as the
    // server being slower at the end of a run than the start.
    let (tx, rx) = mpsc::channel::<usize>();
    for i in 0..args.requests {
        tx.send(i)
            .expect("the receiver is alive until the scope ends");
    }
    drop(tx);
    let rx = std::sync::Mutex::new(rx);

    let timings: Vec<RequestTiming> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..args.concurrency.min(args.requests))
            .map(|_| {
                let base = &base;
                let prompt = &prompt;
                let model = &args.model;
                let rx = &rx;
                scope.spawn(move || {
                    let mut mine = Vec::new();
                    loop {
                        let next = rx.lock().unwrap_or_else(|p| p.into_inner()).recv();
                        if next.is_err() {
                            break;
                        }
                        let body = request_body(model.as_deref(), prompt, &sampling);
                        let dispatched = started.elapsed().as_secs_f64();
                        let mut timing = RequestTiming::started(dispatched);
                        // A failed request contributes a timing with no
                        // tics, which the report counts as failed rather
                        // than dropping: a run where a third of the
                        // requests failed and the rest were fast is not
                        // a fast run.
                        if let Err(e) = stream_request(base, &body, &started, &mut timing) {
                            eprintln!("request failed: {e:#}");
                        }
                        mine.push(timing);
                    }
                    mine
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect()
    });

    let report = BenchReport::of(&timings);
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report_json(&report, &args))?
        );
    } else {
        print_report(&report, &args);
    }
    if report.completed == 0 {
        anyhow::bail!("every request failed; nothing was measured");
    }
    Ok(())
}

/// The body every benchmark request sends.
///
/// `stream: true` is not a preference: TTFT is only observable on a
/// stream, and a buffered request can report nothing but end-to-end.
fn request_body(model: Option<&str>, prompt: &str, sampling: &BenchSampling) -> Value {
    json!({
        "model": model.unwrap_or("bench"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": sampling.output_len,
        "temperature": sampling.temperature,
        "top_k": sampling.top_k,
        "ignore_eos": sampling.ignore_eos,
        "stream": true,
    })
}

/// Filler with no structure a template or a cache could shortcut.
///
/// Deliberately varied rather than one repeated word: a prompt of
/// `"a a a a"` is the best case for any prefix cache and several
/// tokenizers, so a run built on one measures the cache rather than the
/// server.
fn filler_prompt(chars: usize) -> String {
    const WORDS: [&str; 12] = [
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
        "eiusmod",
        "tempor",
    ];
    let mut out = String::with_capacity(chars + 16);
    let mut i = 0usize;
    while out.len() < chars {
        if !out.is_empty() {
            out.push(' ');
        }
        // A cheap deterministic walk: reproducible between runs, which
        // a benchmark needs, without pulling in an RNG.
        out.push_str(WORDS[(i * 7 + i * i) % WORDS.len()]);
        i += 1;
    }
    out.truncate(chars);
    out
}

/// Sends one streamed request and tics per token chunk.
fn stream_request(
    base: &str,
    body: &Value,
    started: &Instant,
    timing: &mut RequestTiming,
) -> Result<()> {
    let (host, port, path) = parse_url(&format!("{base}/v1/chat/completions"))?;
    let bytes = serde_json::to_vec(body)?;
    let stream = TcpStream::connect((host.as_str(), port))
        .with_context(|| format!("connect {host}:{port}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(600)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let mut stream = stream;

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
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
    }
    if !(200..300).contains(&status) {
        let mut rest = String::new();
        use std::io::Read;
        reader.read_to_string(&mut rest)?;
        anyhow::bail!("HTTP {status}: {rest}");
    }

    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let trimmed = line.trim_end();
        if let Some(data) = trimmed.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                // The rule this whole command depends on: a keepalive
                // arrives during exactly the window TTFT measures, and
                // the terminal frame carries no token. See
                // `is_token_chunk` for what each would corrupt.
                if is_token_chunk(&chunk) {
                    timing.tic(started.elapsed().as_secs_f64());
                }
                // The terminal chunk carries the server's own token
                // count. Taken in preference to the chunk count,
                // because a buffered answer arrives as one chunk and
                // was still N tokens of work.
                if let Some(n) = chunk
                    .get("usage")
                    .and_then(|u| u.get("completion_tokens"))
                    .and_then(serde_json::Value::as_u64)
                {
                    timing.report_tokens(n as usize);
                }
            }
        }
        line.clear();
    }
    Ok(())
}

fn parse_url(url: &str) -> Result<(String, u16, String)> {
    let url = url.strip_prefix("http://").ok_or_else(|| {
        anyhow!("only http:// URLs are supported (got {url}); https is not implemented here")
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

fn ms(seconds: Option<f64>) -> String {
    match seconds {
        Some(s) => format!("{:.1}", s * 1000.0),
        // A dash and not a zero: a run with no samples measured
        // nothing, and a zero reads as instantaneous.
        None => "-".to_string(),
    }
}

fn latency_json(l: &Latency) -> Value {
    json!({
        "mean_ms": l.mean.map(|v| v * 1000.0),
        "p50_ms": l.p50.map(|v| v * 1000.0),
        "p90_ms": l.p90.map(|v| v * 1000.0),
        "p99_ms": l.p99.map(|v| v * 1000.0),
    })
}

fn report_json(report: &BenchReport, args: &ServeBenchArgs) -> Value {
    json!({
        "requests": args.requests,
        "concurrency": args.concurrency,
        "output_len": args.output_len,
        "completed": report.completed,
        "failed": report.failed,
        "output_tokens": report.output_tokens,
        "duration_s": report.duration_s,
        "output_throughput_tps": report.output_throughput(),
        "ttft": latency_json(&report.ttft),
        "tpot": latency_json(&report.tpot),
        "end_to_end": latency_json(&report.end_to_end),
    })
}

fn print_report(report: &BenchReport, args: &ServeBenchArgs) {
    println!(
        "\nserve-bench  {} requests, concurrency {}, {} tokens each",
        args.requests, args.concurrency, args.output_len
    );
    println!(
        "completed {}  failed {}  tokens {}  span {:.2}s",
        report.completed, report.failed, report.output_tokens, report.duration_s
    );
    match report.output_throughput() {
        Some(tps) => println!("output throughput: {tps:.1} tok/s (whole run)"),
        None => println!("output throughput: - (nothing completed)"),
    }
    println!();
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>10}",
        "", "mean", "p50", "p90", "p99"
    );
    for (name, l) in [
        ("TTFT (ms)", &report.ttft),
        ("TPOT (ms)", &report.tpot),
        ("E2E (ms)", &report.end_to_end),
    ] {
        println!(
            "{:<12} {:>10} {:>10} {:>10} {:>10}",
            name,
            ms(l.mean),
            ms(l.p50),
            ms(l.p90),
            ms(l.p99)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every field the methodology depends on has to actually reach the
    /// wire. `ignore_eos` is the one worth a test of its own: without
    /// it the requests finish at different lengths and the slowest
    /// percentile is whichever prompt happened to run longest.
    #[test]
    fn a_bench_request_pins_everything_the_methodology_depends_on() {
        let body = request_body(Some("m"), "hello", &BenchSampling::new(64));
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["top_k"], 1);
        assert_eq!(body["ignore_eos"], true);
        assert_eq!(
            body["stream"], true,
            "TTFT is only observable on a stream; a buffered request can \
             report nothing but end-to-end"
        );
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    /// Reproducible between runs, and not one repeated word: a prompt
    /// of `"a a a a"` is the best case for any prefix cache and several
    /// tokenizers, so a run built on one measures the cache.
    #[test]
    fn the_filler_prompt_is_reproducible_and_not_one_repeated_word() {
        let a = filler_prompt(200);
        assert_eq!(a.len(), 200);
        assert_eq!(
            a,
            filler_prompt(200),
            "the same run twice sends the same prompt"
        );

        let distinct: std::collections::HashSet<&str> = a.split_whitespace().collect();
        assert!(
            distinct.len() > 3,
            "a prompt of one repeated word measures the prefix cache: {a}"
        );

        assert!(filler_prompt(0).is_empty());
    }

    /// A missing figure prints as a dash, never as a zero: a zero is
    /// the best possible latency and would make a run in which nothing
    /// answered look like the fastest one.
    #[test]
    fn a_latency_that_was_never_measured_prints_as_a_dash() {
        assert_eq!(ms(None), "-");
        assert_eq!(ms(Some(0.0125)), "12.5");
    }

    #[test]
    fn only_http_urls_are_accepted_and_the_default_port_is_80() {
        let (host, port, path) = parse_url("http://127.0.0.1:8383/v1/chat/completions").unwrap();
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 8383);
        assert_eq!(path, "/v1/chat/completions");

        let (host, port, path) = parse_url("http://example.test").unwrap();
        assert_eq!(
            (host.as_str(), port, path.as_str()),
            ("example.test", 80, "/")
        );

        assert!(parse_url("https://example.test").is_err());
    }
}
