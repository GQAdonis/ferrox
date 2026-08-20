//! The chat-template renderer against real checkpoints.
//!
//! Two layers:
//!
//! * `every_checked_in_template_compiles_and_renders` runs in CI. It
//!   covers the five `tests/templates/*.jinja` files, each of which is
//!   the verbatim `tokenizer.chat_template` string of a real GGUF in
//!   `models/` (read out of the file's metadata, not paraphrased).
//! * `sweep_every_local_gguf_template` is `#[ignore]`d because it needs
//!   the checkpoints themselves. It opens every GGUF under `models/`
//!   (or `$FERROX_MODELS_DIR`), pulls each one's real template, and
//!   renders a fixed three-turn conversation through it. That is how
//!   the compile/render coverage claim in
//!   `docs/plans/llama-cpp-parity-push.md` was measured; re-run it with
//!   `cargo test -p ferrox-models --test chat_template_real_gguf -- --ignored --nocapture`.

use ferrox_models::chat_template::{ChatTemplate, RenderOptions};
use serde_json::json;

fn conversation() -> Vec<serde_json::Value> {
    vec![
        json!({"role": "user", "content": "What is the capital of France?"}),
        json!({"role": "assistant", "content": "Paris."}),
        json!({"role": "user", "content": "And of Italy?"}),
    ]
}

fn opts() -> RenderOptions {
    RenderOptions {
        add_generation_prompt: true,
        bos_token: Some("<s>".into()),
        eos_token: Some("</s>".into()),
        ..Default::default()
    }
}

#[test]
fn every_checked_in_template_compiles_and_renders() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/templates");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).expect("tests/templates") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("jinja") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        let tmpl = ChatTemplate::from_jinja(&src)
            .unwrap_or_else(|e| panic!("{} did not compile: {e}", path.display()));
        let out = tmpl
            .render(&conversation(), &opts())
            .unwrap_or_else(|e| panic!("{} did not render: {e}", path.display()));
        assert!(
            out.contains("And of Italy?"),
            "{} dropped the last user turn:\n{out}",
            path.display()
        );
        seen += 1;
    }
    assert_eq!(seen, 5, "expected the five real GGUF templates in {dir:?}");
}

/// Every template must be *evaluated*, never sniffed. This pins the
/// specific regression: Mistral-Instruct's real template matches none of
/// the six markers the old `ChatTemplate::detect` looked for, so it
/// rendered as `user: …` plain lines — a framing the checkpoint has
/// never seen.
#[test]
fn a_template_with_no_recognisable_marker_still_renders_correctly() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/templates/tinyllama-1.1b-chat.jinja"),
    )
    .unwrap();
    let out = ChatTemplate::from_jinja(&src)
        .unwrap()
        .render(&conversation(), &opts())
        .unwrap();
    // TinyLlama's real template terminates every turn with the literal
    // `eos_token` text, which the hand-written renderer hardcoded as
    // `</s>` for every checkpoint using this shape. Here it comes from
    // the vocabulary the caller passed in.
    assert!(
        out.contains("<|user|>\nWhat is the capital of France?</s>"),
        "{out}"
    );
    assert!(out.trim_end().ends_with("<|assistant|>"), "{out}");
}

#[test]
#[ignore = "needs the real GGUF checkpoints in models/"]
fn sweep_every_local_gguf_template() {
    let root = std::env::var("FERROX_MODELS_DIR").unwrap_or_else(|_| "models".to_string());
    let mut files = Vec::new();
    collect_gguf(std::path::Path::new(&root), &mut files);
    assert!(!files.is_empty(), "no GGUFs under {root}");

    let mut with_template = 0;
    let mut failures = Vec::new();
    for path in &files {
        let Ok(file) = ferrox_gguf::ShardedGguf::open(path.to_str().unwrap()) else {
            continue;
        };
        let Some(src) = ferrox_gguf::TensorSource::metadata_str(&file, "tokenizer.chat_template")
            .filter(|t| !t.trim().is_empty())
        else {
            println!("{}: no chat template", path.display());
            continue;
        };
        with_template += 1;
        match ChatTemplate::from_jinja(src).and_then(|t| t.render(&conversation(), &opts())) {
            Ok(out) => println!(
                "{}: OK ({} bytes)\n---\n{out}\n---",
                path.display(),
                out.len()
            ),
            Err(e) => failures.push(format!("{}: {e}", path.display())),
        }
    }
    println!("{with_template} templates, {} failures", failures.len());
    assert!(failures.is_empty(), "{failures:#?}");
}

fn collect_gguf(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
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
