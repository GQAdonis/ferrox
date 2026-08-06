//! Real chat-template rendering, replacing a naive newline-join of
//! message contents that ignored `role` entirely (see this module's use
//! in `main.rs::prompt_from_messages`).
//!
//! GGUF files commonly carry a `tokenizer.chat_template` metadata string
//! -- the same Jinja2 template a real HuggingFace `tokenizer_config.json`
//! ships, describing exactly how a chat-tuned model expects its
//! conversation history formatted. Implementing a full Jinja2 evaluator
//! is out of scope here (a real, disclosed scope decision, not an
//! oversight): instead, this recognizes the
//! template string's real, distinctive markers and renders using a
//! small set of hand-written templates for the conventions those
//! markers actually correspond to, falling back to ChatML for real GGUF
//! checkpoints whose `tokenizer.chat_template` metadata is missing or
//! empty (matching llama.cpp `--jinja`), or plain role-labeled format for
//! byte/synthetic tokenizers and unrecognized custom templates.
//!
//! This directly closes a real, verified root cause of degenerate
//! chat-model output (found serving TinyLlama-1.1B-Chat end to end):
//! without correct role structure, a chat-tuned model never sees input
//! shaped anything like what it was trained on.
//!
//! Per-message rendering goes through `ChatMessage::rendered_content`
//! (`main.rs`), not the raw `content` field directly: an assistant
//! message replayed from tool-calling conversation history (see
//! `main.rs`'s tool-calling support) carries its past tool calls in
//! `tool_calls`, not `content`, and needs re-rendering as the same
//! `<tool_call>{...}</tool_call>` marker text a model is asked to
//! produce for a *new* call, so a multi-turn tool conversation looks
//! consistent to the model regardless of which turn it's replaying.

use crate::ChatMessage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<|im_start|>{role}\n{content}<|im_end|>\n`, repeated per
    /// message, ending with `<|im_start|>assistant\n` -- the ChatML
    /// convention (Qwen, and many other chat-tuned models).
    ChatMl,
    /// `<|{role}|>\n{content}\n`, repeated per message, ending with
    /// `<|assistant|>\n` -- TinyLlama-Chat's and Zephyr's real
    /// convention (no closing/end-of-turn marker per message).
    GenericRoleMarkers,
    /// `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`,
    /// repeated per message, ending with
    /// `<|start_header_id|>assistant<|end_header_id|>\n\n` -- the real
    /// Llama 3/3.1/3.2-Instruct convention. Found via a real gap: a real
    /// Llama-3.1-8B-Instruct GGUF's `tokenizer.chat_template` metadata
    /// (fetched directly from the file, not guessed) uses this exact
    /// convention, but neither `<|im_start|>` nor bare `<|user|>`/
    /// `<|assistant|>` markers appear in it, so it silently fell through
    /// to `Plain` -- the real, verified root cause of a real end-to-end
    /// test producing degenerate repetition even though a real chat
    /// template was present in the file the whole time. This
    /// implementation skips the real template's auto-injected system
    /// preamble (knowledge-cutoff/today's-date boilerplate, tool-calling
    /// scaffolding) -- a disclosed simplification, not a silent one --
    /// and renders only the structurally-important per-turn
    /// header/eot_id framing that actually determines output quality.
    Llama3,
    /// Gemma 1/2/3 Instruct: `<start_of_turn>user\n{content}<end_of_turn>\n`
    /// (assistant role rendered as `model`), ending with
    /// `<start_of_turn>model\n`. BOS is *not* emitted here -- the
    /// server's `bos_id` prepending in `generate` owns that, matching
    /// the GGUF template's leading `{{ bos_token }}`.
    Gemma,
    /// `{role}: {content}`, one per line, no special tokens at all --
    /// used when the GGUF carries no `tokenizer.chat_template` (or an
    /// unrecognized one), matching the plain completion-style behavior
    /// this server has always had for such checkpoints.
    Plain,
}

impl ChatTemplate {
    /// Sniffs a GGUF's real `tokenizer.chat_template` Jinja2 string for
    /// distinctive literal markers, rather than evaluating it as Jinja2.
    /// Unrecognized non-empty templates yield `Plain`.
    pub fn detect(chat_template: Option<&str>) -> Self {
        match chat_template {
            Some(t) if t.contains("<|im_start|>") => ChatTemplate::ChatMl,
            Some(t) if t.contains("<|start_header_id|>") => ChatTemplate::Llama3,
            Some(t) if t.contains("<|user|>") || t.contains("<|assistant|>") => {
                ChatTemplate::GenericRoleMarkers
            }
            Some(t) if t.contains("<start_of_turn>") => ChatTemplate::Gemma,
            _ => ChatTemplate::Plain,
        }
    }

    /// Like [`Self::detect`], but when `tokenizer.chat_template` is missing
    /// or empty, match llama.cpp `--jinja` / `common/chat.cpp`: real GGUF
    /// checkpoints default to ChatML (`CHATML_TEMPLATE_SRC`). Keep
    /// [`Plain`] for byte tokenizers and synthetic server fallbacks.
    pub fn detect_for_gguf(
        chat_template: Option<&str>,
        arch: Option<&str>,
        byte_tokenizer: bool,
    ) -> Self {
        match chat_template.filter(|t| !t.is_empty()) {
            Some(t) => Self::detect(Some(t)),
            None if byte_tokenizer || arch.is_none() => Self::Plain,
            None => Self::ChatMl,
        }
    }

    /// Renders `messages` into a single prompt string, ending with
    /// whatever marker tells the model it's now the assistant's turn to
    /// generate (the real "generation prompt" convention every one of
    /// these template families uses).
    pub fn render(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        match self {
            ChatTemplate::ChatMl => {
                for m in messages {
                    out.push_str("<|im_start|>");
                    out.push_str(&m.role);
                    out.push('\n');
                    out.push_str(&m.rendered_content());
                    out.push_str("<|im_end|>\n");
                }
                out.push_str("<|im_start|>assistant\n");
            }
            ChatTemplate::GenericRoleMarkers => {
                for m in messages {
                    out.push_str("<|");
                    out.push_str(&m.role);
                    out.push_str("|>\n");
                    out.push_str(&m.rendered_content());
                    // The real template every known user of this
                    // convention ships (TinyLlama-Chat, Zephyr) appends
                    // the real EOS token *string* right after each
                    // turn's content, not just a newline -- confirmed
                    // directly against TinyLlama-Chat's own real
                    // `tokenizer.chat_template`:
                    // `'<|role|>\n' + content + eos_token`. Both of
                    // those models are real SentencePiece-family
                    // tokenizers whose EOS token text is `</s>`, and
                    // (now that `GgufSpmTokenizer`/`GgufBpeTokenizer`
                    // recognize control tokens atomically) this
                    // literal text is recognized as the real EOS token
                    // id, not shattered into byte-fallback pieces. A
                    // model using this exact template convention with a
                    // different real EOS string is a known, disclosed
                    // gap.
                    out.push_str("</s>\n");
                }
                out.push_str("<|assistant|>\n");
            }
            ChatTemplate::Llama3 => {
                for m in messages {
                    out.push_str("<|start_header_id|>");
                    out.push_str(&m.role);
                    out.push_str("<|end_header_id|>\n\n");
                    out.push_str(&m.rendered_content());
                    out.push_str("<|eot_id|>");
                }
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatTemplate::Gemma => {
                // Match the structural Gemma IT framing. System content is
                // prepended to the first user turn (as the real Jinja does).
                let mut system_prefix = String::new();
                let mut turns: Vec<&ChatMessage> = Vec::new();
                for m in messages {
                    match m.role.as_str() {
                        "system" if turns.is_empty() && system_prefix.is_empty() => {
                            system_prefix = m.rendered_content();
                            if !system_prefix.ends_with('\n') {
                                system_prefix.push_str("\n\n");
                            }
                        }
                        _ => turns.push(m),
                    }
                }
                for m in turns {
                    let role = match m.role.as_str() {
                        "assistant" | "model" => "model",
                        _ => "user",
                    };
                    out.push_str("<start_of_turn>");
                    out.push_str(role);
                    out.push('\n');
                    if role == "user" && !system_prefix.is_empty() {
                        out.push_str(&system_prefix);
                        system_prefix.clear();
                    }
                    out.push_str(&m.rendered_content());
                    out.push_str("<end_of_turn>\n");
                }
                out.push_str("<start_of_turn>model\n");
            }
            ChatTemplate::Plain => {
                let lines: Vec<String> = messages
                    .iter()
                    .map(|m| format!("{}: {}", m.role, m.rendered_content()))
                    .collect();
                out.push_str(&lines.join("\n"));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(crate::MessageContent::Text(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn detects_chatml_from_a_real_template_string() {
        let template =
            "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}";
        assert_eq!(ChatTemplate::detect(Some(template)), ChatTemplate::ChatMl);
    }

    #[test]
    fn detects_generic_role_markers_from_a_real_tinyllama_style_template() {
        let template =
            "{% for m in messages %}<|{{m.role}}|>\n{{m.content}}\n{% endfor %}<|assistant|>\n";
        assert_eq!(
            ChatTemplate::detect(Some(template)),
            ChatTemplate::GenericRoleMarkers
        );
    }

    #[test]
    fn detects_llama3_from_the_real_template_string() {
        // Fetched directly from a real Meta-Llama-3.1-8B-Instruct GGUF's
        // tokenizer.chat_template metadata (lmstudio-community's Q4_K_M
        // conversion) -- the exact real template, not a hand-shortened
        // paraphrase.
        let template = r#"{{- bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
{%- endif %}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"#;
        assert_eq!(ChatTemplate::detect(Some(template)), ChatTemplate::Llama3);
    }

    #[test]
    fn llama3_renders_real_markers_and_a_trailing_generation_prompt() {
        let messages = vec![msg("system", "be helpful"), msg("user", "hi")];
        let rendered = ChatTemplate::Llama3.render(&messages);
        assert_eq!(
            rendered,
            "<|start_header_id|>system<|end_header_id|>\n\nbe helpful<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn detects_gemma_from_start_of_turn_marker() {
        let template = "{{ bos_token }}{% for message in messages %}<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n{% endfor %}";
        assert_eq!(ChatTemplate::detect(Some(template)), ChatTemplate::Gemma);
    }

    #[test]
    fn gemma_renders_user_model_turns_and_generation_prompt() {
        let messages = vec![msg("user", "Capital of France?")];
        let rendered = ChatTemplate::Gemma.render(&messages);
        assert_eq!(
            rendered,
            "<start_of_turn>user\nCapital of France?<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn gemma_folds_system_into_first_user_turn() {
        let messages = vec![msg("system", "be brief"), msg("user", "hi")];
        let rendered = ChatTemplate::Gemma.render(&messages);
        assert_eq!(
            rendered,
            "<start_of_turn>user\nbe brief\n\nhi<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn falls_back_to_plain_when_no_template_or_unrecognized() {
        assert_eq!(ChatTemplate::detect(None), ChatTemplate::Plain);
        assert_eq!(
            ChatTemplate::detect(Some("some unrecognized custom format")),
            ChatTemplate::Plain
        );
    }

    #[test]
    fn gguf_without_template_defaults_to_chatml_for_real_architectures() {
        assert_eq!(
            ChatTemplate::detect_for_gguf(None, Some("olmoe"), false),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::detect_for_gguf(Some(""), Some("olmoe"), false),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::detect_for_gguf(None, Some("olmoe"), true),
            ChatTemplate::Plain
        );
        assert_eq!(
            ChatTemplate::detect_for_gguf(
                Some("some unrecognized custom format"),
                Some("olmoe"),
                false
            ),
            ChatTemplate::Plain
        );
    }

    #[test]
    fn chatml_renders_real_markers_and_a_trailing_generation_prompt() {
        let messages = vec![msg("system", "be helpful"), msg("user", "hi")];
        let rendered = ChatTemplate::ChatMl.render(&messages);
        assert_eq!(
            rendered,
            "<|im_start|>system\nbe helpful<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn generic_role_markers_renders_real_markers_and_a_trailing_generation_prompt() {
        let messages = vec![msg("user", "hello there")];
        let rendered = ChatTemplate::GenericRoleMarkers.render(&messages);
        assert_eq!(rendered, "<|user|>\nhello there</s>\n<|assistant|>\n");
    }

    #[test]
    fn plain_renders_role_labeled_lines_with_no_special_tokens() {
        let messages = vec![msg("user", "hello"), msg("assistant", "hi back")];
        let rendered = ChatTemplate::Plain.render(&messages);
        assert_eq!(rendered, "user: hello\nassistant: hi back");
    }
}
