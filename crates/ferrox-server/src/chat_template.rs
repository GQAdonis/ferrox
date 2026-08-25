//! The served checkpoint's own chat template, evaluated.
//!
//! # What this replaced
//!
//! This module used to *sniff* `tokenizer.chat_template` for literal
//! markers -- `<|im_start|>`, `<|start_header_id|>`, `<start_of_turn>`
//! -- and pick one of six hand-written renderers. That was a disclosed
//! scope decision when there was no Jinja evaluator in the workspace.
//! There is one now (`ferrox_models::chat_template`, which compiles the
//! checkpoint's real template with minijinja), and it had no caller, so
//! the server was still serving Mistral-Instruct `user: hi` because
//! `[INST] … [/INST]` matches none of those markers. The sniffer's three
//! silent failure modes are written up in that module's doc comment; all
//! three are gone here by construction, because the template is now
//! evaluated rather than recognised.
//!
//! # What this adds on top of it
//!
//! Three things the evaluator deliberately leaves to its caller:
//!
//! 1. **Message shape.** [`PromptTemplate::render`] converts the
//!    server's `ChatMessage` into the OpenAI-shaped JSON a real template
//!    reads (`message.role`, `message.content`, `message.tool_calls`,
//!    `message.tool_call_id`, and a replayed chain of thought under all
//!    three spellings families use for it).
//! 2. **Who describes the tools.** A template that really consumes
//!    `tools` -- established by rendering with and without one, not by
//!    looking for the word -- gets them as structured JSON and owns the
//!    whole tool grammar. A template that does not gets the text
//!    preamble this server has always used (`tool_preamble` in
//!    `lib.rs`), and replayed calls are folded back into `content` as
//!    the same `<tool_call>{…}</tool_call>` marker text the preamble
//!    asks for -- so a conversation looks consistent to the model on
//!    every turn regardless of which half of that split it lands in.
//! 3. **The effort vocabulary.** Probed once at load
//!    ([`ferrox_edge::probe_effort_profile`]) rather than per request,
//!    because it costs ~30 renders and never changes for a checkpoint.
//!
//! Both probes run once, at load, which is what lets them be renders
//! rather than guesses.
//!
//! # End of turn
//!
//! [`PromptTemplate::end_of_turn`] survives the sniffer, and only that.
//! Gemma IT emits `<end_of_turn>` before `<eos>`, so a served Gemma
//! needs that string in its stop set no matter how the prompt was
//! rendered. It is a property of the *family*, not of the render, which
//! is why one literal marker test is all that is left of the six.

use std::sync::Arc;

use ferrox_edge::{
    derive_think_gears, probe_effort_profile, probe_thinking_profile, EffortProfile, ThinkGears,
};
use ferrox_models::chat_template::{
    BuiltinTemplate, ChatTemplate as JinjaTemplate, RenderOptions, TemplateError,
};
use serde_json::{json, Map, Value};

use crate::{ChatMessage, ToolDef};

/// Everything the server needs to turn a conversation into a prompt.
///
/// Cheap to clone: every `*Loaded` struct carries one and hands it to
/// each request.
#[derive(Clone)]
pub(crate) struct PromptTemplate {
    inner: Arc<Inner>,
}

struct Inner {
    template: JinjaTemplate,
    end_of_turn: Option<&'static str>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    thinking: ferrox_edge::ThinkingProfile,
    handles_tools: bool,
}

impl std::fmt::Debug for PromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PromptTemplate({})", self.inner.template.describe())
    }
}

impl PromptTemplate {
    /// The load-time entry point for a GGUF: compile whatever
    /// `tokenizer.chat_template` holds, or fall back to a builtin when
    /// the checkpoint ships none at all (llama.cpp `--jinja` defaults to
    /// ChatML the same way).
    pub(crate) fn from_gguf_metadata(
        source: Option<&str>,
        arch: Option<&str>,
        byte_tokenizer: bool,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        Self::build(
            JinjaTemplate::from_gguf_metadata(source, arch, byte_tokenizer),
            source,
            bos_token,
            eos_token,
        )
    }

    /// For checkpoints that carry their template outside GGUF metadata
    /// -- Kimi K3 keeps it as a top-level string in
    /// `tokenizer_config.json`, the real HuggingFace convention.
    pub(crate) fn from_source(
        source: Option<&str>,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        Self::build(
            JinjaTemplate::from_gguf_metadata(source, None, false),
            source,
            bos_token,
            eos_token,
        )
    }

    /// Role-labeled lines, no special tokens: the synthetic-weights demo
    /// path, where there is no real vocabulary for markers to live in.
    pub(crate) fn plain() -> Self {
        Self::build(
            JinjaTemplate::builtin(BuiltinTemplate::Plain),
            None,
            None,
            None,
        )
    }

    fn build(
        template: JinjaTemplate,
        source: Option<&str>,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        let (bos, eos) = (bos_token.as_deref(), eos_token.as_deref());
        let thinking = probe_thinking_profile(
            probe_render(&template, bos, eos),
            probe_efforts(&template, bos, eos),
        );
        let handles_tools = probe_tools_consumed(&template, bos, eos);
        Self {
            inner: Arc::new(Inner {
                end_of_turn: end_of_turn_marker(source),
                template,
                bos_token,
                eos_token,
                thinking,
                handles_tools,
            }),
        }
    }

    /// Short human-readable identity, for the load-time log line.
    pub(crate) fn describe(&self) -> String {
        self.inner.template.describe()
    }

    /// The end-of-turn string this family emits before EOS, if it has
    /// one worth adding to the stop set.
    pub(crate) fn end_of_turn(&self) -> Option<&'static str> {
        self.inner.end_of_turn
    }

    /// Whether the template consumes `tools` itself. When it does not,
    /// the caller owes the model a text preamble describing them.
    pub(crate) fn handles_tools(&self) -> bool {
        self.inner.handles_tools
    }

    /// The effort vocabulary this checkpoint's template actually grades,
    /// probed at load.
    pub(crate) fn efforts(&self) -> &EffortProfile {
        &self.inner.thinking.efforts
    }

    /// The thinking controls to advertise on `/v1/models`, so a client
    /// picks a gear instead of guessing one.
    ///
    /// `parser_configured` covers the one case the template cannot
    /// speak for: an always-thinking family whose template has no
    /// observable knob, but whose output really is being split into
    /// `reasoning_content`. Such a checkpoint advertises a single `on`
    /// gear with no kwargs -- there is nothing to send, and the point is
    /// only to stop a reasoning model looking like one with no gears.
    pub(crate) fn think_gears(&self, parser_configured: bool) -> ThinkGears {
        derive_think_gears(&self.inner.thinking, parser_configured)
    }

    /// Render a conversation into a prompt.
    ///
    /// `extra` is the request's `chat_template_kwargs` -- already
    /// sanitized by the caller, since every render path has to quantize
    /// effort identically or a request validates against one prompt and
    /// generates from another.
    pub(crate) fn render(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        extra: Map<String, Value>,
    ) -> Result<String, TemplateError> {
        let structured = self.handles_tools() && !tools.is_empty();
        let json: Vec<Value> = messages
            .iter()
            .map(|m| message_json(m, structured))
            .collect();
        let opts = RenderOptions {
            add_generation_prompt: true,
            bos_token: self.inner.bos_token.clone(),
            eos_token: self.inner.eos_token.clone(),
            tools: if structured {
                tools.iter().map(tool_json).collect()
            } else {
                Vec::new()
            },
            extra,
        };
        self.inner.template.render(&json, &opts)
    }
}

/// One OpenAI-shaped message, as a template reads it.
///
/// `structured` says whether the template is going to iterate
/// `message.tool_calls` itself. When it is not, a replayed assistant
/// turn's calls are folded back into `content` as the marker text the
/// text preamble asked the model to produce -- otherwise a multi-turn
/// tool conversation would show the model its own past calls in a shape
/// it was never asked for.
fn message_json(m: &ChatMessage, structured: bool) -> Value {
    let mut obj = Map::new();
    obj.insert("role".into(), json!(m.role));
    let text = m
        .content
        .as_ref()
        .map(crate::MessageContent::as_text)
        .unwrap_or_default();
    match (&m.tool_calls, structured) {
        (Some(calls), true) => {
            obj.insert("content".into(), json!(text));
            obj.insert(
                "tool_calls".into(),
                Value::Array(calls.iter().map(tool_call_json).collect()),
            );
        }
        (Some(_), false) => {
            obj.insert("content".into(), json!(m.rendered_content()));
        }
        (None, _) => {
            obj.insert("content".into(), json!(text));
        }
    }
    if let Some(id) = &m.tool_call_id {
        obj.insert("tool_call_id".into(), json!(id));
    }
    // Under every spelling, because templates disagree and a template
    // reads a key it does not know as undefined rather than erroring:
    // the DeepSeek/GLM family iterates `message.reasoning_content`, the
    // Qwen lineage `message.reasoning`, harmony `message.thinking`.
    // Emitting all three costs two keys and means a replayed chain of
    // thought reaches whichever one the served checkpoint was written
    // against, instead of being silently dropped by the others.
    //
    // Empty is not the same as absent and is skipped: a template that
    // tests `if message.reasoning_content` would otherwise open the
    // family's thinking markers around nothing.
    if let Some(reasoning) = m.reasoning_content.as_deref().filter(|r| !r.is_empty()) {
        obj.insert("reasoning_content".into(), json!(reasoning));
        obj.insert("reasoning".into(), json!(reasoning));
        // `thinking` has ONE exception, and it is a raise rather than a
        // wrong render: gpt-oss's harmony template rejects an assistant
        // turn that makes tool calls while carrying both visible text
        // and thinking, because harmony puts those on channels that
        // cannot both be final. Visible text wins -- it is what the
        // user was actually shown -- and the other two spellings still
        // carry the reasoning for every family that reads them.
        let harmony_would_raise = m.tool_calls.is_some() && !text.is_empty();
        if !harmony_would_raise {
            obj.insert("thinking".into(), json!(reasoning));
        }
    }
    Value::Object(obj)
}

/// A replayed tool call.
///
/// `arguments` arrives on the wire as a JSON *string*, and templates
/// disagree about what they expect: HuggingFace's own
/// `apply_chat_template` hands the model a dict, so a template that
/// writes `{{ call.function.arguments | tojson }}` would emit a
/// double-encoded string if the wire form were passed through. Parsing
/// it back to an object when it is one gives every template the shape it
/// was written against; a value that is not a JSON object stays a
/// string, because that is all that can honestly be said about it.
fn tool_call_json(call: &crate::ToolCallIn) -> Value {
    let raw = call.function.arguments.as_str();
    let args = match serde_json::from_str::<Value>(raw) {
        Ok(v @ Value::Object(_)) => v,
        _ => json!(raw),
    };
    json!({
        "type": "function",
        "id": call.id,
        "function": {"name": call.function.name, "arguments": args},
    })
}

pub(crate) fn tool_json(t: &ToolDef) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.function.name,
            "description": t.function.description.as_deref().unwrap_or(""),
            "parameters": t.function.parameters.clone().unwrap_or_else(|| json!({
                "type": "object",
                "properties": {},
            })),
        },
    })
}

/// Does this template actually *consume* the tools it is handed?
///
/// `ferrox_models`'s own `handles_tools` is a substring check on the
/// template source, and its doc comment says as much -- llama.cpp
/// probes by rendering and keeps the source check only because it is
/// cheap enough to run per call. This runs once, at load, so it can
/// afford the real answer: render the same conversation with and
/// without a tool and see whether the template did anything with it.
///
/// The difference is not cosmetic. A template that merely *mentions*
/// the word (in a comment, or in prose it prints) would claim to handle
/// tools, the text preamble would be skipped on that basis, and the
/// model would be offered tools nobody ever described to it -- a
/// request that silently cannot call anything. A template that errors
/// when handed tools has not handled them either.
fn probe_tools_consumed(
    template: &JinjaTemplate,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> bool {
    let messages = vec![json!({"role": "user", "content": "probe"})];
    let render = |tools: Vec<Value>| {
        template.render(
            &messages,
            &RenderOptions {
                add_generation_prompt: true,
                bos_token: bos_token.map(str::to_string),
                eos_token: eos_token.map(str::to_string),
                tools,
                extra: Map::new(),
            },
        )
    };
    let probe = json!({
        "type": "function",
        "function": {
            "name": "ferrox_probe_tool",
            "description": "No-op probe tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    });
    match (render(Vec::new()), render(vec![probe])) {
        // It did something with them, or it did not.
        (Ok(without), Ok(with)) => with != without,
        // Handed tools, it failed: it cannot express them.
        (_, Err(_)) => false,
        // It renders *only* when tools are present, which is as strong
        // a statement of dependence as a template can make.
        (Err(_), Ok(_)) => true,
    }
}

/// One render closure over a fixed probe conversation.
///
/// Both probes take the same shape -- vary one thing, render, compare
/// -- so they share the conversation as well as the closure. Fixed, so
/// the only thing that differs between two renders is the thing the
/// probe varied.
fn probe_render<'a>(
    template: &'a JinjaTemplate,
    bos_token: Option<&'a str>,
    eos_token: Option<&'a str>,
) -> impl FnMut(&Map<String, Value>, Option<&[Value]>) -> Result<String, TemplateError> + 'a {
    let messages = vec![json!({"role": "user", "content": "probe"})];
    move |kwargs, tools| {
        let opts = RenderOptions {
            add_generation_prompt: true,
            bos_token: bos_token.map(str::to_string),
            eos_token: eos_token.map(str::to_string),
            tools: tools.map(<[Value]>::to_vec).unwrap_or_default(),
            extra: kwargs.clone(),
        };
        template.render(&messages, &opts)
    }
}

/// Learn the effort vocabulary by rendering probes through the template.
///
/// A template that rejects the probe shape entirely is reported inert
/// by [`ferrox_edge::probe_effort_profile`], which is the safe answer:
/// send no effort at all.
fn probe_efforts(
    template: &JinjaTemplate,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> EffortProfile {
    probe_effort_profile(probe_render(template, bos_token, eos_token))
}

/// The one thing the marker sniffer was still right about: Gemma IT
/// ends a turn with a string that is not EOS, so a served Gemma needs
/// it in the stop set however the prompt was rendered.
fn end_of_turn_marker(source: Option<&str>) -> Option<&'static str> {
    let src = source?;
    if src.contains("<|turn>") || src.contains("<turn|>") {
        Some("<turn|>")
    } else if src.contains("<start_of_turn>") {
        Some("<end_of_turn>")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MessageContent;

    fn tool_def(name: &str) -> ToolDef {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": "a tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }))
        .expect("tool def")
    }

    fn tool_call_in(name: &str, arguments: &str) -> crate::ToolCallIn {
        serde_json::from_value(json!({
            "id": "call_0",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }))
        .expect("tool call")
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    /// The whole reason this module was rewritten: `[INST] … [/INST]`
    /// matches none of the old sniffer's markers, so a real Mistral
    /// checkpoint was served role-labeled lines it has never seen.
    #[test]
    fn a_template_the_old_sniffer_could_not_recognise_now_renders_correctly() {
        let mistral = "{% for m in messages %}{% if m.role == 'user' %}[INST] {{ m.content }} [/INST]{% else %}{{ m.content }}</s>{% endif %}{% endfor %}";
        let tmpl =
            PromptTemplate::from_gguf_metadata(Some(mistral), Some("llama"), false, None, None);
        let rendered = tmpl
            .render(&[msg("user", "hi")], &[], Map::new())
            .expect("renders");
        assert_eq!(rendered, "[INST] hi [/INST]");
    }

    #[test]
    fn a_checkpoint_with_no_template_falls_back_to_chatml_like_llama_cpp() {
        let tmpl = PromptTemplate::from_gguf_metadata(None, Some("olmoe"), false, None, None);
        let rendered = tmpl
            .render(&[msg("user", "hi")], &[], Map::new())
            .expect("renders");
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn a_byte_tokenizer_checkpoint_falls_back_to_role_labeled_lines() {
        let tmpl = PromptTemplate::from_gguf_metadata(None, Some("olmoe"), true, None, None);
        let rendered = tmpl
            .render(
                &[msg("user", "hello"), msg("assistant", "hi back")],
                &[],
                Map::new(),
            )
            .expect("renders");
        assert_eq!(rendered, "user: hello\nassistant: hi back");
    }

    #[test]
    fn chat_template_kwargs_reach_the_template() {
        let src = "{% if enable_thinking %}THINK{% endif %}{{ messages[0].content }}";
        let tmpl = PromptTemplate::from_gguf_metadata(Some(src), Some("qwen3"), false, None, None);
        let mut extra = Map::new();
        extra.insert("enable_thinking".into(), json!(true));
        assert_eq!(
            tmpl.render(&[msg("user", "hi")], &[], extra)
                .expect("renders"),
            "THINKhi"
        );
        assert_eq!(
            tmpl.render(&[msg("user", "hi")], &[], Map::new())
                .expect("renders"),
            "hi"
        );
    }

    #[test]
    fn a_template_that_reads_tools_is_given_them_structurally() {
        let src =
            "{% for t in tools %}TOOL:{{ t.function.name }}{% endfor %}{{ messages[0].content }}";
        let tmpl = PromptTemplate::from_gguf_metadata(Some(src), Some("qwen3"), false, None, None);
        assert!(tmpl.handles_tools());
        let tools = vec![tool_def("get_weather")];
        assert_eq!(
            tmpl.render(&[msg("user", "hi")], &tools, Map::new())
                .expect("renders"),
            "TOOL:get_weatherhi"
        );
    }

    /// A source check says this template handles tools; rendering it
    /// says otherwise. The preamble is what the model would lose.
    #[test]
    fn a_template_that_only_mentions_tools_does_not_count_as_handling_them() {
        let mentions = "{# tools are described by the system prompt #}\
             {% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
        let tmpl =
            PromptTemplate::from_gguf_metadata(Some(mentions), Some("qwen2"), false, None, None);
        assert!(
            !tmpl.handles_tools(),
            "the word alone must not skip the preamble"
        );
    }

    /// A template that never mentions `tools` cannot describe them, so
    /// the caller keeps the text preamble -- and a replayed call has to
    /// come back as the marker text that preamble asked for.
    #[test]
    fn a_template_that_ignores_tools_sees_replayed_calls_as_marker_text() {
        let tmpl =
            PromptTemplate::from_gguf_metadata(Some(CHATML), Some("qwen2"), false, None, None);
        assert!(!tmpl.handles_tools());
        let replayed = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![tool_call_in("get_weather", r#"{"city":"Paris"}"#)]),
            tool_call_id: None,
            reasoning_content: None,
        };
        let rendered = tmpl.render(&[replayed], &[], Map::new()).expect("renders");
        assert!(
            rendered.contains(
                r#"<tool_call>{"name": "get_weather", "arguments": {"city":"Paris"}}</tool_call>"#
            ),
            "{rendered}"
        );
    }

    /// The wire form of `arguments` is a string; a template written
    /// against HuggingFace's `apply_chat_template` expects the parsed
    /// object, and `| tojson` on the string form would double-encode it.
    #[test]
    fn replayed_call_arguments_reach_a_structural_template_as_an_object() {
        let src = "{% for m in messages %}{% for c in m.tool_calls %}{{ c.function.arguments.city }}{% endfor %}{% endfor %}tools:{{ tools | length }}";
        let tmpl = PromptTemplate::from_gguf_metadata(Some(src), Some("qwen3"), false, None, None);
        let replayed = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![tool_call_in("get_weather", r#"{"city":"Paris"}"#)]),
            tool_call_id: None,
            reasoning_content: None,
        };
        let tools = vec![tool_def("get_weather")];
        assert_eq!(
            tmpl.render(&[replayed], &tools, Map::new())
                .expect("renders"),
            "Paristools:1"
        );
    }

    #[test]
    fn a_broken_template_fails_the_request_instead_of_guessing() {
        let tmpl = PromptTemplate::from_gguf_metadata(
            Some("{% for m in messages %}"),
            None,
            false,
            None,
            None,
        );
        assert!(tmpl.render(&[msg("user", "hi")], &[], Map::new()).is_err());
    }

    #[test]
    fn gemma_families_contribute_their_end_of_turn_marker_to_the_stop_set() {
        assert_eq!(
            end_of_turn_marker(Some("{{ bos_token }}<start_of_turn>user\n")),
            Some("<end_of_turn>")
        );
        assert_eq!(
            end_of_turn_marker(Some("{{- '<|turn>' + m.role -}}")),
            Some("<turn|>")
        );
        assert_eq!(end_of_turn_marker(Some(CHATML)), None);
        assert_eq!(end_of_turn_marker(None), None);
    }

    /// The probe is what makes `reasoning_effort` mean anything: a
    /// template that grades only the OpenAI triple must not be sent
    /// `minimal`.
    #[test]
    fn the_effort_vocabulary_is_probed_at_load() {
        let graded = "{% set allowed = ['low','medium','high'] %}\
             {% if reasoning_effort %}\
               {% if reasoning_effort not in allowed %}{{ raise_exception('bad effort') }}{% endif %}\
               E:{{ reasoning_effort }}\
             {% endif %}{{ messages[0].content }}";
        let tmpl =
            PromptTemplate::from_gguf_metadata(Some(graded), Some("qwen3"), false, None, None);
        let profile = tmpl.efforts();
        assert!(profile.consumes_effort);
        assert!(profile.validates);
        assert_eq!(
            profile
                .supported
                .iter()
                .map(|e| e.as_str())
                .collect::<Vec<_>>(),
            vec!["low", "medium", "high"]
        );

        let inert =
            PromptTemplate::from_gguf_metadata(Some(CHATML), Some("qwen2"), false, None, None);
        assert!(!inert.efforts().consumes_effort);
    }

    /// Templates disagree about the key, so both are emitted. The
    /// DeepSeek/GLM lineage iterates `message.reasoning_content`, Qwen
    /// and gpt-oss `message.reasoning`; a template reads the one it
    /// does not know as undefined, so emitting both costs a key and
    /// loses nothing, while emitting one drops a replayed chain of
    /// thought on every checkpoint written against the other.
    #[test]
    fn a_replayed_chain_of_thought_reaches_a_template_under_both_spellings() {
        let mut m = msg("assistant", "the answer is 4");
        m.reasoning_content = Some("2 + 2".to_string());

        let json = message_json(&m, false);
        assert_eq!(json["reasoning_content"], "2 + 2");
        assert_eq!(json["reasoning"], "2 + 2");
        assert_eq!(json["thinking"], "2 + 2");
        assert_eq!(
            json["content"], "the answer is 4",
            "reasoning must never be folded into what the model said"
        );
    }

    /// The one case where a key is deliberately withheld. gpt-oss's
    /// harmony template RAISES on an assistant turn that makes tool
    /// calls while carrying both visible text and `thinking` -- the two
    /// land on channels that cannot both be final -- so a replayed
    /// agent turn would 500 the whole request rather than render
    /// oddly. Visible text wins, and the other two spellings still
    /// carry the reasoning for the families that read them.
    #[test]
    fn a_tool_call_turn_with_visible_text_withholds_only_the_harmony_spelling() {
        let mut m = msg("assistant", "checking the weather");
        m.reasoning_content = Some("I should call the tool".to_string());
        m.tool_calls = Some(vec![tool_call_in("get_weather", "{}")]);

        let json = message_json(&m, true);
        assert!(
            json.get("thinking").is_none(),
            "harmony would raise on this turn"
        );
        assert_eq!(json["reasoning_content"], "I should call the tool");
        assert_eq!(json["reasoning"], "I should call the tool");

        // A tool-call turn with NO visible text is the ordinary harmony
        // shape and keeps all three.
        m.content = None;
        assert_eq!(message_json(&m, true)["thinking"], "I should call the tool");
    }

    /// Empty is not absent. A template that opens the family's thinking
    /// markers on `if message.reasoning_content` would wrap them around
    /// nothing, so an empty string is skipped exactly like `None` --
    /// which also keeps a plain message's rendering byte-identical to
    /// what it was before the field existed.
    #[test]
    fn an_empty_or_absent_chain_of_thought_puts_no_key_in_front_of_a_template() {
        for empty in [None, Some(String::new())] {
            let mut m = msg("assistant", "hi");
            m.reasoning_content = empty;
            let json = message_json(&m, false);
            assert!(json.get("reasoning_content").is_none());
            assert!(json.get("reasoning").is_none());
            assert!(json.get("thinking").is_none());
        }
    }

    /// Both spellings are accepted on the way in, too. A client that
    /// received `reasoning` from `/v1/responses` or `/v1/messages` and
    /// replays the turn verbatim must not have it silently ignored
    /// because the chat surface spells the key the other way.
    #[test]
    fn a_replayed_turn_is_accepted_under_either_spelling_of_the_key() {
        for key in ["reasoning_content", "reasoning"] {
            let m: ChatMessage = serde_json::from_value(json!({
                "role": "assistant",
                "content": "4",
                key: "2 + 2",
            }))
            .expect("deserializes");
            assert_eq!(m.reasoning_content.as_deref(), Some("2 + 2"), "{key}");
        }
    }
}
