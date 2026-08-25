//! Recognizing a tool call in whichever wire format a checkpoint was
//! trained on.
//!
//! There is no standard. Every model family invented its own way of
//! saying "call this function with these arguments", and a served model
//! only ever emits its own:
//!
//! | family | shape |
//! |---|---|
//! | Hermes / Qwen2.5 | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` |
//! | Llama 3 | `<\|python_tag\|>{"name": …, "parameters": {…}}` |
//! | Mistral | `[TOOL_CALLS] [{…}, {…}]` |
//! | Qwen3-Coder | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` |
//! | GLM-4.7 | `<tool_call>name\n<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` |
//! | DeepSeek | `<｜DSML｜invoke name="…"><｜DSML｜parameter name="…">…` |
//! | MiniMax | `<minimax:tool_call><invoke name="…"><parameter name="…">…` |
//! | gpt-oss | `<\|channel\|>commentary to=functions.name<\|message\|>{…}<\|call\|>` |
//! | Gemma 4 | `<\|tool_call>call:name{k: v}<tool_call\|>` |
//!
//! # Why the XML-ish families need a schema
//!
//! A JSON family states its own types: `{"count": 3}` is a number
//! because it is written as one. An XML-ish family does not --
//! `<parameter name="count">3</parameter>` is a *string* on the wire,
//! and whether it should reach the tool as `3` or `"3"` is a fact about
//! the tool's schema, not about the text. So [`ToolSchema`] is passed
//! in, and a declared `integer` parameter is parsed while a declared
//! `string` one is handed over verbatim -- which is what keeps a
//! zero-padded id like `"018956"` from arriving as `18956`.
//!
//! # What streams and what does not
//!
//! The invoke/parameter families stream **prefix-stable** JSON
//! fragments: each fragment is a literal continuation of the arguments
//! JSON, so a client can concatenate them and parse the result. That
//! matters for coding agents, whose arguments are whole files.
//!
//! The JSON-payload families (Hermes, Llama 3, Mistral, gpt-oss,
//! Gemma) are emitted whole, when their block completes. Their payload
//! is not meaningfully parseable until it is complete, and emitting a
//! half-written JSON object as a "fragment" would only move the
//! problem to the client.
//!
//! Ported 1:1 from FreeToken's `server/function_call_parser.py`
//! (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::detokenize::{floor_char_boundary, stop_prefix_holdback};

/// One recognized call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Position in this response's list of calls.
    pub index: usize,
    pub name: String,
    /// The arguments as a JSON object string, ready to hand to a
    /// client. Always valid JSON, `{}` when the call took none.
    pub arguments: String,
}

/// What a streaming parse produced, in wire order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallEvent {
    /// Ordinary output, safe to send.
    Text(String),
    /// A call has begun. Its arguments follow.
    CallStart { index: usize, name: String },
    /// A literal continuation of the call's arguments JSON.
    CallArguments { index: usize, fragment: String },
    /// The call is complete, with its whole arguments object.
    CallEnd { index: usize, arguments: String },
}

/// One tool the request offered, used to type XML-ish parameter values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolSchema {
    pub name: String,
    /// The `parameters` JSON Schema, if the request supplied one.
    pub parameters: Option<Value>,
}

impl ToolSchema {
    pub fn new(name: impl Into<String>) -> Self {
        ToolSchema {
            name: name.into(),
            parameters: None,
        }
    }

    pub fn with_parameters(name: impl Into<String>, parameters: Value) -> Self {
        ToolSchema {
            name: name.into(),
            parameters: Some(parameters),
        }
    }

    /// The declared JSON Schema type of one parameter, if any.
    fn parameter_type(&self, key: &str) -> Option<String> {
        let properties = self.parameters.as_ref()?.get("properties")?;
        let declared = properties.get(key)?;
        declared
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string)
    }
}

/// Which family's tool-call format to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// Hermes-style `<tool_call>` wrapping a JSON object.
    Qwen25,
    /// `<|python_tag|>` followed by one or more JSON objects.
    Llama3,
    /// `[TOOL_CALLS] [ … ]`.
    Mistral,
    /// `<function=`/`<parameter=` inside `<tool_call>`.
    Qwen3Coder,
    /// `<arg_key>`/`<arg_value>` inside `<tool_call>`.
    Glm47,
    /// DeepSeek's DSML invoke/parameter grammar.
    DeepSeekV32,
    /// MiniMax's namespaced invoke/parameter grammar.
    MiniMax,
    /// The gpt-oss harmony commentary channel.
    GptOss,
    /// Gemma 4's `call:name{…}` form.
    Gemma4,
}

/// Every marker that means "a tool call may be starting", across all
/// families. Used to decide, cheaply, whether a response is worth
/// parsing at all.
pub const TOOL_MARKERS: [&str; 10] = [
    "<tool_call>",
    "<function=",
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<minimax:tool_call>",
    "<｜DSML｜function_calls>",
    "<｜DSML｜invoke",
    "<|channel|>",
    "<|tool_call>",
    "to=functions.",
];

impl ToolCallFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            ToolCallFormat::Qwen25 => "qwen25",
            ToolCallFormat::Llama3 => "llama3",
            ToolCallFormat::Mistral => "mistral",
            ToolCallFormat::Qwen3Coder => "qwen3_coder",
            ToolCallFormat::Glm47 => "glm47",
            ToolCallFormat::DeepSeekV32 => "deepseekv32",
            ToolCallFormat::MiniMax => "minimax",
            ToolCallFormat::GptOss => "gpt_oss",
            ToolCallFormat::Gemma4 => "gemma4",
        }
    }

    pub fn parse(name: &str) -> Option<ToolCallFormat> {
        match name {
            "qwen" | "qwen25" => Some(ToolCallFormat::Qwen25),
            "llama3" => Some(ToolCallFormat::Llama3),
            "mistral" => Some(ToolCallFormat::Mistral),
            "qwen3_coder" => Some(ToolCallFormat::Qwen3Coder),
            "glm47" => Some(ToolCallFormat::Glm47),
            "deepseekv32" => Some(ToolCallFormat::DeepSeekV32),
            "minimax" => Some(ToolCallFormat::MiniMax),
            "gpt_oss" | "gpt-oss" => Some(ToolCallFormat::GptOss),
            "gemma4" => Some(ToolCallFormat::Gemma4),
            _ => None,
        }
    }

    /// Which format a checkpoint's identity implies.
    ///
    /// The order of these arms is load-bearing: the specific families
    /// have to be tested before the general ones they look like, or
    /// Qwen3-Coder resolves to plain Qwen and its whole grammar is
    /// missed. Llama 3 is the fallback because its `<|python_tag|>`
    /// form is also what an untrained model most often improvises.
    pub fn infer(marker: &str) -> ToolCallFormat {
        let marker = marker.to_ascii_lowercase();
        let has = |needle: &str| marker.contains(needle);
        if has("gpt_oss") || has("gpt-oss") || has("gptoss") {
            ToolCallFormat::GptOss
        } else if has("minimax") {
            ToolCallFormat::MiniMax
        } else if has("gemma4") {
            ToolCallFormat::Gemma4
        } else if has("qwen3_5") || has("qwen3.5") || (has("qwen3") && has("coder")) {
            ToolCallFormat::Qwen3Coder
        } else if has("qwen") {
            ToolCallFormat::Qwen25
        } else if has("deepseek") && (has("v4") || has("v3.2") || has("v32")) {
            ToolCallFormat::DeepSeekV32
        } else if has("glm") {
            ToolCallFormat::Glm47
        } else if has("mistral") {
            ToolCallFormat::Mistral
        } else {
            ToolCallFormat::Llama3
        }
    }

    /// The marker that opens a call in this format -- what a scheduler
    /// watches for when it wants to checkpoint state at the start of a
    /// tool call.
    pub fn opener(self) -> Option<&'static str> {
        match self {
            ToolCallFormat::Qwen25 | ToolCallFormat::Qwen3Coder | ToolCallFormat::Glm47 => {
                Some("<tool_call>")
            }
            ToolCallFormat::Llama3 => Some("<|python_tag|>"),
            ToolCallFormat::Mistral => Some("[TOOL_CALLS]"),
            ToolCallFormat::DeepSeekV32 => Some("<｜DSML｜function_calls>"),
            ToolCallFormat::MiniMax => Some("<minimax:tool_call>"),
            ToolCallFormat::Gemma4 => Some("<|tool_call>"),
            // A harmony call opens with a channel header, which also
            // opens ordinary messages -- there is no marker that means
            // "tool call" on its own.
            ToolCallFormat::GptOss => None,
        }
    }

    /// Whether this format's arguments stream as prefix-stable
    /// fragments, or arrive whole when the call completes.
    pub fn streams_arguments(self) -> bool {
        matches!(
            self,
            ToolCallFormat::Qwen3Coder
                | ToolCallFormat::DeepSeekV32
                | ToolCallFormat::MiniMax
                | ToolCallFormat::Glm47
        )
    }

    fn markers(self) -> Markers {
        match self {
            ToolCallFormat::Qwen25 => Markers::block("<tool_call>", "</tool_call>"),
            ToolCallFormat::Llama3 => Markers::block("<|python_tag|>", ""),
            ToolCallFormat::Mistral => Markers::block("[TOOL_CALLS]", ""),
            ToolCallFormat::Gemma4 => Markers::block("<|tool_call>", "<tool_call|>"),
            ToolCallFormat::GptOss => Markers::block("<|channel|>", "<|call|>"),
            ToolCallFormat::Qwen3Coder => Markers {
                open: "<tool_call>",
                close: "</tool_call>",
                invoke: Some(TagGrammar {
                    open: "<function=",
                    name: NameStyle::Bare,
                    close: "</function>",
                }),
                param: Some(TagGrammar {
                    open: "<parameter=",
                    name: NameStyle::Bare,
                    close: "</parameter>",
                }),
                trim_newlines: TrimStyle::One,
            },
            ToolCallFormat::MiniMax => Markers {
                open: "<minimax:tool_call>",
                close: "</minimax:tool_call>",
                invoke: Some(TagGrammar {
                    open: "<invoke",
                    name: NameStyle::Attribute,
                    close: "</invoke>",
                }),
                param: Some(TagGrammar {
                    open: "<parameter",
                    name: NameStyle::Attribute,
                    close: "</parameter>",
                }),
                trim_newlines: TrimStyle::All,
            },
            ToolCallFormat::DeepSeekV32 => Markers {
                open: "<｜DSML｜function_calls>",
                close: "</｜DSML｜function_calls>",
                invoke: Some(TagGrammar {
                    open: "<｜DSML｜invoke",
                    name: NameStyle::Attribute,
                    close: "</｜DSML｜invoke>",
                }),
                param: Some(TagGrammar {
                    open: "<｜DSML｜parameter",
                    name: NameStyle::Attribute,
                    close: "</｜DSML｜parameter>",
                }),
                trim_newlines: TrimStyle::None,
            },
            ToolCallFormat::Glm47 => Markers {
                open: "<tool_call>",
                close: "</tool_call>",
                invoke: None,
                param: Some(TagGrammar {
                    open: "<arg_key>",
                    name: NameStyle::Bare,
                    close: "</arg_value>",
                }),
                trim_newlines: TrimStyle::All,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum NameStyle {
    /// The name is everything between the opener and `>`.
    Bare,
    /// The name is the `name="…"` attribute.
    Attribute,
}

#[derive(Debug, Clone, Copy)]
struct TagGrammar {
    open: &'static str,
    name: NameStyle,
    close: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrimStyle {
    /// Values are handed over exactly as written.
    None,
    /// Strip at most one leading and one trailing newline -- the ones
    /// the template's own layout inserts.
    One,
    /// Strip all surrounding whitespace.
    All,
}

#[derive(Debug, Clone, Copy)]
struct Markers {
    open: &'static str,
    close: &'static str,
    invoke: Option<TagGrammar>,
    param: Option<TagGrammar>,
    trim_newlines: TrimStyle,
}

impl Markers {
    const fn block(open: &'static str, close: &'static str) -> Self {
        Markers {
            open,
            close,
            invoke: None,
            param: None,
            trim_newlines: TrimStyle::None,
        }
    }
}

/// Parses one response's tool calls.
#[derive(Debug)]
pub struct ToolCallParser {
    format: ToolCallFormat,
    markers: Markers,
    tools: BTreeMap<String, ToolSchema>,
    buffer: String,
    /// Calls emitted so far, which is also the next call's index.
    emitted: usize,
    /// Set while a streamed call is open.
    open: Option<OpenCall>,
}

#[derive(Debug)]
struct OpenCall {
    index: usize,
    name: String,
    /// The arguments JSON built so far, without its closing brace.
    ledger: String,
    /// Whether any parameter has been written into the ledger yet.
    started: bool,
}

impl ToolCallParser {
    pub fn new(format: ToolCallFormat, tools: Vec<ToolSchema>) -> Self {
        ToolCallParser {
            format,
            markers: format.markers(),
            tools: tools
                .into_iter()
                .map(|tool| (tool.name.clone(), tool))
                .collect(),
            buffer: String::new(),
            emitted: 0,
            open: None,
        }
    }

    pub fn format(&self) -> ToolCallFormat {
        self.format
    }

    /// Whether `text` could contain a call in *any* known format.
    ///
    /// Deliberately format-agnostic and cheap: a response with none of
    /// these markers needs no parsing at all, and that is the common
    /// case.
    pub fn text_may_contain_call(text: &str) -> bool {
        TOOL_MARKERS.iter().any(|marker| text.contains(marker))
    }

    /// Whether `text` contains a call in *this* format.
    pub fn has_tool_call(&self, text: &str) -> bool {
        match self.format {
            ToolCallFormat::GptOss => {
                text.contains("to=functions.") && text.contains("<|message|>")
            }
            ToolCallFormat::Llama3 => {
                text.contains("<|python_tag|>") || text.trim_start().starts_with('{')
            }
            ToolCallFormat::Qwen3Coder => {
                text.contains("<function=") || text.contains("<tool_call>")
            }
            _ => text.contains(self.markers.open),
        }
    }

    /// Parse a complete response into its text and its calls.
    pub fn parse_complete(&self, text: &str) -> (String, Vec<ToolCall>) {
        if self.tools.is_empty() || !self.has_tool_call(text) {
            return (text.to_string(), Vec::new());
        }
        match self.format {
            ToolCallFormat::Qwen25 => self.parse_json_blocks(text, "<tool_call>", "</tool_call>"),
            ToolCallFormat::Llama3 => self.parse_bare_json(text, "<|python_tag|>"),
            ToolCallFormat::Mistral => self.parse_mistral(text),
            ToolCallFormat::GptOss => self.parse_harmony(text),
            ToolCallFormat::Gemma4 => self.parse_gemma(text),
            ToolCallFormat::Glm47 => self.parse_glm(text),
            ToolCallFormat::Qwen3Coder | ToolCallFormat::MiniMax | ToolCallFormat::DeepSeekV32 => {
                self.parse_invoke(text)
            }
        }
    }

    /// Feed one increment and get the events it completed, in wire
    /// order.
    pub fn push(&mut self, chunk: &str) -> Vec<ToolCallEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        if self.tools.is_empty() {
            let text = std::mem::take(&mut self.buffer);
            if !text.is_empty() {
                events.push(ToolCallEvent::Text(text));
            }
            return events;
        }
        loop {
            let progressed = if self.open.is_some() {
                self.step_open_call(&mut events)
            } else {
                self.step_idle(&mut events)
            };
            if !progressed {
                break;
            }
        }
        events
    }

    /// Release whatever is still buffered when the stream ends.
    ///
    /// A call left half-written by a truncated generation is closed
    /// with what it has, so the fragments a client already received
    /// still concatenate to valid JSON. Silently dropping it would
    /// leave the client parsing an unterminated object forever.
    pub fn finish(&mut self) -> Vec<ToolCallEvent> {
        let mut events = Vec::new();
        if let Some(open) = self.open.take() {
            let arguments = close_ledger(&open);
            events.push(ToolCallEvent::CallArguments {
                index: open.index,
                fragment: if open.started {
                    "}".into()
                } else {
                    "{}".into()
                },
            });
            events.push(ToolCallEvent::CallEnd {
                index: open.index,
                arguments,
            });
            self.emitted = open.index + 1;
        }
        let rest = std::mem::take(&mut self.buffer);
        if rest.is_empty() {
            return events;
        }
        if !self.has_tool_call(&rest) {
            self.emit_text(rest, &mut events);
            return events;
        }
        // Llama 3 and Mistral have no closing marker, so their payload
        // is only parseable once the stream ends -- this is the only
        // point at which their calls can be emitted at all. Dropping
        // the buffer here because it "looks like a call" would lose
        // every call those two families make.
        let (text, calls) = self.parse_complete(&rest);
        if !text.trim().is_empty() {
            events.push(ToolCallEvent::Text(text));
        }
        for call in calls {
            let index = self.emitted;
            self.emitted += 1;
            events.push(ToolCallEvent::CallStart {
                index,
                name: call.name,
            });
            events.push(ToolCallEvent::CallArguments {
                index,
                fragment: call.arguments.clone(),
            });
            events.push(ToolCallEvent::CallEnd {
                index,
                arguments: call.arguments,
            });
        }
        events
    }

    /// Looking for the start of a call. Returns whether it made
    /// progress.
    fn step_idle(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        // A model does not always wrap its call in the block marker its
        // template shows -- Qwen3-Coder in particular often emits a
        // bare `<function=…>`. The invoke tag opens a call just as
        // definitively, so whichever comes first counts.
        let opener = match self.markers.invoke {
            Some(invoke) => [self.markers.open, invoke.open]
                .iter()
                .filter_map(|marker| self.buffer.find(marker))
                .min(),
            None => self.buffer.find(self.markers.open),
        };
        match opener {
            Some(index) => {
                if index > 0 {
                    let text: String = self.buffer.drain(..index).collect();
                    self.emit_text(text, events);
                }
                if self.format.streams_arguments() {
                    self.start_streamed_block(events)
                } else {
                    self.take_complete_block(events)
                }
            }
            None => {
                // Withhold any trailing run that could still grow into
                // a marker -- opening OR closing. The closing ones
                // matter because a streamed invoke grammar leaves its
                // block close behind after the call is consumed, and at
                // one character per chunk it would otherwise dribble
                // out as `<`, `/`, `t`, ... before anything could
                // recognize it as structure.
                let markers: Vec<String> = TOOL_MARKERS
                    .iter()
                    .map(|marker| marker.to_string())
                    .chain(self.closing_markers().into_iter().map(str::to_string))
                    .collect();
                let hold = stop_prefix_holdback(&self.buffer, &markers);
                let split = floor_char_boundary(&self.buffer, self.buffer.len() - hold);
                if split == 0 {
                    return false;
                }
                let text: String = self.buffer.drain(..split).collect();
                self.emit_text(text, events);
                false
            }
        }
    }

    /// Release text, minus any structure that outlived its call.
    ///
    /// A streamed call is consumed marker by marker as it is
    /// recognized, and the *block* close can be left over -- an invoke
    /// grammar closes `</function>` and then `</tool_call>`, and only
    /// the first belongs to the call. Emitting the remainder as content
    /// would print raw markup into an answer, which is exactly what a
    /// client renders verbatim.
    fn emit_text(&self, text: String, events: &mut Vec<ToolCallEvent>) {
        let mut text = text;
        for marker in self.closing_markers() {
            if text.contains(marker) {
                text = text.replace(marker, "");
            }
        }
        if !text.is_empty() {
            events.push(ToolCallEvent::Text(text));
        }
    }

    /// This format's closing markers: structure that can outlive the
    /// call it closed.
    fn closing_markers(&self) -> Vec<&'static str> {
        [
            Some(self.markers.close),
            self.markers.invoke.map(|tag| tag.close),
            self.markers.param.map(|tag| tag.close),
        ]
        .into_iter()
        .flatten()
        .filter(|marker| !marker.is_empty())
        .collect()
    }

    /// A non-streaming format: wait for the whole block, then emit the
    /// calls it holds.
    fn take_complete_block(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let close = self.markers.close;
        // Formats with no closing marker (Llama 3, Mistral) only become
        // parseable at the end of the stream.
        if close.is_empty() {
            return false;
        }
        let Some(end) = self.buffer.find(close) else {
            return false;
        };
        let block: String = self.buffer.drain(..end + close.len()).collect();
        let (_, calls) = self.parse_complete(&block);
        for call in calls {
            let index = self.emitted;
            self.emitted += 1;
            events.push(ToolCallEvent::CallStart {
                index,
                name: call.name,
            });
            events.push(ToolCallEvent::CallArguments {
                index,
                fragment: call.arguments.clone(),
            });
            events.push(ToolCallEvent::CallEnd {
                index,
                arguments: call.arguments,
            });
        }
        true
    }

    /// A streaming format: open a call as soon as its name is known.
    fn start_streamed_block(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let Some(invoke) = self.markers.invoke else {
            return self.start_glm_call(events);
        };
        let Some(start) = self.buffer.find(invoke.open) else {
            // The block opened but the first invoke has not arrived. If
            // the block closed with nothing in it, drop it.
            if let Some(end) = self.buffer.find(self.markers.close) {
                self.buffer.drain(..end + self.markers.close.len());
                return true;
            }
            return false;
        };
        let attrs_start = start + invoke.open.len();
        let Some(gt) = self.buffer[attrs_start..]
            .find('>')
            .map(|i| i + attrs_start)
        else {
            return false;
        };
        let attrs = &self.buffer[attrs_start..gt];
        let Some(name) = read_name(attrs, invoke.name) else {
            // A malformed opener: drop it rather than stalling.
            self.buffer.drain(..gt + 1);
            return true;
        };
        let index = self.emitted;
        self.open = Some(OpenCall {
            index,
            name: name.clone(),
            ledger: String::new(),
            started: false,
        });
        self.buffer.drain(..gt + 1);
        events.push(ToolCallEvent::CallStart { index, name });
        true
    }

    /// GLM's grammar names the function in bare text after the block
    /// opener rather than in a tag.
    fn start_glm_call(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let open = self.markers.open;
        let Some(start) = self.buffer.find(open) else {
            return false;
        };
        let after = start + open.len();
        let rest = &self.buffer[after..];
        // The name ends at the first newline, the first argument, or
        // the end of the block -- whichever comes first.
        let end = ["\n", "<arg_key>", "</tool_call>"]
            .iter()
            .filter_map(|marker| rest.find(marker))
            .min();
        let Some(end) = end else {
            return false;
        };
        let name = rest[..end].trim().to_string();
        if name.is_empty() {
            return false;
        }
        let index = self.emitted;
        self.open = Some(OpenCall {
            index,
            name: name.clone(),
            ledger: String::new(),
            started: false,
        });
        self.buffer.drain(..after + end);
        events.push(ToolCallEvent::CallStart { index, name });
        true
    }

    /// Inside a call: consume one parameter, or close it.
    fn step_open_call(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let param = self
            .markers
            .param
            .expect("a streaming format has parameters");
        let invoke_close = self.markers.invoke.map(|tag| tag.close);
        let block_close = self.markers.close;

        let param_at = self.buffer.find(param.open);
        let end_at = invoke_close
            .and_then(|close| self.buffer.find(close))
            .or_else(|| self.buffer.find(block_close));

        // A close before the next parameter ends the call.
        if let Some(end) = end_at {
            if param_at.is_none_or(|p| end < p) {
                let close_len = invoke_close
                    .filter(|close| self.buffer[end..].starts_with(close))
                    .map(str::len)
                    .unwrap_or(block_close.len());
                self.buffer.drain(..end + close_len);
                let open = self.open.take().expect("a call is open");
                let arguments = close_ledger(&open);
                events.push(ToolCallEvent::CallArguments {
                    index: open.index,
                    fragment: if open.started {
                        "}".into()
                    } else {
                        "{}".into()
                    },
                });
                events.push(ToolCallEvent::CallEnd {
                    index: open.index,
                    arguments,
                });
                self.emitted = open.index + 1;
                return true;
            }
        }

        let Some(start) = param_at else {
            return false;
        };
        let (key, is_string_attr, value_start) = match self.read_param_header(start, &param) {
            Some(header) => header,
            None => return false,
        };
        // The value runs to its closing tag; without one it is still
        // arriving.
        let Some(value_end) = self.buffer[value_start..]
            .find(param.close)
            .map(|i| i + value_start)
        else {
            return false;
        };
        let raw = self.buffer[value_start..value_end].to_string();
        self.buffer.drain(..value_end + param.close.len());

        let name = self.open.as_ref().expect("a call is open").name.clone();
        let value = self.convert_value(&name, &key, &raw, is_string_attr);
        let fragment = {
            let open = self.open.as_mut().expect("a call is open");
            let lead = if open.started { "," } else { "{" };
            let fragment = format!(
                "{lead}{}:{}",
                Value::String(key),
                serde_json::to_string(&value).unwrap_or_else(|_| "null".into())
            );
            open.ledger.push_str(&fragment);
            open.started = true;
            fragment
        };
        let index = self.open.as_ref().expect("a call is open").index;
        events.push(ToolCallEvent::CallArguments { index, fragment });
        true
    }

    /// Read a parameter opener at `start`: its key, whether the wire
    /// marked it as a string, and where its value begins.
    fn read_param_header(&self, start: usize, param: &TagGrammar) -> Option<(String, bool, usize)> {
        match param.name {
            // GLM writes `<arg_key>k</arg_key><arg_value>v`.
            NameStyle::Bare if param.open == "<arg_key>" => {
                let key_start = start + param.open.len();
                let key_end = self.buffer[key_start..].find("</arg_key>")? + key_start;
                let key = self.buffer[key_start..key_end].trim().to_string();
                let value_open = "<arg_value>";
                let value_start =
                    self.buffer[key_end..].find(value_open)? + key_end + value_open.len();
                Some((key, false, value_start))
            }
            NameStyle::Bare => {
                let attrs_start = start + param.open.len();
                let gt = self.buffer[attrs_start..].find('>')? + attrs_start;
                let key = self.buffer[attrs_start..gt].trim().to_string();
                Some((key, false, gt + 1))
            }
            NameStyle::Attribute => {
                let attrs_start = start + param.open.len();
                let gt = self.buffer[attrs_start..].find('>')? + attrs_start;
                let attrs = &self.buffer[attrs_start..gt];
                let key = read_name(attrs, NameStyle::Attribute)?;
                // DeepSeek marks a value that must stay a string, which
                // beats whatever the schema says.
                let is_string = read_attribute(attrs, "string").as_deref() == Some("true");
                Some((key, is_string, gt + 1))
            }
        }
    }

    /// Type one XML-ish parameter value.
    ///
    /// Precedence: an explicit wire marker, then the tool's declared
    /// schema, then a best-effort JSON parse. A value the schema calls
    /// a string is never parsed, which is what keeps `"018956"` and
    /// `"1.0"` intact.
    fn convert_value(&self, tool: &str, key: &str, raw: &str, wire_says_string: bool) -> Value {
        let trimmed = match self.markers.trim_newlines {
            TrimStyle::None => raw,
            TrimStyle::One => raw
                .strip_prefix('\n')
                .unwrap_or(raw)
                .strip_suffix('\n')
                .unwrap_or_else(|| raw.strip_prefix('\n').unwrap_or(raw)),
            TrimStyle::All => raw.trim(),
        };
        if wire_says_string {
            return Value::String(trimmed.to_string());
        }
        let declared = self.tools.get(tool).and_then(|t| t.parameter_type(key));
        match declared.as_deref() {
            Some("string") | Some("str") | Some("enum") => Value::String(trimmed.to_string()),
            Some("integer") | Some("int") => trimmed
                .parse::<i64>()
                .map(Value::from)
                .unwrap_or_else(|_| Value::String(trimmed.to_string())),
            Some("number") | Some("float") | Some("double") => trimmed
                .parse::<f64>()
                .map(Value::from)
                .unwrap_or_else(|_| Value::String(trimmed.to_string())),
            Some("boolean") | Some("bool") => Value::Bool(trimmed.eq_ignore_ascii_case("true")),
            Some("object") | Some("array") => {
                serde_json::from_str(trimmed).unwrap_or_else(|_| Value::String(trimmed.to_string()))
            }
            // Undeclared: a best-effort parse, keeping the text
            // whenever it is not obviously something else.
            _ => parse_loose(trimmed),
        }
    }

    // ---- one-shot parsers, per family ----

    fn parse_json_blocks(&self, text: &str, open: &str, close: &str) -> (String, Vec<ToolCall>) {
        let mut normal = String::new();
        let mut calls = Vec::new();
        let mut cursor = 0usize;
        while let Some(start) = text[cursor..].find(open).map(|i| i + cursor) {
            normal.push_str(&text[cursor..start]);
            let body_start = start + open.len();
            let Some(end) = text[body_start..].find(close).map(|i| i + body_start) else {
                normal.push_str(&text[start..]);
                cursor = text.len();
                break;
            };
            if let Some(call) = self.call_from_json(text[body_start..end].trim(), calls.len()) {
                calls.push(call);
            }
            cursor = end + close.len();
        }
        normal.push_str(&text[cursor..]);
        (normal, calls)
    }

    /// Llama 3: one or more bare JSON objects after a marker, or a
    /// response that simply starts with one.
    fn parse_bare_json(&self, text: &str, marker: &str) -> (String, Vec<ToolCall>) {
        let (normal, payload) = match text.find(marker) {
            Some(index) => (text[..index].to_string(), &text[index + marker.len()..]),
            None => (String::new(), text),
        };
        let mut calls = Vec::new();
        let mut rest = payload.trim_start();
        while rest.starts_with('{') {
            let Some(end) = json_object_end(rest) else {
                break;
            };
            if let Some(call) = self.call_from_json(&rest[..end], calls.len()) {
                calls.push(call);
            }
            rest = rest[end..]
                .trim_start()
                .trim_start_matches([';', ','])
                .trim_start();
        }
        (normal, calls)
    }

    fn parse_mistral(&self, text: &str) -> (String, Vec<ToolCall>) {
        let marker = "[TOOL_CALLS]";
        let Some(index) = text.find(marker) else {
            return (text.to_string(), Vec::new());
        };
        let normal = text[..index].to_string();
        let rest = text[index + marker.len()..].trim_start();
        let Some(end) = json_array_end(rest) else {
            return (text.to_string(), Vec::new());
        };
        let Ok(Value::Array(items)) = serde_json::from_str::<Value>(&rest[..end]) else {
            return (text.to_string(), Vec::new());
        };
        let mut calls = Vec::new();
        for item in items {
            if let Some(call) = self.call_from_value(&item, calls.len()) {
                calls.push(call);
            }
        }
        (normal, calls)
    }

    /// gpt-oss: a commentary channel addressed to a function, whose
    /// message body is the arguments JSON.
    fn parse_harmony(&self, text: &str) -> (String, Vec<ToolCall>) {
        let mut normal = String::new();
        let mut calls = Vec::new();
        let mut cursor = 0usize;
        while let Some(channel) = text[cursor..].find("<|channel|>").map(|i| i + cursor) {
            let header_start = channel + "<|channel|>".len();
            let Some(message) = text[header_start..]
                .find("<|message|>")
                .map(|i| i + header_start)
            else {
                break;
            };
            let header = &text[header_start..message];
            let body_start = message + "<|message|>".len();
            let (end, matched) = ["<|end|>", "<|return|>", "<|call|>", "<|start|>"]
                .iter()
                .filter_map(|marker| {
                    text[body_start..]
                        .find(marker)
                        .map(|i| (i + body_start, *marker))
                })
                .min_by_key(|(index, _)| *index)
                .map(|(index, marker)| (index, Some(marker)))
                .unwrap_or((text.len(), None));

            if let Some(name) = header
                .split_whitespace()
                .find_map(|token| token.strip_prefix("to=functions."))
            {
                normal.push_str(&text[cursor..channel]);
                let arguments = normalize_arguments(&text[body_start..end]);
                if self.known(name) {
                    calls.push(ToolCall {
                        index: calls.len(),
                        name: name.to_string(),
                        arguments,
                    });
                }
            } else {
                normal.push_str(&text[cursor..end]);
            }
            cursor = match matched {
                Some(marker) => end + marker.len(),
                None => text.len(),
            };
        }
        normal.push_str(&text[cursor..]);
        (normal, calls)
    }

    /// Gemma 4: `call:name{key: value, …}` with its own quoting.
    fn parse_gemma(&self, text: &str) -> (String, Vec<ToolCall>) {
        let mut normal = String::new();
        let mut calls = Vec::new();
        let mut cursor = 0usize;
        while let Some(start) = text[cursor..].find("<|tool_call>").map(|i| i + cursor) {
            normal.push_str(&text[cursor..start]);
            let body_start = start + "<|tool_call>".len();
            let Some(end) = text[body_start..]
                .find("<tool_call|>")
                .map(|i| i + body_start)
            else {
                normal.push_str(&text[start..]);
                cursor = text.len();
                break;
            };
            let body = text[body_start..end].trim();
            if let Some(rest) = body.strip_prefix("call:") {
                if let Some(brace) = rest.find('{') {
                    let name = rest[..brace].trim();
                    let args = rest[brace + 1..].trim_end().trim_end_matches('}');
                    if self.known(name) {
                        calls.push(ToolCall {
                            index: calls.len(),
                            name: name.to_string(),
                            arguments: gemma_arguments(args),
                        });
                    }
                }
            }
            cursor = end + "<tool_call|>".len();
        }
        normal.push_str(&text[cursor..]);
        (normal, calls)
    }

    fn parse_glm(&self, text: &str) -> (String, Vec<ToolCall>) {
        self.replay_streaming(text)
    }

    fn parse_invoke(&self, text: &str) -> (String, Vec<ToolCall>) {
        self.replay_streaming(text)
    }

    /// Run the streaming machinery on a fresh parser.
    ///
    /// The invoke families have exactly one implementation of their
    /// grammar, so a streamed response and a buffered one cannot
    /// disagree about what the model said.
    fn replay_streaming(&self, text: &str) -> (String, Vec<ToolCall>) {
        let mut parser = ToolCallParser::new(self.format, self.tools.values().cloned().collect());
        let mut events = parser.push(text);
        events.extend(parser.finish());

        let mut normal = String::new();
        let mut calls = Vec::new();
        let mut names: BTreeMap<usize, String> = BTreeMap::new();
        for event in events {
            match event {
                ToolCallEvent::Text(text) => normal.push_str(&text),
                ToolCallEvent::CallStart { index, name } => {
                    names.insert(index, name);
                }
                ToolCallEvent::CallArguments { .. } => {}
                ToolCallEvent::CallEnd { index, arguments } => {
                    if let Some(name) = names.remove(&index) {
                        if self.known(&name) {
                            calls.push(ToolCall {
                                index: calls.len(),
                                name,
                                arguments,
                            });
                        }
                    }
                }
            }
        }
        (normal, calls)
    }

    fn call_from_json(&self, text: &str, index: usize) -> Option<ToolCall> {
        let value: Value = serde_json::from_str(text).ok()?;
        self.call_from_value(&value, index)
    }

    fn call_from_value(&self, value: &Value, index: usize) -> Option<ToolCall> {
        let name = value.get("name")?.as_str()?.to_string();
        if !self.known(&name) {
            return None;
        }
        // Families disagree on the key; they never both appear.
        let arguments = value
            .get("arguments")
            .or_else(|| value.get("parameters"))
            .cloned()
            .unwrap_or(Value::Object(Map::new()));
        Some(ToolCall {
            index,
            name,
            arguments: serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".into()),
        })
    }

    /// Whether the request actually offered this tool.
    ///
    /// A name the request never offered is dropped: forwarding it makes
    /// the client execute something it did not advertise. A namespaced
    /// name (`skills:read`) is forwarded anyway -- those are routed by
    /// the client, which is where the namespace is resolved.
    fn known(&self, name: &str) -> bool {
        self.tools.contains_key(name) || name.contains(':')
    }
}

fn close_ledger(open: &OpenCall) -> String {
    if open.started {
        format!("{}}}", open.ledger)
    } else {
        "{}".to_string()
    }
}

/// Read a `name` out of a tag's attribute text.
fn read_name(attrs: &str, style: NameStyle) -> Option<String> {
    match style {
        NameStyle::Bare => {
            let name = attrs.trim();
            (!name.is_empty()).then(|| name.to_string())
        }
        NameStyle::Attribute => read_attribute(attrs, "name"),
    }
}

/// Read `key="value"` (or `key='value'`) out of attribute text.
fn read_attribute(attrs: &str, key: &str) -> Option<String> {
    let pattern = format!("{key}=");
    let start = attrs.find(&pattern)? + pattern.len();
    let rest = &attrs[start..];
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let body = &rest[quote.len_utf8()..];
    let end = body.find(quote)?;
    Some(body[..end].to_string())
}

/// Parse a value with no declared type: JSON if it plainly is JSON, the
/// text otherwise.
fn parse_loose(text: &str) -> Value {
    if text.is_empty() {
        return Value::String(String::new());
    }
    match text {
        "true" | "True" => return Value::Bool(true),
        "false" | "False" => return Value::Bool(false),
        "null" | "None" => return Value::Null,
        _ => {}
    }
    match serde_json::from_str::<Value>(text) {
        // A bare word parses as nothing; a quoted one parses as the
        // string it already was.
        Ok(Value::String(_)) | Err(_) => Value::String(text.to_string()),
        Ok(value) => value,
    }
}

/// Re-serialize an arguments payload, or wrap it if it is not an
/// object.
fn normalize_arguments(raw: &str) -> String {
    let trimmed = raw.trim();
    match serde_json::from_str::<Value>(trimmed) {
        Ok(value @ Value::Object(_)) => {
            serde_json::to_string(&value).unwrap_or_else(|_| "{}".into())
        }
        _ => "{}".to_string(),
    }
}

/// Gemma's `k: v, k: v` argument list, with `<|"|>` for quotes.
fn gemma_arguments(text: &str) -> String {
    const QUOTE: &str = "<|\"|>";
    let mut map = Map::new();
    for pair in split_top_level(text, ',') {
        let Some((key, value)) = pair.split_once(':') else {
            continue;
        };
        let key = key.trim().trim_matches('"').to_string();
        let value = value.trim();
        let parsed = match value
            .strip_prefix(QUOTE)
            .and_then(|v| v.strip_suffix(QUOTE))
        {
            Some(inner) => Value::String(inner.to_string()),
            None => parse_loose(value),
        };
        map.insert(key, parsed);
    }
    serde_json::to_string(&Value::Object(map)).unwrap_or_else(|_| "{}".into())
}

/// Split on `delimiter`, ignoring delimiters inside brackets or quotes.
fn split_top_level(text: &str, delimiter: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut quoted = false;
    let mut current = String::new();
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if text[index..].starts_with("<|\"|>") {
            quoted = !quoted;
            current.push_str("<|\"|>");
            for _ in 0..4 {
                chars.next();
            }
            continue;
        }
        if !quoted {
            match ch {
                '{' | '[' => depth += 1,
                '}' | ']' => depth -= 1,
                c if c == delimiter && depth == 0 => {
                    parts.push(std::mem::take(&mut current));
                    continue;
                }
                _ => {}
            }
        }
        current.push(ch);
    }
    if !current.trim().is_empty() {
        parts.push(current);
    }
    parts
}

/// The byte just past the JSON object starting at `text[0]`.
fn json_object_end(text: &str) -> Option<usize> {
    balanced_end(text, '{', '}')
}

fn json_array_end(text: &str) -> Option<usize> {
    balanced_end(text, '[', ']')
}

/// Scan a balanced bracket span, respecting JSON strings and escapes.
fn balanced_end(text: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            c if c == open => depth += 1,
            c if c == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(index + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<ToolSchema> {
        vec![
            ToolSchema::with_parameters(
                "get_weather",
                json!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                }}),
            ),
            ToolSchema::with_parameters(
                "write_file",
                json!({"type": "object", "properties": {
                    "path": {"type": "string"},
                    "contents": {"type": "string"},
                    "overwrite": {"type": "boolean"}
                }}),
            ),
        ]
    }

    fn parser(format: ToolCallFormat) -> ToolCallParser {
        ToolCallParser::new(format, tools())
    }

    /// Chunk by characters, not bytes: a token boundary never splits a
    /// codepoint, and the DSML markers are multi-byte.
    fn stream(parser: &mut ToolCallParser, text: &str, width: usize) -> Vec<ToolCallEvent> {
        let chars: Vec<char> = text.chars().collect();
        let mut events = Vec::new();
        for chunk in chars.chunks(width) {
            let piece: String = chunk.iter().collect();
            events.extend(parser.push(&piece));
        }
        events.extend(parser.finish());
        events
    }

    fn calls_of(events: &[ToolCallEvent]) -> Vec<(usize, String, String)> {
        let mut names: BTreeMap<usize, String> = BTreeMap::new();
        let mut out = Vec::new();
        for event in events {
            match event {
                ToolCallEvent::CallStart { index, name } => {
                    names.insert(*index, name.clone());
                }
                ToolCallEvent::CallEnd { index, arguments } => {
                    out.push((*index, names[index].clone(), arguments.clone()));
                }
                _ => {}
            }
        }
        out
    }

    fn text_of(events: &[ToolCallEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                ToolCallEvent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_hermes_call_is_recognized_with_its_arguments() {
        let parser = parser(ToolCallFormat::Qwen25);
        let (text, calls) = parser.parse_complete(
            "Let me check.<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>",
        );
        assert_eq!(text, "Let me check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Rome"}"#);
    }

    #[test]
    fn several_hermes_calls_keep_their_order() {
        let parser = parser(ToolCallFormat::Qwen25);
        let (_, calls) = parser.parse_complete(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>\n\
             <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[1].arguments, r#"{"city":"Oslo"}"#);
    }

    /// A name the request never offered must not reach the client: it
    /// would be asked to run something it did not advertise.
    #[test]
    fn a_tool_the_request_never_offered_is_dropped() {
        let parser = parser(ToolCallFormat::Qwen25);
        let (_, calls) = parser
            .parse_complete("<tool_call>{\"name\": \"rm_rf\", \"arguments\": {}}</tool_call>");
        assert!(calls.is_empty());
    }

    /// ... but a namespaced name is forwarded, because the client is
    /// what resolves the namespace.
    #[test]
    fn a_namespaced_tool_is_forwarded() {
        let parser = parser(ToolCallFormat::Qwen25);
        let (_, calls) = parser.parse_complete(
            "<tool_call>{\"name\": \"skills:read\", \"arguments\": {}}</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "skills:read");
    }

    #[test]
    fn llama_and_mistral_payloads_are_recognized() {
        let llama = parser(ToolCallFormat::Llama3);
        let (text, calls) = llama.parse_complete(
            "sure<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}",
        );
        assert_eq!(text, "sure");
        assert_eq!(calls[0].arguments, r#"{"city":"Rome"}"#);

        let mistral = parser(ToolCallFormat::Mistral);
        let (text, calls) = mistral.parse_complete(
            "ok [TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}]",
        );
        assert_eq!(text, "ok ");
        assert_eq!(calls[0].arguments, r#"{"city":"Oslo"}"#);
    }

    #[test]
    fn a_harmony_commentary_channel_is_a_call() {
        let parser = parser(ToolCallFormat::GptOss);
        let (text, calls) = parser.parse_complete(
            "<|channel|>commentary to=functions.get_weather<|message|>{\"city\": \"Rome\"}<|call|>",
        );
        assert!(text.is_empty(), "{text:?}");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Rome"}"#);
    }

    #[test]
    fn a_gemma_call_uses_its_own_quoting() {
        let parser = parser(ToolCallFormat::Gemma4);
        let (_, calls) = parser.parse_complete(
            "<|tool_call>call:get_weather{city: <|\"|>Rome<|\"|>, days: 3}<tool_call|>",
        );
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["city"], json!("Rome"));
        assert_eq!(args["days"], json!(3));
    }

    #[test]
    fn a_qwen_coder_call_streams_its_arguments() {
        let mut parser = parser(ToolCallFormat::Qwen3Coder);
        let wire = "<tool_call><function=write_file>\
                    <parameter=path>\n/tmp/x\n</parameter>\
                    <parameter=contents>\nhello\n</parameter>\
                    </function></tool_call>";
        let events = stream(&mut parser, wire, 7);
        let calls = calls_of(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "write_file");
        let args: Value = serde_json::from_str(&calls[0].2).unwrap();
        assert_eq!(args["path"], json!("/tmp/x"));
        assert_eq!(args["contents"], json!("hello"));
    }

    /// The streaming contract: the fragments concatenate to exactly the
    /// final arguments, so a client can parse what it accumulated.
    #[test]
    fn streamed_fragments_concatenate_to_the_final_arguments() {
        for format in [
            ToolCallFormat::Qwen3Coder,
            ToolCallFormat::MiniMax,
            ToolCallFormat::DeepSeekV32,
            ToolCallFormat::Glm47,
        ] {
            let wire = wire_for(format);
            for width in [1usize, 5, 17, 4096] {
                let mut parser = parser(format);
                let events = stream(&mut parser, &wire, width);
                let calls = calls_of(&events);
                assert_eq!(calls.len(), 1, "{format:?} width {width}: {events:?}");

                let joined: String = events
                    .iter()
                    .filter_map(|e| match e {
                        ToolCallEvent::CallArguments { fragment, .. } => Some(fragment.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(joined, calls[0].2, "{format:?} width {width}");
                serde_json::from_str::<Value>(&joined)
                    .unwrap_or_else(|e| panic!("{format:?} width {width}: {e} in {joined}"));
            }
        }
    }

    /// And streaming agrees with a buffered parse of the same wire.
    #[test]
    fn streaming_agrees_with_one_shot_for_every_invoke_family() {
        for format in [
            ToolCallFormat::Qwen3Coder,
            ToolCallFormat::MiniMax,
            ToolCallFormat::DeepSeekV32,
            ToolCallFormat::Glm47,
        ] {
            let wire = wire_for(format);
            let (_, complete) = parser(format).parse_complete(&wire);
            let mut streamed = parser(format);
            let events = stream(&mut streamed, &wire, 3);
            let calls = calls_of(&events);
            assert_eq!(calls.len(), complete.len(), "{format:?}");
            assert_eq!(calls[0].1, complete[0].name, "{format:?}");
            assert_eq!(calls[0].2, complete[0].arguments, "{format:?}");
        }
    }

    fn wire_for(format: ToolCallFormat) -> String {
        match format {
            ToolCallFormat::Qwen3Coder => "<tool_call><function=get_weather>\
                 <parameter=city>\nRome\n</parameter>\
                 <parameter=days>\n3\n</parameter>\
                 </function></tool_call>"
                .to_string(),
            ToolCallFormat::MiniMax => "<minimax:tool_call>\
                 <invoke name=\"get_weather\">\
                 <parameter name=\"city\">Rome</parameter>\
                 <parameter name=\"days\">3</parameter>\
                 </invoke></minimax:tool_call>"
                .to_string(),
            ToolCallFormat::DeepSeekV32 => "<｜DSML｜function_calls>\
                 <｜DSML｜invoke name=\"get_weather\">\
                 <｜DSML｜parameter name=\"city\" string=\"true\">Rome</｜DSML｜parameter>\
                 <｜DSML｜parameter name=\"days\">3</｜DSML｜parameter>\
                 </｜DSML｜invoke></｜DSML｜function_calls>"
                .to_string(),
            ToolCallFormat::Glm47 => "<tool_call>get_weather\n\
                 <arg_key>city</arg_key><arg_value>Rome</arg_value>\
                 <arg_key>days</arg_key><arg_value>3</arg_value>\
                 </tool_call>"
                .to_string(),
            _ => unreachable!("only the streaming families have invoke wire"),
        }
    }

    /// A declared string parameter is handed over verbatim, so a
    /// zero-padded id survives.
    #[test]
    fn a_declared_string_is_never_reparsed() {
        let tools = vec![ToolSchema::with_parameters(
            "lookup",
            json!({"type": "object", "properties": {"id": {"type": "string"}}}),
        )];
        let parser = ToolCallParser::new(ToolCallFormat::MiniMax, tools);
        let (_, calls) = parser.parse_complete(
            "<minimax:tool_call><invoke name=\"lookup\">\
             <parameter name=\"id\">018956</parameter></invoke></minimax:tool_call>",
        );
        assert_eq!(calls[0].arguments, r#"{"id":"018956"}"#);
    }

    /// ... and a declared number stays a number, including a float
    /// written with a trailing zero.
    #[test]
    fn a_declared_number_keeps_its_type() {
        let tools = vec![ToolSchema::with_parameters(
            "scale",
            json!({"type": "object", "properties": {
                "factor": {"type": "number"},
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"}
            }}),
        )];
        let parser = ToolCallParser::new(ToolCallFormat::MiniMax, tools);
        let (_, calls) = parser.parse_complete(
            "<minimax:tool_call><invoke name=\"scale\">\
             <parameter name=\"factor\">5.0</parameter>\
             <parameter name=\"count\">7</parameter>\
             <parameter name=\"enabled\">true</parameter>\
             </invoke></minimax:tool_call>",
        );
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert!(args["factor"].is_f64(), "{args}");
        assert_eq!(args["count"], json!(7));
        assert_eq!(args["enabled"], json!(true));
    }

    /// DeepSeek marks a value as a string on the wire, and that beats
    /// any guess.
    #[test]
    fn the_wire_string_marker_wins_over_a_guess() {
        let tools = vec![ToolSchema::new("run")];
        let parser = ToolCallParser::new(ToolCallFormat::DeepSeekV32, tools);
        let (_, calls) = parser.parse_complete(
            "<｜DSML｜function_calls><｜DSML｜invoke name=\"run\">\
             <｜DSML｜parameter name=\"cmd\" string=\"true\">123</｜DSML｜parameter>\
             </｜DSML｜invoke></｜DSML｜function_calls>",
        );
        assert_eq!(calls[0].arguments, r#"{"cmd":"123"}"#);
    }

    /// A file's contents must survive verbatim -- the case a coding
    /// agent lives on.
    #[test]
    fn a_multiline_value_survives_verbatim() {
        let wire = "<minimax:tool_call><invoke name=\"write_file\">\
                    <parameter name=\"path\">/tmp/x.rs</parameter>\
                    <parameter name=\"contents\">fn main() {\n    println!(\"hi\");\n}</parameter>\
                    </invoke></minimax:tool_call>";
        let (_, calls) = parser(ToolCallFormat::MiniMax).parse_complete(wire);
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(
            args["contents"],
            json!("fn main() {\n    println!(\"hi\");\n}")
        );
    }

    #[test]
    fn a_call_with_no_arguments_still_reports_an_object() {
        let tools = vec![ToolSchema::new("ping")];
        let parser = ToolCallParser::new(ToolCallFormat::MiniMax, tools);
        let (_, calls) = parser.parse_complete(
            "<minimax:tool_call><invoke name=\"ping\"></invoke></minimax:tool_call>",
        );
        assert_eq!(calls[0].arguments, "{}");
    }

    /// Text either side of a call must arrive as text, in order.
    #[test]
    fn text_around_a_call_is_preserved_in_order() {
        let mut parser = parser(ToolCallFormat::Qwen25);
        let events = stream(
            &mut parser,
            "before <tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call> after",
            6,
        );
        let text = text_of(&events);
        assert!(text.starts_with("before "), "{text:?}");
        assert!(text.ends_with(" after"), "{text:?}");
        assert_eq!(calls_of(&events).len(), 1);
    }

    /// No prefix of a marker may reach the wire as text, or a client
    /// renders `<tool` and then has to take it back.
    #[test]
    fn a_partial_marker_is_never_streamed_as_text() {
        let mut parser = parser(ToolCallFormat::Qwen25);
        let events = parser.push("hello <tool");
        assert_eq!(text_of(&events), "hello ");
        let events =
            parser.push("_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>");
        assert_eq!(text_of(&events), "");
        assert_eq!(calls_of(&events).len(), 1);
    }

    /// A streamed invoke grammar closes twice -- the call, then the
    /// block -- and only the first belongs to the call. The leftover
    /// must not be printed into the answer.
    #[test]
    fn a_block_close_left_over_from_a_call_is_not_content() {
        let wire = "<tool_call><function=get_weather>\
                    <parameter=city>\nRome\n</parameter>\
                    </function></tool_call>";
        for width in [1usize, 6, 4096] {
            let mut parser = parser(ToolCallFormat::Qwen3Coder);
            let events = stream(&mut parser, wire, width);
            assert_eq!(calls_of(&events).len(), 1, "width {width}");
            assert_eq!(
                text_of(&events),
                "",
                "width {width}: structure leaked as content"
            );
        }
    }

    /// A held partial that turns out to be ordinary text is released,
    /// not dropped.
    #[test]
    fn a_partial_that_is_not_a_marker_comes_back() {
        let mut parser = parser(ToolCallFormat::Qwen25);
        let events = stream(&mut parser, "compare a <tool b", 4);
        assert_eq!(text_of(&events), "compare a <tool b");
    }

    /// A generation truncated mid-call still leaves the client with
    /// parseable JSON.
    #[test]
    fn a_truncated_call_is_closed_with_what_it_has() {
        let mut parser = parser(ToolCallFormat::MiniMax);
        let events = stream(
            &mut parser,
            "<minimax:tool_call><invoke name=\"get_weather\">\
             <parameter name=\"city\">Rome</parameter>",
            9,
        );
        let calls = calls_of(&events);
        assert_eq!(calls.len(), 1);
        serde_json::from_str::<Value>(&calls[0].2).expect("valid JSON despite truncation");
    }

    /// Llama 3 and Mistral have no closing marker, so the end of the
    /// stream is the only place their calls can be recognized. A parser
    /// that dropped its buffer there because it "looked like a call"
    /// would lose every call those two families make.
    #[test]
    fn a_format_with_no_closing_marker_still_emits_at_the_end() {
        for (format, wire) in [
            (
                ToolCallFormat::Llama3,
                "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}",
            ),
            (
                ToolCallFormat::Mistral,
                "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}]",
            ),
        ] {
            let mut parser = parser(format);
            let events = stream(&mut parser, wire, 6);
            let calls = calls_of(&events);
            assert_eq!(calls.len(), 1, "{format:?}: {events:?}");
            assert_eq!(calls[0].1, "get_weather", "{format:?}");
            assert_eq!(calls[0].2, r#"{"city":"Rome"}"#, "{format:?}");
        }
    }

    /// Qwen3-Coder often emits a bare `<function=…>` without the
    /// `<tool_call>` wrapper its template shows. The invoke tag opens a
    /// call just as definitively.
    #[test]
    fn an_unwrapped_invoke_tag_still_opens_a_call() {
        let wire = "sure: <function=get_weather><parameter=city>\nRome\n</parameter></function>";
        let mut streamed = parser(ToolCallFormat::Qwen3Coder);
        // Push everything but do NOT finish: the call must be
        // recognized while the stream is running, not salvaged from the
        // buffer at the end.
        let mut events = Vec::new();
        for chunk in wire.chars().collect::<Vec<_>>().chunks(5) {
            events.extend(streamed.push(&chunk.iter().collect::<String>()));
        }
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ToolCallEvent::CallStart { .. })),
            "the call must open mid-stream: {events:?}"
        );
        events.extend(streamed.finish());
        let calls = calls_of(&events);
        assert_eq!(calls.len(), 1, "{events:?}");
        assert_eq!(calls[0].2, r#"{"city":"Rome"}"#);
        assert_eq!(text_of(&events), "sure: ");

        let (_, complete) = parser(ToolCallFormat::Qwen3Coder).parse_complete(wire);
        assert_eq!(complete.len(), 1);
        assert_eq!(complete[0].arguments, r#"{"city":"Rome"}"#);
    }

    #[test]
    fn a_response_with_no_tools_offered_is_all_text() {
        let mut parser = ToolCallParser::new(ToolCallFormat::Qwen25, Vec::new());
        let wire = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>";
        let events = stream(&mut parser, wire, 8);
        assert_eq!(text_of(&events), wire);
        assert!(calls_of(&events).is_empty());
        assert_eq!(parser.parse_complete(wire).1.len(), 0);
    }

    /// The inference order is load-bearing: the specific families must
    /// win over the general ones they resemble.
    #[test]
    fn format_inference_prefers_the_specific_family() {
        assert_eq!(
            ToolCallFormat::infer("Qwen3-Coder-30B"),
            ToolCallFormat::Qwen3Coder
        );
        assert_eq!(ToolCallFormat::infer("Qwen2.5-7B"), ToolCallFormat::Qwen25);
        assert_eq!(
            ToolCallFormat::infer("DeepSeek-V4-Flash"),
            ToolCallFormat::DeepSeekV32
        );
        assert_eq!(ToolCallFormat::infer("GLM-5.2"), ToolCallFormat::Glm47);
        assert_eq!(
            ToolCallFormat::infer("gpt-oss-120b"),
            ToolCallFormat::GptOss
        );
        assert_eq!(
            ToolCallFormat::infer("something-unknown"),
            ToolCallFormat::Llama3,
            "the fallback is the shape an untrained model improvises"
        );
        assert_eq!(ToolCallFormat::parse("qwen"), Some(ToolCallFormat::Qwen25));
        assert_eq!(ToolCallFormat::parse("nonsense"), None);
    }

    #[test]
    fn the_cheap_marker_test_answers_before_any_parsing() {
        assert!(!ToolCallParser::text_may_contain_call("just an answer"));
        assert!(ToolCallParser::text_may_contain_call("<tool_call>{}"));
        assert!(ToolCallParser::text_may_contain_call("[TOOL_CALLS] []"));
    }

    #[test]
    fn balanced_scanning_respects_strings_and_escapes() {
        assert_eq!(json_object_end(r#"{"a": "}"} tail"#), Some(10));
        assert_eq!(json_object_end(r#"{"a": "\""} tail"#), Some(11));
        assert_eq!(json_object_end("{unterminated"), None);
        assert_eq!(json_array_end("[1, [2], 3] tail"), Some(11));
    }
}
