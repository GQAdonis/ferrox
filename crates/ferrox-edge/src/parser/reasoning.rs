//! Splitting a reasoning model's output into its chain of thought and
//! its answer.
//!
//! A reasoning model emits both on one token stream, separated by
//! markers its template taught it. The API contract is that they arrive
//! as two fields -- `reasoning_content` and `content` -- so something
//! has to cut the stream, and it has to do it *incrementally*, because
//! the marker that separates them can straddle any token boundary.
//!
//! # The three things that make this hard
//!
//! **A marker can arrive in pieces.** `</think>` may come as `</thi` +
//! `nk>`. So any trailing run of the buffer that is a proper prefix of
//! a marker is withheld until the next token settles it -- the same
//! rule as [`crate::detokenize::stop_prefix_holdback`], for the same
//! reason: SSE cannot retract.
//!
//! **A tool call inside reasoning is not reasoning.** Some families
//! emit tool calls without ever closing the reasoning block. When a
//! tool marker appears mid-reasoning the text from the marker on is
//! *held*, not emitted, until enough of it has arrived to tell a real
//! tool call from the model quoting one in its own reasoning
//! ([`TOOL_HOLD_MAX`]). A complete `</think>` always wins over a tool
//! marker, which is what keeps a quoted marker inside a properly closed
//! reasoning block classified as reasoning.
//!
//! **Some templates open the block in the prompt.** A model whose
//! prompt already ends in `<think>` never emits the opening marker, so
//! the parser has to start *inside* the block. That is
//! `force_reasoning`, and it is derived from the request
//! ([`crate::effort::resolve_thinking_mode`]) rather than from the
//! model name, because the same checkpoint does it or does not
//! depending on how the request was rendered.
//!
//! Ported 1:1 from FreeToken's `server/reasoning_parser.py`
//! (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

/// How much text may be held after a mid-reasoning tool marker before
/// it is committed as a real tool call.
///
/// A model quoting `<tool_call>` inside its reasoning writes a few more
/// words and then closes the block; a model actually calling a tool
/// writes a long JSON payload. The bound turns "which is this?" into a
/// question that answers itself.
pub const TOOL_HOLD_MAX: usize = 512;

/// Which family's markers to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// `<think>` / `</think>`, with a DSML tool marker that can appear
    /// before the block closes. Its template always opens the block.
    DeepSeekV32,
    /// Plain `<think>` / `</think>`: Qwen3, GLM.
    Think,
    /// Plain `<think>` / `</think>`, but the template always opens the
    /// block.
    ThinkAlwaysOpen,
    /// MiniMax-M3's namespaced markers, plus its adaptive-thinking
    /// quirk.
    MiniMaxM3,
    /// Gemma-4's channel markers.
    Gemma4,
    /// The gpt-oss "harmony" channel format, which is a different shape
    /// of grammar entirely.
    GptOss,
}

impl ReasoningFormat {
    /// The parser name a client or config would use.
    pub fn as_str(self) -> &'static str {
        match self {
            ReasoningFormat::DeepSeekV32 => "deepseekv32",
            ReasoningFormat::Think => "qwen3",
            ReasoningFormat::ThinkAlwaysOpen => "minimax",
            ReasoningFormat::MiniMaxM3 => "minimax_m3",
            ReasoningFormat::Gemma4 => "gemma4",
            ReasoningFormat::GptOss => "gpt_oss",
        }
    }

    /// Resolve a configured parser name.
    pub fn parse(name: &str) -> Option<ReasoningFormat> {
        match name {
            "deepseekv32" => Some(ReasoningFormat::DeepSeekV32),
            "qwen3" | "glm" => Some(ReasoningFormat::Think),
            "minimax" => Some(ReasoningFormat::ThinkAlwaysOpen),
            "minimax_m3" => Some(ReasoningFormat::MiniMaxM3),
            "gemma4" => Some(ReasoningFormat::Gemma4),
            "gpt_oss" | "gpt-oss" => Some(ReasoningFormat::GptOss),
            _ => None,
        }
    }

    /// Which parser a checkpoint's identity implies.
    ///
    /// Order matters: the specific arms have to be tested before the
    /// family arms they are a special case of, or MiniMax-M3 resolves
    /// to MiniMax and loses its namespaced markers.
    pub fn infer(marker: &str) -> Option<ReasoningFormat> {
        let marker = marker.to_ascii_lowercase();
        let has = |needle: &str| marker.contains(needle);
        if has("gpt_oss") || has("gpt-oss") || has("gptoss") {
            return Some(ReasoningFormat::GptOss);
        }
        if has("deepseek") && (has("v4") || has("v3.2") || has("v32")) {
            return Some(ReasoningFormat::DeepSeekV32);
        }
        if has("qwen3") {
            return Some(ReasoningFormat::Think);
        }
        if has("glm") {
            return Some(ReasoningFormat::Think);
        }
        if has("minimax_m3") || has("minimax-m3") || has("minimaxm3") {
            return Some(ReasoningFormat::MiniMaxM3);
        }
        if has("minimax") {
            return Some(ReasoningFormat::ThinkAlwaysOpen);
        }
        if has("gemma4") {
            return Some(ReasoningFormat::Gemma4);
        }
        None
    }

    fn markers(self) -> Markers {
        match self {
            ReasoningFormat::DeepSeekV32 => Markers {
                start: "<think>",
                end: "</think>",
                // The tool grammar can open before the reasoning block
                // closes, so the reasoning parser has to know about it.
                tool_start: Some("<｜DSML｜"),
                always_open: true,
            },
            ReasoningFormat::Think => Markers {
                start: "<think>",
                end: "</think>",
                tool_start: None,
                always_open: false,
            },
            ReasoningFormat::ThinkAlwaysOpen => Markers {
                start: "<think>",
                end: "</think>",
                tool_start: None,
                always_open: true,
            },
            ReasoningFormat::MiniMaxM3 => Markers {
                start: "<mm:think>",
                end: "</mm:think>",
                tool_start: Some("]<]minimax[>[<tool_call>"),
                always_open: false,
            },
            ReasoningFormat::Gemma4 => Markers {
                start: "<|channel>thought\n",
                end: "<channel|>",
                tool_start: None,
                always_open: false,
            },
            // Not marker-delimited; see `HarmonyState`.
            ReasoningFormat::GptOss => Markers {
                start: "",
                end: "",
                tool_start: None,
                always_open: false,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Markers {
    start: &'static str,
    end: &'static str,
    tool_start: Option<&'static str>,
    /// Whether this family's template opens the reasoning block itself,
    /// so the model never emits the opening marker.
    always_open: bool,
}

/// One increment's worth of split output. Either side may be empty.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReasoningDelta {
    pub reasoning: String,
    pub content: String,
}

impl ReasoningDelta {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty()
    }
}

/// The gpt-oss channel format's incremental state.
#[derive(Debug, Default)]
struct HarmonyState {
    buffer: String,
    emitted_reasoning: usize,
    emitted_content: usize,
    /// Whether the last scan ran in passthrough mode -- output with no
    /// channel markers in it at all. The emitted counters index into
    /// the scan's own output, so they are meaningless across a mode
    /// change and are reset when one happens.
    passthrough: bool,
}

const HARMONY_CHANNEL: &str = "<|channel|>";
const HARMONY_MESSAGE: &str = "<|message|>";
/// Anything that can end a harmony message body.
const HARMONY_BOUNDARIES: [&str; 5] = [
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|start|>",
    "<|channel|>",
];
/// The subset that is a *closing* token, and so belongs to the message
/// it closes rather than to whatever follows.
const HARMONY_CLOSERS: [&str; 3] = ["<|end|>", "<|return|>", "<|call|>"];

/// Splits a model's output into reasoning and answer.
#[derive(Debug)]
pub struct ReasoningParser {
    format: ReasoningFormat,
    markers: Markers,
    /// Whether output is currently inside a reasoning block.
    in_reasoning: bool,
    /// Whether to stream reasoning as it arrives, or hold it until the
    /// block closes.
    stream_reasoning: bool,
    buffer: String,
    stripped_start: bool,
    harmony: HarmonyState,
    /// MiniMax-M3 only: the template may or may not have opened the
    /// block, and the model says which by whether it leads with a bare
    /// closer.
    leading_closer_pending: bool,
    head_buffer: String,
}

impl ReasoningParser {
    /// `force_reasoning` starts the parser inside the block, for a
    /// template that opened it in the prompt.
    pub fn new(format: ReasoningFormat, force_reasoning: bool, stream_reasoning: bool) -> Self {
        let markers = format.markers();
        let force = force_reasoning || markers.always_open;
        ReasoningParser {
            format,
            markers,
            in_reasoning: force,
            stream_reasoning,
            buffer: String::new(),
            stripped_start: false,
            harmony: HarmonyState::default(),
            leading_closer_pending: format == ReasoningFormat::MiniMaxM3 && !force,
            head_buffer: String::new(),
        }
    }

    pub fn format(&self) -> ReasoningFormat {
        self.format
    }

    /// Split a complete response.
    ///
    /// Runs the streaming machinery on a fresh parser rather than
    /// having a second implementation: two implementations of a rule
    /// this fiddly disagree eventually, and the disagreement shows up
    /// as a streamed answer that differs from the non-streamed one.
    pub fn parse_complete(&self, text: &str) -> ReasoningDelta {
        let mut clone = ReasoningParser::new(self.format, self.in_reasoning, true);
        clone.leading_closer_pending = self.leading_closer_pending;
        let mut out = clone.push(text);
        let tail = clone.flush();
        out.reasoning.push_str(&tail.reasoning);
        out.content.push_str(&tail.content);
        ReasoningDelta {
            reasoning: out.reasoning.trim().to_string(),
            content: out.content.trim().to_string(),
        }
    }

    /// Feed one increment of model output.
    pub fn push(&mut self, chunk: &str) -> ReasoningDelta {
        if self.format == ReasoningFormat::GptOss {
            return self.push_harmony(chunk);
        }
        if self.leading_closer_pending {
            if let Some(delta) = self.push_leading_closer(chunk) {
                return delta;
            }
        }
        self.push_marker(chunk)
    }

    /// Whatever is still held when the stream ends.
    ///
    /// Released, not dropped: it was withheld against a marker that
    /// never arrived, so it is ordinary output, and discarding it would
    /// truncate every answer whose tail happens to look like the start
    /// of a marker.
    pub fn flush(&mut self) -> ReasoningDelta {
        if self.format == ReasoningFormat::GptOss {
            let (reasoning, content) = self.harmony_scan(false);
            let delta = self.harmony_delta(reasoning, content);
            self.harmony = HarmonyState::default();
            return delta;
        }
        if self.leading_closer_pending && !self.head_buffer.is_empty() {
            // The model never produced the bare closer, so the head was
            // ordinary output all along.
            let head = std::mem::take(&mut self.head_buffer);
            self.leading_closer_pending = false;
            let mut delta = self.push_marker(&head);
            let tail = self.flush();
            delta.reasoning.push_str(&tail.reasoning);
            delta.content.push_str(&tail.content);
            return delta;
        }
        let held = std::mem::take(&mut self.buffer);
        if held.is_empty() {
            return ReasoningDelta::default();
        }
        // A held run that begins with a tool marker is a tool call,
        // whatever the block state said.
        if let Some(tool) = self.markers.tool_start {
            if held.trim_start().starts_with(tool) {
                self.in_reasoning = false;
                return ReasoningDelta {
                    content: held,
                    ..Default::default()
                };
            }
        }
        if self.in_reasoning {
            ReasoningDelta {
                reasoning: held,
                ..Default::default()
            }
        } else {
            ReasoningDelta {
                content: held,
                ..Default::default()
            }
        }
    }

    /// MiniMax-M3's adaptive mode: the model signals "no thinking this
    /// turn" by leading with a bare closing marker, so the head of the
    /// stream is held until it is clear which it is.
    fn push_leading_closer(&mut self, chunk: &str) -> Option<ReasoningDelta> {
        self.head_buffer.push_str(chunk);
        let head = self.head_buffer.trim_start();
        if let Some(rest) = head.strip_prefix(self.markers.end) {
            let rest = rest.to_string();
            self.head_buffer.clear();
            self.leading_closer_pending = false;
            self.in_reasoning = false;
            return Some(self.push_marker(&rest));
        }
        if is_partial_prefix(head, self.markers.end) {
            // Still could become the bare closer; keep holding.
            return Some(ReasoningDelta::default());
        }
        // It diverged: replay the held head verbatim through the normal
        // machinery.
        let head = std::mem::take(&mut self.head_buffer);
        self.leading_closer_pending = false;
        Some(self.push_marker(&head))
    }

    fn push_marker(&mut self, chunk: &str) -> ReasoningDelta {
        self.buffer.push_str(chunk);

        if !self.stripped_start && !self.markers.start.is_empty() {
            if let Some(index) = self.buffer.find(self.markers.start) {
                let mut rebuilt = String::with_capacity(self.buffer.len());
                rebuilt.push_str(&self.buffer[..index]);
                rebuilt.push_str(&self.buffer[index + self.markers.start.len()..]);
                self.buffer = rebuilt;
                self.stripped_start = true;
                self.in_reasoning = true;
            }
        }

        // A complete closing marker beats everything: it is the only
        // thing that definitively ends the block, and honouring a tool
        // marker first would misclassify a tool marker the model merely
        // quoted inside its reasoning.
        if self.in_reasoning && !self.markers.end.is_empty() {
            if let Some(end) = self.buffer.find(self.markers.end) {
                let reasoning = self.buffer[..end].trim_end().to_string();
                let content = self.buffer[end + self.markers.end.len()..].to_string();
                self.buffer.clear();
                self.in_reasoning = false;
                return ReasoningDelta { reasoning, content };
            }
        }

        // A tool marker while still in reasoning: hold from the marker
        // on, and commit only once enough has arrived to be sure.
        if self.in_reasoning {
            if let Some(tool) = self.markers.tool_start {
                if let Some(index) = self.buffer.find(tool) {
                    let reasoning = self.buffer[..index].to_string();
                    let held = self.buffer[index..].to_string();
                    self.buffer = held.clone();
                    if held.len() > TOOL_HOLD_MAX {
                        self.buffer.clear();
                        self.in_reasoning = false;
                        return ReasoningDelta {
                            reasoning,
                            content: held,
                        };
                    }
                    return ReasoningDelta {
                        reasoning,
                        ..Default::default()
                    };
                }
            }
        }

        let hold = self.trailing_partial_len();
        let safe_end = self.buffer.len() - hold;
        if self.in_reasoning && !self.stream_reasoning {
            // Reasoning is being withheld wholesale; nothing goes out
            // until the block closes.
            return ReasoningDelta::default();
        }
        let safe: String = self.buffer.drain(..safe_end).collect();
        if safe.is_empty() {
            return ReasoningDelta::default();
        }
        if self.in_reasoning {
            ReasoningDelta {
                reasoning: safe,
                ..Default::default()
            }
        } else {
            ReasoningDelta {
                content: safe,
                ..Default::default()
            }
        }
    }

    /// The longest trailing run of the buffer that could still grow
    /// into one of the markers this parser tracks.
    fn trailing_partial_len(&self) -> usize {
        let mut candidates: Vec<&str> = vec![self.markers.end];
        if !self.stripped_start && !self.markers.start.is_empty() {
            candidates.push(self.markers.start);
        }
        if let Some(tool) = self.markers.tool_start {
            candidates.push(tool);
        }
        candidates.retain(|c| !c.is_empty());
        let owned: Vec<String> = candidates.iter().map(|c| c.to_string()).collect();
        let hold = crate::detokenize::stop_prefix_holdback(&self.buffer, &owned);
        let split = crate::detokenize::floor_char_boundary(&self.buffer, self.buffer.len() - hold);
        self.buffer.len() - split
    }

    fn push_harmony(&mut self, chunk: &str) -> ReasoningDelta {
        self.harmony.buffer.push_str(chunk);
        let (reasoning, content) = self.harmony_scan(true);
        self.harmony_delta(reasoning, content)
    }

    /// Emit only what this scan added beyond what has already gone out.
    fn harmony_delta(&mut self, reasoning: String, content: String) -> ReasoningDelta {
        let passthrough = !self.harmony.buffer.contains(HARMONY_CHANNEL);
        if passthrough != self.harmony.passthrough {
            // The scan's output is a different string now; offsets into
            // the old one would skip the head of the new one.
            self.harmony.emitted_reasoning = 0;
            self.harmony.emitted_content = 0;
            self.harmony.passthrough = passthrough;
        }
        let new_reasoning = reasoning
            .get(self.harmony.emitted_reasoning..)
            .unwrap_or("")
            .to_string();
        let new_content = content
            .get(self.harmony.emitted_content..)
            .unwrap_or("")
            .to_string();
        self.harmony.emitted_reasoning = reasoning.len();
        self.harmony.emitted_content = content.len();
        ReasoningDelta {
            reasoning: new_reasoning,
            content: new_content,
        }
    }

    /// Re-scan the whole harmony buffer into (reasoning, content).
    ///
    /// Re-scanning rather than advancing an index is what makes a
    /// header that arrives in pieces work: a `<|channel|>` with no
    /// `<|message|>` yet is simply not a message this time round, and
    /// becomes one when the rest lands.
    fn harmony_scan(&self, hold_partial: bool) -> (String, String) {
        let text = &self.harmony.buffer;
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut cursor = 0usize;

        if !text.contains(HARMONY_CHANNEL) {
            // Not harmony output at all -- a model that answered
            // plainly. Pass it through as content rather than
            // discarding it, holding back anything that could still
            // become a channel header.
            let mut body = text.as_str();
            if hold_partial {
                let markers = [HARMONY_CHANNEL.to_string()];
                let hold = crate::detokenize::stop_prefix_holdback(body, &markers);
                body = &body[..crate::detokenize::floor_char_boundary(body, body.len() - hold)];
            }
            return (String::new(), body.to_string());
        }

        while let Some(channel) = text[cursor..].find(HARMONY_CHANNEL).map(|i| i + cursor) {
            let header_start = channel + HARMONY_CHANNEL.len();
            let Some(message) = text[header_start..]
                .find(HARMONY_MESSAGE)
                .map(|i| i + header_start)
            else {
                // The header is still streaming.
                break;
            };
            let header = &text[header_start..message];
            let body_start = message + HARMONY_MESSAGE.len();
            let (end, matched) = earliest_marker(text, body_start, &HARMONY_BOUNDARIES);
            let channel_name = header.split_whitespace().next().unwrap_or("");
            // A commentary channel addressed to a function is a tool
            // call, and belongs to the tool parser verbatim -- markers
            // and all -- not to either text field.
            let is_tool = channel_name == "commentary" && header.contains("to=functions");

            if is_tool {
                let slice_end = match matched {
                    Some(marker) if HARMONY_CLOSERS.contains(&marker) => end + marker.len(),
                    _ => end,
                };
                content.push_str(&text[channel..slice_end]);
            } else {
                let mut body = &text[body_start..end];
                if matched.is_none() && hold_partial {
                    let markers: Vec<String> =
                        HARMONY_BOUNDARIES.iter().map(|m| m.to_string()).collect();
                    let hold = crate::detokenize::stop_prefix_holdback(body, &markers);
                    body = &body[..crate::detokenize::floor_char_boundary(body, body.len() - hold)];
                }
                if channel_name == "analysis" {
                    reasoning.push_str(body);
                } else {
                    content.push_str(body);
                }
            }
            if matched.is_none() {
                break;
            }
            cursor = end;
        }
        (reasoning, content)
    }
}

/// The earliest of `markers` at or after `from`, and which one it was.
fn earliest_marker<'a>(text: &str, from: usize, markers: &[&'a str]) -> (usize, Option<&'a str>) {
    let mut best = text.len();
    let mut which = None;
    for marker in markers {
        if let Some(index) = text[from..].find(marker).map(|i| i + from) {
            if index < best {
                best = index;
                which = Some(*marker);
            }
        }
    }
    (best, which)
}

/// Whether `text` is a non-empty proper prefix of `marker`.
fn is_partial_prefix(text: &str, marker: &str) -> bool {
    !text.is_empty() && text.len() < marker.len() && marker.starts_with(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(parser: &mut ReasoningParser, chunks: &[&str]) -> ReasoningDelta {
        let mut out = ReasoningDelta::default();
        for chunk in chunks {
            let delta = parser.push(chunk);
            out.reasoning.push_str(&delta.reasoning);
            out.content.push_str(&delta.content);
        }
        let tail = parser.flush();
        out.reasoning.push_str(&tail.reasoning);
        out.content.push_str(&tail.content);
        out
    }

    #[test]
    fn a_closed_block_splits_into_reasoning_and_answer() {
        let parser = ReasoningParser::new(ReasoningFormat::Think, false, true);
        let split = parser.parse_complete("<think>let me see</think>The answer is 4.");
        assert_eq!(split.reasoning, "let me see");
        assert_eq!(split.content, "The answer is 4.");
    }

    #[test]
    fn text_with_no_block_is_all_answer() {
        let parser = ReasoningParser::new(ReasoningFormat::Think, false, true);
        let split = parser.parse_complete("Just the answer.");
        assert_eq!(split.reasoning, "");
        assert_eq!(split.content, "Just the answer.");
    }

    /// A template that opened the block in the prompt means the model
    /// never emits `<think>`, so the parser must start inside.
    #[test]
    fn a_forced_parser_starts_inside_the_block() {
        let parser = ReasoningParser::new(ReasoningFormat::Think, true, true);
        let split = parser.parse_complete("thinking hard</think>done");
        assert_eq!(split.reasoning, "thinking hard");
        assert_eq!(split.content, "done");

        let unforced = ReasoningParser::new(ReasoningFormat::Think, false, true);
        let split = unforced.parse_complete("thinking hard</think>done");
        assert_eq!(split.reasoning, "");
        assert_eq!(split.content, "thinking hard</think>done");
    }

    /// The whole reason for the hold-back: a marker split across two
    /// chunks must not leak half of itself into the answer.
    #[test]
    fn a_marker_split_across_chunks_never_leaks() {
        let mut parser = ReasoningParser::new(ReasoningFormat::Think, true, true);
        let out = stream(&mut parser, &["thinking", "</thi", "nk>", "answer"]);
        assert_eq!(out.reasoning, "thinking");
        assert_eq!(out.content, "answer");
        assert!(!out.reasoning.contains("</"), "no partial marker leaked");
    }

    /// Streaming and one-shot must agree, whatever the chunking.
    #[test]
    fn streaming_agrees_with_one_shot() {
        let text = "<think>step one. step two.</think>The answer is 4.";
        let reference =
            ReasoningParser::new(ReasoningFormat::Think, false, true).parse_complete(text);
        for width in [1usize, 3, 7, 13, 64] {
            let mut parser = ReasoningParser::new(ReasoningFormat::Think, false, true);
            let chunks: Vec<String> = text
                .as_bytes()
                .chunks(width)
                .map(|c| String::from_utf8_lossy(c).into_owned())
                .collect();
            let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let out = stream(&mut parser, &refs);
            assert_eq!(out.reasoning.trim(), reference.reasoning, "width {width}");
            assert_eq!(out.content.trim(), reference.content, "width {width}");
        }
    }

    /// With `stream_reasoning` off, nothing goes out until the block
    /// closes -- but nothing is lost either.
    #[test]
    fn withheld_reasoning_still_arrives_in_full() {
        let mut parser = ReasoningParser::new(ReasoningFormat::Think, true, false);
        let first = parser.push("thinking ");
        assert!(first.is_empty(), "nothing yet");
        let second = parser.push("more</think>answer");
        assert_eq!(second.reasoning, "thinking more");
        assert_eq!(second.content, "answer");
    }

    /// A closing marker beats a tool marker, which is what keeps a tool
    /// marker the model merely quoted inside its reasoning classified
    /// as reasoning.
    #[test]
    fn a_closed_block_wins_over_a_quoted_tool_marker() {
        let parser = ReasoningParser::new(ReasoningFormat::DeepSeekV32, true, true);
        let split = parser.parse_complete(
            "I could call <｜DSML｜function_calls> here, but I won't.</think>No tool needed.",
        );
        assert!(split.reasoning.contains("but I won't"));
        assert_eq!(split.content, "No tool needed.");
    }

    /// ... while a tool marker followed by a real payload, with no
    /// closing marker, ends the reasoning there.
    #[test]
    fn a_real_tool_call_ends_reasoning_without_a_closing_marker() {
        let parser = ReasoningParser::new(ReasoningFormat::DeepSeekV32, true, true);
        let payload = "x".repeat(TOOL_HOLD_MAX + 1);
        let split = parser.parse_complete(&format!(
            "I should look it up.<｜DSML｜function_calls>{payload}"
        ));
        assert_eq!(split.reasoning, "I should look it up.");
        assert!(split.content.starts_with("<｜DSML｜function_calls>"));
    }

    /// A truncated stream releases what it was holding rather than
    /// dropping it.
    #[test]
    fn a_truncated_stream_releases_what_it_held() {
        let mut parser = ReasoningParser::new(ReasoningFormat::Think, true, true);
        let out = stream(&mut parser, &["thinking</thi"]);
        assert_eq!(out.reasoning, "thinking</thi");
        assert!(out.content.is_empty());
    }

    #[test]
    fn the_harmony_format_routes_analysis_to_reasoning() {
        let parser = ReasoningParser::new(ReasoningFormat::GptOss, false, true);
        let split = parser.parse_complete(
            "<|channel|>analysis<|message|>weighing it up<|end|>\
             <|channel|>final<|message|>The answer is 4.<|return|>",
        );
        assert_eq!(split.reasoning, "weighing it up");
        assert_eq!(split.content, "The answer is 4.");
    }

    /// A harmony tool channel is neither reasoning nor prose: it is
    /// handed on verbatim, markers included, for the tool parser.
    #[test]
    fn a_harmony_tool_channel_survives_verbatim() {
        let parser = ReasoningParser::new(ReasoningFormat::GptOss, false, true);
        let split = parser.parse_complete(
            "<|channel|>analysis<|message|>need the weather<|end|>\
             <|channel|>commentary to=functions.get_weather<|message|>{\"city\":\"Rome\"}<|call|>",
        );
        assert_eq!(split.reasoning, "need the weather");
        assert!(split.content.contains("to=functions.get_weather"));
        assert!(split.content.contains("{\"city\":\"Rome\"}"));
        assert!(split.content.ends_with("<|call|>"));
    }

    #[test]
    fn harmony_text_with_no_channels_passes_through() {
        let parser = ReasoningParser::new(ReasoningFormat::GptOss, false, true);
        let split = parser.parse_complete("plain answer");
        assert_eq!(split.content, "plain answer");
        assert_eq!(split.reasoning, "");
    }

    #[test]
    fn harmony_streams_incrementally_and_matches_one_shot() {
        let text = "<|channel|>analysis<|message|>step one<|end|><|channel|>final<|message|>done<|return|>";
        let reference =
            ReasoningParser::new(ReasoningFormat::GptOss, false, true).parse_complete(text);
        let mut parser = ReasoningParser::new(ReasoningFormat::GptOss, false, true);
        let chunks: Vec<String> = text
            .as_bytes()
            .chunks(5)
            .map(|c| String::from_utf8_lossy(c).into_owned())
            .collect();
        let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let out = stream(&mut parser, &refs);
        assert_eq!(out.reasoning.trim(), reference.reasoning);
        assert_eq!(out.content.trim(), reference.content);
    }

    /// MiniMax-M3's adaptive mode: a leading bare closer means the
    /// model chose not to think this turn.
    #[test]
    fn a_leading_bare_closer_means_no_thinking_this_turn() {
        let parser = ReasoningParser::new(ReasoningFormat::MiniMaxM3, false, true);
        let split = parser.parse_complete("</mm:think>Straight to the answer.");
        assert_eq!(split.reasoning, "");
        assert_eq!(split.content, "Straight to the answer.");
    }

    /// ... and it is only special at the head. A closer later in the
    /// stream is ordinary.
    #[test]
    fn a_later_closer_is_not_the_adaptive_signal() {
        let parser = ReasoningParser::new(ReasoningFormat::MiniMaxM3, false, true);
        let split = parser.parse_complete("<mm:think>weighing</mm:think>done");
        assert_eq!(split.reasoning, "weighing");
        assert_eq!(split.content, "done");
    }

    #[test]
    fn the_adaptive_head_is_replayed_when_it_is_not_a_closer() {
        let mut parser = ReasoningParser::new(ReasoningFormat::MiniMaxM3, false, true);
        let out = stream(&mut parser, &["</m", "m:th", "ought> hmm"]);
        assert!(
            out.content.starts_with("</mm:thought>"),
            "the held head came back verbatim: {out:?}"
        );
    }

    #[test]
    fn parser_names_round_trip_and_specific_families_win_inference() {
        assert_eq!(
            ReasoningFormat::parse("deepseekv32"),
            Some(ReasoningFormat::DeepSeekV32)
        );
        assert_eq!(ReasoningFormat::parse("nonsense"), None);
        assert_eq!(
            ReasoningFormat::infer("MiniMax-M3-Instruct"),
            Some(ReasoningFormat::MiniMaxM3),
            "the specific arm must beat the bare minimax arm"
        );
        assert_eq!(
            ReasoningFormat::infer("MiniMax-M2"),
            Some(ReasoningFormat::ThinkAlwaysOpen)
        );
        assert_eq!(
            ReasoningFormat::infer("DeepSeek-V4-Flash"),
            Some(ReasoningFormat::DeepSeekV32)
        );
        assert_eq!(ReasoningFormat::infer("llama-3.1-8b"), None);
    }

    #[test]
    fn gemma_channel_markers_split_the_same_way() {
        let parser = ReasoningParser::new(ReasoningFormat::Gemma4, false, true);
        let split = parser.parse_complete("<|channel>thought\nmulling<channel|>the answer");
        assert_eq!(split.reasoning, "mulling");
        assert_eq!(split.content, "the answer");
    }
}
