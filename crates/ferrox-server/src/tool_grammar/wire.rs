//! One root rule per wire format, derived from the description the
//! PARSER reads that format with.
//!
//! # Why this is not a table of eleven framings
//!
//! `policy::parser::tool_call` already knows how every family spells a
//! call, as [`Markers`]: the block markers, the invoke and parameter
//! tags, how a value is trimmed. A second table here -- one to read a
//! framing and one to write it -- is this repo's dominant bug shape
//! spelled out in full, and it would decay the usual way: a marker
//! corrected on the reading side, a forced call still emitted in the old
//! spelling, and a 200 whose tool call this server cannot read back.
//!
//! So there is one description. [`shape`] says what KIND of root rule a
//! format needs, and everything else -- every literal in the grammar --
//! comes from `format.markers()`. The formats whose framing cannot be
//! written as a root rule are refused BY NAME with the reason, because a
//! forced call served with a 200 that does not parse is worse than the
//! 501: the caller stops checking.
//!
//! # What each shape is
//!
//! | Shape | Formats | Root |
//! |---|---|---|
//! | [`Shape::Json`] | hermes/qwen2.5, llama3, mistral | a marker, a JSON object naming the tool, a closing marker |
//! | [`Shape::Elements`] | qwen3_coder, glm47, minimax, deepseekv32 | an invoke element holding one element per argument |
//! | [`Shape::Harmony`] | gpt_oss | a channel header addressed to `functions.<name>`, then JSON |
//!
//! # What the element shape does about types
//!
//! An XML-ish family writes every value as TEXT, and this server decides
//! what it MEANS from the tool's own schema
//! (`ToolCallParser::convert_value`). The grammar has to agree with that
//! decision or the forced call arrives with arguments the schema does
//! not describe, so the value rule is chosen from the same declared
//! `type`:
//!
//! * `string` -- free text, everything up to the closing tag
//!   ([`super::exclude`]), or the `enum` members verbatim when the schema
//!   lists them. Not JSON: a `"…"`-quoted string reaches
//!   `convert_declared` as a string WITH its quotes.
//! * everything else -- the property's own JSON, through the same
//!   converter the JSON families use, because `convert_declared` parses
//!   an `integer` / `number` / `boolean` / `object` / `array` / `null`
//!   value as JSON.
//! * a property with no declared `type`, or one this server does not map
//!   -- refused, naming the property. `parse_loose` would guess, and a
//!   guess is the thing a forced call may not be built on.

use serde_json::Value;

use ferrox_models::grammar::json_schema::GrammarBuilder;
use ferrox_models::grammar::LazyTriggers;

use super::exclude::text_excluding;
use super::{escape, internal, invalid, schema_refused, unsupported, ToolSpec};
use crate::policy::parser::tool_call::{harmony, Markers, NameStyle, TagGrammar};
use crate::policy::parser::ToolCallFormat;
use crate::ApiError;

/// What kind of root rule a format's calls need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// A marker, a JSON object naming the tool, and whatever closes it.
    /// `array` is Mistral's one-element list around the object.
    Json { array: bool },
    /// An element grammar: an invoke tag naming the call, one parameter
    /// tag per argument. Entirely described by [`Markers`].
    Elements,
    /// gpt-oss's harmony channel addressed to a function.
    Harmony,
}

/// The shape a format's root rule takes, or the refusal naming it.
///
/// Exhaustive on purpose: a twelfth wire format must decide what a
/// forced call in it looks like, or say why it cannot, before this
/// compiles.
fn shape(format: ToolCallFormat) -> Result<Shape, ApiError> {
    match format {
        ToolCallFormat::Qwen25 | ToolCallFormat::Llama3 => Ok(Shape::Json { array: false }),
        ToolCallFormat::Mistral => Ok(Shape::Json { array: true }),
        ToolCallFormat::Qwen3Coder
        | ToolCallFormat::Glm47
        | ToolCallFormat::MiniMax
        | ToolCallFormat::DeepSeekV32 => Ok(Shape::Elements),
        ToolCallFormat::GptOss => Ok(Shape::Harmony),
        // The three that stay refused, each for a reason about the
        // format rather than about effort.
        ToolCallFormat::Gemma4 => Err(refused(
            format,
            "a gemma4 call's arguments are a comma-separated list in gemma's own quoting rather \
             than a JSON object, so which of them are required cannot be expressed by the object \
             rule every other format here shares; writing a second one beside the JSON Schema \
             converter is the drift this refusal exists to avoid",
        )),
        ToolCallFormat::MiniMaxM3 => Err(refused(
            format,
            "a minimax_m3 call names each argument with an ELEMENT of its own, and what a \
             repeated element means -- an array rather than a value -- depends on siblings that \
             have not been written yet, so no root rule can force a call whose arguments this \
             server would read back the way the schema declares them",
        )),
        ToolCallFormat::MuseGlimmer => Err(refused(
            format,
            "a muse_glimmer call's boundary is not syntactic: the same <atem:function_calls> \
             block is a call inside a channel addressed to a tool and prose inside one addressed \
             to the user, so a grammar over the block alone would force text this server reads \
             back as content",
        )),
    }
}

/// Build the body of `root` for `format`, and the trigger that switches
/// the lazy grammar on.
///
/// Every rule this adds to `builder` is added in dependency order, and
/// the exclusion automata come FIRST: their states reference each other
/// by name, so they must be added while nothing but the builtins is
/// bound. See [`text_excluding`].
pub(super) fn build_root(
    builder: &mut GrammarBuilder,
    format: ToolCallFormat,
    tools: &[ToolSpec<'_>],
) -> Result<(String, LazyTriggers), ApiError> {
    match shape(format)? {
        Shape::Json { array } => json_root(builder, format, tools, array),
        Shape::Elements => elements_root(builder, format, tools),
        Shape::Harmony => harmony_root(builder, tools),
    }
}

// ---- the JSON-payload families ----

/// `OPEN {"name": …, "arguments": …} CLOSE`, the shape whose payload is
/// a JSON object.
fn json_root(
    builder: &mut GrammarBuilder,
    format: ToolCallFormat,
    tools: &[ToolSpec<'_>],
    array: bool,
) -> Result<(String, LazyTriggers), ApiError> {
    let markers = format.markers();
    let mut alternatives = Vec::with_capacity(tools.len());
    for tool in tools {
        let args = builder
            .add_schema_value(&format!("tool-{}-args", tool.name), parameters(tool))
            .map_err(|e| schema_refused(tool.name, &e))?;
        let body = format!(
            r#""{{" space "\"name\"" space ":" space "\"{name}\"" space "," space "\"arguments\"" space ":" space {args} space "}}""#,
            name = tool.name,
        );
        alternatives.push(builder.add_rule(&format!("tool-{}-call", tool.name), &body));
    }

    let call = builder.add_rule("tool-call", &alternatives.join(" | "));
    let payload = if array {
        builder.add_rule("tool-call-list", &format!(r#""[" space {call} space "]""#))
    } else {
        call
    };
    Ok((
        block(
            markers.open,
            &format!("space {payload} space"),
            markers.close,
        ),
        trigger(markers.open)?,
    ))
}

// ---- the element families ----

/// `OPEN <invoke name> <param>value</param> … </invoke> CLOSE`.
fn elements_root(
    builder: &mut GrammarBuilder,
    format: ToolCallFormat,
    tools: &[ToolSpec<'_>],
) -> Result<(String, LazyTriggers), ApiError> {
    // No `..`: a new field of `Markers` must be looked at here before
    // this compiles again, which is the only thing that keeps a reader
    // and a writer of one framing honest.
    let Markers {
        open,
        close,
        invoke,
        param,
        // Both of these are about a value this grammar does not write.
        // `trim_newlines` strips whitespace from around a value, and the
        // only values here that may carry any are the JSON ones, which
        // `convert_declared` trims before it parses whatever this leaves.
        // `undeclared` decides what an argument the schema never
        // mentioned is worth, and this grammar can only write arguments
        // the schema declares.
        trim_newlines: _,
        undeclared: _,
    } = format.markers();

    let Some(param) = param else {
        return Err(internal(format!(
            "{} was given the element shape but its framing declares no parameter tag",
            format.as_str()
        )));
    };
    // Every value ends at this tag, so it is also the one string a value
    // may not contain.
    let text = text_excluding(builder, "arg-text", param.close)?;

    let mut alternatives = Vec::with_capacity(tools.len());
    for tool in tools {
        let mut body = invoke_open(invoke, tool.name)?;
        for arg in element_args(builder, param, tool, &text)? {
            body.push_str(" space ");
            body.push_str(&arg);
        }
        if let Some(tag) = invoke {
            body.push_str(&format!(r#" space "{}""#, escape(tag.close)));
        }
        alternatives.push(builder.add_rule(&format!("tool-{}-call", tool.name), &body));
    }

    let call = builder.add_rule("tool-call", &alternatives.join(" | "));
    // A format that names its call in a TAG may have whitespace before
    // it. One that names it in bare text (GLM) may not: its name ends at
    // the first newline, so a newline before it is an empty name.
    let lead = if invoke.is_some() { "space " } else { "" };
    Ok((
        block(open, &format!("{lead}{call} space"), close),
        trigger(open)?,
    ))
}

/// The rules for one tool's arguments, in the order a call writes them:
/// the required ones, then each optional one on its own.
fn element_args(
    builder: &mut GrammarBuilder,
    param: TagGrammar,
    tool: &ToolSpec<'_>,
    text: &str,
) -> Result<Vec<String>, ApiError> {
    let schema = parameters(tool);
    let Some(object) = schema.as_object() else {
        return Err(object_expected(tool.name));
    };
    match object.get("type").and_then(Value::as_str) {
        Some("object") | None => {}
        Some(_) => return Err(object_expected(tool.name)),
    }
    let properties = match object.get("properties") {
        None => return Ok(Vec::new()),
        Some(Value::Object(map)) => map,
        Some(_) => return Err(object_expected(tool.name)),
    };
    let required: Vec<&str> = object
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let mut args = Vec::new();
    for key in required.iter().copied() {
        let Some(property) = properties.get(key) else {
            return Err(invalid(
                format!(
                    "tool {:?} cannot be forced: it requires the argument {key:?}, which its \
                     \"parameters\" schema does not declare",
                    tool.name
                ),
                "tools",
            ));
        };
        args.push(param_rule(builder, param, tool, key, property, text)?);
    }
    for (key, property) in properties {
        if required.contains(&key.as_str()) {
            continue;
        }
        let rule = param_rule(builder, param, tool, key, property, text)?;
        args.push(format!("{rule}?"));
    }
    Ok(args)
}

/// One argument: the parameter tag, its value, and the tag that closes
/// it -- all four spellings taken from [`TagGrammar`].
fn param_rule(
    builder: &mut GrammarBuilder,
    param: TagGrammar,
    tool: &ToolSpec<'_>,
    key: &str,
    property: &Value,
    text: &str,
) -> Result<String, ApiError> {
    check_key(tool.name, key)?;
    // A JSON value may be surrounded by whitespace -- the template's own
    // layout puts a newline there, and `convert_declared` trims before it
    // parses. A text or `enum` value may NOT: it reaches the tool
    // verbatim, so whitespace around it would be part of it.
    let value = match value_shape(tool.name, key, property, param.close)? {
        ValueShape::Text => text.to_string(),
        ValueShape::Literals(body) => {
            builder.add_rule(&format!("tool-{}-enum-{key}", tool.name), &body)
        }
        ValueShape::Json => format!(
            "space {} space",
            builder
                .add_schema_value(&format!("tool-{}-arg-{key}", tool.name), property)
                .map_err(|e| schema_refused(tool.name, &e))?
        ),
    };
    let head = match param.name {
        NameStyle::Bare => format!(r#""{}{}>""#, escape(param.open), escape(key)),
        NameStyle::Attribute => format!(r#""{} name=\"{}\">""#, escape(param.open), escape(key)),
        NameStyle::Paired {
            key_close,
            value_open,
        } => format!(
            r#""{}{}{}{}""#,
            escape(param.open),
            escape(key),
            escape(key_close),
            escape(value_open)
        ),
    };
    let body = format!(r#"{head} {value} "{}""#, escape(param.close));
    Ok(builder.add_rule(&format!("tool-{}-param-{key}", tool.name), &body))
}

/// How one argument's value must be written so that
/// `ToolCallParser::convert_value` reads it back as the schema declares
/// it.
enum ValueShape {
    /// Free text up to the closing tag.
    Text,
    /// The `enum` members, written as they are.
    Literals(String),
    /// The property's own JSON.
    Json,
}

/// Keywords that constrain nothing, so a value rule may ignore them.
/// The same list `json_schema` ignores.
const ANNOTATIONS: [&str; 10] = [
    "title",
    "description",
    "default",
    "examples",
    "$schema",
    "$id",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
];

fn value_shape(
    tool: &str,
    key: &str,
    property: &Value,
    param_close: &str,
) -> Result<ValueShape, ApiError> {
    let Some(object) = property.as_object() else {
        return Err(untyped(tool, key, "it is not a schema object"));
    };
    let declared = object.get("type").and_then(Value::as_str);
    let Some(declared) = declared else {
        return Err(untyped(
            tool,
            key,
            "it declares no \"type\", and this server would have to GUESS whether the text the \
             model writes there is a string, a number or JSON",
        ));
    };
    if declared != "string" {
        // Every other declared type reaches `convert_declared` as JSON,
        // which is exactly what the schema converter emits.
        return Ok(ValueShape::Json);
    }

    // A declared string is handed to the tool verbatim, so its value is
    // TEXT and the schema converter -- which would quote it -- is the
    // wrong instrument.
    if let Some(members) = object.get("enum").or_else(|| object.get("const")) {
        let members = match members {
            Value::Array(members) => members.clone(),
            single => vec![single.clone()],
        };
        if members.is_empty() {
            return Err(untyped(tool, key, "its \"enum\" lists no members"));
        }
        let mut alternatives = Vec::with_capacity(members.len());
        for member in &members {
            let Some(member) = member.as_str() else {
                return Err(untyped(
                    tool,
                    key,
                    "it is a string whose \"enum\" holds a member that is not a string",
                ));
            };
            if member.contains(param_close) {
                return Err(untyped(
                    tool,
                    key,
                    "one of its \"enum\" members contains the tag that ends an argument, so \
                     writing it would end the argument early",
                ));
            }
            alternatives.push(format!("\"{}\"", escape(member)));
        }
        return Ok(ValueShape::Literals(alternatives.join(" | ")));
    }

    for keyword in object.keys() {
        if keyword == "type" || ANNOTATIONS.contains(&keyword.as_str()) {
            continue;
        }
        return Err(untyped(
            tool,
            key,
            &format!(
                "it is a string carrying {keyword:?}, which this server cannot honour in a value \
                 that is written as bare text rather than as JSON"
            ),
        ));
    }
    Ok(ValueShape::Text)
}

/// The invoke element that names the call.
fn invoke_open(invoke: Option<TagGrammar>, name: &str) -> Result<String, ApiError> {
    match invoke {
        Some(tag) => match tag.name {
            NameStyle::Bare => Ok(format!(r#""{}{}>""#, escape(tag.open), escape(name))),
            NameStyle::Attribute => Ok(format!(
                r#""{} name=\"{}\">""#,
                escape(tag.open),
                escape(name)
            )),
            NameStyle::Paired { .. } => Err(internal(format!(
                "the invoke tag {:?} is named the way a parameter is, which has no reader",
                tag.open
            ))),
        },
        // GLM has no invoke tag: the name is bare text right after the
        // block opener, and it ends at the first newline.
        None => Ok(format!(r#""{}\n""#, escape(name))),
    }
}

// ---- gpt-oss ----

/// `<|channel|>commentary to=functions.NAME<|message|>{…}<|call|>`.
///
/// Every literal comes from [`harmony`], which `parse_harmony` reads the
/// same call with.
fn harmony_root(
    builder: &mut GrammarBuilder,
    tools: &[ToolSpec<'_>],
) -> Result<(String, LazyTriggers), ApiError> {
    // The `<|constrain|>json` hint the harmony spec allows between the
    // recipient and the message. Optional: the header is read by
    // splitting on whitespace, so it changes nothing about the call.
    let constrain = builder.add_rule(
        "harmony-constrain",
        &format!(r#"| " {}json""#, escape(harmony::CONSTRAIN)),
    );
    let channel = builder.add_rule(
        "harmony-channel",
        &harmony::CHANNELS
            .iter()
            .map(|name| format!("\"{}\"", escape(name)))
            .collect::<Vec<_>>()
            .join(" | "),
    );

    let mut alternatives = Vec::with_capacity(tools.len());
    for tool in tools {
        let schema = parameters(tool);
        match schema.get("type").and_then(Value::as_str) {
            Some("object") | None => {}
            // A harmony message body that is not a JSON object is read
            // back as `{}` (`normalize_arguments`), so forcing one would
            // serve a call with its arguments thrown away.
            Some(_) => return Err(object_expected(tool.name)),
        }
        let args = builder
            .add_schema_value(&format!("tool-{}-args", tool.name), schema)
            .map_err(|e| schema_refused(tool.name, &e))?;
        let body = format!(
            r#""{name}" {constrain} "{message}" {args} "{call}""#,
            name = escape(tool.name),
            message = escape(harmony::MESSAGE_OPEN),
            call = escape(harmony::CALL_CLOSE),
        );
        alternatives.push(builder.add_rule(&format!("tool-{}-call", tool.name), &body));
    }
    let call = builder.add_rule("tool-call", &alternatives.join(" | "));

    let root = format!(
        r#""{open}" {channel} " {key}{namespace}" {call}"#,
        open = escape(harmony::CHANNEL_OPEN),
        key = escape(harmony::RECIPIENT_KEY),
        namespace = escape(harmony::FUNCTION_NAMESPACE),
    );

    // One trigger per channel a call may be written on, each ending at
    // `to=` rather than at the whole recipient: the name that follows is
    // then still ahead of the sampler, and so still constrained. A
    // trigger on `<|channel|>` alone would fire on the reasoning channel
    // and force a call inside the model's own thinking.
    let mut triggers = LazyTriggers::new().mandatory();
    for name in harmony::CHANNELS {
        triggers = triggers
            .with_word(&format!(
                "{}{name} {}",
                harmony::CHANNEL_OPEN,
                harmony::RECIPIENT_KEY
            ))
            .map_err(|e| internal(format!("tool-call trigger does not compile: {e}")))?;
    }
    Ok((root, triggers))
}

// ---- shared ----

/// `OPEN body CLOSE`, with the close omitted for the formats that end at
/// end-of-text.
fn block(open: &str, body: &str, close: &str) -> String {
    let mut root = format!(r#""{}" {body}"#, escape(open));
    if !close.is_empty() {
        root.push_str(&format!(r#" "{}""#, escape(close)));
    }
    root
}

/// The lazy trigger for a format whose calls open with one marker.
fn trigger(open: &str) -> Result<LazyTriggers, ApiError> {
    LazyTriggers::new()
        .with_word(open)
        .map_err(|e| internal(format!("tool-call trigger does not compile: {e}")))
        .map(LazyTriggers::mandatory)
}

/// The tool's `parameters`, or the schema of a call that takes none.
fn parameters<'a>(tool: &ToolSpec<'a>) -> &'a Value {
    static EMPTY: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    tool.parameters.unwrap_or_else(|| {
        EMPTY.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
            })
        })
    })
}

/// Every character of an argument's name reaches the grammar as a
/// literal, and several formats end the name at a `>` or a `"`. Held to
/// the same rule as a tool name rather than escaped into something the
/// checkpoint was never trained to write.
fn check_key(tool: &str, key: &str) -> Result<(), ApiError> {
    let ok = !key.is_empty()
        && key.len() <= 64
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.');
    if ok {
        return Ok(());
    }
    Err(invalid(
        format!(
            "tool {tool:?} cannot be forced: its argument {key:?} is written into the wire format \
             as a bare name, and this server accepts only names of 1..=64 characters from \
             [A-Za-z0-9_.-] there"
        ),
        "tools",
    ))
}

fn object_expected(tool: &str) -> ApiError {
    invalid(
        format!(
            "tool {tool:?} cannot be forced: this checkpoint's wire format writes a call's \
             arguments as named members, so its \"parameters\" must be an object schema"
        ),
        "tools",
    )
}

fn untyped(tool: &str, key: &str, why: &str) -> ApiError {
    invalid(
        format!(
            "tool {tool:?} cannot be forced: its argument {key:?} cannot be given a value \
                 rule, because {why}"
        ),
        "tools",
    )
}

fn refused(format: ToolCallFormat, why: &str) -> ApiError {
    unsupported(format!(
        "tool_choice cannot be enforced for a {} checkpoint: {why}. Use tool_choice \"auto\", \
         which asks for a call in the prompt instead of forcing one.",
        format.as_str()
    ))
}
