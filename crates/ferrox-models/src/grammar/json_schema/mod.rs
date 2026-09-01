//! JSON Schema to GBNF, a port of llama.cpp's
//! `common/json-schema-to-grammar.cpp` (MIT; see
//! `docs/THIRD_PARTY_NOTICES.md`).
//!
//! This is what `response_format: {"type": "json_schema"}` needs. The
//! output is GBNF text that [`crate::grammar::parse`] accepts, so it feeds
//! the same pushdown machine as a hand-written grammar -- and every
//! conversion re-parses its own output before returning it, because a
//! grammar that does not parse is a bug in this module and should not
//! reach a caller as a string.
//!
//! # What is ported
//!
//! | Keyword | Behaviour |
//! |---|---|
//! | `type` | `object` `array` `string` `number` `integer` `boolean` `null`, and an array of them as a union |
//! | `properties`, `required` | in declaration order, required first |
//! | `additionalProperties` | `false`/absent closes the object, a schema types the tail, `true` opens it |
//! | `items`, `prefixItems` | a single schema is a list, an array is a fixed tuple |
//! | `minItems`, `maxItems` | beside a single-schema `items` |
//! | `minLength`, `maxLength` | beside an explicit `"type": "string"` |
//! | `enum`, `const` | literal alternatives, JSON-encoded |
//! | `oneOf`, `anyOf` | union of alternatives |
//! | `$ref`, `$defs`, `definitions` | same-document `#/` pointers, recursion included |
//! | `format` | `date`, `time`, `date-time`, `uuid`, `uuid1`..`uuid5` |
//!
//! Annotations (`title`, `description`, `default`, `examples`, `$schema`,
//! `$id`, `$comment`, `deprecated`, `readOnly`, `writeOnly`) are ignored,
//! which is safe: they constrain nothing.
//!
//! # What is refused, by name
//!
//! Everything else, via [`SchemaError::UnsupportedKeyword`]. The rule is
//! in [`branch`]: a keyword the chosen branch did not act on is refused
//! unless the schema's declared `type` makes it vacuous -- `items` beside
//! `"type": "null"` says nothing about a null, so it is dropped, while
//! `pattern` beside `"type": "string"` is refused.
//!
//! The notable refusals are `pattern` (llama.cpp's regex-to-GBNF compiler
//! is not ported), `allOf` (schema intersection), the numeric bounds
//! `minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum`
//! (llama.cpp builds a digit-by-digit range grammar for integers), `not` /
//! `if` / `then` / `else`, `patternProperties`, `propertyNames`,
//! `uniqueItems`, `minProperties` / `maxProperties`, `multipleOf`, the
//! `dependent*` family, and any `format` outside the six above.
//!
//! **This is stricter than llama.cpp on purpose.** Upstream's `visit` is a
//! chain of `if`s: a keyword no branch tested falls off the end and is
//! discarded, so `{"type": "string", "pattern": "^[a-z]+$"}` compiles to a
//! grammar for *any* string. That grammar accepts documents the caller
//! declared invalid, which is a wrong answer, not a missing feature. Per
//! `CLAUDE.md`, a refusal is coverage.
//!
//! # Where this differs from llama.cpp on schemas it accepts
//!
//! - `_not_strings` writes property-name bytes into a GBNF character class
//!   unescaped, so upstream emits an unparseable grammar for a property
//!   named `a-b`, `a]b` or `añb`. This port escapes the class and keys the
//!   trie on `char` rather than `u8`.
//! - JSON Pointer escapes (`~1`, `~0`) in a `$ref` are decoded. Upstream
//!   splits on `/` and cannot address such a member at all.
//! - Remote (`https://`) `$ref`s are refused rather than fetched.
//! - A top-level schema whose rule ends up named something other than
//!   `root` is refused ([`SchemaError::RootDisplaced`]) instead of silently
//!   producing a grammar that starts from a subschema; upstream reaches
//!   this whenever a property is named `""`.
//!
//! # Property order
//!
//! The order of `properties` is part of the accepted language: required
//! members are emitted in declaration order. [`json_schema_to_grammar`]
//! parses the schema text with [`value::JsonValue`], which keeps document
//! order the way llama.cpp's `nlohmann::ordered_json` does.
//! [`json_schema_to_grammar_value`] can only carry the order its
//! `serde_json::Value` already has, and this workspace builds `serde_json`
//! without `preserve_order`, so that entry point orders properties
//! lexicographically. Prefer the text entry point when the caller has the
//! raw JSON.

mod branch;
mod converter;
mod error;
mod primitives;
mod refs;
mod value;

#[cfg(test)]
mod tests;

pub use error::SchemaError;
pub use value::JsonValue;

use converter::Converter;

/// Convert JSON Schema text to a GBNF grammar.
///
/// This is the entry point to prefer: it keeps `properties` in the order
/// the schema declares them.
pub fn json_schema_to_grammar(schema_text: &str) -> Result<String, SchemaError> {
    let schema = JsonValue::parse(schema_text).map_err(|e| SchemaError::NotJson(e.to_string()))?;
    convert(&schema)
}

/// Convert an already-parsed `serde_json::Value` schema.
///
/// See the module docs on property order: without `serde_json`'s
/// `preserve_order` feature a `Value` has already lost the schema's
/// declaration order, and required members will be required in
/// lexicographic order instead.
pub fn json_schema_to_grammar_value(schema: &serde_json::Value) -> Result<String, SchemaError> {
    convert(&JsonValue::from(schema))
}

/// `json_schema_to_grammar` / `build_grammar`: resolve refs, visit the
/// root, emit, and check the result against this repo's own GBNF parser.
fn convert(schema: &JsonValue) -> Result<String, SchemaError> {
    let refs = refs::collect(schema)?;
    let mut converter = Converter::new(refs);
    let root = converter.visit(schema, "")?;
    if root != "root" {
        return Err(SchemaError::RootDisplaced { got: root });
    }
    let grammar = converter.finish();
    // A grammar this module cannot hand to its own parser is a defect
    // here, and it is cheaper to find it now than in a sampler.
    crate::grammar::parse(&grammar)?;
    Ok(grammar)
}
