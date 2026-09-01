//! Where a request's stated constraint becomes a compiled grammar.
//!
//! Exactly one function answers "what grammar does this request run
//! under", because a request can state the constraint two ways and only
//! one of them may win:
//!
//! | Request field | Today |
//! |---|---|
//! | `"grammar": "<GBNF>"` | compiled and applied (llama.cpp's field, same spelling) |
//! | `response_format: {"type": "json_schema", ...}` | refused, naming what is missing |
//!
//! The second is the one callers actually reach for, and it is one
//! function away: a JSON schema becomes a GBNF grammar
//! (`common/json-schema-to-grammar.cpp` upstream), and everything after
//! that -- masking, accepting, both decode loops, the batcher -- is the
//! same path `"grammar"` already takes. That converter is a separate
//! piece of work and is NOT approximated here: emitting a grammar that
//! is *nearly* the caller's schema would be served with a 200 and read
//! as compliance.
//!
//! # Why the refusal is not a silent ignore
//!
//! `logit_bias` was declared on these requests only so it could be
//! refused by name, because serde dropped an undeclared field and the
//! caller got a 200 whose answer was sampled from unbiased logits. A
//! `json_schema` that is accepted and not applied is the same failure
//! with more expensive consequences: the caller parses the answer.

use std::sync::Arc;

use axum::http::StatusCode;
use axum::Json;
use ferrox_models::grammar::Grammar;

use crate::ApiError;

/// The root rule name every grammar here starts from. llama.cpp's
/// convention, and the one every published `.gbnf` uses.
const ROOT: &str = "root";

/// Compile GBNF text, or refuse with the parser's own diagnostic.
///
/// A 400: a grammar that does not parse is a fact about the request,
/// and the message carries the position the parser stopped at, which is
/// the only thing that makes one debuggable from the client side.
pub(crate) fn compile(src: &str) -> Result<Arc<Grammar>, ApiError> {
    Grammar::from_str_with_root(src, ROOT)
        .map(Arc::new)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("grammar: {e}"),
                        "type": "invalid_request_error",
                        "param": "grammar",
                    }
                })),
            )
        })
}

/// The grammar one request runs under, from every field that can state
/// one.
///
/// `response_format` is checked FIRST and refuses rather than falls
/// through: a request that asked for a schema and also supplied a
/// grammar has asked for two different constraints, and serving the one
/// we happen to be able to compile is not answering the question.
pub(crate) fn for_request(
    grammar: Option<&str>,
    response_format: Option<&serde_json::Value>,
) -> Result<Option<Arc<Grammar>>, ApiError> {
    if let Some(kind) = response_format
        .and_then(|v| v.get("type"))
        .and_then(|v| v.as_str())
    {
        if kind == "json_schema" {
            return Err(json_schema_not_implemented());
        }
    }
    match grammar.map(str::trim).filter(|s| !s.is_empty()) {
        Some(src) => compile(src).map(Some),
        None => Ok(None),
    }
}

/// The one place `response_format: json_schema` is refused, so the
/// converter that closes it has one call site to land in.
fn json_schema_not_implemented() -> ApiError {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message":
                    "response_format json_schema is not implemented: the grammar engine and the \
                     sampler hook are wired, but converting a JSON schema to a GBNF grammar is \
                     not. Send the equivalent grammar in the \"grammar\" field, or use \
                     response_format json_object for the best-effort character-class mode.",
                "type": "invalid_request_error",
                "param": "response_format",
            }
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gbnf_grammar_compiles_and_starts_at_root() {
        let g = for_request(Some(r#"root ::= "a"+"#), None)
            .expect("a valid grammar is not an error")
            .expect("a grammar was asked for");
        assert!(!g.stacks().is_empty(), "the machine has a live stack");
    }

    #[test]
    fn no_grammar_field_means_no_grammar() {
        assert!(for_request(None, None).unwrap().is_none());
        // Whitespace is not a grammar. Left as `None` rather than sent
        // to the parser, which would 400 on an empty string a client
        // library sent by defaulting the field.
        assert!(for_request(Some("   "), None).unwrap().is_none());
    }

    #[test]
    fn an_unparseable_grammar_is_a_400_naming_the_field() {
        // An unterminated string literal. Note that `root ::=` alone is
        // NOT this case: an empty alternate is legal GBNF, and it
        // compiles to a grammar that is satisfied before it starts.
        let (status, Json(body)) =
            for_request(Some(r#"root ::= "a"#), None).expect_err("this does not parse");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "grammar");
    }

    /// A grammar whose root rule is spelled something else is refused
    /// too, rather than starting from whichever rule came first.
    #[test]
    fn a_grammar_with_no_root_rule_is_refused() {
        let (status, _) =
            for_request(Some(r#"start ::= "a""#), None).expect_err("there is no root rule");
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// The seam. It must NOT quietly become "unconstrained", and it
    /// must not be answered by the `grammar` field either.
    #[test]
    fn json_schema_is_refused_by_name_rather_than_ignored() {
        let fmt = serde_json::json!({"type": "json_schema", "json_schema": {"schema": {}}});
        let (status, Json(body)) = for_request(None, Some(&fmt)).expect_err("not implemented yet");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("JSON schema"), "{message}");

        let (status, _) = for_request(Some(r#"root ::= "a""#), Some(&fmt))
            .expect_err("two constraints, one of which cannot be honoured");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    }

    /// `json_object` is a different feature (a character-class mask) and
    /// is handled elsewhere; it must not be caught by the schema arm.
    #[test]
    fn json_object_is_not_the_schema_arm() {
        let fmt = serde_json::json!({"type": "json_object"});
        assert!(for_request(None, Some(&fmt)).unwrap().is_none());
    }
}
