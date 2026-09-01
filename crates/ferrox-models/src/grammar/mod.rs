//! GBNF grammar-constrained decoding.
//!
//! A port of llama.cpp's `src/llama-grammar.cpp` and `src/llama-grammar.h`
//! (MIT; see `docs/THIRD_PARTY_NOTICES.md`). The algorithm is transcribed
//! from that source rather than reconstructed from how EBNF engines
//! usually work, because the two differ in places that matter -- the
//! negation of a character class lives on its first element only, the
//! repetition rewrites produce observable rule ids, and a token piece that
//! ends mid-codepoint has to stay viable rather than be rejected.
//!
//! What this replaces, and why it is not the same kind of thing:
//! `crates/ferrox-server/src/json_mode.rs` masks logits by a fixed
//! character class. It has no state, so it cannot know whether a `}`
//! closes an object that was opened. This is a pushdown machine over a
//! parsed grammar, so it can.
//!
//! # Layout
//!
//! | Module | Role |
//! |---|---|
//! | [`element`] | the compiled element / rule / cursor types |
//! | [`error`] | every refusal, naming what is missing |
//! | [`utf8`] | llama.cpp's two UTF-8 decoders, partial sequences included |
//! | [`parser`] | GBNF text to a rule table |
//! | [`machine`] | the pushdown stack machine over that table |
//! | [`candidates`] | which candidate tokens no viable stack accepts |
//! | [`json_schema`] | JSON Schema to GBNF text, for `response_format` |
//!
//! # Status
//!
//! Steps 1 and 2 of the three-step spine in
//! `docs/plans/llama-cpp-gap-inventory.md` §8 item 8: the parser and the
//! stack machine, with the candidate-rejection core the sampler hook would
//! sit on. **Not wired into sampling or the server.** The seam a caller
//! needs is [`machine::Grammar::accept_token`] after each accepted token
//! and [`candidates::reject_candidates`] before each sample; nothing calls
//! either yet.
//!
//! [`json_schema`] now ports `common/json-schema-to-grammar.cpp`, which is
//! what `response_format: json_schema` needs on top of a grammar. It is
//! stricter than upstream by design -- a keyword it cannot honour is a
//! typed refusal rather than a grammar that accepts too much -- and its
//! module docs list what is ported, what is refused by name, and where its
//! output differs from llama.cpp's. It is not wired into the server
//! either; `ferrox-server` still answers `response_format: json_schema`
//! with a 501.
//!
//! Still not ported, and a separate piece of work:
//!
//! - **Lazy grammars** (`trigger_patterns` / `trigger_tokens`), which is
//!   what `tool_choice: "required"` needs on top of a grammar.

pub mod candidates;
pub mod element;
pub mod error;
pub mod json_schema;
pub mod machine;
pub mod parser;
pub mod utf8;

#[cfg(test)]
mod grammars;
#[cfg(test)]
mod machine_tests;
#[cfg(test)]
mod parser_tests;

pub use candidates::{reject_candidates, Candidate, DecodedPiece};
pub use element::{GrammarElement, GrammarRule, GrammarStack, GreType, RulePos};
pub use error::GrammarError;
pub use json_schema::{json_schema_to_grammar, json_schema_to_grammar_value, SchemaError};
pub use machine::Grammar;
pub use parser::{parse, parse_with_vocab, GrammarVocab, ParsedGrammar};
pub use utf8::PartialUtf8;
