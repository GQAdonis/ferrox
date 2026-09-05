//! "Every string that does not contain this literal", as GBNF.
//!
//! The XML-ish wire formats end an argument's value at a closing tag,
//! so the value rule is *everything up to* `</parameter>` (or whatever
//! that family calls it), and a grammar that FORCES a call has to be
//! able to say exactly that. `[^<]*` says something else: it forbids
//! every `<`, and a coding agent's arguments are whole files.
//!
//! This is the complement of the literal's KMP automaton, one
//! right-recursive rule per state -- llama.cpp's
//! `gbnf_excluding_grammar` (`common/peg-parser.cpp`, added by
//! ggml-org/llama.cpp#24839) for the single-pattern case, where its
//! Aho-Corasick automaton degenerates to KMP. Every state accepts, and
//! the transition that would COMPLETE the literal is the one
//! alternative that is never written, so the literal can never be
//! matched.
//!
//! Right recursion is deliberate and is not a stack leak: a rule
//! reference in final position is not pushed as a continuation
//! (`machine::advance_stack`, transcribed from
//! `llama_grammar_advance_stack`), so a value of any length costs one
//! stack entry.

use std::collections::{BTreeMap, BTreeSet};

use ferrox_models::grammar::json_schema::GrammarBuilder;

use crate::ApiError;

/// Emit the rules for text that cannot contain `forbidden`, and return
/// the name of the rule to reference.
///
/// `prefix` must be free in `builder`: these rules reference each other
/// by name, and `GrammarBuilder::add_rule` renames a name already bound
/// to a different body, which would silently point a state at somebody
/// else's rule. Call this BEFORE adding any schema, while nothing but
/// the builtins is bound. The check below is what makes a caller that
/// gets that order wrong a refusal rather than a wrong grammar.
pub(super) fn text_excluding(
    builder: &mut GrammarBuilder,
    prefix: &str,
    forbidden: &str,
) -> Result<String, ApiError> {
    let chars: Vec<char> = forbidden.chars().collect();
    if chars.is_empty() {
        // Every string contains the empty string, so the honest
        // language here is the empty one, which GBNF cannot spell. The
        // empty string is the only under-approximation, and it is safe:
        // it never lets through text the reader would not read back.
        // Every format `wire::shape` gives the element treatment to has
        // a non-empty closing tag, so this is a guard on the call rather
        // than a case a request can reach.
        return Ok(builder.add_rule(prefix, r#""""#));
    }

    let failure = kmp_failure(&chars);
    let alphabet: BTreeSet<char> = chars.iter().copied().collect();
    let name_of = |state: usize| {
        if state == 0 {
            prefix.to_string()
        } else {
            format!("{prefix}-{state}")
        }
    };

    for state in 0..chars.len() {
        // Chars whose transition leads somewhere other than the start
        // state, grouped by where; plus every char that has any
        // explicit transition at all, so the rest can be swept up by
        // one negated class.
        let mut buckets: BTreeMap<usize, Vec<char>> = BTreeMap::new();
        let mut specific: Vec<char> = Vec::new();
        for &c in &alphabet {
            let next = step(&chars, &failure, state, c);
            if next == chars.len() {
                // Completing the literal. Listed as "explicit" so the
                // catch-all cannot match it, and given no alternative
                // of its own: that is the whole exclusion.
                specific.push(c);
            } else if next != 0 {
                buckets.entry(next).or_default().push(c);
                specific.push(c);
            }
        }

        // The empty first alternative: every state of the complement
        // accepts, because a string that has not completed the literal
        // does not contain it.
        let mut alternatives = vec![String::new()];
        for (next, group) in &buckets {
            alternatives.push(format!("{} {}", char_class(group, false), name_of(*next)));
        }
        alternatives.push(format!("{} {}", char_class(&specific, true), name_of(0)));

        let name = name_of(state);
        let got = builder.add_rule(&name, &alternatives.join(" | "));
        if got != name {
            return Err(super::internal(format!(
                "tool-call grammar: the rule {name:?} that holds text excluding {forbidden:?} was \
                 renamed to {got:?}, so its own states would reference the wrong rule"
            )));
        }
    }
    Ok(name_of(0))
}

/// `failure[q]` is the length of the longest proper prefix of
/// `chars[..=q]` that is also a suffix of it.
fn kmp_failure(chars: &[char]) -> Vec<usize> {
    let mut failure = vec![0usize; chars.len()];
    let mut k = 0usize;
    for i in 1..chars.len() {
        while k > 0 && chars[i] != chars[k] {
            k = failure[k - 1];
        }
        if chars[i] == chars[k] {
            k += 1;
        }
        failure[i] = k;
    }
    failure
}

/// The state reached from `state` on `c`: how much of the literal is
/// matched by the longest suffix of the input read so far.
fn step(chars: &[char], failure: &[usize], state: usize, c: char) -> usize {
    let mut state = state;
    loop {
        if chars[state] == c {
            return state + 1;
        }
        if state == 0 {
            return 0;
        }
        state = failure[state - 1];
    }
}

/// A GBNF character class over `chars`, negated or not.
///
/// `-` is spelled `\x2D` rather than `\-` for the reason
/// `json_schema::primitives::escape_in_range` gives: llama.cpp's table
/// lists `\-` but its GBNF *parser* has no such escape, and this repo's
/// parser transcribes the parser.
fn char_class(chars: &[char], negated: bool) -> String {
    let mut out = String::from("[");
    if negated {
        out.push('^');
    }
    for &c in chars {
        match c {
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '"' => out.push_str("\\\""),
            '-' => out.push_str("\\x2D"),
            ']' => out.push_str("\\]"),
            '[' => out.push_str("\\["),
            '\\' => out.push_str("\\\\"),
            '^' => out.push_str("\\x5E"),
            c => out.push(c),
        }
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_models::grammar::Grammar;

    /// Compile `root ::= <text excluding forbidden> "END"` and report
    /// whether `text` parses -- the shape the value rules use, so the
    /// exclusion is tested where it has to be exact: right before the
    /// literal it excludes.
    fn accepts(forbidden: &str, text: &str) -> bool {
        let mut builder = GrammarBuilder::new();
        let body = text_excluding(&mut builder, "not", forbidden).expect("a rule");
        builder.add_rule("root", &format!("{body} \"END\""));
        let grammar = Grammar::from_str_with_root(&builder.finish().expect("grammar"), "root")
            .expect("compiles");
        let mut g = grammar.clone();
        let whole = format!("{text}END");
        if g.accept_token(0, whole.as_bytes()).is_err() {
            return false;
        }
        g.allows_eog()
    }

    /// The headline: a value may hold anything at all except the tag
    /// that ends it.
    #[test]
    fn only_the_forbidden_literal_is_refused() {
        assert!(accepts("</parameter>", "plain text"));
        assert!(accepts("</parameter>", "<html><body>a < b</body></html>"));
        assert!(accepts(
            "</parameter>",
            "</param> </parameters> <parameter>"
        ));
        assert!(accepts("</parameter>", ""));
        assert!(!accepts("</parameter>", "before</parameter>after"));
        assert!(!accepts("</parameter>", "</parameter>"));
    }

    /// The exclusion has to survive a partial match that restarts, which
    /// is the whole reason it is an automaton and not an alternation of
    /// "a prefix then a mismatching character".
    #[test]
    fn a_restarted_partial_match_still_completes_the_literal() {
        // "aab" contains "ab" from index 1. The naive
        // `([^a] | "a" [^b])*` construction accepts it.
        assert!(!accepts("ab", "aab"));
        assert!(accepts("ab", "aa"));
        assert!(!accepts("aa", "baaa"));
        assert!(accepts("aa", "aba"));
        // A literal with a border, where the failure function matters.
        assert!(!accepts("aba", "xxababa"));
        assert!(accepts("aba", "xxabb"));
    }

    /// The multi-byte tags are the ones a byte-wise automaton would get
    /// wrong: DeepSeek spells its tags with U+FF5C.
    #[test]
    fn a_multi_byte_literal_is_excluded_by_codepoint() {
        assert!(accepts("</｜DSML｜parameter>", "値 with a ｜ in it"));
        assert!(accepts("</｜DSML｜parameter>", "</｜DSML｜invoke>"));
        assert!(!accepts("</｜DSML｜parameter>", "x</｜DSML｜parameter>y"));
    }

    /// Every character that has to be escaped to reach a GBNF class
    /// alive. A literal holding one of these used to be the way this
    /// emitted a grammar that does not parse.
    #[test]
    fn a_literal_of_class_metacharacters_still_compiles() {
        for forbidden in ["]-^", "[\\]", "\"a\"", "\n\t"] {
            assert!(
                accepts(forbidden, "harmless"),
                "{forbidden:?} should compile and accept text without it"
            );
            assert!(
                !accepts(forbidden, forbidden),
                "{forbidden:?} should exclude itself"
            );
        }
    }
}
