//! Incremental detokenization for streaming: turning one more token id
//! into the *text you can safely send now*.
//!
//! Two things make this harder than "decode the new token".
//!
//! A BPE token is a byte string, not a character: a multi-byte
//! codepoint (or an emoji) arrives across several tokens, and decoding
//! each one alone yields a replacement char. So a decode always
//! re-decodes a small trailing window of ids -- `surr_offset ..` -- and
//! keeps only what that window added beyond the part already accounted
//! for, which is what lets a codepoint materialize once its last byte
//! lands.
//!
//! And a stop string may straddle a token boundary. Streaming `<` when
//! `<|end|>` is a stop string means retracting it later, which SSE
//! cannot do -- so any trailing run that is still a proper prefix of a
//! stop string is withheld until a later token decides it.
//!
//! Ported 1:1 from FreeToken's
//! `python/freetoken/tokenizer/detokenize.py` (Apache-2.0), which in
//! turn borrows the printable-text heuristic from SGLang and
//! `transformers`' `TextStreamer`; see `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::{HashMap, HashSet};

/// Whether `cp` is a CJK codepoint.
///
/// This defines a "chinese character" as anything in the CJK Unicode
/// block. Note that the CJK block is *not* all Japanese and Korean
/// characters, despite its name: modern Hangul is a different block, as
/// are Hiragana and Katakana. Those alphabets write space-separated
/// words, so they need no special case -- they fall through to the
/// word-boundary heuristic like every other language.
fn is_chinese_char(cp: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

/// The longest prefix of `text` that contains only whole words.
///
/// A newline flushes everything. CJK text is not space-delimited, so a
/// trailing CJK char is printable as-is and a penultimate one makes
/// everything but the last char printable. Otherwise the text is cut at
/// the last space: the run after it is a partial word that the next
/// token may still change.
pub fn find_printable_text(text: &str) -> &str {
    if text.ends_with('\n') {
        return text;
    }
    let mut chars = text.chars().rev();
    match (chars.next(), chars.next()) {
        (Some(last), _) if is_chinese_char(last as u32) => text,
        (Some(last), Some(penultimate)) if is_chinese_char(penultimate as u32) => {
            &text[..text.len() - last.len_utf8()]
        }
        _ => match text.rfind(' ') {
            Some(idx) => &text[..idx + 1],
            None => "",
        },
    }
}

/// Length in bytes of the longest trailing run of `text` that is a
/// *proper* prefix of some stop string.
///
/// Those bytes are withheld until a later token resolves whether the
/// stop completes, so a partial stop is never streamed and then needs
/// retracting. A run equal to a whole stop string is not held back --
/// that is a match, and the caller ends the generation instead.
///
/// Byte comparison rather than character comparison is deliberate: the
/// answer is a split point, and callers pass it through
/// [`floor_char_boundary`], so a run that lands mid-character withholds
/// slightly more rather than slicing a `str` invalidly.
///
/// This is the single implementation of the rule in the workspace --
/// `ferrox-server`'s `stop::StopMatcher`, which owns the same promise
/// for the non-batched decode path, delegates here rather than keeping
/// its own copy.
pub fn stop_prefix_holdback(text: &str, stop_strs: &[String]) -> usize {
    let bytes = text.as_bytes();
    let longest = stop_strs
        .iter()
        .map(|s| s.len().saturating_sub(1))
        .max()
        .unwrap_or(0);
    let max_k = longest.min(bytes.len());
    (1..=max_k)
        .rev()
        .find(|&k| {
            let tail = &bytes[bytes.len() - k..];
            stop_strs
                .iter()
                .any(|s| s.len() > k && s.as_bytes().starts_with(tail))
        })
        .unwrap_or(0)
}

/// The largest index `<= at` that is a char boundary of `text`.
///
/// `str::floor_char_boundary` is still unstable, and every split point
/// derived from a byte-length rule above needs one.
pub fn floor_char_boundary(text: &str, at: usize) -> usize {
    let mut idx = at.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// One sequence's decode state.
#[derive(Debug, Clone, Default)]
pub struct DecodeStatus {
    pub decoded_ids: Vec<u32>,
    pub decoded_str: String,
    /// Number of ids whose text is already committed.
    pub read_offset: usize,
    /// Start of the re-decoded trailing window.
    pub surr_offset: usize,
    /// Bytes of `decoded_str` already streamed to the client.
    pub sent_offset: usize,
}

/// One decode step for one sequence.
#[derive(Debug, Clone)]
pub struct DetokenizeMsg {
    pub uid: u64,
    pub next_token: u32,
    pub finished: bool,
    /// The stop string that ended this generation, when one did. The
    /// final chunk is trimmed at it, so the stop text itself is never
    /// part of the response.
    pub matched_stop: Option<String>,
    /// Stop strings still in play, for the hold-back above.
    pub stop_strs: Vec<String>,
}

/// Anything that can turn a run of token ids back into text.
/// `ferrox-models`' tokenizers implement this; tests use a toy.
pub trait Detokenizer {
    fn decode(&self, ids: &[u32]) -> String;
}

/// Per-sequence incremental detokenization, keyed by request uid.
pub struct DetokenizeManager<D: Detokenizer> {
    decode_map: HashMap<u64, DecodeStatus>,
    tokenizer: D,
    eos_token_ids: HashSet<u32>,
}

impl<D: Detokenizer> DetokenizeManager<D> {
    pub fn new(tokenizer: D, eos_token_ids: HashSet<u32>) -> Self {
        DetokenizeManager {
            decode_map: HashMap::new(),
            tokenizer,
            eos_token_ids,
        }
    }

    /// Drop a uid's decode state without a finished `DetokenizeMsg`. An
    /// aborted or errored request never sends one -- its terminal reply
    /// is an error -- so without this its accumulated ids and text stay
    /// in the map for the life of the worker.
    pub fn discard(&mut self, uid: u64) {
        self.decode_map.remove(&uid);
    }

    pub fn is_tracking(&self, uid: u64) -> bool {
        self.decode_map.contains_key(&uid)
    }

    /// Fold one step per sequence into the text to stream for each.
    pub fn detokenize(&mut self, msgs: &[DetokenizeMsg]) -> Vec<String> {
        let mut windows: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(msgs.len());
        for msg in msgs {
            let state = self.decode_map.entry(msg.uid).or_default();
            // A terminal EOS is a control token, not output: it is
            // never appended, so it can never be decoded into the
            // response.
            if !(msg.finished && self.eos_token_ids.contains(&msg.next_token)) {
                state.decoded_ids.push(msg.next_token);
            }
            let read = state.decoded_ids[state.surr_offset..].to_vec();
            let surr = state.decoded_ids[state.surr_offset..state.read_offset].to_vec();
            windows.push((read, surr));
        }

        let mut out = Vec::with_capacity(msgs.len());
        for (msg, (read_ids, surr_ids)) in msgs.iter().zip(windows.iter()) {
            let read_str = self.tokenizer.decode(read_ids);
            let surr_str = self.tokenizer.decode(surr_ids);
            let state = self
                .decode_map
                .get_mut(&msg.uid)
                .expect("decode state was just inserted");

            // What this step added beyond the already-accounted-for
            // part of the same window.
            let mut new_text = read_str.get(surr_str.len()..).unwrap_or("").to_string();
            let output_str;
            if !new_text.is_empty() && !new_text.ends_with('\u{fffd}') {
                // A complete decode: commit it and slide the window.
                output_str = format!("{}{}", state.decoded_str, new_text);
                state.decoded_str.clone_from(&output_str);
                state.surr_offset = state.read_offset;
                state.read_offset = state.decoded_ids.len();
            } else {
                // A partial codepoint (or nothing): show only what is
                // printable and re-decode the same window next step.
                new_text = find_printable_text(&new_text).to_string();
                output_str = format!("{}{}", state.decoded_str, new_text);
            }

            let prev_sent = state.sent_offset;
            let emit_end = if msg.finished {
                // Generation is over: flush everything, trimming at the
                // matched stop string.
                match msg.matched_stop.as_deref() {
                    Some(stop) => output_str.find(stop).unwrap_or(output_str.len()),
                    None => output_str.len(),
                }
            } else if !msg.stop_strs.is_empty() {
                let keep = stop_prefix_holdback(&output_str, &msg.stop_strs);
                floor_char_boundary(&output_str, output_str.len() - keep)
            } else {
                output_str.len()
            };

            let incremental = if emit_end > prev_sent {
                output_str
                    .get(prev_sent..emit_end)
                    .unwrap_or("")
                    .to_string()
            } else {
                String::new()
            };
            state.sent_offset = prev_sent.max(emit_end);
            out.push(incremental);
            if msg.finished {
                self.decode_map.remove(&msg.uid);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy vocabulary: each id maps to a byte string, and decode is
    /// concatenation interpreted as UTF-8 -- exactly the failure mode
    /// (a codepoint split across ids) the manager exists to handle.
    struct Bytes(Vec<Vec<u8>>);

    impl Detokenizer for Bytes {
        fn decode(&self, ids: &[u32]) -> String {
            let bytes: Vec<u8> = ids
                .iter()
                .flat_map(|id| self.0[*id as usize].clone())
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }

    fn msg(uid: u64, token: u32) -> DetokenizeMsg {
        DetokenizeMsg {
            uid,
            next_token: token,
            finished: false,
            matched_stop: None,
            stop_strs: Vec::new(),
        }
    }

    #[test]
    fn printable_text_cuts_at_the_last_word_boundary() {
        assert_eq!(find_printable_text("hello wor"), "hello ");
        assert_eq!(find_printable_text("hello"), "");
        assert_eq!(find_printable_text("hello\n"), "hello\n");
        assert_eq!(find_printable_text(""), "");
    }

    /// CJK is not space-delimited, so waiting for a space would stall a
    /// Chinese response forever.
    #[test]
    fn printable_text_streams_cjk_immediately() {
        assert_eq!(find_printable_text("上海"), "上海");
        assert_eq!(find_printable_text("上海a"), "上海");
    }

    #[test]
    fn holdback_covers_only_proper_prefixes() {
        let stops = vec!["<|end|>".to_string(), "STOP".to_string()];
        assert_eq!(stop_prefix_holdback("hello <|en", &stops), 4);
        assert_eq!(stop_prefix_holdback("hello ST", &stops), 2);
        assert_eq!(stop_prefix_holdback("hello", &stops), 0);
        // A whole stop string is a match, not a partial one: the caller
        // ends the generation rather than withholding it forever.
        assert_eq!(stop_prefix_holdback("hello STOP", &stops), 0);
    }

    /// A multi-byte codepoint split across two ids must surface exactly
    /// once, when its last byte lands -- never as a replacement char.
    #[test]
    fn split_codepoint_is_emitted_once_it_completes() {
        // "é" is 0xC3 0xA9.
        let vocab = Bytes(vec![b"hi ".to_vec(), vec![0xC3], vec![0xA9]]);
        let mut mgr = DetokenizeManager::new(vocab, HashSet::new());
        assert_eq!(mgr.detokenize(&[msg(1, 0)]), vec!["hi ".to_string()]);
        assert_eq!(mgr.detokenize(&[msg(1, 1)]), vec![String::new()]);
        assert_eq!(mgr.detokenize(&[msg(1, 2)]), vec!["é".to_string()]);
    }

    #[test]
    fn partial_stop_is_withheld_then_released_when_it_does_not_complete() {
        let vocab = Bytes(vec![b"a".to_vec(), b"ST".to_vec(), b"X".to_vec()]);
        let mut mgr = DetokenizeManager::new(vocab, HashSet::new());
        let with_stop = |uid, token| DetokenizeMsg {
            stop_strs: vec!["STOP".to_string()],
            ..msg(uid, token)
        };
        assert_eq!(mgr.detokenize(&[with_stop(1, 0)]), vec!["a".to_string()]);
        // "ST" could still become "STOP": hold it.
        assert_eq!(mgr.detokenize(&[with_stop(1, 1)]), vec![String::new()]);
        // "X" settles it -- the held run is released with the new text.
        assert_eq!(mgr.detokenize(&[with_stop(1, 2)]), vec!["STX".to_string()]);
    }

    #[test]
    fn a_finished_step_trims_at_the_matched_stop_and_drops_the_state() {
        let vocab = Bytes(vec![b"a ".to_vec(), b"STOP".to_vec()]);
        let mut mgr = DetokenizeManager::new(vocab, HashSet::new());
        mgr.detokenize(&[DetokenizeMsg {
            stop_strs: vec!["STOP".to_string()],
            ..msg(7, 0)
        }]);
        let out = mgr.detokenize(&[DetokenizeMsg {
            finished: true,
            matched_stop: Some("STOP".to_string()),
            stop_strs: vec!["STOP".to_string()],
            ..msg(7, 1)
        }]);
        assert_eq!(out, vec![String::new()]);
        assert!(!mgr.is_tracking(7));
    }

    /// The EOS that ends a generation is a control token: it is never
    /// appended, so its text can never reach the response.
    #[test]
    fn terminal_eos_is_not_decoded_into_the_output() {
        let vocab = Bytes(vec![b"done".to_vec(), b"<|eos|>".to_vec()]);
        let mut mgr = DetokenizeManager::new(vocab, HashSet::from([1]));
        assert_eq!(mgr.detokenize(&[msg(3, 0)]), vec!["done".to_string()]);
        let out = mgr.detokenize(&[DetokenizeMsg {
            finished: true,
            ..msg(3, 1)
        }]);
        assert_eq!(out, vec![String::new()], "the EOS renders to nothing");
        assert!(!mgr.is_tracking(3));
    }

    #[test]
    fn sequences_are_independent() {
        let vocab = Bytes(vec![b"a ".to_vec(), b"b ".to_vec()]);
        let mut mgr = DetokenizeManager::new(vocab, HashSet::new());
        let out = mgr.detokenize(&[msg(1, 0), msg(2, 1)]);
        assert_eq!(out, vec!["a ".to_string(), "b ".to_string()]);
        mgr.discard(1);
        assert!(!mgr.is_tracking(1));
        assert!(mgr.is_tracking(2));
    }
}
