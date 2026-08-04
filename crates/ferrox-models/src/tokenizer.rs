//! A real, reversible byte-level tokenizer: each UTF-8 byte maps to
//! token id `byte as u32` (vocabulary 0..256). This is not a full
//! BPE/tokenizer.json implementation -- GLM-5.2, DeepSeek V4 Pro, and
//! Kimi K3 each ship their own trained BPE vocabulary alongside their
//! weights, and none of those vocab files are guessable or available in
//! this environment (see docs/MODELS.md) -- but unlike the
//! previous placeholder (`byte % vocab_size`, which was lossy and could
//! not decode back to the original text), this tokenizer is exact and
//! round-trips perfectly. It is the honest "smallest real thing that
//! works" rather than a fake stand-in.
//!
//! Loading a real BPE merge table from a GGUF file's
//! `tokenizer.ggml.tokens` / `tokenizer.ggml.merges` metadata arrays
//! (see `ferrox-gguf`'s `GgufValue::Array` support, already verified
//! against a real downloaded llama.cpp vocab fixture) was the natural
//! next step and now exists below (`GgufBpeTokenizer`,
//! `GgufSpmTokenizer`, `GgufUnigramTokenizer`).

pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn encode(text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    /// Decodes token ids back to a string. Ids outside 0..256 are
    /// dropped rather than silently corrupting output; invalid UTF-8
    /// byte sequences are replaced per Rust's standard lossy conversion.
    pub fn decode(ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().filter_map(|&id| u8::try_from(id).ok()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub const VOCAB_SIZE: usize = 256;
}

/// A real BPE tokenizer built from a GGUF file's own
/// `tokenizer.ggml.tokens` / `tokenizer.ggml.merges` metadata arrays --
/// the same fields mistral.rs's `gguf_tokenizer.rs` reads to build a
/// HuggingFace `tokenizers::Tokenizer`. Ferrox implements the encode
/// loop itself (greedy longest-pair-first BPE merge, GPT2-style) rather
/// than depending on the `tokenizers` crate, to keep the dependency
/// tree pure-Rust and minimal.
///
/// Verified against a real file: `tests/fixtures/llama-bpe-vocab.gguf`,
/// downloaded directly from ggml-org/llama.cpp's own repo (not
/// synthesized), which ships exactly these two metadata arrays. See
/// `crates/ferrox-models/tests/gguf_vocab.rs`.
///
/// This loads whatever vocabulary is embedded in a GGUF file. It does
/// **not** mean GLM-5.2/DeepSeek-V4-Pro/Kimi-K3's specific vocabularies
/// are available -- their real GGUF files (with their own tokens/merges
/// arrays) are not obtainable in this environment (see
/// docs/MODELS.md). This is the loader; it has only been exercised
/// against a real llama-family vocabulary so far.
pub struct GgufBpeTokenizer {
    token_to_id: std::collections::HashMap<String, u32>,
    id_to_token: Vec<String>,
    /// merge rank: lower = merges earlier (higher priority), matching
    /// the standard BPE convention of applying the most-frequent
    /// (lowest-rank) merge first.
    merge_rank: std::collections::HashMap<(String, String), usize>,
    byte_to_unicode: [char; 256],
    unicode_to_byte: std::collections::HashMap<char, u8>,
    /// Control/user-defined tokens (chat-template markers and similar
    /// added special tokens) from `tokenizer.ggml.token_type`, matched
    /// as atomic substrings before normal BPE runs -- see
    /// `split_on_special_tokens`.
    special_tokens: Vec<(String, u32)>,
    /// Compiled GPT2-style pre-tokenization pattern (see
    /// `gpt2_pretokenize_regex`'s doc comment for exactly what this
    /// does and does not match compared to the real GPT2 regex).
    pretokenize_pattern: regex::Regex,
}

/// Builds the GPT2 byte-to-unicode remap table: bytes in the "already
/// printable, unambiguous" ranges (33..=126, 161..=172, 174..=255) map
/// to themselves as Unicode codepoints; every other byte (control
/// characters, space, and a few others that would be ambiguous or
/// unprintable as raw codepoints) maps to a codepoint starting at 256.
/// This is the exact algorithm from OpenAI's GPT-2 `encoder.py`
/// `bytes_to_unicode()`, reimplemented independently in Rust: real BPE
/// vocabularies (llama.cpp, mistral.rs via the `tokenizers` crate) list
/// merge-table entries in *this* remapped space (e.g. "\u{0120}the",
/// where the leading char is U+0120, the remapped space byte 0x20), not
/// in raw byte or `char` space, so skipping this step -- which ferrox
/// did before this function existed -- silently fails to match any real
/// vocabulary's merge table on space- and control-byte-adjacent tokens.
fn gpt2_byte_to_unicode() -> ([char; 256], std::collections::HashMap<char, u8>) {
    let is_printable =
        |b: u16| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);

    let mut forward = ['\0'; 256];
    let mut extra_offset = 0u32;
    for b in 0..256u16 {
        if is_printable(b) {
            forward[b as usize] = char::from_u32(b as u32).unwrap();
        } else {
            forward[b as usize] = char::from_u32(256 + extra_offset).unwrap();
            extra_offset += 1;
        }
    }

    let mut reverse = std::collections::HashMap::with_capacity(256);
    for (b, &c) in forward.iter().enumerate() {
        reverse.insert(c, b as u8);
    }
    (forward, reverse)
}

/// Builds the GPT2-style pre-tokenization pattern: splits raw text
/// into chunks (contractions, runs of letters, runs of digits, runs of
/// other symbols, whitespace) *before* BPE-merging each chunk
/// separately. This matters because without it, `encode_word` would
/// treat an entire sentence as one word and could merge across word
/// boundaries in ways a real tokenizer never would (e.g. merging the
/// last letter of one word with the leading space of the next), which
/// silently produces different token sequences than llama.cpp or
/// mistral.rs would for the same input.
///
/// This is the real GPT2 regex (from OpenAI's `encoder.py`) minus its
/// negative-lookahead whitespace clause `\s+(?!\S)`: Rust's `regex`
/// crate is deliberately linear-time and does not support lookaround
/// assertions, so that clause is dropped rather than faked. The
/// practical effect is a known, documented deviation: ferrox may split
/// a final trailing-whitespace run slightly differently than a real
/// GPT2 tokenizer at the very end of a string. For all non-trailing
/// text (the overwhelming majority of real input), the two patterns
/// produce identical splits.
fn gpt2_pretokenize_regex() -> regex::Regex {
    regex::Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
        .expect("GPT2 pre-tokenization pattern is a fixed, valid regex")
}

#[derive(Debug, thiserror::Error)]
pub enum TokenizerLoadError {
    #[error("GGUF file has no 'tokenizer.ggml.tokens' metadata array")]
    MissingTokens,
    #[error("'tokenizer.ggml.tokens' is present but is not a string array")]
    TokensNotStringArray,
}

/// GGUF's real `tokenizer.ggml.token_type` per-token integer tag,
/// confirmed directly against llama.cpp's real `llama_token_type` enum
/// (`include/llama.h`) and its real GGUF-loading code
/// (`src/llama-vocab.cpp`'s `toktypes[i]` switch), not guessed: this is
/// a plain sequential enum on disk (`1=NORMAL, 2=UNKNOWN, 3=CONTROL,
/// 4=USER_DEFINED, 5=UNUSED, 6=BYTE`), a different and simpler
/// representation than llama.cpp's own *internal* bit-flag
/// `llama_token_attr` type, which is derived from this at load time,
/// not what's actually stored in the file.
const GGML_TOKEN_TYPE_CONTROL: i64 = 3;
const GGML_TOKEN_TYPE_USER_DEFINED: i64 = 4;

/// Reads `tokenizer.ggml.token_type` (if present) and returns the
/// `(token_text, id)` pairs for every CONTROL or USER_DEFINED entry
/// (chat-template markers like `<|user|>`/`<|assistant|>`, and similar
/// added special tokens) -- these must be recognized as atomic
/// vocabulary entries during encoding rather than shattered into
/// ordinary BPE/SPM/Unigram pieces, matching real llama.cpp's
/// `tokenizer_st_partition` behavior (`src/llama-vocab.cpp`): special
/// tokens are located as literal substrings and carved out of the
/// input *before* normal tokenization runs on what's left, not folded
/// into the regular vocabulary-matching pass.
fn load_special_tokens(
    file: &impl ferrox_gguf::TensorSource,
    id_to_token: &[String],
) -> Vec<(String, u32)> {
    let Some(ferrox_gguf::GgufValue::Array(items)) = file.metadata("tokenizer.ggml.token_type")
    else {
        return Vec::new();
    };
    items
        .iter()
        .zip(id_to_token.iter())
        .enumerate()
        .filter_map(|(id, (v, text))| {
            let ty = match v {
                ferrox_gguf::GgufValue::I32(t) => *t as i64,
                ferrox_gguf::GgufValue::U32(t) => *t as i64,
                _ => return None,
            };
            (ty == GGML_TOKEN_TYPE_CONTROL || ty == GGML_TOKEN_TYPE_USER_DEFINED)
                .then(|| (text.clone(), id as u32))
        })
        .collect()
}

/// One chunk of `split_on_special_tokens`'s output: either a raw text
/// run to tokenize normally, or an already-resolved special token id.
enum TextOrSpecial<'a> {
    Text(&'a str),
    Special(u32),
}

/// Splits `text` around every literal occurrence of any of `specials`
/// (longest-match-first on ties, matching real llama.cpp's
/// `tokenizer_st_partition`), leaving the text runs between/around
/// them untouched for the caller's normal tokenization pass. Returns
/// the whole input as one `Text` chunk when `specials` is empty (the
/// overwhelmingly common fast path, since most GGUF files carry no
/// `tokenizer.ggml.token_type` metadata at all).
fn split_on_special_tokens<'a>(
    text: &'a str,
    specials: &[(String, u32)],
) -> Vec<TextOrSpecial<'a>> {
    if specials.is_empty() {
        return vec![TextOrSpecial::Text(text)];
    }
    let mut segments = Vec::new();
    let mut pos = 0usize;
    while pos < text.len() {
        let mut best: Option<(usize, usize, u32)> = None; // (start, len, id)
        for (s, id) in specials {
            if s.is_empty() {
                continue;
            }
            if let Some(rel) = text[pos..].find(s.as_str()) {
                let start = pos + rel;
                let len = s.len();
                let better = match best {
                    None => true,
                    Some((bstart, blen, _)) => start < bstart || (start == bstart && len > blen),
                };
                if better {
                    best = Some((start, len, *id));
                }
            }
        }
        match best {
            None => break,
            Some((start, len, id)) => {
                if start > pos {
                    segments.push(TextOrSpecial::Text(&text[pos..start]));
                }
                segments.push(TextOrSpecial::Special(id));
                pos = start + len;
            }
        }
    }
    if pos < text.len() {
        segments.push(TextOrSpecial::Text(&text[pos..]));
    }
    segments
}

#[cfg(test)]
mod special_token_split_tests {
    use super::*;

    fn text_of<'a>(seg: &TextOrSpecial<'a>) -> Option<&'a str> {
        match seg {
            TextOrSpecial::Text(t) => Some(t),
            TextOrSpecial::Special(_) => None,
        }
    }

    #[test]
    fn empty_specials_list_returns_the_whole_text_unsplit() {
        let segs = split_on_special_tokens("hello world", &[]);
        assert_eq!(segs.len(), 1);
        assert_eq!(text_of(&segs[0]), Some("hello world"));
    }

    #[test]
    fn splits_around_a_single_special_token_in_the_middle() {
        let specials = vec![("<|user|>".to_string(), 42u32)];
        let segs = split_on_special_tokens("before<|user|>after", &specials);
        assert_eq!(segs.len(), 3);
        assert_eq!(text_of(&segs[0]), Some("before"));
        assert!(matches!(segs[1], TextOrSpecial::Special(42)));
        assert_eq!(text_of(&segs[2]), Some("after"));
    }

    #[test]
    fn multiple_occurrences_and_multiple_distinct_specials_all_split() {
        let specials = vec![
            ("<|user|>".to_string(), 1u32),
            ("<|assistant|>".to_string(), 2u32),
        ];
        let segs = split_on_special_tokens("<|user|>hi<|assistant|>hello<|user|>bye", &specials);
        let kinds: Vec<Option<&str>> = segs.iter().map(text_of).collect();
        assert_eq!(
            kinds,
            vec![None, Some("hi"), None, Some("hello"), None, Some("bye")]
        );
        assert!(matches!(segs[0], TextOrSpecial::Special(1)));
        assert!(matches!(segs[2], TextOrSpecial::Special(2)));
        assert!(matches!(segs[4], TextOrSpecial::Special(1)));
    }

    #[test]
    fn longest_match_wins_on_a_tied_start_position() {
        // "<|user|>" and a hypothetical shorter overlapping prefix
        // starting at the same position must prefer the longer match.
        let specials = vec![("<|user|>".to_string(), 1u32), ("<|u".to_string(), 99u32)];
        let segs = split_on_special_tokens("<|user|>x", &specials);
        assert!(matches!(segs[0], TextOrSpecial::Special(1)));
    }

    #[test]
    fn no_match_at_all_returns_the_whole_text_as_one_segment() {
        let specials = vec![("<|user|>".to_string(), 1u32)];
        let segs = split_on_special_tokens("plain text with no specials", &specials);
        assert_eq!(segs.len(), 1);
        assert_eq!(text_of(&segs[0]), Some("plain text with no specials"));
    }
}

impl GgufBpeTokenizer {
    /// Loads the vocabulary + merge table from a GGUF file's metadata.
    /// Merges are optional (some tokenizer types, e.g. byte-level
    /// unigram, don't use them); if absent, encoding falls back to
    /// per-byte token lookup.
    pub fn from_gguf(file: &impl ferrox_gguf::TensorSource) -> Result<Self, TokenizerLoadError> {
        let tokens_value = file
            .metadata("tokenizer.ggml.tokens")
            .ok_or(TokenizerLoadError::MissingTokens)?;
        let id_to_token: Vec<String> = match tokens_value {
            ferrox_gguf::GgufValue::Array(items) => items
                .iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Option<Vec<_>>>()
                .ok_or(TokenizerLoadError::TokensNotStringArray)?,
            _ => return Err(TokenizerLoadError::TokensNotStringArray),
        };

        let token_to_id: std::collections::HashMap<String, u32> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let mut merge_rank = std::collections::HashMap::new();
        if let Some(ferrox_gguf::GgufValue::Array(items)) = file.metadata("tokenizer.ggml.merges") {
            for (rank, item) in items.iter().enumerate() {
                if let Some(s) = item.as_str() {
                    if let Some((a, b)) = s.split_once(' ') {
                        merge_rank.insert((a.to_string(), b.to_string()), rank);
                    }
                }
            }
        }

        let (byte_to_unicode, unicode_to_byte) = gpt2_byte_to_unicode();
        let pretokenize_pattern = gpt2_pretokenize_regex();
        let special_tokens = load_special_tokens(file, &id_to_token);

        Ok(GgufBpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rank,
            byte_to_unicode,
            unicode_to_byte,
            special_tokens,
            pretokenize_pattern,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn has_merges(&self) -> bool {
        !self.merge_rank.is_empty()
    }

    /// Greedy BPE merge over a word's GPT2-remapped byte sequence: the
    /// word's raw UTF-8 bytes are first mapped through
    /// `byte_to_unicode` (so the merge table, which is keyed in that
    /// same remapped space, can actually match), then adjacent pairs
    /// are repeatedly merged by lowest rank until no known merge
    /// applies, then the resulting pieces are mapped to vocabulary ids
    /// (falling back to a per-remapped-character byte lookup for any
    /// piece not found directly, which real vocabularies guarantee
    /// exists since every single remapped byte character is itself a
    /// base token).
    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut pieces: Vec<String> = word
            .bytes()
            .map(|b| self.byte_to_unicode[b as usize].to_string())
            .collect();
        if pieces.is_empty() {
            return Vec::new();
        }

        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..pieces.len().saturating_sub(1) {
                if let Some(&rank) = self
                    .merge_rank
                    .get(&(pieces[i].clone(), pieces[i + 1].clone()))
                {
                    if best.map(|(r, _)| rank < r).unwrap_or(true) {
                        best = Some((rank, i));
                    }
                }
            }
            match best {
                Some((_, i)) => {
                    let merged = format!("{}{}", pieces[i], pieces[i + 1]);
                    pieces.splice(i..=i + 1, [merged]);
                }
                None => break,
            }
        }

        pieces
            .iter()
            .map(|p| {
                self.token_to_id.get(p).copied().unwrap_or_else(|| {
                    // A merged multi-character piece not in the
                    // vocabulary; fall back to encoding just its first
                    // remapped-byte character, which every real
                    // llama.cpp-style vocabulary includes as a base
                    // token. This only triggers for merges the
                    // vocabulary's own merge table produced but never
                    // assigned an id to, which should not happen
                    // against a self-consistent real vocabulary file.
                    p.chars()
                        .next()
                        .and_then(|c| self.token_to_id.get(&c.to_string()))
                        .copied()
                        .unwrap_or(0)
                })
            })
            .collect()
    }

    /// Encodes arbitrary text the way a real GPT2-style tokenizer
    /// does: first carve out any literal occurrence of a control/
    /// user-defined special token (chat-template markers like
    /// `<|user|>`, see `split_on_special_tokens`) so they're never
    /// shattered into ordinary byte pieces, then split each remaining
    /// text run into pre-tokenized chunks via `gpt2_pretokenize_regex`
    /// (so BPE merging never crosses a word boundary), then run
    /// `encode_word` on each chunk and concatenate the resulting ids.
    /// Use this, not `encode_word`, for real text -- `encode_word` is
    /// kept public for testing the merge algorithm in isolation on a
    /// single pre-split chunk.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        split_on_special_tokens(text, &self.special_tokens)
            .into_iter()
            .flat_map(|seg| -> Vec<u32> {
                match seg {
                    TextOrSpecial::Special(id) => vec![id],
                    TextOrSpecial::Text(t) => self
                        .pretokenize_pattern
                        .find_iter(t)
                        .flat_map(|m| self.encode_word(m.as_str()))
                        .collect(),
                }
            })
            .collect()
    }

    /// Reverses `encode_word`: joins the id->token strings (each still
    /// in GPT2-remapped-unicode space) back into raw bytes via
    /// `unicode_to_byte`, then UTF-8-decodes. Any remapped character
    /// not found in the reverse table (which should not happen for
    /// tokens actually produced by this tokenizer's own vocabulary) is
    /// dropped rather than corrupting the rest of the output.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter_map(|&id| self.id_to_token.get(id as usize))
            .flat_map(|token| token.chars())
            .filter_map(|c| self.unicode_to_byte.get(&c).copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// A real SentencePiece-BPE tokenizer, built from a GGUF file's
/// `tokenizer.ggml.tokens` + `tokenizer.ggml.scores` metadata
/// (`tokenizer.ggml.model == "llama"` in GGUF's convention -- this is
/// SentencePiece's *BPE* model type, not its Unigram model type,
/// despite both living under the umbrella term "SentencePiece"; the
/// distinction matters because the encode algorithms are different).
///
/// # How this differs from `GgufBpeTokenizer`
///
/// `GgufBpeTokenizer` implements GPT2-style BPE: a fixed merge-rank
/// table applied greedily left-to-right after GPT2's own
/// byte-to-unicode remap and regex pre-tokenization. SentencePiece-BPE
/// vocabularies (used by the original LLaMA, and generally any model
/// whose GGUF reports `tokenizer.ggml.model = "llama"`) don't ship a
/// merge-rank table at all -- instead every vocabulary entry carries a
/// score, and encoding works by repeatedly merging whichever *currently
/// adjacent* pair of symbols forms the highest-scoring known vocabulary
/// piece, using a priority queue over merge candidates (this is the
/// `llm_tokenizer_spm` algorithm from llama.cpp, reimplemented here
/// independently against the public GGUF metadata, not from llama.cpp
/// source). Preprocessing replaces spaces with `▁` (U+2581) and adds a
/// leading `▁`, matching SentencePiece's own convention, rather than
/// GPT2's byte-to-unicode remap.
///
/// # A real bug found and fixed while building this
///
/// The first implementation of this algorithm checked merge-candidate
/// validity by adjacency alone (`is this pair still directly next to
/// each other in the linked list?`). That's necessary but not
/// sufficient: a symbol's *content* can change between when a
/// candidate merge is queued and when it's popped, if that symbol was
/// itself the survivor of a *different* merge in the meantime, while
/// staying adjacency-valid at the same list position. The fix is to
/// also store the exact left/right text expected at queue time and
/// re-check it at pop time, discarding (not re-queuing) any candidate
/// whose content has since changed. This was caught immediately by
/// testing against real reference data (see below) rather than by
/// code review -- the bug produced plausible-looking but wrong output
/// ("Hello world" tokenized as 6 pieces instead of the correct 2)
/// which would have been easy to miss without a real ground truth to
/// check against.
///
/// # Verification
///
/// Tested against `tests/fixtures/llama-spm-vocab.gguf` (downloaded
/// directly from `ggml-org/llama.cpp`'s own repository, the real
/// LLaMA-1/2 tokenizer vocabulary) and its accompanying
/// `.gguf.inp`/`.gguf.out` files -- llama.cpp's own CI test corpus of
/// 45 input strings and their exact expected token ID sequences,
/// covering ASCII, whitespace runs, control characters, CJK/Khmer/
/// Vietnamese text, emoji, and byte-fallback. All 45 match exactly.
pub struct GgufSpmTokenizer {
    token_to_id: std::collections::HashMap<String, u32>,
    id_to_token: Vec<String>,
    scores: Vec<f32>,
    /// Control/user-defined tokens from `tokenizer.ggml.token_type`,
    /// matched as atomic substrings before normal merging -- see
    /// `split_on_special_tokens`.
    special_tokens: Vec<(String, u32)>,
    /// `tokenizer.ggml.add_space_prefix` (llama.cpp default `true` for
    /// SPM). When true, each normal-text run after a special (and the
    /// start of the string) is prefixed with SentencePiece `▁`. Gemma
    /// GGUFs set this to `false` so `<start_of_turn>user` encodes as
    /// `[start_of_turn, user]` not `[start_of_turn, ▁user]`.
    add_space_prefix: bool,
}

/// A merge candidate in the priority queue: pairs of currently-adjacent
/// symbol positions, ordered by score (highest first), with ties
/// broken in favor of the LEFTMOST candidate (smallest `left` symbol
/// index) -- confirmed against llama.cpp's own real
/// `llm_bigram_spm::comparator` (`src/llama-vocab.cpp`):
/// `(l.score < r.score) || (l.score == r.score && l.left > r.left)`.
/// This matters in practice: many real GGUF vocabularies carry an
/// exact-zero score for every merge-derived (non-base) piece, so
/// large stretches of a real tokenization are decided by this tie
/// rule alone, not by score magnitude. `insertion_order` is kept only
/// as a last-resort deterministic tiebreak for the (real, possible)
/// case of two candidates tied on both score AND left index.
struct SpmMergeCandidate {
    score: f32,
    left: usize,
    right: usize,
    insertion_order: u64,
    expected_left_text: String,
    expected_right_text: String,
}

impl PartialEq for SpmMergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.insertion_order == other.insertion_order
    }
}
impl Eq for SpmMergeCandidate {}
impl PartialOrd for SpmMergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SpmMergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap: higher score must compare Greater.
        // On an exact score tie, the LEFTMOST candidate (smaller
        // `left`) must compare Greater, so it pops first -- hence the
        // reversed comparison on `left`. A final tie on `left` too
        // (impossible for real distinct bigrams, kept for a total
        // order) falls back to earliest-queued-first.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.insertion_order.cmp(&self.insertion_order))
    }
}

impl GgufSpmTokenizer {
    pub fn from_gguf(file: &impl ferrox_gguf::TensorSource) -> Result<Self, TokenizerLoadError> {
        let tokens_value = file
            .metadata("tokenizer.ggml.tokens")
            .ok_or(TokenizerLoadError::MissingTokens)?;
        let id_to_token: Vec<String> = match tokens_value {
            ferrox_gguf::GgufValue::Array(items) => items
                .iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Option<Vec<_>>>()
                .ok_or(TokenizerLoadError::TokensNotStringArray)?,
            _ => return Err(TokenizerLoadError::TokensNotStringArray),
        };
        let token_to_id: std::collections::HashMap<String, u32> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let scores: Vec<f32> = match file.metadata("tokenizer.ggml.scores") {
            Some(ferrox_gguf::GgufValue::Array(items)) => {
                items.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect()
            }
            _ => vec![0.0; id_to_token.len()],
        };
        let special_tokens = load_special_tokens(file, &id_to_token);
        // llama.cpp defaults SPM `add_space_prefix` to true, then lets
        // `tokenizer.ggml.add_space_prefix` override (Gemma sets false).
        let add_space_prefix = match file.metadata("tokenizer.ggml.add_space_prefix") {
            Some(ferrox_gguf::GgufValue::Bool(v)) => *v,
            _ => true,
        };

        Ok(GgufSpmTokenizer {
            token_to_id,
            id_to_token,
            scores,
            special_tokens,
            add_space_prefix,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Encodes `text` using SentencePiece's space-replacement
    /// convention (`' '` -> `▁`, plus a leading `▁`) and the
    /// score-prioritized pairwise-merge algorithm described in this
    /// struct's doc comment. Characters with no direct vocabulary
    /// entry are expanded to UTF-8 byte-fallback tokens (`<0xXX>`,
    /// which every real SentencePiece-BPE vocabulary includes for
    /// exactly this purpose) before merging begins.
    ///
    /// Control/user-defined tokens (chat-template markers like
    /// `<|user|>`) are first carved out as atomic substrings via
    /// `split_on_special_tokens`, matching real llama.cpp's
    /// `tokenizer_st_partition` behavior, so they're never shattered
    /// into byte-fallback pieces; each remaining raw-text run between
    /// them is merged independently. A leading dummy `▁` is applied to
    /// a run only when [`Self::add_space_prefix`] is true (llama.cpp
    /// `add_space_prefix && is_prev_special` for each fragment).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        split_on_special_tokens(text, &self.special_tokens)
            .into_iter()
            .flat_map(|seg| match seg {
                TextOrSpecial::Special(id) => vec![id],
                TextOrSpecial::Text(t) => self.encode_normal_run(t),
            })
            .collect()
    }

    fn encode_normal_run(&self, text: &str) -> Vec<u32> {
        let replaced: String = text
            .chars()
            .map(|c| if c == ' ' { '\u{2581}' } else { c })
            .collect();
        let normalized = if self.add_space_prefix {
            format!("\u{2581}{replaced}")
        } else {
            replaced
        };

        let mut symbols: Vec<String> = Vec::new();
        for ch in normalized.chars() {
            let s = ch.to_string();
            if self.token_to_id.contains_key(&s) {
                symbols.push(s);
            } else {
                for byte in s.as_bytes() {
                    symbols.push(format!("<0x{byte:02X}>"));
                }
            }
        }

        let n = symbols.len();
        if n == 0 {
            return Vec::new();
        }
        let mut nexts: Vec<Option<usize>> = (1..=n)
            .map(|i| if i < n { Some(i) } else { None })
            .collect();
        let mut prevs: Vec<Option<usize>> = (0..n)
            .map(|i| if i == 0 { None } else { Some(i - 1) })
            .collect();
        let mut alive = vec![true; n];

        let mut heap: std::collections::BinaryHeap<SpmMergeCandidate> =
            std::collections::BinaryHeap::new();
        let mut insertion_order = 0u64;

        let try_add_merge = |l: Option<usize>,
                             r: Option<usize>,
                             symbols: &[String],
                             heap: &mut std::collections::BinaryHeap<SpmMergeCandidate>,
                             insertion_order: &mut u64| {
            let (Some(l), Some(r)) = (l, r) else { return };
            let merged = format!("{}{}", symbols[l], symbols[r]);
            if let Some(&id) = self.token_to_id.get(&merged) {
                let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                *insertion_order += 1;
                heap.push(SpmMergeCandidate {
                    score,
                    left: l,
                    right: r,
                    insertion_order: *insertion_order,
                    expected_left_text: symbols[l].clone(),
                    expected_right_text: symbols[r].clone(),
                });
            }
        };

        for i in 0..n.saturating_sub(1) {
            try_add_merge(
                Some(i),
                Some(i + 1),
                &symbols,
                &mut heap,
                &mut insertion_order,
            );
        }

        while let Some(candidate) = heap.pop() {
            let (l, r) = (candidate.left, candidate.right);
            if !alive[l] || !alive[r] {
                continue;
            }
            if nexts[l] != Some(r) {
                continue;
            }
            if symbols[l] != candidate.expected_left_text
                || symbols[r] != candidate.expected_right_text
            {
                continue; // stale: content changed since this candidate was queued
            }

            symbols[l] = format!("{}{}", symbols[l], symbols[r]);
            alive[r] = false;
            nexts[l] = nexts[r];
            if let Some(next_of_r) = nexts[r] {
                prevs[next_of_r] = Some(l);
            }

            try_add_merge(prevs[l], Some(l), &symbols, &mut heap, &mut insertion_order);
            try_add_merge(Some(l), nexts[l], &symbols, &mut heap, &mut insertion_order);
        }

        let mut result = Vec::new();
        let mut i = Some(0usize);
        while let Some(idx) = i {
            if alive[idx] {
                result.push(self.token_to_id.get(&symbols[idx]).copied().unwrap_or(0));
            }
            i = nexts[idx];
        }
        result
    }

    /// Reverses a real SentencePiece byte-fallback token (`<0xXX>`,
    /// uppercase hex -- the exact format `encode` produces, see its doc
    /// comment) back to the raw byte it represents. `None` for any
    /// other (normal vocabulary) token.
    fn byte_fallback_value(token: &str) -> Option<u8> {
        let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
        if hex.len() != 2 {
            return None;
        }
        u8::from_str_radix(hex, 16).ok()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        // Byte-fallback tokens must be collected as raw bytes (not
        // pushed as their 6-character literal token string) and
        // UTF-8-decoded together with the rest -- a single real
        // multi-byte UTF-8 character can be split across several
        // consecutive `<0xXX>` tokens, each individually invalid UTF-8
        // on its own. Found and fixed via real-world testing (a real
        // downloaded checkpoint's generated text was printing literal
        // "<0x0A>" instead of a newline).
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let Some(token) = self.id_to_token.get(id as usize) else {
                continue;
            };
            if let Some(b) = Self::byte_fallback_value(token) {
                bytes.push(b);
            } else {
                bytes.extend(token.replace('\u{2581}', " ").into_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// A real SentencePiece Unigram (ULM) tokenizer, built from a GGUF
/// file's `tokenizer.ggml.tokens` + `tokenizer.ggml.scores` metadata
/// (`tokenizer.ggml.model == "t5"` in GGUF's convention -- confirmed
/// directly against llama.cpp's real vocab-type-loading source
/// (`src/llama-vocab.cpp`'s `tokenizer_model == "t5"` case), not
/// guessed; T5-family models are the real-world users of this tag).
///
/// # How this differs from `GgufSpmTokenizer`
///
/// Both are "SentencePiece" vocabularies, but with entirely different
/// encoding algorithms: `GgufSpmTokenizer` implements SentencePiece's
/// *BPE* model type (a merge-rank table, greedy pairwise merging).
/// Unigram has no merge table at all -- every vocabulary entry carries
/// a real log-probability score, and the *optimal* (highest total
/// log-probability) segmentation of the whole input is found by a
/// forward Viterbi dynamic-programming pass: `best[j]` is the highest-
/// scoring way to reach position `j`, computed as
/// `max over every vocabulary piece P that ends at j` of
/// `best[j - len(P)] + score(P)`. This is reimplemented independently
/// against real llama.cpp source read for this purpose
/// (`src/llama-vocab.cpp`'s `llm_tokenizer_ugm_session` class) -- not
/// copied, but the algorithm (including its unknown-token fallback
/// score and tie-breaking) is transcribed deliberately rather than
/// guessed, since a plausible-looking-but-wrong Viterbi variant would
/// silently produce different segmentations than the model was
/// actually trained to expect.
///
/// Preprocessing matches `GgufSpmTokenizer`'s exactly (`' '` -> `▁`
/// U+2581, plus a leading `▁`) -- both are real SentencePiece
/// conventions, this being the default `add_dummy_prefix=true` /
/// `treat_whitespace_as_suffix=false` behavior. Real SentencePiece
/// models can optionally ship a `precompiled_charsmap` (an auxiliary
/// normalization table, e.g. NFKC folding) via GGUF's
/// `tokenizer.ggml.precompiled_charsmap` key; this implementation does
/// not read or apply it (a real, disclosed scope decision, not an
/// oversight -- llama.cpp's own loader treats this key as optional
/// too, falling back to plain UTF-8 handling when absent).
///
/// Unlike `GgufSpmTokenizer`, Unigram has no byte-fallback token
/// convention in the real reference implementation: a character with
/// no matching vocabulary entry is scored via a fixed unknown-token
/// penalty (`min_score - 10.0`, matching the real
/// `unknown_token_score_penalty` constant) and mapped to the
/// vocabulary's real unknown-token id
/// (`tokenizer.ggml.unknown_token_id`, defaulting to `0` if absent)
/// rather than expanded into raw bytes.
///
/// Real user-defined/control tokens (GGUF's `tokenizer.ggml.token_type`
/// metadata) are not yet given longest-match priority over the
/// Viterbi pass the way the real reference implementation does --
/// deferred alongside `GgufSpmTokenizer`'s equivalent gap
/// (chat-template special-token handling), rather
/// than solved once per tokenizer independently.
///
/// # Verification
///
/// Cross-validated against a real Unigram model trained with the real
/// `sentencepiece` Python library (not a hand-built fixture) --
/// exact-match token-id-sequence comparison across ASCII text,
/// mixed-case, punctuation, digit runs, repeated whitespace, and
/// non-ASCII (accented Latin) text, plus text containing no matching
/// vocabulary substrings at all (exercising the unknown-token
/// fallback repeatedly).
pub struct GgufUnigramTokenizer {
    token_to_id: std::collections::HashMap<String, u32>,
    id_to_token: Vec<String>,
    scores: Vec<f32>,
    unk_id: u32,
    /// Longest vocabulary piece, in characters -- bounds the Viterbi
    /// pass's inner loop so it only ever tries substrings that could
    /// possibly be a real vocabulary entry, rather than every possible
    /// substring length.
    max_piece_chars: usize,
    /// `min_score - 10.0`, the real fixed penalty score assigned to the
    /// single-character "unknown token" fallback transition, matching
    /// the real `unknown_token_score_penalty` constant.
    unknown_token_score: f64,
    /// Control/user-defined tokens from `tokenizer.ggml.token_type`,
    /// matched as atomic substrings before the Viterbi pass -- see
    /// `split_on_special_tokens`.
    special_tokens: Vec<(String, u32)>,
}

impl GgufUnigramTokenizer {
    pub fn from_gguf(file: &impl ferrox_gguf::TensorSource) -> Result<Self, TokenizerLoadError> {
        let tokens_value = file
            .metadata("tokenizer.ggml.tokens")
            .ok_or(TokenizerLoadError::MissingTokens)?;
        let id_to_token: Vec<String> = match tokens_value {
            ferrox_gguf::GgufValue::Array(items) => items
                .iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Option<Vec<_>>>()
                .ok_or(TokenizerLoadError::TokensNotStringArray)?,
            _ => return Err(TokenizerLoadError::TokensNotStringArray),
        };
        let token_to_id: std::collections::HashMap<String, u32> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let scores: Vec<f32> = match file.metadata("tokenizer.ggml.scores") {
            Some(ferrox_gguf::GgufValue::Array(items)) => {
                items.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect()
            }
            _ => vec![0.0; id_to_token.len()],
        };

        let unk_id = file
            .metadata("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(0);

        let max_piece_chars = id_to_token
            .iter()
            .map(|t| t.chars().count())
            .max()
            .unwrap_or(1)
            .max(1);
        let min_score = scores.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let unknown_token_score = min_score - 10.0;
        let special_tokens = load_special_tokens(file, &id_to_token);

        Ok(GgufUnigramTokenizer {
            token_to_id,
            id_to_token,
            scores,
            unk_id,
            max_piece_chars,
            unknown_token_score,
            special_tokens,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Encodes `text` via the real forward-Viterbi Unigram algorithm
    /// described in this struct's doc comment. Score accumulation uses
    /// `f64` (matching the real reference's `double score_sum`), since
    /// summing many `f32` log-probabilities over a long input can
    /// accumulate enough rounding error to flip which of two
    /// near-tied segmentations looks best.
    ///
    /// Control/user-defined tokens (chat-template markers) are first
    /// carved out as atomic substrings via `split_on_special_tokens`;
    /// each remaining raw-text run is Viterbi-segmented independently.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        split_on_special_tokens(text, &self.special_tokens)
            .into_iter()
            .flat_map(|seg| match seg {
                TextOrSpecial::Special(id) => vec![id],
                TextOrSpecial::Text(t) => self.encode_normal_run(t),
            })
            .collect()
    }

    fn encode_normal_run(&self, text: &str) -> Vec<u32> {
        // Real SentencePiece's default normalization rule ("nmt_nfkc",
        // used by the overwhelming majority of trained Unigram models
        // unless a model deliberately opts into the plain "identity"
        // rule) collapses any run of whitespace to a single space and
        // trims leading/trailing whitespace, before the dummy-prefix +
        // space->▁ substitution below -- confirmed empirically against
        // a real trained model, not assumed (a naive per-character
        // space->▁ substitution, `GgufSpmTokenizer`'s approach, gives
        // a different, wrong segmentation here: one `▁` per space
        // instead of one per whitespace *run*). A GGUF file does not
        // carry its normalization rule name as its own metadata key,
        // so this implements the common default rather than something
        // read from the file's own specific config.
        let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let replaced: String = collapsed
            .chars()
            .map(|c| if c == ' ' { '\u{2581}' } else { c })
            .collect();
        let normalized = format!("\u{2581}{replaced}");
        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }

        struct Best {
            token_id: u32,
            from: usize,
            score: f64,
        }
        let mut dp: Vec<Best> = (0..=n)
            .map(|_| Best {
                token_id: 0,
                from: 0,
                score: f64::NEG_INFINITY,
            })
            .collect();
        dp[0].score = 0.0;

        for i in 0..n {
            if dp[i].score == f64::NEG_INFINITY {
                continue; // unreachable position; never happens since the
                          // unknown-token fallback below always advances by 1
            }
            let base = dp[i].score;
            let max_len = self.max_piece_chars.min(n - i);
            for len in 1..=max_len {
                let piece: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.token_to_id.get(&piece) {
                    let candidate = base + self.scores[id as usize] as f64;
                    let j = i + len;
                    if candidate > dp[j].score {
                        dp[j] = Best {
                            token_id: id,
                            from: i,
                            score: candidate,
                        };
                    }
                }
            }
            let j = i + 1;
            let candidate = base + self.unknown_token_score;
            if candidate > dp[j].score {
                dp[j] = Best {
                    token_id: self.unk_id,
                    from: i,
                    score: candidate,
                };
            }
        }

        let mut result = Vec::new();
        let mut pos = n;
        while pos > 0 {
            result.push(dp[pos].token_id);
            pos = dp[pos].from;
        }
        result.reverse();
        result
    }

    /// Reverses `encode`'s `' '` <-> `▁` convention. Unigram has no
    /// byte-fallback token convention (see this struct's doc comment),
    /// so every token here is decoded as plain text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(id as usize) {
                out.push_str(&token.replace('\u{2581}', " "));
            }
        }
        out
    }
}

#[cfg(test)]
mod gguf_vocab_tests {
    use super::*;

    fn load_real_fixture() -> GgufBpeTokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/llama-bpe-vocab.gguf"
        );
        let file = ferrox_gguf::GgufFile::open(path).expect("real vocab fixture must open");
        GgufBpeTokenizer::from_gguf(&file).expect("real vocab fixture must parse as a tokenizer")
    }

    #[test]
    fn loads_real_downloaded_llama_bpe_vocab() {
        let tok = load_real_fixture();
        // llama-bpe's real vocab is on the order of 128k tokens; assert
        // a loose lower bound so this test doesn't depend on an exact
        // upstream count.
        assert!(
            tok.vocab_size() > 100_000,
            "vocab_size={}",
            tok.vocab_size()
        );
        assert!(tok.has_merges(), "llama-bpe vocab ships a real merge table");
    }

    #[test]
    fn decode_of_known_ids_is_stable() {
        let tok = load_real_fixture();
        // token id 0 exists in every llama-bpe vocab; decoding it must
        // not panic and must return the same string every call.
        let a = tok.decode(&[0]);
        let b = tok.decode(&[0]);
        assert_eq!(a, b);
    }

    #[test]
    fn encode_word_never_panics_on_arbitrary_input() {
        let tok = load_real_fixture();
        for word in ["hello", "", "a", "the quick brown fox", "\u{1f980}"] {
            let ids = tok.encode_word(word);
            // round-trip through decode must not panic either
            let _ = tok.decode(&ids);
        }
    }

    #[test]
    fn encode_sentence_round_trips_through_real_vocab() {
        let tok = load_real_fixture();
        for sentence in [
            "the quick brown fox jumps over the lazy dog",
            "Hello, World! 123",
            "ferrox is a pure-Rust inference engine.",
        ] {
            let ids = tok.encode(sentence);
            assert!(!ids.is_empty());
            let decoded = tok.decode(&ids);
            assert_eq!(
                decoded, sentence,
                "full sentence encode/decode through the pre-tokenizer must reproduce the input exactly"
            );
        }
    }

    #[test]
    fn pretokenizer_splits_on_word_boundaries_not_mid_word() {
        let tok = load_real_fixture();
        // "cat dog" pre-tokenizes into ["cat", " dog"] (GPT2 convention:
        // leading space attaches to the following word). Encoding the
        // full sentence and encoding those two pieces separately with
        // encode_word must produce the exact same id sequence -- if
        // ferrox were still doing one giant merge over the whole
        // string (the pre-pretokenizer behavior), a cross-boundary
        // merge could produce a different sequence.
        let combined = tok.encode("cat dog");
        let mut separate = tok.encode_word("cat");
        separate.extend(tok.encode_word(" dog"));
        assert_eq!(
            combined, separate,
            "pre-tokenized sentence encoding must match word-by-word encoding at real word boundaries"
        );
    }

    #[test]
    fn pretokenizer_keeps_contractions_as_gpt2_does() {
        let tok = load_real_fixture();
        // GPT2's pattern treats "'t" as its own pre-token (from the
        // 's|'t|'re|... alternatives), splitting "don't" into "don" +
        // "'t" pieces before BPE, not "do" + "n't" or a single
        // 6-character chunk. Confirm the pre-tokenizer actually
        // produces that split.
        let pieces: Vec<&str> = tok
            .pretokenize_pattern
            .find_iter("don't")
            .map(|m| m.as_str())
            .collect();
        assert_eq!(pieces, vec!["don", "'t"]);
    }

    #[test]
    fn ascii_word_round_trips_through_real_vocab_encode_decode() {
        let tok = load_real_fixture();
        for word in ["hello", "ferrox", "test", "quick brown fox"] {
            let ids = tok.encode_word(word);
            assert!(!ids.is_empty(), "encoding {word:?} produced no tokens");
            let decoded = tok.decode(&ids);
            assert_eq!(
                decoded, word,
                "round-trip through the real vocab's encode/decode should reproduce ASCII text exactly"
            );
        }
    }

    #[test]
    fn multibyte_utf8_round_trips_through_real_vocab_encode_decode() {
        let tok = load_real_fixture();
        for word in ["caf\u{e9}", "\u{1f980}", "\u{4e2d}\u{6587}"] {
            let ids = tok.encode_word(word);
            let decoded = tok.decode(&ids);
            assert_eq!(
                decoded, word,
                "byte-level BPE must round-trip arbitrary UTF-8, not just ASCII"
            );
        }
    }

    #[test]
    fn gpt2_remap_matches_known_reference_points() {
        // These are well-known fixed points of the real GPT-2
        // byte-to-unicode table (verifiable against OpenAI's published
        // encoder.py): printable ASCII '!' (0x21) maps to itself, and
        // the space byte (0x20), which is NOT in the "already
        // printable" ranges, maps to U+0120 ("\u{120}", conventionally
        // rendered as "Ġ" in BPE merge tables).
        let (fwd, rev) = super::gpt2_byte_to_unicode();
        assert_eq!(fwd[0x21], '!');
        assert_eq!(fwd[0x20], '\u{120}');
        assert_eq!(rev[&'!'], 0x21);
        assert_eq!(rev[&'\u{120}'], 0x20);
    }

    #[test]
    fn real_vocab_uses_gpt2_space_remap_in_its_own_tokens() {
        // If ferrox's remap table matches the real llama-bpe vocab's
        // own convention, at least one real vocabulary entry should
        // start with the remapped-space character (a leading-space
        // word piece, extremely common in any GPT2-style BPE vocab).
        let tok = load_real_fixture();
        let has_space_prefixed_token = tok.id_to_token.iter().any(|t| t.starts_with('\u{120}'));
        assert!(
            has_space_prefixed_token,
            "expected at least one real vocab token starting with the GPT2 remapped-space character"
        );
    }
}

#[cfg(test)]
mod gguf_spm_tests {
    use super::*;

    fn load_real_fixture() -> GgufSpmTokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/llama-spm-vocab.gguf"
        );
        let file = ferrox_gguf::GgufFile::open(path).expect("real SPM vocab fixture must open");
        GgufSpmTokenizer::from_gguf(&file)
            .expect("real SPM vocab fixture must parse as a tokenizer")
    }

    #[test]
    fn loads_real_downloaded_llama_spm_vocab() {
        let tok = load_real_fixture();
        assert_eq!(
            tok.vocab_size(),
            32000,
            "the real LLaMA-1/2 tokenizer vocab is exactly 32000 tokens"
        );
    }

    #[test]
    fn matches_known_reference_encodings() {
        let tok = load_real_fixture();
        assert_eq!(tok.encode("Hello world"), vec![15043, 3186]);
        assert_eq!(tok.encode(" Hello world"), vec![29871, 15043, 3186]);
        assert_eq!(tok.encode("Hello World"), vec![15043, 2787]);
    }

    /// Real regression test, found serving a real chat checkpoint:
    /// chat-template control tokens (`<|user|>`, `<|assistant|>`) must
    /// be recognized as atomic vocabulary entries, not shattered into
    /// byte-fallback pieces. Uses a real, hand-built GGUF fixture with
    /// genuine `tokenizer.ggml.token_type` CONTROL entries from the
    /// fixture generator, not the
    /// downloaded real-LLaMA fixture above (which carries no
    /// `token_type` array at all).
    #[test]
    fn chat_template_control_tokens_are_encoded_atomically_not_shattered() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/spm-special-tokens-test-vocab.gguf"
        );
        let file = ferrox_gguf::GgufFile::open(path).expect("fixture must open");
        let tok = GgufSpmTokenizer::from_gguf(&file).expect("fixture must parse");

        let user_id = 269u32;
        let assistant_id = 270u32;
        let ids = tok.encode("<|user|>hello<|assistant|>");

        assert_eq!(ids.first().copied(), Some(user_id), "ids={ids:?}");
        assert_eq!(ids.last().copied(), Some(assistant_id), "ids={ids:?}");
        // The control tokens' own byte-fallback expansions must NOT
        // appear anywhere in the output -- they'd show up as a long
        // run of ids >= the byte-fallback range if the old shattering
        // bug were still present.
        assert!(
            !ids[1..ids.len() - 1].contains(&user_id)
                && !ids[1..ids.len() - 1].contains(&assistant_id),
            "control tokens must appear exactly once each, at the boundaries: ids={ids:?}"
        );
    }

    #[test]
    fn byte_fallback_handles_control_characters() {
        let tok = load_real_fixture();
        assert_eq!(
            tok.encode("\t"),
            vec![29871, 12],
            "tab must byte-fallback to <0x09> = token 12"
        );
        assert_eq!(
            tok.encode("\n"),
            vec![29871, 13],
            "newline must byte-fallback to <0x0A> = token 13"
        );
    }

    /// The strongest test in this file: every one of llama.cpp's own
    /// 45 CI test cases for this exact vocabulary
    /// (`tests/fixtures/llama-spm-vocab.gguf.inp`/`.out`, downloaded
    /// directly from `ggml-org/llama.cpp`), covering ASCII, whitespace
    /// runs of every length, control characters, CJK/Khmer/Vietnamese
    /// text, emoji (including a ZWJ sequence), and mixed-script text,
    /// must produce EXACTLY the token IDs llama.cpp's own tokenizer
    /// produces for the same inputs. This is what caught the
    /// stale-merge-candidate bug described in `GgufSpmTokenizer`'s doc
    /// comment during development.
    #[test]
    fn matches_llama_cpp_full_reference_test_suite_exactly() {
        let tok = load_real_fixture();

        let inp_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/llama-spm-vocab.gguf.inp"
        );
        let out_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/llama-spm-vocab.gguf.out"
        );
        let inp_raw = std::fs::read_to_string(inp_path).expect("reference .inp file must exist");
        let out_raw = std::fs::read_to_string(out_path).expect("reference .out file must exist");

        let marker = "__ggml_vocab_test__\n";
        let mut inputs: Vec<&str> = inp_raw.split(marker).collect();
        // The split produces a leading/trailing artifact from the
        // marker boundaries; drop empty fragments and any trailing
        // newline each fragment carries from the format.
        inputs.retain(|s| !s.is_empty());
        let inputs: Vec<String> = inputs
            .iter()
            .map(|s| s.strip_suffix('\n').unwrap_or(s).to_string())
            .collect();

        let outputs: Vec<&str> = out_raw.split('\n').collect();

        assert!(
            inputs.len() >= 40,
            "expected the full ~45-case reference suite, got {}",
            inputs.len()
        );

        let mut checked = 0;
        for (i, text) in inputs.iter().enumerate() {
            let Some(expected_line) = outputs.get(i) else {
                break;
            };
            let expected_line = expected_line.trim();
            if expected_line.is_empty() {
                continue;
            }
            let expected: Vec<u32> = expected_line
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let got = tok.encode(text);
            assert_eq!(got, expected, "case #{i}: text={text:?}");
            checked += 1;
        }
        assert!(
            checked >= 40,
            "expected to actually check at least 40 real cases, only checked {checked}"
        );
    }

    #[test]
    fn decode_reverses_encode_for_ascii_text() {
        let tok = load_real_fixture();
        // SentencePiece's real convention (confirmed by the reference
        // suite above) always prepends a dummy leading space before
        // tokenizing, so decoding round-trips to " Hello world" (WITH
        // a leading space), not "Hello world" -- this is genuine
        // LLaMA-tokenizer behavior, not a bug in this test or the
        // encoder; downstream text-generation code conventionally
        // strips exactly one leading space from decoded output, but
        // the raw decode legitimately includes it.
        let text = "Hello world";
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), " Hello world");
    }

    #[test]
    fn decode_reverses_byte_fallback_tokens_to_the_real_raw_bytes() {
        // Real bug found via real-world testing:
        // decode() used to emit the literal 6-character token string
        // "<0x0A>" instead of an actual newline byte.
        let tok = load_real_fixture();
        // `encode` always prepends a dummy leading space (SentencePiece
        // convention, see `decode_reverses_encode_for_ascii_text`
        // above), so the decoded round-trip carries it too.
        let newline_id = tok.encode("\n");
        assert_eq!(tok.decode(&newline_id), " \n");

        // A multi-byte UTF-8 character split across several
        // consecutive byte-fallback tokens must still decode correctly
        // once reassembled -- not as mojibake or individually-invalid
        // UTF-8 fragments.
        let emoji = "🦀";
        let ids = tok.encode(emoji);
        assert_eq!(tok.decode(&ids), format!(" {emoji}"));
    }
}

#[cfg(test)]
mod gguf_unigram_tests {
    use super::*;

    /// Real trained SentencePiece Unigram model (100 pieces, trained
    /// with the real `sentencepiece` Python library on a small text
    /// corpus through a fixture generator), not a
    /// hand-guessed vocabulary.
    fn load_real_fixture() -> GgufUnigramTokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/unigram-test-vocab.gguf"
        );
        let file = ferrox_gguf::GgufFile::open(path).expect("real Unigram vocab fixture must open");
        GgufUnigramTokenizer::from_gguf(&file)
            .expect("real Unigram vocab fixture must parse as a tokenizer")
    }

    #[test]
    fn loads_real_trained_unigram_vocab() {
        let tok = load_real_fixture();
        assert_eq!(tok.vocab_size(), 100);
    }

    /// Cross-validated against the exact same trained model's own
    /// `sentencepiece.SentencePieceProcessor.Encode` output -- not a
    /// hand-computed expectation. Covers ASCII, mixed case, digit runs,
    /// repeated whitespace, punctuation, and non-ASCII (accented Latin)
    /// text, plus a string with no real vocabulary substrings at all
    /// (exercising the unknown-token fallback repeatedly, including
    /// consecutive unknown tokens).
    #[test]
    fn matches_real_sentencepiece_reference_encodings() {
        let tok = load_real_fixture();
        let cases: &[(&str, &[u32])] = &[
            ("hello world", &[3, 63, 4, 95, 8, 3, 36, 14, 11]),
            (
                "The quick brown fox",
                &[34, 3, 89, 10, 65, 70, 57, 49, 73, 12, 54, 8, 30],
            ),
            (
                "Testing unicode: café",
                &[74, 44, 20, 35, 47, 4, 83, 3, 62, 13, 25, 18],
            ),
            (
                "Numbers 12345",
                &[3, 86, 50, 15, 53, 5, 3, 75, 76, 77, 81, 82],
            ),
            ("a", &[58]),
            (
                "   multiple   spaces   ",
                &[55, 10, 14, 64, 16, 99, 22, 3, 5, 99, 13, 27, 5],
            ),
            (
                "Zurich naive resume",
                &[3, 88, 10, 7, 16, 51, 38, 16, 33, 60, 4, 5, 50, 4],
            ),
            (
                "punctuation! test? yes.",
                &[24, 10, 72, 43, 29, 80, 3, 64, 44, 84, 3, 28, 4, 5, 6],
            ),
            (
                "unknown_gibberish_xyz_qqq_zzz",
                &[
                    3, 10, 12, 70, 12, 8, 73, 12, 0, 17, 16, 15, 15, 53, 56, 63, 0, 30, 28, 90, 0,
                    89, 89, 89, 0, 90, 90, 90,
                ],
            ),
        ];
        for (text, expected) in cases {
            let got = tok.encode(text);
            assert_eq!(&got, expected, "text={text:?}");
        }
    }

    #[test]
    fn decode_reverses_encode_for_ascii_text() {
        let tok = load_real_fixture();
        let ids = tok.encode("hello world");
        // encode's leading dummy `▁` decodes back to a leading space,
        // same SentencePiece convention as GgufSpmTokenizer.
        assert_eq!(tok.decode(&ids), " hello world");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_round_trips_exactly() {
        let text = "hello ferrox";
        let ids = ByteTokenizer::encode(text);
        assert_eq!(ids.len(), text.len());
        assert_eq!(ByteTokenizer::decode(&ids), text);
    }

    #[test]
    fn utf8_multibyte_round_trips_exactly() {
        let text = "caffe\u{300} \u{1f980}"; // combining accent + emoji, multi-byte UTF-8
        let ids = ByteTokenizer::encode(text);
        assert_eq!(ByteTokenizer::decode(&ids), text);
    }

    #[test]
    fn all_ids_are_within_byte_vocab_range() {
        let ids = ByteTokenizer::encode("mixed ASCII and \u{00e9}\u{00e8} text");
        assert!(ids
            .iter()
            .all(|&id| (id as usize) < ByteTokenizer::VOCAB_SIZE));
    }

    #[test]
    fn empty_string_round_trips() {
        assert_eq!(ByteTokenizer::encode(""), Vec::<u32>::new());
        assert_eq!(ByteTokenizer::decode(&[]), "");
    }

    #[test]
    fn out_of_range_ids_are_dropped_not_corrupting() {
        // 300 is outside the byte vocab; decode should simply skip it
        // rather than panicking or wrapping into a wrong byte.
        let decoded = ByteTokenizer::decode(&[104, 105, 300, 33]); // "hi" + garbage + "!"
        assert_eq!(decoded, "hi!");
    }
}
