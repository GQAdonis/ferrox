//! `/v1/rerank`'s model half, on a real cross-encoder checkpoint, with
//! the ORDER checked against an independent implementation.
//!
//! # Why this file exists
//!
//! The rerank route (PR #42) shipped assembled from pieces that were
//! each unit-tested and never once run together, because there was no
//! reranker checkpoint on the development machine (issue #43). "It
//! returns 200" is not evidence for a route whose entire contract is
//! that the ordering is right, and a wrong ordering is invisible: it
//! produces a ranking, just not the model's.
//!
//! Running it end to end found two defects that green CI could not:
//!
//! 1. **The checkpoint would not load at all.** `cls.output.weight` is
//!    a `1 x 384` projection, and GGUF stores it with `n_dims = 1`
//!    because ggml's trailing dimensions are implicitly 1. ferrox's
//!    `load_weight_matrix` demanded two, so every reranker died with
//!    `UnsupportedDtype("cls.output.weight (expected 2D, got shape
//!    [384])")`.
//! 2. **The ordering was wrong** — see
//!    [`scoring_both_halves_as_segment_zero_ranks_the_relevant_document_last`],
//!    which is the old behaviour, preserved as a test so the fix cannot
//!    silently revert. That is issue #44: the document half of the pair
//!    was embedded as "Sentence A".
//!
//! # Getting the checkpoint
//!
//! ```text
//! ferrox download sinjab/ms-marco-MiniLM-L6-v2-Q8_0-GGUF --local-dir models
//! cargo test -p ferrox-models --test rerank_cross_encoder_ordering -- --ignored --nocapture
//! ```
//!
//! 24 MB, `bert` / WordPiece, six layers, `n_embd = 384`, one output
//! labelled `LABEL_0`. `bge-reranker-*` cannot stand in for it: those
//! are XLM-R with a SentencePiece vocabulary and refuse at
//! `EmbedError::UnsupportedTokenizer`.
//!
//! **These tests FAIL rather than skip when the file is absent.** The
//! repo's other checkpoint tests print `SKIP` and return, which passes
//! in 0.00 s — and a `--ignored` run that silently passes is exactly
//! how a route ships unverified twice. `#[ignore]` already means "only
//! when asked"; asking and getting nothing is a failure.
//!
//! # Where the expected numbers come from
//!
//! `scripts/rerank_reference_ms_marco.py` — a NumPy transcription of
//! HuggingFace `BertForSequenceClassification` read straight from the
//! checkpoint's safetensors, with no `transformers` model classes and
//! no torch, so it is a genuinely second implementation rather than a
//! recording of ferrox's own output.
//!
//! # The one place ferrox does NOT match HuggingFace, deliberately
//!
//! llama.cpp's converter drops `bert.pooler.dense` ("we are only using
//! BERT for embeddings so we don't need the pooling layer",
//! `conversion/bert.py`), so the GGUF does not contain it and the head
//! runs as `classifier(cls_hidden)` instead of
//! `classifier(tanh(pooler(cls_hidden)))`. No engine can apply a tensor
//! that is not in the file. The measured effect is on the score SCALE
//! (about +-0.2 rather than about +-11), not on which document wins;
//! the reference script prints both columns so the difference stays
//! visible. The assertions below are against the column ferrox can
//! actually reach, plus the *ordering* of the full HuggingFace
//! reference.

use std::path::{Path, PathBuf};

use ferrox_models::{pool, EmbeddingModel, PoolingType};

/// The Q8_0 conversion of `cross-encoder/ms-marco-MiniLM-L6-v2`.
///
/// Absence is a failure, not a skip: see the module docs.
fn checkpoint() -> PathBuf {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ms-marco-MiniLM-L6-v2-Q8_0.gguf");
    assert!(
        path.exists(),
        "{} is missing. These tests are `#[ignore]`d precisely so that running them means \
         you want them to run, so this fails instead of passing in 0.00s.\n    \
         ferrox download sinjab/ms-marco-MiniLM-L6-v2-Q8_0-GGUF --local-dir models\n\
         (a git worktree has no models/ of its own — symlink the one in the main checkout)",
        path.display()
    );
    path
}

/// The query, and documents in an order that is NOT the answer.
///
/// Document 1 answers the question and documents 2 and 3 are about
/// something else entirely, so "the relevant one ranks first" is a
/// property of the model and not of the fixture. Document 4 is loosely
/// on-topic and document 0 is about the right city and the wrong thing,
/// which is what distinguishes a reranker from a keyword match.
const QUERY: &str = "How many people live in Berlin?";
const DOCUMENTS: [&str; 5] = [
    "Berlin is well known for its museums.",
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "The capital of France is Paris.",
    "Elephants are the largest land animals.",
    "Berlin is the capital and largest city of Germany by both area and population.",
];

/// `scripts/rerank_reference_ms_marco.py`, `ferrox` row: the NumPy
/// reference restricted to what the GGUF carries (no pooler, real
/// segment ids).
const REFERENCE_SCORES: [f32; 5] = [-0.027210, 0.073057, -0.172285, -0.245995, 0.010248];

/// `scripts/rerank_reference_ms_marco.py`, `hf` row: the ordering the
/// checkpoint's full HuggingFace head produces. ferrox must reproduce
/// this ORDER even though it cannot reproduce those scores.
const REFERENCE_ORDER: [usize; 5] = [1, 4, 0, 2, 3];

/// Q8_0 weights against an f64 reference. The largest deviation
/// measured across all four query sets in the script is 1.7e-3.
const TOLERANCE: f32 = 4e-3;

fn ranking(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    order
}

/// The load itself, which is where this route died before anything
/// could be said about its ordering.
///
/// Three claims in one, because each was broken or untested:
///
/// * a `1 x n_embd` `cls.output.weight` stored with `n_dims = 1` loads;
/// * `load_rank_head` runs before `load_bert_encoder`, so
///   `assert_every_tensor_consumed` does not reject `cls.*` — the
///   ordering carried a load-bearing comment and no test;
/// * the head is the checkpoint's own, with its declared label.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn a_real_reranker_checkpoint_loads_with_its_classification_head() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    assert_eq!(model.architecture(), "bert");
    assert_eq!(model.n_embd(), 384);
    let head = model
        .rank_head()
        .expect("a reranker checkpoint with no head is not a reranker");
    assert_eq!(head.n_cls_out(), 1);
    assert_eq!(head.labels(), ["LABEL_0"]);
}

/// The input the head was trained on: `[CLS] query [SEP] document
/// [SEP]`, with the query half segment 0 and the document half segment
/// 1 — `tokenizer(query, document)`'s `token_type_ids`.
///
/// Checked against ids this checkpoint's own vocabulary produces, so a
/// tokenizer change that silently drops the boundary shows up here and
/// not as a slightly worse ranking.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn the_pair_is_cls_query_sep_document_sep_with_the_document_on_segment_one() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    let pair = model.rerank_input(QUERY, DOCUMENTS[0]).expect("pair input");

    // [CLS] how many people live in berlin ? [SEP] berlin is well known
    // for its museums . [SEP]
    assert_eq!(
        pair.tokens,
        vec![
            101, 2129, 2116, 2111, 2444, 1999, 4068, 1029, 102, 4068, 2003, 2092, 2124, 2005, 2049,
            9941, 1012, 102
        ]
    );
    assert_eq!(
        pair.segments,
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    );
    assert_eq!(pair.tokens.len(), pair.segments.len());
    // The boundary is the first [SEP] and it belongs to the query half.
    assert_eq!(pair.tokens[8], 102);
    assert_eq!(pair.segments[8], 0);
    assert_eq!(pair.segments[9], 1);
}

/// **The point of the whole file.** The document that answers the
/// question must rank FIRST, and the assertion is on the INDEX, not on
/// a score threshold: a score says nothing without the others to
/// compare it against, and it is the index a client joins its own array
/// back onto.
///
/// Also asserts the scores themselves against the NumPy reference. That
/// is what rules out "it ranks plausibly for the wrong reason": a
/// cosine similarity, or a head that ran on the wrong row, would still
/// put document 1 near the top on a fixture this easy.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn the_relevant_document_ranks_first_and_the_scores_match_the_numpy_reference() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    let scores: Vec<f32> = DOCUMENTS
        .iter()
        .map(|d| {
            let pair = model.rerank_input(QUERY, d).expect("pair input");
            model.rerank_score(&pair).expect("score")
        })
        .collect();

    let order = ranking(&scores);
    assert_eq!(
        order[0],
        1,
        "the document that answers the query ranked {} of {}: {scores:?}",
        order.iter().position(|&i| i == 1).unwrap() + 1,
        DOCUMENTS.len()
    );
    assert_eq!(
        order,
        REFERENCE_ORDER.to_vec(),
        "ferrox ordered {order:?}, HuggingFace {REFERENCE_ORDER:?}, from {scores:?}"
    );

    for (i, (got, want)) in scores.iter().zip(REFERENCE_SCORES).enumerate() {
        assert!(
            (got - want).abs() < TOLERANCE,
            "document {i}: ferrox {got}, NumPy reference {want}"
        );
    }
    // A relevance logit, not a similarity: it is signed and it is not
    // confined to -1..1, so nothing here could be a cosine standing in
    // for the head.
    assert!(scores.iter().any(|s| *s < 0.0) && scores.iter().any(|s| *s > 0.0));
}

/// The number `/v1/rerank` reports is the classification head's output
/// applied to the CLS row, and this rebuilds that composition from its
/// parts to prove it.
///
/// `rerank_score` is one call; this is `encode`, then CLS pooling, then
/// [`ferrox_models::RankHead::score`], assembled here. If they disagree,
/// the route is reporting something other than the head — which is the
/// failure the whole route was written to avoid, and the reason
/// `/v1/embeddings` refuses a RANK checkpoint rather than serving a
/// cosine.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn the_reported_score_is_the_head_run_on_the_cls_row() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    let head = model.rank_head().expect("head is present");

    let pair = model.rerank_input(QUERY, DOCUMENTS[1]).expect("pair input");
    let hidden = model.pair_hidden_states(&pair).expect("hidden states");
    assert_eq!(hidden.len(), pair.tokens.len() * model.n_embd());
    let cls = pool(&hidden, model.n_embd(), PoolingType::Cls).expect("cls row");
    let by_hand = head.score(&cls);

    let by_route = model.rerank_score(&pair).expect("score");
    assert_eq!(
        by_hand, by_route,
        "the score the model reports is not the head applied to the CLS row"
    );
    // The head is doing real work: 384 floats in, one out, and it is
    // not the CLS row's own first element passed through.
    assert_eq!(cls.len(), 384);
    assert_ne!(by_route, cls[0]);
}

/// **The old behaviour, kept as a test so the fix cannot revert.**
///
/// Before issue #44 was fixed, `encode` added row 0 of
/// `token_types.weight` at every position, matching llama.cpp — whose
/// `llama_batch` carries no segment ids, so its `bert.cpp` views
/// `type_embd` at offset 0 and says token types are hardcoded to zero.
/// #44 recorded that as a defensible shared deviation and asked for it
/// to be MEASURED before being changed. Measured, on this checkpoint,
/// it puts the document that answers the question DEAD LAST out of
/// five, and it does the same in two of the other three query sets in
/// `scripts/rerank_reference_ms_marco.py`.
///
/// So the deviation is not a scale difference, it is a different
/// ranking, and this asserts the size of what was fixed rather than
/// leaving "it changes the order" as a claim in a commit message.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn scoring_both_halves_as_segment_zero_ranks_the_relevant_document_last() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    let head = model.rank_head().expect("head is present");

    let segment_blind: Vec<f32> = DOCUMENTS
        .iter()
        .map(|d| {
            let mut pair = model.rerank_input(QUERY, d).expect("pair input");
            // The old graph, expressed as data rather than as a second
            // copy of the forward pass: every position on "Sentence A".
            pair.segments.fill(0);
            let hidden = model.pair_hidden_states(&pair).expect("hidden states");
            let cls = pool(&hidden, model.n_embd(), PoolingType::Cls).expect("cls row");
            head.score(&cls)
        })
        .collect();

    let order = ranking(&segment_blind);
    assert_eq!(
        *order.last().unwrap(),
        1,
        "the segment-blind graph no longer ranks the relevant document last ({order:?} from \
         {segment_blind:?}); if the reference changed, re-run \
         scripts/rerank_reference_ms_marco.py before relaxing this"
    );
    assert_ne!(order, REFERENCE_ORDER.to_vec());
}
