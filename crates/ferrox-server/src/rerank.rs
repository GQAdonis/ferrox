//! `/v1/rerank` (and llama.cpp's `/rerank`): one query, N documents,
//! ordered by the checkpoint's **own** classification head.
//!
//! # Why this is not `/v1/embeddings` with a cosine
//!
//! The obvious cheap implementation — embed the query, embed each
//! document, sort by cosine similarity — is a different model. A
//! bi-encoder embeds the two texts independently and can only compare
//! them afterwards; a cross-encoder reranker reads the pair *together*,
//! `[CLS] query [SEP] document [SEP]`, so every layer's attention runs
//! across the boundary, and it reports a relevance logit from a trained
//! classification head ([`ferrox_models::RankHead`]) rather than an
//! angle between two vectors. Reranking exists precisely because the
//! two disagree. Serving a cosine here would return a plausible
//! ordering under a name that promises the model's, which is this
//! engine's one prohibited failure — so a checkpoint with no head
//! **refuses** (501) instead, and never substitutes a similarity.
//!
//! # What the checkpoint has to be
//!
//! A `bert` GGUF carrying `cls` / `cls.output` (see
//! [`ferrox_models::load_rank_head`]), reached the same two ways an
//! embedding model is: as `FERROX_MODEL_PATH`, or as the
//! `FERROX_EMBEDDING_MODEL_PATH` side-car. Such a checkpoint declares
//! `pooling_type = RANK`, and `/v1/embeddings` against it still refuses
//! in [`ferrox_models::pooling::pool`] — deliberately. RANK is not a
//! pooling rule, and the refusal is what sends the caller here.
//!
//! # The wire shape
//!
//! Jina's and Cohere's, which is what every existing rerank client
//! speaks (OpenAI has no reranker endpoint to be compatible with):
//! `{"query": …, "documents": [...], "top_n": N, "return_documents":
//! bool}` in, `{"results": [{"index": i, "relevance_score": f}]}` out,
//! best first.
//!
//! `index` is the position in the **caller's** `documents` array, not
//! in the sorted output. That is the standing bug in this route shape
//! and it is invisible in the easy cases — a one-document request, or
//! any input that happened to arrive already ordered — so
//! [`ranking`] carries it and is tested directly.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;

use ferrox_models::{EmbedError, EmbeddingModel};

use crate::openai_extra::Call;
use crate::{join_error_response, unsupported_feature, ApiError, AppState};

/// Jina/Cohere's request body. Every field declared here is honoured;
/// a field of another dialect (`rank_fields`, `return_text`) is ignored
/// rather than refused, the same way [`crate::embeddings`] treats one.
#[derive(Debug, Deserialize)]
pub(crate) struct RerankRequest {
    /// Echoed back. Purely cosmetic: this server serves whatever is
    /// loaded, and does not route on a model name.
    #[serde(default)]
    model: Option<String>,
    query: String,
    documents: Vec<String>,
    /// Keep only the best `top_n`. Absent means "all of them".
    #[serde(default)]
    top_n: Option<usize>,
    /// Echo each document's text back inside its result.
    #[serde(default)]
    return_documents: bool,
}

/// Which routes an encoder checkpoint actually answers.
///
/// `/health` and `/v1/models` both tell a client this, and before this
/// route existed both simply said `/v1/embeddings` — true while an
/// encoder was necessarily an embedding model. A reranker checkpoint is
/// an encoder too, and it is the *opposite* case: its `pooling_type` is
/// RANK, which [`ferrox_models::pooling::pool`] refuses, so
/// `/v1/embeddings` answers it with a 400 while `/v1/rerank` serves it.
/// Two reporting surfaces restating one list is this repo's dominant
/// bug shape, so they read it from here instead.
///
/// Each line is derived from the refusal it mirrors, not from a guess
/// about what a "reranker" is: embeddings is listed exactly when `pool`
/// would accept the checkpoint's pooling type, and rerank exactly when
/// there is a head for [`rerank_inner`] to run. A checkpoint that
/// satisfies neither is listed as answering neither, which is true.
pub(crate) fn encoder_endpoints(encoder: &EmbeddingModel) -> Vec<&'static str> {
    let mut out = Vec::with_capacity(2);
    if encoder.pooling_type() != ferrox_models::pooling::PoolingType::Rank {
        out.push(ferrox_api::routes::V1_EMBEDDINGS);
    }
    if encoder.rank_head().is_some() {
        out.push(ferrox_api::routes::V1_RERANK);
    }
    out
}

fn bad_request(message: String) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": { "message": message } })),
    )
}

/// The output order: indices into the caller's `documents`, best first.
///
/// Two properties, and this route shape gets both wrong by default:
///
/// * **The values are the caller's indices.** A client joins the
///   results back onto what it sent, so reporting the position in the
///   sorted list instead silently mislabels every document whenever the
///   input was not already in score order — and it *is* already in
///   score order for a single-document request and for any test that
///   does not deliberately shuffle.
/// * **Ties keep the caller's order.** `sort_by` is stable and equal
///   scores are ordinary here: duplicate documents score identically,
///   and a head whose `tanh` saturates flattens whole runs of them. An
///   unstable sort would make the same request answer differently on
///   different days.
///
/// `total_cmp` rather than `partial_cmp(..).unwrap()`: a total order
/// cannot panic on a score the head produced. Non-finite scores are
/// refused before they get here (see [`finite_score`]), so this is
/// belt-and-braces, not the check.
///
/// Truncation is one rule with no special cases: `top_n` absent keeps
/// everything, `top_n` past the end keeps everything (`Vec::truncate`
/// is a no-op then), `top_n` of 0 keeps nothing.
fn ranking(scores: &[f32], top_n: Option<usize>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    order.truncate(top_n.unwrap_or(order.len()));
    order
}

/// Refuses a score that JSON cannot carry.
///
/// `serde_json` has no NaN and no infinity: `json!(f32::NAN)` is
/// `null`, silently. A client reading `relevance_score` would get a
/// null where it expected a number, or — worse — a JSON parser that
/// coerces null to 0 would rank that document last with no error
/// anywhere. A non-finite score means the head or the encoder went
/// wrong, which is the server's fault and so a 500.
fn finite_score(score: f32, index: usize) -> Result<f32, ApiError> {
    if score.is_finite() {
        return Ok(score);
    }
    Err((
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": {
            "message": format!(
                "the classification head produced a non-finite relevance score ({score}) \
                 for document {index}. JSON cannot carry it, and reporting it as null \
                 would read as a valid score of zero"
            ),
            "type": "internal_error",
        }})),
    ))
}

/// 501 for an embedding model that carries no classification head.
///
/// The sentence itself is [`EmbedError::NoRankHead`], which lives
/// beside the head loader rather than being restated here, and the 501
/// comes from [`unsupported_feature`] — the server's one constructor
/// for "ferrox does not implement this". Both halves are borrowed on
/// purpose: a second wording of either is a second thing to keep true.
fn no_rank_head(name: &str, arch: &str) -> ApiError {
    let why = EmbedError::NoRankHead {
        name: name.to_string(),
        arch: arch.to_string(),
    };
    unsupported_feature(&format!(
        "{why}. POST {} to use it for embeddings, or load a reranker checkpoint",
        ferrox_api::routes::V1_EMBEDDINGS,
    ))
}

/// 501 for a generative model, which is the mirror image of the
/// refusal [`crate::loaded`] gives an encoder on a generation route:
/// same status, same reason (this endpoint is not implemented *for
/// this model*), same rule that it names the checkpoint rather than
/// blaming a tensor.
fn not_a_reranker(model: &str) -> ApiError {
    unsupported_feature(&format!(
        "the loaded model '{model}' is not a reranker. {} needs a cross-encoder checkpoint \
         carrying a classification head (a `bert` GGUF with `cls` / `cls.output` tensors), \
         loaded as FERROX_MODEL_PATH or beside this one as FERROX_EMBEDDING_MODEL_PATH. A \
         generative model cannot answer this route, and the cosine of its embeddings is \
         not a substitute for a rerank score",
        ferrox_api::routes::V1_RERANK,
    ))
}

/// The encoder that can actually answer this route, or the refusal
/// naming what is loaded instead.
///
/// Three different answers, and none of them may be given for another:
/// nothing loaded is 503 (from [`AppState::require_active`], the one
/// place this server says that), a generative model is
/// [`not_a_reranker`], and an embedding model with no head is
/// [`no_rank_head`].
fn require_reranker(state: &AppState) -> Result<Arc<EmbeddingModel>, ApiError> {
    let Some(encoder) = state.embedding_model() else {
        return Err(state.require_active().err().unwrap_or_else(|| {
            not_a_reranker(&state.active_model_name().unwrap_or_else(|| "?".to_string()))
        }));
    };
    if encoder.rank_head().is_none() {
        return Err(no_rank_head(encoder.name(), encoder.architecture()));
    }
    Ok(encoder)
}

/// What a rerank needs before it is worth loading a model for.
///
/// Both refusals run before the encoder is touched, which is the point
/// of them: an empty `documents` has nothing to rank, and an empty
/// `query` would pay one full encoder pass per document to score them
/// all against nothing.
fn validate(query: &str, documents: &[String]) -> Result<(), ApiError> {
    if documents.is_empty() {
        return Err(bad_request(
            "documents must be a non-empty array of strings".to_string(),
        ));
    }
    if query.is_empty() {
        return Err(bad_request(
            "query must be a non-empty string; a rerank scores each document against it"
                .to_string(),
        ));
    }
    Ok(())
}

/// A load-time refusal from the model, given the status it deserves.
///
/// `NoRankHead` / `NoPairInput` are "ferrox cannot do this at all" and
/// stay 501 even though [`require_reranker`] has already ruled the
/// first one out; anything else here is a property of this particular
/// document (too long for the position table, an out-of-range id) and
/// is the caller's 400, named with the document it came from.
fn embed_error(index: usize, e: EmbedError) -> ApiError {
    match e {
        EmbedError::NoRankHead { .. } | EmbedError::NoPairInput { .. } => {
            unsupported_feature(&e.to_string())
        }
        other => bad_request(format!("document {index}: {other}")),
    }
}

/// One encoder pass per document, on the blocking pool.
///
/// Every document is scored even when `top_n` is small, and there is no
/// way around that: the top N cannot be known without scoring all of
/// them. `top_n` truncates the answer, it does not shorten the work.
async fn score_documents(
    encoder: Arc<EmbeddingModel>,
    query: String,
    documents: Arc<Vec<String>>,
) -> Result<(Vec<f32>, usize), ApiError> {
    tokio::task::spawn_blocking(move || {
        let mut scores = Vec::with_capacity(documents.len());
        let mut prompt_tokens = 0usize;
        for (i, doc) in documents.iter().enumerate() {
            let ids = encoder
                .rerank_token_ids(&query, doc)
                .map_err(|e| embed_error(i, e))?;
            prompt_tokens += ids.len();
            let score = encoder.rerank_score(&ids).map_err(|e| embed_error(i, e))?;
            scores.push(finite_score(score, i)?);
        }
        Ok::<_, ApiError>((scores, prompt_tokens))
    })
    .await
    .map_err(join_error_response)?
}

/// The `results` array, in `order`. Pure, so the index property above
/// is testable without a checkpoint.
fn results_json(
    order: &[usize],
    scores: &[f32],
    documents: &[String],
    return_documents: bool,
) -> Vec<serde_json::Value> {
    order
        .iter()
        .map(|&i| {
            let mut row = serde_json::json!({
                "index": i,
                "relevance_score": scores[i],
            });
            if return_documents {
                row["document"] = serde_json::json!({ "text": documents[i] });
            }
            row
        })
        .collect()
}

pub async fn rerank(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<RerankRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let call = Call::new(&headers);
    let result = rerank_inner(&state, req).await;
    // Same accounting rule as `/v1/embeddings`: the encoder passes are
    // real prompt tokens, and there is no decode loop, so the
    // completion half is 0 rather than borrowing from the total.
    let usage = result
        .as_ref()
        .ok()
        .map(|(_, prompt_tokens)| ferrox_api::Usage::new(*prompt_tokens, 0));
    // Recorded under the `/v1` spelling for both mounts, the way
    // `/tokenize` is: one route in the stats ring, two ways to reach it.
    call.record(
        &state,
        ferrox_api::routes::V1_RERANK,
        state.embedding_model_name(),
        &result,
        usage.as_ref(),
    );
    result.map(|(body, _)| Json(body))
}

async fn rerank_inner(
    state: &AppState,
    req: RerankRequest,
) -> Result<(serde_json::Value, usize), ApiError> {
    validate(&req.query, &req.documents)?;
    let encoder = require_reranker(state)?;
    let model_name = req
        .model
        .clone()
        .unwrap_or_else(|| encoder.name().to_string());
    // The label output 0 carries, when the head named its outputs.
    // `RankHead::score` reports `embd[0]` the way upstream's
    // `send_rerank` does, and `load_rank_head` refuses a multi-output
    // head that names nothing -- so whenever there IS more than one
    // output, there is a name for the one being reported, and saying it
    // is cheaper than leaving the choice silent.
    let score_label = encoder
        .rank_head()
        .and_then(|h| h.labels().first().cloned());

    let documents = Arc::new(req.documents);
    let (scores, prompt_tokens) =
        score_documents(encoder, req.query, Arc::clone(&documents)).await?;

    let order = ranking(&scores, req.top_n);
    let mut body = serde_json::json!({
        "object": "list",
        "model": model_name,
        "results": results_json(&order, &scores, &documents, req.return_documents),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        }
    });
    if let Some(label) = score_label {
        body["ferrox_score_label"] = serde_json::json!(label);
    }
    Ok((body, prompt_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("doc{i}")).collect()
    }

    /// **The classic bug in this route shape.** `index` must address
    /// the caller's array, so a client can join the results back onto
    /// what it sent; the position in the sorted output is a different
    /// number that happens to agree whenever the input arrived already
    /// ordered. The fixture is deliberately out of order in both
    /// directions so neither reading passes by accident.
    #[test]
    fn the_index_is_the_position_in_the_request_not_in_the_sorted_output() {
        let scores = [0.1f32, 0.9, 0.5];
        assert_eq!(ranking(&scores, None), vec![1, 2, 0]);
        let rows = results_json(&ranking(&scores, None), &scores, &docs(3), false);
        assert_eq!(rows[0]["index"], 1);
        assert_eq!(rows[1]["index"], 2);
        assert_eq!(rows[2]["index"], 0);
        // And the score travels with its own document, not with its
        // rank: row 0 is document 1, so it carries document 1's score.
        assert_eq!(rows[0]["relevance_score"], 0.9f32);
        assert_eq!(rows[2]["relevance_score"], 0.1f32);
    }

    /// Equal scores are ordinary (duplicate documents, a saturating
    /// `tanh`), so the tie-break has to be defined. It is the caller's
    /// own order, which needs a STABLE sort -- an unstable one would
    /// answer the same request differently on different days.
    ///
    /// The fixture is 64 documents over two distinct scores on purpose.
    /// A four-element version of this test passes with
    /// `sort_unstable_by` as well, because Rust's pattern-defeating
    /// quicksort insertion-sorts anything under ~20 elements and is
    /// accidentally stable there -- so the small version asserts the
    /// property on paper and enforces nothing. Above that threshold the
    /// real algorithm runs and permutes the tie groups.
    #[test]
    fn ties_keep_the_order_the_caller_sent() {
        // Odd indices score 1.0, even indices 0.0: two tie groups of 32.
        let scores: Vec<f32> = (0..64).map(|i| (i % 2) as f32).collect();
        // The highs in the caller's order, then the lows in the
        // caller's order. Any permutation inside either group is the
        // failure this is here for.
        let expected: Vec<usize> = (0..64)
            .filter(|i| i % 2 == 1)
            .chain((0..64).filter(|i| i % 2 == 0))
            .collect();
        assert_eq!(ranking(&scores, None), expected);
    }

    /// A `top_n` past the end is a clamp, not an error and not a panic:
    /// asking for more than exists returns everything that exists.
    #[test]
    fn a_top_n_larger_than_the_document_list_returns_every_document() {
        let scores = [0.1f32, 0.9];
        assert_eq!(ranking(&scores, Some(1000)), vec![1, 0]);
        assert_eq!(ranking(&scores, Some(2)), vec![1, 0]);
        assert_eq!(ranking(&scores, None), vec![1, 0]);
    }

    /// `top_n: 0` is the same truncation rule with a smaller argument:
    /// the top nothing is nothing. It is NOT read as "unset" -- that
    /// would silently return every document to a caller who asked for
    /// none.
    #[test]
    fn a_top_n_of_zero_returns_no_results() {
        let scores = [0.1f32, 0.9, 0.5];
        assert!(ranking(&scores, Some(0)).is_empty());
        assert!(results_json(&ranking(&scores, Some(0)), &scores, &docs(3), true).is_empty());
    }

    /// An empty `documents` array is refused rather than answered with
    /// an empty list: there is nothing to rank, and a 200 saying so is
    /// indistinguishable from "your documents all scored badly". An
    /// empty query is refused for the mirror reason -- it would pay one
    /// full encoder pass per document to score them all against
    /// nothing.
    #[test]
    fn an_empty_documents_array_and_an_empty_query_are_refused_by_name() {
        assert!(validate("q", &docs(1)).is_ok());
        // An empty document is a document: it is the LIST that must be
        // non-empty, not each string in it.
        assert!(validate("q", &[String::new()]).is_ok());

        let (status, body) = validate("q", &[]).unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.0["error"]["message"]
            .as_str()
            .unwrap()
            .contains("documents"));

        let (status, body) = validate("", &docs(1)).unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.0["error"]["message"]
            .as_str()
            .unwrap()
            .contains("query"));
    }

    /// A checkpoint with no classification head must be told apart
    /// from a decoder, and both from "nothing is loaded". Each refusal
    /// names the model it is about -- an operator reading "unsupported"
    /// alone goes hunting a missing tensor, which is the exact failure
    /// the encoder seam in `crate::loaded` already exists to prevent.
    #[test]
    fn the_two_501s_name_the_model_and_the_route_that_would_serve_it() {
        let (status, body) = no_rank_head("bge-small-en-v1.5", "bert");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let msg = body.0["error"]["message"].as_str().unwrap().to_string();
        for fact in ["bge-small-en-v1.5", "bert", "cls", "/v1/embeddings"] {
            assert!(msg.contains(fact), "{msg} does not carry {fact}");
        }

        let (status, body) = not_a_reranker("llama-3.2-3b");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let msg = body.0["error"]["message"].as_str().unwrap().to_string();
        for fact in ["llama-3.2-3b", "cross-encoder", "/v1/rerank"] {
            assert!(msg.contains(fact), "{msg} does not carry {fact}");
        }
        // 501 and not 503: no amount of retrying turns either of these
        // into a reranker, and a supervisor acts on that difference.
        assert_ne!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    /// `serde_json` has no NaN: it serializes as `null`, which a client
    /// reads as a missing score or coerces to 0. A head that produces
    /// one is broken, and saying so is the only honest answer.
    #[test]
    fn a_non_finite_score_is_refused_rather_than_serialized_as_null() {
        // The trap this guards, stated as an assertion so it cannot
        // quietly stop being true.
        assert!(serde_json::json!(f32::NAN).is_null());
        assert!(serde_json::json!(f32::INFINITY).is_null());

        assert_eq!(finite_score(0.0, 0).unwrap(), 0.0);
        assert_eq!(finite_score(-12.5, 0).unwrap(), -12.5);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let (status, body) = finite_score(bad, 3).unwrap_err();
            assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
            let msg = body.0["error"]["message"].as_str().unwrap();
            assert!(msg.contains("document 3"), "{msg}");
        }
    }

    /// `return_documents` echoes the text of the document each row
    /// refers to -- which is the index property again, from the other
    /// side: row 0 must carry document 1's text, not document 0's.
    #[test]
    fn return_documents_echoes_the_row_s_own_document_and_omitting_it_omits_the_key() {
        let scores = [0.1f32, 0.9, 0.5];
        let order = ranking(&scores, None);
        let rows = results_json(&order, &scores, &docs(3), true);
        assert_eq!(rows[0]["document"]["text"], "doc1");
        assert_eq!(rows[2]["document"]["text"], "doc0");

        let bare = results_json(&order, &scores, &docs(3), false);
        assert!(bare[0].get("document").is_none());
    }
}
