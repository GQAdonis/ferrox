//! Extra OpenAI-shaped endpoints: tokenize / detokenize, embeddings
//! (GGUF Decoder hidden-state pool), and legacy `/v1/completions`.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use ferrox_models::sampling::SamplingParams;
use serde::Deserialize;

use crate::attribution::Attribution;
use crate::generate::{FinishReason, GenerationParams};
use crate::{
    decode_error_response, join_error_response, run_generation, unsupported_feature, ApiError,
    AppState,
};

/// What one of the small endpoints knows before it does any work.
///
/// Captured at entry so the ring entry can be written from the same
/// facts however the handler ends -- and so the three of them travel
/// together instead of as three positional arguments that are each
/// easy to pass in the wrong order.
struct Call {
    request_id: String,
    started: std::time::Instant,
    attribution: Attribution,
}

impl Call {
    fn new(headers: &axum::http::HeaderMap) -> Self {
        Call {
            request_id: ferrox_api::next_request_id(),
            started: std::time::Instant::now(),
            attribution: Attribution::from_headers(headers),
        }
    }

    /// Records a call that reached a response. Spelled out rather than
    /// left as a `&Ok(())` at the call site, which reads like a
    /// mistake.
    fn record_success(
        &self,
        state: &AppState,
        route: &str,
        model: Option<String>,
        usage: Option<&ferrox_api::Usage>,
    ) {
        self.record(state, route, model, &Ok::<(), ApiError>(()), usage);
    }

    /// Records this call in the `/admin/stats` ring, with the status
    /// the caller actually saw.
    ///
    /// These endpoints used not to be recorded at all, which made the
    /// monitor quietly wrong rather than merely incomplete: an editor
    /// hammering `/v1/embeddings` showed up as an idle server. A
    /// failure is recorded with its own status for the same reason -- a
    /// 400 that leaves no trace is indistinguishable from a request
    /// that was never sent.
    fn record<T>(
        &self,
        state: &AppState,
        route: &str,
        model: Option<String>,
        result: &Result<T, ApiError>,
        usage: Option<&ferrox_api::Usage>,
    ) {
        let status = match result {
            Ok(_) => 200,
            Err((code, _)) => code.as_u16(),
        };
        state.record_request(crate::stats::Record {
            request_id: &self.request_id,
            route,
            model,
            status,
            stream: false,
            duration_ms: self.started.elapsed().as_millis() as u64,
            usage: result.is_ok().then_some(usage).flatten(),
            attribution: &self.attribution,
        });
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct TokenizeRequest {
    prompt: String,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DetokenizeRequest {
    tokens: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingsRequest {
    input: EmbeddingInput,
    #[serde(default)]
    model: Option<String>,
    /// Only `"float"` is supported (OpenAI also has `base64`).
    #[serde(default)]
    encoding_format: Option<String>,
    /// Pooling over token hidden states: `mean` (default) or `last`.
    #[serde(default)]
    embedding_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CompletionsRequest {
    /// Accepted as a `Value` rather than a `String` so a token-id prompt
    /// -- `[int]` or `[[int]]`, which OpenAI allows and this server does
    /// not implement -- is REFUSED BY NAME rather than dying as a serde
    /// type error the caller has to guess at.
    prompt: serde_json::Value,
    /// The legacy 16 is right *here*, and only here: a caller asking to
    /// complete a fragment usually wants a fragment back. The chat and
    /// Responses surfaces deliberately keep it out (see
    /// `DEFAULT_CHAT_MAX_TOKENS`).
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    /// Honoured. Silently dropping this is the dangerous one: the caller
    /// believes generation halts at their sentinel and instead gets the
    /// full budget of text past it.
    #[serde(default)]
    stop: Option<StopParam>,
    // Fields this server does not implement. Deserialized ONLY so they
    // can be refused by name -- serde would otherwise drop each one
    // silently, which is indistinguishable from having honoured it.
    #[serde(default)]
    logprobs: Option<serde_json::Value>,
    #[serde(default)]
    echo: Option<bool>,
    #[serde(default)]
    suffix: Option<String>,
    #[serde(default)]
    logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
}

/// `stop` as OpenAI sends it: one string or a list of them.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum StopParam {
    One(String),
    Many(Vec<String>),
}

impl CompletionsRequest {
    /// The prompt text, or a named refusal.
    ///
    /// A token-id prompt is a real OpenAI shape and a real thing this
    /// server cannot serve; saying so beats a deserialization error, and
    /// beats silently treating `[1,2,3]` as the string it is not.
    fn prompt_text(&self) -> Result<&str, ApiError> {
        match &self.prompt {
            serde_json::Value::String(s) => Ok(s),
            serde_json::Value::Array(_) => Err(crate::unsupported_feature(
                "token-id prompts are not implemented; send `prompt` as a string",
            )),
            _ => Err(crate::invalid_request("prompt must be a string", "prompt")),
        }
    }

    fn stop_sequences(&self) -> Vec<String> {
        match &self.stop {
            Some(StopParam::One(s)) => vec![s.clone()],
            Some(StopParam::Many(v)) => v.clone(),
            None => Vec::new(),
        }
    }

    /// Refuse what this server does not implement, by name.
    fn validate(&self) -> Result<(), ApiError> {
        let unsupported = [
            (self.logprobs.is_some(), "logprobs"),
            (self.echo == Some(true), "echo"),
            (self.suffix.is_some(), "suffix"),
            (self.logit_bias.is_some(), "logit_bias"),
        ];
        for (present, name) in unsupported {
            if present {
                return Err(crate::unsupported_feature(&format!(
                    "`{name}` is not implemented on /v1/completions (see docs/API.md)"
                )));
            }
        }
        // `{"type": "text"}` is the default and means nothing to refuse.
        if let Some(fmt) = &self.response_format {
            let kind = fmt.get("type").and_then(|v| v.as_str());
            if kind != Some("text") {
                return Err(crate::unsupported_feature(
                    "only `response_format: {\"type\": \"text\"}` is implemented on \
                     /v1/completions (see docs/API.md)",
                ));
            }
        }
        Ok(())
    }
}

fn default_max_tokens() -> usize {
    16
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let call = Call::new(&headers);
    let result = tokenize_inner(&state, req);
    // No `usage`, deliberately. Tokenizing runs the tokenizer and not
    // the model, and `prompt_tokens` here feeds `tokens_prompt_total`,
    // which means "tokens this server put through a forward pass".
    // Counting a tokenize call into it would inflate every throughput
    // number derived from it.
    call.record(
        &state,
        ferrox_api::routes::V1_TOKENIZE,
        state.active_model_name(),
        &result,
        None,
    );
    result
}

fn tokenize_inner(
    state: &AppState,
    req: TokenizeRequest,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = req.model;
    // Tokenizing needs the loaded vocabulary, so this is a 503 like any
    // other generation endpoint when nothing is loaded -- answering
    // with byte-fallback ids would silently be the wrong vocabulary.
    let tokens = state.require_model()?.encode(&req.prompt);
    let count = tokens.len();
    Ok(Json(serde_json::json!({
        "tokens": tokens,
        "count": count,
    })))
}

pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<DetokenizeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let call = Call::new(&headers);
    let result = detokenize_inner(&state, req);
    // Same reasoning as `tokenize`: no forward pass, so no usage.
    call.record(
        &state,
        ferrox_api::routes::V1_DETOKENIZE,
        state.active_model_name(),
        &result,
        None,
    );
    result
}

fn detokenize_inner(
    state: &AppState,
    req: DetokenizeRequest,
) -> Result<Json<serde_json::Value>, ApiError> {
    let text = state.require_model()?.decode(&req.tokens);
    Ok(Json(serde_json::json!({ "text": text })))
}

fn pool_hidden(hiddens: &[Vec<f32>], pooling: &str) -> Result<Vec<f32>, ApiError> {
    if hiddens.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {"message": "input encoded to zero tokens; cannot embed empty sequence"}
            })),
        ));
    }
    match pooling {
        "last" => Ok(hiddens.last().unwrap().clone()),
        "mean" => {
            let dim = hiddens[0].len();
            let mut acc = vec![0.0f32; dim];
            for h in hiddens {
                for (a, &v) in acc.iter_mut().zip(h.iter()) {
                    *a += v;
                }
            }
            let n = hiddens.len() as f32;
            for a in &mut acc {
                *a /= n;
            }
            Ok(acc)
        }
        other => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "embedding_type must be \"mean\" or \"last\", got {other:?}"
                    )
                }
            })),
        )),
    }
}

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let call = Call::new(&headers);
    let result = embeddings_inner(&state, req).await;
    // Embeddings *do* run the model, so the prompt tokens they paid for
    // are real prompt tokens and are recorded as such. There is no
    // decode loop, so `decode_ms` stays null rather than borrowing the
    // total -- the same rule the two duration columns exist for.
    let usage = result
        .as_ref()
        .ok()
        .map(|(_, prompt_tokens)| ferrox_api::Usage::new(*prompt_tokens, 0));
    call.record(
        &state,
        ferrox_api::routes::V1_EMBEDDINGS,
        state.active_model_name(),
        &result,
        usage.as_ref(),
    );
    result.map(|(body, _)| Json(body))
}

async fn embeddings_inner(
    state: &AppState,
    req: EmbeddingsRequest,
) -> Result<(serde_json::Value, usize), ApiError> {
    if let Some(fmt) = req.encoding_format.as_deref() {
        if fmt != "float" {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!(
                            "encoding_format {fmt:?} is not supported (only \"float\")"
                        )
                    }
                })),
            ));
        }
    }
    let pooling = req.embedding_type.as_deref().unwrap_or("mean");
    if !matches!(pooling, "mean" | "last") {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "embedding_type must be \"mean\" or \"last\", got {pooling:?}"
                    )
                }
            })),
        ));
    }

    let active_model = state.require_model()?;
    // Fail fast for non-GGUF engines before paying encode cost.
    if active_model.embed_tokens(&[]).is_none() {
        return Err(unsupported_feature(
            "embeddings engine not yet available for this model",
        ));
    }

    let inputs: Vec<String> = match req.input {
        EmbeddingInput::One(s) => vec![s],
        EmbeddingInput::Many(v) => v,
    };
    if inputs.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {"message": "input must be a non-empty string or array of strings"}
            })),
        ));
    }

    let model = Arc::clone(&active_model);
    let pooling = pooling.to_string();
    let (data, prompt_tokens) = tokio::task::spawn_blocking(move || {
        let mut out = Vec::with_capacity(inputs.len());
        let mut prompt_tokens = 0usize;
        for (i, text) in inputs.iter().enumerate() {
            let tokens = model.encode(text);
            prompt_tokens += tokens.len();
            if let Some(vocab) = model.vocab_size() {
                if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab) {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": {
                                "message": format!(
                                    "token id {bad} is outside this model's vocabulary of {vocab}"
                                )
                            }
                        })),
                    ));
                }
            }
            let hiddens = model.embed_tokens(&tokens).ok_or_else(|| {
                unsupported_feature("embeddings engine not yet available for this model")
            })?;
            let embedding = pool_hidden(&hiddens, &pooling)?;
            out.push(serde_json::json!({
                "object": "embedding",
                "index": i,
                "embedding": embedding,
            }));
        }
        Ok::<_, ApiError>((out, prompt_tokens))
    })
    .await
    .map_err(join_error_response)??;

    let model_name = req.model.unwrap_or_else(|| active_model.name().to_string());
    Ok((
        serde_json::json!({
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            }
        }),
        prompt_tokens,
    ))
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<CompletionsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let call = Call::new(&headers);
    crate::cache_admin::check_admission(&state)?;
    req.validate()?;
    let prompt = req.prompt_text()?.to_string();
    let active = state.require_active()?;
    let params = GenerationParams {
        max_tokens: req.max_tokens,
        sampling: SamplingParams {
            temperature: req.temperature.unwrap_or(0.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        seed: req.seed.unwrap_or(0),
        stop: req.stop_sequences(),
        json_object: false,
        // `/v1/completions` is buffered rather than streamed here, so
        // there is no first chunk on which to state a request id and
        // nothing for a client to name in a cancel.
        stop_token_ids: Vec::new(),
        cancel: None,
    };
    let model = Arc::clone(&active.model);
    let kv_pool = state.kv_pool.clone();
    let prefix_cache = state.prefix_cache.clone();
    let batcher = active.batcher.clone();
    let ceiling = active.ceiling.clone();

    let (chunks, finish, usage) = tokio::task::spawn_blocking(move || {
        run_generation(
            &model,
            &prompt,
            &params,
            kv_pool.as_ref(),
            prefix_cache.as_deref(),
            batcher.as_ref(),
            ceiling.as_deref(),
        )
    })
    .await
    .map_err(join_error_response)?
    .map_err(decode_error_response)?;

    let text = chunks.concat();
    let finish_reason = match finish {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        // Unreachable today -- this path passes `cancel: None` -- but
        // written out rather than defaulted so that wiring cancellation
        // into `/v1/completions` later is a compile error here first,
        // instead of a completion silently reported as finished.
        FinishReason::Cancelled => "cancelled",
    };
    let model_name = req.model.unwrap_or_else(|| active.model.name().to_string());
    call.record_success(
        &state,
        ferrox_api::routes::V1_COMPLETIONS,
        Some(active.model.name().to_string()),
        Some(&usage),
    );

    Ok(Json(serde_json::json!({
        "id": call.request_id,
        "object": "text_completion",
        "model": model_name,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: serde_json::Value) -> CompletionsRequest {
        serde_json::from_value(value).expect("request")
    }

    /// The dangerous silent drop. A caller who sets `stop` believes
    /// generation halts at their sentinel; dropping it hands them the
    /// full budget of text past it instead.
    #[test]
    fn a_completion_stop_string_reaches_the_generation_params() {
        let one = request(serde_json::json!({"prompt": "hi", "stop": "END"}));
        assert_eq!(one.stop_sequences(), vec!["END".to_string()]);

        let many = request(serde_json::json!({"prompt": "hi", "stop": ["A", "B"]}));
        assert_eq!(
            many.stop_sequences(),
            vec!["A".to_string(), "B".to_string()]
        );

        let none = request(serde_json::json!({"prompt": "hi"}));
        assert!(none.stop_sequences().is_empty());
    }

    /// Each of these is a real OpenAI field this server does not
    /// implement. Serde drops an undeclared field silently, which a
    /// caller cannot tell apart from having had it honoured -- so each
    /// is declared purely in order to be refused BY NAME.
    #[test]
    fn every_unimplemented_completion_field_is_refused_by_name() {
        for (field, value) in [
            ("logprobs", serde_json::json!(5)),
            ("echo", serde_json::json!(true)),
            ("suffix", serde_json::json!("tail")),
            ("logit_bias", serde_json::json!({"5": -100})),
        ] {
            let mut body = serde_json::json!({"prompt": "hi"});
            body[field] = value;
            let (status, payload) = request(body)
                .validate()
                .expect_err(&format!("`{field}` must be refused"));
            assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
            assert!(
                payload["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(field),
                "the refusal must name `{field}`"
            );
        }
    }

    /// `{"type": "text"}` is the default and means nothing to refuse;
    /// anything else is a promise this endpoint cannot keep.
    #[test]
    fn only_the_text_response_format_is_accepted() {
        let text =
            request(serde_json::json!({"prompt": "hi", "response_format": {"type": "text"}}));
        assert!(text.validate().is_ok());

        let json = request(
            serde_json::json!({"prompt": "hi", "response_format": {"type": "json_object"}}),
        );
        assert!(json.validate().is_err());
    }

    /// A token-id prompt is a real OpenAI shape. Saying so beats a
    /// deserialization error the caller has to guess at, and beats
    /// treating `[1,2,3]` as a string it is not.
    #[test]
    fn a_token_id_prompt_is_refused_by_name_rather_than_mis_parsed() {
        let ids = request(serde_json::json!({"prompt": [1, 2, 3]}));
        let (status, payload) = ids.prompt_text().expect_err("refused");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(payload["error"]["message"]
            .as_str()
            .unwrap()
            .contains("token-id"));

        let text = request(serde_json::json!({"prompt": "hi"}));
        assert_eq!(text.prompt_text().expect("a string prompt"), "hi");
    }
}
