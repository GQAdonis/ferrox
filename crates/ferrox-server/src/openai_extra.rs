//! Extra OpenAI-shaped endpoints: tokenize / detokenize, embeddings
//! (GGUF Decoder hidden-state pool), and legacy `/v1/completions`.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use ferrox_models::sampling::SamplingParams;
use serde::Deserialize;

use crate::generate::{FinishReason, GenerationParams};
use crate::{
    decode_error_response, join_error_response, run_generation, unsupported_feature, ApiError,
    AppState,
};

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
    prompt: String,
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
}

fn default_max_tokens() -> usize {
    16
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> Json<serde_json::Value> {
    let _ = req.model;
    let tokens = state.model.encode(&req.prompt);
    let count = tokens.len();
    Json(serde_json::json!({
        "tokens": tokens,
        "count": count,
    }))
}

pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DetokenizeRequest>,
) -> Json<serde_json::Value> {
    let text = state.model.decode(&req.tokens);
    Json(serde_json::json!({ "text": text }))
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
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
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

    // Fail fast for non-GGUF engines before paying encode cost.
    if state.model.embed_tokens(&[]).is_none() {
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

    let model = Arc::clone(&state.model);
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

    let model_name = req
        .model
        .unwrap_or_else(|| state.model.name().to_string());
    Ok(Json(serde_json::json!({
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        }
    })))
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let params = GenerationParams {
        max_tokens: req.max_tokens,
        sampling: SamplingParams {
            temperature: req.temperature.unwrap_or(0.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: 0,
            repetition_penalty: 1.0,
        },
        seed: req.seed.unwrap_or(0),
        stop: Vec::new(),
    };
    let model = Arc::clone(&state.model);
    let kv_pool = state.kv_pool.clone();
    let prefix_cache = state.prefix_cache.clone();
    let batcher = state.continuous_batcher.clone();
    let prompt = req.prompt;

    let (chunks, finish, usage) = tokio::task::spawn_blocking(move || {
        run_generation(
            &model,
            &prompt,
            &params,
            kv_pool.as_ref(),
            prefix_cache.as_deref(),
            batcher.as_ref(),
        )
    })
    .await
    .map_err(join_error_response)?
    .map_err(decode_error_response)?;

    let text = chunks.concat();
    let finish_reason = match finish {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    };
    let model_name = req
        .model
        .unwrap_or_else(|| state.model.name().to_string());

    Ok(Json(serde_json::json!({
        "id": "ferrox-cmpl-0",
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
