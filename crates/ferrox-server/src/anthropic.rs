//! Minimal Anthropic Messages API (`POST /v1/messages`).
//!
//! Thin adapter onto the same chat-template + [`crate::run_generation`] path as
//! OpenAI `/v1/chat/completions`. Supports non-streaming text turns only;
//! tools, images, and SSE are rejected with a clear error.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use ferrox_models::sampling::SamplingParams;
use serde::Deserialize;

use crate::generate::{FinishReason, GenerationParams};
use crate::{
    decode_error_response, join_error_response, prompt_from_messages, run_generation,
    unsupported_feature, ApiError, AppState, ChatMessage,
};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ContentIn {
    Text(String),
    Blocks(Vec<ContentBlockIn>),
}

#[derive(Debug, Deserialize)]
struct ContentBlockIn {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: ContentIn,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SystemIn {
    Text(String),
    Blocks(Vec<ContentBlockIn>),
}

#[derive(Debug, Deserialize)]
pub(crate) struct MessagesRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: usize,
    #[serde(default)]
    system: Option<SystemIn>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

fn blocks_to_text(blocks: &[ContentBlockIn]) -> Result<String, ApiError> {
    let mut out = String::new();
    for b in blocks {
        if b.kind != "text" {
            return Err(unsupported_feature(
                "Anthropic Messages: only text content blocks are supported",
            ));
        }
        if let Some(t) = &b.text {
            out.push_str(t);
        }
    }
    Ok(out)
}

fn content_to_text(content: &ContentIn) -> Result<String, ApiError> {
    match content {
        ContentIn::Text(s) => Ok(s.clone()),
        ContentIn::Blocks(blocks) => blocks_to_text(blocks),
    }
}

fn system_to_text(system: &SystemIn) -> Result<String, ApiError> {
    match system {
        SystemIn::Text(s) => Ok(s.clone()),
        SystemIn::Blocks(blocks) => blocks_to_text(blocks),
    }
}

fn to_chat_messages(req: &MessagesRequest) -> Result<Vec<ChatMessage>, ApiError> {
    let mut out = Vec::new();
    if let Some(system) = &req.system {
        out.push(ChatMessage {
            role: "system".to_string(),
            content: Some(system_to_text(system)?),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    for m in &req.messages {
        let role = m.role.as_str();
        if role != "user" && role != "assistant" {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": format!("unsupported message role '{role}' (user/assistant only)")
                    }
                })),
            ));
        }
        out.push(ChatMessage {
            role: m.role.clone(),
            content: Some(content_to_text(&m.content)?),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    Ok(out)
}

fn stop_reason(finish: FinishReason) -> &'static str {
    match finish {
        FinishReason::Stop => "end_turn",
        FinishReason::Length => "max_tokens",
    }
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MessagesRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = req.metadata;
    if req.stream == Some(true) {
        return Err(unsupported_feature(
            "Anthropic Messages streaming is not implemented yet; omit stream or set stream=false",
        ));
    }
    if req.max_tokens == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "max_tokens must be > 0"
                }
            })),
        ));
    }

    let history = to_chat_messages(&req)?;
    let template = state.model.chat_template();
    let prompt = prompt_from_messages(&history, template, &[]);
    let mut stop = req.stop_sequences.clone().unwrap_or_default();
    if matches!(template, crate::chat_template::ChatTemplate::Gemma)
        && !stop.iter().any(|s| s == "<end_of_turn>")
    {
        stop.push("<end_of_turn>".to_string());
    }
    let params = GenerationParams {
        max_tokens: req.max_tokens,
        sampling: SamplingParams {
            temperature: req.temperature.unwrap_or(0.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: req.top_k.unwrap_or(0),
            repetition_penalty: 1.0,
        },
        seed: 0,
        stop,
    };

    let model = Arc::clone(&state.model);
    let kv_pool = state.kv_pool.clone();
    let prefix_cache = state.prefix_cache.clone();
    let batcher = state.continuous_batcher.clone();
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
    Ok(Json(serde_json::json!({
        "id": "msg_ferrox_0",
        "type": "message",
        "role": "assistant",
        "model": req.model,
        "content": [{ "type": "text", "text": text }],
        "stop_reason": stop_reason(finish),
        "stop_sequence": serde_json::Value::Null,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_string_and_blocks() {
        let t = content_to_text(&ContentIn::Text("hi".into())).unwrap();
        assert_eq!(t, "hi");
        let blocks = content_to_text(&ContentIn::Blocks(vec![ContentBlockIn {
            kind: "text".into(),
            text: Some("a".into()),
        }]))
        .unwrap();
        assert_eq!(blocks, "a");
    }
}
