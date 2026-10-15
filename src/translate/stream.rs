use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;

use crate::types::{
    AnthropicContentBlock, AnthropicContentBlockDelta, AnthropicStreamEvent, OpenAIRole,
};

pub struct StreamState {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_creation_input_tokens: Option<u32>,
    pub cache_read_input_tokens: Option<u32>,
    pub sent_role: bool,
    pub tool_calls: HashMap<u32, StreamToolCall>,
    pub accumulated_text: String,
}

pub struct StreamToolCall {
    pub id: String,
    pub name: String,
}

impl Default for StreamState {
    fn default() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            sent_role: false,
            tool_calls: HashMap::new(),
            accumulated_text: String::new(),
        }
    }
}

pub fn translate_stream_event(
    event_data: &str,
    model: &str,
    message_id: &str,
    state: &mut StreamState,
) -> Vec<String> {
    let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(event_data) else {
        return Vec::new();
    };

    match event {
        AnthropicStreamEvent::MessageStart { message } => {
            state.input_tokens = message.usage.input_tokens;
            state.cache_creation_input_tokens = message.usage.cache_creation_input_tokens;
            state.cache_read_input_tokens = message.usage.cache_read_input_tokens;
            Vec::new()
        }
        AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            if let AnthropicContentBlock::ToolUse { id, name, .. } = content_block {
                state.tool_calls.insert(index, StreamToolCall { id, name });
            }
            Vec::new()
        }
        AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
            AnthropicContentBlockDelta::TextDelta { text } => {
                state.accumulated_text.push_str(&text);
                let mut chunks = Vec::new();
                maybe_emit_role(model, message_id, state, &mut chunks);
                chunks.push(sse_line(json!({
                    "id": format!("chatcmpl-{message_id}"),
                    "object": "chat.completion.chunk",
                    "created": now_unix(),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": text },
                        "finish_reason": serde_json::Value::Null
                    }]
                })));
                chunks
            }
            AnthropicContentBlockDelta::InputJsonDelta { partial_json } => {
                let mut chunks = Vec::new();
                maybe_emit_role(model, message_id, state, &mut chunks);
                let tool_call = state.tool_calls.get(&index);
                chunks.push(sse_line(json!({
                    "id": format!("chatcmpl-{message_id}"),
                    "object": "chat.completion.chunk",
                    "created": now_unix(),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": tool_call.map(|call| call.id.clone()),
                                "type": "function",
                                "function": {
                                    "name": tool_call.map(|call| call.name.clone()),
                                    "arguments": partial_json,
                                }
                            }]
                        },
                        "finish_reason": serde_json::Value::Null
                    }]
                })));
                chunks
            }
            AnthropicContentBlockDelta::ThinkingDelta { .. }
            | AnthropicContentBlockDelta::SignatureDelta { .. } => Vec::new(),
        },
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            if let Some(usage) = usage {
                state.output_tokens = usage.output_tokens;
            }

            let mut usage_json = json!({
                "prompt_tokens": state.input_tokens,
                "completion_tokens": state.output_tokens,
                "total_tokens": state.input_tokens.saturating_add(state.output_tokens)
            });
            if let Some(tokens) = state.cache_creation_input_tokens {
                usage_json["cache_creation_input_tokens"] = json!(tokens);
            }
            if let Some(tokens) = state.cache_read_input_tokens {
                usage_json["cache_read_input_tokens"] = json!(tokens);
            }

            vec![sse_line(json!({
                "id": format!("chatcmpl-{message_id}"),
                "object": "chat.completion.chunk",
                "created": now_unix(),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": map_stop_reason(delta.stop_reason.as_deref())
                }],
                "usage": usage_json
            }))]
        }
        AnthropicStreamEvent::MessageStop => vec!["data: [DONE]\n\n".to_string()],
        AnthropicStreamEvent::Error { error } => vec![sse_line(json!({
            "error": {
                "message": error.message,
                "type": error.error_type,
            }
        }))],
        AnthropicStreamEvent::ContentBlockStop { .. } | AnthropicStreamEvent::Ping => Vec::new(),
    }
}

fn maybe_emit_role(
    model: &str,
    message_id: &str,
    state: &mut StreamState,
    chunks: &mut Vec<String>,
) {
    if state.sent_role {
        return;
    }

    state.sent_role = true;
    chunks.push(sse_line(json!({
        "id": format!("chatcmpl-{message_id}"),
        "object": "chat.completion.chunk",
        "created": now_unix(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "role": OpenAIRole::Assistant },
            "finish_reason": serde_json::Value::Null
        }]
    })));
}

fn sse_line(value: serde_json::Value) -> String {
    format!("data: {}\n\n", value)
}

fn map_stop_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        _ => "stop",
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs()
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{StreamState, translate_stream_event};

    #[test]
    fn streaming_anthropic_usage_preserves_cache_counters() {
        let mut state = StreamState::default();
        let start = json!({
            "type": "message_start",
            "message": {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 30,
                    "cache_read_input_tokens": 40
                }
            }
        });
        assert!(translate_stream_event(&start.to_string(), "model", "id", &mut state).is_empty());

        let delta = json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 5}
        });
        let chunks = translate_stream_event(&delta.to_string(), "model", "id", &mut state);
        let chunk: Value =
            serde_json::from_str(chunks[0].trim_start_matches("data: ").trim()).unwrap();

        assert_eq!(chunk["usage"]["prompt_tokens"], 10);
        assert_eq!(chunk["usage"]["cache_creation_input_tokens"], 30);
        assert_eq!(chunk["usage"]["cache_read_input_tokens"], 40);
        assert_eq!(chunk["usage"]["total_tokens"], 15);
    }

    #[test]
    fn streaming_anthropic_usage_distinguishes_absent_from_zero() {
        fn usage_for(cache_fields: Value) -> Value {
            let mut state = StreamState::default();
            let mut usage = json!({"input_tokens": 10, "output_tokens": 0});
            usage
                .as_object_mut()
                .unwrap()
                .extend(cache_fields.as_object().unwrap().clone());
            let start = json!({
                "type": "message_start",
                "message": {
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-opus-4-6",
                    "usage": usage
                }
            });
            translate_stream_event(&start.to_string(), "model", "id", &mut state);
            let delta = json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5}
            });
            let chunks = translate_stream_event(&delta.to_string(), "model", "id", &mut state);
            serde_json::from_str::<Value>(chunks[0].trim_start_matches("data: ").trim()).unwrap()
        }

        let absent = usage_for(json!({}));
        let zero = usage_for(json!({
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0
        }));

        assert!(absent["usage"].get("cache_creation_input_tokens").is_none());
        assert!(absent["usage"].get("cache_read_input_tokens").is_none());
        assert_eq!(zero["usage"]["cache_creation_input_tokens"], 0);
        assert_eq!(zero["usage"]["cache_read_input_tokens"], 0);
    }
}
