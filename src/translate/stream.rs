use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;

use crate::types::{
    AnthropicContentBlock, AnthropicContentBlockDelta, AnthropicStreamEvent, OpenAIRole,
};

pub struct StreamState {
    pub input_tokens: u32,
    pub output_tokens: u32,
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
                "usage": {
                    "prompt_tokens": state.input_tokens,
                    "completion_tokens": state.output_tokens,
                    "total_tokens": state.input_tokens + state.output_tokens
                }
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
