use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;

use super::anthropic_to_openai::map_stop_reason;
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
    open_blocks: HashSet<u32>,
    stop_reason: Option<String>,
    finished: bool,
    failure: Option<String>,
}

pub struct StreamToolCall {
    pub id: String,
    initial_input: serde_json::Value,
    arguments: String,
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
            open_blocks: HashSet::new(),
            stop_reason: None,
            finished: false,
            failure: None,
        }
    }
}

impl StreamState {
    pub fn failed(&self) -> bool {
        self.failure.is_some()
    }

    pub fn fail(&mut self, message: impl Into<String>) -> Vec<String> {
        let message = message.into();
        if self.failure.is_some() {
            return vec![];
        }
        self.failure = Some(message.clone());
        vec![super::sse::error(&message)]
    }

    pub fn finish(&self) -> Result<(), String> {
        if let Some(error) = &self.failure {
            Err(error.clone())
        } else if !self.finished {
            Err("Anthropic stream ended without message_stop and a terminal stop_reason".into())
        } else {
            Ok(())
        }
    }
}

pub fn translate_stream_event(
    event_data: &str,
    model: &str,
    message_id: &str,
    state: &mut StreamState,
) -> Vec<String> {
    if state.failure.is_some() {
        return vec![];
    }
    if event_data == "[DONE]" {
        return if state.finished {
            vec![]
        } else {
            state.fail("Anthropic [DONE] arrived without message_stop")
        };
    }
    let event = match serde_json::from_str::<AnthropicStreamEvent>(event_data) {
        Ok(event) => event,
        Err(_) => return state.fail("invalid or unsupported Anthropic SSE event"),
    };
    if state.finished {
        return if matches!(event, AnthropicStreamEvent::Ping) {
            vec![]
        } else {
            state.fail("Anthropic event arrived after message_stop")
        };
    }

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
            if !state.open_blocks.insert(index) || state.tool_calls.contains_key(&index) {
                return state.fail("duplicate Anthropic content block index");
            }
            if let AnthropicContentBlock::ToolUse {
                id, name, input, ..
            } = content_block
            {
                if id.is_empty()
                    || name.is_empty()
                    || state.tool_calls.values().any(|call| call.id == id)
                {
                    return state.fail("Anthropic tool_use missing or duplicate identity");
                }
                let delta = json!({"tool_calls":[{"index":index,"id":id,"type":"function","function":{"name":name,"arguments":""}}]});
                state.tool_calls.insert(
                    index,
                    StreamToolCall {
                        id,
                        initial_input: input,
                        arguments: String::new(),
                    },
                );
                let mut chunks = Vec::new();
                maybe_emit_role(model, message_id, state, &mut chunks);
                chunks.push(delta_chunk(model, message_id, delta));
                chunks
            } else {
                Vec::new()
            }
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
                if !state.open_blocks.contains(&index) {
                    return state.fail("arguments reference a closed or unknown Anthropic block");
                }
                let Some(call) = state.tool_calls.get_mut(&index) else {
                    return state.fail("arguments reference a non-tool Anthropic block");
                };
                call.arguments.push_str(&partial_json);
                vec![delta_chunk(
                    model,
                    message_id,
                    json!({"tool_calls":[{"index":index,"function":{"arguments":partial_json}}]}),
                )]
            }
            AnthropicContentBlockDelta::ThinkingDelta { .. }
            | AnthropicContentBlockDelta::SignatureDelta { .. } => Vec::new(),
        },
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            if let Some(usage) = usage {
                state.output_tokens = usage.output_tokens;
            }
            if delta.stop_reason.is_some() {
                if state.stop_reason.is_some() {
                    return state.fail("duplicate Anthropic terminal stop_reason");
                }
                match map_stop_reason(delta.stop_reason.as_deref()) {
                    Ok(reason) => state.stop_reason = Some(reason),
                    Err(error) => return state.fail(error),
                }
            }
            // A message_delta alone does not prove that the stream completed.
            Vec::new()
        }
        AnthropicStreamEvent::MessageStop => {
            let Some(reason) = state.stop_reason.clone() else {
                return state.fail("Anthropic message_stop has no terminal stop_reason");
            };
            if !state.open_blocks.is_empty() {
                return state.fail("Anthropic message_stop arrived with unclosed content blocks");
            }
            if reason == "tool_calls"
                && (state.tool_calls.is_empty()
                    || state.tool_calls.values().any(|call| {
                        !serde_json::from_str::<serde_json::Value>(&call.arguments)
                            .is_ok_and(|v| v.is_object())
                    }))
            {
                return state.fail("Anthropic tool turn contains incomplete function arguments");
            }
            state.finished = true;
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

            vec![
                sse_line(json!({
                    "id": format!("chatcmpl-{message_id}"),
                    "object": "chat.completion.chunk",
                    "created": now_unix(),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": reason
                    }],
                    "usage": usage_json
                })),
                "data: [DONE]\n\n".to_string(),
            ]
        }
        AnthropicStreamEvent::Error { error } => state.fail(error.message),
        AnthropicStreamEvent::ContentBlockStop { index } => {
            if !state.open_blocks.remove(&index) {
                return state.fail("Anthropic content_block_stop has no open block");
            }
            if let Some(call) = state.tool_calls.get_mut(&index) {
                if call.arguments.is_empty() {
                    // Empty-input tools have no input_json_delta. Do not prepend
                    // '{}' at start: ordinary streamed arguments would concatenate.
                    call.arguments = call.initial_input.to_string();
                    return vec![delta_chunk(
                        model,
                        message_id,
                        json!({"tool_calls":[{"index":index,"function":{"arguments":call.arguments}}]}),
                    )];
                }
            }
            Vec::new()
        }
        AnthropicStreamEvent::Ping => Vec::new(),
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

fn delta_chunk(model: &str, message_id: &str, delta: serde_json::Value) -> String {
    sse_line(json!({
        "id":format!("chatcmpl-{message_id}"), "object":"chat.completion.chunk",
        "created":now_unix(), "model":model,
        "choices":[{"index":0,"delta":delta,"finish_reason":null}]
    }))
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
        assert!(translate_stream_event(&delta.to_string(), "model", "id", &mut state).is_empty());
        let chunks =
            translate_stream_event(r#"{"type":"message_stop"}"#, "model", "id", &mut state);
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
            translate_stream_event(&delta.to_string(), "model", "id", &mut state);
            let chunks =
                translate_stream_event(r#"{"type":"message_stop"}"#, "model", "id", &mut state);
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
