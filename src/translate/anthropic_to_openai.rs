use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::types::{
    AnthropicContentBlock, AnthropicErrorResponse, AnthropicResponse, OpenAIChoice, OpenAIError,
    OpenAIErrorResponse, OpenAIFunctionCall, OpenAIMessage, OpenAIMessageContent, OpenAIResponse,
    OpenAIRole, OpenAIToolCall, OpenAIUsage,
};

pub fn translate_response(
    resp: &AnthropicResponse,
    original_model: &str,
    request_id: &str,
) -> Result<crate::types::OpenAIResponse, String> {
    let finish_reason = map_stop_reason(resp.stop_reason.as_deref())?;
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &resp.content {
        match block {
            AnthropicContentBlock::Text { text, .. } => text_parts.push(text.clone()),
            AnthropicContentBlock::ToolUse {
                id, name, input, ..
            } => {
                if finish_reason == "tool_calls"
                    && (id.is_empty() || name.is_empty() || !input.is_object())
                {
                    return Err("Anthropic tool turn contains incomplete function call".into());
                }
                tool_calls.push(OpenAIToolCall {
                    id: id.clone(),
                    kind: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                });
            }
            AnthropicContentBlock::Image { .. }
            | AnthropicContentBlock::ToolResult { .. }
            | AnthropicContentBlock::Thinking { .. } => {}
        }
    }

    if finish_reason == "tool_calls" && tool_calls.is_empty() {
        return Err("Anthropic tool turn has no tool calls".into());
    }

    Ok(OpenAIResponse {
        id: format!("chatcmpl-{request_id}"),
        object: "chat.completion".to_string(),
        created: now_unix(),
        model: original_model.to_string(),
        choices: vec![OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: OpenAIRole::Assistant,
                content: OpenAIMessageContent::Text(text_parts.join("")),
                name: None,
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                tool_call_id: None,
                cache_control: None,
            },
            finish_reason: Some(finish_reason),
        }],
        usage: Some(OpenAIUsage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp
                .usage
                .input_tokens
                .saturating_add(resp.usage.output_tokens),
            cache_creation_input_tokens: resp.usage.cache_creation_input_tokens,
            cache_read_input_tokens: resp.usage.cache_read_input_tokens,
        }),
    })
}

pub fn translate_error(err: &AnthropicErrorResponse) -> crate::types::OpenAIErrorResponse {
    OpenAIErrorResponse {
        error: OpenAIError {
            message: err.error.message.clone(),
            error_type: err.error.error_type.clone(),
            param: None,
            code: None,
        },
    }
}

pub fn map_stop_reason(reason: Option<&str>) -> Result<String, String> {
    Ok(match reason {
        Some("end_turn" | "stop_sequence") => "stop".into(),
        Some("max_tokens" | "model_context_window_exceeded") => "length".into(),
        Some("tool_use") => "tool_calls".into(),
        Some("refusal") => "content_filter".into(),
        Some("pause_turn") => "pause_turn".into(),
        Some(reason) if !reason.is_empty() => format!("unknown:{reason}"),
        _ => return Err("Anthropic response has no terminal stop_reason".into()),
    })
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::translate_response;
    use crate::types::AnthropicResponse;

    #[test]
    fn preserves_anthropic_cache_usage_as_additive_input() {
        let response: AnthropicResponse = serde_json::from_value(json!({
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 30,
                "cache_read_input_tokens": 40
            }
        }))
        .unwrap();

        let translated =
            serde_json::to_value(translate_response(&response, "model", "req").unwrap()).unwrap();

        assert_eq!(translated["usage"]["prompt_tokens"], 10);
        assert_eq!(translated["usage"]["completion_tokens"], 5);
        assert_eq!(translated["usage"]["cache_creation_input_tokens"], 30);
        assert_eq!(translated["usage"]["cache_read_input_tokens"], 40);
        assert_eq!(translated["usage"]["total_tokens"], 15);
    }

    #[test]
    fn absent_anthropic_cache_usage_remains_absent() {
        let response: AnthropicResponse = serde_json::from_value(json!({
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }))
        .unwrap();

        let translated =
            serde_json::to_value(translate_response(&response, "model", "req").unwrap()).unwrap();

        assert!(
            translated["usage"]
                .get("cache_creation_input_tokens")
                .is_none()
        );
        assert!(translated["usage"].get("cache_read_input_tokens").is_none());
        assert_eq!(translated["usage"]["total_tokens"], 15);
    }

    #[test]
    fn measured_zero_anthropic_cache_usage_is_not_omitted() {
        let response: AnthropicResponse = serde_json::from_value(json!({
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        }))
        .unwrap();

        let translated =
            serde_json::to_value(translate_response(&response, "model", "req").unwrap()).unwrap();

        assert_eq!(translated["usage"]["cache_creation_input_tokens"], 0);
        assert_eq!(translated["usage"]["cache_read_input_tokens"], 0);
    }
}
