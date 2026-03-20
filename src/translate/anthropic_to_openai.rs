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
) -> crate::types::OpenAIResponse {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &resp.content {
        match block {
            AnthropicContentBlock::Text { text } => text_parts.push(text.clone()),
            AnthropicContentBlock::ToolUse { id, name, input } => {
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
            AnthropicContentBlock::ToolResult { .. } | AnthropicContentBlock::Thinking { .. } => {}
        }
    }

    OpenAIResponse {
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
            },
            finish_reason: Some(map_stop_reason(resp.stop_reason.as_deref())),
        }],
        usage: Some(OpenAIUsage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        }),
    }
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

fn map_stop_reason(reason: Option<&str>) -> String {
    match reason {
        Some("max_tokens") => "length".to_string(),
        Some("tool_use") => "tool_calls".to_string(),
        _ => "stop".to_string(),
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs()
}
