/// Translate OpenAI Chat Completions <-> ChatGPT Responses API format.
///
/// The Codex endpoint (https://chatgpt.com/backend-api/codex/responses) uses
/// the Responses API, not the Chat Completions API.
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::types::{OpenAIMessageContent, OpenAIRequest, OpenAIRole};

// ── Request translation ───────────────────────────────────────────────────────

/// Build a Responses API request body from an OpenAI chat completions request.
pub fn to_responses_request(req: &OpenAIRequest) -> Value {
    // Extract system messages to use as `instructions`; remaining go into `input`
    let system_text: String = req
        .messages
        .iter()
        .filter(|m| m.role == OpenAIRole::System)
        .map(|m| extract_text(&m.content))
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    let non_system: Vec<_> = req
        .messages
        .iter()
        .filter(|m| m.role != OpenAIRole::System)
        .cloned()
        .collect();

    let input = build_input(&non_system);

    let instructions = if system_text.is_empty() {
        "You are a helpful assistant.".to_string()
    } else {
        system_text
    };

    let mut body = json!({
        "model": req.model,
        "input": input,
        "instructions": instructions,
        "store": false,
        "stream": req.stream.unwrap_or(false),
    });

    let obj = body.as_object_mut().unwrap();

    if let Some(t) = req.temperature {
        obj.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.top_p {
        obj.insert("top_p".into(), json!(p));
    }
    // max_output_tokens is not supported by the Codex endpoint — omit it

    // reasoning_effort -> reasoning config
    if let Some(effort) = &req.reasoning_effort {
        obj.insert(
            "reasoning".into(),
            json!({
                "effort": effort,
                "summary": "auto"
            }),
        );
        // reasoning models don't support temperature/top_p
        obj.remove("temperature");
        obj.remove("top_p");
    } else {
        // default reasoning for gpt-5.* models
        obj.insert(
            "reasoning".into(),
            json!({
                "effort": "medium",
                "summary": "auto"
            }),
        );
        obj.remove("temperature");
        obj.remove("top_p");
    }

    if let Some(tools) = &req.tools {
        let resp_tools: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters,
                    "strict": false,
                })
            })
            .collect();
        obj.insert("tools".into(), json!(resp_tools));

        // tool_choice
        if let Some(tc) = &req.tool_choice {
            let tc_val = match tc {
                crate::types::OpenAIToolChoice::Mode(s) => json!(s),
                crate::types::OpenAIToolChoice::Tool(t) => json!({
                    "type": "function",
                    "name": t.function.name
                }),
            };
            obj.insert("tool_choice".into(), tc_val);
        }
    }

    body
}

fn build_input(messages: &[crate::types::OpenAIMessage]) -> Value {
    let mut items: Vec<Value> = Vec::new();

    for msg in messages {
        match msg.role {
            OpenAIRole::System => {
                // system messages are handled via `instructions` field, skip here
            }
            OpenAIRole::User => {
                let content = build_user_content(&msg.content);
                items.push(json!({ "role": "user", "content": content }));
            }
            OpenAIRole::Assistant => {
                // check for tool calls
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        items.push(json!({
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }));
                    }
                    // also include any text content
                    let text = extract_text(&msg.content);
                    if !text.is_empty() {
                        items.push(json!({
                            "role": "assistant",
                            "content": [{ "type": "output_text", "text": text }],
                        }));
                    }
                } else {
                    let text = extract_text(&msg.content);
                    items.push(json!({
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": text }],
                    }));
                }
            }
            OpenAIRole::Tool => {
                if let Some(call_id) = &msg.tool_call_id {
                    items.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": extract_text(&msg.content),
                    }));
                }
            }
        }
    }

    Value::Array(items)
}

fn build_user_content(content: &OpenAIMessageContent) -> Value {
    match content {
        OpenAIMessageContent::Text(text) => {
            json!([{ "type": "input_text", "text": text }])
        }
        OpenAIMessageContent::Parts(parts) => {
            let items: Vec<Value> = parts
                .iter()
                .map(|part| match part {
                    crate::types::OpenAIContentPart::Text { text } => {
                        json!({ "type": "input_text", "text": text })
                    }
                    crate::types::OpenAIContentPart::ImageUrl { image_url } => {
                        json!({ "type": "input_image", "image_url": image_url.url })
                    }
                })
                .collect();
            Value::Array(items)
        }
        OpenAIMessageContent::Null => json!([]),
    }
}

fn extract_text(content: &OpenAIMessageContent) -> String {
    match content {
        OpenAIMessageContent::Text(t) => t.clone(),
        OpenAIMessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                crate::types::OpenAIContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        OpenAIMessageContent::Null => String::new(),
    }
}

// ── Response translation ──────────────────────────────────────────────────────

/// Translate a Responses API JSON response to an OpenAI chat completions response.
pub fn from_responses_response(resp: &Value, original_model: &str, request_id: &str) -> Value {
    let output = resp.get("output").and_then(|v| v.as_array());
    let usage = resp.get("usage");

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut finish_reason = "stop";

    if let Some(output) = output {
        for item in output {
            match item.get("type").and_then(|t| t.as_str()) {
                Some("message") => {
                    if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                        for block in content {
                            if block.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                    text_parts.push(text.to_string());
                                }
                            }
                        }
                    }
                }
                Some("function_call") => {
                    finish_reason = "tool_calls";
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let arguments = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    tool_calls.push(json!({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        }
                    }));
                }
                _ => {}
            }
        }
    }

    let mut message = json!({
        "role": "assistant",
        "content": text_parts.join(""),
    });
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }

    let (input_tokens, output_tokens) = if let Some(u) = usage {
        (
            u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
        )
    } else {
        (0, 0)
    };

    json!({
        "id": format!("chatcmpl-{request_id}"),
        "object": "chat.completion",
        "created": now_unix(),
        "model": original_model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    })
}

// ── Stream translation ────────────────────────────────────────────────────────

#[derive(Default)]
pub struct ChatgptStreamState {
    pub sent_role: bool,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// call_id -> (index, name)
    pub tool_calls: HashMap<String, (usize, String)>,
    pub tool_call_count: usize,
}

/// Translate a Responses API SSE event data string to OpenAI SSE lines.
pub fn translate_stream_event(
    event_type: &str,
    event_data: &str,
    model: &str,
    message_id: &str,
    state: &mut ChatgptStreamState,
) -> Vec<String> {
    let Ok(data) = serde_json::from_str::<Value>(event_data) else {
        return vec![];
    };

    match event_type {
        "response.output_text.delta" => {
            let delta = data.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if delta.is_empty() {
                return vec![];
            }
            let mut chunks = vec![];
            maybe_role_chunk(model, message_id, state, &mut chunks);
            chunks.push(sse(json!({
                "id": format!("chatcmpl-{message_id}"),
                "object": "chat.completion.chunk",
                "created": now_unix(),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": null}]
            })));
            chunks
        }

        "response.output_item.added" => {
            // When a function_call item appears, register it
            if let Some(item) = data.get("item") {
                if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let idx = state.tool_call_count;
                    state.tool_call_count += 1;
                    state
                        .tool_calls
                        .insert(call_id.clone(), (idx, name.clone()));

                    let mut chunks = vec![];
                    maybe_role_chunk(model, message_id, state, &mut chunks);
                    chunks.push(sse(json!({
                        "id": format!("chatcmpl-{message_id}"),
                        "object": "chat.completion.chunk",
                        "created": now_unix(),
                        "model": model,
                        "choices": [{"index": 0, "delta": {
                            "tool_calls": [{"index": idx, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}]
                        }, "finish_reason": null}]
                    })));
                    return chunks;
                }
            }
            vec![]
        }

        "response.function_call_arguments.delta" => {
            // Streaming tool call arguments
            let delta = data.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if delta.is_empty() {
                return vec![];
            }
            // Find the tool call index by item_id (call_id in this context)
            let item_id = data.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
            let idx = state.tool_calls.get(item_id).map(|(i, _)| *i).unwrap_or(0);
            vec![sse(json!({
                "id": format!("chatcmpl-{message_id}"),
                "object": "chat.completion.chunk",
                "created": now_unix(),
                "model": model,
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": idx, "function": {"arguments": delta}}]
                }, "finish_reason": null}]
            }))]
        }

        "response.completed" | "response.incomplete" => {
            if let Some(usage) = data.get("response").and_then(|r| r.get("usage")) {
                state.input_tokens = usage
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                state.output_tokens = usage
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
            }
            let finish_reason = if !state.tool_calls.is_empty() {
                "tool_calls"
            } else if event_type == "response.incomplete" {
                "length"
            } else {
                "stop"
            };
            vec![
                sse(json!({
                    "id": format!("chatcmpl-{message_id}"),
                    "object": "chat.completion.chunk",
                    "created": now_unix(),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    "usage": {
                        "prompt_tokens": state.input_tokens,
                        "completion_tokens": state.output_tokens,
                        "total_tokens": state.input_tokens + state.output_tokens,
                    }
                })),
                "data: [DONE]\n\n".to_string(),
            ]
        }

        "error" => {
            let msg = data
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            vec![sse(json!({
                "error": { "message": msg, "type": "upstream_error" }
            }))]
        }

        _ => vec![],
    }
}

fn maybe_role_chunk(
    model: &str,
    message_id: &str,
    state: &mut ChatgptStreamState,
    chunks: &mut Vec<String>,
) {
    if state.sent_role {
        return;
    }
    state.sent_role = true;
    chunks.push(sse(json!({
        "id": format!("chatcmpl-{message_id}"),
        "object": "chat.completion.chunk",
        "created": now_unix(),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    })));
}

fn sse(value: Value) -> String {
    format!("data: {value}\n\n")
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
