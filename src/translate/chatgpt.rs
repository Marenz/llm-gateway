/// Translate OpenAI Chat Completions <-> ChatGPT Responses API format.
///
/// The Codex endpoint (https://chatgpt.com/backend-api/codex/responses) uses
/// the Responses API, not the Chat Completions API.
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

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
            // No "summary" key: requesting summaries ("auto") makes the
            // backend emit reasoning-summary text that also arrives as
            // response.output_text.delta events and gets concatenated with
            // the final answer (duplicated-text bug). "none" is rejected
            // by the API, so omit the key entirely.
            json!({
                "effort": effort
            }),
        );
        // reasoning models don't support temperature/top_p
        obj.remove("temperature");
        obj.remove("top_p");
    } else {
        // default reasoning for gpt-5.* models
        obj.insert(
            "reasoning".into(),
            // No "summary" key: see comment above (duplicated-text bug).
            json!({
                "effort": "medium"
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
                    crate::types::OpenAIContentPart::Text { text, .. } => {
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
                crate::types::OpenAIContentPart::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        OpenAIMessageContent::Null => String::new(),
    }
}

// ── Stream translation ────────────────────────────────────────────────────────

#[derive(Default)]
pub struct ChatgptStreamState {
    pub sent_role: bool,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: Option<u64>,
    /// Responses item.id -> stable OpenAI tool-call index. NOT the public call_id.
    tool_indices: HashMap<String, usize>,
    tool_calls: Vec<Value>,
    pub accumulated_text: String,
    refusal: String,
    finish_reason: Option<String>,
    failure: Option<String>,
}

impl ChatgptStreamState {
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
        } else if self.finish_reason.is_none() {
            Err("ChatGPT stream ended without a terminal response event".into())
        } else {
            Ok(())
        }
    }

    fn usage(&self) -> Value {
        let mut usage = json!({
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        });
        if let Some(cached_tokens) = self.cached_tokens {
            usage["cache_read_input_tokens"] = json!(cached_tokens);
        }
        usage
    }

    /// The same state machine assembles buffered and streaming responses.
    pub fn completion(&self, model: &str, request_id: &str) -> Result<Value, String> {
        self.finish()?;
        let mut message = json!({"role": "assistant", "content": self.accumulated_text});
        if !self.refusal.is_empty() {
            message["refusal"] = json!(self.refusal);
        }
        if !self.tool_calls.is_empty() {
            message["tool_calls"] = json!(self.tool_calls);
        }
        Ok(json!({
            "id": format!("chatcmpl-{request_id}"), "object": "chat.completion",
            "created": now_unix(), "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}],
            "usage": self.usage()
        }))
    }

    fn register_call(&mut self, item: &Value) -> Result<Value, String> {
        let item_id = required_string(item, "id")?;
        let call_id = required_string(item, "call_id")?;
        let name = required_string(item, "name")?;
        if self.tool_indices.contains_key(item_id)
            || self.tool_calls.iter().any(|call| call["id"] == call_id)
        {
            return Err("duplicate Responses function-call identity".into());
        }
        let index = self.tool_calls.len();
        let arguments = item.get("arguments").and_then(Value::as_str).unwrap_or("");
        let call =
            json!({"id":call_id,"type":"function","function":{"name":name,"arguments":arguments}});
        self.tool_calls.push(call.clone());
        self.tool_indices.insert(item_id.to_string(), index);
        let mut delta = call;
        delta["index"] = json!(index);
        Ok(json!({"tool_calls":[delta]}))
    }

    fn append_arguments(&mut self, item_id: &str, delta: &str) -> Result<Value, String> {
        let index = *self
            .tool_indices
            .get(item_id)
            .ok_or("arguments reference an unknown Responses item_id")?;
        let args = self.tool_calls[index]["function"]["arguments"]
            .as_str()
            .unwrap();
        self.tool_calls[index]["function"]["arguments"] = json!(format!("{args}{delta}"));
        Ok(json!({"tool_calls":[{"index":index,"function":{"arguments":delta}}]}))
    }

    fn final_arguments(&mut self, item_id: &str, arguments: &str) -> Result<Value, String> {
        let index = *self
            .tool_indices
            .get(item_id)
            .ok_or("arguments reference an unknown Responses item_id")?;
        let prefix = self.tool_calls[index]["function"]["arguments"]
            .as_str()
            .unwrap();
        let remaining = arguments
            .strip_prefix(prefix)
            .ok_or("final function arguments contradict streamed arguments")?
            .to_string();
        self.append_arguments(item_id, &remaining)
    }

    fn reconcile_call(&mut self, item: &Value) -> Result<Value, String> {
        let item_id = required_string(item, "id")?;
        let index = *self
            .tool_indices
            .get(item_id)
            .ok_or("unknown completed function call")?;
        let call = &self.tool_calls[index];
        if call["id"] != required_string(item, "call_id")?
            || call["function"]["name"] != required_string(item, "name")?
        {
            return Err("completed function-call identity changed".into());
        }
        self.final_arguments(
            item_id,
            item.get("arguments")
                .and_then(Value::as_str)
                .ok_or("completed function call has no arguments")?,
        )
    }
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("Responses event missing {key}"))
}

/// Translate a Responses API SSE event data string to OpenAI SSE lines.
pub fn translate_stream_event(
    event_type: &str,
    event_data: &str,
    model: &str,
    message_id: &str,
    state: &mut ChatgptStreamState,
) -> Vec<String> {
    if state.failure.is_some() {
        return vec![];
    }
    if event_data == "[DONE]" {
        return if state.finish_reason.is_some() {
            vec![]
        } else {
            state.fail("ChatGPT [DONE] arrived without a terminal response event")
        };
    }
    let data = match serde_json::from_str::<Value>(event_data) {
        Ok(data) => data,
        Err(_) => return state.fail("invalid JSON in ChatGPT SSE event"),
    };
    let event_type = if event_type.is_empty() {
        data.get("type").and_then(Value::as_str).unwrap_or("")
    } else {
        event_type
    };
    if state.finish_reason.is_some() {
        return state.fail("ChatGPT event arrived after terminal response");
    }
    match translate_event(event_type, &data, model, message_id, state) {
        Ok(chunks) => chunks,
        Err(error) => state.fail(error),
    }
}

fn translate_event(
    event_type: &str,
    data: &Value,
    model: &str,
    message_id: &str,
    state: &mut ChatgptStreamState,
) -> Result<Vec<String>, String> {
    let mut deltas = Vec::new();
    let mut terminal = None;
    match event_type {
        "response.output_text.delta" => {
            let delta = data
                .get("delta")
                .and_then(Value::as_str)
                .ok_or("missing text delta")?;
            state.accumulated_text.push_str(delta);
            deltas.push(json!({"content": delta}));
        }
        "response.refusal.delta" => {
            let delta = data
                .get("delta")
                .and_then(Value::as_str)
                .ok_or("missing refusal delta")?;
            state.refusal.push_str(delta);
            deltas.push(json!({"refusal":delta}));
        }
        "response.output_item.added" => {
            if let Some(item) = data.get("item") {
                if item["type"] == "function_call" {
                    deltas.push(state.register_call(item)?);
                }
            }
        }
        "response.function_call_arguments.delta" => {
            deltas.push(
                state.append_arguments(
                    required_string(data, "item_id")?,
                    data.get("delta")
                        .and_then(Value::as_str)
                        .ok_or("missing arguments delta")?,
                )?,
            );
        }
        "response.function_call_arguments.done" => {
            deltas.push(
                state.final_arguments(
                    required_string(data, "item_id")?,
                    data.get("arguments")
                        .and_then(Value::as_str)
                        .ok_or("missing final arguments")?,
                )?,
            );
        }
        "response.output_item.done" => {
            if let Some(item) = data
                .get("item")
                .filter(|item| item["type"] == "function_call")
            {
                deltas.push(state.reconcile_call(item)?);
            }
        }
        "response.completed" | "response.incomplete" => {
            let response = data
                .get("response")
                .filter(|response| response.is_object())
                .ok_or("terminal event has no response")?;
            let expected = if event_type == "response.completed" {
                "completed"
            } else {
                "incomplete"
            };
            if response
                .get("status")
                .is_some_and(|status| status != expected)
            {
                return Err("terminal event contradicts response status".into());
            }
            if let Some(usage) = data.get("response").and_then(|r| r.get("usage")) {
                apply_usage(state, usage);
            }
            if let Some(output) = response.get("output").and_then(Value::as_array) {
                for item in output.iter().filter(|item| item["type"] == "function_call") {
                    if !state
                        .tool_indices
                        .contains_key(required_string(item, "id")?)
                    {
                        deltas.push(state.register_call(item)?);
                    }
                    deltas.push(state.reconcile_call(item)?);
                }
            }
            let reason = if event_type == "response.incomplete" {
                match response
                    .pointer("/incomplete_details/reason")
                    .and_then(Value::as_str)
                {
                    Some("max_output_tokens") => "length",
                    Some("content_filter") => "content_filter",
                    _ => "incomplete",
                }
            } else {
                for call in &state.tool_calls {
                    let args = call["function"]["arguments"].as_str().unwrap();
                    if !serde_json::from_str::<Value>(args).is_ok_and(|v| v.is_object()) {
                        return Err(
                            "completed response contains incomplete function arguments".into()
                        );
                    }
                }
                if !state.refusal.is_empty() {
                    "content_filter"
                } else if state.tool_calls.is_empty() {
                    "stop"
                } else {
                    "tool_calls"
                }
            };
            terminal = Some(reason.to_string());
        }
        "error" | "response.failed" => {
            return Err(data
                .pointer("/response/error/message")
                .or_else(|| data.pointer("/error/message"))
                .or_else(|| data.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("ChatGPT response failed")
                .to_string());
        }
        _ => {}
    }
    let mut chunks = Vec::new();
    if !deltas.is_empty() {
        maybe_role_chunk(model, message_id, state, &mut chunks);
    }
    for delta in deltas {
        chunks.push(sse(json!({
            "id":format!("chatcmpl-{message_id}"), "object":"chat.completion.chunk",
            "created":now_unix(), "model":model,
            "choices":[{"index":0,"delta":delta,"finish_reason":null}]
        })));
    }
    if let Some(reason) = terminal {
        state.finish_reason = Some(reason.clone());
        chunks.push(sse(json!({
            "id":format!("chatcmpl-{message_id}"), "object":"chat.completion.chunk",
            "created":now_unix(), "model":model,
            "choices":[{"index":0,"delta":{},"finish_reason":reason}], "usage":state.usage()
        })));
        chunks.push("data: [DONE]\n\n".into());
    }
    Ok(chunks)
}

pub fn apply_usage(state: &mut ChatgptStreamState, usage: &Value) {
    state.input_tokens = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    state.output_tokens = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    state.cached_tokens = usage
        .get("input_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64);
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

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ChatgptStreamState, translate_stream_event};

    #[test]
    fn collected_response_shape_reports_cached_subset_without_double_counting() {
        let response = json!({
            "output": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 70}
            }
        });

        let mut state = ChatgptStreamState::default();
        translate_stream_event(
            "response.completed",
            &json!({"response":response}).to_string(),
            "model",
            "id",
            &mut state,
        );
        let completion = state.completion("model", "id").unwrap();

        assert_eq!(completion["usage"]["prompt_tokens"], 100);
        assert_eq!(completion["usage"]["cache_read_input_tokens"], 70);
        assert_eq!(completion["usage"]["total_tokens"], 120);
    }

    #[test]
    fn streaming_chatgpt_usage_reports_cached_subset_without_double_counting() {
        let mut state = ChatgptStreamState::default();
        let event = json!({
            "response": {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 70}
                }
            }
        });

        let chunks = translate_stream_event(
            "response.completed",
            &event.to_string(),
            "model",
            "id",
            &mut state,
        );
        let chunk: Value =
            serde_json::from_str(chunks[0].trim_start_matches("data: ").trim()).unwrap();

        assert_eq!(chunk["usage"]["prompt_tokens"], 100);
        assert_eq!(chunk["usage"]["cache_read_input_tokens"], 70);
        assert_eq!(chunk["usage"]["total_tokens"], 120);
        assert!(chunk["usage"].get("cache_creation_input_tokens").is_none());
    }

    #[test]
    fn missing_chatgpt_cache_details_remain_absent() {
        let mut state = ChatgptStreamState::default();
        let event = json!({
            "response": {"usage": {"input_tokens": 100, "output_tokens": 20}}
        });

        let chunks = translate_stream_event(
            "response.completed",
            &event.to_string(),
            "model",
            "id",
            &mut state,
        );
        let chunk: Value =
            serde_json::from_str(chunks[0].trim_start_matches("data: ").trim()).unwrap();

        assert!(chunk["usage"].get("cache_read_input_tokens").is_none());
    }
}
