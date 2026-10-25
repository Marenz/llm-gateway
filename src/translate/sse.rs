//! SSE framing shared by translated streams and the buffered Responses adapter.
use serde_json::{Value, json};

#[derive(Default)]
pub struct Decoder {
    line: Vec<u8>,
    event: String,
    data: Vec<String>,
}

impl Decoder {
    /// Decode only complete events. Byte buffering preserves split UTF-8 codepoints.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<(String, String)>, String> {
        let mut events = Vec::new();
        for &byte in bytes {
            if byte != b'\n' {
                self.line.push(byte);
                continue;
            }
            let line = String::from_utf8(std::mem::take(&mut self.line))
                .map_err(|_| "invalid UTF-8 in upstream SSE".to_string())?;
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                if !self.data.is_empty() {
                    events.push((std::mem::take(&mut self.event), self.data.join("\n")));
                    self.data.clear();
                } else {
                    self.event.clear();
                }
            } else if let Some(value) = line.strip_prefix("data:") {
                self.data
                    .push(value.strip_prefix(' ').unwrap_or(value).to_string());
            } else if let Some(value) = line.strip_prefix("event:") {
                self.event = value.strip_prefix(' ').unwrap_or(value).to_string();
            }
        }
        Ok(events)
    }

    pub fn finish(&self) -> Result<(), String> {
        if self.line.is_empty() && self.data.is_empty() && self.event.is_empty() {
            Ok(())
        } else {
            Err("upstream SSE ended inside an event".into())
        }
    }
}

pub fn line(value: Value) -> String {
    format!("data: {value}\n\n")
}

pub fn error(message: &str) -> String {
    line(json!({"error": {"type": "upstream_error", "message": message}}))
}

/// Re-emit an already buffered completion without inventing terminal evidence.
pub fn completion_to_sse(completion: &Value) -> Result<String, String> {
    let choices = completion
        .get("choices")
        .and_then(Value::as_array)
        .filter(|choices| !choices.is_empty())
        .ok_or("upstream completion has no choices")?;
    let mut deltas = Vec::new();
    let mut finishes = Vec::new();
    for choice in choices {
        let index = choice
            .get("index")
            .and_then(Value::as_u64)
            .ok_or("upstream completion choice has no index")?;
        let reason = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
            .ok_or("upstream completion has no terminal reason")?;
        let mut message = choice
            .get("message")
            .filter(|m| m.is_object())
            .ok_or("upstream completion has no message")?
            .clone();
        if let Some(calls) = message.get_mut("tool_calls").filter(|v| !v.is_null()) {
            let calls = calls
                .as_array_mut()
                .ok_or("invalid completion tool_calls")?;
            for (index, call) in calls.iter_mut().enumerate() {
                if call
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .is_none()
                    || call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .is_none()
                    || call
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .is_none()
                    || call.get("type").and_then(Value::as_str) != Some("function")
                {
                    return Err("incomplete upstream tool call".into());
                }
                call["index"] = json!(index);
            }
        }
        deltas.push(json!({"index": index, "delta": message, "finish_reason": null}));
        finishes.push(json!({"index": index, "delta": {}, "finish_reason": reason}));
    }
    let mut chunk = completion.clone();
    let object = chunk.as_object_mut().ok_or("invalid upstream completion")?;
    object.remove("usage");
    object.insert("object".into(), json!("chat.completion.chunk"));
    object.insert("choices".into(), json!(deltas));
    let mut result = line(chunk.clone());
    chunk["choices"] = json!(finishes);
    if let Some(usage) = completion.get("usage") {
        chunk["usage"] = usage.clone();
    }
    result.push_str(&line(chunk));
    result.push_str("data: [DONE]\n\n");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_only_completion_preserves_calls_usage_and_reason() {
        let input = json!({"id":"c", "model":"m", "created":7, "choices":[{
            "index":0, "message":{"role":"assistant", "content":null, "tool_calls":[
                {"id":"call-A", "type":"function", "function":{"name":"one", "arguments":"{ \"x\": 1 }"}},
                {"id":"call-B", "type":"function", "function":{"name":"two", "arguments":"{}"}}
            ]}, "finish_reason":"tool_calls"}], "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,"cache_read_input_tokens":3}});
        let sse = completion_to_sse(&input).unwrap();
        let events = Decoder::default().push(sse.as_bytes()).unwrap();
        let chunk: Value = serde_json::from_str(&events[0].1).unwrap();
        for index in 0..2 {
            let mut call = chunk["choices"][0]["delta"]["tool_calls"][index].clone();
            assert_eq!(
                call.as_object_mut().unwrap().remove("index"),
                Some(json!(index))
            );
            assert_eq!(call, input["choices"][0]["message"]["tool_calls"][index]);
        }
        assert!(chunk["choices"][0]["delta"]["content"].is_null());
        let finish: Value = serde_json::from_str(&events[1].1).unwrap();
        assert_eq!(finish["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(finish["usage"], input["usage"]);
        assert_eq!(events[2].1, "[DONE]");
    }

    #[test]
    fn formatter_preserves_non_success_reasons_and_rejects_missing_terminal() {
        for reason in ["length", "content_filter", "pause_turn", "unknown"] {
            let completion = json!({"choices":[{"index":0,"message":{"role":"assistant","content":"partial"},"finish_reason":reason}]});
            let events = Decoder::default()
                .push(completion_to_sse(&completion).unwrap().as_bytes())
                .unwrap();
            let finish: Value = serde_json::from_str(&events[1].1).unwrap();
            assert_eq!(finish["choices"][0]["finish_reason"], reason);
        }
        assert!(
            completion_to_sse(&json!({"choices":[{"index":0,"message":{"content":"partial"}}]}))
                .is_err()
        );
    }

    #[test]
    fn decoder_handles_byte_fragments_crlf_multiline_and_truncation() {
        let mut decoder = Decoder::default();
        let mut events = Vec::new();
        for byte in
            "event: test\r\ndata: {\"text\":\"👋\",\r\ndata: \"ok\":true}\r\n\r\n".as_bytes()
        {
            events.extend(decoder.push(&[*byte]).unwrap());
        }
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "test");
        assert_eq!(
            serde_json::from_str::<Value>(&events[0].1).unwrap()["text"],
            "👋"
        );
        assert!(decoder.finish().is_ok());
        decoder.push(b"data: {\"incomplete\"").unwrap();
        assert!(decoder.finish().is_err());
    }
}
