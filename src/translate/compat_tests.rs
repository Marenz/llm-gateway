//! Sanitized raw provider fixtures and loopback-only HTTP regressions for #2.
use axum::body::{Body, to_bytes};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{anthropic_to_openai, chatgpt, openai_to_anthropic, sse, stream, transport};
use crate::types::{AnthropicResponse, OpenAIRequest};

const CHATGPT: &str = include_str!("fixtures/chatgpt_tools.sse");
const ANTHROPIC: &str = include_str!("fixtures/anthropic_tools.sse");

fn events(body: &str) -> Vec<(String, String)> {
    let mut decoder = sse::Decoder::default();
    let result = decoder.push(body.as_bytes()).unwrap();
    decoder.finish().unwrap();
    result
}

fn values(body: &str) -> Vec<Value> {
    events(body)
        .into_iter()
        .filter(|(_, data)| data != "[DONE]")
        .map(|(_, data)| serde_json::from_str(&data).unwrap())
        .collect()
}

fn assemble_calls(chunks: &[Value]) -> BTreeMap<u64, Value> {
    let mut calls: BTreeMap<u64, Value> = BTreeMap::new();
    for chunk in chunks {
        if let Some(deltas) = chunk
            .pointer("/choices/0/delta/tool_calls")
            .and_then(Value::as_array)
        {
            for delta in deltas {
                let call = calls
                    .entry(delta["index"].as_u64().unwrap())
                    .or_insert_with(|| json!({"function":{"arguments":""}}));
                for key in ["id", "type"] {
                    if let Some(value) = delta.get(key) {
                        assert!(
                            call.get(key).is_none(),
                            "identity must only be emitted once"
                        );
                        call[key] = value.clone();
                    }
                }
                if let Some(name) = delta.pointer("/function/name") {
                    assert!(call["function"].get("name").is_none());
                    call["function"]["name"] = name.clone();
                }
                if let Some(args) = delta.pointer("/function/arguments").and_then(Value::as_str) {
                    call["function"]["arguments"] = json!(format!(
                        "{}{args}",
                        call["function"]["arguments"].as_str().unwrap()
                    ));
                }
            }
        }
    }
    calls
}

fn call(id: &str, name: &str, args: &str) -> Value {
    json!({"id":id,"type":"function","function":{"name":name,"arguments":args}})
}

fn chatgpt_fixture(body: &str) -> (chatgpt::ChatgptStreamState, String) {
    let mut state = chatgpt::ChatgptStreamState::default();
    let mut result = String::new();
    // Split at every byte, including within UTF-8 and JSON escape sequences.
    let mut decoder = sse::Decoder::default();
    for byte in body.as_bytes() {
        for (event, data) in decoder.push(&[*byte]).unwrap() {
            result.extend(chatgpt::translate_stream_event(
                &event, &data, "m", "r", &mut state,
            ));
        }
    }
    decoder.finish().unwrap();
    (state, result)
}

#[test]
fn chatgpt_distinct_item_ids_interleaved_arguments_match_buffered_and_streamed() {
    let (state, output) = chatgpt_fixture(CHATGPT);
    let completion = state.completion("m", "r").unwrap();
    let chunks = values(&output);
    let calls = assemble_calls(&chunks);
    let expected = vec![
        call("call-A", "lookup", "{ \"city\": \"Zürich\" }"),
        call("call-B", "clock", "{}"),
    ];
    assert_eq!(calls.into_values().collect::<Vec<_>>(), expected);
    assert_eq!(
        completion["choices"][0]["message"]["tool_calls"],
        json!(expected)
    );
    assert_eq!(
        completion["choices"][0]["message"]["content"],
        "Checking 👋"
    );
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "tool_calls"
    );
    assert_eq!(chunks.last().unwrap()["usage"], completion["usage"]);
    assert_eq!(completion["usage"]["cache_read_input_tokens"], 70);
    assert!(output.ends_with("data: [DONE]\n\n"));
}

#[test]
fn anthropic_zero_input_and_sparse_blocks_keep_identities_arguments_usage() {
    let mut state = stream::StreamState::default();
    let mut output = String::new();
    for (_, data) in events(ANTHROPIC) {
        output.extend(stream::translate_stream_event(&data, "m", "r", &mut state));
    }
    state.finish().unwrap();
    let chunks = values(&output);
    let calls = assemble_calls(&chunks);
    assert_eq!(calls.keys().copied().collect::<Vec<_>>(), vec![2, 5]);
    assert_eq!(calls[&2], call("toolu-A", "clock", "{}"));
    assert_eq!(
        calls[&5],
        call("toolu-B", "lookup", "{ \"city\": \"Zürich\" }")
    );
    let terminal = chunks.last().unwrap();
    assert_eq!(terminal["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        terminal["usage"],
        json!({"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cache_creation_input_tokens":30,"cache_read_input_tokens":40})
    );
    assert!(output.ends_with("data: [DONE]\n\n"));
}

#[test]
fn chatgpt_incomplete_calls_never_have_successful_terminal_reason() {
    let prefix = CHATGPT.split("event: response.completed").next().unwrap();
    for (upstream_reason, expected) in [
        ("max_output_tokens", "length"),
        ("content_filter", "content_filter"),
        ("unexpected", "incomplete"),
    ] {
        let fixture = format!(
            "{prefix}event: response.incomplete\ndata: {}\n\n",
            json!({"response":{"status":"incomplete","incomplete_details":{"reason":upstream_reason}}})
        );
        let (state, output) = chatgpt_fixture(&fixture);
        let completion = state.completion("m", "r").unwrap();
        assert_eq!(completion["choices"][0]["finish_reason"], expected);
        assert_eq!(
            values(&output).last().unwrap()["choices"][0]["finish_reason"],
            expected
        );
    }
}

#[test]
fn chatgpt_failed_error_unknown_call_and_invalid_arguments_are_sticky_errors() {
    let prefix = CHATGPT
        .split("event: response.function_call_arguments.delta")
        .next()
        .unwrap();
    for suffix in [
        "event: response.failed\ndata: {\"response\":{\"error\":{\"message\":\"failed fixture\"}}}\n\n",
        "event: error\ndata: {\"message\":\"error fixture\"}\n\n",
        "event: response.function_call_arguments.delta\ndata: {\"item_id\":\"absent\",\"delta\":\"{}\"}\n\n",
        "event: response.completed\ndata: {\"response\":{\"status\":\"completed\"}}\n\n",
        "event: response.completed\ndata: {\"response\":{\"status\":\"failed\"}}\n\n",
        "event: response.completed\ndata: {\"response\":null}\n\n",
        "data: [DONE]\n\n",
        "data: not-json\n\n",
    ] {
        let (mut state, output) = chatgpt_fixture(&format!("{prefix}{suffix}"));
        assert!(state.completion("m", "r").is_err(), "{suffix}");
        assert!(
            values(&output).iter().any(|v| v.get("error").is_some()),
            "{suffix}"
        );
        assert!(!output.contains("data: [DONE]"));
        assert!(
            chatgpt::translate_stream_event(
                "response.completed",
                r#"{"response":{}}"#,
                "m",
                "r",
                &mut state
            )
            .is_empty()
        );
        assert!(state.finish().is_err());
    }
}

#[test]
fn chatgpt_refusal_has_non_success_reason() {
    let (state, output) = chatgpt_fixture(
        "event: response.refusal.delta\ndata: {\"delta\":\"Cannot comply\"}\n\nevent: response.completed\ndata: {\"response\":{\"status\":\"completed\"}}\n\n",
    );
    assert_eq!(
        state.completion("m", "r").unwrap()["choices"][0]["finish_reason"],
        "content_filter"
    );
    assert_eq!(
        values(&output).last().unwrap()["choices"][0]["finish_reason"],
        "content_filter"
    );
}

#[test]
fn missing_terminal_evidence_and_anthropic_errors_cannot_succeed() {
    let (state, _) = chatgpt_fixture(CHATGPT.split("event: response.completed").next().unwrap());
    assert!(state.completion("m", "r").is_err());
    let mut state = stream::StreamState::default();
    let before_stop = ANTHROPIC.split("event: message_stop").next().unwrap();
    let mut output = String::new();
    for (_, data) in events(before_stop) {
        output.extend(stream::translate_stream_event(&data, "m", "r", &mut state));
    }
    assert!(state.finish().is_err());
    assert!(
        values(&output)
            .iter()
            .all(|v| v["choices"][0]["finish_reason"].is_null())
    );
    output.extend(stream::translate_stream_event(
        r#"{"type":"error","error":{"type":"overloaded_error","message":"fixture overload"}}"#,
        "m",
        "r",
        &mut state,
    ));
    assert!(values(&output).last().unwrap().get("error").is_some());
    assert!(
        stream::translate_stream_event(r#"{"type":"message_stop"}"#, "m", "r", &mut state)
            .is_empty()
    );
    assert!(!output.contains("[DONE]"));
    let mut missing = stream::StreamState::default();
    assert!(
        values(
            &stream::translate_stream_event(r#"{"type":"message_stop"}"#, "m", "r", &mut missing)
                .concat()
        )[0]
        .get("error")
        .is_some()
    );
}

#[test]
fn anthropic_stop_reasons_match_in_json_and_stream_and_missing_is_error() {
    for (reason, expected) in [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("refusal", "content_filter"),
        ("pause_turn", "pause_turn"),
        ("future_reason", "unknown:future_reason"),
    ] {
        let response: AnthropicResponse = serde_json::from_value(json!({"id":"a","type":"message","role":"assistant","model":"m","content":[],"stop_reason":reason,"usage":{}})).unwrap();
        let json = anthropic_to_openai::translate_response(&response, "m", "r").unwrap();
        assert_eq!(json.choices[0].finish_reason.as_deref(), Some(expected));
        let mut state = stream::StreamState::default();
        assert!(
            stream::translate_stream_event(
                &json!({"type":"message_delta","delta":{"stop_reason":reason}}).to_string(),
                "m",
                "r",
                &mut state
            )
            .is_empty()
        );
        let output =
            stream::translate_stream_event(r#"{"type":"message_stop"}"#, "m", "r", &mut state)
                .concat();
        assert_eq!(values(&output)[0]["choices"][0]["finish_reason"], expected);
    }
    let response: AnthropicResponse = serde_json::from_value(
        json!({"id":"a","type":"message","role":"assistant","model":"m","content":[],"usage":{}}),
    )
    .unwrap();
    assert!(anthropic_to_openai::translate_response(&response, "m", "r").is_err());
}

#[test]
fn tool_result_roundtrip_preserves_schemas_auto_choice_ids_and_cache_boundaries() {
    let request: OpenAIRequest = serde_json::from_value(json!({
        "model":"fixture", "tools":[{"type":"function","function":{"name":"lookup","description":"find","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}], "tool_choice":"auto",
        "messages":[
            {"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"https://example.com/fixture.png"}}]},
            {"role":"assistant","content":null,"tool_calls":[call("call-A","lookup","{\"city\":\"Zürich\"}"),call("call-B","clock","{}")],"cache_control":{"type":"ephemeral"}},
            {"role":"tool","tool_call_id":"call-A","content":"result A"},
            {"role":"tool","tool_call_id":"call-B","content":"result B","cache_control":{"type":"ephemeral"}}
        ]})).unwrap();
    let anthropic =
        serde_json::to_value(openai_to_anthropic::translate_request(&request).unwrap()).unwrap();
    assert_eq!(
        anthropic["tools"][0]["input_schema"],
        request.tools.as_ref().unwrap()[0].function.parameters
    );
    assert_eq!(anthropic["tool_choice"], json!({"type":"auto"}));
    assert_eq!(anthropic["messages"][0]["content"][1]["type"], "image");
    assert_eq!(anthropic["messages"][1]["content"][0]["id"], "call-A");
    assert_eq!(anthropic["messages"][1]["content"][1]["id"], "call-B");
    assert_eq!(
        anthropic["messages"][1]["content"][1]["cache_control"],
        json!({"type":"ephemeral"})
    );
    for (index, id) in ["call-A", "call-B"].iter().enumerate() {
        assert_eq!(
            anthropic["messages"][2]["content"][index]["tool_use_id"],
            *id
        );
    }
    assert_eq!(
        anthropic["messages"][2]["content"][1]["cache_control"],
        json!({"type":"ephemeral"})
    );
    let responses = chatgpt::to_responses_request(&request);
    assert_eq!(responses["tool_choice"], "auto");
    assert_eq!(
        responses["tools"][0]["parameters"],
        request.tools.as_ref().unwrap()[0].function.parameters
    );
    assert_eq!(responses["input"][1]["call_id"], "call-A");
    assert_eq!(responses["input"][2]["call_id"], "call-B");
    assert_eq!(
        responses["input"][3],
        json!({"type":"function_call_output","call_id":"call-A","output":"result A"})
    );
    assert_eq!(
        responses["input"][4],
        json!({"type":"function_call_output","call_id":"call-B","output":"result B"})
    );
}

/// Raw loopback HTTP, optionally closing before its advertised Content-Length.
async fn mock_response(body: &str, disconnect: bool) -> reqwest::Response {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let body = body.to_string();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut byte = [0];
        while !request.ends_with(b"\r\n\r\n") {
            if socket.read_exact(&mut byte).await.is_err() {
                return;
            }
            request.push(byte[0]);
        }
        let length = body.len() + if disconnect { 50 } else { 0 };
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n"
        );
        socket.write_all(headers.as_bytes()).await.unwrap();
        for bytes in body.as_bytes().chunks(7) {
            if socket.write_all(bytes).await.is_err() {
                return;
            }
            tokio::task::yield_now().await;
        }
    });
    reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap()
        .get(format!("http://{address}"))
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn raw_http_chatgpt_collector_matches_streaming() {
    let buffered = transport::collect_chatgpt(mock_response(CHATGPT, false).await, "m", "r")
        .await
        .unwrap();
    let body = transport::translated_stream(
        mock_response(CHATGPT, false).await,
        "m".into(),
        "r".into(),
        transport::ProviderStream::Chatgpt(Default::default()),
    );
    let bytes = to_bytes(Body::from_stream(body), usize::MAX).await.unwrap();
    let chunks = values(std::str::from_utf8(&bytes).unwrap());
    assert_eq!(
        json!(assemble_calls(&chunks).into_values().collect::<Vec<_>>()),
        buffered["choices"][0]["message"]["tool_calls"]
    );
    assert_eq!(chunks.last().unwrap()["usage"], buffered["usage"]);
}

#[tokio::test]
async fn raw_http_disconnect_and_clean_eof_without_terminal_are_errors() {
    for disconnect in [false, true] {
        let partial = "event: response.output_text.delta\ndata: {\"delta\":\"partial\"}\n\n";
        assert!(
            transport::collect_chatgpt(mock_response(partial, disconnect).await, "m", "r")
                .await
                .is_err()
        );
        for anthropic in [false, true] {
            let raw = if anthropic {
                "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"
            } else {
                partial
            };
            let provider = if anthropic {
                transport::ProviderStream::Anthropic(Default::default())
            } else {
                transport::ProviderStream::Chatgpt(Default::default())
            };
            let body = transport::translated_stream(
                mock_response(raw, disconnect).await,
                "m".into(),
                "r".into(),
                provider,
            );
            let bytes = to_bytes(Body::from_stream(body), usize::MAX).await.unwrap();
            let output = std::str::from_utf8(&bytes).unwrap();
            assert!(!output.contains("[DONE]"));
            let chunks = values(output);
            assert!(
                chunks
                    .iter()
                    .all(|v| v["choices"][0]["finish_reason"].is_null())
            );
            assert!(chunks.last().unwrap().get("error").is_some());
        }
    }
}

#[tokio::test]
async fn raw_http_anthropic_sparse_tool_stream() {
    let body = transport::translated_stream(
        mock_response(ANTHROPIC, false).await,
        "m".into(),
        "r".into(),
        transport::ProviderStream::Anthropic(Default::default()),
    );
    let bytes = to_bytes(Body::from_stream(body), usize::MAX).await.unwrap();
    let output = std::str::from_utf8(&bytes).unwrap();
    assert!(output.ends_with("data: [DONE]\n\n"));
    assert_eq!(
        assemble_calls(&values(output))[&2],
        call("toolu-A", "clock", "{}")
    );
}
