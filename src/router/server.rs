use std::convert::Infallible;
use std::sync::Arc;

use anyhow::{Context, anyhow};
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Json, Response, Sse};
use axum::routing::{get, post};
use axum::{Router, serve};
use futures::{Stream, StreamExt};
use serde::Serialize;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::config::{
    AnthropicProviderConfig, ChatgptProviderConfig, GatewayConfig, OpenaiCompatibleProviderConfig,
    OpenaiProviderConfig, ProviderConfig, XiaomiMimoProviderConfig,
};
use crate::oauth;
use crate::router::model_resolver::ModelResolver;
use crate::translate::anthropic_to_openai::{translate_error, translate_response};
use crate::translate::chatgpt::{self as chatgpt_translate, ChatgptStreamState};
use crate::translate::openai_to_anthropic::translate_request;
use crate::translate::stream::{StreamState, translate_stream_event};
use crate::types::{
    AnthropicErrorResponse, AnthropicRequest, AnthropicResponse, OpenAIRequest, ProviderKind,
};

pub struct AppState {
    pub anthropic: Option<AnthropicProvider>,
    pub chatgpt: Option<ChatgptProvider>,
    pub openai_compat_providers: std::collections::HashMap<String, OpenAICompatProvider>,
    pub model_resolver: ModelResolver,
    pub master_key: Option<String>,
}

pub async fn run(config: GatewayConfig) -> anyhow::Result<()> {
    let model_resolver = ModelResolver::from_config(&config);
    let mut anthropic = None;
    let mut chatgpt = None;
    let mut openai_compat_providers = std::collections::HashMap::new();

    for provider in &config.providers {
        match provider {
            ProviderConfig::Anthropic(cfg) => anthropic = Some(AnthropicProvider::new(cfg.clone())),
            ProviderConfig::Chatgpt(cfg) => chatgpt = Some(ChatgptProvider::new(cfg.clone())),
            ProviderConfig::Openai(cfg) => {
                openai_compat_providers.insert(
                    cfg.name.clone(),
                    OpenAICompatProvider::from_openai(cfg.clone()),
                );
            }
            ProviderConfig::XiaomiMimo(cfg) => {
                openai_compat_providers.insert(
                    cfg.name.clone(),
                    OpenAICompatProvider::from_mimo(cfg.clone()),
                );
            }
            ProviderConfig::OpenaiCompatible(cfg) => {
                openai_compat_providers
                    .insert(cfg.name.clone(), OpenAICompatProvider::new(cfg.clone()));
            }
        }
    }

    let state = Arc::new(AppState {
        anthropic,
        chatgpt,
        openai_compat_providers,
        model_resolver,
        master_key: config.master_key.clone(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/models", get(models))
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state, auth_middleware));

    let listener = tokio::net::TcpListener::bind((config.host.as_str(), config.port))
        .await
        .with_context(|| format!("failed to bind {}:{}", config.host, config.port))?;

    info!(address = %listener.local_addr()?, "llm-gateway listening");
    let _ = std::any::TypeId::of::<Sse<futures::stream::Empty<Result<axum::response::sse::Event, Infallible>>>>();
    serve(listener, app).await.context("server error")?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut discovered: Vec<serde_json::Value> = Vec::new();

    // Anthropic — live discovery
    if let Some(provider) = &state.anthropic {
        match provider.fetch_models().await {
            Ok(models) => discovered.extend(models),
            Err(err) => warn!(error = %err, "failed to fetch Anthropic models"),
        }
    }

    // OpenAI-compat providers (MiMo, OpenAI, etc.) — live discovery via /v1/models
    for (name, provider) in &state.openai_compat_providers {
        match provider.fetch_models().await {
            Ok(models) => discovered.extend(models),
            Err(err) => warn!(provider = %name, error = %err, "failed to fetch models"),
        }
    }

    // ChatGPT — live discovery
    if let Some(provider) = &state.chatgpt {
        match provider.fetch_models().await {
            Ok(models) => discovered.extend(models),
            Err(err) => warn!(error = %err, "failed to fetch ChatGPT models"),
        }
    }

    // Fall back to statically configured routes if nothing discovered
    if discovered.is_empty() {
        let mut model_ids = state
            .model_resolver
            .routes
            .iter()
            .filter(|route| route.strip_prefix.is_none())
            .map(|route| route.pattern.clone())
            .collect::<Vec<_>>();
        model_ids.extend(state.model_resolver.aliases.keys().cloned());
        model_ids.sort();
        model_ids.dedup();
        discovered = model_ids
            .into_iter()
            .map(|id| json!({ "id": id, "object": "model" }))
            .collect();
    }

    Json(json!({ "object": "list", "data": discovered }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<OpenAIRequest>,
) -> Response {
    let request_id = Uuid::new_v4().simple().to_string();
    let started_at = std::time::Instant::now();

    let Some(resolved) = state.model_resolver.resolve(&req.model) else {
        return error_response(StatusCode::BAD_REQUEST, format!("unknown model: {}", req.model));
    };

    // Log the incoming conversation at DEBUG level
    tracing::debug!(
        request_id = %request_id,
        model = %req.model,
        stream = req.stream.unwrap_or(false),
        messages = %{
            req.messages.iter().map(|m| {
                let role = format!("{:?}", m.role).to_lowercase();
                let content = match &m.content {
                    crate::types::OpenAIMessageContent::Text(t) => t.clone(),
                    crate::types::OpenAIMessageContent::Parts(p) => format!("[{} parts]", p.len()),
                    crate::types::OpenAIMessageContent::Null => String::new(),
                };
                format!("{role}: {content}")
            }).collect::<Vec<_>>().join(" | ")
        },
        "→ request"
    );

    let response = match resolved.provider_kind {
        ProviderKind::Anthropic => {
            let Some(provider) = state.anthropic.as_ref() else {
                return error_response(StatusCode::BAD_GATEWAY, "anthropic provider not configured");
            };

            let mut upstream_request = translate_request(&req);
            upstream_request.model = resolved.upstream_model.clone();

            if req.stream.unwrap_or(false) {
                match provider
                    .send_json(
                        "/v1/messages",
                        &upstream_request,
                        extract_auth_header(&headers),
                    )
                    .await
                {
                    Ok(response) => {
                        translate_anthropic_sse(response, req.model.clone(), request_id).await
                    }
                    Err(err) => {
                        error!(error = %err, "anthropic stream request failed");
                        error_response(StatusCode::BAD_GATEWAY, err.to_string())
                    }
                }
            } else {
                match provider
                    .send_json(
                        "/v1/messages",
                        &upstream_request,
                        extract_auth_header(&headers),
                    )
                    .await
                {
                    Ok(response) => match response.status() {
                        status if status.is_success() => match response.json::<AnthropicResponse>().await {
                            Ok(body) => {
                                let content = body.content.iter().filter_map(|b| match b {
                                    crate::types::AnthropicContentBlock::Text { text, cache_control: None } => Some(text.as_str()),
                                    _ => None,
                                }).collect::<Vec<_>>().join("");
                                tracing::debug!(
                                    "← done model={} tokens={}/{} finish={} content={:?}",
                                    req.model,
                                    body.usage.input_tokens,
                                    body.usage.output_tokens,
                                    body.stop_reason.as_deref().unwrap_or("unknown"),
                                    content.clone(),
                                );
                                Json(translate_response(&body, &req.model, &request_id)).into_response()
                            },
                            Err(err) => error_response(StatusCode::BAD_GATEWAY, err.to_string()),
                        },
                        status => match response.json::<AnthropicErrorResponse>().await {
                            Ok(body) => (status, Json(translate_error(&body))).into_response(),
                            Err(err) => error_response(StatusCode::BAD_GATEWAY, err.to_string()),
                        },
                    },
                    Err(err) => {
                        error!(error = %err, "anthropic request failed");
                        error_response(StatusCode::BAD_GATEWAY, err.to_string())
                    }
                }
            }
        }
        ProviderKind::Chatgpt => {
            let Some(provider) = state.chatgpt.as_ref() else {
                return error_response(StatusCode::BAD_GATEWAY, "chatgpt provider not configured");
            };
            let mut upstream_req = chatgpt_translate::to_responses_request(&req);
            upstream_req["model"] = serde_json::Value::String(resolved.upstream_model.clone());

            if req.stream.unwrap_or(false) {
                upstream_req["stream"] = serde_json::Value::Bool(true);
                match provider.send_json("/responses", &upstream_req).await {
                    Ok(response) => {
                        translate_chatgpt_sse(response, req.model.clone(), request_id).await
                    }
                    Err(err) => {
                        error!(error = %err, "chatgpt stream request failed");
                        error_response(StatusCode::BAD_GATEWAY, err.to_string())
                    }
                }
            } else {
                // Codex endpoint requires stream:true — send streaming and collect into one response
                upstream_req["stream"] = serde_json::Value::Bool(true);
                match provider.send_json("/responses", &upstream_req).await {
                    Ok(response) => {
                        if !response.status().is_success() {
                            return proxy_json_response(response).await;
                        }
                        collect_chatgpt_stream(response, req.model.clone(), request_id).await
                    }
                    Err(err) => {
                        error!(error = %err, "chatgpt request failed");
                        error_response(StatusCode::BAD_GATEWAY, err.to_string())
                    }
                }
            }
        }
        ProviderKind::Openai | ProviderKind::XiaomiMimo | ProviderKind::OpenaiCompatible => {
            let Some(provider) = state.openai_compat_providers.get(&resolved.provider_name) else {
                return error_response(StatusCode::BAD_GATEWAY, "provider not configured");
            };
            let mut upstream_request = req.clone();
            upstream_request.model = resolved.upstream_model.clone();
            // XiaoMiMo supports OpenAI-style cache_control markers on content parts
            if resolved.provider_kind == ProviderKind::XiaomiMimo {
                apply_openai_cache_markers(&mut upstream_request.messages);
            }
            proxy_openai(provider, &upstream_request).await
        }
    };

    let elapsed_ms = started_at.elapsed().as_millis();
    let status = response.status();
    let label = if req.stream.unwrap_or(false) { "← response (first byte)" } else { "← response" };
    info!(model = %req.model, status = %status.as_u16(), elapsed_ms = elapsed_ms, "{label}");

    response
}

async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut req): Json<AnthropicRequest>,
) -> Response {
    let Some(provider) = state.anthropic.as_ref() else {
        return error_response(StatusCode::BAD_GATEWAY, "anthropic provider not configured");
    };

    if let Some(resolved) = state.model_resolver.resolve(&req.model) {
        if matches!(resolved.provider_kind, ProviderKind::Anthropic) {
            req.model = resolved.upstream_model;
        }
    }

    match provider
        .send_json("/v1/messages", &req, extract_auth_header(&headers))
        .await
    {
        Ok(response) => {
            if req.stream.unwrap_or(false) {
                passthrough_sse(response)
            } else {
                proxy_json_response(response).await
            }
        }
        Err(err) => {
            error!(error = %err, "anthropic passthrough failed");
            error_response(StatusCode::BAD_GATEWAY, err.to_string())
        }
    }
}

async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    if let Some(master_key) = &state.master_key {
        let provided = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok());
        let expected_bearer = format!("Bearer {master_key}");

        if provided != Some(expected_bearer.as_str()) && provided != Some(master_key.as_str()) {
            return error_response(StatusCode::UNAUTHORIZED, "unauthorized");
        }
    }

    next.run(req).await
}

async fn proxy_openai(provider: &OpenAICompatProvider, req: &OpenAIRequest) -> Response {
    match provider.send_json("/chat/completions", req).await {
        Ok(response) => {
            if req.stream.unwrap_or(false) {
                passthrough_sse(response)
            } else {
                proxy_json_response(response).await
            }
        }
        Err(err) => {
            error!(error = %err, provider = %provider.name, "openai-compatible request failed");
            error_response(StatusCode::BAD_GATEWAY, err.to_string())
        }
    }
}

async fn proxy_json_response(response: reqwest::Response) -> Response {
    let status = response.status();
    let headers = clone_content_type(response.headers());
    match response.bytes().await {
        Ok(body) => build_response(status, headers, Body::from(body)),
        Err(err) => error_response(StatusCode::BAD_GATEWAY, err.to_string()),
    }
}

/// For non-streaming ChatGPT requests: consume the SSE stream and assemble a single JSON response.
async fn collect_chatgpt_stream(
    response: reqwest::Response,
    model: String,
    request_id: String,
) -> Response {
    use futures::StreamExt;

    let mut upstream = response.bytes_stream();
    let mut line_buf = String::new();
    let mut event_type = String::new();
    let mut data_lines: Vec<String> = Vec::new();
    let mut state = ChatgptStreamState::default();

    // Collect all text content and tool calls
    let mut text = String::new();
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut finish_reason = "stop".to_string();

    while let Some(chunk) = upstream.next().await {
        let Ok(chunk) = chunk else { break };
        for byte in chunk {
            let ch = byte as char;
            if ch == '\n' {
                let line = std::mem::take(&mut line_buf);
                let line = line.trim_end_matches('\r');
                if line.is_empty() {
                    if !data_lines.is_empty() {
                        let data = data_lines.join("\n");
                        data_lines.clear();
                        let et = std::mem::take(&mut event_type);
                        // Process event directly
                        match et.as_str() {
                            "response.output_text.delta" => {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                    if let Some(delta) = v.get("delta").and_then(|d| d.as_str()) {
                                        text.push_str(delta);
                                    }
                                }
                            }
                            "response.output_item.added" => {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                    if let Some(item) = v.get("item") {
                                        if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                                            let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or_default();
                                            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or_default();
                                            let idx = state.tool_call_count;
                                            state.tool_call_count += 1;
                                            state.tool_calls.insert(call_id.to_string(), (idx, name.to_string()));
                                            tool_calls.push(json!({
                                                "id": call_id,
                                                "type": "function",
                                                "function": {"name": name, "arguments": ""}
                                            }));
                                        }
                                    }
                                }
                            }
                            "response.function_call_arguments.delta" => {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                    let delta = v.get("delta").and_then(|d| d.as_str()).unwrap_or("");
                                    let item_id = v.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
                                    if let Some((idx, _)) = state.tool_calls.get(item_id) {
                                        if let Some(tc) = tool_calls.get_mut(*idx) {
                                            if let Some(args) = tc["function"]["arguments"].as_str() {
                                                let mut new_args = args.to_string();
                                                new_args.push_str(delta);
                                                tc["function"]["arguments"] = json!(new_args);
                                            }
                                        }
                                    }
                                }
                            }
                            "response.completed" | "response.incomplete" => {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                    if let Some(usage) = v.get("response").and_then(|r| r.get("usage")) {
                                        state.input_tokens = usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                                        state.output_tokens = usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                                    }
                                }
                                finish_reason = if !tool_calls.is_empty() {
                                    "tool_calls".to_string()
                                } else if et == "response.incomplete" {
                                    "length".to_string()
                                } else {
                                    "stop".to_string()
                                };
                            }
                            _ => {}
                        }
                    }
                } else if let Some(val) = line.strip_prefix("event:") {
                    event_type = val.trim().to_string();
                } else if let Some(val) = line.strip_prefix("data:") {
                    data_lines.push(val.trim_start().to_string());
                }
            } else {
                line_buf.push(ch);
            }
        }
    }

    use std::time::{SystemTime, UNIX_EPOCH};
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

    tracing::debug!(
        "← done model={} tokens={}/{} finish={} content={:?}",
        model,
        state.input_tokens,
        state.output_tokens,
        finish_reason,
        text.clone(),
    );

    let mut message = json!({ "role": "assistant", "content": text });
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }

    Json(json!({
        "id": format!("chatcmpl-{request_id}"),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": state.input_tokens,
            "completion_tokens": state.output_tokens,
            "total_tokens": state.input_tokens + state.output_tokens,
        }
    })).into_response()
}

async fn translate_chatgpt_sse(
    response: reqwest::Response,
    model: String,
    request_id: String,
) -> Response {
    let status = response.status();
    if !status.is_success() {
        return proxy_json_response(response).await;
    }

    let stream = stream_responses_sse(response, move |event_type, event_data, state| {
        let chunks = chatgpt_translate::translate_stream_event(event_type, event_data, &model, &request_id, state);
        if chunks.iter().any(|c| c.contains("[DONE]")) {
            tracing::debug!(
                "← done (stream) model={} tokens={}/{} content={:?}",
                model,
                state.input_tokens,
                state.output_tokens,
                state.accumulated_text.clone(),
            );
        }
        chunks
    });
    build_response(status, sse_headers(), Body::from_stream(stream))
}

async fn translate_anthropic_sse(
    response: reqwest::Response,
    model: String,
    request_id: String,
) -> Response {
    let status = response.status();
    if !status.is_success() {
        return proxy_json_response(response).await;
    }

    let stream = stream_response_body(response, move |event_data, state| {
        let chunks = translate_stream_event(event_data, &model, &request_id, state);
        // Log accumulated content when stream completes
        if chunks.iter().any(|c| c.contains("[DONE]")) {
            tracing::debug!(
                "← done (stream) model={} tokens={}/{} content={:?}",
                model,
                state.input_tokens,
                state.output_tokens,
                state.accumulated_text.clone(),
            );
        }
        chunks
    });
    build_response(status, sse_headers(), Body::from_stream(stream))
}

/// SSE parser for Responses API format: handles both `event:` and `data:` lines.
fn stream_responses_sse<F>(
    response: reqwest::Response,
    translator: F,
) -> impl Stream<Item = Result<Bytes, Infallible>>
where
    F: Fn(&str, &str, &mut ChatgptStreamState) -> Vec<String> + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<Bytes>(32);
    tokio::spawn(async move {
        let mut upstream = response.bytes_stream();
        let mut line_buf = String::new();
        let mut event_type = String::new();
        let mut data_lines: Vec<String> = Vec::new();
        let mut state = ChatgptStreamState::default();

        while let Some(chunk) = upstream.next().await {
            let Ok(chunk) = chunk else {
                warn!("chatgpt SSE stream ended with error");
                break;
            };
            for byte in chunk {
                let ch = byte as char;
                if ch == '\n' {
                    let line = std::mem::take(&mut line_buf);
                    let line = line.trim_end_matches('\r');

                    if line.is_empty() {
                        // Dispatch event
                        if !data_lines.is_empty() {
                            let data = data_lines.join("\n");
                            data_lines.clear();
                            let et = std::mem::take(&mut event_type);
                            for out in translator(&et, &data, &mut state) {
                                if tx.send(Bytes::from(out)).await.is_err() {
                                    return;
                                }
                            }
                        }
                    } else if let Some(val) = line.strip_prefix("event:") {
                        event_type = val.trim().to_string();
                    } else if let Some(val) = line.strip_prefix("data:") {
                        data_lines.push(val.trim_start().to_string());
                    }
                } else {
                    line_buf.push(ch);
                }
            }
        }
    });
    futures::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|b| (Ok(b), rx)) })
}

fn passthrough_sse(response: reqwest::Response) -> Response {
    let status = response.status();
    let stream = response.bytes_stream().map(|result| {
        result
            .map(Bytes::from)
            .map_err(|err| anyhow!(err))
    });
    build_response(status, sse_headers(), Body::from_stream(stream))
}

fn stream_response_body<F>(response: reqwest::Response, translator: F) -> impl Stream<Item = Result<Bytes, Infallible>>
where
    F: Fn(&str, &mut StreamState) -> Vec<String> + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<Bytes>(32);
    tokio::spawn(async move {
        let mut upstream = response.bytes_stream();
        let mut event_buffer = String::new();
        let mut data_lines = Vec::new();
        let mut state = StreamState::default();

        while let Some(chunk) = upstream.next().await {
            let Ok(chunk) = chunk else {
                warn!("upstream SSE stream ended with an error");
                break;
            };

            for byte in chunk {
                let ch = byte as char;
                event_buffer.push(ch);

                if ch != '\n' {
                    continue;
                }

                let line = event_buffer.trim_end_matches('\n').trim_end_matches('\r').to_string();
                event_buffer.clear();

                if line.is_empty() {
                    if data_lines.is_empty() {
                        continue;
                    }

                    let data = data_lines.join("\n");
                    data_lines.clear();

                    if data == "[DONE]" {
                        let _ = tx.send(Bytes::from_static(b"data: [DONE]\n\n")).await;
                        continue;
                    }

                    for event in translator(&data, &mut state) {
                        if tx.send(Bytes::from(event)).await.is_err() {
                            return;
                        }
                    }
                } else if let Some(data) = line.strip_prefix("data:") {
                    data_lines.push(data.trim_start().to_string());
                }
            }
        }

        if !data_lines.is_empty() {
            let data = data_lines.join("\n");
            for event in translator(&data, &mut state) {
                if tx.send(Bytes::from(event)).await.is_err() {
                    return;
                }
            }
        }
    });

    futures::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|item| (Ok(item), rx)) })
}

fn build_response(status: reqwest::StatusCode, headers: Vec<(header::HeaderName, HeaderValue)>, body: Body) -> Response {
    let mut builder = Response::builder().status(status);
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    builder.body(body).unwrap_or_else(|_| Response::new(Body::from("internal error")))
}

fn clone_content_type(headers: &reqwest::header::HeaderMap) -> Vec<(header::HeaderName, HeaderValue)> {
    headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| HeaderValue::from_bytes(value.as_bytes()).ok())
        .map(|value| vec![(header::CONTENT_TYPE, value)])
        .unwrap_or_default()
}

fn sse_headers() -> Vec<(header::HeaderName, HeaderValue)> {
    vec![(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"))]
}

/// Apply cache_control markers to the last 3 messages for providers that support it (e.g. XiaoMiMo).
/// Converts string content to a content-parts array with cache_control on the last part.
fn apply_openai_cache_markers(messages: &mut Vec<crate::types::OpenAIMessage>) {
    let cache_control = serde_json::json!({"type": "ephemeral"});
    let n = messages.len();
    let targets: Vec<usize> = (0..3).filter_map(|i| n.checked_sub(i + 1)).collect();

    for idx in targets {
        if let Some(msg) = messages.get_mut(idx) {
            match &msg.content {
                crate::types::OpenAIMessageContent::Text(t) => {
                    let text = t.clone();
                    msg.content = crate::types::OpenAIMessageContent::Parts(vec![
                        crate::types::OpenAIContentPart::Text {
                            text,
                            cache_control: Some(cache_control.clone()),
                        },
                    ]);
                }
                crate::types::OpenAIMessageContent::Parts(parts) => {
                    let mut new_parts = parts.clone();
                    if let Some(last) = new_parts.last_mut() {
                        if let crate::types::OpenAIContentPart::Text { cache_control: cc, .. } = last {
                            *cc = Some(cache_control.clone());
                        }
                    }
                    msg.content = crate::types::OpenAIMessageContent::Parts(new_parts);
                }
                crate::types::OpenAIMessageContent::Null => {}
            }
        }
    }
}

fn extract_auth_header(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    let body = Json(json!({
        "error": {
            "message": message.into(),
            "type": "gateway_error"
        }
    }));
    (status, body).into_response()
}

#[derive(Clone)]
pub struct AnthropicProvider {
    name: String,
    api_base: String,
    api_key: Option<String>,
    allow_bearer_passthrough: bool,
    token_store: Option<Arc<crate::oauth::token_store::TokenStore>>,
    client: reqwest::Client,
}

impl AnthropicProvider {
    fn new(config: AnthropicProviderConfig) -> Self {
        let token_path = config.oauth_token_file.clone().or_else(|| {
            dirs::config_dir().map(|d| d.join("llm-gateway").join("anthropic-oauth.json"))
        });
        let token_store = token_path
            .map(|path| Arc::new(crate::oauth::token_store::TokenStore::new(path)));
        Self {
            name: config.name,
            api_base: config.api_base,
            api_key: config.api_key,
            allow_bearer_passthrough: config.allow_bearer_passthrough,
            token_store,
            client: reqwest::Client::new(),
        }
    }

    /// Get a valid access token, refreshing from the token store if needed.
    async fn get_access_token(&self) -> anyhow::Result<Option<String>> {
        // Try token store first (OAuth token file with auto-refresh)
        if let Some(store) = &self.token_store {
            if let Some(tokens) = store.load().await? {
                if !crate::oauth::token_store::TokenStore::is_expired(&tokens) {
                    return Ok(Some(tokens.access_token));
                }
                // Token expired — try to refresh
                if let Some(refresh_token) = &tokens.refresh_token {
                    info!(provider = %self.name, "refreshing expired Anthropic OAuth token");
                    match oauth::anthropic::refresh_token(&self.client, refresh_token).await {
                        Ok(mut refreshed) => {
                            // Preserve old refresh token if new response doesn't include one
                            if refreshed.refresh_token.is_none() {
                                refreshed.refresh_token = tokens.refresh_token;
                            }
                            store.save(&refreshed).await?;
                            return Ok(Some(refreshed.access_token));
                        }
                        Err(err) => {
                            warn!(provider = %self.name, error = %err, "OAuth token refresh failed, falling back to API key");
                        }
                    }
                }
            }
        }
        // Fall back to static API key
        Ok(self.api_key.clone())
    }

    async fn send_json<T: Serialize>(
        &self,
        path: &str,
        body: &T,
        passthrough_auth: Option<String>,
    ) -> anyhow::Result<reqwest::Response> {
        let mut request = self
            .client
            .post(join_url(&self.api_base, path))
            .header("anthropic-version", "2023-06-01")
            .json(body);

        if let Some(auth_value) = passthrough_auth.filter(|_| self.allow_bearer_passthrough) {
            request = request.header(header::AUTHORIZATION, auth_value);
        } else if let Some(api_key) = self.get_access_token().await? {
            if oauth::anthropic::is_oauth_token(&api_key) {
                request = request
                    .bearer_auth(&api_key)
                    .header("anthropic-beta", "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14,prompt-caching-2024-07-31")
                    .header("user-agent", "claude-cli/2.1.80 (external, cli)")
                    .header("x-app", "cli");
            } else {
                request = request.header("x-api-key", &api_key);
            }
        }

        request
            .send()
            .await
            .with_context(|| format!("request to Anthropic provider {} failed", self.name))
    }

    /// Fetch available models from the Anthropic API and return them in OpenAI model list format.
    pub async fn fetch_models(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let api_key = self
            .get_access_token()
            .await?
            .ok_or_else(|| anyhow!("no Anthropic credentials configured"))?;

        let mut request = self
            .client
            .get(join_url(&self.api_base, "/v1/models"))
            .header("anthropic-version", "2023-06-01");

        if oauth::anthropic::is_oauth_token(&api_key) {
            request = request
                .bearer_auth(&api_key)
                .header("anthropic-beta", "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14,prompt-caching-2024-07-31")
                .header("user-agent", "claude-cli/2.1.80 (external, cli)")
                .header("x-app", "cli");
        } else {
            request = request.header("x-api-key", &api_key);
        }

        let resp = request.send().await.context("Anthropic /v1/models request failed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic /v1/models returned {status}: {body}");
        }

        let body: serde_json::Value = resp.json().await.context("failed to decode Anthropic models")?;
        let models = body
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_str()?;
                        Some(json!({
                            "id": format!("anthropic/{id}"),
                            "object": "model",
                            "owned_by": "anthropic",
                            "created": m.get("created_at").cloned().unwrap_or(json!(0)),
                        }))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}

#[derive(Clone)]
pub struct ChatgptProvider {
    name: String,
    api_base: String,
    token_store: Arc<crate::oauth::token_store::TokenStore>,
    client: reqwest::Client,
}

impl ChatgptProvider {
    fn new(config: ChatgptProviderConfig) -> Self {
        let token_path = config.token_file.unwrap_or_else(|| {
            dirs::config_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("llm-gateway")
                .join("chatgpt-auth.json")
        });
        Self {
            name: config.name,
            api_base: config.api_base,
            token_store: Arc::new(crate::oauth::token_store::TokenStore::new(token_path)),
            client: reqwest::Client::new(),
        }
    }

    /// Get a valid access token, refreshing if expired.
    async fn get_access_token(&self) -> anyhow::Result<String> {
        let tokens = self
            .token_store
            .load()
            .await?
            .ok_or_else(|| anyhow!("ChatGPT not logged in. Run: llm-gateway login chatgpt"))?;

        if !crate::oauth::token_store::TokenStore::is_expired_with_buffer(&tokens, 60) {
            return Ok(tokens.access_token);
        }

        // Token expired — refresh
        let refresh_tok = tokens.refresh_token.as_deref().ok_or_else(|| {
            anyhow!("ChatGPT token expired and no refresh token. Run: llm-gateway login chatgpt")
        })?;

        info!(provider = %self.name, "refreshing expired ChatGPT OAuth token");
        let mut refreshed = crate::oauth::chatgpt::refresh_token(&self.client, refresh_tok).await?;

        if refreshed.refresh_token.is_none() {
            refreshed.refresh_token = tokens.refresh_token;
        }

        self.token_store.save(&refreshed).await?;
        Ok(refreshed.access_token)
    }

    pub async fn fetch_models(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let access_token = self.get_access_token().await?;
        let resp = self.client
            .get("https://chatgpt.com/backend-api/codex/models")
            .query(&[("client_version", "1.0.0")])
            .bearer_auth(&access_token)
            .header("originator", "codex_cli_rs")
            .send()
            .await
            .context("ChatGPT /models request failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("ChatGPT /models returned {}", resp.status());
        }

        let body: serde_json::Value = resp.json().await.context("failed to decode ChatGPT models")?;
        let models = body
            .get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter(|m| m.get("visibility").and_then(|v| v.as_str()) == Some("list"))
                    .filter_map(|m| {
                        let slug = m.get("slug")?.as_str()?;
                        let name = m.get("display_name").and_then(|v| v.as_str()).unwrap_or(slug);
                        Some(json!({
                            "id": format!("chatgpt/{slug}"),
                            "object": "model",
                            "owned_by": "openai",
                            "display_name": name,
                        }))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    async fn send_json<T: Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> anyhow::Result<reqwest::Response> {
        let access_token = self.get_access_token().await?;
        let account_id = self
            .token_store
            .load()
            .await?
            .and_then(|t| t.extra.get("account_id").and_then(|v| v.as_str().map(String::from)));

        let mut request = self
            .client
            .post(join_url(&self.api_base, path))
            .bearer_auth(&access_token)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header("originator", "codex_cli_rs")
            .header("user-agent", "codex_cli_rs/0.0.0 (Unknown 0; unknown) unknown")
            .json(body);

        if let Some(account_id) = &account_id {
            request = request.header("ChatGPT-Account-Id", account_id);
        }

        request
            .send()
            .await
            .with_context(|| format!("request to ChatGPT provider {} failed", self.name))
    }
}

#[derive(Clone)]
pub struct OpenAICompatProvider {
    name: String,
    api_base: String,
    api_key: Option<String>,
    auth_header: Option<String>,
    client: reqwest::Client,
}

impl OpenAICompatProvider {
    fn new(config: OpenaiCompatibleProviderConfig) -> Self {
        Self {
            name: config.name,
            api_base: config.api_base,
            api_key: config.api_key,
            auth_header: config.auth_header,
            client: reqwest::Client::new(),
        }
    }

    fn from_openai(config: OpenaiProviderConfig) -> Self {
        Self {
            name: config.name,
            api_base: config.api_base,
            api_key: config.api_key,
            auth_header: None,
            client: reqwest::Client::new(),
        }
    }

    fn from_mimo(config: XiaomiMimoProviderConfig) -> Self {
        Self {
            name: config.name,
            api_base: config.api_base,
            api_key: config.api_key,
            auth_header: None,
            client: reqwest::Client::new(),
        }
    }

    async fn send_json<T: Serialize>(&self, path: &str, body: &T) -> anyhow::Result<reqwest::Response> {
        let mut request = self.client.post(join_url(&self.api_base, path)).json(body);

        if let Some(api_key) = &self.api_key {
            if let Some(auth_header) = &self.auth_header {
                request = request.header(auth_header, api_key);
            } else {
                request = request.bearer_auth(api_key);
            }
        }

        request
            .send()
            .await
            .with_context(|| format!("request to provider {} failed", self.name))
    }

    pub async fn fetch_models(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut request = self.client.get(join_url(&self.api_base, "/models"));

        if let Some(api_key) = &self.api_key {
            if let Some(auth_header) = &self.auth_header {
                request = request.header(auth_header.as_str(), api_key.as_str());
            } else {
                request = request.bearer_auth(api_key);
            }
        }

        let resp = request.send().await
            .with_context(|| format!("GET /models failed for {}", self.name))?;

        if !resp.status().is_success() {
            anyhow::bail!("{} /models returned {}", self.name, resp.status());
        }

        let body: serde_json::Value = resp.json().await
            .with_context(|| format!("failed to decode {} /models response", self.name))?;

        let prefix = self.name.replace('-', "_");
        let models = body
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_str()?;
                        // Use mimo/ prefix for xiaomi-mimo, otherwise use name/
                        let prefixed = if self.name.contains("mimo") {
                            format!("mimo/{id}")
                        } else {
                            format!("{prefix}/{id}")
                        };
                        Some(json!({
                            "id": prefixed,
                            "object": "model",
                            "owned_by": m.get("owned_by").cloned().unwrap_or(json!(self.name)),
                        }))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}

fn join_url(base: &str, path: &str) -> String {
    format!("{}{}", base.trim_end_matches('/'), path)
}
