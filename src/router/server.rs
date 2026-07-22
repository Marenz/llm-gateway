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
    MoonshotProviderConfig, OpenaiProviderConfig, OpencodeGoProviderConfig, ProviderConfig,
    XiaomiMimoProviderConfig,
};
use crate::oauth;
use crate::providers::deepseek::DeepSeekProvider;
use crate::providers::deepinfra::DeepInfraProvider;
use crate::providers::zen::ZenProvider;
use crate::router::model_resolver::ModelResolver;
use crate::translate::anthropic_to_openai::{translate_error, translate_response};
use crate::translate::chatgpt::{self as chatgpt_translate, ChatgptStreamState};
use crate::translate::openai_to_anthropic::translate_request;
use crate::translate::stream::{StreamState, translate_stream_event};
use crate::types::{
    AnthropicErrorResponse, AnthropicRequest, AnthropicResponse, OpenAIRequest, ProviderKind,
};

pub struct AppState {
    pub anthropic_pool: AnthropicPool,
    pub chatgpt: Option<ChatgptProvider>,
    pub openai_compat_providers: std::collections::HashMap<String, OpenAICompatProvider>,
    pub opencode_go: Option<OpenCodeGoProvider>,
    pub zen: Option<ZenProvider>,
    pub deepseek: Option<DeepSeekProvider>,
    pub deepinfra: Option<DeepInfraProvider>,
    pub model_resolver: ModelResolver,
    pub master_key: Option<String>,
    /// Virtual model name -> ordered list of real target model names.
    pub virtual_models: std::collections::HashMap<String, Vec<String>>,
}

/// Decision returned by [`classify_failover_status`]: whether to return the
/// response immediately or advance to the next failover target.
#[derive(Debug, PartialEq)]
pub enum FailoverDecision {
    /// Return this response to the client immediately (success or non-retryable client error).
    Return,
    /// Retry with the next target (rate-limit, server error, or transport failure).
    Advance,
}

/// Classify an HTTP status code for virtual-model failover purposes.
///
/// - 2xx, 400, 401, 403, 422 → Return immediately (a malformed request or an
///   auth failure won't be fixed by trying a different model on the same chain).
/// - 402, 404, 408, 429, 5xx → Advance to the next target. These all mean "this
///   particular target can't serve the request right now" — model not found,
///   account out of balance/credit, request timeout, rate-limited, or upstream
///   server error — exactly the cases a backup provider should cover.
pub fn classify_failover_status(status: reqwest::StatusCode) -> FailoverDecision {
    if status.is_success() {
        return FailoverDecision::Return;
    }
    match status.as_u16() {
        // Bad-request / auth errors won't be fixed by a different target.
        400 | 401 | 403 | 422 => FailoverDecision::Return,
        // Provider-unusable conditions → advance to the next target:
        //   402 = out of balance/credit, 404 = model not found,
        //   408 = upstream timeout, 429 = rate-limited.
        402 | 404 | 408 | 429 => FailoverDecision::Advance,
        // Upstream server errors are retryable elsewhere.
        s if s >= 500 => FailoverDecision::Advance,
        // All other client errors: return as-is.
        _ => FailoverDecision::Return,
    }
}

pub async fn run(config: GatewayConfig) -> anyhow::Result<()> {
    let model_resolver = ModelResolver::from_config(&config);
    let mut anthropic_providers = Vec::new();
    let mut chatgpt = None;
    let mut openai_compat_providers = std::collections::HashMap::new();
    let mut opencode_go = None;
    let mut zen = None;
    let mut deepseek = None;
    let mut deepinfra = None;

    for provider in &config.providers {
        match provider {
            ProviderConfig::Anthropic(cfg) => anthropic_providers.push(AnthropicProvider::new(cfg.clone())),
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
            ProviderConfig::OpencodeGo(cfg) => {
                let mut provider = OpenCodeGoProvider::new(cfg.clone());
                provider.refresh_anthropic_models().await;
                opencode_go = Some(provider);
            }
            ProviderConfig::Zen(cfg) => {
                zen = Some(ZenProvider::new(cfg.clone()));
            }
            ProviderConfig::DeepSeek(cfg) => {
                deepseek = Some(DeepSeekProvider::new(cfg.clone()));
            }
            ProviderConfig::DeepInfra(cfg) => {
                deepinfra = Some(DeepInfraProvider::new(cfg.clone()));
            }
            ProviderConfig::Moonshot(cfg) => {
                openai_compat_providers.insert(
                    cfg.name.clone(),
                    OpenAICompatProvider::from_moonshot(cfg.clone()),
                );
            }
        }
    }

    let state = Arc::new(AppState {
        anthropic_pool: AnthropicPool::new(anthropic_providers),
        chatgpt,
        openai_compat_providers,
        opencode_go,
        zen,
        deepseek,
        deepinfra,
        model_resolver,
        master_key: config.master_key.clone(),
        virtual_models: config.virtual_models.clone(),
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
    // Anthropic â live discovery from pool
    if !state.anthropic_pool.is_empty() {
        match state.anthropic_pool.fetch_models().await {
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

    // OpenCode Go — live discovery
    if let Some(provider) = &state.opencode_go {
        match provider.fetch_models().await {
            Ok(models) => discovered.extend(models),
            Err(err) => warn!(error = %err, "failed to fetch OpenCode Go models"),
        }
    }

    // Zen — live discovery
    if let Some(provider) = &state.zen {
        match provider.fetch_models().await {
            Ok(models) => discovered.extend(models),
            Err(err) => warn!(error = %err, "failed to fetch Zen models"),
        }
    }

    // DeepSeek — live discovery
    if let Some(provider) = &state.deepseek {
        match provider.fetch_models().await {
            Ok(models) => discovered.extend(models),
            Err(err) => warn!(error = %err, "failed to fetch DeepSeek models"),
        }
    }

    // DeepInfra — live discovery
    if let Some(provider) = &state.deepinfra {
        match provider.fetch_models().await {
            Ok(models) => discovered.extend(models),
            Err(err) => warn!(error = %err, "failed to fetch DeepInfra models"),
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

    let response = if let Some(targets) = state.virtual_models.get(&req.model) {
        // Virtual model: try each target in order, advancing on retryable failures.
        virtual_model_dispatch(&state, &headers, &req, &request_id, targets).await
    } else {
        // Normal model: resolve once and dispatch.
        let Some(resolved) = state.model_resolver.resolve(&req.model) else {
            return error_response(StatusCode::BAD_REQUEST, format!("unknown model: {}", req.model));
        };
        dispatch_resolved(&state, resolved, &req, &headers, &request_id).await
    };

    let elapsed_ms = started_at.elapsed().as_millis();
    let status = response.status();
    let label = if req.stream.unwrap_or(false) { "← response (first byte)" } else { "← response" };
    info!(model = %req.model, status = %status.as_u16(), elapsed_ms = elapsed_ms, "{label}");

    response
}

/// Dispatch a request for a virtual model, iterating over the target list and
/// failing over on retryable errors (429 / 5xx / transport failure).
///
/// **Streaming note:** For streaming virtual-model requests we force non-streaming
/// upstream calls so we can inspect the HTTP status before committing to a
/// streaming response body. If all targets succeed with 2xx we return the last
/// non-streaming response (the client receives a normal JSON completion instead
/// of an SSE stream). This trades streaming for correctness in failover scenarios;
/// a single-target virtual model used purely for model aliasing will still stream
/// correctly because the first attempt will succeed and streaming is enabled below
/// only when there is exactly one remaining candidate that returned 2xx.
///
/// Concretely: we attempt each target with `stream: false`. On the first 2xx we
/// return that JSON response. The client (e.g. opencode) handles non-streamed
/// responses just fine.
async fn virtual_model_dispatch(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    req: &OpenAIRequest,
    request_id: &str,
    targets: &[String],
) -> Response {
    let mut last_response: Option<Response> = None;
    let mut tried_targets: Vec<String> = Vec::new();

    // Whether the CLIENT asked for streaming. We force non-streaming upstream so
    // we can inspect status for failover, but if the client wanted SSE we must
    // re-emit the chosen completion as a stream (clients with SSE-only parsers,
    // e.g. adapsis llm_takeover, get 0 content from a raw JSON body otherwise).
    let client_wants_stream = req.stream.unwrap_or(false);

    // Guard against self-referential virtual models: cap at list length.
    let max_attempts = targets.len();

    for (attempt, target) in targets.iter().enumerate().take(max_attempts) {
        // Guard: virtual model must not reference itself by the same name as the
        // outer virtual model key (prevents trivial infinite loops).
        // Deeper cycles (A → virtual B → virtual A) are not guarded here; that
        // would require a per-request visited set and is out of scope.

        // Build a non-streaming per-target request so we can inspect status.
        let mut target_req = req.clone();
        target_req.model = target.clone();
        // Force non-streaming: see function doc above.
        target_req.stream = Some(false);

        let Some(resolved) = state.model_resolver.resolve(target) else {
            warn!(target = %target, "virtual model target failed to resolve, skipping");
            tried_targets.push(format!("{target} (unresolvable)"));
            continue;
        };

        info!(
            attempt = attempt + 1,
            total = max_attempts,
            target = %target,
            "virtual model: trying target"
        );

        let response = dispatch_resolved(state, resolved, &target_req, headers, request_id).await;
        let status = response.status();

        match classify_failover_status(reqwest::StatusCode::from_u16(status.as_u16()).unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR)) {
            FailoverDecision::Return => {
                if client_wants_stream && status.is_success() {
                    return json_completion_to_sse(response, request_id).await;
                }
                return response;
            }
            FailoverDecision::Advance => {
                warn!(
                    target = %target,
                    status = status.as_u16(),
                    "virtual model target failed, advancing to next"
                );
                tried_targets.push(format!("{target} ({})", status.as_u16()));
                last_response = Some(response);
            }
        }
    }

    // All targets exhausted — return a 502 listing what was tried.
    let tried_str = tried_targets.join(", ");
    warn!(tried = %tried_str, "virtual model: all targets exhausted");
    last_response.unwrap_or_else(|| {
        error_response(
            StatusCode::BAD_GATEWAY,
            format!("all virtual model targets failed: {tried_str}"),
        )
    })
}

/// Convert a non-streamed OpenAI chat-completion `Response` into a minimal SSE
/// stream (one content delta chunk + `[DONE]`). Used when a virtual model was
/// dispatched non-streaming (for failover) but the client requested streaming,
/// so SSE-only clients still receive the completion.
async fn json_completion_to_sse(response: Response, request_id: &str) -> Response {
    let status = response.status();
    let body_bytes = match axum::body::to_bytes(response.into_body(), usize::MAX).await {
        Ok(b) => b,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("failed to read virtual-model response body: {e}"),
            )
        }
    };

    let parsed: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => {
            // Not JSON we understand — fall back to returning it verbatim.
            return build_response(status, vec![(header::CONTENT_TYPE, HeaderValue::from_static("application/json"))], Body::from(body_bytes));
        }
    };

    let content = parsed
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let model = parsed.get("model").and_then(|m| m.as_str()).unwrap_or("");
    let id = parsed
        .get("id")
        .and_then(|i| i.as_str())
        .unwrap_or(request_id);
    let created = parsed.get("created").and_then(|c| c.as_i64()).unwrap_or(0);

    // First chunk: role+content delta. Second: finish. Then [DONE].
    let chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": serde_json::Value::Null}],
    });
    let finish = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    });
    let sse = format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        chunk, finish
    );
    build_response(reqwest::StatusCode::OK, sse_headers(), Body::from(sse))
}

/// Dispatch a single already-resolved model request to its provider.
///
/// This is the canonical dispatch path shared by both the normal flow and the
/// virtual-model failover loop. The `req.model` field should already have been
/// set to the target model name (or left as-is for non-virtual requests).
async fn dispatch_resolved(
    state: &Arc<AppState>,
    resolved: crate::types::ResolvedModel,
    req: &OpenAIRequest,
    headers: &HeaderMap,
    request_id: &str,
) -> Response {
    match resolved.provider_kind {
        ProviderKind::Anthropic => {
            if state.anthropic_pool.is_empty() {
                return error_response(StatusCode::BAD_GATEWAY, "anthropic provider not configured");
            }

            let mut upstream_request = translate_request(req);
            upstream_request.model = resolved.upstream_model.clone();

            // Try with failover across providers
            let auth_header = extract_auth_header(headers);
            anthropic_with_failover(state, &upstream_request, auth_header, req, request_id).await
        }
        ProviderKind::Chatgpt => {
            let Some(provider) = state.chatgpt.as_ref() else {
                return error_response(StatusCode::BAD_GATEWAY, "chatgpt provider not configured");
            };
            let mut upstream_req = chatgpt_translate::to_responses_request(req);
            upstream_req["model"] = serde_json::Value::String(resolved.upstream_model.clone());

            if req.stream.unwrap_or(false) {
                upstream_req["stream"] = serde_json::Value::Bool(true);
                match provider.send_json("/responses", &upstream_req).await {
                    Ok(response) => {
                        translate_chatgpt_sse(response, req.model.clone(), request_id.to_string()).await
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
                        collect_chatgpt_stream(response, req.model.clone(), request_id.to_string()).await
                    }
                    Err(err) => {
                        error!(error = %err, "chatgpt request failed");
                        error_response(StatusCode::BAD_GATEWAY, err.to_string())
                    }
                }
            }
        }
        ProviderKind::Openai
        | ProviderKind::XiaomiMimo
        | ProviderKind::OpenaiCompatible
        | ProviderKind::Moonshot => {
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
        ProviderKind::OpenCodeGo => {
            let Some(provider) = state.opencode_go.as_ref() else {
                return error_response(StatusCode::BAD_GATEWAY, "opencode-go provider not configured");
            };
            if provider.uses_anthropic_format(&resolved.upstream_model) {
                // Anthropic-compatible models (MiniMax)
                let mut upstream_request = translate_request(req);
                upstream_request.model = resolved.upstream_model.clone();
                match provider
                    .send_anthropic_json("/v1/messages", &upstream_request)
                    .await
                {
                    Ok(response) => {
                        if req.stream.unwrap_or(false) {
                            translate_anthropic_sse(response, req.model.clone(), request_id.to_string()).await
                        } else {
                            handle_anthropic_json_response(response, req, request_id).await
                        }
                    }
                    Err(err) => {
                        error!(error = %err, "opencode-go anthropic request failed");
                        error_response(StatusCode::BAD_GATEWAY, err.to_string())
                    }
                }
            } else {
                // OpenAI-compatible models (GLM-5, Kimi, MiMo)
                let mut upstream_request = req.clone();
                upstream_request.model = resolved.upstream_model.clone();
                match provider
                    .send_openai_json("/v1/chat/completions", &upstream_request)
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
                        error!(error = %err, "opencode-go request failed");
                        error_response(StatusCode::BAD_GATEWAY, err.to_string())
                    }
                }
            }
        }
        ProviderKind::Zen => {
            let Some(provider) = state.zen.as_ref() else {
                return error_response(StatusCode::BAD_GATEWAY, "zen provider not configured");
            };
            // Zen models: most use /v1/chat/completions, Claude uses /v1/messages
            // For simplicity, try chat/completions first (works for most models)
            let mut upstream_request = req.clone();
            upstream_request.model = resolved.upstream_model.clone();
            match provider
                .send_openai_json("/v1/chat/completions", &upstream_request)
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
                    error!(error = %err, "zen request failed");
                    error_response(StatusCode::BAD_GATEWAY, err.to_string())
                }
            }
        }
        ProviderKind::DeepSeek => {
            let Some(provider) = state.deepseek.as_ref() else {
                return error_response(StatusCode::BAD_GATEWAY, "deepseek provider not configured");
            };
            let mut upstream_request = req.clone();
            upstream_request.model = resolved.upstream_model.clone();
            match provider
                .send_json("/chat/completions", &upstream_request)
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
                    error!(error = %err, "deepseek request failed");
                    error_response(StatusCode::BAD_GATEWAY, err.to_string())
                }
            }
        }
        ProviderKind::DeepInfra => {
            let Some(provider) = state.deepinfra.as_ref() else {
                return error_response(StatusCode::BAD_GATEWAY, "deepinfra provider not configured");
            };
            let mut upstream_request = req.clone();
            upstream_request.model = resolved.upstream_model.clone();
            match provider
                .send_json("/chat/completions", &upstream_request)
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
                    error!(error = %err, "deepinfra request failed");
                    error_response(StatusCode::BAD_GATEWAY, err.to_string())
                }
            }
        }
    }
}

async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut req): Json<AnthropicRequest>,
) -> Response {
    let Some(provider) = state.anthropic_pool.active_provider() else {
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

/// Try sending an Anthropic request with automatic failover across pool providers.
async fn anthropic_with_failover(
    state: &Arc<AppState>,
    upstream_request: &AnthropicRequest,
    auth_header: Option<String>,
    original_req: &OpenAIRequest,
    request_id: &str,
) -> Response {
    let is_stream = original_req.stream.unwrap_or(false);

    // Try each provider, starting with the best available
    let mut tried = std::collections::HashSet::new();
    loop {
        let Some((idx, provider)) = state.anthropic_pool.pick_provider().await else {
            return error_response(StatusCode::BAD_GATEWAY, "no Anthropic providers available");
        };

        if !tried.insert(idx) {
            // Already tried all providers â return the error from the last one
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "all Anthropic providers are rate-limited",
            );
        }

        info!(provider = %provider.name, "trying Anthropic provider");

        match provider
            .send_json("/v1/messages", upstream_request, auth_header.clone())
            .await
        {
            Ok(response) => {
                let status = response.status();

                // On rate-limit or server error, record failure and try next provider
                if status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || status.is_server_error()
                {
                    state.anthropic_pool.record_failure(idx, &response).await;
                    continue;
                }

                // Success or client error (4xx) Ã¢ return the response
                if is_stream {
                    return translate_anthropic_sse(
                        response,
                        original_req.model.clone(),
                        request_id.to_string(),
                    )
                    .await;
                } else {
                    return handle_anthropic_json_response(
                        response,
                        original_req,
                        request_id,
                    )
                    .await;
                }
            }
            Err(err) => {
                error!(provider = %provider.name, error = %err, "anthropic request failed");
                // Network error Ã¢ record a short cooldown and try next
                let mut available = state.anthropic_pool.available_at.lock().await;
                if idx < available.len() {
                    available[idx] = now_secs() + 10;
                }
                drop(available);
                continue;
            }
        }
    }
}

/// Handle a successful (non-streaming) Anthropic JSON response.
async fn handle_anthropic_json_response(
    response: reqwest::Response,
    original_req: &OpenAIRequest,
    request_id: &str,
) -> Response {
    match response.status() {
        status if status.is_success() => match response.json::<AnthropicResponse>().await {
            Ok(body) => {
                let content = body
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        crate::types::AnthropicContentBlock::Text { text, .. } => {
                            Some(text.as_str())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                tracing::debug!(
                    "Ã¢ done model={} tokens={}/{} finish={} content={:?}",
                    original_req.model,
                    body.usage.input_tokens,
                    body.usage.output_tokens,
                    body.stop_reason.as_deref().unwrap_or("unknown"),
                    content,
                );
                Json(translate_response(&body, &original_req.model, request_id)).into_response()
            }
            Err(err) => error_response(StatusCode::BAD_GATEWAY, err.to_string()),
        },
        status => match response.json::<AnthropicErrorResponse>().await {
            Ok(body) => (status, Json(translate_error(&body))).into_response(),
            Err(err) => error_response(StatusCode::BAD_GATEWAY, err.to_string()),
        },
    }
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
    // Accumulate raw bytes per line and decode as UTF-8 at line end. Decoding
    // byte-by-byte via `byte as char` would Latin-1-mangle any multi-byte UTF-8
    // (e.g. an emoji f0 9f 91 8b became four separate code points -> mojibake).
    let mut line_buf: Vec<u8> = Vec::new();
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
            if byte == b'\n' {
                let line_bytes = std::mem::take(&mut line_buf);
                let line = String::from_utf8_lossy(&line_bytes);
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
                line_buf.push(byte);
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
        let mut raw_buf: Vec<u8> = Vec::new();

        while let Some(chunk) = upstream.next().await {
            let Ok(chunk) = chunk else {
                warn!("chatgpt SSE stream ended with error");
                break;
            };
            raw_buf.extend_from_slice(&chunk);
            let valid_up_to = match std::str::from_utf8(&raw_buf) {
                Ok(_) => raw_buf.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid_up_to == 0 {
                continue;
            }
            let text = std::str::from_utf8(&raw_buf[..valid_up_to]).unwrap();
            for ch in text.chars() {
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
            raw_buf.drain(..valid_up_to);
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
        // Buffer for incomplete UTF-8 sequences at chunk boundaries
        let mut raw_buf: Vec<u8> = Vec::new();

        let mut chunk_count: u64 = 0;
        let stream_start = std::time::Instant::now();
        while let Some(chunk) = upstream.next().await {
            let Ok(chunk) = chunk else {
                warn!("upstream SSE stream ended with an error after {chunk_count} chunks");
                break;
            };
            chunk_count += 1;
            if chunk_count <= 3 || chunk_count % 50 == 0 {
                tracing::debug!("upstream chunk #{chunk_count} ({} bytes, +{}ms)", chunk.len(), stream_start.elapsed().as_millis());
            }

            raw_buf.extend_from_slice(&chunk);

            // Decode as much valid UTF-8 as possible, leaving incomplete sequences
            let valid_up_to = match std::str::from_utf8(&raw_buf) {
                Ok(_) => raw_buf.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid_up_to == 0 {
                continue;
            }
            let text = std::str::from_utf8(&raw_buf[..valid_up_to]).unwrap();

            for ch in text.chars() {
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
                        tracing::debug!("upstream [DONE] after {chunk_count} chunks, +{}ms", stream_start.elapsed().as_millis());
                        let _ = tx.send(Bytes::from_static(b"data: [DONE]\n\n")).await;
                        continue;
                    }

                    let translated = translator(&data, &mut state);
                    for event in translated {
                        if tx.send(Bytes::from(event)).await.is_err() {
                            tracing::warn!("downstream closed after {chunk_count} chunks");
                            return;
                        }
                    }
                } else if let Some(data) = line.strip_prefix("data:") {
                    data_lines.push(data.trim_start().to_string());
                }
            }
            raw_buf.drain(..valid_up_to);
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
    let mut result = vec![];
    if let Some(value) = headers.get(reqwest::header::CONTENT_TYPE) {
        if let Ok(v) = HeaderValue::from_bytes(value.as_bytes()) {
            result.push((header::CONTENT_TYPE, v));
        }
    }
    result.extend(clone_ratelimit_headers(headers));
    result
}

fn clone_ratelimit_headers(headers: &reqwest::header::HeaderMap) -> Vec<(header::HeaderName, HeaderValue)> {
    // Forward rate-limit / retry headers so downstream clients can honor
    // server-provided backoff hints on 429 responses.
    let mut result = Vec::new();
    for (name, value) in headers.iter() {
        let n = name.as_str();
        let dominated = n == "retry-after"
            || n == "x-should-retry"
            || n.starts_with("x-ratelimit-")
            || n.starts_with("anthropic-ratelimit-");
        if dominated {
            if let Ok(v) = HeaderValue::from_bytes(value.as_bytes()) {
                result.push((header::HeaderName::from_bytes(name.as_ref()).unwrap(), v));
            }
        }
    }
    result
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

use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Mutex;

/// Pool of Anthropic providers with sticky failover.
///
/// Tracks which provider is currently active. On rate-limit (429) or server error (5xx),
/// switches to the provider with the earliest recovery time and stays there until it fails too.
pub struct AnthropicPool {
    providers: Vec<AnthropicProvider>,
    active: AtomicUsize,
    /// Per-provider: unix timestamp (seconds) when the provider becomes available again.
    /// 0 means available now.
    available_at: Mutex<Vec<u64>>,
}

impl AnthropicPool {
    pub fn new(providers: Vec<AnthropicProvider>) -> Self {
        let n = providers.len();
        Self {
            providers,
            active: AtomicUsize::new(0),
            available_at: Mutex::new(vec![0; n]),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Get the currently active provider.
    pub fn active_provider(&self) -> Option<&AnthropicProvider> {
        self.providers.get(self.active.load(Ordering::Relaxed))
    }

    /// Pick the best available provider: the active one if it's available,
    /// otherwise the one with the earliest available_at.
    pub async fn pick_provider(&self) -> Option<(usize, &AnthropicProvider)> {
        if self.providers.is_empty() {
            return None;
        }

        let now = now_secs();
        let available = self.available_at.lock().await;
        let active_idx = self.active.load(Ordering::Relaxed);

        // If the active provider is available, use it (sticky)
        if active_idx < available.len() && available[active_idx] <= now {
            return Some((active_idx, &self.providers[active_idx]));
        }

        // Find the provider with the earliest recovery time
        let best = available
            .iter()
            .enumerate()
            .min_by_key(|(_, t)| **t)
            .map(|(i, _)| i)?;

        drop(available);
        self.active.store(best, Ordering::Relaxed);
        info!(
            provider = %self.providers[best].name,
            "switched active Anthropic provider"
        );
        Some((best, &self.providers[best]))
    }

    /// Record a rate-limit / failure for a provider, parsed from response headers.
    pub async fn record_failure(&self, idx: usize, response: &reqwest::Response) {
        let mut available = self.available_at.lock().await;
        if idx >= available.len() {
            return;
        }

        let recovery_time = parse_recovery_time(response);
        available[idx] = recovery_time;

        let provider_name = &self.providers[idx].name;
        let secs_until = recovery_time.saturating_sub(now_secs());
        warn!(
            provider = %provider_name,
            status = %response.status().as_u16(),
            retry_in_secs = secs_until,
            "Anthropic provider rate-limited"
        );

        // If this was the active provider, switch to the best alternative
        let active_idx = self.active.load(Ordering::Relaxed);
        if active_idx == idx {
            let best = available
                .iter()
                .enumerate()
                .min_by_key(|(_, t)| **t)
                .map(|(i, _)| i)
                .unwrap_or(0);
            if best != idx {
                self.active.store(best, Ordering::Relaxed);
                let secs_until_best = available[best].saturating_sub(now_secs());
                info!(
                    from = %provider_name,
                    to = %self.providers[best].name,
                    available_in_secs = secs_until_best,
                    "failover: switched active Anthropic provider"
                );
            }
        }
    }

    /// Fetch models from the first available provider.
    pub async fn fetch_models(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        for provider in &self.providers {
            match provider.fetch_models().await {
                Ok(models) if !models.is_empty() => return Ok(models),
                Ok(_) => continue,
                Err(e) => {
                    warn!(provider = %provider.name, error = %e, "failed to fetch models");
                    continue;
                }
            }
        }
        Ok(vec![])
    }
}

/// Parse the recovery timestamp from Anthropic rate-limit response headers.
fn parse_recovery_time(response: &reqwest::Response) -> u64 {
    let headers = response.headers();
    let now = now_secs();

    // Try anthropic-ratelimit-unified-reset (unix timestamp)
    if let Some(reset) = headers
        .get("anthropic-ratelimit-unified-reset")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
    {
        return reset;
    }

    // Try retry-after (seconds from now)
    if let Some(retry_after) = headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
    {
        return now + retry_after;
    }

    // Default: 60 seconds cooldown
    now + 60
}

fn now_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
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

    fn from_moonshot(config: MoonshotProviderConfig) -> Self {
        let api_key = config.resolve_key();
        Self {
            name: config.name,
            api_base: config.api_base,
            api_key,
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

#[derive(Clone)]
pub struct OpenCodeGoProvider {
    name: String,
    openai_api_base: String,
    anthropic_api_base: String,
    api_key: Option<String>,
    client: reqwest::Client,
    /// Models that use Anthropic Messages API format (fetched from models.dev)
    anthropic_models: std::collections::HashSet<String>,
}

impl OpenCodeGoProvider {
    fn new(config: OpencodeGoProviderConfig) -> Self {
        Self {
            name: config.name,
            openai_api_base: config.openai_api_base,
            anthropic_api_base: config.anthropic_api_base,
            api_key: config.api_key,
            client: reqwest::Client::new(),
            anthropic_models: std::collections::HashSet::new(),
        }
    }

    /// Check whether a model uses the Anthropic Messages API format.
    pub fn uses_anthropic_format(&self, model: &str) -> bool {
        self.anthropic_models.contains(model)
    }

    async fn send_openai_json<T: Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> anyhow::Result<reqwest::Response> {
        let mut request = self
            .client
            .post(join_url(&self.openai_api_base, path))
            .json(body);

        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        request
            .send()
            .await
            .with_context(|| format!("opencode-go openai request to {} failed", self.name))
    }

    async fn send_anthropic_json<T: Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> anyhow::Result<reqwest::Response> {
        let mut request = self
            .client
            .post(join_url(&self.anthropic_api_base, path))
            .header("anthropic-version", "2023-06-01")
            .json(body);

        if let Some(api_key) = &self.api_key {
            request = request.header("x-api-key", api_key);
        }

        request
            .send()
            .await
            .with_context(|| format!("opencode-go anthropic request to {} failed", self.name))
    }

    /// Fetch available Go models from models.dev (same source opencode uses).
    /// Also populates the Anthropic-format model set from provider overrides.
    pub async fn fetch_models(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let resp = self
            .client
            .get("https://models.dev/api.json")
            .send()
            .await
            .context("failed to fetch models.dev")?;

        if !resp.status().is_success() {
            anyhow::bail!("models.dev returned {}", resp.status());
        }

        let body: serde_json::Value = resp.json().await.context("failed to decode models.dev")?;

        let provider = body
            .get("opencode-go")
            .context("opencode-go not found in models.dev")?;

        let models = provider
            .get("models")
            .and_then(|m| m.as_object())
            .map(|obj| {
                obj.values()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_str()?;
                        Some(json!({
                            "id": format!("opencode-go/{id}"),
                            "object": "model",
                            "owned_by": "opencode-go",
                        }))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(models)
    }

    /// Populate Anthropic-format model set from models.dev.
    /// Models with provider.npm == "@ai-sdk/anthropic" use the Anthropic
    /// Messages API; all others use OpenAI-compatible.
    pub async fn refresh_anthropic_models(&mut self) {
        if let Ok(resp) = self.client.get("https://models.dev/api.json").send().await {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                if let Some(models) = body
                    .get("opencode-go")
                    .and_then(|p| p.get("models"))
                    .and_then(|m| m.as_object())
                {
                    let mut set = std::collections::HashSet::new();
                    for (id, m) in models {
                        let is_anthropic = m
                            .get("provider")
                            .and_then(|p| p.get("npm"))
                            .and_then(|n| n.as_str())
                            == Some("@ai-sdk/anthropic");
                        if is_anthropic {
                            set.insert(id.clone());
                        }
                    }
                    self.anthropic_models = set;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── classify_failover_status tests ─────────────────────────────────────

    #[test]
    fn status_200_returns_immediately() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::OK),
            FailoverDecision::Return
        );
    }

    #[test]
    fn status_201_returns_immediately() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::CREATED),
            FailoverDecision::Return
        );
    }

    #[test]
    fn status_400_returns_immediately() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::BAD_REQUEST),
            FailoverDecision::Return
        );
    }

    #[test]
    fn status_401_returns_immediately() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::UNAUTHORIZED),
            FailoverDecision::Return
        );
    }

    #[test]
    fn status_403_returns_immediately() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::FORBIDDEN),
            FailoverDecision::Return
        );
    }

    #[test]
    fn status_404_advances() {
        // A target returning "model not found" should fall through to the next
        // target in the chain rather than failing the whole request.
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::NOT_FOUND),
            FailoverDecision::Advance
        );
    }

    #[test]
    fn status_402_advances() {
        // "Insufficient balance" on a provider should fail over to the next.
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::PAYMENT_REQUIRED),
            FailoverDecision::Advance
        );
    }

    #[test]
    fn status_408_advances() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::REQUEST_TIMEOUT),
            FailoverDecision::Advance
        );
    }

    #[test]
    fn status_422_returns_immediately() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::UNPROCESSABLE_ENTITY),
            FailoverDecision::Return
        );
    }

    #[test]
    fn status_429_advances() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::TOO_MANY_REQUESTS),
            FailoverDecision::Advance
        );
    }

    #[test]
    fn status_500_advances() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR),
            FailoverDecision::Advance
        );
    }

    #[test]
    fn status_502_advances() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::BAD_GATEWAY),
            FailoverDecision::Advance
        );
    }

    #[test]
    fn status_503_advances() {
        assert_eq!(
            classify_failover_status(reqwest::StatusCode::SERVICE_UNAVAILABLE),
            FailoverDecision::Advance
        );
    }

    // ── Target-iteration / failover logic unit tests ────────────────────────
    //
    // The virtual-model dispatch loop is async and tied to AppState (which requires
    // real provider instances). Rather than spinning up an HTTP mock server, we
    // test the pure classification helper exhaustively above, and verify the
    // iteration logic through the status classifier: the loop advances iff
    // classify_failover_status returns Advance, and returns iff it returns Return.
    //
    // This mirrors the actual loop in virtual_model_dispatch:
    //   FailoverDecision::Return  → return response immediately
    //   FailoverDecision::Advance → record and continue
    //
    // A full integration test would require a mock HTTP server (e.g. wiremock)
    // which is not a current dependency; the classifier tests above give us high
    // confidence in the decision boundary.

    #[test]
    fn failover_iteration_logic_return_on_success() {
        // Simulate: first target returns 200 → should stop
        let statuses = vec![200u16, 429, 500];
        let mut tried = Vec::new();
        let mut final_returned = None;

        for (i, &s) in statuses.iter().enumerate() {
            let code = reqwest::StatusCode::from_u16(s).unwrap();
            match classify_failover_status(code) {
                FailoverDecision::Return => {
                    final_returned = Some(i);
                    break;
                }
                FailoverDecision::Advance => {
                    tried.push(s);
                }
            }
        }

        assert_eq!(final_returned, Some(0), "should stop at first 200");
        assert!(tried.is_empty(), "nothing should have been advanced past");
    }

    #[test]
    fn failover_iteration_logic_advances_past_429_then_stops_on_200() {
        let statuses = vec![429u16, 503, 200];
        let mut tried = Vec::new();
        let mut final_returned = None;

        for (i, &s) in statuses.iter().enumerate() {
            let code = reqwest::StatusCode::from_u16(s).unwrap();
            match classify_failover_status(code) {
                FailoverDecision::Return => {
                    final_returned = Some(i);
                    break;
                }
                FailoverDecision::Advance => {
                    tried.push(s);
                }
            }
        }

        assert_eq!(final_returned, Some(2), "should stop at third target (200)");
        assert_eq!(tried, vec![429, 503], "first two should have been advanced past");
    }

    #[test]
    fn failover_iteration_logic_all_fail_exhausts_list() {
        let statuses = vec![429u16, 500, 502];
        let mut tried = Vec::new();
        let mut final_returned: Option<usize> = None;

        for (i, &s) in statuses.iter().enumerate() {
            let code = reqwest::StatusCode::from_u16(s).unwrap();
            match classify_failover_status(code) {
                FailoverDecision::Return => {
                    final_returned = Some(i);
                    break;
                }
                FailoverDecision::Advance => {
                    tried.push(s);
                }
            }
        }

        assert!(final_returned.is_none(), "no target should have returned");
        assert_eq!(tried.len(), 3, "all 3 targets should have been tried");
    }

    #[test]
    fn failover_stops_on_non_retryable_client_error() {
        // 401 Unauthorized: not worth retrying, return immediately
        let statuses = vec![429u16, 401, 200];
        let mut tried = Vec::new();
        let mut final_returned = None;

        for (i, &s) in statuses.iter().enumerate() {
            let code = reqwest::StatusCode::from_u16(s).unwrap();
            match classify_failover_status(code) {
                FailoverDecision::Return => {
                    final_returned = Some(i);
                    break;
                }
                FailoverDecision::Advance => {
                    tried.push(s);
                }
            }
        }

        assert_eq!(final_returned, Some(1), "should stop at 401, not continue to 200");
        assert_eq!(tried, vec![429u16]);
    }
}
