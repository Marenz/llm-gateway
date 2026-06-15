# llm-gateway

## Overview

Generic LLM proxy/router in Rust. Exposes a unified OpenAI-compatible API (`/v1/chat/completions`) that routes to multiple backend providers: Anthropic, OpenAI/ChatGPT (OAuth), XiaoMiMo, and any OpenAI-compatible endpoint.

Ported from [anthropic-max-router](https://github.com/Nayjest/lm-proxy) (TypeScript) with a generic multi-provider architecture.

## Build & Run

```bash
cargo build --release
./target/release/llm-gateway --config gateway.json
# Or with defaults from env vars:
ANTHROPIC_API_KEY=sk-ant-... ./target/release/llm-gateway
```

## Test

```bash
cargo test
cargo check
```

## Project Structure

```
src/
  main.rs            — CLI parsing, config loading, entry point
  config.rs          — GatewayConfig, ProviderConfig types, env var resolution
  types.rs           — OpenAI + Anthropic wire format types (serde)
  oauth/
    mod.rs
    token_store.rs   — Generic OAuth token persistence with file locking
    anthropic.rs     — Anthropic PKCE OAuth flow + token refresh
    chatgpt.rs       — ChatGPT device code OAuth flow + token refresh
  providers/
    mod.rs
    anthropic.rs     — Full Anthropic provider with OAuth token store (not used by server currently)
    chatgpt.rs       — ChatGPT provider with device code login
    openai_compat.rs — Generic OpenAI-compatible provider
  router/
    mod.rs
    server.rs        — Axum HTTP server, all endpoints, inline provider structs
    model_resolver.rs — Model name -> provider routing
  translate/
    mod.rs
    openai_to_anthropic.rs — OpenAI request -> Anthropic request
    anthropic_to_openai.rs — Anthropic response -> OpenAI response
    stream.rs              — Anthropic SSE -> OpenAI SSE stream translation
```

## Key Design Decisions

- **server.rs has inline provider structs** — The `AnthropicProvider`, `ChatgptProvider`, `OpenAICompatProvider` in `router/server.rs` are the ones actually used. The `providers/` module has richer standalone implementations that could replace them later.
- **Token refresh is lazy** — Tokens are refreshed on-demand when a request comes in and the token is expired. No background refresh.
- **Model routing** — Models can be routed by exact name, prefix (e.g. `anthropic/claude-*`), or alias.
- **Config** — JSON config file with `env:VAR_NAME` syntax for secrets. Falls back to env vars when no config file exists.

## Endpoints

- `GET /health` — Health check
- `POST /v1/chat/completions` — OpenAI-compatible chat completions (routes to any provider)
- `POST /v1/messages` — Anthropic Messages API passthrough
- `GET /v1/models` — List available models

## Providers

| Provider | Auth | Config type |
|----------|------|-------------|
| Anthropic | API key or OAuth (sk-ant-oat-*) with auto-refresh | `anthropic` |
| ChatGPT | OAuth device code flow | `chatgpt` |
| OpenAI | API key | `openai` |
| XiaoMiMo | API key | `xiaomi_mimo` |
| OpenCode Go | API key (subscription) | `opencode_go` |
| OpenCode Zen | API key (subscription) | `zen` |
| DeepSeek | API key | `deep_seek` |
| Generic | API key + custom header | `openai_compatible` |

## Virtual Models (failover chains)

A **virtual model** maps a single client-facing model name to an ordered list of
real model names. When a client requests a virtual model, the gateway tries each
target in order, advancing to the next on a retryable failure.

```json
"virtual_models": {
  "family-bot": ["chatgpt/gpt-5.5", "anthropic/claude-sonnet-4-5-20250929", "deepseek-v4-pro"]
}
```

- Lookup happens in `chat_completions` BEFORE normal model resolution; a name in
  `virtual_models` shadows any same-named real route.
- Each target is resolved through `ModelResolver::resolve` (so targets may be
  prefixed names or aliases). Unresolvable targets are skipped.
- **Failover trigger** (`classify_failover_status`): `429` and `5xx` (and
  transport/gateway errors surfacing as 502) → advance; `2xx` and non-retryable
  client errors (`400/401/403/404/422`) → return immediately (auth/bad-request
  won't be fixed by retrying elsewhere).
- All targets exhausted → returns the last failure response (502 listing tried
  targets + statuses).
- **Streaming:** virtual-model attempts are forced to `stream:false` so the HTTP
  status is visible before committing to a body; the client receives a normal
  JSON completion. Non-virtual requests stream normally.
- Shared dispatch: both the virtual loop and the normal path call
  `dispatch_resolved` — the per-provider match lives in exactly one place.

## DeepSeek Models

The DeepSeek provider supports all models via the OpenAI-compatible API at `api.deepseek.com/v1`. Passthrough proxies handle both standard and thinking output transparently.

| Model | Upstream ID | Notes |
|-------|-------------|-------|
| DeepSeek V4 Flash | `deepseek-v4-flash` | Fast chat model. DeepSeek-chat compatibility alias maps here. |
| DeepSeek V4 Pro | `deepseek-v4-pro` | Thinking/reasoning model. Use `thinking: {"type": "enabled"}` + `reasoning_effort`. |

Backwards-compatible aliases: `deepseek-chat` -> `deepseek-v4-flash`, `deepseek-reasoner` -> `deepseek-v4-flash`, `deepseek-thinking` -> `deepseek-v4-pro`. All deprecate 2026/07/24.

## Conventions

- Use `anyhow` for error handling
- Use `tracing` for logging
- Use `serde_json::Value` for passthrough fields
- Types use `#[serde(skip_serializing_if = "Option::is_none")]` for optional fields
