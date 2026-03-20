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
| Generic | API key + custom header | `openai_compatible` |

## Conventions

- Use `anyhow` for error handling
- Use `tracing` for logging
- Use `serde_json::Value` for passthrough fields
- Types use `#[serde(skip_serializing_if = "Option::is_none")]` for optional fields
