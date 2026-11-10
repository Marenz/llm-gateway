# llm-gateway

## Overview

Generic LLM proxy/router in Rust. Exposes a unified OpenAI-compatible API (`/v1/chat/completions`) that routes to multiple backend providers: Anthropic, OpenAI/ChatGPT (OAuth), XiaoMiMo, Moonshot AI, and any OpenAI-compatible endpoint.

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
    chatgpt.rs             — Responses request + shared stream/collection state machine
    sse.rs                 — Incremental byte-safe SSE decoder and buffered completion re-emission
    transport.rs           — HTTP body adapters for translated streams and ChatGPT collection
    compat_tests.rs        — Offline raw fixtures + loopback HTTP compatibility regressions
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

## Login / Auth CLI

- `login anthropic` and `login chatgpt` both auto-open the browser via `xdg-open`
  by default. Pass `--show-url-only` to skip the browser and just print the URL
  (headless/remote use). The shared `open_browser(url, show_url_only)` helper in
  `main.rs` handles this for Anthropic; `chatgpt::login_browser` takes a
  `show_url_only: bool`.
  - **Default ChatGPT mode** runs a local callback server on port 1455 and
    `wait_for_callback` races it against `tokio::signal::ctrl_c()` + a 5-min
    timeout, so Ctrl+C cancels cleanly (the old code blocked on the channel and
    ignored SIGINT).
  - **`--show-url-only` ChatGPT mode** does NOT bind a callback server (the
    browser is on another machine, so `localhost:1455` can't reach this host).
    Instead `prompt_for_pasted_callback` asks the user to paste the redirect URL
    (or `code#state`, or a bare `code`); `parse_pasted_callback` extracts
    `(code, state)`. A bare code skips the CSRF/state check with a warning.
  - Anthropic always requires a manual `code#state` paste (hosted redirect at
    `console.anthropic.com`).
- `login deepseek` stores a pasted API key to `~/.config/llm-gateway/deepseek-key.txt`
  (chmod 0600). `DeepSeekProviderConfig::resolve_key()` prefers `api_key`, then
  falls back to `api_key_file` (or the default key path). DeepSeek is now always
  registered in `default_config()` so the key file is picked up even without
  `DEEPSEEK_API_KEY`.
- **No-effect warnings:** `handle_login` loads the effective config and warns (to
  stderr) when a just-completed login won't be used by the gateway:
  - DeepSeek: a configured `api_key` (or `DEEPSEEK_API_KEY`) shadows the key file.
  - Anthropic/ChatGPT: the provider isn't present in the config at all, or the
    `--token-file` save path differs from the provider's configured
    `oauth_token_file`/`token_file` (compared via `same_path`, which canonicalizes).
  - With no config file present, `default_config()` registers all providers at
    default paths, so default-path saves never warn.
- `login moonshot` stores a pasted API key at
  `~/.config/llm-gateway/moonshot-key.txt` (chmod 0600). The dedicated
  `moonshot` provider defaults to `https://api.moonshot.ai/v1`, is registered
  by `default_config()`, and routes `moonshot/*` through the shared
  OpenAI-compatible proxy.

## Local deployment

Installed binary lives at `~/.cargo/bin/llm-gateway`, run as the
`llm-gateway.service` **systemd --user** unit (config at
`~/.config/llm-gateway/config.json`). To deploy changes:

```bash
cargo install --path . --force
systemctl --user restart llm-gateway.service
curl -s http://127.0.0.1:4000/health   # {"status":"ok"}
```

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
| DeepInfra | API key | `deep_infra` |
| Moonshot AI | API key | `moonshot` |
| Generic | API key + custom header | `openai_compatible` |

### Request body limit

- `max_request_bytes` (config, default 64 MiB = `DEFAULT_MAX_REQUEST_BYTES`)
  replaces axum's implicit 2 MB `DefaultBodyLimit` on all routes via
  `build_router`. Before this, bodies >2 MB (≈450k tokens) failed with
  "Failed to buffer the request body: length limit exceeded".
- Handlers take `Result<Json<T>, JsonRejection>`; `body_rejection` turns an
  oversized body into a JSON 413 (`type: request_too_large`,
  `max_request_bytes: <n>`). Other rejections keep their status, JSON-wrapped.
- No other size caps exist on the path: upstream reqwest clients, response
  buffering (`bytes()`, `to_bytes(usize::MAX)`) and the SSE decoder are
  unbounded. reqwest clients have no total timeout either.

### Image inputs

- `/v1/chat/completions` accepts OpenAI multipart user content with `image_url`
  parts. ChatGPT and OpenAI-compatible providers receive those parts directly.
- Anthropic translation preserves user images: `data:<media-type>;base64,...`
  becomes an Anthropic `base64` image source, while other URLs become `url`
  image sources. Images are not valid in Anthropic system or assistant content.
- Native `/v1/messages` requests also deserialize Anthropic `image` blocks with
  `base64` or `url` sources.
- Provider transport support does not imply every configured model has vision;
  test the specific model when adding it to a route or virtual-model chain.

### Prompt cache passthrough

- OpenAI-shaped `message.cache_control` is an explicit Anthropic cache boundary.
  Explicit boundaries replace the legacy last-three-message heuristic.
- A boundary stays on the last content block of its SOURCE message. Adjacent
  same-role messages (tool results, user messages) still merge into one
  Anthropic message, but the marker is placed right after its own message is
  merged, so later messages are appended behind it and stay outside the cached
  prefix. One merged message can carry several block-level markers.
  - Two consecutive unmarked assistant texts are still joined into one string;
    a marked one becomes its own text block.
  - A marked message with no content of its own (user `content: null`, empty
    assistant) shares the block merged before it: same marker, one breakpoint;
    a different `ttl` on that block is a 400. With nothing before it, it is a
    400 as before.
- Validated before any upstream call (400, and no virtual-model failover): at
  most four breakpoints, counted per block; a `ttl: "1h"` boundary must not come
  after a 5-minute one in prompt order (system text, then messages). Anthropic
  rejects that order itself (`validate_ttl_order`, confirmed live 2026-11-10).
- `src/translate/fixtures/anthropic_request_pins.json` pins the exact upstream
  bytes for the shapes Adapsis sends (boundaries on system text and assistant
  messages), for a boundary on the last message of a merged run, and for the
  heuristic. `expected` was recorded at c4f0d36; change it only on purpose.
- Empty text and thinking blocks are not cacheable; text, image, tool-use, and
  tool-result blocks are cacheable.
- Cache-validation 400s are client errors and must stop virtual-model failover;
  ordinary upstream provider errors retain the normal failover policy.
- Anthropic `input_tokens` excludes cache-read and cache-creation tokens, so they
  are separate additive input counters. Keep OpenAI-shaped `total_tokens` equal
  to `prompt_tokens + completion_tokens`; clients account for Anthropic cache
  fields separately. ChatGPT `input_tokens_details.cached_tokens` is already
  included in `input_tokens`; expose it without adding it to prompt or total.

### ChatGPT prompt cache key

- The ChatGPT backend only reuses a cached prefix when requests carry a stable
  identity. Measured 2026-11-10 on `gpt-6-astra`, follow-up requests 12 s
  apart: neither field 0/4 hits, `prompt_cache_key` only 1/8, `session-id`
  header only 6/8, both 7/8. The gateway sends both with the same value, as the
  Codex CLI does (`codex-rs/core/src/client.rs`,
  `codex-api/src/requests/headers.rs`).
- `/v1/chat/completions` accepts an optional `prompt_cache_key`: a string of
  1–64 printable ASCII characters without surrounding whitespace
  (`OpenAIRequest::client_prompt_cache_key`, `MAX_PROMPT_CACHE_KEY_LEN`). The
  field is raw JSON in the request type so a wrong type is a gateway 400, not a
  serde 422. It is checked once in `chat_completions`, before any route.
- chatgpt route: `chatgpt::apply_prompt_cache_key` uses the client key, or
  derives a UUID-shaped key from SHA-256 over the upstream model, the
  `instructions` and the first `input` item. A conversation keeps its key as it
  grows; a different model, system text or first message gives another key. The
  key is a routing hint and never changes `instructions` or `input`.
- openai route: a client key is forwarded; none is invented. All other routes
  drop the field in `openai_passthrough_request`, as before.
- Hits are probabilistic, not guaranteed. Measured 2026-11-10 with the derived
  key on 1.6k-token prefixes: 18 of 29 second requests hit after 12 s, and 15
  of 26 single later reads hit (60 s 1/5, 4 min 1/2, 7 min 2/2, 20 min 4/7,
  35 min 4/5, 65 min 3/5). There is no downward trend up to 65 minutes, so no
  expiry was observed; the upper bound and whether a read extends the lifetime
  are not measured. Batches of the same test varied from 1/3 to 9/9 at 12 s
  with no clear relation to request rate.
- Usage: upstream reports `input_tokens_details.cached_tokens` (a multiple of
  128, inside `input_tokens`) and `cache_write_tokens`, which was 0 on every
  request on this route. The gateway exposes only `cache_read_input_tokens`.
  `RUST_LOG=llm_gateway::translate::chatgpt=debug` logs the raw details.
- Conversations with the same model, system text and first message share a
  derived key. Clients that start many conversations identically should send
  their own per-conversation `prompt_cache_key`.

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
- **Failover trigger** (`classify_failover_status`): every non-`2xx` response
  advances to the next target, including authentication and malformed-request
  errors. Each new request starts from the first target, so recovered providers
  and refreshed credentials are retried automatically.
- All targets exhausted → returns the last failure response (502 listing tried
  targets + statuses).
- **Streaming:** virtual-model attempts are forced to `stream:false` so the HTTP
  status is visible before committing to a body. For `stream:true` clients the
  buffered completion is re-emitted as SSE, preserving tool calls (with indices),
  content, usage and finish reason. Missing terminal reasons yield 502 instead
  of an invented `stop`. Non-virtual requests stream normally.
- Shared dispatch: both the virtual loop and the normal path call
  `dispatch_resolved` — the per-provider match lives in exactly one place.

## Native tool-call compatibility

- ChatGPT Responses `item.id` identifies argument events; `call_id` is the public
  OpenAI tool-call id used by subsequent `role:tool` messages. Never key argument
  assembly by `call_id` or fall back to index zero on a missing item.
- ChatGPT buffered and streaming paths share one state machine. Only upstream
  `response.completed` / `response.incomplete` establishes terminal evidence.
  Incomplete responses never report successful `tool_calls`: token limits map
  to `length`, filtering to `content_filter`, other causes to `incomplete`.
  Failed/error events and EOF without terminal evidence produce an error (502
  when buffered; an SSE error payload without a success terminal when streamed).
- Anthropic stream tool headers are emitted at block start; argument fragments
  keep their original content-block indices, which can be sparse. Block stop
  emits the initial input for tools without argument deltas (including `{}`).
  A terminal chunk is deferred until `message_stop`, with a stop reason required.
- Anthropic `max_tokens` / `model_context_window_exceeded` map to `length`,
  `refusal` to `content_filter`, `pause_turn` stays `pause_turn`; unrecognized
  reasons become `unknown:<upstream-reason>`. Missing reasons are errors. Clients
  must treat only `stop` and `tool_calls` as successful terminal reasons and
  require a complete stream before executing fully assembled calls.
- This slice supports ordinary function schemas, `tool_choice:auto`, assistant
  `content:null` with tool calls and text tool results correlated by id. Signed
  Anthropic thinking replay is not represented; tool-result images are flattened
  away by existing text extraction. `required` tool choice combined with thinking
  can be rejected upstream. These cases are **not** certified by these fixes.
- `src/translate/fixtures/*.sse` contains synthetic public-shape payloads only.
  Compatibility tests use in-memory framing or loopback HTTP and no providers.
  Offline source tests do not certify the deployed binary or live providers;
  live certification requires separately approved probes.

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
