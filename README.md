# llm-gateway

A generic LLM proxy/router written in Rust. Exposes a unified OpenAI-compatible API
(`/v1/chat/completions`) and routes to multiple backend providers.

## Providers

| Provider | Auth | Model prefix |
|----------|------|-------------|
| **Anthropic** | OAuth token (auto-refresh) or API key | `anthropic/` |
| **ChatGPT** (subscription) | OAuth device/browser flow | `chatgpt/` |
| **XiaoMiMo** | API key | `mimo/` |
| **OpenAI** | API key | `openai/` |
| **Generic** | API key + custom header | configured name |

Models are discovered live from each provider's API at startup — no hardcoded lists needed.

## Installation

```bash
cargo build --release
cp gateway.example.json ~/.config/llm-gateway/config.json
# edit config.json with your keys
```

## Authentication

### Anthropic (OAuth)

```bash
llm-gateway login anthropic
# opens browser → paste code#state back → tokens saved to
# ~/.config/llm-gateway/anthropic-oauth.json
# tokens auto-refresh on expiry
```

Or set `ANTHROPIC_API_KEY` for standard API key auth.

### ChatGPT (subscription)

```bash
llm-gateway login chatgpt
# prints URL + code → visit URL → enter code → tokens saved to
# ~/.config/llm-gateway/chatgpt-auth.json
# tokens auto-refresh on expiry
# requires ChatGPT Plus/Pro/Max subscription
```

### XiaoMiMo / OpenAI

Set `XIAOMI_MIMO_API_KEY` / `OPENAI_API_KEY`, or configure in `config.json`.

## Usage

```bash
# Start the server (default: 127.0.0.1:4000)
llm-gateway

# Check auth status
llm-gateway status

# List available models
curl http://127.0.0.1:4000/v1/models

# Chat completions (standard OpenAI format)
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-sonnet-4-6", "messages": [{"role": "user", "content": "hello"}]}'

# Streaming
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "chatgpt/gpt-5.4", "messages": [{"role": "user", "content": "hello"}], "stream": true}'
```

## Configuration

Default config path: `~/.config/llm-gateway/config.json`

```json
{
  "host": "127.0.0.1",
  "port": 4000,
  "providers": [
    {
      "type": "anthropic",
      "name": "anthropic",
      "api_key": "env:ANTHROPIC_API_KEY",
      "oauth_token_file": null,
      "api_base": "https://api.anthropic.com"
    },
    {
      "type": "chatgpt",
      "name": "chatgpt",
      "token_file": null,
      "api_base": "https://chatgpt.com/backend-api/codex"
    },
    {
      "type": "xiaomi_mimo",
      "name": "xiaomi-mimo",
      "api_key": "env:XIAOMI_MIMO_API_KEY",
      "api_base": "https://api.xiaomimimo.com/v1"
    },
    {
      "type": "openai_compatible",
      "name": "my-provider",
      "api_key": "env:MY_API_KEY",
      "api_base": "https://api.example.com/v1"
    }
  ],
  "model_aliases": {
    "claude": "anthropic/claude-sonnet-4-6",
    "gpt": "chatgpt/gpt-5.4-mini"
  }
}
```

`api_key` values support `"env:VAR_NAME"` to read from environment variables.

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models (live discovery) |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/messages` | Anthropic Messages API passthrough |

## Attribution

This project is a Rust port and generalization of
[anthropic-max-router](https://github.com/Nayjest/anthropic-max-router) by
[@Nayjest](https://github.com/Nayjest), which provided the original Anthropic OAuth
token refresh logic and proxy concept.

The ChatGPT OAuth flow and Responses API integration is based on the implementation
in [opencode](https://github.com/sst/opencode) by SST.
