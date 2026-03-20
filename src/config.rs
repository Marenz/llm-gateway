use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Top-level gateway configuration, loaded from a TOML/JSON file or built programmatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Listen address (default: 127.0.0.1)
    #[serde(default = "default_host")]
    pub host: String,

    /// Listen port (default: 4000)
    #[serde(default = "default_port")]
    pub port: u16,

    /// Master API key for authenticating proxy requests (optional)
    #[serde(default)]
    pub master_key: Option<String>,

    /// Configured providers
    pub providers: Vec<ProviderConfig>,

    /// Model name aliases: maps incoming model names to provider-specific model names.
    /// e.g. "gpt-4" -> "chatgpt/gpt-5.4"
    #[serde(default)]
    pub model_aliases: HashMap<String, String>,

    /// Log level: quiet, minimal, normal, verbose
    #[serde(default = "default_log_level")]
    pub log_level: LogLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Quiet,
    Minimal,
    Normal,
    Verbose,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderConfig {
    Anthropic(AnthropicProviderConfig),
    Chatgpt(ChatgptProviderConfig),
    Openai(OpenaiProviderConfig),
    XiaomiMimo(XiaomiMimoProviderConfig),
    /// Generic OpenAI-compatible provider
    OpenaiCompatible(OpenaiCompatibleProviderConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicProviderConfig {
    /// Display name for this provider instance
    #[serde(default = "default_anthropic_name")]
    pub name: String,

    /// API key or OAuth token (sk-ant-oat-* for OAuth, sk-ant-api* for regular)
    /// Supports "env:VAR_NAME" syntax.
    #[serde(default)]
    pub api_key: Option<String>,

    /// Path to .oauth-tokens.json for auto-refresh
    #[serde(default)]
    pub oauth_token_file: Option<PathBuf>,

    /// API base URL
    #[serde(default = "default_anthropic_api_base")]
    pub api_base: String,

    /// Whether to allow bearer token passthrough from client requests
    #[serde(default)]
    pub allow_bearer_passthrough: bool,

    /// Required system prompt to prepend (for MAX plan enforcement)
    #[serde(default)]
    pub required_system_prompt: Option<String>,

    /// Models to expose through this provider
    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatgptProviderConfig {
    #[serde(default = "default_chatgpt_name")]
    pub name: String,

    /// Path to auth.json token file
    #[serde(default)]
    pub token_file: Option<PathBuf>,

    /// API base URL
    #[serde(default = "default_chatgpt_api_base")]
    pub api_base: String,

    /// Models to expose
    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenaiProviderConfig {
    #[serde(default = "default_openai_name")]
    pub name: String,

    /// API key. Supports "env:VAR_NAME" syntax.
    pub api_key: Option<String>,

    /// Organization ID
    #[serde(default)]
    pub organization: Option<String>,

    #[serde(default = "default_openai_api_base")]
    pub api_base: String,

    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiaomiMimoProviderConfig {
    #[serde(default = "default_mimo_name")]
    pub name: String,

    /// API key. Supports "env:VAR_NAME" syntax.
    pub api_key: Option<String>,

    #[serde(default = "default_mimo_api_base")]
    pub api_base: String,

    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenaiCompatibleProviderConfig {
    pub name: String,

    /// API key. Supports "env:VAR_NAME" syntax.
    #[serde(default)]
    pub api_key: Option<String>,

    /// Base URL for the OpenAI-compatible API
    pub api_base: String,

    /// Custom auth header name (default: "Authorization" with Bearer prefix)
    #[serde(default)]
    pub auth_header: Option<String>,

    #[serde(default)]
    pub models: Vec<String>,
}

// Defaults
fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    4000
}
fn default_log_level() -> LogLevel {
    LogLevel::Normal
}
fn default_anthropic_name() -> String {
    "anthropic".to_string()
}
fn default_anthropic_api_base() -> String {
    "https://api.anthropic.com".to_string()
}
fn default_chatgpt_name() -> String {
    "chatgpt".to_string()
}
fn default_chatgpt_api_base() -> String {
    "https://chatgpt.com/backend-api/codex".to_string()
}
fn default_openai_name() -> String {
    "openai".to_string()
}
fn default_openai_api_base() -> String {
    "https://api.openai.com/v1".to_string()
}
fn default_mimo_name() -> String {
    "xiaomi-mimo".to_string()
}
fn default_mimo_api_base() -> String {
    "https://api.xiaomimimo.com/v1".to_string()
}

impl GatewayConfig {
    /// Load config from a JSON file path.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Resolve "env:VAR_NAME" references in API keys.
    pub fn resolve_env_vars(&mut self) {
        for provider in &mut self.providers {
            match provider {
                ProviderConfig::Anthropic(cfg) => {
                    cfg.api_key = cfg.api_key.as_ref().and_then(|k| resolve_env(k));
                }
                ProviderConfig::Openai(cfg) => {
                    cfg.api_key = cfg.api_key.as_ref().and_then(|k| resolve_env(k));
                    cfg.organization = cfg.organization.as_ref().and_then(|k| resolve_env(k));
                }
                ProviderConfig::XiaomiMimo(cfg) => {
                    cfg.api_key = cfg.api_key.as_ref().and_then(|k| resolve_env(k));
                }
                ProviderConfig::OpenaiCompatible(cfg) => {
                    cfg.api_key = cfg.api_key.as_ref().and_then(|k| resolve_env(k));
                }
                ProviderConfig::Chatgpt(_) => {
                    // ChatGPT uses OAuth, no API key to resolve
                }
            }
        }
        if let Some(ref key) = self.master_key {
            self.master_key = resolve_env(key);
        }
    }
}

/// Resolve a value that may be "env:VAR_NAME" to the environment variable value.
fn resolve_env(value: &str) -> Option<String> {
    if let Some(var_name) = value.strip_prefix("env:") {
        std::env::var(var_name).ok()
    } else {
        Some(value.to_string())
    }
}
