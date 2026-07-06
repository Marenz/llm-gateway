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

    /// Virtual models: a client-facing name -> ordered list of real model names tried in sequence.
    /// On a target-unusable failure (402/404/408/429/5xx/transport error) the next target is tried.
    #[serde(default)]
    pub virtual_models: HashMap<String, Vec<String>>,

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
    /// OpenCode Go subscription provider
    OpencodeGo(OpencodeGoProviderConfig),
    /// OpenCode Zen provider
    Zen(ZenProviderConfig),
    /// DeepSeek provider (OpenAI-compatible)
    DeepSeek(DeepSeekProviderConfig),
    /// DeepInfra provider (OpenAI-compatible)
    DeepInfra(DeepInfraProviderConfig),
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
pub struct OpencodeGoProviderConfig {
    #[serde(default = "default_opencode_go_name")]
    pub name: String,

    /// API key from opencode.ai. Supports "env:VAR_NAME" syntax.
    pub api_key: Option<String>,

    /// OpenAI-compatible base URL for most models
    #[serde(default = "default_opencode_go_openai_base")]
    pub openai_api_base: String,

    /// Anthropic-compatible base URL for MiniMax models
    #[serde(default = "default_opencode_go_anthropic_base")]
    pub anthropic_api_base: String,

    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZenProviderConfig {
    #[serde(default = "default_zen_name")]
    pub name: String,

    /// API key from opencode.ai/zen. Supports "env:VAR_NAME" syntax.
    pub api_key: Option<String>,

    /// OpenAI-compatible base URL (for /v1/chat/completions)
    #[serde(default = "default_zen_api_base")]
    pub api_base: String,

    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekProviderConfig {
    #[serde(default = "default_deepseek_name")]
    pub name: String,

    /// API key. Supports "env:VAR_NAME" syntax.
    pub api_key: Option<String>,

    /// Path to a file containing the API key (written by `llm-gateway login deepseek`).
    /// Used as a fallback when `api_key` is not set. Defaults to
    /// ~/.config/llm-gateway/deepseek-key.txt
    #[serde(default)]
    pub api_key_file: Option<PathBuf>,

    #[serde(default = "default_deepseek_api_base")]
    pub api_base: String,

    #[serde(default)]
    pub models: Vec<String>,
}

impl DeepSeekProviderConfig {
    /// Resolve the effective API key: prefer `api_key`, else read `api_key_file`
    /// (or the default key path).
    pub fn resolve_key(&self) -> Option<String> {
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                return Some(key.clone());
            }
        }
        let path = self
            .api_key_file
            .clone()
            .or_else(default_deepseek_key_path)?;
        let contents = std::fs::read_to_string(path).ok()?;
        let trimmed = contents.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }
}

/// Default path for the DeepSeek API key file.
pub fn default_deepseek_key_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("llm-gateway").join("deepseek-key.txt"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepInfraProviderConfig {
    #[serde(default = "default_deepinfra_name")]
    pub name: String,

    /// API key. Supports "env:VAR_NAME" syntax.
    pub api_key: Option<String>,

    /// Path to a file containing the API key (written by `llm-gateway login deepinfra`).
    /// Used as a fallback when `api_key` is not set. Defaults to
    /// ~/.config/llm-gateway/deepinfra-key.txt
    #[serde(default)]
    pub api_key_file: Option<PathBuf>,

    #[serde(default = "default_deepinfra_api_base")]
    pub api_base: String,

    #[serde(default)]
    pub models: Vec<String>,
}

impl DeepInfraProviderConfig {
    /// Resolve the effective API key: prefer `api_key`, else read `api_key_file`
    /// (or the default key path).
    pub fn resolve_key(&self) -> Option<String> {
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                return Some(key.clone());
            }
        }
        let path = self
            .api_key_file
            .clone()
            .or_else(default_deepinfra_key_path)?;
        let contents = std::fs::read_to_string(path).ok()?;
        let trimmed = contents.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }
}

/// Default path for the DeepInfra API key file.
pub fn default_deepinfra_key_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("llm-gateway").join("deepinfra-key.txt"))
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
fn default_opencode_go_name() -> String {
    "opencode-go".to_string()
}
fn default_opencode_go_openai_base() -> String {
    "https://opencode.ai/zen/go".to_string()
}
fn default_opencode_go_anthropic_base() -> String {
    "https://opencode.ai/zen/go".to_string()
}
fn default_zen_name() -> String {
    "zen".to_string()
}
fn default_zen_api_base() -> String {
    "https://opencode.ai/zen".to_string()
}
fn default_deepseek_name() -> String {
    "deepseek".to_string()
}
fn default_deepseek_api_base() -> String {
    "https://api.deepseek.com/v1".to_string()
}
fn default_deepinfra_name() -> String {
    "deepinfra".to_string()
}
fn default_deepinfra_api_base() -> String {
    "https://api.deepinfra.com/v1/openai".to_string()
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
                ProviderConfig::OpencodeGo(cfg) => {
                    cfg.api_key = cfg.api_key.as_ref().and_then(|k| resolve_env(k));
                }
                ProviderConfig::Zen(cfg) => {
                    cfg.api_key = cfg.api_key.as_ref().and_then(|k| resolve_env(k));
                }
                ProviderConfig::DeepSeek(cfg) => {
                    cfg.api_key = cfg.api_key.as_ref().and_then(|k| resolve_env(k));
                }
                ProviderConfig::DeepInfra(cfg) => {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config_json(extra: &str) -> String {
        if extra.is_empty() {
            r#"{ "providers": [] }"#.to_string()
        } else {
            format!(r#"{{ "providers": [], {} }}"#, extra)
        }
    }

    #[test]
    fn virtual_models_parses_from_json() {
        let json = minimal_config_json(
            r#"
            "virtual_models": {
                "family-bot": ["chatgpt/gpt-5.5", "anthropic/claude-sonnet", "deepseek-v4-pro"]
            }
            "#,
        );
        let config: GatewayConfig = serde_json::from_str(&json).expect("should parse");
        let targets = config.virtual_models.get("family-bot").expect("key missing");
        assert_eq!(targets.len(), 3);
        assert_eq!(targets[0], "chatgpt/gpt-5.5");
        assert_eq!(targets[1], "anthropic/claude-sonnet");
        assert_eq!(targets[2], "deepseek-v4-pro");
    }

    #[test]
    fn virtual_models_defaults_to_empty_when_absent() {
        let json = minimal_config_json("");
        let config: GatewayConfig = serde_json::from_str(&json).expect("should parse");
        assert!(
            config.virtual_models.is_empty(),
            "virtual_models should default to empty HashMap"
        );
    }

    #[test]
    fn virtual_models_empty_map_parses() {
        let json = minimal_config_json(r#""virtual_models": {}"#);
        let config: GatewayConfig = serde_json::from_str(&json).expect("should parse");
        assert!(config.virtual_models.is_empty());
    }
}
