use std::sync::Arc;

use anyhow::{anyhow, Context};
use reqwest::header::{
    AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue,
};
use tracing::{debug, info};

use crate::config::AnthropicProviderConfig;
use crate::oauth::anthropic as anthropic_oauth;
use crate::oauth::token_store::TokenStore;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

pub struct AnthropicProvider {
    pub config: AnthropicProviderConfig,
    pub http_client: reqwest::Client,
    pub token_store: Option<Arc<TokenStore>>,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicProviderConfig, http_client: reqwest::Client) -> Self {
        let token_store = config
            .oauth_token_file
            .clone()
            .map(TokenStore::new)
            .map(Arc::new);

        Self {
            config,
            http_client,
            token_store,
        }
    }

    pub async fn get_access_token(&self) -> anyhow::Result<String> {
        if let Some(token_store) = &self.token_store {
            let Some(tokens) = token_store.load().await.context("failed to load Anthropic tokens")? else {
                return self
                    .config
                    .api_key
                    .clone()
                    .ok_or_else(|| anyhow!("Anthropic provider has no OAuth tokens or API key configured"));
            };

            if !TokenStore::is_expired(&tokens) {
                return Ok(tokens.access_token);
            }

            let refresh_token = tokens
                .refresh_token
                .as_deref()
                .ok_or_else(|| anyhow!("Anthropic OAuth token is expired and no refresh token is available"))?;

            info!(provider = %self.config.name, "refreshing Anthropic OAuth token");
            let mut refreshed = anthropic_oauth::refresh_token(&self.http_client, refresh_token)
                .await
                .context("failed to refresh Anthropic access token")?;

            if refreshed.refresh_token.is_none() {
                refreshed.refresh_token = tokens.refresh_token;
            }

            token_store
                .save(&refreshed)
                .await
                .context("failed to save refreshed Anthropic tokens")?;

            return Ok(refreshed.access_token);
        }

        self.config
            .api_key
            .clone()
            .ok_or_else(|| anyhow!("Anthropic provider has no API key configured"))
    }

    pub async fn forward_request(
        &self,
        request: serde_json::Value,
        bearer_override: Option<&str>,
    ) -> anyhow::Result<reqwest::Response> {
        let access_token = if let Some(token) = bearer_override {
            token.to_owned()
        } else {
            self.get_access_token().await?
        };

        let request = self.apply_required_system_prompt(request)?;
        let url = format!("{}/v1/messages", self.config.api_base.trim_end_matches('/'));
        let headers = self.build_headers(&access_token)?;

        debug!(provider = %self.config.name, url = %url, "forwarding Anthropic request");

        self.http_client
            .post(url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .context("failed to send Anthropic request")
    }

    pub fn build_headers(&self, access_token: &str) -> anyhow::Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );

        let beta_header = if access_token.starts_with(anthropic_oauth::OAUTH_TOKEN_PREFIX) {
            format!(
                "{},{}",
                anthropic_oauth::OAUTH_BETA_HEADER,
                INTERLEAVED_THINKING_BETA
            )
        } else {
            INTERLEAVED_THINKING_BETA.to_string()
        };

        headers.insert(
            HeaderName::from_static("anthropic-beta"),
            HeaderValue::from_str(&beta_header).context("invalid Anthropic beta header")?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {access_token}"))
                .context("invalid Anthropic authorization header")?,
        );

        Ok(headers)
    }

    fn apply_required_system_prompt(
        &self,
        mut request: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        let Some(required_prompt) = &self.config.required_system_prompt else {
            return Ok(request);
        };

        let object = request
            .as_object_mut()
            .ok_or_else(|| anyhow!("Anthropic request payload must be a JSON object"))?;

        let required_entry = serde_json::json!({
            "type": "text",
            "text": required_prompt,
        });

        match object.get_mut("system") {
            None => {
                object.insert(
                    "system".to_string(),
                    serde_json::Value::Array(vec![required_entry]),
                );
            }
            Some(serde_json::Value::String(existing)) => {
                let current = std::mem::take(existing);
                object.insert(
                    "system".to_string(),
                    serde_json::Value::Array(vec![
                        required_entry,
                        serde_json::json!({ "type": "text", "text": current }),
                    ]),
                );
            }
            Some(serde_json::Value::Array(existing)) => {
                existing.insert(0, required_entry);
            }
            Some(_) => {
                return Err(anyhow!("Anthropic request field 'system' must be a string or array"));
            }
        }

        Ok(request)
    }
}
