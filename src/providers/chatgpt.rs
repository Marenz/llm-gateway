use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context};
use reqwest::header::{
    ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue,
};
use tracing::{debug, info};

use crate::config::ChatgptProviderConfig;
use crate::oauth::chatgpt as chatgpt_oauth;
use crate::oauth::token_store::TokenStore;

pub struct ChatgptProvider {
    pub config: ChatgptProviderConfig,
    pub http_client: reqwest::Client,
    pub token_store: Arc<TokenStore>,
}

impl ChatgptProvider {
    pub fn new(config: ChatgptProviderConfig, http_client: reqwest::Client) -> Self {
        let token_path = config
            .token_file
            .clone()
            .unwrap_or_else(default_token_path);

        Self {
            config,
            http_client,
            token_store: Arc::new(TokenStore::new(token_path)),
        }
    }

    pub async fn get_access_token(&self) -> anyhow::Result<String> {
        let tokens = self
            .token_store
            .load()
            .await
            .context("failed to load ChatGPT tokens")?
            .ok_or_else(|| anyhow!("ChatGPT is not logged in; run the login flow first"))?;

        if !TokenStore::is_expired(&tokens) {
            return Ok(tokens.access_token);
        }

        let refresh_token = tokens
            .refresh_token
            .as_deref()
            .ok_or_else(|| anyhow!("ChatGPT OAuth token is expired and no refresh token is available"))?;

        info!(provider = %self.config.name, "refreshing ChatGPT OAuth token");
        let mut refreshed = chatgpt_oauth::refresh_token(&self.http_client, refresh_token)
            .await
            .context("failed to refresh ChatGPT access token")?;

        if refreshed.refresh_token.is_none() {
            refreshed.refresh_token = tokens.refresh_token;
        }

        self.token_store
            .save(&refreshed)
            .await
            .context("failed to save refreshed ChatGPT tokens")?;

        Ok(refreshed.access_token)
    }

    pub async fn login(&self) -> anyhow::Result<()> {
        let device_code = chatgpt_oauth::request_device_code(&self.http_client)
            .await
            .context("failed to start ChatGPT device login")?;

        println!("Open this URL: {}", chatgpt_oauth::DEVICE_VERIFY_URL);
        println!("Enter this code: {}", device_code.user_code);

        let auth_code = chatgpt_oauth::poll_for_authorization(&self.http_client, &device_code)
            .await
            .context("failed while waiting for ChatGPT device authorization")?;
        let redirect_uri = format!("{}/deviceauth/callback", chatgpt_oauth::AUTH_BASE);
        let tokens = chatgpt_oauth::exchange_code(&self.http_client, &auth_code, &redirect_uri)
            .await
            .context("failed to exchange ChatGPT authorization code")?;

        self.token_store
            .save(&tokens)
            .await
            .context("failed to save ChatGPT tokens")?;

        info!(provider = %self.config.name, "ChatGPT login completed");
        Ok(())
    }

    pub async fn forward_request(&self, request: serde_json::Value) -> anyhow::Result<reqwest::Response> {
        let access_token = self.get_access_token().await?;
        let account_id = chatgpt_oauth::extract_account_id(&access_token);
        let headers = self.build_headers(&access_token, account_id.as_deref())?;
        let url = self.config.api_base.clone();

        debug!(provider = %self.config.name, url = %url, "forwarding ChatGPT request");

        self.http_client
            .post(url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .context("failed to send ChatGPT request")
    }

    pub fn build_headers(
        &self,
        access_token: &str,
        account_id: Option<&str>,
    ) -> anyhow::Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {access_token}"))
                .context("invalid ChatGPT authorization header")?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        headers.insert(
            HeaderName::from_static("originator"),
            HeaderValue::from_static("llm-gateway"),
        );
        headers.insert(
            HeaderName::from_static("user-agent"),
            HeaderValue::from_static("llm-gateway/0.1.0"),
        );

        if let Some(account_id) = account_id {
            headers.insert(
                HeaderName::from_static("chatgpt-account-id"),
                HeaderValue::from_str(account_id).context("invalid ChatGPT account id header")?,
            );
        }

        Ok(headers)
    }
}

fn default_token_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llm-gateway")
        .join("chatgpt")
        .join("auth.json")
}
