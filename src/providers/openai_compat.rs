use anyhow::Context;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use tracing::debug;

pub struct OpenAICompatProvider {
    pub name: String,
    pub api_base: String,
    pub api_key: Option<String>,
    pub auth_header: Option<String>,
    pub http_client: reqwest::Client,
}

impl OpenAICompatProvider {
    pub fn new(
        name: String,
        api_base: String,
        api_key: Option<String>,
        auth_header: Option<String>,
        http_client: reqwest::Client,
    ) -> Self {
        Self {
            name,
            api_base,
            api_key,
            auth_header,
            http_client,
        }
    }

    pub fn from_openai_config(
        config: &crate::config::OpenaiProviderConfig,
        http_client: reqwest::Client,
    ) -> Self {
        Self::new(
            config.name.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            None,
            http_client,
        )
    }

    pub fn from_mimo_config(
        config: &crate::config::XiaomiMimoProviderConfig,
        http_client: reqwest::Client,
    ) -> Self {
        Self::new(
            config.name.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            Some("api-key".to_string()),
            http_client,
        )
    }

    pub fn from_generic_config(
        config: &crate::config::OpenaiCompatibleProviderConfig,
        http_client: reqwest::Client,
    ) -> Self {
        Self::new(
            config.name.clone(),
            config.api_base.clone(),
            config.api_key.clone(),
            config.auth_header.clone(),
            http_client,
        )
    }

    pub async fn forward_request(&self, request: serde_json::Value) -> anyhow::Result<reqwest::Response> {
        let url = format!("{}/chat/completions", self.api_base.trim_end_matches('/'));
        let headers = self.build_headers()?;

        debug!(provider = %self.name, url = %url, "forwarding OpenAI-compatible request");

        self.http_client
            .post(url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .context("failed to send OpenAI-compatible request")
    }

    fn build_headers(&self) -> anyhow::Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(api_key) = &self.api_key {
            if let Some(auth_header) = &self.auth_header {
                headers.insert(
                    HeaderName::from_bytes(auth_header.as_bytes())
                        .context("invalid custom auth header name")?,
                    HeaderValue::from_str(api_key).context("invalid custom auth header value")?,
                );
            } else {
                headers.insert(
                    AUTHORIZATION,
                    HeaderValue::from_str(&format!("Bearer {api_key}"))
                        .context("invalid authorization header")?,
                );
            }
        }

        Ok(headers)
    }
}
