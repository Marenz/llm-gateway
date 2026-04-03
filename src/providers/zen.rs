use anyhow::Context;
use reqwest::Client;
use serde::Serialize;

use crate::config::ZenProviderConfig;

#[derive(Clone)]
pub struct ZenProvider {
    pub name: String,
    pub api_base: String,
    pub api_key: Option<String>,
    pub client: Client,
}

impl ZenProvider {
    pub fn new(config: ZenProviderConfig) -> Self {
        Self {
            name: config.name,
            api_base: config.api_base,
            api_key: config.api_key,
            client: Client::new(),
        }
    }

    pub async fn send_openai_json<T: Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> anyhow::Result<reqwest::Response> {
        let mut request = self
            .client
            .post(join_url(&self.api_base, path))
            .json(body);

        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        request
            .send()
            .await
            .with_context(|| format!("zen request to {} failed", self.name))
    }

    pub async fn send_anthropic_json<T: Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> anyhow::Result<reqwest::Response> {
        let mut request = self
            .client
            .post(join_url(&self.api_base, path))
            .header("anthropic-version", "2023-06-01")
            .json(body);

        if let Some(api_key) = &self.api_key {
            request = request.header("x-api-key", api_key);
        }

        request
            .send()
            .await
            .with_context(|| format!("zen anthropic request to {} failed", self.name))
    }

    pub async fn fetch_models(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut request = self.client.get(join_url(&self.api_base, "/v1/models"));

        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let resp = request
            .send()
            .await
            .with_context(|| format!("zen GET /models failed for {}", self.name))?;

        if !resp.status().is_success() {
            anyhow::bail!("{} /models returned {}", self.name, resp.status());
        }

        let body: serde_json::Value = resp
            .json()
            .await
            .with_context(|| format!("failed to decode {} /models response", self.name))?;

        let models = body
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_str()?;
                        Some(serde_json::json!({
                            "id": format!("zen/{id}"),
                            "object": "model",
                            "owned_by": m.get("owned_by").cloned().unwrap_or(serde_json::json!("zen")),
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
