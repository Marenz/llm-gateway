use anyhow::Context;
use reqwest::Client;
use serde::Serialize;

use crate::config::DeepInfraProviderConfig;

#[derive(Clone)]
pub struct DeepInfraProvider {
    pub name: String,
    pub api_base: String,
    pub api_key: Option<String>,
    pub client: Client,
}

impl DeepInfraProvider {
    pub fn new(config: DeepInfraProviderConfig) -> Self {
        let api_key = config.resolve_key();
        Self {
            name: config.name,
            api_base: config.api_base,
            api_key,
            client: Client::new(),
        }
    }

    pub async fn send_json<T: Serialize>(
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
            .with_context(|| format!("deepinfra request to {} failed", self.name))
    }

    pub async fn fetch_models(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut request = self.client.get(join_url(&self.api_base, "/models"));

        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let resp = request
            .send()
            .await
            .with_context(|| format!("deepinfra GET /models failed for {}", self.name))?;

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
                            "id": format!("deepinfra/{id}"),
                            "object": "model",
                            "owned_by": m.get("owned_by").cloned().unwrap_or(serde_json::json!("deepinfra")),
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