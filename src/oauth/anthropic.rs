use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::RngCore;
use reqwest::Url;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::token_store::OAuthTokens;

pub const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
pub const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
pub const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
pub const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
pub const SCOPE: &str = "org:create_api_key user:profile user:inference";
pub const OAUTH_TOKEN_PREFIX: &str = "sk-ant-oat";
pub const OAUTH_BETA_HEADER: &str = "oauth-2025-04-20";

pub fn generate_pkce() -> (String, String) {
    let mut verifier_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut verifier_bytes);

    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));

    (verifier, challenge)
}

pub fn generate_state() -> String {
    let mut state = [0u8; 32];
    rand::rng().fill_bytes(&mut state);
    URL_SAFE_NO_PAD.encode(state)
}

pub fn get_authorization_url(code_challenge: &str, state: &str) -> String {
    let mut url = Url::parse(AUTHORIZE_URL).expect("invalid authorize URL");
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPE)
        .append_pair("state", state)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256");
    url.to_string()
}

pub async fn exchange_code(
    client: &reqwest::Client,
    code: &str,
    code_verifier: &str,
    state: &str,
) -> anyhow::Result<OAuthTokens> {
    let response = client
        .post(TOKEN_URL)
        .header("anthropic-beta", OAUTH_BETA_HEADER)
        .json(&json!({
            "grant_type": "authorization_code",
            "code": code,
            "state": state,
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": code_verifier,
        }))
        .send()
        .await
        .context("failed to exchange Anthropic authorization code")?
        .error_for_status()
        .context("Anthropic authorization code exchange failed")?;

    let payload: Value = response
        .json()
        .await
        .context("failed to decode Anthropic token response")?;
    build_tokens(payload)
}

pub async fn refresh_token(
    client: &reqwest::Client,
    refresh_token: &str,
) -> anyhow::Result<OAuthTokens> {
    let response = client
        .post(TOKEN_URL)
        .header("anthropic-beta", OAUTH_BETA_HEADER)
        .json(&json!({
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        }))
        .send()
        .await
        .context("failed to refresh Anthropic token")?
        .error_for_status()
        .context("Anthropic token refresh failed")?;

    let payload: Value = response
        .json()
        .await
        .context("failed to decode Anthropic refresh response")?;
    build_tokens(payload)
}

pub fn is_oauth_token(key: &str) -> bool {
    key.starts_with(OAUTH_TOKEN_PREFIX)
}

fn build_tokens(payload: Value) -> anyhow::Result<OAuthTokens> {
    let object = payload
        .as_object()
        .ok_or_else(|| anyhow!("Anthropic token response was not a JSON object"))?;

    let access_token = object
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing access_token in Anthropic response"))?
        .to_owned();

    let refresh_token = object
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::to_owned);

    let token_type = object
        .get("token_type")
        .and_then(Value::as_str)
        .unwrap_or("Bearer")
        .to_owned();

    let scope = object.get("scope").and_then(Value::as_str).map(str::to_owned);
    let created_at = object
        .get("created_at")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let expires_at = object
        .get("expires_in")
        .and_then(Value::as_i64)
        .map(|expires_in| now_millis().saturating_add(expires_in.saturating_mul(1000)));

    let mut extra = HashMap::new();
    for (key, value) in object {
        if matches!(
            key.as_str(),
            "access_token"
                | "refresh_token"
                | "token_type"
                | "scope"
                | "expires_in"
                | "created_at"
        ) {
            continue;
        }
        extra.insert(key.clone(), value.clone());
    }

    Ok(OAuthTokens {
        access_token,
        refresh_token,
        token_type,
        scope,
        expires_at,
        created_at,
        extra,
    })
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as i64
}
