use std::collections::HashMap;
use std::time::Duration;

use anyhow::{anyhow, Context};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::token_store::OAuthTokens;

pub const AUTH_BASE: &str = "https://auth.openai.com";
pub const DEVICE_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
pub const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
pub const OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const DEVICE_VERIFY_URL: &str = "https://auth.openai.com/codex/device";
pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const TOKEN_EXPIRY_SKEW_SECS: i64 = 60;
pub const DEVICE_CODE_TIMEOUT_SECS: u64 = 900;
pub const DEVICE_CODE_POLL_SLEEP_SECS: u64 = 5;
pub const OAUTH_REDIRECT_PORT: u16 = 1455;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_auth_id: String,
    pub user_code: String,
    #[serde(default, deserialize_with = "deserialize_string_or_number")]
    pub interval: Option<u64>,
}

fn deserialize_string_or_number<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct StringOrNumber;

    impl<'de> de::Visitor<'de> for StringOrNumber {
        type Value = Option<u64>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a number or numeric string")
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v))
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(Some(v as u64))
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            v.parse::<u64>().map(Some).map_err(de::Error::custom)
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
    }

    deserializer.deserialize_any(StringOrNumber)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationCodeResponse {
    pub authorization_code: String,
    pub code_challenge: String,
    pub code_verifier: String,
}

pub async fn request_device_code(client: &reqwest::Client) -> anyhow::Result<DeviceCodeResponse> {
    client
        .post(DEVICE_CODE_URL)
        .json(&json!({ "client_id": CLIENT_ID }))
        .send()
        .await
        .context("failed to request ChatGPT device code")?
        .error_for_status()
        .context("ChatGPT device code request failed")?
        .json()
        .await
        .context("failed to decode ChatGPT device code response")
}

pub async fn poll_for_authorization(
    client: &reqwest::Client,
    device_code: &DeviceCodeResponse,
) -> anyhow::Result<AuthorizationCodeResponse> {
    let poll_interval = device_code
        .interval
        .unwrap_or(DEVICE_CODE_POLL_SLEEP_SECS)
        .max(DEVICE_CODE_POLL_SLEEP_SECS);
    let started = tokio::time::Instant::now();

    loop {
        if started.elapsed() >= Duration::from_secs(DEVICE_CODE_TIMEOUT_SECS) {
            return Err(anyhow!("timed out waiting for ChatGPT device authorization"));
        }

        let response = client
            .post(DEVICE_TOKEN_URL)
            .json(&json!({
                "client_id": CLIENT_ID,
                "device_auth_id": device_code.device_auth_id,
                "user_code": device_code.user_code,
            }))
            .send()
            .await
            .context("failed to poll ChatGPT device authorization")?;

        match response.status().as_u16() {
            200 => {
                let auth: AuthorizationCodeResponse = response
                    .json()
                    .await
                    .context("failed to decode ChatGPT authorization response")?;
                return Ok(auth);
            }
            403 | 404 => {
                tokio::time::sleep(Duration::from_secs(poll_interval)).await;
            }
            _ => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!(
                    "ChatGPT authorization polling failed with status {}: {}",
                    status,
                    body
                ));
            }
        }
    }
}

pub async fn exchange_code(
    client: &reqwest::Client,
    auth_code: &AuthorizationCodeResponse,
    redirect_uri: &str,
) -> anyhow::Result<OAuthTokens> {
    let response = client
        .post(OAUTH_TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", auth_code.authorization_code.as_str()),
            ("redirect_uri", redirect_uri),
            ("client_id", CLIENT_ID),
            ("code_verifier", auth_code.code_verifier.as_str()),
        ])
        .send()
        .await
        .context("failed to exchange ChatGPT authorization code")?
        .error_for_status()
        .context("ChatGPT authorization code exchange failed")?;

    let payload: Value = response
        .json()
        .await
        .context("failed to decode ChatGPT token response")?;
    build_tokens(payload)
}

pub async fn refresh_token(
    client: &reqwest::Client,
    refresh_token: &str,
) -> anyhow::Result<OAuthTokens> {
    let response = client
        .post(OAUTH_TOKEN_URL)
        .json(&json!({
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
            "scope": "openid profile email",
        }))
        .send()
        .await
        .context("failed to refresh ChatGPT token")?
        .error_for_status()
        .context("ChatGPT token refresh failed")?;

    let payload: Value = response
        .json()
        .await
        .context("failed to decode ChatGPT refresh response")?;
    build_tokens(payload)
}

pub fn decode_jwt_exp(token: &str) -> Option<i64> {
    let payload = decode_jwt_payload(token)?;
    let exp = payload.get("exp")?.as_i64()?;
    Some(exp.saturating_sub(TOKEN_EXPIRY_SKEW_SECS).saturating_mul(1000))
}

pub fn extract_account_id(token: &str) -> Option<String> {
    let payload = decode_jwt_payload(token)?;
    payload
        .get("https://api.openai.com/auth")?
        .get("chatgpt_account_id")?
        .as_str()
        .map(str::to_owned)
}

fn build_tokens(payload: Value) -> anyhow::Result<OAuthTokens> {
    let object = payload
        .as_object()
        .ok_or_else(|| anyhow!("ChatGPT token response was not a JSON object"))?;

    let access_token = object
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing access_token in ChatGPT response"))?
        .to_owned();

    let refresh_token = object
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let id_token = object.get("id_token").and_then(Value::as_str).map(str::to_owned);
    let token_type = object
        .get("token_type")
        .and_then(Value::as_str)
        .unwrap_or("Bearer")
        .to_owned();
    let scope = object.get("scope").and_then(Value::as_str).map(str::to_owned);

    let expires_at = decode_jwt_exp(&access_token);
    let created_at = object
        .get("created_at")
        .and_then(Value::as_str)
        .map(str::to_owned);

    let mut extra = HashMap::new();
    if let Some(id_token) = id_token {
        extra.insert("id_token".to_string(), Value::String(id_token.clone()));
        if let Some(account_id) = extract_account_id(&id_token) {
            extra.insert("account_id".to_string(), Value::String(account_id));
        }
    }

    for (key, value) in object {
        if matches!(
            key.as_str(),
            "access_token" | "refresh_token" | "id_token" | "token_type" | "scope" | "created_at"
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

fn decode_jwt_payload(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let decoded = URL_SAFE_NO_PAD.decode(payload).ok()?;
    serde_json::from_slice(&decoded).ok()
}

/// Full browser-based OAuth flow: spins up a local server on port 1455,
/// builds the authorize URL with PKCE, opens the browser, and waits for the callback.
pub async fn login_browser(
    http_client: &reqwest::Client,
    show_url_only: bool,
) -> anyhow::Result<OAuthTokens> {
    use rand::RngCore;
    use sha2::{Digest, Sha256};
    use tokio::sync::oneshot;

    // Generate PKCE
    let mut verifier_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut verifier_bytes);
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));

    // Generate state
    let mut state_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut state_bytes);
    let state = URL_SAFE_NO_PAD.encode(state_bytes);

    let redirect_uri = format!("http://localhost:{OAUTH_REDIRECT_PORT}/auth/callback");

    // Build authorize URL
    let auth_url = {
        let mut url = reqwest::Url::parse(AUTHORIZE_URL).unwrap();
        url.query_pairs_mut()
            .append_pair("response_type", "code")
            .append_pair("client_id", CLIENT_ID)
            .append_pair("redirect_uri", &redirect_uri)
            .append_pair("scope", "openid profile email offline_access")
            .append_pair("code_challenge", &challenge)
            .append_pair("code_challenge_method", "S256")
            .append_pair("id_token_add_organizations", "true")
            .append_pair("codex_cli_simplified_flow", "true")
            .append_pair("originator", "llm-gateway")
            .append_pair("state", &state);
        url.to_string()
    };

    println!("Open this URL in your browser:\n");
    println!("  {auth_url}\n");

    // In show-url-only mode the browser is (typically) on a different machine, so
    // the localhost:1455 redirect can't reach us. Ask the user to paste the value
    // they're redirected to back into this terminal instead.
    let (code, returned_state) = if show_url_only {
        println!("(--show-url-only set — not opening a browser, not listening for a callback)");
        prompt_for_pasted_callback().await?
    } else {
        match std::process::Command::new("xdg-open").arg(&auth_url).spawn() {
            Ok(_) => println!("Opening browser..."),
            Err(_) => println!("(could not auto-open browser — open the URL above manually)"),
        }
        println!("Waiting for authorization callback on localhost:{OAUTH_REDIRECT_PORT}... (Ctrl+C to cancel)");
        wait_for_callback(&state).await?
    };

    // Validate state when we got one back. In show-url-only mode the user may
    // paste a bare code with no state; we can't verify CSRF then, so we warn
    // instead of failing.
    if returned_state.is_empty() {
        eprintln!("warning: no `state` in pasted value — skipping CSRF check");
    } else if returned_state != state {
        anyhow::bail!("OAuth state mismatch — possible CSRF");
    }

    println!("Authorized! Exchanging code for tokens...");
    exchange_code(
        http_client,
        &AuthorizationCodeResponse {
            authorization_code: code,
            // The browser flow returns code directly; we use our own verifier
            code_challenge: challenge,
            code_verifier: verifier.clone(),
        },
        &redirect_uri,
    )
    .await
}

/// Run the local callback server and wait for the OAuth redirect, racing against
/// Ctrl+C so the user can cancel. Returns (code, returned_state).
async fn wait_for_callback(state: &str) -> anyhow::Result<(String, String)> {
    use tokio::sync::oneshot;

    let (tx, rx) = oneshot::channel::<anyhow::Result<(String, String)>>();
    let tx = std::sync::Arc::new(tokio::sync::Mutex::new(Some(tx)));

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", OAUTH_REDIRECT_PORT))
        .await
        .with_context(|| format!("failed to bind localhost:{OAUTH_REDIRECT_PORT} for OAuth callback — is another instance running?"))?;

    let state_owned = state.to_string();
    let tx_clone = tx.clone();
    let server = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else { break };
            let tx = tx_clone.clone();
            let state = state_owned.clone();
            tokio::spawn(async move {
                handle_oauth_callback(stream, tx, &state).await;
            });
        }
    });

    // Race the callback against a timeout and Ctrl+C.
    let result = tokio::select! {
        biased;
        _ = tokio::signal::ctrl_c() => {
            server.abort();
            anyhow::bail!("cancelled (Ctrl+C)");
        }
        timed = tokio::time::timeout(Duration::from_secs(300), rx) => {
            server.abort();
            timed
                .context("timed out waiting for OAuth callback")?
                .context("OAuth callback channel closed")?
        }
    };

    result
}

/// Prompt the user to paste the value they were redirected to (full URL,
/// `code#state`, or just the `code`) and parse out (code, state).
async fn prompt_for_pasted_callback() -> anyhow::Result<(String, String)> {
    println!(
        "\nAfter authorizing, your browser is redirected to a URL like:\n  \
         http://localhost:{OAUTH_REDIRECT_PORT}/auth/callback?code=...&state=...\n\
         The page won't load (this machine isn't the browser's localhost) — that's fine.\n\
         Copy that URL from the address bar and paste it here.\n"
    );

    // Read a line from stdin without blocking the Ctrl+C signal handler.
    let line = tokio::select! {
        biased;
        _ = tokio::signal::ctrl_c() => anyhow::bail!("cancelled (Ctrl+C)"),
        line = read_line_async("paste redirect URL (or code#state)> ") => line?,
    };

    parse_pasted_callback(&line)
}

/// Parse a pasted callback value into (code, state). Accepts a full URL,
/// `code#state`, `code&state=...`, or a bare code (state defaults to empty,
/// which the caller treats as a match only if its own state is empty).
fn parse_pasted_callback(input: &str) -> anyhow::Result<(String, String)> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        anyhow::bail!("no input provided");
    }

    // Full URL with query params.
    if let Ok(url) = reqwest::Url::parse(trimmed) {
        if url.query().is_some() || url.scheme().starts_with("http") {
            let params: HashMap<_, _> = url.query_pairs().collect();
            if let Some(err) = params.get("error") {
                anyhow::bail!("authorization failed: {err}");
            }
            let code = params
                .get("code")
                .map(|c| c.to_string())
                .ok_or_else(|| anyhow!("no `code` parameter found in pasted URL"))?;
            let state = params.get("state").map(|s| s.to_string()).unwrap_or_default();
            return Ok((code, state));
        }
    }

    // `code#state` (mirrors the Anthropic flow).
    if let Some((code, state)) = trimmed.split_once('#') {
        return Ok((code.to_string(), state.to_string()));
    }

    // Bare code, no state.
    Ok((trimmed.to_string(), String::new()))
}

/// Read a single line from stdin on a blocking thread so it can be raced in `select!`.
async fn read_line_async(prompt: &str) -> anyhow::Result<String> {
    use std::io::Write;
    eprint!("{prompt}");
    std::io::stderr().flush().ok();
    tokio::task::spawn_blocking(|| {
        let mut buf = String::new();
        std::io::stdin()
            .read_line(&mut buf)
            .context("failed to read input")?;
        Ok::<_, anyhow::Error>(buf)
    })
    .await
    .context("stdin read task failed")?
}

async fn handle_oauth_callback(
    mut stream: tokio::net::TcpStream,
    tx: std::sync::Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Sender<anyhow::Result<(String, String)>>>>>,
    expected_state: &str,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut buf = vec![0u8; 4096];
    let n = match stream.read(&mut buf).await {
        Ok(n) => n,
        Err(_) => return,
    };
    let request = String::from_utf8_lossy(&buf[..n]);

    // Parse the request line: GET /auth/callback?... HTTP/1.1
    let path = request.lines().next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");

    let url = match reqwest::Url::parse(&format!("http://localhost{path}")) {
        Ok(u) => u,
        Err(_) => return,
    };

    let params: HashMap<_, _> = url.query_pairs().collect();
    let code = params.get("code").map(|c| c.to_string());
    let returned_state = params.get("state").map(|s| s.to_string());
    let error = params.get("error").map(|e| e.to_string());

    let (status, body) = if let Some(error) = error {
        (
            "400 Bad Request",
            format!("<html><body><h1>Authorization Failed</h1><p>{error}</p></body></html>"),
        )
    } else if let (Some(code), Some(state)) = (code, returned_state) {
        let result = if state == expected_state {
            Ok((code, state))
        } else {
            Err(anyhow!("state mismatch"))
        };
        let html = match &result {
            Ok(_) => "<html><body><h1>Authorization Successful</h1><p>You can close this tab.</p><script>setTimeout(()=>window.close(),1500)</script></body></html>".to_string(),
            Err(e) => format!("<html><body><h1>Error</h1><p>{e}</p></body></html>"),
        };
        if let Some(sender) = tx.lock().await.take() {
            let _ = sender.send(result);
        }
        ("200 OK", html)
    } else {
        ("400 Bad Request", "<html><body><h1>Missing parameters</h1></body></html>".to_string())
    };

    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
}

#[cfg(test)]
mod tests {
    use super::parse_pasted_callback;

    #[test]
    fn parses_full_redirect_url() {
        let (code, state) = parse_pasted_callback(
            "http://localhost:1455/auth/callback?code=abc123&state=xyz",
        )
        .unwrap();
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz");
    }

    #[test]
    fn parses_code_hash_state() {
        let (code, state) = parse_pasted_callback("abc123#xyz").unwrap();
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz");
    }

    #[test]
    fn parses_bare_code_with_empty_state() {
        let (code, state) = parse_pasted_callback("  abc123  ").unwrap();
        assert_eq!(code, "abc123");
        assert_eq!(state, "");
    }

    #[test]
    fn empty_input_errors() {
        assert!(parse_pasted_callback("   ").is_err());
    }

    #[test]
    fn url_with_error_param_errors() {
        assert!(parse_pasted_callback(
            "http://localhost:1455/auth/callback?error=access_denied"
        )
        .is_err());
    }

    #[test]
    fn url_without_code_errors() {
        assert!(parse_pasted_callback("http://localhost:1455/auth/callback?state=xyz").is_err());
    }
}
