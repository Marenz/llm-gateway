mod config;
mod oauth;
mod providers;
mod router;
mod translate;
mod types;

use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "llm-gateway", version, about = "Generic LLM proxy/router")]
struct Cli {
    /// Path to configuration file (JSON)
    #[arg(short, long, global = true)]
    config: Option<String>,

    /// Log level (quiet, minimal, normal, verbose)
    #[arg(long, global = true)]
    log_level: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start the proxy server (default when no subcommand given)
    Serve {
        /// Listen host
        #[arg(long)]
        host: Option<String>,
        /// Listen port
        #[arg(long)]
        port: Option<u16>,
    },

    /// Authenticate with a provider's OAuth flow
    Login {
        /// Provider to log in to
        #[command(subcommand)]
        provider: LoginProvider,
    },

    /// Show current auth status for all providers
    Status,
}

#[derive(Subcommand)]
enum LoginProvider {
    /// Run the Anthropic OAuth PKCE flow
    Anthropic {
        /// Path to save the OAuth tokens (default: ~/.config/llm-gateway/anthropic-oauth.json)
        #[arg(long)]
        token_file: Option<PathBuf>,
        /// Only print the authorization URL; do not try to open a browser
        /// (useful on headless/remote machines)
        #[arg(long)]
        show_url_only: bool,
    },
    /// Run the ChatGPT/OpenAI device code OAuth flow
    Chatgpt {
        /// Path to save the auth tokens
        #[arg(long)]
        token_file: Option<PathBuf>,
        /// Only print the authorization URL; do not try to open a browser
        /// (useful on headless/remote machines)
        #[arg(long)]
        show_url_only: bool,
    },
    /// Store a DeepSeek API key (paste it when prompted)
    Deepseek {
        /// API key to store. If omitted, you'll be prompted to paste it.
        #[arg(long)]
        api_key: Option<String>,
        /// Path to save the key (default: ~/.config/llm-gateway/deepseek-key.txt)
        #[arg(long)]
        key_file: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = match cli.log_level.as_deref() {
        Some("quiet") => "warn",
        Some("verbose") => "debug",
        Some("minimal") => "info",
        _ => "info",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .init();

    let config_path = cli.config.unwrap_or_else(default_config_path);
    match cli.command {
        Some(Command::Login { provider }) => handle_login(provider, &config_path).await,
        Some(Command::Status) => handle_status(&config_path).await,
        Some(Command::Serve { host, port }) => handle_serve(&config_path, host, port).await,
        None => handle_serve(&config_path, None, None).await,
    }
}

async fn handle_login(provider: LoginProvider, config_path: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();

    // Load the effective config (if any) so we can warn when a login won't take effect.
    let loaded_config = {
        let path = std::path::Path::new(config_path);
        if path.exists() {
            match config::GatewayConfig::from_file(path) {
                Ok(mut cfg) => {
                    cfg.resolve_env_vars();
                    Some(cfg)
                }
                Err(err) => {
                    eprintln!("warning: could not read config at {config_path}: {err}");
                    None
                }
            }
        } else {
            None
        }
    };

    match provider {
        LoginProvider::Anthropic { token_file, show_url_only } => {
            let token_path = token_file.unwrap_or_else(default_anthropic_token_path);
            let store = oauth::token_store::TokenStore::new(token_path.clone());

            println!("Starting Anthropic OAuth flow...\n");

            // Generate PKCE challenge
            let (verifier, challenge) = oauth::anthropic::generate_pkce();
            let state = oauth::anthropic::generate_state();
            let auth_url = oauth::anthropic::get_authorization_url(&challenge, &state);

            println!("Open this URL in your browser:\n");
            println!("  {auth_url}\n");
            open_browser(&auth_url, show_url_only);
            println!("After authorizing, you'll be redirected to a page showing a code.");
            println!("The URL will look like:  ...callback?code=CODE#STATE");
            println!("Paste the full 'code#state' value below.\n");

            // Read code#state from stdin
            let mut input = String::new();
            eprint!("code#state> ");
            std::io::stdin()
                .read_line(&mut input)
                .context("failed to read input")?;
            let input = input.trim();

            let (code, returned_state) = input
                .split_once('#')
                .ok_or_else(|| anyhow::anyhow!("expected format: CODE#STATE"))?;

            if returned_state != state {
                anyhow::bail!("state mismatch: expected {state}, got {returned_state}");
            }

            // Exchange code for tokens
            println!("Exchanging authorization code...");
            let tokens =
                oauth::anthropic::exchange_code(&client, code, &verifier, &state).await?;

            store.save(&tokens).await?;
            println!("Anthropic OAuth tokens saved to {}", token_path.display());

            warn_anthropic_no_effect(loaded_config.as_ref(), &token_path, config_path);

            if let Some(expires_at) = tokens.expires_at {
                let hours = (expires_at - now_millis()) / 1000 / 3600;
                println!("Token expires in ~{hours} hours");
            }
            if tokens.refresh_token.is_some() {
                println!("Refresh token stored — tokens will auto-refresh");
            }
        }

        LoginProvider::Chatgpt { token_file, show_url_only } => {
            let token_path = token_file.unwrap_or_else(default_chatgpt_token_path);
            let store = oauth::token_store::TokenStore::new(token_path.clone());

            println!("Starting ChatGPT OAuth flow...\n");

            let tokens = oauth::chatgpt::login_browser(&client, show_url_only).await?;

            if let Some(parent) = token_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            store.save(&tokens).await?;
            println!("\nChatGPT OAuth tokens saved to {}", token_path.display());

            warn_chatgpt_no_effect(loaded_config.as_ref(), &token_path, config_path);

            if let Some(account_id) = tokens.extra.get("account_id") {
                println!("Account ID: {account_id}");
            }
            if tokens.refresh_token.is_some() {
                println!("Refresh token stored — tokens will auto-refresh");
            }
        }

        LoginProvider::Deepseek { api_key, key_file } => {
            let key_path = key_file
                .or_else(config::default_deepseek_key_path)
                .ok_or_else(|| anyhow::anyhow!("could not determine config dir for key file"))?;

            let key = match api_key {
                Some(key) => key,
                None => {
                    eprint!("Paste your DeepSeek API key: ");
                    std::io::Write::flush(&mut std::io::stderr()).ok();
                    let mut input = String::new();
                    std::io::stdin()
                        .read_line(&mut input)
                        .context("failed to read input")?;
                    input.trim().to_string()
                }
            };

            if key.is_empty() {
                anyhow::bail!("no API key provided");
            }

            if let Some(parent) = key_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&key_path, format!("{key}\n"))
                .with_context(|| format!("failed to write {}", key_path.display()))?;

            // Restrict permissions to the user (0600) on Unix.
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600));
            }

            println!("DeepSeek API key saved to {}", key_path.display());

            // Warn if a higher-precedence key will shadow this one.
            let shadowed_by_config = loaded_config.as_ref().is_some_and(|cfg| {
                cfg.providers.iter().any(|p| {
                    matches!(p, config::ProviderConfig::DeepSeek(c)
                        if c.api_key.as_ref().is_some_and(|k| !k.is_empty()))
                })
            });
            if shadowed_by_config {
                eprintln!(
                    "warning: this key will have NO EFFECT — the DeepSeek provider in {config_path} \
                     already has `api_key` set, which takes precedence over the key file. \
                     Remove `api_key` (or set it to \"env:DEEPSEEK_API_KEY\") to use the saved key."
                );
            } else if loaded_config.is_none() && std::env::var("DEEPSEEK_API_KEY").is_ok() {
                eprintln!(
                    "warning: this key may have NO EFFECT — DEEPSEEK_API_KEY is set in the \
                     environment and takes precedence over the key file."
                );
            } else {
                println!("Restart the gateway to pick up the new key.");
            }
        }
    }

    Ok(())
}

/// Warn if Anthropic OAuth tokens just saved won't be used by the gateway.
fn warn_anthropic_no_effect(
    config: Option<&config::GatewayConfig>,
    saved_path: &std::path::Path,
    config_path: &str,
) {
    let Some(config) = config else {
        // No config file → default_config() always registers Anthropic at the
        // default token path, so a default-path save is fine.
        return;
    };
    let provider = config.providers.iter().find_map(|p| match p {
        config::ProviderConfig::Anthropic(c) => Some(c),
        _ => None,
    });
    let Some(provider) = provider else {
        eprintln!(
            "warning: this login will have NO EFFECT — no `anthropic` provider is configured \
             in {config_path}, so the gateway will never load these tokens."
        );
        return;
    };
    let expected = provider.oauth_token_file.clone().unwrap_or_else(default_anthropic_token_path);
    if !same_path(&expected, saved_path) {
        eprintln!(
            "warning: this login may have NO EFFECT — the `anthropic` provider in {config_path} \
             reads tokens from {}, but they were saved to {}. Use --token-file {} (or update the \
             config's `oauth_token_file`).",
            expected.display(),
            saved_path.display(),
            expected.display(),
        );
    }
}

/// Warn if ChatGPT OAuth tokens just saved won't be used by the gateway.
fn warn_chatgpt_no_effect(
    config: Option<&config::GatewayConfig>,
    saved_path: &std::path::Path,
    config_path: &str,
) {
    let Some(config) = config else {
        return;
    };
    let provider = config.providers.iter().find_map(|p| match p {
        config::ProviderConfig::Chatgpt(c) => Some(c),
        _ => None,
    });
    let Some(provider) = provider else {
        eprintln!(
            "warning: this login will have NO EFFECT — no `chatgpt` provider is configured \
             in {config_path}, so the gateway will never load these tokens."
        );
        return;
    };
    let expected = provider.token_file.clone().unwrap_or_else(default_chatgpt_token_path);
    if !same_path(&expected, saved_path) {
        eprintln!(
            "warning: this login may have NO EFFECT — the `chatgpt` provider in {config_path} \
             reads tokens from {}, but they were saved to {}. Use --token-file {} (or update the \
             config's `token_file`).",
            expected.display(),
            saved_path.display(),
            expected.display(),
        );
    }
}

/// Compare two paths, canonicalizing where possible so relative/symlinked paths
/// that point to the same file are treated as equal.
fn same_path(a: &std::path::Path, b: &std::path::Path) -> bool {
    let canon = |p: &std::path::Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    canon(a) == canon(b)
}

/// Try to open the given URL in the user's browser unless `show_url_only` is set.
fn open_browser(url: &str, show_url_only: bool) {
    if show_url_only {
        println!("(--show-url-only set — not opening a browser)");
        return;
    }
    match std::process::Command::new("xdg-open").arg(url).spawn() {
        Ok(_) => println!("Opening browser..."),
        Err(_) => println!("(could not auto-open browser — open the URL above manually)"),
    }
}

async fn handle_status(config_path: &str) -> anyhow::Result<()> {
    let path = std::path::Path::new(config_path);
    let config = if path.exists() {
        config::GatewayConfig::from_file(path)?
    } else {
        default_config()
    };

    println!("Provider status:\n");

    for provider in &config.providers {
        match provider {
            config::ProviderConfig::Anthropic(cfg) => {
                print!("  anthropic ({}): ", cfg.name);
                let token_path = cfg.oauth_token_file.clone().or_else(|| {
                    dirs::config_dir().map(|d| d.join("llm-gateway").join("anthropic-oauth.json"))
                });
                let store = token_path.map(oauth::token_store::TokenStore::new);
                match store {
                    Some(store) => match store.load().await? {
                        Some(tokens) => {
                            if oauth::token_store::TokenStore::is_expired(&tokens) {
                                if tokens.refresh_token.is_some() {
                                    println!("token expired (has refresh token, will auto-refresh)");
                                } else {
                                    println!("token expired (no refresh token — run: llm-gateway login anthropic)");
                                }
                            } else {
                                let hours_left = tokens.expires_at
                                    .map(|e| (e - now_millis()) / 1000 / 3600)
                                    .unwrap_or(0);
                                print!("authenticated (expires in ~{hours_left}h)");
                                if cfg.api_key.is_some() {
                                    print!(", api key also set");
                                }
                                println!();
                            }
                        }
                        None => {
                            if cfg.api_key.is_some() {
                                println!("api key configured (no oauth token — run: llm-gateway login anthropic)");
                            } else {
                                println!("not configured (run: llm-gateway login anthropic)");
                            }
                        }
                    },
                    None => println!("not configured"),
                }
            }
            config::ProviderConfig::Chatgpt(cfg) => {
                print!("  chatgpt ({}): ", cfg.name);
                let token_path = cfg.token_file.clone()
                    .unwrap_or_else(|| default_chatgpt_token_path());
                let store = oauth::token_store::TokenStore::new(token_path);
                match store.load().await? {
                    Some(tokens) => {
                        if oauth::token_store::TokenStore::is_expired(&tokens) {
                            if tokens.refresh_token.is_some() {
                                println!("token expired (has refresh token, will auto-refresh)");
                            } else {
                                println!("token expired (no refresh token!)");
                            }
                        } else {
                            let hours_left = tokens.expires_at
                                .map(|e| (e - now_millis()) / 1000 / 3600)
                                .unwrap_or(0);
                            println!("authenticated (expires in ~{hours_left}h)");
                        }
                    }
                    None => println!("not logged in (run: llm-gateway login chatgpt)"),
                }
            }
            config::ProviderConfig::Openai(cfg) => {
                print!("  openai ({}): ", cfg.name);
                if cfg.api_key.is_some() {
                    println!("api key configured");
                } else {
                    println!("not configured");
                }
            }
            config::ProviderConfig::XiaomiMimo(cfg) => {
                print!("  xiaomi-mimo ({}): ", cfg.name);
                if cfg.api_key.is_some() {
                    println!("api key configured");
                } else {
                    println!("not configured");
                }
            }
            config::ProviderConfig::OpenaiCompatible(cfg) => {
                print!("  {} ({}): ", cfg.name, cfg.api_base);
                if cfg.api_key.is_some() {
                    println!("api key configured");
                } else {
                    println!("no api key");
                }
            }
            config::ProviderConfig::OpencodeGo(cfg) => {
                print!("  opencode-go ({}): ", cfg.name);
                if cfg.api_key.is_some() {
                    println!("api key configured");
                } else {
                    println!("no api key");
                }
            }
            config::ProviderConfig::Zen(cfg) => {
                print!("  zen ({}): ", cfg.name);
                if cfg.api_key.is_some() {
                    println!("api key configured");
                } else {
                    println!("no api key");
                }
            }
            config::ProviderConfig::DeepSeek(cfg) => {
                print!("  deepseek ({}): ", cfg.name);
                if cfg.api_key.is_some() {
                    println!("api key configured");
                } else if cfg.resolve_key().is_some() {
                    println!("api key configured (from key file)");
                } else {
                    println!("no api key (run: llm-gateway login deepseek)");
                }
            }
        }
    }

    Ok(())
}

async fn handle_serve(
    config_path: &str,
    host: Option<String>,
    port: Option<u16>,
) -> anyhow::Result<()> {
    let path = std::path::Path::new(config_path);
    let mut config = if path.exists() {
        config::GatewayConfig::from_file(path)?
    } else {
        tracing::warn!(path = %config_path, "config file not found, using default config");
        default_config()
    };

    if let Some(host) = host {
        config.host = host;
    }
    if let Some(port) = port {
        config.port = port;
    }

    config.resolve_env_vars();
    router::server::run(config).await
}

/// Build a default config from environment variables when no config file exists.
fn default_config() -> config::GatewayConfig {
    let mut providers = Vec::new();

    // Always add Anthropic — token store at ~/.config/llm-gateway/anthropic-oauth.json
    // is checked automatically; api_key is a fallback if the env var is set.
    providers.push(config::ProviderConfig::Anthropic(
        config::AnthropicProviderConfig {
            name: "anthropic".to_string(),
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            oauth_token_file: None,
            api_base: "https://api.anthropic.com".to_string(),
            allow_bearer_passthrough: false,
            required_system_prompt: None,
            models: vec![],
        },
    ));

    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        providers.push(config::ProviderConfig::Openai(
            config::OpenaiProviderConfig {
                name: "openai".to_string(),
                api_key: Some(key),
                organization: std::env::var("OPENAI_ORGANIZATION").ok(),
                api_base: "https://api.openai.com/v1".to_string(),
                models: vec![],
            },
        ));
    }

    if let Ok(key) = std::env::var("XIAOMI_MIMO_API_KEY") {
        providers.push(config::ProviderConfig::XiaomiMimo(
            config::XiaomiMimoProviderConfig {
                name: "xiaomi-mimo".to_string(),
                api_key: Some(key),
                api_base: "https://api.xiaomimimo.com/v1".to_string(),
                models: vec![],
            },
        ));
    }

    if let Ok(key) = std::env::var("OPENCODE_GO_API_KEY") {
        providers.push(config::ProviderConfig::OpencodeGo(
            config::OpencodeGoProviderConfig {
                name: "opencode-go".to_string(),
                api_key: Some(key),
                openai_api_base: "https://opencode.ai/zen/go".to_string(),
                anthropic_api_base: "https://opencode.ai/zen/go".to_string(),
                models: vec![],
            },
        ));
    }

    // Always register DeepSeek: the env var is optional, since the key may have
    // been stored via `llm-gateway login deepseek` (read from the key file).
    providers.push(config::ProviderConfig::DeepSeek(
        config::DeepSeekProviderConfig {
            name: "deepseek".to_string(),
            api_key: std::env::var("DEEPSEEK_API_KEY").ok(),
            api_key_file: None,
            api_base: "https://api.deepseek.com/v1".to_string(),
            models: vec![],
        },
    ));

    providers.push(config::ProviderConfig::Chatgpt(
        config::ChatgptProviderConfig {
            name: "chatgpt".to_string(),
            token_file: None,
            api_base: "https://chatgpt.com/backend-api/codex".to_string(),
            models: vec![],
        },
    ));

    config::GatewayConfig {
        host: "127.0.0.1".to_string(),
        port: 4000,
        master_key: std::env::var("LLM_GATEWAY_MASTER_KEY").ok(),
        providers,
        model_aliases: Default::default(),
        virtual_models: Default::default(),
        log_level: config::LogLevel::Normal,
    }
}

fn default_config_path() -> String {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llm-gateway")
        .join("config.json")
        .to_string_lossy()
        .into_owned()
}

fn default_anthropic_token_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llm-gateway")
        .join("anthropic-oauth.json")
}

fn default_chatgpt_token_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llm-gateway")
        .join("chatgpt-auth.json")
}

fn now_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
