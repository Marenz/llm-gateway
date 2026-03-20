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
    },
    /// Run the ChatGPT/OpenAI device code OAuth flow
    Chatgpt {
        /// Path to save the auth tokens
        #[arg(long)]
        token_file: Option<PathBuf>,
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
        Some(Command::Login { provider }) => handle_login(provider).await,
        Some(Command::Status) => handle_status(&config_path).await,
        Some(Command::Serve { host, port }) => handle_serve(&config_path, host, port).await,
        None => handle_serve(&config_path, None, None).await,
    }
}

async fn handle_login(provider: LoginProvider) -> anyhow::Result<()> {
    let client = reqwest::Client::new();

    match provider {
        LoginProvider::Anthropic { token_file } => {
            let token_path = token_file.unwrap_or_else(default_anthropic_token_path);
            let store = oauth::token_store::TokenStore::new(token_path.clone());

            println!("Starting Anthropic OAuth flow...\n");

            // Generate PKCE challenge
            let (verifier, challenge) = oauth::anthropic::generate_pkce();
            let state = oauth::anthropic::generate_state();
            let auth_url = oauth::anthropic::get_authorization_url(&challenge, &state);

            println!("Open this URL in your browser:\n");
            println!("  {auth_url}\n");
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

            if let Some(expires_at) = tokens.expires_at {
                let hours = (expires_at - now_millis()) / 1000 / 3600;
                println!("Token expires in ~{hours} hours");
            }
            if tokens.refresh_token.is_some() {
                println!("Refresh token stored — tokens will auto-refresh");
            }
        }

        LoginProvider::Chatgpt { token_file } => {
            let token_path = token_file.unwrap_or_else(default_chatgpt_token_path);
            let store = oauth::token_store::TokenStore::new(token_path.clone());

            println!("Starting ChatGPT OAuth flow...\n");

            let tokens = oauth::chatgpt::login_browser(&client).await?;

            if let Some(parent) = token_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            store.save(&tokens).await?;
            println!("\nChatGPT OAuth tokens saved to {}", token_path.display());

            if let Some(account_id) = tokens.extra.get("account_id") {
                println!("Account ID: {account_id}");
            }
            if tokens.refresh_token.is_some() {
                println!("Refresh token stored — tokens will auto-refresh");
            }
        }
    }

    Ok(())
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
