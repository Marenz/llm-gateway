use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthTokens {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    pub token_type: String,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub expires_at: Option<i64>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

pub struct TokenStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl TokenStore {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            lock: Mutex::new(()),
        }
    }

    pub async fn load(&self) -> anyhow::Result<Option<OAuthTokens>> {
        // Fast path: if the file doesn't exist, skip locking entirely
        if !self.path.exists() {
            return Ok(None);
        }

        let _guard = self.lock.lock().await;
        let lock_path = self.lock_path();
        // Ensure parent directory exists for the lock file
        if let Some(parent) = lock_path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        let _file_lock = FileLock::acquire(&lock_path).await?;

        match tokio::fs::read_to_string(&self.path).await {
            Ok(contents) => {
                let tokens = serde_json::from_str(&contents)
                    .with_context(|| format!("failed to parse token file {}", self.path.display()))?;
                Ok(Some(tokens))
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err)
                .with_context(|| format!("failed to read token file {}", self.path.display())),
        }
    }

    pub async fn save(&self, tokens: &OAuthTokens) -> anyhow::Result<()> {
        let _guard = self.lock.lock().await;
        let lock_path = self.lock_path();
        if let Some(parent) = lock_path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        let _file_lock = FileLock::acquire(&lock_path).await?;

        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }

        let json = serde_json::to_string_pretty(tokens).context("failed to serialize tokens")?;
        tokio::fs::write(&self.path, json)
            .await
            .with_context(|| format!("failed to write token file {}", self.path.display()))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let permissions = std::fs::Permissions::from_mode(0o600);
            tokio::fs::set_permissions(&self.path, permissions)
                .await
                .with_context(|| format!("failed to set permissions on {}", self.path.display()))?;
        }

        Ok(())
    }

    pub async fn delete(&self) -> anyhow::Result<()> {
        let _guard = self.lock.lock().await;
        let lock_path = self.lock_path();
        let _file_lock = FileLock::acquire(&lock_path).await?;

        match tokio::fs::remove_file(&self.path).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err)
                .with_context(|| format!("failed to delete token file {}", self.path.display())),
        }
    }

    pub fn is_expired(tokens: &OAuthTokens) -> bool {
        Self::is_expired_with_buffer(tokens, 300)
    }

    pub fn is_expired_with_buffer(tokens: &OAuthTokens, buffer_secs: i64) -> bool {
        let Some(expires_at) = tokens.expires_at else {
            return true;
        };

        let now_ms = now_millis();
        let buffer_ms = buffer_secs.saturating_mul(1000);
        expires_at <= now_ms.saturating_add(buffer_ms)
    }

    fn lock_path(&self) -> PathBuf {
        let mut lock_name = self
            .path
            .file_name()
            .map(|name| name.to_os_string())
            .unwrap_or_else(|| "tokens".into());
        lock_name.push(".lock");
        self.path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(lock_name)
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as i64
}

struct FileLock {
    path: PathBuf,
}

impl FileLock {
    async fn acquire(path: &Path) -> anyhow::Result<Self> {
        loop {
            match tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .await
            {
                Ok(_) => {
                    return Ok(Self {
                        path: path.to_path_buf(),
                    });
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(err) => {
                    return Err(err)
                        .with_context(|| format!("failed to acquire lock {}", path.display()));
                }
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
