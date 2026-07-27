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

/// A lock file older than this is treated as abandoned. The owner is only
/// supposed to hold it for the few milliseconds of a token read/write, so any
/// lock this old belongs to a process that died (crash, SIGKILL, `systemctl
/// restart`) before `Drop` could clean up.
const LOCK_STALE_AFTER: Duration = Duration::from_secs(30);

/// Upper bound on how long to wait for a lock held by a *live* owner. Failing
/// with an actionable error beats hanging forever.
const LOCK_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(15);

const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug)]
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    async fn acquire(path: &Path) -> anyhow::Result<Self> {
        Self::acquire_with(path, LOCK_STALE_AFTER, LOCK_ACQUIRE_TIMEOUT).await
    }

    async fn acquire_with(
        path: &Path,
        stale_after: Duration,
        timeout: Duration,
    ) -> anyhow::Result<Self> {
        use tokio::io::AsyncWriteExt;

        let start = std::time::Instant::now();
        loop {
            match tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .await
            {
                Ok(mut file) => {
                    // Record the owner so a stale lock can be attributed.
                    let _ = file.write_all(std::process::id().to_string().as_bytes()).await;
                    let _ = file.flush().await;
                    return Ok(Self {
                        path: path.to_path_buf(),
                    });
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                    if Self::reclaim_if_stale(path, stale_after).await {
                        continue;
                    }
                    if start.elapsed() >= timeout {
                        let owner = tokio::fs::read_to_string(path)
                            .await
                            .ok()
                            .filter(|s| !s.trim().is_empty())
                            .map(|s| format!(" (created by pid {})", s.trim()))
                            .unwrap_or_default();
                        anyhow::bail!(
                            "timed out after {:?} waiting for lock {}{owner}. \
                             If no llm-gateway process is running, delete that file.",
                            timeout,
                            path.display(),
                        );
                    }
                    tokio::time::sleep(LOCK_RETRY_INTERVAL).await;
                }
                Err(err) => {
                    return Err(err)
                        .with_context(|| format!("failed to acquire lock {}", path.display()));
                }
            }
        }
    }

    /// Delete a lock file whose owner is gone. Returns true if it was reclaimed,
    /// so the caller can immediately retry the create.
    async fn reclaim_if_stale(path: &Path, stale_after: Duration) -> bool {
        let Ok(metadata) = tokio::fs::metadata(path).await else {
            // Vanished between the failed create and now — retrying is correct.
            return true;
        };
        let age = metadata
            .modified()
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .unwrap_or_default();
        if age < stale_after {
            return false;
        }
        tokio::fs::remove_file(path).await.is_ok()
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod lock_tests {
    use super::*;

    fn temp_lock_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("llm-gateway-{name}-{}.lock", std::process::id()))
    }

    #[tokio::test]
    async fn acquire_creates_and_drop_removes() {
        let path = temp_lock_path("basic");
        let _ = std::fs::remove_file(&path);

        {
            let _lock = FileLock::acquire(&path).await.expect("should acquire");
            assert!(path.exists(), "lock file should exist while held");
        }
        assert!(!path.exists(), "Drop should remove the lock file");
    }

    /// Regression: a lock left behind by a killed process used to make every
    /// later token load/save spin at 50ms forever, hanging both `login` and
    /// every inference request that had to read the token.
    #[tokio::test]
    async fn stale_lock_is_reclaimed_instead_of_hanging() {
        let path = temp_lock_path("stale");
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, b"").expect("write orphan lock");

        tokio::time::sleep(Duration::from_millis(60)).await;

        let lock = tokio::time::timeout(
            Duration::from_secs(2),
            FileLock::acquire_with(&path, Duration::from_millis(50), Duration::from_secs(1)),
        )
        .await
        .expect("must not hang on a stale lock")
        .expect("must reclaim a stale lock");

        drop(lock);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn live_lock_times_out_with_actionable_error() {
        let path = temp_lock_path("live");
        let _ = std::fs::remove_file(&path);

        let held = FileLock::acquire(&path).await.expect("should acquire");

        // stale_after is long, so the held lock is never reclaimed and we must
        // hit the timeout rather than spin forever.
        let err = FileLock::acquire_with(&path, Duration::from_secs(60), Duration::from_millis(150))
            .await
            .expect_err("must not acquire a live lock");
        let msg = err.to_string();
        assert!(msg.contains("timed out"), "unexpected error: {msg}");
        assert!(
            msg.contains(&std::process::id().to_string()),
            "error should name the lock owner: {msg}"
        );

        drop(held);
    }
}
