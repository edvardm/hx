//! Cache management for hx.
//!
//! This crate handles:
//! - Global cache directories
//! - Project-local cache (.hx/)
//! - Cache keys and artifact storage
//! - Shared build store with fingerprint tracking
//! - Source fingerprinting for incremental builds
//! - Build state tracking
//! - Binary artifact caching for compiled modules

pub mod artifacts;
pub mod bhc_store;
pub mod build_state;
pub mod source;
pub mod store;

use directories::ProjectDirs;
use hx_core::error::{Error, Result};
use std::path::{Path, PathBuf};
use tracing::debug;

pub use artifacts::{
    ArtifactEntry, ArtifactIndex, ArtifactStats, PruneResult, clear_artifacts,
    compute_artifact_hash, hash_file, prune_artifacts, retrieve_artifacts, store_artifacts,
};
pub use bhc_store::{BhcPackageCacheEntry, BhcPackageCacheIndex, calculate_bhc_cache_key};
pub use build_state::{BuildState, PackageBuildInfo, PackageStatus};
pub use source::{
    SourceFingerprint, compute_source_fingerprint, load_source_fingerprint, save_source_fingerprint,
};
pub use store::{
    BuildCacheEntry, PackageCacheEntry, PackageCacheIndex, PackageCacheStats, StoreIndex,
    StoreStats, calculate_fingerprint, calculate_package_cache_key, store_disk_size,
};

/// Get the global cache directory.
///
/// Respects `$XDG_CACHE_HOME` if set to a non-empty value, on any platform.
/// Otherwise falls back to the platform default:
/// - Linux: `~/.cache/hx`
/// - macOS: `~/Library/Caches/hx`
/// - Windows: `%LOCALAPPDATA%\hx\cache`
pub fn global_cache_dir() -> Result<PathBuf> {
    if let Some(dir) = xdg_dir(std::env::var("XDG_CACHE_HOME").ok(), "hx") {
        return Ok(dir);
    }
    let dirs = ProjectDirs::from("io", "raskell", "hx")
        .ok_or_else(|| Error::config("could not determine home directory for cache"))?;
    Ok(dirs.cache_dir().to_path_buf())
}

/// Get the Cabal store directory within the global cache.
pub fn cabal_store_dir() -> Result<PathBuf> {
    Ok(global_cache_dir()?.join("cabal").join("store"))
}

/// Age of Cabal's own Hackage package index, or whether it exists at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CabalIndexStatus {
    /// No package index found at any known Cabal directory.
    Missing,
    /// Index found, with its age since the last `cabal update`.
    Present { age: std::time::Duration },
}

/// Check Cabal's own Hackage package index (not an hx-managed directory).
///
/// hx doesn't control where this lives, and Cabal's own layout has changed
/// across versions — try the XDG-style cache dir (Cabal 3.10+, respects
/// `$XDG_CACHE_HOME` the same way hx's own dirs do), then the pre-3.10
/// `~/.cabal` layout.
pub fn cabal_index_status() -> CabalIndexStatus {
    let candidates = [
        xdg_dir(std::env::var("XDG_CACHE_HOME").ok(), "cabal"),
        directories::BaseDirs::new().map(|d| d.home_dir().join(".cache").join("cabal")),
        directories::BaseDirs::new().map(|d| d.home_dir().join(".cabal")),
    ];

    for dir in candidates.into_iter().flatten() {
        let timestamp = dir
            .join("packages")
            .join("hackage.haskell.org")
            .join("01-index.timestamp");
        if let Ok(age) = std::fs::metadata(&timestamp)
            .and_then(|m| m.modified())
            .and_then(|t| {
                t.elapsed()
                    .map_err(|e| std::io::Error::other(e.to_string()))
            })
        {
            return CabalIndexStatus::Present { age };
        }
    }

    CabalIndexStatus::Missing
}

/// Get the global config directory.
///
/// Respects `$XDG_CONFIG_HOME` if set to a non-empty value, on any platform.
/// Otherwise falls back to the platform default:
/// - Linux: `~/.config/hx`
/// - macOS: `~/Library/Application Support/hx`
/// - Windows: `%APPDATA%\hx\config`
pub fn global_config_dir() -> Result<PathBuf> {
    if let Some(dir) = xdg_dir(std::env::var("XDG_CONFIG_HOME").ok(), "hx") {
        return Ok(dir);
    }
    let dirs = ProjectDirs::from("io", "raskell", "hx")
        .ok_or_else(|| Error::config("could not determine home directory for config"))?;
    Ok(dirs.config_dir().to_path_buf())
}

fn xdg_dir(env_value: Option<String>, subdir: &str) -> Option<PathBuf> {
    env_value
        .filter(|v| !v.is_empty())
        .map(|v| PathBuf::from(v).join(subdir))
}

/// Get the global config file path.
///
/// - Linux: `~/.config/hx/config.toml`
/// - macOS: `~/Library/Application Support/hx/config.toml`
/// - Windows: `%APPDATA%\hx\config\config.toml`
pub fn global_config_file() -> Result<PathBuf> {
    Ok(global_config_dir()?.join("config.toml"))
}

/// Get the toolchain directory for hx-managed installations.
///
/// Uses `~/.hx/toolchains` on all platforms. This avoids paths with spaces
/// which cause issues with GHC's build system (especially on macOS where
/// the standard data directory contains "Application Support").
///
/// Structure:
/// ```text
/// ~/.hx/toolchains/
///   ghc/
///     9.8.2/bin/ghc, ghc-pkg, ...
///     9.6.4/...
///   cabal/
///     3.12.1.0/bin/cabal
///   downloads/
///     ghc-9.8.2-aarch64-apple-darwin.tar.xz
///   manifest.json
/// ```
pub fn toolchain_dir() -> Result<PathBuf> {
    let home = directories::BaseDirs::new()
        .ok_or_else(|| Error::config("could not determine home directory"))?;
    Ok(home.home_dir().join(".hx").join("toolchains"))
}

/// Get the bin directory for hx-managed tool symlinks.
///
/// Respects `$XDG_BIN_HOME` if set to a non-empty value (used as-is — this is
/// a `PATH` directory of executables, not per-app data, so no `hx`
/// subdirectory is appended). Otherwise defaults to `~/.local/bin`, the de
/// facto standard for user-installed executables (systemd's
/// `file-hierarchy(7)`; what `pipx`, `uv tool install`, and `cargo install`
/// effectively use).
///
/// This directory can be added to `PATH` for direct access to hx-managed
/// tools; `hx doctor` reports when it isn't.
pub fn toolchain_bin_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("XDG_BIN_HOME")
        && !dir.is_empty()
    {
        return Ok(PathBuf::from(dir));
    }
    let home = directories::BaseDirs::new()
        .ok_or_else(|| Error::config("could not determine home directory"))?;
    Ok(home.home_dir().join(".local").join("bin"))
}

/// Get ghcup's own bin directory, if it has one installed.
///
/// hx doesn't manage this — it's ghcup's own directory, and where that is
/// depends on ghcup's own env vars (see
/// <https://www.haskell.org/ghcup/guide/config/#env-variables>):
/// - `GHCUP_USE_XDG_DIRS` set: ghcup uses `$XDG_BIN_HOME` (default
///   `~/.local/bin`) — the same resolution [`toolchain_bin_dir`] uses, so
///   ghcup and hx share a bin directory in this mode.
/// - Otherwise: `$GHCUP_INSTALL_BASE_PREFIX/.ghcup/bin` (base defaults to
///   `$HOME`). ghcup's official installer script is supposed to add this to
///   `PATH` itself, but package manager-installed ghcup (e.g. via Homebrew)
///   doesn't always.
pub fn ghcup_bin_dir() -> Option<PathBuf> {
    let dir = if std::env::var_os("GHCUP_USE_XDG_DIRS").is_some() {
        toolchain_bin_dir().ok()?
    } else {
        let base = std::env::var("GHCUP_INSTALL_BASE_PREFIX")
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| directories::BaseDirs::new().map(|d| d.home_dir().to_path_buf()))?;
        base.join(".ghcup").join("bin")
    };
    dir.exists().then_some(dir)
}

/// Ensure a directory exists.
///
/// On Unix, newly created cache directories are restricted to the owner
/// (0o700) so other local users cannot tamper with cached artifacts.
pub fn ensure_dir(path: &PathBuf) -> Result<()> {
    if !path.exists() {
        debug!("Creating directory: {}", path.display());
        std::fs::create_dir_all(path).map_err(|e| Error::Io {
            message: format!("failed to create directory: {}", path.display()),
            path: Some(path.clone()),
            source: e,
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).map_err(
                |e| Error::Io {
                    message: format!("failed to set permissions on: {}", path.display()),
                    path: Some(path.clone()),
                    source: e,
                },
            )?;
        }
    }
    Ok(())
}

/// Clean the global cache.
pub fn clean_global_cache() -> Result<()> {
    let cache_dir = global_cache_dir()?;
    if cache_dir.exists() {
        debug!("Removing global cache: {}", cache_dir.display());
        std::fs::remove_dir_all(&cache_dir).map_err(|e| Error::Io {
            message: "failed to remove global cache".to_string(),
            path: Some(cache_dir),
            source: e,
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn xdg_dir_appends_subdir_to_set_value() {
        assert_eq!(
            xdg_dir(Some("/custom/config".to_string()), "hx"),
            Some(PathBuf::from("/custom/config/hx"))
        );
    }

    #[test]
    fn xdg_dir_falls_back_when_unset() {
        assert_eq!(xdg_dir(None, "hx"), None);
    }

    #[test]
    fn xdg_dir_falls_back_when_empty() {
        assert_eq!(xdg_dir(Some(String::new()), "hx"), None);
    }

    // std::env::set_var is process-global and `cargo test` runs tests in
    // parallel by default, so any test that touches real XDG_* env vars
    // must serialize against the others via this lock.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn global_config_and_cache_dirs_use_distinct_xdg_vars() {
        let _guard = ENV_LOCK.lock().unwrap();

        // SAFETY: serialized by ENV_LOCK against other tests in this module.
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", "/tmp/hx-xdg-test/config");
            std::env::set_var("XDG_CACHE_HOME", "/tmp/hx-xdg-test/cache");
        }

        let config = global_config_dir().unwrap();
        let cache = global_cache_dir().unwrap();

        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
            std::env::remove_var("XDG_CACHE_HOME");
        }

        assert_ne!(
            config, cache,
            "config and cache dirs must not collapse to the same path"
        );
        assert_eq!(config, PathBuf::from("/tmp/hx-xdg-test/config/hx"));
        assert_eq!(cache, PathBuf::from("/tmp/hx-xdg-test/cache/hx"));
    }

    #[test]
    fn global_config_dir_ignores_xdg_cache_home() {
        let _guard = ENV_LOCK.lock().unwrap();

        // SAFETY: serialized by ENV_LOCK against other tests in this module.
        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
            std::env::set_var("XDG_CACHE_HOME", "/tmp/hx-xdg-test/cache-only");
        }

        let config = global_config_dir().unwrap();

        unsafe {
            std::env::remove_var("XDG_CACHE_HOME");
        }

        assert_ne!(
            config,
            PathBuf::from("/tmp/hx-xdg-test/cache-only/hx"),
            "XDG_CACHE_HOME must not leak into the config dir resolution"
        );
    }
}

/// Clean a project's local cache.
pub fn clean_project_cache(project_root: &Path) -> Result<()> {
    let cache_dir = project_root.join(".hx");
    if cache_dir.exists() {
        debug!("Removing project cache: {}", cache_dir.display());
        std::fs::remove_dir_all(&cache_dir).map_err(|e| Error::Io {
            message: "failed to remove project cache".to_string(),
            path: Some(cache_dir),
            source: e,
        })?;
    }
    Ok(())
}
