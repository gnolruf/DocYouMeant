//! Application configuration module.
//!
//! This module provides configuration management for the DocYouMeant application.
//! Configuration is loaded from a JSON file.

use super::error::ConfigError;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Default configuration file path
pub const DEFAULT_CONFIG_PATH: &str = "config/app_config.json";

/// Global configuration instance
static CONFIG_INSTANCE: OnceCell<AppConfig> = OnceCell::new();

/// Application configuration structure.
///
/// This struct represents the application's configuration settings
/// that are loaded from a JSON configuration file. String fields use
/// `Box<str>` for memory efficiency since they are set once and never modified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Maximum allowed file size in bytes
    pub max_file_size: u64,

    /// Directory path for model files
    pub model_directory: Box<str>,

    /// Host URL for the server
    pub host_url: Box<str>,

    /// Model set to use
    pub model_set: Option<Box<str>>,
}

impl AppConfig {
    /// Load configuration from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the configuration JSON file
    ///
    /// # Returns
    ///
    /// Returns the parsed `AppConfig` or a `ConfigError` if loading fails.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path)?;
        let config: AppConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from the default path.
    ///
    /// This loads configuration from `config/app_config.json`.
    ///
    /// # Returns
    ///
    /// Returns the parsed `AppConfig` or a `ConfigError` if loading fails.
    pub fn load_default() -> Result<Self, ConfigError> {
        Self::from_file(DEFAULT_CONFIG_PATH)
    }

    /// Initialize the global configuration instance.
    ///
    /// This should be called once at application startup. If not called,
    /// `get()` will initialize with default values.
    ///
    /// # Returns
    ///
    /// Returns a reference to the initialized configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The config file cannot be loaded
    /// - `model_set` is not specified in the config
    pub fn init() -> Result<&'static Self, ConfigError> {
        CONFIG_INSTANCE.get_or_try_init(|| {
            let config = Self::load_default()?;

            if config.model_set.is_none() {
                return Err(ConfigError::MissingModelSet);
            }

            Ok(config)
        })
    }

    /// Get the global configuration instance.
    ///
    /// If the configuration hasn't been initialized, returns default values.
    ///
    /// # Returns
    ///
    /// Returns a reference to the global configuration.
    #[must_use]
    pub fn get() -> &'static Self {
        CONFIG_INSTANCE.get_or_init(Self::default)
    }

    /// Create a new configuration with default values.
    ///
    /// # Returns
    ///
    /// Returns an `AppConfig` with sensible default values.
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            max_file_size: 1024 * 1024 * 1024, // 1 GB
            model_directory: "models".into(),
            host_url: "0.0.0.0:3000".into(),
            model_set: Some("edge".into()),
        }
    }

    /// Get the active model set name.
    ///
    /// # Panics
    ///
    /// Panics if called before `init()` or if no model set was configured.
    #[must_use]
    pub fn model_set(&self) -> &str {
        self.model_set
            .as_ref()
            .expect("model_set not initialized - call init() first")
    }

    /// Get the path to a model file within the active model set directory.
    ///
    /// # Arguments
    ///
    /// * `relative_path` - The relative path to the model file (e.g., "onnx/text_detection.onnx")
    ///
    /// # Returns
    ///
    /// Returns the full path to a model file
    #[must_use]
    pub fn model_path(&self, relative_path: &str) -> String {
        format!(
            "{}/{}/{}",
            self.model_directory,
            self.model_set(),
            relative_path
        )
    }

    /// Get the path to a shared file in the model directory (not model-set specific).
    ///
    /// Used for files shared across model sets, like dictionary files.
    ///
    /// # Arguments
    ///
    /// * `relative_path` - The relative path to the file (e.g., "dict/en_dict.txt")
    ///
    /// # Returns
    ///
    /// Returns the full path: `{model_directory}/{relative_path}`
    #[must_use]
    pub fn shared_path(&self, relative_path: &str) -> String {
        format!("{}/{}", self.model_directory, relative_path)
    }

    /// Get the TensorRT cache directory for the active model set.
    ///
    /// # Returns
    ///
    /// Returns the path: `{model_directory}/{model_set}/trt_engines`
    #[must_use]
    pub fn rt_cache_directory(&self) -> String {
        format!("{}/{}/trt_engines", self.model_directory, self.model_set())
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_config_from_json() {
        let json = r#"{
            "max_file_size": 52428800,
            "model_directory": "models",
            "host_url": "127.0.0.1:8080",
            "model_set": "edge"
        }"#;

        let config: AppConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.max_file_size, 52428800);
        assert_eq!(&*config.model_directory, "models");
        assert_eq!(&*config.host_url, "127.0.0.1:8080");
        assert_eq!(config.model_set.as_deref(), Some("edge"));
    }

    #[test]
    fn test_load_config_from_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let json = r#"{
            "max_file_size": 104857600,
            "model_directory": "models",
            "host_url": "0.0.0.0:3000",
            "model_set": "server"
        }"#;
        temp_file.write_all(json.as_bytes()).unwrap();

        let config = AppConfig::from_file(temp_file.path()).unwrap();

        assert_eq!(config.max_file_size, 104857600);
        assert_eq!(&*config.model_directory, "models");
        assert_eq!(&*config.host_url, "0.0.0.0:3000");
        assert_eq!(config.model_set.as_deref(), Some("server"));
    }

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();

        assert_eq!(config.max_file_size, 1024 * 1024 * 1024);
        assert_eq!(&*config.model_directory, "models");
        assert_eq!(&*config.host_url, "0.0.0.0:3000");
        assert_eq!(config.model_set.as_deref(), Some("edge"));
    }

    #[test]
    fn test_config_without_model_set() {
        let json = r#"{
            "max_file_size": 52428800,
            "model_directory": "models",
            "host_url": "127.0.0.1:8080"
        }"#;

        let config: AppConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.max_file_size, 52428800);
        assert_eq!(config.model_set, None);
    }

    #[test]
    fn test_serialize_config() {
        let json = r#"{
            "max_file_size": 1000,
            "model_directory": "test/models",
            "host_url": "localhost:9000",
            "model_set": "custom"
        }"#;

        let config: AppConfig = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&config).unwrap();
        let parsed: AppConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.max_file_size, parsed.max_file_size);
        assert_eq!(config.model_directory, parsed.model_directory);
        assert_eq!(config.host_url, parsed.host_url);
        assert_eq!(config.model_set, parsed.model_set);
    }
}
