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

    /// Default model set to use if not specified via CLI
    pub default_model_set: Option<Box<str>>,

    /// Active model set (set at runtime, not serialized)
    #[serde(skip)]
    model_set: Option<Box<str>>,
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
    #[must_use]
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
    #[must_use]
    pub fn load_default() -> Result<Self, ConfigError> {
        Self::from_file(DEFAULT_CONFIG_PATH)
    }

    /// Initialize the global configuration instance with an optional model set override.
    ///
    /// This should be called once at application startup. If not called,
    /// `get()` will initialize with default values.
    ///
    /// # Arguments
    ///
    /// * `model_set` - Optional model set name from CLI. If None, uses default_model_set from config.
    ///
    /// # Returns
    ///
    /// Returns a reference to the initialized configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The config file cannot be loaded
    /// - Neither model_set argument nor default_model_set is provided
    pub fn init(model_set: Option<String>) -> Result<&'static Self, ConfigError> {
        CONFIG_INSTANCE.get_or_try_init(|| {
            let mut config = Self::load_default()?;

            let effective_model_set = model_set
                .or_else(|| config.default_model_set.as_ref().map(|s| s.to_string()))
                .ok_or(ConfigError::MissingModelSet)?;

            config.model_set = Some(effective_model_set.into());
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
            default_model_set: Some("edge".into()),
            model_set: None,
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
