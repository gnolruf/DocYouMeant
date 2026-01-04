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

    /// Directory path for TensorRT engine cache files
    pub rt_cache_directory: Box<str>,

    /// Directory path for model files
    pub model_directory: Box<str>,

    /// Host URL for the server
    pub host_url: Box<str>,
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

    /// Initialize the global configuration instance.
    ///
    /// This should be called once at application startup. If not called,
    /// `get()` will initialize with default values.
    ///
    /// # Returns
    ///
    /// Returns a reference to the initialized configuration.
    pub fn init() -> Result<&'static Self, ConfigError> {
        CONFIG_INSTANCE.get_or_try_init(Self::load_default)
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
            rt_cache_directory: "models/trt_engines".into(),
            model_directory: "models".into(),
            host_url: "0.0.0.0:3000".into(),
        }
    }

    /// Get the path to a model file within the model directory.
    ///
    /// # Arguments
    ///
    /// * `relative_path` - The relative path to the model file (e.g., "onnx/text_detection.onnx")
    ///
    /// # Returns
    ///
    /// Returns the full path to the model file.
    #[must_use]
    pub fn model_path(&self, relative_path: &str) -> String {
        format!("{}/{}", self.model_directory, relative_path)
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self::default_config()
    }
}
