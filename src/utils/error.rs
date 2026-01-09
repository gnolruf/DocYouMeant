use thiserror::Error;

#[derive(Error, Debug)]
pub enum ImageError {
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Failed to create perspective transformation")]
    ProjectionFailed,
}

#[derive(Error, Debug)]
pub enum BoxError {
    #[error("Polygon offset operation failed")]
    OffsetFailed,
}

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read configuration file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse configuration JSON: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Model set not specified. Provide --model-set CLI argument or set default_model_set in config")]
    MissingModelSet,
}
