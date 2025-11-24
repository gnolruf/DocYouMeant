use std::path::PathBuf;

use ort::Error as OrtError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("ONNX Runtime error: {source}")]
    Ort {
        #[from]
        source: OrtError,
    },

    #[error("Failed to load data file: {path}")]
    DataFileLoadError {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse data file: {path}")]
    DataFileParseError {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to load model file: {path}")]
    ModelFileLoadError {
        path: PathBuf,
        #[source]
        source: OrtError,
    },

    #[error("Model execution failed: {operation}")]
    ModelExecutionError {
        operation: String,
        #[source]
        source: OrtError,
    },

    #[error("Image preprocessing failed: {operation}")]
    PreprocessingError { operation: String, message: String },

    #[error("Prediction processing failed: {operation}")]
    PredictionError { operation: String, message: String },

    #[error("Processing error: {message}")]
    ProcessingError { message: String },
}
