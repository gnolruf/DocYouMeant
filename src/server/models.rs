use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};

use super::error::ValidationError;
use crate::document::AnalysisResult;

/// Maximum allowed file size in bytes (1 GB)
const MAX_FILE_SIZE_BYTES: usize = 1024 * 1024 * 1024;

/// Base64 encoding expands data by ~4/3, so we calculate the max encoded length
const MAX_BASE64_LENGTH: usize = (MAX_FILE_SIZE_BYTES / 3 + 1) * 4;

const FORBIDDEN_FILENAME_CHARS: &[char] = &['/', '\0'];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeRequest {
    /// Base64-encoded document data
    pub data: String,

    /// Filename with extension
    pub filename: String,

    /// Optional list of questions to answer about the document
    #[serde(skip_serializing_if = "Option::is_none")]
    pub questions: Option<Vec<String>>,

    /// The process ID to use for analysis (e.g., "general", "read")
    #[serde(default = "default_process_id")]
    pub process_id: String,

    /// The OCR language model to use for text recognition (e.g., "english", "chinese")
    /// If not provided, language will be automatically detected
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

fn default_process_id() -> String {
    "general".to_string()
}

impl AnalyzeRequest {
    pub fn validate_and_decode(&self) -> Result<Vec<u8>, ValidationError> {
        self.validate_filename()?;
        self.validate_and_decode_base64()
    }

    fn validate_and_decode_base64(&self) -> Result<Vec<u8>, ValidationError> {
        if self.data.len() > MAX_BASE64_LENGTH {
            return Err(ValidationError::Base64DataTooLarge);
        }

        let decoded = STANDARD
            .decode(&self.data)
            .map_err(|e| ValidationError::InvalidBase64(e.to_string()))?;

        if decoded.len() > MAX_FILE_SIZE_BYTES {
            return Err(ValidationError::FileSizeTooLarge);
        }

        Ok(decoded)
    }

    fn validate_filename(&self) -> Result<(), ValidationError> {
        let filename = self.filename.trim();

        if filename.is_empty() {
            return Err(ValidationError::EmptyFilename);
        }

        if filename.len() > 255 {
            return Err(ValidationError::FilenameTooLong);
        }

        for ch in filename.chars() {
            if FORBIDDEN_FILENAME_CHARS.contains(&ch) {
                return Err(ValidationError::ForbiddenCharacter(ch));
            }
        }

        if !filename.contains('.') || filename.ends_with('.') {
            return Err(ValidationError::MissingExtension);
        }

        if filename.starts_with('.') || filename.starts_with(' ') || filename.ends_with(' ') {
            return Err(ValidationError::InvalidFilenameEdges);
        }

        Ok(())
    }

    pub fn sanitized_filename(&self) -> String {
        self.filename.trim().to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeResponse {
    pub status: String,
    pub result: AnalysisResult,
}

impl AnalyzeResponse {
    pub fn success(result: AnalysisResult) -> Self {
        Self {
            status: "success".to_string(),
            result,
        }
    }
}

/// Error response body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub status: String,
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl ErrorResponse {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            status: "error".to_string(),
            error: error.into(),
            details: None,
        }
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

impl HealthResponse {
    pub fn ok() -> Self {
        Self {
            status: "ok".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}
