use crate::document::AnalysisResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeRequest {
    /// Base64-encoded document data
    pub data: String,

    /// Filename with extension (e.g., "document.pdf", "image.jpg")
    /// The extension is used to determine the document type
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
