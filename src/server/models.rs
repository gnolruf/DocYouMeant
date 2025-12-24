//! Request and response models for the document analysis server API.
//!
//! This module defines the data structures used for communication between
//! clients and the document analysis server, including request validation
//! and response formatting.

use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};

use super::error::ValidationError;
use crate::document::AnalysisResult;

/// Maximum allowed file size in bytes (1 GB).
///
/// This limit prevents memory exhaustion from excessively large uploads.
const MAX_FILE_SIZE_BYTES: usize = 1024 * 1024 * 1024;

/// Maximum allowed length of Base64-encoded data.
///
/// Base64 encoding expands data by approximately 4/3, so this value is
/// calculated from [`MAX_FILE_SIZE_BYTES`] to ensure the decoded data
/// fits within the file size limit.
const MAX_BASE64_LENGTH: usize = (MAX_FILE_SIZE_BYTES / 3 + 1) * 4;

/// Characters that are forbidden in filenames.
///
/// These characters are rejected during filename validation to prevent
/// path traversal attacks and filesystem issues:
/// - `/` - Directory separator, could allow path traversal
/// - `\0` - Null byte, can cause string truncation issues
const FORBIDDEN_FILENAME_CHARS: &[char] = &['/', '\0'];

/// Request payload for document analysis.
///
/// This struct represents the JSON body expected by the `/analyze` endpoint.
/// It contains the document data, metadata, and optional analysis parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisRequest {
    /// Base64-encoded document data.
    ///
    /// The document (PDF, image, etc.) must be encoded using standard Base64.
    /// Maximum decoded size is 1 GB.
    pub data: String,

    /// Filename with extension (e.g., `"document.pdf"`).
    ///
    /// The extension is used to determine the document type for processing.
    /// Must be 1-255 characters and cannot contain `/` or null bytes.
    pub filename: String,

    /// Optional list of questions to answer about the document.
    ///
    /// When provided, the analysis will attempt to extract answers to these
    /// specific questions from the document content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub questions: Option<Vec<String>>,

    /// The process ID specifying the analysis pipeline to use.
    ///
    /// Common values:
    /// - `"general"` - Full document analysis (default)
    /// - `"read"` - Text extraction only
    ///
    /// Defaults to `"general"` if not specified.
    #[serde(default = "default_process_id")]
    pub process_id: String,

    /// The OCR language model to use for text recognition.
    ///
    /// Supported values include `"english"`, `"chinese"`, `"arabic"`, etc.
    /// If not provided, the language will be automatically detected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// Returns the default process ID for analysis requests.
///
/// This function is used by serde to provide a default value when
/// `process_id` is not specified in the JSON request.
fn default_process_id() -> String {
    "general".to_string()
}

impl AnalysisRequest {
    /// Validates the request and decodes the Base64 document data.
    ///
    /// This method performs all validation checks and returns the decoded
    /// document bytes if successful.
    ///
    /// # Errors
    ///
    /// Returns a [`ValidationError`] if:
    /// - The filename is invalid (empty, too long, missing extension, etc.)
    /// - The Base64 data is invalid or too large
    /// - The decoded file exceeds the size limit
    pub fn validate_and_decode(&self) -> Result<Vec<u8>, ValidationError> {
        self.validate_filename()?;
        self.validate_and_decode_base64()
    }

    /// Validates and decodes the Base64-encoded document data.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::Base64DataTooLarge`] if the encoded data exceeds the limit.
    /// Returns [`ValidationError::InvalidBase64`] if the data is not valid Base64.
    /// Returns [`ValidationError::FileSizeTooLarge`] if the decoded data exceeds 1 GB.
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

    /// Validates the filename for security and filesystem compatibility.
    ///
    /// # Validation Rules
    ///
    /// - Must not be empty or whitespace-only
    /// - Must not exceed 255 characters
    /// - Must not contain `/` or null bytes
    /// - Must have an extension (contain `.` but not end with it)
    /// - Must not start with `.` or space, or end with space
    ///
    /// # Errors
    ///
    /// Returns the appropriate [`ValidationError`] variant for each validation failure.
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

    /// Returns the filename with leading and trailing whitespace removed.
    ///
    /// Use this method to get a clean filename for logging or storage purposes
    /// after validation has been performed.
    pub fn sanitized_filename(&self) -> String {
        self.filename.trim().to_string()
    }
}

/// Successful response from the document analysis endpoint.
///
/// This struct wraps the analysis result with a status indicator for
/// consistent API response formatting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeResponse {
    /// Status indicator, always `"success"` for this response type.
    pub status: String,

    /// The document analysis result containing extracted text, tables, and layout.
    pub result: AnalysisResult,
}

impl AnalyzeResponse {
    /// Creates a new successful analysis response.
    ///
    /// # Arguments
    ///
    /// * `result` - The [`AnalysisResult`] from document processing.
    pub fn success(result: AnalysisResult) -> Self {
        Self {
            status: "success".to_string(),
            result,
        }
    }
}

/// Error response returned when document analysis fails.
///
/// This struct provides a consistent error format across all API endpoints,
/// with an optional details field for additional context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Status indicator, always `"error"` for this response type.
    pub status: String,

    /// A brief description of the error that occurred.
    pub error: String,

    /// Optional additional details about the error.
    ///
    /// This field may contain more specific information about what went wrong,
    /// such as validation error details or internal error messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl ErrorResponse {
    /// Creates a new error response with the given error message.
    ///
    /// # Arguments
    ///
    /// * `error` - The error message to include in the response.
    #[must_use]
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            status: "error".to_string(),
            error: error.into(),
            details: None,
        }
    }

    /// Adds additional details to the error response.
    ///
    /// This method follows the builder pattern, allowing chained calls.
    ///
    /// # Arguments
    ///
    /// * `details` - Additional context or details about the error.
    #[must_use]
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

/// Response from the health check endpoint.
///
/// This struct is returned by the `/health` endpoint to indicate that
/// the server is running and responsive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Health status indicator, `"ok"` when the server is healthy.
    pub status: String,

    /// The server version from `Cargo.toml`.
    pub version: String,
}

impl HealthResponse {
    /// Creates a healthy status response with the current package version.
    ///
    /// The version is automatically read from `Cargo.toml` at compile time.
    pub fn ok() -> Self {
        Self {
            status: "ok".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}
