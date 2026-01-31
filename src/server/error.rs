use axum::{
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use thiserror::Error;

use super::models::ErrorResponse;
use crate::document::DocumentError;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ValidationError {
    #[error("File size exceeds maximum allowed size")]
    FileSizeTooLarge,

    #[error("Base64 data length exceeds maximum allowed size")]
    Base64DataTooLarge,

    #[error("Invalid base64 encoding: {0}")]
    InvalidBase64(String),

    #[error("Filename is empty")]
    EmptyFilename,

    #[error("Filename contains forbidden character: '{0}'")]
    ForbiddenCharacter(char),

    #[error("Filename is too long (max 255 characters)")]
    FilenameTooLong,

    #[error("Filename must have an extension")]
    MissingExtension,

    #[error("Filename cannot start with a period or space, or end with a space")]
    InvalidFilenameEdges,

    #[error("Filename contains path traversal sequence")]
    PathTraversal,
}

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("Document processing error")]
    Document {
        #[from]
        source: DocumentError,
    },

    #[error("Internal server error: {message}")]
    Internal { message: String },
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message, details) = match self {
            AppError::Validation(err) => (
                StatusCode::BAD_REQUEST,
                "Validation Error".to_string(),
                Some(err.to_string()),
            ),
            AppError::Document { source } => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "Document Processing Error".to_string(),
                Some(source.to_string()),
            ),
            AppError::Internal { message } => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal Server Error".to_string(),
                Some(message),
            ),
        };

        let mut error_response = ErrorResponse::new(error_message);
        if let Some(details) = details {
            error_response = error_response.with_details(details);
        }

        (status, Json(error_response)).into_response()
    }
}
