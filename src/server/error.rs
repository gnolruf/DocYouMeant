use axum::{
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use thiserror::Error;

use super::models::ErrorResponse;
use crate::document::DocumentError;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Document processing error")]
    Document {
        #[from]
        source: DocumentError,
    },
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message, details) = match self {
            AppError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "Bad Request".to_string(),
                Some(msg),
            ),
            AppError::Document { source } => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "Document Processing Error".to_string(),
                Some(source.to_string()),
            ),
        };

        let mut error_response = ErrorResponse::new(error_message);
        if let Some(details) = details {
            error_response = error_response.with_details(details);
        }

        (status, Json(error_response)).into_response()
    }
}
