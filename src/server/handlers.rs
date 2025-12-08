//! HTTP request handlers for the document analysis API.
//!
//! This module contains the handler functions that process incoming HTTP requests
//! and produce responses. Each handler corresponds to an API endpoint defined in
//! the router.
//!
//! # Handlers
//!
//! | Handler | Endpoint | Description |
//! |---------|----------|-------------|
//! | [`health`] | `GET /health` | Server health check |
//! | [`analyze_document`] | `POST /api/v1/analyze` | Document analysis |

use axum::response::Json;

use super::error::AppError;
use super::models::{AnalysisRequest, AnalyzeResponse, HealthResponse};
use crate::document::Document;

/// Health check endpoint handler.
///
/// Returns the server status and version information. This endpoint is useful
/// for load balancers, orchestration systems, and monitoring tools to verify
/// that the server is running and responsive.
///
/// # Endpoint
///
/// `GET /health`
///
/// # Response
///
/// Returns a [`HealthResponse`] with status `"ok"` and the current server version.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse::ok())
}

/// Main document analysis endpoint handler.
///
/// Processes uploaded documents through the OCR and layout analysis pipeline,
/// extracting text, tables, and structural information.
///
/// # Endpoint
///
/// `POST /api/v1/analyze`
///
/// # Request Body
///
/// Expects an [`AnalysisRequest`] JSON payload.
///
/// # Response
///
/// On success, returns an [`AnalyzeResponse`] containing the analysis results for the document.
///
/// # Errors
///
/// Returns an [`AppError`] (mapped to appropriate HTTP status codes) when:
/// - Request validation fails (400 Bad Request)
/// - Document format is unsupported (400 Bad Request)
/// - Document processing fails (500 Internal Server Error)
///
/// # Processing Steps
///
/// 1. Validate the request
/// 2. Decode the Base64 document data
/// 3. Load the document
/// 4. Run the analysis pipeline
/// 5. Return structured results
pub async fn analyze_document(
    Json(request): Json<AnalysisRequest>,
) -> Result<Json<AnalyzeResponse>, AppError> {
    tracing::info!(
        "Received analysis request for filename: {}",
        request.filename
    );

    let document_bytes = request.validate_and_decode()?;
    let filename = request.sanitized_filename();

    let mut document = Document::new(&document_bytes, &filename)?;

    tracing::info!("Document loaded with type: {:?}", document.doc_type());

    let questions = request.questions.as_deref();
    document.analyze(questions, &request.process_id, request.language.as_deref())?;

    tracing::info!("Document analysis completed successfully");

    let result = document.to_analyze_result()?;

    Ok(Json(AnalyzeResponse::success(result)))
}
