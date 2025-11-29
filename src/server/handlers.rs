use axum::response::Json;

use super::error::AppError;
use super::models::{AnalyzeRequest, AnalyzeResponse, HealthResponse};
use crate::document::Document;

/// Health check endpoint
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse::ok())
}

/// Main document analysis endpoint
pub async fn analyze_document(
    Json(request): Json<AnalyzeRequest>,
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
