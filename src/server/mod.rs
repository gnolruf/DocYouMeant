//! HTTP server for the document analysis API.
//!
//! This module provides an HTTP server built with [Axum](https://docs.rs/axum) that exposes
//! document analysis capabilities via a REST API. The server handles document uploads,
//! performs OCR and layout analysis, and returns structured results.

pub mod error;
pub mod handlers;
pub mod models;

use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::inference::crnn::Crnn;
use crate::inference::dbnet::DBNet;
use crate::inference::lcnet::{LCNet, LCNetMode};
use crate::inference::rtdetr::{RtDetr, RtDetrMode};
use crate::inference::tasks::question_and_answer_task::QuestionAndAnswerTask;

/// Creates and configures the Axum router with all API routes.
///
/// This function sets up the complete routing configuration including:
/// - Health check endpoint at `/health`
/// - Document analysis endpoint at `/api/v1/analyze`
/// - CORS middleware (permissive configuration)
/// - HTTP request tracing middleware
///
/// # Returns
///
/// A configured [`Router`] ready to be served.
pub fn create_app() -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/api/v1/analyze", post(handlers::analyze_document))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}

/// Preloads all machine learning models into memory.
///
/// This function initializes and caches all inference models used by the
/// document analysis pipeline. Calling this at server startup ensures that
/// the first request doesn't incur model loading latency.
///
/// # Arguments
///
/// * `ocr_language` - Optional language code for the OCR model to preload.
pub async fn initialize_models(
    ocr_language: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Preloading models...");

    tracing::info!("  Loading DBNet (text detection)...");
    DBNet::get_or_init()?;

    tracing::info!("  Loading LCNet (document orientation)...");
    LCNet::get_or_init(LCNetMode::DocumentOrientation)?;

    tracing::info!("  Loading LCNet (text orientation)...");
    LCNet::get_or_init(LCNetMode::TextOrientation)?;

    tracing::info!("  Loading LCNet (table type classification)...");
    LCNet::get_or_init(LCNetMode::TableType)?;

    tracing::info!("  Loading RtDetr (layout detection)...");
    RtDetr::get_or_init(RtDetrMode::Layout)?;

    tracing::info!("  Loading RtDetr (wired table cell detection)...");
    RtDetr::get_or_init(RtDetrMode::WiredTableCell)?;

    tracing::info!("  Loading RtDetr (wireless table cell detection)...");
    RtDetr::get_or_init(RtDetrMode::WirelessTableCell)?;

    tracing::info!("  Loading Phi4Mini (language model)...");
    QuestionAndAnswerTask::get_or_init()?;

    if let Some(language) = ocr_language {
        tracing::info!("  Loading Crnn ({} text recognition)...", language);
        let _ = Crnn::new(language)?;
    }

    tracing::info!("All models preloaded successfully.");

    Ok(())
}

/// Starts the HTTP server and listens for incoming requests.
///
/// This function binds to the specified address and begins serving the
/// document analysis API. It blocks until the server is shut down.
///
/// # Arguments
///
/// * `addr` - The socket address to bind to (e.g., `0.0.0.0:3000`).
///
/// # Errors
///
/// Returns an error if:
/// - The address is already in use
/// - The address cannot be bound (e.g., permission denied for privileged ports)
/// - The server encounters a fatal error during operation
pub async fn start_server(addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Starting server on {}", addr);

    let app = create_app();
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!("Server listening on http://{}", addr);
    tracing::info!("API endpoint: http://{}/api/v1/analyze", addr);
    tracing::info!("Health check: http://{}/health", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
