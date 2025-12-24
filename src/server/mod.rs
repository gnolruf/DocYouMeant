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
