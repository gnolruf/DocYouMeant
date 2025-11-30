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

pub fn create_app() -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/api/v1/analyze", post(handlers::analyze_document))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}

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
