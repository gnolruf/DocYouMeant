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

use crate::inference::dbnet::DBNet;
use crate::inference::lcnet::LCNet;
use crate::inference::rtdetr::RtDetr;
use crate::inference::tasks::key_value_pair_extraction_task::KeyValuePairExtractionTask;
use crate::inference::tasks::question_and_answer_task::QuestionAndAnswerTask;

pub fn create_app() -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/api/v1/analyze", post(handlers::analyze_document))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}

pub async fn initialize_models() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Preloading models...");

    tracing::info!("  Loading DBNet (text detection)...");
    DBNet::get_or_init()?;

    tracing::info!("  Loading LCNet (document orientation)...");
    LCNet::get_or_init(false)?;

    tracing::info!("  Loading LCNet (text orientation)...");
    LCNet::get_or_init(true)?;

    tracing::info!("  Loading RtDetr (layout detection)...");
    RtDetr::get_or_init()?;

    tracing::info!("  Loading Phi4Mini (language model)...");
    KeyValuePairExtractionTask::get_or_init()?;
    QuestionAndAnswerTask::get_or_init()?;

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
