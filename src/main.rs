use std::env;

mod document;
mod inference;
mod server;
mod utils;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_ort()?;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "docyoumeant=info,tower_http=debug".into()),
        )
        .init();

    run_server().await?;

    Ok(())
}

fn setup_ort() -> Result<(), Box<dyn std::error::Error>> {
    let dylib_path =
        env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "/usr/lib/libonnxruntime.so".to_string());

    ort::init_from(dylib_path)
        .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default()
            .with_device_id(0)
            .build()])
        .commit()?;

    Ok(())
}

async fn run_server() -> Result<(), Box<dyn std::error::Error>> {
    let addr = std::env::var("DOCYOUMEANT_ADDR").unwrap_or_else(|_| "127.0.0.1:3000".to_string());

    let socket_addr: std::net::SocketAddr = addr.parse()?;

    server::initialize_models().await?;

    server::start_server(socket_addr).await?;

    Ok(())
}
