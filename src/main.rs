use clap::Parser;
use docyoumeant::inference::crnn::Crnn;
use docyoumeant::inference::dbnet::DBNet;
use docyoumeant::inference::lcnet::{LCNet, LCNetMode};
use docyoumeant::inference::rtdetr::{RtDetr, RtDetrMode};
use docyoumeant::inference::tasks::question_and_answer_task::QuestionAndAnswerTask;
use docyoumeant::server;
use docyoumeant::utils::config::AppConfig;
use docyoumeant::utils::lang_utils::LangUtils;
use lingua::Language;
use std::env;

#[derive(Parser, Debug)]
#[command(name = "docyoumeant")]
#[command(about = "A configurable document understanding pipeline server")]
struct Args {
    #[arg(long, short = 'l')]
    language: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let _config = AppConfig::init()?;

    setup_ort()?;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "docyoumeant=info,tower_http=debug".into()),
        )
        .init();

    // Parse language from command line argument (None if not provided)
    let language = args
        .language
        .as_ref()
        .and_then(|s| LangUtils::parse_language(s));

    run_server(language).await?;

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

async fn run_server(language: Option<Language>) -> Result<(), Box<dyn std::error::Error>> {
    let config = AppConfig::get();
    let addr = &config.host_url;

    let socket_addr: std::net::SocketAddr = addr.parse()?;

    initialize_models(language).await?;

    server::start_server(socket_addr, language).await?;

    Ok(())
}

/// Preloads all machine learning models into memory.
///
/// This function initializes and caches all inference models used by the
/// document analysis pipeline. Calling this at server startup ensures that
/// the first request doesn't incur model loading latency.
///
/// # Arguments
///
/// * `ocr_language` - Optional `Language` enum for the OCR model to preload.
async fn initialize_models(
    ocr_language: Option<Language>,
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
        let lang_str = LangUtils::map_from_lingua_language(language);
        tracing::info!("  Loading Crnn ({} text recognition)...", lang_str);

        let model_info = LangUtils::get_language_model_info(language)
            .ok_or_else(|| format!("Unsupported language: {}", lang_str))?;

        Crnn::get_or_init(model_info.model_file)?;
    } else {
        tracing::info!("  Loading CRNN models (OCR)...");
        let model_groups = LangUtils::get_model_groups(false)?;
        for (model_file, _) in model_groups {
            Crnn::get_or_init(model_file)?;
        }
    }

    tracing::info!("All models preloaded successfully.");

    Ok(())
}
