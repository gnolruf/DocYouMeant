use docyoumeant::utils::config::AppConfig;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_parse_config_from_json() {
    let json = r#"{
        "max_file_size": 52428800,
        "rt_cache_directory": "cache/trt",
        "model_directory": "models",
        "host_url": "127.0.0.1:8080"
    }"#;

    let config: AppConfig = serde_json::from_str(json).unwrap();

    assert_eq!(config.max_file_size, 52428800);
    assert_eq!(config.rt_cache_directory, "cache/trt");
    assert_eq!(config.model_directory, "models");
    assert_eq!(config.host_url, "127.0.0.1:8080");
}

#[test]
fn test_load_config_from_file() {
    let mut temp_file = NamedTempFile::new().unwrap();
    let json = r#"{
        "max_file_size": 104857600,
        "rt_cache_directory": "models/trt_engines",
        "model_directory": "models",
        "host_url": "0.0.0.0:3000"
    }"#;
    temp_file.write_all(json.as_bytes()).unwrap();

    let config = AppConfig::from_file(temp_file.path()).unwrap();

    assert_eq!(config.max_file_size, 104857600);
    assert_eq!(config.rt_cache_directory, "models/trt_engines");
    assert_eq!(config.model_directory, "models");
    assert_eq!(config.host_url, "0.0.0.0:3000");
}

#[test]
fn test_default_config() {
    let config = AppConfig::default();

    assert_eq!(config.max_file_size, 1024 * 1024 * 1024);
    assert_eq!(config.rt_cache_directory, "models/trt_engines");
    assert_eq!(config.model_directory, "models");
    assert_eq!(config.host_url, "0.0.0.0:3000");
}

#[test]
fn test_serialize_config() {
    let config = AppConfig {
        max_file_size: 1000,
        rt_cache_directory: "test/cache".to_string(),
        model_directory: "test/models".to_string(),
        host_url: "localhost:9000".to_string(),
    };

    let json = serde_json::to_string(&config).unwrap();
    let parsed: AppConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.max_file_size, parsed.max_file_size);
    assert_eq!(config.rt_cache_directory, parsed.rt_cache_directory);
    assert_eq!(config.model_directory, parsed.model_directory);
    assert_eq!(config.host_url, parsed.host_url);
}
