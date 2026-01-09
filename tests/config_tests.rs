use docyoumeant::utils::config::AppConfig;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_parse_config_from_json() {
    let json = r#"{
        "max_file_size": 52428800,
        "model_directory": "models",
        "host_url": "127.0.0.1:8080",
        "model_set": "edge"
    }"#;

    let config: AppConfig = serde_json::from_str(json).unwrap();

    assert_eq!(config.max_file_size, 52428800);
    assert_eq!(&*config.model_directory, "models");
    assert_eq!(&*config.host_url, "127.0.0.1:8080");
    assert_eq!(config.model_set.as_deref(), Some("edge"));
}

#[test]
fn test_load_config_from_file() {
    let mut temp_file = NamedTempFile::new().unwrap();
    let json = r#"{
        "max_file_size": 104857600,
        "model_directory": "models",
        "host_url": "0.0.0.0:3000",
        "model_set": "server"
    }"#;
    temp_file.write_all(json.as_bytes()).unwrap();

    let config = AppConfig::from_file(temp_file.path()).unwrap();

    assert_eq!(config.max_file_size, 104857600);
    assert_eq!(&*config.model_directory, "models");
    assert_eq!(&*config.host_url, "0.0.0.0:3000");
    assert_eq!(config.model_set.as_deref(), Some("server"));
}

#[test]
fn test_default_config() {
    let config = AppConfig::default();

    assert_eq!(config.max_file_size, 1024 * 1024 * 1024);
    assert_eq!(&*config.model_directory, "models");
    assert_eq!(&*config.host_url, "0.0.0.0:3000");
    assert_eq!(config.model_set.as_deref(), Some("edge"));
}

#[test]
fn test_config_without_model_set() {
    let json = r#"{
        "max_file_size": 52428800,
        "model_directory": "models",
        "host_url": "127.0.0.1:8080"
    }"#;

    let config: AppConfig = serde_json::from_str(json).unwrap();

    assert_eq!(config.max_file_size, 52428800);
    assert_eq!(config.model_set, None);
}

#[test]
fn test_serialize_config() {
    let json = r#"{
        "max_file_size": 1000,
        "model_directory": "test/models",
        "host_url": "localhost:9000",
        "model_set": "custom"
    }"#;

    let config: AppConfig = serde_json::from_str(json).unwrap();
    let serialized = serde_json::to_string(&config).unwrap();
    let parsed: AppConfig = serde_json::from_str(&serialized).unwrap();

    assert_eq!(config.max_file_size, parsed.max_file_size);
    assert_eq!(config.model_directory, parsed.model_directory);
    assert_eq!(config.host_url, parsed.host_url);
    assert_eq!(config.model_set, parsed.model_set);
}
