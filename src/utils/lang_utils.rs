use std::collections::HashMap;
use std::fs;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageConfig {
    pub name: String,
    pub model_file: String,
    pub dict_file: String,
}

pub struct LangUtils;

impl LangUtils {
    pub fn get_language_config(language: &str) -> Option<LanguageConfig> {
        match Self::get_all_language_configs() {
            Ok(configs) => configs.get(language).cloned(),
            Err(_) => {
                if language == "english" {
                    Some(LanguageConfig {
                        name: "english".to_string(),
                        model_file: "models/onnx/text_recognition_en.onnx".to_string(),
                        dict_file: "models/dict/en_dict.txt".to_string(),
                    })
                } else {
                    None
                }
            }
        }
    }

    pub fn get_all_language_configs(
    ) -> Result<HashMap<String, LanguageConfig>, Box<dyn std::error::Error>> {
        let config_path = "models/lang_models.json";
        let config_content = fs::read_to_string(config_path)?;
        let configs: HashMap<String, LanguageConfig> = serde_json::from_str(&config_content)?;
        Ok(configs)
    }
}
