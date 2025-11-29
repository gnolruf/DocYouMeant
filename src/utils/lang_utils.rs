use std::collections::HashMap;
use std::fs;

use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageConfig {
    pub name: String,
    pub model_file: String,
    pub dict_file: String,
    pub is_script_model: bool,
}

pub struct LangUtils;

impl LangUtils {
    pub fn get_language_config(language: &str) -> Option<LanguageConfig> {
        match Self::get_all_language_configs(false) {
            Ok(configs) => configs.get(language).cloned(),
            Err(_) => {
                if language == "english" {
                    Some(LanguageConfig {
                        name: "english".to_string(),
                        model_file: "models/onnx/text_recognition_en.onnx".to_string(),
                        dict_file: "models/dict/en_dict.txt".to_string(),
                        is_script_model: false,
                    })
                } else {
                    None
                }
            }
        }
    }

    pub fn get_all_language_configs(
        script_models_only: bool,
    ) -> Result<HashMap<String, LanguageConfig>, Box<dyn std::error::Error>> {
        let config_path = "models/ocr_lang_models.json";
        let config_content = fs::read_to_string(config_path)?;
        let configs: HashMap<String, LanguageConfig> = serde_json::from_str(&config_content)?;

        if script_models_only {
            Ok(configs
                .into_iter()
                .filter(|(_, config)| config.is_script_model)
                .collect())
        } else {
            Ok(configs)
        }
    }

    pub fn detect_language(texts: &[String], candidate_languages: &[String]) -> Option<String> {
        if texts.is_empty() || candidate_languages.is_empty() {
            return None;
        }

        let lingua_languages: Vec<Language> = candidate_languages
            .iter()
            .filter_map(|lang| Self::map_to_lingua_language(lang))
            .collect();

        if lingua_languages.is_empty() {
            return candidate_languages.first().cloned();
        }

        let detector: LanguageDetector = LanguageDetectorBuilder::from_languages(&lingua_languages)
            .with_preloaded_language_models()
            .build();

        let combined_text = texts.join(" ");

        detector
            .detect_language_of(&combined_text)
            .map(Self::map_from_lingua_language)
            .or_else(|| candidate_languages.first().cloned())
    }

    pub fn map_to_lingua_language(language: &str) -> Option<Language> {
        match language.to_lowercase().as_str() {
            // A
            "afrikaans" => Some(Language::Afrikaans),
            "albanian" => Some(Language::Albanian),
            "arabic" => Some(Language::Arabic),
            "armenian" => Some(Language::Armenian),
            "azerbaijani" => Some(Language::Azerbaijani),
            // B
            "basque" => Some(Language::Basque),
            "belarusian" => Some(Language::Belarusian),
            "bengali" => Some(Language::Bengali),
            "bokmal" | "norwegian bokmal" | "norwegian" => Some(Language::Bokmal),
            "bosnian" => Some(Language::Bosnian),
            "bulgarian" => Some(Language::Bulgarian),
            // C
            "catalan" => Some(Language::Catalan),
            "chinese" => Some(Language::Chinese),
            "croatian" => Some(Language::Croatian),
            "czech" => Some(Language::Czech),
            // D
            "danish" => Some(Language::Danish),
            "dutch" => Some(Language::Dutch),
            // E
            "english" => Some(Language::English),
            "esperanto" => Some(Language::Esperanto),
            "estonian" => Some(Language::Estonian),
            // F
            "finnish" => Some(Language::Finnish),
            "french" => Some(Language::French),
            // G
            "ganda" => Some(Language::Ganda),
            "georgian" => Some(Language::Georgian),
            "german" => Some(Language::German),
            "greek" => Some(Language::Greek),
            "gujarati" => Some(Language::Gujarati),
            // H
            "hebrew" => Some(Language::Hebrew),
            "hindi" => Some(Language::Hindi),
            "hungarian" => Some(Language::Hungarian),
            // I
            "icelandic" => Some(Language::Icelandic),
            "indonesian" => Some(Language::Indonesian),
            "irish" => Some(Language::Irish),
            "italian" => Some(Language::Italian),
            // J
            "japanese" => Some(Language::Japanese),
            // K
            "kazakh" => Some(Language::Kazakh),
            "korean" => Some(Language::Korean),
            // L
            "latin" => Some(Language::Latin),
            "latvian" => Some(Language::Latvian),
            "lithuanian" => Some(Language::Lithuanian),
            // M
            "macedonian" => Some(Language::Macedonian),
            "malay" => Some(Language::Malay),
            "maori" => Some(Language::Maori),
            "marathi" => Some(Language::Marathi),
            "mongolian" => Some(Language::Mongolian),
            // N
            "nynorsk" | "norwegian nynorsk" => Some(Language::Nynorsk),
            // P
            "persian" => Some(Language::Persian),
            "polish" => Some(Language::Polish),
            "portuguese" => Some(Language::Portuguese),
            "punjabi" => Some(Language::Punjabi),
            // R
            "romanian" => Some(Language::Romanian),
            "russian" => Some(Language::Russian),
            // S
            "serbian" => Some(Language::Serbian),
            "shona" => Some(Language::Shona),
            "slovak" => Some(Language::Slovak),
            "slovene" => Some(Language::Slovene),
            "somali" => Some(Language::Somali),
            "sotho" => Some(Language::Sotho),
            "spanish" => Some(Language::Spanish),
            "swahili" => Some(Language::Swahili),
            "swedish" => Some(Language::Swedish),
            // T
            "tagalog" => Some(Language::Tagalog),
            "tamil" => Some(Language::Tamil),
            "telugu" | "telegu" => Some(Language::Telugu),
            "thai" => Some(Language::Thai),
            "tsonga" => Some(Language::Tsonga),
            "tswana" => Some(Language::Tswana),
            "turkish" => Some(Language::Turkish),
            // U
            "ukrainian" => Some(Language::Ukrainian),
            "urdu" => Some(Language::Urdu),
            // V
            "vietnamese" => Some(Language::Vietnamese),
            // W
            "welsh" => Some(Language::Welsh),
            // X
            "xhosa" => Some(Language::Xhosa),
            // Y
            "yoruba" => Some(Language::Yoruba),
            // Z
            "zulu" => Some(Language::Zulu),
            _ => None,
        }
    }

    pub fn map_from_lingua_language(language: Language) -> String {
        match language {
            // A
            Language::Afrikaans => "afrikaans".to_string(),
            Language::Albanian => "albanian".to_string(),
            Language::Arabic => "arabic".to_string(),
            Language::Armenian => "armenian".to_string(),
            Language::Azerbaijani => "azerbaijani".to_string(),
            // B
            Language::Basque => "basque".to_string(),
            Language::Belarusian => "belarusian".to_string(),
            Language::Bengali => "bengali".to_string(),
            Language::Bokmal => "bokmal".to_string(),
            Language::Bosnian => "bosnian".to_string(),
            Language::Bulgarian => "bulgarian".to_string(),
            // C
            Language::Catalan => "catalan".to_string(),
            Language::Chinese => "chinese".to_string(),
            Language::Croatian => "croatian".to_string(),
            Language::Czech => "czech".to_string(),
            // D
            Language::Danish => "danish".to_string(),
            Language::Dutch => "dutch".to_string(),
            // E
            Language::English => "english".to_string(),
            Language::Esperanto => "esperanto".to_string(),
            Language::Estonian => "estonian".to_string(),
            // F
            Language::Finnish => "finnish".to_string(),
            Language::French => "french".to_string(),
            // G
            Language::Ganda => "ganda".to_string(),
            Language::Georgian => "georgian".to_string(),
            Language::German => "german".to_string(),
            Language::Greek => "greek".to_string(),
            Language::Gujarati => "gujarati".to_string(),
            // H
            Language::Hebrew => "hebrew".to_string(),
            Language::Hindi => "hindi".to_string(),
            Language::Hungarian => "hungarian".to_string(),
            // I
            Language::Icelandic => "icelandic".to_string(),
            Language::Indonesian => "indonesian".to_string(),
            Language::Irish => "irish".to_string(),
            Language::Italian => "italian".to_string(),
            // J
            Language::Japanese => "japanese".to_string(),
            // K
            Language::Kazakh => "kazakh".to_string(),
            Language::Korean => "korean".to_string(),
            // L
            Language::Latin => "latin".to_string(),
            Language::Latvian => "latvian".to_string(),
            Language::Lithuanian => "lithuanian".to_string(),
            // M
            Language::Macedonian => "macedonian".to_string(),
            Language::Malay => "malay".to_string(),
            Language::Maori => "maori".to_string(),
            Language::Marathi => "marathi".to_string(),
            Language::Mongolian => "mongolian".to_string(),
            // N
            Language::Nynorsk => "nynorsk".to_string(),
            // P
            Language::Persian => "persian".to_string(),
            Language::Polish => "polish".to_string(),
            Language::Portuguese => "portuguese".to_string(),
            Language::Punjabi => "punjabi".to_string(),
            // R
            Language::Romanian => "romanian".to_string(),
            Language::Russian => "russian".to_string(),
            // S
            Language::Serbian => "serbian".to_string(),
            Language::Shona => "shona".to_string(),
            Language::Slovak => "slovak".to_string(),
            Language::Slovene => "slovene".to_string(),
            Language::Somali => "somali".to_string(),
            Language::Sotho => "sotho".to_string(),
            Language::Spanish => "spanish".to_string(),
            Language::Swahili => "swahili".to_string(),
            Language::Swedish => "swedish".to_string(),
            // T
            Language::Tagalog => "tagalog".to_string(),
            Language::Tamil => "tamil".to_string(),
            Language::Telugu => "telugu".to_string(),
            Language::Thai => "thai".to_string(),
            Language::Tsonga => "tsonga".to_string(),
            Language::Tswana => "tswana".to_string(),
            Language::Turkish => "turkish".to_string(),
            // U
            Language::Ukrainian => "ukrainian".to_string(),
            Language::Urdu => "urdu".to_string(),
            // V
            Language::Vietnamese => "vietnamese".to_string(),
            // W
            Language::Welsh => "welsh".to_string(),
            // X
            Language::Xhosa => "xhosa".to_string(),
            // Y
            Language::Yoruba => "yoruba".to_string(),
            // Z
            Language::Zulu => "zulu".to_string(),
        }
    }
}
