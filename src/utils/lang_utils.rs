//! Utility functions for language configurations.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;

use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};
use serde::{Deserialize, Serialize};

use super::config::AppConfig;

/// Text directionality for a language.
///
/// Indicates whether a language is read left-to-right (LTR) or right-to-left (RTL).
/// This affects text line ordering, word sorting, and OCR image padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Directionality {
    /// Left-to-right text direction
    #[default]
    Ltr,
    /// Right-to-left text direction
    Rtl,
}

/// Configuration for a language's OCR model and dictionary.
///
/// This struct holds the necessary file paths and metadata for loading
/// language-specific OCR recognition models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageModelInfo {
    /// The human-readable name of the language (e.g., "english", "chinese").
    pub name: String,
    /// Path to the ONNX model file for text recognition.
    pub model_file: String,
    /// Path to the dictionary file containing valid characters for the language.
    pub dict_file: String,
    /// Whether this is a script-based model (covers multiple languages using the same script).
    pub is_script_model: bool,
    /// Text directionality for this language (LTR or RTL).
    #[serde(default)]
    pub directionality: Directionality,
}

/// A group of languages that share the same OCR model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGroup {
    /// The filename of the OCR model for this group.
    pub model_file: String,
    /// The list of language names that use this model.
    pub languages: Vec<String>,
}

/// Utility struct for language detection and configuration management.
///
/// Provides static methods for:
/// - Loading language configurations from JSON files
/// - Detecting languages from text using the Lingua library
/// - Mapping between internal language names and Lingua's `Language` enum
pub struct LangUtils;

impl LangUtils {
    /// Retrieves the information about the model for a specific language.
    ///
    /// Looks up the language model info from the JSON model config file using the
    /// Lingua `Language` enum. If the model config file cannot be read and the
    /// requested language is English, returns a default English configuration as a fallback.
    ///
    /// # Arguments
    ///
    /// * `language` - The Lingua `Language` enum value to look up.
    ///
    /// # Returns
    ///
    /// * `Some(LanguageModelInfo)` - The configuration for the requested language if found.
    /// * `None` - If the language is not supported or not found in the configuration.
    #[must_use]
    pub fn get_language_model_info(language: Language) -> Option<LanguageModelInfo> {
        let language_str = Self::map_from_lingua_language(language);
        let config = AppConfig::get();
        match Self::get_all_language_configs(false) {
            Ok(configs) => configs.get(language_str.as_ref()).cloned(),
            Err(_) => {
                if language == Language::English {
                    Some(LanguageModelInfo {
                        name: "english".to_string(),
                        model_file: config.model_path("onnx/text_recognition_en.onnx"),
                        dict_file: config.model_path("dict/en_dict.txt"),
                        is_script_model: false,
                        directionality: Directionality::Ltr,
                    })
                } else {
                    None
                }
            }
        }
    }

    /// Gets the text directionality for a language.
    ///
    /// Returns the directionality (LTR or RTL) for the specified language
    /// based on its configuration. Defaults to LTR if the language configuration
    /// is not found.
    ///
    /// # Arguments
    ///
    /// * `language` - The Lingua `Language` enum value to look up.
    ///
    /// # Returns
    ///
    /// The text directionality for the language, defaulting to LTR.
    #[must_use]
    pub fn get_directionality(language: Language) -> Directionality {
        Self::get_language_model_info(language)
            .map(|info| info.directionality)
            .unwrap_or_default()
    }

    /// Parses a language string and returns the corresponding Lingua Language enum.
    ///
    /// This is useful for converting command-line arguments or configuration strings
    /// to type-safe Language enum values.
    ///
    /// # Arguments
    ///
    /// * `language_str` - The language name string to parse (case-insensitive).
    ///
    /// # Returns
    ///
    /// * `Some(Language)` - The corresponding Lingua `Language` enum variant.
    /// * `None` - If the language string is not recognized.
    #[must_use]
    pub fn parse_language(language_str: &str) -> Option<Language> {
        Self::map_to_lingua_language(language_str)
    }

    /// Retrieves all available language configurations.
    ///
    /// Loads and parses the language configurations from `models/ocr_lang_models.json`.
    /// Optionally filters to return only script-based models.
    ///
    /// # Arguments
    ///
    /// * `script_models_only` - If `true`, returns only configurations where `is_script_model` is `true`.
    ///   Script models cover multiple languages that share the same writing system.
    ///
    /// # Returns
    ///
    /// * `Ok(HashMap<String, LanguageConfig>)` - A map of language names to their configurations.
    /// * `Err` - If the config file cannot be read or parsed.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The config file at `models/ocr_lang_models.json` cannot be read.
    /// - The JSON content cannot be deserialized into the expected format.
    pub fn get_all_language_configs(
        script_models_only: bool,
    ) -> Result<HashMap<String, LanguageModelInfo>, Box<dyn std::error::Error>> {
        let config = AppConfig::get();
        let config_path = config.model_path("ocr_lang_models.json");
        let config_content = fs::read_to_string(config_path)?;
        let configs: HashMap<String, LanguageModelInfo> = serde_json::from_str(&config_content)?;

        if script_models_only {
            Ok(configs
                .into_iter()
                .filter(|(_, config)| config.is_script_model)
                .collect())
        } else {
            Ok(configs)
        }
    }

    /// Builds a mapping of OCR model files to their supported languages.
    ///
    /// # Arguments
    ///
    /// * `script_models_only` - If `true`, only includes script-based models.
    ///
    /// # Returns
    ///
    /// A `HashMap` where keys are model filenames and values are [`ModelGroup`]
    /// instances containing the model file and list of supported languages.
    pub fn get_model_groups(
        script_models_only: bool,
    ) -> Result<HashMap<String, ModelGroup>, Box<dyn std::error::Error>> {
        let configs = Self::get_all_language_configs(script_models_only)?;
        let mut model_groups: HashMap<String, ModelGroup> = HashMap::new();

        for (lang_name, config) in configs {
            let entry = model_groups
                .entry(config.model_file.clone())
                .or_insert_with(|| ModelGroup {
                    model_file: config.model_file.clone(),
                    languages: Vec::new(),
                });

            entry.languages.push(lang_name);
        }

        Ok(model_groups)
    }

    /// Retrieves the model configuration for a specific model file.
    ///
    /// # Arguments
    ///
    /// * `model_file` - The path to the model file.
    ///
    /// # Returns
    ///
    /// * `Some(LanguageModelInfo)` - The configuration for the model if found.
    /// * `None` - If no language uses this model file.
    #[must_use]
    pub fn get_model_info_by_file(model_file: &str) -> Option<LanguageModelInfo> {
        let configs = Self::get_all_language_configs(false).ok()?;
        configs
            .values()
            .find(|c| c.model_file == model_file)
            .cloned()
    }

    /// Detects the most likely language from a collection of text samples.
    ///
    /// Uses the Lingua library to perform statistical language detection on the combined
    /// text samples, constrained to a set of candidate languages. This improves accuracy
    /// by limiting the detection scope to plausible languages.
    ///
    /// # Arguments
    ///
    /// * `texts` - A slice of text strings to analyze. These are concatenated for detection.
    /// * `candidate_languages` - A slice of language names to consider (e.g., `["english", "german", "french"]`).
    ///
    /// # Returns
    ///
    /// * `Some(Language)` - The detected language if detection succeeds.
    /// * `None` - If `texts` or `candidate_languages` is empty.
    ///
    /// # Notes
    ///
    /// - If none of the candidate languages can be mapped to Lingua's supported languages,
    ///   the first parseable candidate language is returned as a fallback.
    /// - If Lingua cannot confidently detect a language, the first candidate is returned.
    #[must_use]
    pub fn detect_language(texts: &[String], candidate_languages: &[String]) -> Option<Language> {
        if texts.is_empty() || candidate_languages.is_empty() {
            return None;
        }

        let lingua_languages: Vec<Language> = candidate_languages
            .iter()
            .filter_map(|lang| Self::map_to_lingua_language(lang))
            .collect();

        if lingua_languages.is_empty() {
            return candidate_languages
                .first()
                .and_then(|l| Self::map_to_lingua_language(l));
        }

        let detector: LanguageDetector = LanguageDetectorBuilder::from_languages(&lingua_languages)
            .with_preloaded_language_models()
            .build();

        let combined_text = texts.join(" ");

        detector
            .detect_language_of(&combined_text)
            .or_else(|| lingua_languages.first().copied())
    }

    /// Maps an internal language name to a Lingua `Language` enum variant.
    ///
    /// Performs case-insensitive matching of language names to their corresponding
    /// Lingua enum values. Supports common aliases (e.g., "norwegian" maps to `Bokmal`).
    ///
    /// # Arguments
    ///
    /// * `language` - The language name to map (case-insensitive).
    ///
    /// # Returns
    ///
    /// * `Some(Language)` - The corresponding Lingua `Language` enum variant.
    /// * `None` - If the language name is not recognized or not supported by Lingua.
    #[must_use]
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

    /// Maps a Lingua `Language` enum variant back to an internal language name string.
    ///
    /// This is the inverse operation of [`map_to_lingua_language`]. Converts Lingua's
    /// detected language back to the lowercase string format used internally.
    ///
    /// # Arguments
    ///
    /// * `language` - The Lingua `Language` enum variant to convert.
    ///
    /// # Returns
    ///
    /// A [`Cow<'static, str>`] containing a lowercase string representation of the language
    /// (e.g., `Language::English` → `"english"`). Returns a borrowed static string to avoid
    /// allocations.
    ///
    /// # Note
    ///
    /// This function handles all Lingua language variants. The returned string matches
    /// the keys used in the language configuration files.
    #[must_use]
    pub fn map_from_lingua_language(language: Language) -> Cow<'static, str> {
        match language {
            // A
            Language::Afrikaans => "afrikaans".into(),
            Language::Albanian => "albanian".into(),
            Language::Arabic => "arabic".into(),
            Language::Armenian => "armenian".into(),
            Language::Azerbaijani => "azerbaijani".into(),
            // B
            Language::Basque => "basque".into(),
            Language::Belarusian => "belarusian".into(),
            Language::Bengali => "bengali".into(),
            Language::Bokmal => "bokmal".into(),
            Language::Bosnian => "bosnian".into(),
            Language::Bulgarian => "bulgarian".into(),
            // C
            Language::Catalan => "catalan".into(),
            Language::Chinese => "chinese".into(),
            Language::Croatian => "croatian".into(),
            Language::Czech => "czech".into(),
            // D
            Language::Danish => "danish".into(),
            Language::Dutch => "dutch".into(),
            // E
            Language::English => "english".into(),
            Language::Esperanto => "esperanto".into(),
            Language::Estonian => "estonian".into(),
            // F
            Language::Finnish => "finnish".into(),
            Language::French => "french".into(),
            // G
            Language::Ganda => "ganda".into(),
            Language::Georgian => "georgian".into(),
            Language::German => "german".into(),
            Language::Greek => "greek".into(),
            Language::Gujarati => "gujarati".into(),
            // H
            Language::Hebrew => "hebrew".into(),
            Language::Hindi => "hindi".into(),
            Language::Hungarian => "hungarian".into(),
            // I
            Language::Icelandic => "icelandic".into(),
            Language::Indonesian => "indonesian".into(),
            Language::Irish => "irish".into(),
            Language::Italian => "italian".into(),
            // J
            Language::Japanese => "japanese".into(),
            // K
            Language::Kazakh => "kazakh".into(),
            Language::Korean => "korean".into(),
            // L
            Language::Latin => "latin".into(),
            Language::Latvian => "latvian".into(),
            Language::Lithuanian => "lithuanian".into(),
            // M
            Language::Macedonian => "macedonian".into(),
            Language::Malay => "malay".into(),
            Language::Maori => "maori".into(),
            Language::Marathi => "marathi".into(),
            Language::Mongolian => "mongolian".into(),
            // N
            Language::Nynorsk => "nynorsk".into(),
            // P
            Language::Persian => "persian".into(),
            Language::Polish => "polish".into(),
            Language::Portuguese => "portuguese".into(),
            Language::Punjabi => "punjabi".into(),
            // R
            Language::Romanian => "romanian".into(),
            Language::Russian => "russian".into(),
            // S
            Language::Serbian => "serbian".into(),
            Language::Shona => "shona".into(),
            Language::Slovak => "slovak".into(),
            Language::Slovene => "slovene".into(),
            Language::Somali => "somali".into(),
            Language::Sotho => "sotho".into(),
            Language::Spanish => "spanish".into(),
            Language::Swahili => "swahili".into(),
            Language::Swedish => "swedish".into(),
            // T
            Language::Tagalog => "tagalog".into(),
            Language::Tamil => "tamil".into(),
            Language::Telugu => "telugu".into(),
            Language::Thai => "thai".into(),
            Language::Tsonga => "tsonga".into(),
            Language::Tswana => "tswana".into(),
            Language::Turkish => "turkish".into(),
            // U
            Language::Ukrainian => "ukrainian".into(),
            Language::Urdu => "urdu".into(),
            // V
            Language::Vietnamese => "vietnamese".into(),
            // W
            Language::Welsh => "welsh".into(),
            // X
            Language::Xhosa => "xhosa".into(),
            // Y
            Language::Yoruba => "yoruba".into(),
            // Z
            Language::Zulu => "zulu".into(),
        }
    }
}
