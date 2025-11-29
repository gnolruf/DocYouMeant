use std::collections::HashMap;

use image::RgbImage;
use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

use crate::document::text_box::TextBox;
use crate::inference::crnn::Crnn;
use crate::inference::error::InferenceError;
use crate::utils::lang_utils::LangUtils;

/// Maximum number of text lines to sample for language detection
const MAX_SAMPLE_LINES: usize = 3;

/// Result of language detection for a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    /// The detected language name
    pub language: String,
    /// The model file that was used for detection
    pub model_file: String,
    /// The confidence score from the OCR model
    pub confidence: f32,
    /// Whether Lingua was used for disambiguation
    pub used_lingua: bool,
}

impl LanguageDetectionResult {
    pub fn new(language: String, model_file: String, confidence: f32, used_lingua: bool) -> Self {
        Self {
            language,
            model_file,
            confidence,
            used_lingua,
        }
    }
}

/// Represents a model group with its associated languages
#[derive(Debug, Clone)]
struct ModelGroup {
    model_file: String,
    languages: Vec<String>,
}

pub struct LanguageDetectionTask;

impl LanguageDetectionTask {
    /// Builds a mapping of model files to their supported languages (script models only)
    fn build_model_groups() -> Result<HashMap<String, ModelGroup>, InferenceError> {
        let configs = LangUtils::get_all_language_configs(true).map_err(|e| {
            InferenceError::PreprocessingError {
                operation: "load language configs".to_string(),
                message: e.to_string(),
            }
        })?;

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

    /// Samples up to MAX_SAMPLE_LINES text line images for classification
    fn sample_text_lines<'a>(
        text_boxes: &'a [TextBox],
        part_images: &'a [RgbImage],
    ) -> Vec<(&'a TextBox, &'a RgbImage)> {
        if text_boxes.len() != part_images.len() {
            return Vec::new();
        }

        let pairs: Vec<_> = text_boxes.iter().zip(part_images.iter()).collect();

        if pairs.len() <= MAX_SAMPLE_LINES {
            return pairs;
        }

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..pairs.len()).collect();
        indices.shuffle(&mut rng);

        indices
            .into_iter()
            .take(MAX_SAMPLE_LINES)
            .map(|i| pairs[i])
            .collect()
    }

    /// Processes text lines with a specific language model and returns the average confidence
    fn process_with_model(
        language: &str,
        samples: &[(&TextBox, &RgbImage)],
    ) -> Result<(f32, Vec<String>), InferenceError> {
        let mut crnn = Crnn::new(language)?;

        let mut total_score = 0.0;
        let mut recognized_texts = Vec::new();

        for (text_box, image) in samples {
            let mut text_box_clone = (*text_box).clone();
            let (_, _) = crnn.get_text(image, &mut text_box_clone, 0)?;

            if let Some(text) = &text_box_clone.text {
                recognized_texts.push(text.clone());
            }
            total_score += text_box_clone.text_score;
        }

        let avg_score = if samples.is_empty() {
            0.0
        } else {
            total_score / samples.len() as f32
        };

        Ok((avg_score, recognized_texts))
    }

    /// Uses Lingua to detect the specific language from a set of candidate languages
    fn detect_language_with_lingua(
        texts: &[String],
        candidate_languages: &[String],
    ) -> Option<String> {
        if texts.is_empty() || candidate_languages.is_empty() {
            return None;
        }

        // Map language names to Lingua Language enum
        let lingua_languages: Vec<Language> = candidate_languages
            .iter()
            .filter_map(|lang| Self::map_to_lingua_language(lang))
            .collect();

        if lingua_languages.is_empty() {
            // If no languages could be mapped, return the first candidate
            return candidate_languages.first().cloned();
        }

        let detector: LanguageDetector = LanguageDetectorBuilder::from_languages(&lingua_languages)
            .with_preloaded_language_models()
            .build();

        // Combine all recognized texts for detection
        let combined_text = texts.join(" ");

        detector
            .detect_language_of(&combined_text)
            .and_then(|detected| Self::map_from_lingua_language(detected))
            .or_else(|| candidate_languages.first().cloned())
    }

    /// Maps our language names to Lingua Language enum
    fn map_to_lingua_language(language: &str) -> Option<Language> {
        match language.to_lowercase().as_str() {
            "spanish" => Some(Language::Spanish),
            "french" => Some(Language::French),
            "german" => Some(Language::German),
            "italian" => Some(Language::Italian),
            "portuguese" => Some(Language::Portuguese),
            "dutch" => Some(Language::Dutch),
            "turkish" => Some(Language::Turkish),
            "indonesian" => Some(Language::Indonesian),
            "malay" => Some(Language::Malay),
            "vietnamese" => Some(Language::Vietnamese),
            "tagalog" => Some(Language::Tagalog),
            "swahili" => Some(Language::Swahili),
            "swedish" => Some(Language::Swedish),
            "norwegian" => Some(Language::Bokmal),
            "danish" => Some(Language::Danish),
            "finnish" => Some(Language::Finnish),
            "hungarian" => Some(Language::Hungarian),
            "romanian" => Some(Language::Romanian),
            "catalan" => Some(Language::Catalan),
            "basque" => Some(Language::Basque),
            "welsh" => Some(Language::Welsh),
            "irish" => Some(Language::Irish),
            "icelandic" => Some(Language::Icelandic),
            "latvian" => Some(Language::Latvian),
            "lithuanian" => Some(Language::Lithuanian),
            "estonian" => Some(Language::Estonian),
            "afrikaans" => Some(Language::Afrikaans),
            "zulu" => Some(Language::Zulu),
            "xhosa" => Some(Language::Xhosa),
            "somali" => Some(Language::Somali),
            "yoruba" => Some(Language::Yoruba),
            "chinese" => Some(Language::Chinese),
            "japanese" => Some(Language::Japanese),
            "thai" => Some(Language::Thai),
            "greek" => Some(Language::Greek),
            "arabic" => Some(Language::Arabic),
            "hindi" => Some(Language::Hindi),
            "marathi" => Some(Language::Marathi),
            "kazakh" => Some(Language::Kazakh),
            "mongolian" => Some(Language::Mongolian),
            "telegu" => Some(Language::Telugu),
            "tamil" => Some(Language::Tamil),
            _ => None,
        }
    }

    /// Maps Lingua Language enum back to our language names
    fn map_from_lingua_language(language: Language) -> Option<String> {
        match language {
            Language::Spanish => Some("spanish".to_string()),
            Language::French => Some("french".to_string()),
            Language::German => Some("german".to_string()),
            Language::Italian => Some("italian".to_string()),
            Language::Portuguese => Some("portuguese".to_string()),
            Language::Dutch => Some("dutch".to_string()),
            Language::Turkish => Some("turkish".to_string()),
            Language::Indonesian => Some("indonesian".to_string()),
            Language::Malay => Some("malay".to_string()),
            Language::Vietnamese => Some("vietnamese".to_string()),
            Language::Tagalog => Some("tagalog".to_string()),
            Language::Swahili => Some("swahili".to_string()),
            Language::Swedish => Some("swedish".to_string()),
            Language::Bokmal => Some("norwegian".to_string()),
            Language::Danish => Some("danish".to_string()),
            Language::Finnish => Some("finnish".to_string()),
            Language::Hungarian => Some("hungarian".to_string()),
            Language::Romanian => Some("romanian".to_string()),
            Language::Catalan => Some("catalan".to_string()),
            Language::Basque => Some("basque".to_string()),
            Language::Welsh => Some("welsh".to_string()),
            Language::Irish => Some("irish".to_string()),
            Language::Icelandic => Some("icelandic".to_string()),
            Language::Latvian => Some("latvian".to_string()),
            Language::Lithuanian => Some("lithuanian".to_string()),
            Language::Estonian => Some("estonian".to_string()),
            Language::Afrikaans => Some("afrikaans".to_string()),
            Language::Zulu => Some("zulu".to_string()),
            Language::Xhosa => Some("xhosa".to_string()),
            Language::Somali => Some("somali".to_string()),
            Language::Yoruba => Some("yoruba".to_string()),
            Language::Chinese => Some("chinese".to_string()),
            Language::Japanese => Some("japanese".to_string()),
            Language::Thai => Some("thai".to_string()),
            Language::Greek => Some("greek".to_string()),
            Language::Arabic => Some("arabic".to_string()),
            Language::Hindi => Some("hindi".to_string()),
            Language::Marathi => Some("marathi".to_string()),
            Language::Kazakh => Some("kazakh".to_string()),
            Language::Mongolian => Some("mongolian".to_string()),
            Language::Telugu => Some("telegu".to_string()),
            Language::Tamil => Some("tamil".to_string()),
            _ => None,
        }
    }

    /// Detects the language of the document based on text line images
    ///
    /// This function samples up to 3 text lines and tests them against all available
    /// script-based OCR models to find the one with the highest confidence score.
    /// If the best model supports multiple languages, Lingua is used for disambiguation.
    ///
    /// # Arguments
    /// * `text_boxes` - The detected text boxes from the page
    /// * `part_images` - The corresponding cropped images for each text box
    ///
    /// # Returns
    /// A `LanguageDetectionResult` containing the detected language and confidence
    pub fn detect(
        text_boxes: &[TextBox],
        part_images: &[RgbImage],
    ) -> Result<LanguageDetectionResult, InferenceError> {
        if text_boxes.is_empty() || part_images.is_empty() {
            return Err(InferenceError::PreprocessingError {
                operation: "validate input".to_string(),
                message: "No text boxes or images provided for language detection".to_string(),
            });
        }

        // Build model groups from script models
        let model_groups = Self::build_model_groups()?;

        if model_groups.is_empty() {
            return Err(InferenceError::PreprocessingError {
                operation: "load models".to_string(),
                message: "No script-based OCR models found".to_string(),
            });
        }

        // Sample text lines for detection
        let samples = Self::sample_text_lines(text_boxes, part_images);

        if samples.is_empty() {
            return Err(InferenceError::PreprocessingError {
                operation: "sample text lines".to_string(),
                message: "Failed to sample text lines for language detection".to_string(),
            });
        }

        let mut best_result: Option<(f32, ModelGroup, Vec<String>)> = None;

        // Test each model group
        for (_, model_group) in model_groups {
            // Use the first language in the group to load the model
            // (they all share the same model file)
            let test_language = model_group.languages.first().ok_or_else(|| {
                InferenceError::PreprocessingError {
                    operation: "get test language".to_string(),
                    message: "Model group has no languages".to_string(),
                }
            })?;

            match Self::process_with_model(test_language, &samples) {
                Ok((score, texts)) => {
                    let is_better = best_result
                        .as_ref()
                        .map(|(best_score, _, _)| score > *best_score)
                        .unwrap_or(true);

                    if is_better {
                        best_result = Some((score, model_group.clone(), texts));
                    }
                }
                Err(e) => {
                    // Log error but continue with other models
                    tracing::warn!(
                        "Failed to process with model {}: {}",
                        model_group.model_file,
                        e
                    );
                }
            }
        }

        let (confidence, best_group, recognized_texts) =
            best_result.ok_or_else(|| InferenceError::PredictionError {
                operation: "language detection".to_string(),
                message: "All models failed to process the text lines".to_string(),
            })?;

        // Determine the final language
        let (language, used_lingua) = if best_group.languages.len() == 1 {
            // Single language model - no disambiguation needed
            (best_group.languages[0].clone(), false)
        } else {
            // Multiple languages share this model - use Lingua for disambiguation
            let detected =
                Self::detect_language_with_lingua(&recognized_texts, &best_group.languages);
            match detected {
                Some(lang) => (lang, true),
                None => (best_group.languages[0].clone(), false),
            }
        };

        Ok(LanguageDetectionResult::new(
            language,
            best_group.model_file,
            confidence,
            used_lingua,
        ))
    }
}
