use std::collections::HashMap;

use image::RgbImage;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

use crate::document::text_box::TextBox;
use crate::inference::crnn::Crnn;
use crate::inference::error::InferenceError;
use crate::utils::lang_utils::LangUtils;

/// Maximum number of text lines to sample for language detection
const MAX_SAMPLE_LINES: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    /// The detected language name
    pub language: String,
    /// The model file that was used for detection
    pub model_file: String,
    /// The confidence score from the OCR model
    pub confidence: f32,
}

impl LanguageDetectionResult {
    pub fn new(language: String, model_file: String, confidence: f32) -> Self {
        Self {
            language,
            model_file,
            confidence,
        }
    }
}

#[derive(Debug, Clone)]
struct ModelGroup {
    model_file: String,
    languages: Vec<String>,
}

pub struct LanguageDetectionTask;

impl LanguageDetectionTask {
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

        let model_groups = Self::build_model_groups()?;

        if model_groups.is_empty() {
            return Err(InferenceError::PreprocessingError {
                operation: "load models".to_string(),
                message: "No script-based OCR models found".to_string(),
            });
        }

        let samples = Self::sample_text_lines(text_boxes, part_images);

        if samples.is_empty() {
            return Err(InferenceError::PreprocessingError {
                operation: "sample text lines".to_string(),
                message: "Failed to sample text lines for language detection".to_string(),
            });
        }

        let mut best_result: Option<(f32, ModelGroup, Vec<String>)> = None;

        for (_, model_group) in model_groups {
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

        let mut candidate_languages = best_group.languages.clone();

        if !candidate_languages.iter().any(|l| l == "english") {
            candidate_languages.push("english".to_string());
        }
        if !candidate_languages.iter().any(|l| l == "chinese") {
            candidate_languages.push("chinese".to_string());
        }

        let language = LangUtils::detect_language(&recognized_texts, &candidate_languages)
            .unwrap_or_else(|| best_group.languages.first().cloned().unwrap_or_default());

        Ok(LanguageDetectionResult::new(
            language,
            best_group.model_file,
            confidence,
        ))
    }

    pub fn detect_from_text(texts: &[String]) -> Result<LanguageDetectionResult, InferenceError> {
        if texts.is_empty() {
            return Err(InferenceError::PreprocessingError {
                operation: "validate input".to_string(),
                message: "No text provided for language detection".to_string(),
            });
        }

        let configs = LangUtils::get_all_language_configs(false).map_err(|e| {
            InferenceError::PreprocessingError {
                operation: "load language configs".to_string(),
                message: e.to_string(),
            }
        })?;

        let candidate_languages: Vec<String> = configs.keys().cloned().collect();

        let language = LangUtils::detect_language(texts, &candidate_languages)
            .unwrap_or_else(|| "english".to_string());

        Ok(LanguageDetectionResult::new(language, String::new(), 1.0))
    }
}
