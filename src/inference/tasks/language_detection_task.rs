//! Language detection task.
//!
//! This module provides automatic language detection capabilities for documents.
//!
//! Language detection works by:
//! 1. Sampling up to [`MAX_SAMPLE_LINES`] text regions from the document
//! 2. If the document does not contain embedded text, running each sample through different OCR models (grouped by script)
//!    2a. Selecting the model group with the highest average confidence score
//! 3. Using linguistic analysis to narrow down to a specific language

use std::collections::HashMap;

use image::RgbImage;
use lingua::Language;
use rand::rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::document::text_box::TextBox;
use crate::inference::crnn::Crnn;
use crate::inference::error::InferenceError;
use crate::utils::lang_utils::LangUtils;

/// Maximum number of text lines to sample for language detection
const MAX_SAMPLE_LINES: usize = 3;

/// The result of a language detection operation.
///
/// Contains information about the detected language, including which
/// OCR model was used and the confidence level of the detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    /// The detected language as a Lingua `Language` enum.
    pub language: Language,
    /// The filename of the OCR model that produced the best results.
    ///
    /// This may be empty for text-based detection where no OCR model is used.
    pub model_file: String,
    /// The confidence score from the detection process (0.0 to 1.0).
    ///
    /// For image-based detection, this is the average OCR confidence score
    /// across sampled text regions. For text-based detection, this is
    /// typically set to 1.0.
    pub confidence: f32,
}

impl LanguageDetectionResult {
    /// Creates a new `LanguageDetectionResult`.
    ///
    /// # Arguments
    ///
    /// * `language` - The detected language as a Lingua `Language` enum.
    /// * `model_file` - The OCR model filename used for detection.
    /// * `confidence` - The confidence score (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// A new `LanguageDetectionResult` instance.
    #[must_use]
    pub fn new(language: Language, model_file: String, confidence: f32) -> Self {
        Self {
            language,
            model_file,
            confidence,
        }
    }

    /// Gets the detected language name as a lowercase string.
    ///
    /// This is useful for logging or when a string representation is needed.
    ///
    /// # Returns
    ///
    /// A lowercase string representation of the language (e.g., "english", "chinese").
    #[must_use]
    pub fn language_name(&self) -> String {
        LangUtils::map_from_lingua_language(self.language)
    }
}

/// A group of languages that share the same OCR model.
///
/// Many languages share a common script (e.g., Latin alphabet), allowing
/// them to use the same OCR model. This struct groups languages by their
/// associated model file for efficient detection.
#[derive(Debug, Clone)]
struct ModelGroup {
    /// The filename of the OCR model for this group.
    model_file: String,
    /// The list of language names that use this model.
    languages: Vec<String>,
}

/// Task handler for automatic language detection in documents.
///
/// This struct provides methods for detecting the language of document content
/// using either image-based OCR analysis or text-based linguistic analysis.
/// The image-based approach is more accurate for scanned documents, while
/// text-based detection is faster when text has already been extracted.
pub struct LanguageDetectionTask;

impl LanguageDetectionTask {
    /// Builds a mapping of OCR model files to their supported languages.
    ///
    /// Loads all available language configurations and groups them by the
    /// OCR model file they use.
    ///
    /// # Returns
    ///
    /// A `HashMap` where keys are model filenames and values are [`ModelGroup`]
    /// instances containing the model file and list of supported languages.
    ///
    /// # Errors
    ///
    /// Returns an [`InferenceError::PreprocessingError`] if the language
    /// configuration files cannot be loaded.
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

    /// Samples text lines for language detection.
    ///
    /// Randomly selects up to [`MAX_SAMPLE_LINES`] text regions to use for
    /// language detection. Using a sample rather than all lines improves
    /// performance while maintaining accuracy.
    ///
    /// # Arguments
    ///
    /// * `text_boxes` - The detected text boxes from the document.
    /// * `part_images` - The cropped images corresponding to each text box.
    ///
    /// # Returns
    ///
    /// A vector of tuples pairing text boxes with their corresponding images.
    /// Returns an empty vector if the input arrays have mismatched lengths.
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

        let mut rng = rng();
        let mut indices: Vec<usize> = (0..pairs.len()).collect();
        indices.shuffle(&mut rng);

        indices
            .into_iter()
            .take(MAX_SAMPLE_LINES)
            .map(|i| pairs[i])
            .collect()
    }

    /// Processes sampled text lines with a specific OCR model.
    ///
    /// Runs OCR on all sampled text regions using the specified language's
    /// model and calculates the average confidence score.
    ///
    /// # Arguments
    ///
    /// * `language` - The language identifier to use for loading the OCR model.
    /// * `samples` - The sampled text box and image pairs to process.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The average confidence score across all samples
    /// - A vector of recognized text strings
    ///
    /// # Errors
    ///
    /// Returns an [`InferenceError`] if the OCR model fails to load or
    /// process any of the samples.
    fn process_with_model(
        language: &str,
        samples: &[(&TextBox, &RgbImage)],
    ) -> Result<(f32, Vec<String>), InferenceError> {
        let lang_enum = LangUtils::parse_language(language).ok_or_else(|| {
            InferenceError::PreprocessingError {
                operation: "parse language".to_string(),
                message: format!("Unknown language: {language}"),
            }
        })?;

        let images: Vec<&RgbImage> = samples.iter().map(|(_, img)| *img).collect();
        let mut text_boxes: Vec<TextBox> = samples.iter().map(|(tb, _)| (*tb).clone()).collect();

        Crnn::with_instance(lang_enum, |crnn| {
            let images_owned: Vec<RgbImage> = images.iter().map(|img| (*img).clone()).collect();
            crnn.get_texts(&images_owned, &mut text_boxes)
        })?;

        let mut total_score = 0.0;
        let mut recognized_texts = Vec::new();

        for text_box in &text_boxes {
            if let Some(text) = &text_box.text {
                recognized_texts.push(text.clone());
            }
            total_score += text_box.text_score;
        }

        let avg_score = if samples.is_empty() {
            0.0
        } else {
            total_score / samples.len() as f32
        };

        Ok((avg_score, recognized_texts))
    }

    /// Detects the language of a document using image-based OCR analysis.
    ///
    /// This method samples text regions from the document, runs them through
    /// multiple OCR models with different script support, and determines which
    /// model produces the highest confidence results. It then uses linguistic
    /// analysis to narrow down to a specific language within the winning
    /// model's supported language group.
    ///
    /// # Arguments
    ///
    /// * `text_boxes` - The detected text boxes from layout analysis.
    /// * `part_images` - The cropped RGB images for each text box.
    ///   Must be the same length as `text_boxes`.
    ///
    /// # Returns
    ///
    /// Returns a [`LanguageDetectionResult`] containing the detected language,
    /// the model file used, and the confidence score.
    ///
    /// # Errors
    ///
    /// Returns an [`InferenceError`] if:
    /// - The input arrays are empty
    /// - No OCR models are available
    /// - Text line sampling fails
    /// - All OCR models fail to process the samples
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
            .unwrap_or_else(|| {
                best_group
                    .languages
                    .first()
                    .and_then(|l| LangUtils::parse_language(l))
                    .unwrap_or(Language::English)
            });

        Ok(LanguageDetectionResult::new(
            language,
            best_group.model_file,
            confidence,
        ))
    }

    /// Detects the language of text using linguistic analysis.
    ///
    /// This method analyzes text to determine the language.
    ///
    /// # Arguments
    ///
    /// * `texts` - A slice of text strings extracted from the document.
    ///   Multiple strings are analyzed together for better accuracy.
    ///
    /// # Returns
    ///
    /// Returns a [`LanguageDetectionResult`] with the detected language.
    /// The `model_file` field will be empty since no OCR model was used,
    /// and the confidence will be set to 1.0.
    ///
    /// # Errors
    ///
    /// Returns an [`InferenceError::PreprocessingError`] if:
    /// - The input text array is empty
    /// - Language configuration files cannot be loaded
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

        let language =
            LangUtils::detect_language(texts, &candidate_languages).unwrap_or(Language::English);

        Ok(LanguageDetectionResult::new(language, String::new(), 1.0))
    }
}
