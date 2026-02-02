//! Convolutional Recurrent Neural Network (CRNN) for text recognition.
//!
//! This module provides text recognition capabilities using a CRNN architecture
//! that combines convolutional layers for feature extraction with recurrent layers
//! for sequence modeling. It supports multiple languages through language-specific
//! models and dictionaries.

use std::fs::File;
use std::io::{BufRead, BufReader};

use image::{imageops, RgbImage};
use lingua::Language;
use ort::{inputs, session::builder::PrepackedWeights, session::Session, value::Value};
use std::sync::OnceLock;

use crate::document::bounds::Bounds;
use crate::document::text_box::TextBox;
use crate::inference::error::InferenceError;
use crate::inference::KeyedSessionPool;
use crate::utils::image_utils;
use crate::utils::lang_utils::{Directionality, LangUtils};
use geo::Coord;

/// Text recognition engine using CRNN (Convolutional Recurrent Neural Network).
///
/// `Crnn` performs optical character recognition (OCR) on cropped text line images.
/// It uses a CTC (Connectionist Temporal Classification) decoder to convert
/// the model's output sequence into readable text with word-level bounding boxes.
///
/// # Fields
///
/// - `session`: ONNX Runtime session for model inference
/// - `keys`: Character dictionary mapping model outputs to characters
/// - `mean_values`: Per-channel mean values for image normalization
/// - `norm_values`: Per-channel normalization divisors
/// - `dst_height`: Target height for input images (maintains aspect ratio)
///
/// # Thread Safety
///
/// `Crnn` uses the keyed singleton pattern with `Language` as the key.
/// Each language has its own singleton instance, allowing thread-safe access
/// through the `with_instance` method.
pub struct Crnn {
    session: Session,
    keys: Vec<Box<str>>,
    mean_values: [f32; 3],
    norm_values: [f32; 3],
    dst_height: u32,
}

static CRNN_POOLS: OnceLock<KeyedSessionPool<String, Crnn>> = OnceLock::new();

fn crnn_pools() -> &'static KeyedSessionPool<String, Crnn> {
    CRNN_POOLS.get_or_init(KeyedSessionPool::new)
}

impl Crnn {
    /// Number of threads for ONNX Runtime inter-op parallelism.
    const NUM_THREADS: usize = 4;

    /// Creates a new CRNN text recognizer for the specified model file.
    ///
    /// Loads the appropriate ONNX model and character dictionary based on
    /// the model file configuration.
    ///
    /// # Arguments
    ///
    /// * `model_file` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// * `Ok(Crnn)` - Initialized recognizer ready for inference
    /// * `Err(InferenceError)` - If the model file is unsupported or cannot be loaded
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The specified model file is not supported
    /// - The model file cannot be loaded
    /// - The dictionary file cannot be read or parsed
    /// Pre-initializes the session pool for the specified model file key.
    pub fn get_or_init(model_file: String) -> Result<(), InferenceError> {
        let pool_size = crate::utils::config::AppConfig::get().inference_pool_size;
        crnn_pools().get_or_init(model_file.clone(), pool_size, |w| {
            Self::new(model_file.clone(), w)
        })
    }

    /// Executes a closure with exclusive access to the model for the given key.
    pub fn with_instance<F, R>(key: String, f: F) -> Result<R, InferenceError>
    where
        F: FnOnce(&mut Crnn) -> Result<R, InferenceError>,
    {
        Self::get_or_init(key.clone())?;
        crnn_pools().with(&key, f)
    }

    fn new(model_file: String, prepacked: &PrepackedWeights) -> Result<Self, InferenceError> {
        let config = LangUtils::get_model_info_by_file(&model_file).ok_or_else(|| {
            InferenceError::UnsupportedModel {
                name: model_file.clone(),
            }
        })?;

        let app_config = crate::utils::config::AppConfig::get();
        let session = Session::builder()
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: config.model_file.clone().into(),
                source,
            })?
            .with_execution_providers([ort::ep::TensorRT::default()
                .with_device_id(0)
                .with_engine_cache(true)
                .with_engine_cache_path(app_config.rt_cache_directory()?)
                .with_engine_cache_prefix("docyoumeant_")
                .with_max_workspace_size(5 << 30)
                .with_fp16(true)
                .with_timing_cache(true)
                .with_profile_min_shapes("x:1x3x48x32")
                .with_profile_max_shapes("x:1x3x48x1280")
                .with_profile_opt_shapes("x:1x3x48x320")
                .build()])?
            .with_inter_threads(Self::NUM_THREADS)?
            .with_prepacked_weights(prepacked)?
            .commit_from_file(&config.model_file)
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: config.model_file.clone().into(),
                source,
            })?;

        let file =
            File::open(&config.dict_file).map_err(|source| InferenceError::DataFileLoadError {
                path: config.dict_file.clone().into(),
                source,
            })?;
        let reader = BufReader::new(file);
        let lines: Result<Vec<_>, _> = reader.lines().collect();
        let mut keys: Vec<Box<str>> = lines
            .map_err(|source| InferenceError::DataFileParseError {
                path: config.dict_file.clone().into(),
                source,
            })?
            .into_iter()
            .map(String::into_boxed_str)
            .collect();

        Self::insert_special_characters(&mut keys);

        Ok(Self {
            session,
            keys,
            mean_values: [127.5, 127.5, 127.5],
            norm_values: [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5],
            dst_height: 48,
        })
    }

    /// Inserts special characters into the character dictionary.
    ///
    /// Adds the CTC blank token ("#") at the beginning and a space character
    /// at the end of the dictionary, as required by the CTC decoding algorithm.
    fn insert_special_characters(keys: &mut Vec<Box<str>>) {
        keys.insert(0, "#".into());
        keys.push(" ".into());
    }

    /// Recognizes text from images using the singleton instance for the specified language.
    ///
    /// # Arguments
    ///
    /// * `language` - The language to use for text recognition
    /// * `part_imgs` - Slice of RGB images, one per text line
    /// * `text_boxes` - Mutable slice of text boxes to update with results
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * `Vec<TextBox>` - All word-level text boxes from all lines
    /// * `Directionality` - The text directionality for this language
    ///
    /// # Errors
    ///
    /// Returns an error if recognition fails.
    pub fn recognize(
        language: Language,
        part_imgs: &[RgbImage],
        text_boxes: &mut [TextBox],
    ) -> Result<(Vec<TextBox>, Directionality), InferenceError> {
        let model_info = LangUtils::get_language_model_info(language).ok_or_else(|| {
            let lang_str = LangUtils::map_from_lingua_language(language);
            InferenceError::UnsupportedModel {
                name: lang_str.into_owned(),
            }
        })?;
        let directionality = model_info.directionality;
        let words = Self::with_instance(model_info.model_file, |crnn| {
            crnn.get_texts(part_imgs, text_boxes, directionality)
        })?;
        Ok((words, directionality))
    }

    /// Recognizes text from multiple text line images in batch.
    ///
    /// Processes multiple cropped text line images in a single batched forward pass.
    /// Images are padded to uniform width before batching. For RTL languages,
    /// padding is applied on the left side to maintain proper text alignment.
    /// Maintains a consistent global character offset across all lines for
    /// document-wide span tracking, starting from offset 0.
    ///
    /// # Arguments
    ///
    /// * `part_imgs` - Slice of RGB images, one per text line
    /// * `text_boxes` - Mutable slice of text boxes to update with results
    /// * `directionality` - Text direction (LTR or RTL) affecting padding and word bounds
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TextBox>)` - All word-level text boxes from all lines
    /// * `Err(InferenceError)` - If any recognition fails
    pub fn get_texts(
        &mut self,
        part_imgs: &[RgbImage],
        text_boxes: &mut [TextBox],
        directionality: Directionality,
    ) -> Result<Vec<TextBox>, InferenceError> {
        if part_imgs.is_empty() {
            return Ok(Vec::new());
        }

        let count = part_imgs.len().min(text_boxes.len());
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut all_words = Vec::new();
        let mut current_offset = 0;

        for i in 0..count {
            let img = &part_imgs[i];
            let scale = self.dst_height as f32 / img.height() as f32;
            let dst_width = (img.width() as f32 * scale) as u32;
            let resized = imageops::resize(
                img,
                dst_width,
                self.dst_height,
                imageops::FilterType::Lanczos3,
            );

            let input_array = image_utils::subtract_mean_normalize(
                &resized,
                &self.mean_values,
                &self.norm_values,
            );

            let shape = input_array.shape().to_vec();
            let (data, _offset) = input_array.into_raw_vec_and_offset();
            let input_value = Value::from_array((shape.as_slice(), data)).map_err(|e| {
                InferenceError::PreprocessingError {
                    operation: "create input value".to_string(),
                    message: e.to_string(),
                }
            })?;

            let (output_data, output_shape) = {
                let outputs = self
                    .session
                    .run(inputs!["x" => input_value])
                    .map_err(|source| InferenceError::ModelExecutionError {
                        operation: "CrnnNet forward pass".to_string(),
                        source,
                    })?;

                let output_tensor = outputs
                    .get("fetch_name_0")
                    .ok_or_else(|| InferenceError::PredictionError {
                        operation: "get model outputs".to_string(),
                        message: "Output 'fetch_name_0' not found".to_string(),
                    })?
                    .try_extract_tensor::<f32>()
                    .map_err(|source| InferenceError::PredictionError {
                        operation: "extract output tensor".to_string(),
                        message: source.to_string(),
                    })?;

                (output_tensor.1.to_vec(), output_tensor.0.clone())
            };

            let seq_len = output_shape[1] as usize;
            let vocab_size = output_shape[2] as usize;

            let (text, score, words, new_offset) = self.score_to_text(
                &output_data,
                seq_len,
                vocab_size,
                &text_boxes[i],
                current_offset,
                directionality,
            )?;

            text_boxes[i].text = Some(text.clone());
            text_boxes[i].text_score = score;
            text_boxes[i].span = Some(crate::document::text_box::DocumentSpan::new(
                current_offset,
                text.len(),
            ));

            current_offset = new_offset;
            all_words.extend(words);
        }

        Ok(all_words)
    }

    /// Converts model output scores to recognized text with word segmentation.
    ///
    /// Implements CTC (Connectionist Temporal Classification) decoding to convert
    /// the raw model output probabilities into text. Also segments the text into
    /// individual words and calculates their bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `output_data` - Raw model output probabilities (flattened)
    /// * `h` - Height dimension of the output (sequence length)
    /// * `w` - Width dimension of the output (vocabulary size)
    /// * `text_box` - Parent text box for calculating word positions
    /// * `global_offset` - Starting character offset for document span tracking
    /// * `directionality` - Text direction (LTR or RTL) affecting word bounds calculation
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `String` - The full recognized text
    /// - `f32` - Average confidence score for the recognition
    /// - `Vec<TextBox>` - Word-level text boxes with positions and scores
    /// - `usize` - Updated global offset after processing
    fn score_to_text(
        &self,
        output_data: &[f32],
        h: usize,
        w: usize,
        text_box: &TextBox,
        global_offset: usize,
        directionality: Directionality,
    ) -> Result<(String, f32, Vec<TextBox>, usize), InferenceError> {
        let mut str_res = String::new();
        let mut scores = Vec::new();
        let mut last_index = 0;

        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut word_scores = Vec::new();
        let mut word_start_pos: Option<usize> = None;
        let mut global_offset = global_offset;

        for i in 0..h {
            let start = i * w;
            let stop = (i + 1) * w;
            let stop = if stop > output_data.len() - 1 {
                output_data.len() - 1
            } else {
                stop
            };

            let (max_index, max_value) = output_data[start..stop]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap_or((0, &0.0));

            if max_index > 0 && max_index < self.keys.len() && !(i > 0 && max_index == last_index) {
                scores.push(*max_value);
                let char_str = &self.keys[max_index];
                str_res.push_str(char_str);

                if &**char_str == " " {
                    let mut processed = false;
                    if !current_word.is_empty() {
                        if let Some(start_pos) = word_start_pos {
                            let word_bounds = self.calculate_word_bounds(
                                text_box,
                                start_pos,
                                i - 1,
                                h,
                                directionality,
                            );

                            let word_score = if word_scores.is_empty() {
                                0.0
                            } else {
                                word_scores.iter().sum::<f32>() / word_scores.len() as f32
                            };

                            let word_len = current_word.len();
                            words.push(TextBox {
                                text: Some(current_word.clone()),
                                bounds: Bounds::new(word_bounds),
                                angle: text_box.angle,
                                box_score: word_score,
                                text_score: word_score,
                                span: Some(crate::document::text_box::DocumentSpan::new(
                                    global_offset,
                                    word_len,
                                )),
                            });

                            global_offset += word_len + 1;
                            current_word.clear();
                            word_scores.clear();
                            word_start_pos = None;
                            processed = true;
                        }
                    }

                    if !processed {
                        global_offset += 1;
                    }
                } else {
                    if word_start_pos.is_none() {
                        word_start_pos = Some(i);
                    }
                    current_word.push_str(char_str);
                    word_scores.push(*max_value);
                }
            }

            last_index = max_index;
        }

        if !current_word.is_empty() {
            if let Some(start_pos) = word_start_pos {
                let word_bounds =
                    self.calculate_word_bounds(text_box, start_pos, h - 1, h, directionality);

                let word_score = if word_scores.is_empty() {
                    0.0
                } else {
                    word_scores.iter().sum::<f32>() / word_scores.len() as f32
                };

                let word_len = current_word.len();
                words.push(TextBox {
                    text: Some(current_word),
                    bounds: Bounds::new(word_bounds),
                    angle: text_box.angle,
                    box_score: word_score,
                    text_score: word_score,
                    span: Some(crate::document::text_box::DocumentSpan::new(
                        global_offset,
                        word_len,
                    )),
                });
                global_offset += word_len;
            }
        }

        let average_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        };

        Ok((str_res, average_score, words, global_offset))
    }

    /// Calculates the bounding box coordinates for a word within a text line.
    ///
    /// Given a text line's bounding box and a word's position within the CTC output
    /// sequence, computes the approximate bounding box for that word using linear
    /// interpolation along the text line. For RTL text, positions are calculated
    /// from right to left.
    ///
    /// # Arguments
    ///
    /// * `text_box` - The parent text line's bounding box
    /// * `start_pos` - Starting position in the output sequence
    /// * `end_pos` - Ending position in the output sequence
    /// * `total_length` - Total length of the output sequence
    /// * `directionality` - Text direction (LTR or RTL)
    ///
    /// # Returns
    ///
    /// Four corner coordinates `[top_left, top_right, bottom_right, bottom_left]`
    /// representing the word's bounding box.
    fn calculate_word_bounds(
        &self,
        text_box: &TextBox,
        start_pos: usize,
        end_pos: usize,
        total_length: usize,
        directionality: Directionality,
    ) -> [Coord<i32>; 4] {
        let top_left = text_box.bounds[0];
        let top_right = text_box.bounds[1];
        let bottom_right = text_box.bounds[2];
        let bottom_left = text_box.bounds[3];

        let (left_ratio, right_ratio) = match directionality {
            Directionality::Ltr => {
                let start_ratio = start_pos as f32 / total_length as f32;
                let end_ratio = (end_pos + 1) as f32 / total_length as f32;
                (start_ratio, end_ratio)
            }
            Directionality::Rtl => {
                let start_ratio = 1.0 - ((end_pos + 1) as f32 / total_length as f32);
                let end_ratio = 1.0 - (start_pos as f32 / total_length as f32);
                (start_ratio, end_ratio)
            }
        };

        let word_top_left = Coord {
            x: top_left.x + ((top_right.x - top_left.x) as f32 * left_ratio) as i32,
            y: top_left.y + ((top_right.y - top_left.y) as f32 * left_ratio) as i32,
        };

        let word_top_right = Coord {
            x: top_left.x + ((top_right.x - top_left.x) as f32 * right_ratio) as i32,
            y: top_left.y + ((top_right.y - top_left.y) as f32 * right_ratio) as i32,
        };

        let word_bottom_right = Coord {
            x: bottom_left.x + ((bottom_right.x - bottom_left.x) as f32 * right_ratio) as i32,
            y: bottom_left.y + ((bottom_right.y - bottom_left.y) as f32 * right_ratio) as i32,
        };

        let word_bottom_left = Coord {
            x: bottom_left.x + ((bottom_right.x - bottom_left.x) as f32 * left_ratio) as i32,
            y: bottom_left.y + ((bottom_right.y - bottom_left.y) as f32 * left_ratio) as i32,
        };

        [
            word_top_left,
            word_top_right,
            word_bottom_right,
            word_bottom_left,
        ]
    }
}
