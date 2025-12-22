//! Convolutional Recurrent Neural Network (CRNN) for text recognition.
//!
//! This module provides text recognition capabilities using a CRNN architecture
//! that combines convolutional layers for feature extraction with recurrent layers
//! for sequence modeling. It supports multiple languages through language-specific
//! models and dictionaries.

use std::fs::File;
use std::io::{BufRead, BufReader};

use image::{imageops, Rgb, RgbImage};
use ndarray::{Array4, Axis};
use ort::{inputs, session::Session, value::Value};

use crate::document::bounds::Bounds;
use crate::document::text_box::TextBox;
use crate::inference::error::InferenceError;
use crate::utils::image_utils;
use crate::utils::lang_utils::LangUtils;
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
/// Unlike other inference modules, `Crnn` is not a singleton. Each language
/// requires its own instance with the appropriate model and dictionary.
/// For multi-threaded usage, create separate instances or wrap in `Arc<Mutex<Crnn>>`.
pub struct Crnn {
    session: Session,
    keys: Vec<String>,
    mean_values: [f32; 3],
    norm_values: [f32; 3],
    dst_height: u32,
}

impl Crnn {
    /// Number of threads for ONNX Runtime inter-op parallelism.
    const NUM_THREADS: usize = 4;

    /// Padding color for batched images (gray, matching normalization mean).
    const PAD_COLOR: Rgb<u8> = Rgb([127, 127, 127]);

    /// Inserts special characters into the character dictionary.
    ///
    /// Adds the CTC blank token ("#") at the beginning and a space character
    /// at the end of the dictionary, as required by the CTC decoding algorithm.
    fn insert_special_characters(keys: &mut Vec<String>) {
        keys.insert(0, "#".to_string());
        keys.push(" ".to_string());
    }

    /// Creates a new CRNN text recognizer for the specified language.
    ///
    /// Loads the appropriate ONNX model and character dictionary based on
    /// the language configuration from `models/ocr_lang_models.json`.
    ///
    /// # Arguments
    ///
    /// * `language` - Language identifier (e.g., "english", "chinese", "arabic")
    ///
    /// # Returns
    ///
    /// * `Ok(Crnn)` - Initialized recognizer ready for inference
    /// * `Err(InferenceError)` - If the language is unsupported or model files cannot be loaded
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The specified language is not supported
    /// - The model file cannot be loaded
    /// - The dictionary file cannot be read or parsed
    pub fn new(language: &str) -> Result<Self, InferenceError> {
        let config = LangUtils::get_language_config(language).ok_or_else(|| {
            InferenceError::ModelFileLoadError {
                path: format!("Unsupported language: {language}").into(),
                source: ort::Error::new(format!("Unsupported language: {language}")),
            }
        })?;

        let session = Session::builder()
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: config.model_file.clone().into(),
                source,
            })?
            .with_inter_threads(Self::NUM_THREADS)?
            .commit_from_file(&config.model_file)
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: config.model_file.clone().into(),
                source,
            })?;

        let mut keys = Vec::new();
        let file =
            File::open(&config.dict_file).map_err(|source| InferenceError::DataFileLoadError {
                path: config.dict_file.clone().into(),
                source,
            })?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            keys.push(line.map_err(|source| InferenceError::DataFileParseError {
                path: config.dict_file.clone().into(),
                source,
            })?);
        }

        Self::insert_special_characters(&mut keys);

        Ok(Self {
            session,
            keys,
            mean_values: [127.5, 127.5, 127.5],
            norm_values: [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5],
            dst_height: 48,
        })
    }

    /// Pads an image to the target width with the padding color.
    ///
    /// Creates a new image of the target dimensions and copies the source
    /// image into the left portion, leaving the right side padded.
    ///
    /// # Arguments
    ///
    /// * `src` - Source image to pad
    /// * `target_width` - Desired width of the output image
    ///
    /// # Returns
    ///
    /// A new image with the source content left-aligned and padding on the right.
    fn pad_image(src: &RgbImage, target_width: u32) -> RgbImage {
        if src.width() >= target_width {
            return src.clone();
        }

        let mut padded = RgbImage::from_pixel(target_width, src.height(), Self::PAD_COLOR);
        imageops::replace(&mut padded, src, 0, 0);
        padded
    }

    /// Recognizes text from a single text line image.
    ///
    /// Performs text recognition on a cropped image containing a single line of text.
    /// Updates the provided `TextBox` with recognized text and confidence score,
    /// and returns individual word bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `src` - RGB image of the text line (will be resized internally)
    /// * `text_box` - Text box to update with recognition results
    /// * `global_offset` - Character offset in the document for span calculation
    ///
    /// # Returns
    ///
    /// * `Ok((words, new_offset))` - Vector of word-level TextBoxes and updated global offset
    /// * `Err(InferenceError)` - If recognition fails
    pub fn get_text(
        &mut self,
        src: &RgbImage,
        text_box: &mut TextBox,
        global_offset: usize,
    ) -> Result<(Vec<TextBox>, usize), InferenceError> {
        let images = vec![src.clone()];
        let mut text_boxes = vec![text_box.clone()];

        let words = self.get_texts_internal(&images, &mut text_boxes, global_offset)?;

        // Copy results back to the original text_box
        if let Some(result) = text_boxes.first() {
            text_box.text = result.text.clone();
            text_box.text_score = result.text_score;
            text_box.span = result.span;
        }

        let new_offset = text_boxes
            .first()
            .and_then(|tb| tb.span.as_ref())
            .map(|span| span.offset + span.length)
            .unwrap_or(global_offset);

        Ok((words, new_offset))
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
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &0.0));

            if max_index > 0 && max_index < self.keys.len() && !(i > 0 && max_index == last_index) {
                scores.push(*max_value);
                let char_str = &self.keys[max_index];
                str_res.push_str(char_str);

                if char_str == " " {
                    if !current_word.is_empty() && word_start_pos.is_some() {
                        let word_bounds =
                            self.calculate_word_bounds(text_box, word_start_pos.unwrap(), i - 1, h);

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
                    } else {
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
                let word_bounds = self.calculate_word_bounds(text_box, start_pos, h - 1, h);

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
    /// interpolation along the text line.
    ///
    /// # Arguments
    ///
    /// * `text_box` - The parent text line's bounding box
    /// * `start_pos` - Starting position in the output sequence
    /// * `end_pos` - Ending position in the output sequence
    /// * `total_length` - Total length of the output sequence
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
    ) -> [Coord<i32>; 4] {
        let start_ratio = start_pos as f32 / total_length as f32;
        let end_ratio = (end_pos + 1) as f32 / total_length as f32;

        let top_left = text_box.bounds[0];
        let top_right = text_box.bounds[1];
        let bottom_right = text_box.bounds[2];
        let bottom_left = text_box.bounds[3];

        let word_top_left = Coord {
            x: top_left.x + ((top_right.x - top_left.x) as f32 * start_ratio) as i32,
            y: top_left.y + ((top_right.y - top_left.y) as f32 * start_ratio) as i32,
        };

        let word_top_right = Coord {
            x: top_left.x + ((top_right.x - top_left.x) as f32 * end_ratio) as i32,
            y: top_left.y + ((top_right.y - top_left.y) as f32 * end_ratio) as i32,
        };

        let word_bottom_right = Coord {
            x: bottom_left.x + ((bottom_right.x - bottom_left.x) as f32 * end_ratio) as i32,
            y: bottom_left.y + ((bottom_right.y - bottom_left.y) as f32 * end_ratio) as i32,
        };

        let word_bottom_left = Coord {
            x: bottom_left.x + ((bottom_right.x - bottom_left.x) as f32 * start_ratio) as i32,
            y: bottom_left.y + ((bottom_right.y - bottom_left.y) as f32 * start_ratio) as i32,
        };

        [
            word_top_left,
            word_top_right,
            word_bottom_right,
            word_bottom_left,
        ]
    }

    /// Recognizes text from multiple text line images in batch.
    ///
    /// Processes multiple cropped text line images in a single batched forward pass
    /// Images are padded to uniform width before batching.
    /// Maintains a consistent global character offset across all lines for
    /// document-wide span tracking.
    ///
    /// # Arguments
    ///
    /// * `part_imgs` - Slice of RGB images, one per text line
    /// * `text_boxes` - Mutable slice of text boxes to update with results
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TextBox>)` - All word-level text boxes from all lines
    /// * `Err(InferenceError)` - If any recognition fails
    pub fn get_texts(
        &mut self,
        part_imgs: &[RgbImage],
        text_boxes: &mut [TextBox],
    ) -> Result<Vec<TextBox>, InferenceError> {
        self.get_texts_internal(part_imgs, text_boxes, 0)
    }

    /// Internal batched text recognition implementation.
    ///
    /// Processes all images in a single forward pass by:
    /// 1. Resizing all images to the target height while preserving aspect ratio
    /// 2. Padding all images to the maximum width in the batch
    /// 3. Stacking into a single batch tensor
    /// 4. Running a single forward pass through the model
    /// 5. Decoding each result with CTC and calculating word boundaries
    ///
    /// # Arguments
    ///
    /// * `part_imgs` - Slice of RGB images, one per text line
    /// * `text_boxes` - Mutable slice of text boxes to update with results
    /// * `initial_offset` - Starting character offset for span tracking
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TextBox>)` - All word-level text boxes from all lines
    /// * `Err(InferenceError)` - If recognition fails
    fn get_texts_internal(
        &mut self,
        part_imgs: &[RgbImage],
        text_boxes: &mut [TextBox],
        initial_offset: usize,
    ) -> Result<Vec<TextBox>, InferenceError> {
        if part_imgs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = part_imgs.len().min(text_boxes.len());
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let resized_images: Vec<RgbImage> = part_imgs
            .iter()
            .take(batch_size)
            .map(|img| {
                let scale = self.dst_height as f32 / img.height() as f32;
                let dst_width = (img.width() as f32 * scale) as u32;
                imageops::resize(
                    img,
                    dst_width,
                    self.dst_height,
                    imageops::FilterType::Lanczos3,
                )
            })
            .collect();

        let original_widths: Vec<u32> = resized_images.iter().map(|img| img.width()).collect();
        let max_width = original_widths.iter().copied().max().unwrap_or(1);

        let normalized_arrays: Vec<_> = resized_images
            .iter()
            .map(|img| {
                let padded = Self::pad_image(img, max_width);
                image_utils::subtract_mean_normalize(&padded, &self.mean_values, &self.norm_values)
            })
            .collect();

        let batch_array = Self::stack_arrays(&normalized_arrays)?;

        let shape = batch_array.shape().to_vec();
        let (data, _offset) = batch_array.into_raw_vec_and_offset();
        let input_value = Value::from_array((shape.as_slice(), data)).map_err(|e| {
            InferenceError::PreprocessingError {
                operation: "create batched input value".to_string(),
                message: e.to_string(),
            }
        })?;

        let (output_data, output_shape) = {
            let outputs = self
                .session
                .run(inputs!["x" => input_value])
                .map_err(|source| InferenceError::ModelExecutionError {
                    operation: "CrnnNet batched forward pass".to_string(),
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
        let output_batch_size = output_shape[0] as usize;

        let mut all_words = Vec::new();
        let mut current_offset = initial_offset;

        for i in 0..output_batch_size.min(batch_size) {
            let item_start = i * seq_len * vocab_size;
            let item_end = (i + 1) * seq_len * vocab_size;
            let item_output = &output_data[item_start..item_end];

            let width_ratio = original_widths[i] as f32 / max_width as f32;
            let effective_seq_len = ((seq_len as f32) * width_ratio).ceil() as usize;

            let (text, score, words, new_offset) = self.score_to_text(
                item_output,
                effective_seq_len,
                vocab_size,
                &text_boxes[i],
                current_offset,
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

    /// Stacks multiple 3D arrays into a single 4D batch array.
    ///
    /// Takes normalized image arrays of shape [C, H, W] and stacks them
    /// into a batch tensor of shape [N, C, H, W].
    ///
    /// # Arguments
    ///
    /// * `arrays` - Slice of 4D arrays (each [1, C, H, W]) to concatenate
    ///
    /// # Returns
    ///
    /// * `Ok(Array4<f32>)` - Concatenated 4D array [N, C, H, W]
    /// * `Err(InferenceError)` - If arrays have inconsistent shapes
    fn stack_arrays(arrays: &[Array4<f32>]) -> Result<Array4<f32>, InferenceError> {
        if arrays.is_empty() {
            return Err(InferenceError::PreprocessingError {
                operation: "stack arrays".to_string(),
                message: "Cannot stack empty array list".to_string(),
            });
        }

        let shape = arrays[0].shape();
        let (c, h, w) = (shape[1], shape[2], shape[3]);
        let batch_size = arrays.len();

        let mut batch = Array4::<f32>::zeros((batch_size, c, h, w));

        for (i, arr) in arrays.iter().enumerate() {
            let view = arr.index_axis(Axis(0), 0);
            batch.index_axis_mut(Axis(0), i).assign(&view);
        }

        Ok(batch)
    }
}
