use std::fs::File;
use std::io::{BufRead, BufReader};

use image::{imageops, RgbImage};
use ort::{inputs, session::Session, value::Value};

use crate::document::text_box::TextBox;
use crate::inference::error::InferenceError;
use crate::utils::image_utils;
use crate::utils::lang_utils::LangUtils;
use geo::Coord;

pub struct Crnn {
    session: Session,
    keys: Vec<String>,
    mean_values: [f32; 3],
    norm_values: [f32; 3],
    dst_height: u32,
}

impl Crnn {
    const NUM_THREADS: usize = 4;

    fn insert_special_characters(keys: &mut Vec<String>) {
        keys.insert(0, "#".to_string());
        keys.push(" ".to_string());
    }

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
            .with_execution_providers([
                ort::execution_providers::TensorRTExecutionProvider::default()
                    .with_device_id(0)
                    .with_engine_cache(true)
                    .with_engine_cache_path("/workspaces/DocYouMeant/models/trt_engines")
                    .with_engine_cache_prefix("docyoumeant_")
                    .with_max_workspace_size(5 << 30)
                    .with_fp16(true)
                    .with_timing_cache(true)
                    .build(),
            ])?
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

    pub fn get_text(
        &mut self,
        src: &RgbImage,
        text_box: &mut TextBox,
        global_offset: usize,
    ) -> Result<(Vec<TextBox>, usize), InferenceError> {
        let scale = self.dst_height as f32 / src.height() as f32;
        let dst_width = (src.width() as f32 * scale) as u32;

        let src_resize = imageops::resize(
            src,
            dst_width,
            self.dst_height,
            imageops::FilterType::Lanczos3,
        );

        let input_array =
            image_utils::subtract_mean_normalize(&src_resize, &self.mean_values, &self.norm_values)
                .map_err(|e| InferenceError::PreprocessingError {
                    operation: "normalize image".to_string(),
                    message: e.to_string(),
                })?;

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

        let h = output_shape[1] as usize;
        let w = output_shape[2] as usize;

        let (text, score, words, new_offset) =
            self.score_to_text(&output_data, h, w, text_box, global_offset)?;
        text_box.text = Some(text);
        text_box.text_score = score;

        Ok((words, new_offset))
    }

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
                            bounds: word_bounds,
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
                    bounds: word_bounds,
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

    pub fn get_texts(
        &mut self,
        part_imgs: &[RgbImage],
        text_boxes: &mut [TextBox],
    ) -> Result<Vec<TextBox>, InferenceError> {
        let mut all_words = Vec::new();
        let mut current_offset = 0;

        for (i, img) in part_imgs.iter().enumerate() {
            if i < text_boxes.len() {
                let (words, new_offset) = self.get_text(img, &mut text_boxes[i], current_offset)?;
                if let Some(ref text) = text_boxes[i].text {
                    let length = text.len();
                    text_boxes[i].span = Some(crate::document::text_box::DocumentSpan::new(
                        current_offset,
                        length,
                    ));
                }
                current_offset = new_offset;
                all_words.extend(words);
            }
        }

        Ok(all_words)
    }
}
