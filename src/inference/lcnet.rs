use image::{imageops, Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use once_cell::sync::OnceCell;
use ort::{inputs, session::Session, value::Value};
use std::sync::Mutex;

use crate::document::text_box::Orientation;
use crate::inference::error::InferenceError;
use crate::utils::image_utils;

static TEXT_ORIENTATION_INSTANCE: OnceCell<Mutex<LCNet>> = OnceCell::new();
static DOCUMENT_ORIENTATION_INSTANCE: OnceCell<Mutex<LCNet>> = OnceCell::new();

pub struct LCNet {
    session: Session,
    mean_values: [f32; 3],
    norm_values: [f32; 3],
    dst_height: u32,
    dst_width: u32,
    is_text_orientation: bool,
}

impl LCNet {
    const TEXT_ORIENTATION_MODEL_PATH: &'static str =
        "models/onnx/text_orientation_classification.onnx";
    const DOCUMENT_ORIENTATION_MODEL_PATH: &'static str =
        "models/onnx/document_orientation_classification.onnx";
    const NUM_THREADS: usize = 4;

    pub fn new(is_text_orientation: bool) -> Result<Self, InferenceError> {
        let model_path = if is_text_orientation {
            Self::TEXT_ORIENTATION_MODEL_PATH
        } else {
            Self::DOCUMENT_ORIENTATION_MODEL_PATH
        };

        let session = Session::builder()
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: model_path.into(),
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
            .commit_from_file(model_path)
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: model_path.into(),
                source,
            })?;

        let (dst_height, dst_width) = if is_text_orientation {
            (80, 160) // Text orientation dimensions
        } else {
            (224, 224) // Document orientation dimensions
        };

        Ok(Self {
            session,
            mean_values: [127.5, 127.5, 127.5],
            norm_values: [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5],
            dst_height,
            dst_width,
            is_text_orientation,
        })
    }

    pub fn get_or_init(is_text_orientation: bool) -> Result<(), InferenceError> {
        if is_text_orientation {
            TEXT_ORIENTATION_INSTANCE.get_or_try_init(|| Self::new(true).map(Mutex::new))?;
        } else {
            DOCUMENT_ORIENTATION_INSTANCE.get_or_try_init(|| Self::new(false).map(Mutex::new))?;
        }
        Ok(())
    }

    fn instance(is_text_orientation: bool) -> Result<&'static Mutex<LCNet>, InferenceError> {
        if is_text_orientation {
            TEXT_ORIENTATION_INSTANCE.get_or_try_init(|| Self::new(true).map(Mutex::new))
        } else {
            DOCUMENT_ORIENTATION_INSTANCE.get_or_try_init(|| Self::new(false).map(Mutex::new))
        }
    }

    fn apply_most_angle(angles: &mut [Orientation]) -> Result<(), InferenceError> {
        if angles.is_empty() {
            return Ok(());
        }

        let mut horizontal_counts = [0; 2]; // [Angle0, Angle180]
        let mut vertical_counts = [0; 2]; // [Angle90, Angle270]

        for angle in angles.iter() {
            match angle {
                Orientation::Oriented0 => horizontal_counts[0] += 1,
                Orientation::Oriented180 => horizontal_counts[1] += 1,
                Orientation::Oriented90 => vertical_counts[0] += 1,
                Orientation::Oriented270 => vertical_counts[1] += 1,
            }
        }

        let most_common_horizontal = if horizontal_counts[0] >= horizontal_counts[1] {
            Orientation::Oriented0
        } else {
            Orientation::Oriented180
        };

        let most_common_vertical = if vertical_counts[0] >= vertical_counts[1] {
            Orientation::Oriented90
        } else {
            Orientation::Oriented270
        };

        for angle in angles.iter_mut() {
            match angle {
                Orientation::Oriented0 | Orientation::Oriented180 => {
                    *angle = most_common_horizontal
                }
                Orientation::Oriented90 | Orientation::Oriented270 => *angle = most_common_vertical,
            }
        }

        Ok(())
    }

    pub fn infer_angle(
        &mut self,
        src: &RgbImage,
        is_vertical: bool,
    ) -> Result<Orientation, InferenceError> {
        let input_array =
            image_utils::subtract_mean_normalize(src, &self.mean_values, &self.norm_values)
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

        let outputs = self.session.run(inputs!["x" => input_value]).map_err(|e| {
            println!("Error running the session: {e:?}");
            InferenceError::ModelExecutionError {
                operation: "AngleNet forward pass".to_string(),
                source: e,
            }
        })?;

        let output = outputs
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

        let output_data = output.1;

        if self.is_text_orientation {
            // Text orientation: 2-class output (0 vs 180 degrees)
            let score_0 = output_data[0];
            let score_1 = output_data[1];

            let base_angle = if score_0 > score_1 {
                Orientation::Oriented0
            } else {
                Orientation::Oriented180
            };

            if is_vertical {
                match base_angle {
                    Orientation::Oriented0 => Ok(Orientation::Oriented90),
                    Orientation::Oriented180 => Ok(Orientation::Oriented270),
                    _ => Ok(Orientation::Oriented90),
                }
            } else {
                Ok(base_angle)
            }
        } else {
            // Document orientation: 4-class output (0, 90, 180, 270 degrees)
            let score_0 = output_data[0];
            let score_90 = output_data[1];
            let score_180 = output_data[2];
            let score_270 = output_data[3];

            let scores = [score_0, score_90, score_180, score_270];
            let max_index = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap_or(0);

            match max_index {
                0 => Ok(Orientation::Oriented0),
                1 => Ok(Orientation::Oriented90),
                2 => Ok(Orientation::Oriented180),
                3 => Ok(Orientation::Oriented270),
                _ => Ok(Orientation::Oriented0),
            }
        }
    }

    pub fn get_angles(
        part_imgs: &[RgbImage],
        is_text_orientation: bool,
        most_angle: bool,
    ) -> Result<Vec<Orientation>, InferenceError> {
        let instance = Self::instance(is_text_orientation)?;
        let mut model = instance
            .lock()
            .map_err(|e| InferenceError::ProcessingError {
                message: format!("Failed to lock LCNet instance: {e}"),
            })?;
        let size = part_imgs.len();
        let mut angles = Vec::with_capacity(size);

        for (i, part_img) in part_imgs.iter().enumerate() {
            if part_img.width() == 0 || part_img.height() == 0 {
                println!("Warning: Empty part image at index {i}, skipping");
                angles.push(Orientation::Oriented0);
                continue;
            }

            if part_img.width() < 2 || part_img.height() < 2 {
                println!(
                    "Warning: Part image at index {} is too small ({}x{}), skipping",
                    i,
                    part_img.width(),
                    part_img.height()
                );
                angles.push(Orientation::Oriented0);
                continue;
            }

            let angle = match model.preprocess(part_img) {
                Ok((resized_img, is_vertical)) => {
                    match model.infer_angle(&resized_img, is_vertical) {
                        Ok(result) => result,
                        Err(e) => {
                            println!("Error inferring angle for image {i}: {e:?}");
                            angles.push(Orientation::Oriented0);
                            continue;
                        }
                    }
                }
                Err(e) => {
                    println!("Error preprocessing image {i}: {e:?}");
                    angles.push(Orientation::Oriented0);
                    continue;
                }
            };

            angles.push(angle);
        }

        if most_angle {
            Self::apply_most_angle(&mut angles)?;
        }

        Ok(angles)
    }

    pub fn preprocess(&self, src: &RgbImage) -> Result<(RgbImage, bool), InferenceError> {
        let is_vertical = (src.height() as f32) >= (src.width() as f32) * 1.5;

        let processed_img = if self.is_text_orientation && is_vertical {
            rotate_about_center(
                src,
                -std::f32::consts::PI / 2.0,
                Interpolation::Bilinear,
                Rgb([255u8, 255u8, 255u8]),
            )
        } else {
            src.clone()
        };

        let resized_img = imageops::resize(
            &processed_img,
            self.dst_width,
            self.dst_height,
            imageops::FilterType::Lanczos3,
        );

        Ok((resized_img, is_vertical))
    }

    pub fn run(
        part_imgs: &[RgbImage],
        is_text_orientation: bool,
    ) -> Result<Vec<Orientation>, InferenceError> {
        Self::get_angles(part_imgs, is_text_orientation, true)
    }
}
