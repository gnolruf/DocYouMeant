//! LCNet-based classification models for orientation and table type detection.
//!
//! This module provides lightweight classification models based on the LCNet architecture
//! for three distinct tasks:
//!
//! - **Text Orientation**: Classifies individual text line images as 0° or 180° rotated
//! - **Document Orientation**: Classifies full document images (0°, 90°, 180°, or 270°)
//! - **Table Classification**: Determines if a table has visible borders (wired) or not (wireless)

use image::{imageops, Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use once_cell::sync::OnceCell;
use ort::{inputs, session::Session, value::Value};
use std::sync::Mutex;

use crate::document::table::TableType;
use crate::document::text_box::Orientation;
use crate::inference::error::InferenceError;
use crate::utils::image_utils;

static TEXT_ORIENTATION_INSTANCE: OnceCell<Mutex<LCNet>> = OnceCell::new();
static DOCUMENT_ORIENTATION_INSTANCE: OnceCell<Mutex<LCNet>> = OnceCell::new();
static TABLE_CLASSIFICATION_INSTANCE: OnceCell<Mutex<LCNet>> = OnceCell::new();

/// Operating mode for the LCNet classifier.
///
/// Determines which model variant is loaded and what type of classification
/// is performed. Each mode uses a separate singleton instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LCNetMode {
    /// Classifies the orientation of individual text lines (0° or 180°)
    TextOrientation,
    /// Classifies the orientation of entire documents (0°, 90°, 180°, or 270°)
    DocumentOrientation,
    /// Classifies tables as wired (with borders) or wireless (without borders)
    TableType,
}

/// Result type returned by LCNet classification.
///
/// The variant matches the mode used for inference:
/// - `TextOrientation` and `DocumentOrientation` return `Orientations`
/// - `TableType` returns `TableTypes`
#[derive(Debug, Clone)]
pub enum LCNetResult {
    Orientations(Vec<Orientation>),
    TableTypes(Vec<TableType>),
}

/// Lightweight classification network for orientation and table type detection.
///
/// `LCNet` provides fast image classification using efficient CNN architectures.
/// It supports three operating modes, each with its own model weights and
/// input preprocessing requirements.
///
/// # Fields
///
/// - `session`: ONNX Runtime session for model inference
/// - `mean_values`: Per-channel mean values for normalization (127.5 for all channels)
/// - `norm_values`: Per-channel normalization divisors
/// - `dst_height`: Target input height (80 for text, 224 for document/table)
/// - `dst_width`: Target input width (160 for text, 224 for document/table)
/// - `mode`: Current operating mode determining the classification task
///
/// # Thread Safety
///
/// Each mode maintains a separate singleton wrapped in `Mutex`, allowing
/// thread-safe access while permitting different modes to run concurrently.
pub struct LCNet {
    session: Session,
    mean_values: [f32; 3],
    norm_values: [f32; 3],
    dst_height: u32,
    dst_width: u32,
    mode: LCNetMode,
}

impl LCNet {
    /// Path to the text orientation classification model.
    const TEXT_ORIENTATION_MODEL_PATH: &'static str =
        "models/onnx/text_orientation_classification.onnx";
    /// Path to the document orientation classification model.
    const DOCUMENT_ORIENTATION_MODEL_PATH: &'static str =
        "models/onnx/document_orientation_classification.onnx";
    /// Path to the table type classification model.
    const TABLE_CLASSIFICATION_MODEL_PATH: &'static str = "models/onnx/table_classification.onnx";
    /// Number of threads for ONNX Runtime inter-op parallelism.
    const NUM_THREADS: usize = 4;

    /// Creates a new LCNet instance for the specified mode.
    ///
    /// Loads the appropriate ONNX model based on the mode and configures
    /// preprocessing parameters. This method is typically called internally
    /// by the singleton initialization.
    ///
    /// # Arguments
    ///
    /// * `mode` - The classification mode determining which model to load
    ///
    /// # Returns
    ///
    /// * `Ok(LCNet)` - Initialized classifier ready for inference
    /// * `Err(InferenceError)` - If the model file cannot be loaded
    pub fn new(mode: LCNetMode) -> Result<Self, InferenceError> {
        let model_path = match mode {
            LCNetMode::TextOrientation => Self::TEXT_ORIENTATION_MODEL_PATH,
            LCNetMode::DocumentOrientation => Self::DOCUMENT_ORIENTATION_MODEL_PATH,
            LCNetMode::TableType => Self::TABLE_CLASSIFICATION_MODEL_PATH,
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

        let (dst_height, dst_width) = match mode {
            LCNetMode::TextOrientation => (80, 160),
            LCNetMode::DocumentOrientation | LCNetMode::TableType => (224, 224),
        };

        Ok(Self {
            session,
            mean_values: [127.5, 127.5, 127.5],
            norm_values: [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5],
            dst_height,
            dst_width,
            mode,
        })
    }

    /// Pre-initializes the LCNet singleton for the specified mode.
    ///
    /// Call this method during application startup to eagerly load models
    /// rather than waiting for the first classification request.
    ///
    /// # Arguments
    ///
    /// * `mode` - The classification mode to initialize
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Model successfully initialized
    /// * `Err(InferenceError)` - If initialization fails
    pub fn get_or_init(mode: LCNetMode) -> Result<(), InferenceError> {
        match mode {
            LCNetMode::TextOrientation => {
                TEXT_ORIENTATION_INSTANCE
                    .get_or_try_init(|| Self::new(LCNetMode::TextOrientation).map(Mutex::new))?;
            }
            LCNetMode::DocumentOrientation => {
                DOCUMENT_ORIENTATION_INSTANCE.get_or_try_init(|| {
                    Self::new(LCNetMode::DocumentOrientation).map(Mutex::new)
                })?;
            }
            LCNetMode::TableType => {
                TABLE_CLASSIFICATION_INSTANCE
                    .get_or_try_init(|| Self::new(LCNetMode::TableType).map(Mutex::new))?;
            }
        }
        Ok(())
    }

    /// Returns a reference to the singleton instance for the specified mode.
    ///
    /// Initializes the instance if it hasn't been created yet.
    fn instance(mode: LCNetMode) -> Result<&'static Mutex<LCNet>, InferenceError> {
        match mode {
            LCNetMode::TextOrientation => TEXT_ORIENTATION_INSTANCE
                .get_or_try_init(|| Self::new(LCNetMode::TextOrientation).map(Mutex::new)),
            LCNetMode::DocumentOrientation => DOCUMENT_ORIENTATION_INSTANCE
                .get_or_try_init(|| Self::new(LCNetMode::DocumentOrientation).map(Mutex::new)),
            LCNetMode::TableType => TABLE_CLASSIFICATION_INSTANCE
                .get_or_try_init(|| Self::new(LCNetMode::TableType).map(Mutex::new)),
        }
    }

    /// Applies majority voting to normalize orientation predictions.
    ///
    /// When processing multiple text lines from the same document, individual
    /// classification errors can occur. This method uses majority voting to
    /// replace each prediction with the most common orientation in its category
    /// (horizontal vs vertical).
    ///
    /// # Arguments
    ///
    /// * `angles` - Mutable slice of orientation predictions to normalize
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Angles successfully normalized
    /// * `Err(InferenceError)` - (Currently never returns an error)
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

    /// Infers the orientation of a single image.
    ///
    /// Runs the classification model on a preprocessed image and interprets
    /// the output logits to determine the orientation.
    ///
    /// # Arguments
    ///
    /// * `src` - Preprocessed RGB image (already resized)
    /// * `is_vertical` - Whether the original image was vertical (height > 1.5× width)
    ///
    /// # Returns
    ///
    /// * `Ok(Orientation)` - Detected orientation (0°, 90°, 180°, or 270°)
    /// * `Err(InferenceError)` - If inference fails
    fn infer_angle(
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

        match self.mode {
            LCNetMode::TextOrientation => {
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
            }
            LCNetMode::DocumentOrientation => {
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
            LCNetMode::TableType => {
                // This branch should not be reached as run() routes to infer_table_type
                Ok(Orientation::Oriented0)
            }
        }
    }

    /// Infers the table type (wired or wireless) for a single image.
    ///
    /// Runs the table classification model to determine whether a table
    /// has visible grid lines/borders.
    ///
    /// # Arguments
    ///
    /// * `src` - Preprocessed RGB image of the table region
    ///
    /// # Returns
    ///
    /// * `Ok(TableType)` - Either `Wired` (with borders) or `Wireless` (borderless)
    /// * `Err(InferenceError)` - If inference fails
    fn infer_table_type(&mut self, src: &RgbImage) -> Result<TableType, InferenceError> {
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
                operation: "TableClassification forward pass".to_string(),
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

        // Table classification: 2-class output (wired vs wireless)
        let score_wired = output_data[0];
        let score_wireless = output_data[1];

        if score_wired > score_wireless {
            Ok(TableType::Wired)
        } else {
            Ok(TableType::Wireless)
        }
    }

    /// Classifies the orientation of multiple images.
    ///
    /// Processes a batch of images through the orientation classifier,
    /// with optional majority voting to smooth out prediction errors.
    ///
    /// # Arguments
    ///
    /// * `part_imgs` - Slice of RGB images to classify
    /// * `mode` - Classification mode (TextOrientation or DocumentOrientation)
    /// * `most_angle` - Whether to apply majority voting after classification
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Orientation>)` - Orientation for each input image
    /// * `Err(InferenceError)` - If classification fails
    fn get_angles(
        part_imgs: &[RgbImage],
        mode: LCNetMode,
        most_angle: bool,
    ) -> Result<Vec<Orientation>, InferenceError> {
        let instance = Self::instance(mode)?;
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

    /// Classifies the type of multiple table images.
    ///
    /// Processes a batch of table region images to determine whether each
    /// table has visible borders (wired) or not (wireless).
    ///
    /// # Arguments
    ///
    /// * `table_imgs` - Slice of RGB images of table regions
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TableType>)` - Classification for each table image
    /// * `Err(InferenceError)` - If classification fails
    fn get_table_types(table_imgs: &[RgbImage]) -> Result<Vec<TableType>, InferenceError> {
        let instance = Self::instance(LCNetMode::TableType)?;
        let mut model = instance
            .lock()
            .map_err(|e| InferenceError::ProcessingError {
                message: format!("Failed to lock LCNet instance: {e}"),
            })?;
        let size = table_imgs.len();
        let mut table_types = Vec::with_capacity(size);

        for (i, table_img) in table_imgs.iter().enumerate() {
            if table_img.width() == 0 || table_img.height() == 0 {
                println!("Warning: Empty table image at index {i}, skipping");
                table_types.push(TableType::Wired); // Default to wired
                continue;
            }

            if table_img.width() < 2 || table_img.height() < 2 {
                println!(
                    "Warning: Table image at index {} is too small ({}x{}), skipping",
                    i,
                    table_img.width(),
                    table_img.height()
                );
                table_types.push(TableType::Wired); // Default to wired
                continue;
            }

            let table_type = match model.preprocess(table_img) {
                Ok((resized_img, _)) => match model.infer_table_type(&resized_img) {
                    Ok(result) => result,
                    Err(e) => {
                        println!("Error inferring table type for image {i}: {e:?}");
                        table_types.push(TableType::Wired);
                        continue;
                    }
                },
                Err(e) => {
                    println!("Error preprocessing table image {i}: {e:?}");
                    table_types.push(TableType::Wired);
                    continue;
                }
            };

            table_types.push(table_type);
        }

        Ok(table_types)
    }

    /// Preprocesses an image for classification.
    ///
    /// Resizes the image to the model's expected dimensions. For text orientation
    /// mode, vertical images (height ≥ 1.5× width) are rotated 90° clockwise
    /// before resizing.
    ///
    /// # Arguments
    ///
    /// * `src` - Input RGB image
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `RgbImage` - Resized (and possibly rotated) image
    /// - `bool` - Whether the original image was vertical
    fn preprocess(&self, src: &RgbImage) -> Result<(RgbImage, bool), InferenceError> {
        let is_vertical = (src.height() as f32) >= (src.width() as f32) * 1.5;

        let processed_img = if self.mode == LCNetMode::TextOrientation && is_vertical {
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

    /// Runs classification on a batch of images (main entry point).
    ///
    /// This is the primary method for LCNet classification. It handles
    /// singleton initialization, preprocessing, and inference based on the
    /// specified mode.
    ///
    /// # Arguments
    ///
    /// * `imgs` - Slice of RGB images to classify
    /// * `mode` - Classification mode determining the task
    /// * `most_angle` - Whether to apply majority voting (only for orientation modes)
    ///
    /// # Returns
    ///
    /// * `Ok(LCNetResult)` - Classification results matching the requested mode
    /// * `Err(InferenceError)` - If classification fails
    pub fn run(
        imgs: &[RgbImage],
        mode: LCNetMode,
        most_angle: bool,
    ) -> Result<LCNetResult, InferenceError> {
        match mode {
            LCNetMode::TextOrientation | LCNetMode::DocumentOrientation => Ok(
                LCNetResult::Orientations(Self::get_angles(imgs, mode, most_angle)?),
            ),
            LCNetMode::TableType => Ok(LCNetResult::TableTypes(Self::get_table_types(imgs)?)),
        }
    }
}
