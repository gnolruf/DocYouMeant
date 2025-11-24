use geo::Coord;
use image::RgbImage;
use ndarray::Array2;
use once_cell::sync::OnceCell;
use ort::{inputs, session::Session, value::Value};
use std::sync::Mutex;

use crate::document::{LayoutBox, LayoutClass};
use crate::inference::error::InferenceError;
use crate::utils::{box_utils, image_utils};

static RTDETR_INSTANCE: OnceCell<Mutex<RtDetr>> = OnceCell::new();

pub struct RtDetr {
    session: Session,
    mean_values: [f32; 3],
    norm_values: [f32; 3],
    src_width: i32,
    src_height: i32,
    input_size: u32,
    conf_threshold: f32,
    nms_threshold: f32,
    scale_x: f32,
    scale_y: f32,
}

impl RtDetr {
    const MODEL_PATH: &'static str = "models/onnx/layout_detection.onnx";
    const NUM_THREADS: usize = 4;

    pub fn new() -> Result<Self, InferenceError> {
        let session = Session::builder()
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: Self::MODEL_PATH.into(),
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
            .commit_from_file(Self::MODEL_PATH)
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: Self::MODEL_PATH.into(),
                source,
            })?;

        Ok(Self {
            session,
            mean_values: [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
            norm_values: [
                1.0 / 0.229 / 255.0,
                1.0 / 0.224 / 255.0,
                1.0 / 0.225 / 255.0,
            ],
            src_height: 0,
            src_width: 0,
            input_size: 800,
            conf_threshold: 0.5,
            nms_threshold: 0.4,
            scale_x: 1.0,
            scale_y: 1.0,
        })
    }

    pub fn get_or_init() -> Result<(), InferenceError> {
        RTDETR_INSTANCE.get_or_try_init(|| Self::new().map(Mutex::new))?;
        Ok(())
    }

    fn instance() -> Result<&'static Mutex<RtDetr>, InferenceError> {
        RTDETR_INSTANCE.get_or_try_init(|| Self::new().map(Mutex::new))
    }

    pub fn preprocess(&mut self, image: &RgbImage) -> Result<RgbImage, InferenceError> {
        self.src_width = image.width() as i32;
        self.src_height = image.height() as i32;

        let resized_img = image::imageops::resize(
            image,
            self.input_size,
            self.input_size,
            image::imageops::FilterType::Lanczos3,
        );

        self.scale_x = self.input_size as f32 / self.src_width as f32;
        self.scale_y = self.input_size as f32 / self.src_height as f32;

        Ok(resized_img)
    }

    fn create_model_inputs(
        &self,
        image: &RgbImage,
    ) -> Result<(Value, Value, Value), InferenceError> {
        let input_array =
            image_utils::subtract_mean_normalize(image, &self.mean_values, &self.norm_values)
                .map_err(|e| InferenceError::PreprocessingError {
                    operation: "normalize image".to_string(),
                    message: e.to_string(),
                })?;

        let image_shape = input_array.shape().to_vec();
        let (image_data, _offset) = input_array.into_raw_vec_and_offset();
        let image_input = Value::from_array((image_shape.as_slice(), image_data)).map_err(|e| {
            InferenceError::PreprocessingError {
                operation: "create image input".to_string(),
                message: e.to_string(),
            }
        })?;

        let im_shape_data = vec![self.src_height as f32, self.src_width as f32];
        let im_shape_array = Array2::from_shape_vec((1, 2), im_shape_data).map_err(|e| {
            InferenceError::PreprocessingError {
                operation: "create im_shape array".to_string(),
                message: e.to_string(),
            }
        })?;
        let im_shape_shape = im_shape_array.shape().to_vec();
        let (im_shape_data, _offset) = im_shape_array.into_raw_vec_and_offset();
        let im_shape_input = Value::from_array((im_shape_shape.as_slice(), im_shape_data))
            .map_err(|e| InferenceError::PreprocessingError {
                operation: "create im_shape input".to_string(),
                message: e.to_string(),
            })?;

        let scale_y = self.input_size as f32 / self.src_height as f32;
        let scale_x = self.input_size as f32 / self.src_width as f32;
        let scale_factor_data = vec![scale_y, scale_x];
        let scale_factor_array =
            Array2::from_shape_vec((1, 2), scale_factor_data).map_err(|e| {
                InferenceError::PreprocessingError {
                    operation: "create scale_factor array".to_string(),
                    message: e.to_string(),
                }
            })?;
        let scale_shape = scale_factor_array.shape().to_vec();
        let (scale_data, _offset) = scale_factor_array.into_raw_vec_and_offset();
        let scale_factor_input =
            Value::from_array((scale_shape.as_slice(), scale_data)).map_err(|e| {
                InferenceError::PreprocessingError {
                    operation: "create scale_factor input".to_string(),
                    message: e.to_string(),
                }
            })?;

        Ok((
            image_input.into(),
            im_shape_input.into(),
            scale_factor_input.into(),
        ))
    }

    pub fn detect(&mut self, image: &RgbImage) -> Result<Vec<LayoutBox>, InferenceError> {
        let (image_input, im_shape_input, scale_factor_input) = self.create_model_inputs(image)?;

        let (bbox_data, num_detections) = {
            let outputs = self
                .session
                .run(inputs![
                    "image" => image_input,
                    "im_shape" => im_shape_input,
                    "scale_factor" => scale_factor_input
                ])
                .map_err(|source| InferenceError::ModelExecutionError {
                    operation: "RT-DETR forward pass".to_string(),
                    source,
                })?;

            let bbox_output = outputs
                .get("fetch_name_0")
                .ok_or_else(|| InferenceError::PredictionError {
                    operation: "get bbox output".to_string(),
                    message: "Output 'fetch_name_0' not found".to_string(),
                })?
                .try_extract_tensor::<f32>()
                .map_err(|source| InferenceError::PredictionError {
                    operation: "extract bbox tensor".to_string(),
                    message: source.to_string(),
                })?;

            let num_output = outputs
                .get("fetch_name_1")
                .ok_or_else(|| InferenceError::PredictionError {
                    operation: "get num output".to_string(),
                    message: "Output 'fetch_name_1' not found".to_string(),
                })?
                .try_extract_tensor::<i32>()
                .map_err(|source| InferenceError::PredictionError {
                    operation: "extract num tensor".to_string(),
                    message: source.to_string(),
                })?;

            (bbox_output.1.to_vec(), num_output.1[0] as usize)
        };

        let mut detection_tuples = Vec::new();
        let num_detections = num_detections.min(bbox_data.len() / 6);

        let conf_threshold = self.conf_threshold;

        for i in 0..num_detections {
            let idx = i * 6;

            let class_id = bbox_data[idx] as usize;
            let confidence = bbox_data[idx + 1];
            let x1 = bbox_data[idx + 2];
            let y1 = bbox_data[idx + 3];
            let x2 = bbox_data[idx + 4];
            let y2 = bbox_data[idx + 5];

            if confidence < conf_threshold {
                continue;
            }

            if LayoutClass::from_id(class_id).is_some() {
                let orig_x1 = (x1 * self.scale_x).round() as i32;
                let orig_y1 = (y1 * self.scale_y).round() as i32;
                let orig_x2 = (x2 * self.scale_x).round() as i32;
                let orig_y2 = (y2 * self.scale_y).round() as i32;

                if orig_x2 > orig_x1 && orig_y2 > orig_y1 {
                    let points = [
                        Coord {
                            x: orig_x1,
                            y: orig_y1,
                        }, // top-left
                        Coord {
                            x: orig_x2,
                            y: orig_y1,
                        }, // top-right
                        Coord {
                            x: orig_x2,
                            y: orig_y2,
                        }, // bottom-right
                        Coord {
                            x: orig_x1,
                            y: orig_y2,
                        }, // bottom-left
                    ];
                    detection_tuples.push((points, class_id, confidence));
                }
            }
        }

        let filtered_tuples = box_utils::apply_nms(detection_tuples, self.nms_threshold);

        let filtered_detections: Vec<LayoutBox> = filtered_tuples
            .into_iter()
            .filter_map(|(points, class_id, confidence)| {
                LayoutClass::from_id(class_id).map(|class| LayoutBox {
                    bounds: points,
                    class,
                    confidence,
                })
            })
            .collect();

        Ok(filtered_detections)
    }

    pub fn run(image: &RgbImage) -> Result<Vec<LayoutBox>, InferenceError> {
        let instance = Self::instance()?;
        let mut model = instance
            .lock()
            .map_err(|e| InferenceError::ProcessingError {
                message: format!("Failed to lock RtDetr instance: {e}"),
            })?;

        let preprocessed = model.preprocess(image)?;
        model.detect(&preprocessed)
    }
}
