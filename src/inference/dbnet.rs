//! Differentiable Binarization Network (DBNet) for text detection.
//!
//! This module provides text detection capabilities using DBNet, a segmentation-based
//! text detector that produces pixel-level probability maps and uses differentiable
//! binarization to generate text region proposals.

use geo::Coord;
use image::{ImageBuffer, Luma, RgbImage};
use imageproc::contours::{find_contours_with_threshold, Contour};
use imageproc::distance_transform::Norm;
use imageproc::drawing::draw_polygon_mut;
use imageproc::geometry::min_area_rect;
use imageproc::morphology::dilate;
use imageproc::point::Point;
use ndarray::Array2;
use ort::{inputs, session::builder::PrepackedWeights, session::Session, value::Value};
use std::sync::OnceLock;

use crate::document::bounds::Bounds;
use crate::document::TextBox;
use crate::inference::error::InferenceError;
use crate::inference::{once_lock_try_init, SessionPool};
use crate::utils::config::AppConfig;
use crate::utils::{box_utils, image_utils};

/// Preprocessing metadata produced by [`DBNet::preprocess`].
///
/// Contains the dimensional information needed to rescale detected boxes
/// back to original image coordinates.
pub(crate) struct DBNetPreprocessResult {
    /// Original input image dimensions `[width, height]`
    src_dimensions: [i32; 2],
    /// Processed image dimensions after padding `[width, height]`
    dst_dimensions: [i32; 2],
    /// Applied padding `[top, bottom, left, right]`
    padding: [i32; 4],
}

/// Text detection model using Differentiable Binarization Network.
///
/// `DBNet` detects text regions in document images by producing a probability map
/// and extracting bounding boxes through contour analysis. It handles arbitrary
/// text orientations and produces rotated bounding boxes when necessary.
///
/// # Fields
///
/// - `session`: ONNX Runtime session for model inference
/// - `mean_values`: ImageNet mean values for normalization [R, G, B]
/// - `norm_values`: Normalization divisors for each channel
/// - `max_side_len`: Maximum image dimension (larger images are resized)
/// - `box_thresh`: Threshold for binary mask creation (probability cutoff)
/// - `box_score_thresh`: Minimum average confidence score to keep a detection
/// - `unclip_ratio`: Expansion ratio for detected boxes (compensates for shrinkage)
pub struct DBNet {
    session: Session,
    mean_values: [f32; 3],
    norm_values: [f32; 3],
    max_side_len: i32,
    box_thresh: f32,
    box_score_thresh: f32,
    unclip_ratio: f32,
}

static DBNET_POOL: OnceLock<SessionPool<DBNet>> = OnceLock::new();

impl DBNet {
    /// Path to the DBNet ONNX model file.
    const MODEL_PATH: &'static str = "onnx/text_detection.onnx";
    /// Number of threads for ONNX Runtime inter-op parallelism.
    const NUM_THREADS: usize = 4;

    /// Creates a new DBNet instance.
    ///
    /// Loads the ONNX model and initializes default parameters for text detection.
    /// This is typically called internally by the singleton initialization.
    ///
    /// # Returns
    ///
    /// * `Ok(DBNet)` - Initialized detector ready for inference
    /// * `Err(InferenceError)` - If the model file cannot be loaded
    ///
    /// # Default Parameters
    ///
    /// - `max_side_len`: 960 pixels
    /// - `box_thresh`: 0.3 (probability threshold)
    /// - `box_score_thresh`: 0.5 (minimum confidence)
    /// - `unclip_ratio`: 1.5 (box expansion factor)
    /// Pre-initializes the session pool.
    pub fn get_or_init() -> Result<(), InferenceError> {
        once_lock_try_init(&DBNET_POOL, || {
            let pool_size = AppConfig::get().inference_pool_size;
            SessionPool::new(pool_size, |w| Self::new(w))
        })?;
        Ok(())
    }

    fn new(prepacked: &PrepackedWeights) -> Result<Self, InferenceError> {
        let config = AppConfig::get();
        let model_path = config.model_path(Self::MODEL_PATH)?;

        let session = Session::builder()
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: model_path.clone().into(),
                source,
            })?
            .with_execution_providers([ort::ep::TensorRT::default()
                .with_device_id(0)
                .with_engine_cache(true)
                .with_engine_cache_path(config.rt_cache_directory()?)
                .with_engine_cache_prefix("docyoumeant_")
                .with_max_workspace_size(5 << 30)
                .with_fp16(true)
                .with_timing_cache(true)
                .with_profile_min_shapes("x:1x3x32x32")
                .with_profile_max_shapes("x:1x3x960x960")
                .with_profile_opt_shapes("x:1x3x640x640")
                .build()])?
            .with_inter_threads(Self::NUM_THREADS)?
            .with_prepacked_weights(prepacked)?
            .commit_from_file(&model_path)
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: model_path.into(),
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
            max_side_len: 960,
            box_thresh: 0.3,
            box_score_thresh: 0.5,
            unclip_ratio: 1.5,
        })
    }

    /// Preprocesses an image for text detection.
    ///
    /// Resizes the image to fit within `max_side_len` while maintaining aspect ratio,
    /// then pads to dimensions divisible by 32 (required by the network architecture).
    ///
    /// # Arguments
    ///
    /// * `image` - Input RGB image
    ///
    /// # Returns
    ///
    /// Preprocessed image ready for inference
    pub(crate) fn preprocess(&self, image: &RgbImage) -> (RgbImage, DBNetPreprocessResult) {
        let width = image.width() as i32;
        let height = image.height() as i32;
        let orig_max_side = width.max(height);

        let resize_ratio = if self.max_side_len <= 0 || self.max_side_len > orig_max_side {
            1.0
        } else {
            self.max_side_len as f32 / orig_max_side as f32
        };

        let resized_width = (width as f32 * resize_ratio) as u32;
        let resized_height = (height as f32 * resize_ratio) as u32;

        let resized_img = image::imageops::resize(
            image,
            resized_width,
            resized_height,
            image::imageops::FilterType::Triangle,
        );

        let pad_width_total = (32 - (resized_width % 32)) % 32;
        let pad_height_total = (32 - (resized_height % 32)) % 32;

        let pad_left = pad_width_total / 2;
        let pad_right = pad_width_total - pad_left;
        let pad_top = pad_height_total / 2;
        let pad_bottom = pad_height_total - pad_top;

        let final_width = resized_width + pad_width_total;
        let final_height = resized_height + pad_height_total;

        let padded_final_img = image_utils::add_image_padding(
            &resized_img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            None,
        );

        let meta = DBNetPreprocessResult {
            src_dimensions: [width, height],
            dst_dimensions: [final_width as i32, final_height as i32],
            padding: [
                pad_top as i32,
                pad_bottom as i32,
                pad_left as i32,
                pad_right as i32,
            ],
        };

        (padded_final_img, meta)
    }

    /// Performs text detection on a preprocessed image.
    ///
    /// Runs the DBNet model and post-processes the output probability map
    /// to extract text region bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `image` - Preprocessed image from `preprocess()`
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TextBox>)` - Detected text regions with confidence scores
    /// * `Err(InferenceError)` - If detection fails
    ///
    /// # Post-processing Pipeline
    ///
    /// 1. Normalize input with ImageNet statistics
    /// 2. Run model inference to get probability map
    /// 3. Apply threshold to create binary mask
    /// 4. Dilate mask to connect nearby regions
    /// 5. Find contours in the binary mask
    /// 6. Fit minimum area rectangles to contours
    /// 7. Filter boxes by size and confidence
    /// 8. Rescale to original image coordinates
    pub(crate) fn detect(
        &mut self,
        image: &RgbImage,
        meta: &DBNetPreprocessResult,
    ) -> Result<Vec<TextBox>, InferenceError> {
        let input_array =
            image_utils::subtract_mean_normalize(image, &self.mean_values, &self.norm_values);

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
                    operation: "DBNet forward pass".to_string(),
                    source,
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

            (output.1.to_vec(), output.0.clone())
        };

        let out_height = output_shape[2] as usize;
        let out_width = output_shape[3] as usize;
        let area = out_height * out_width;

        let mut pred_data = vec![0.0f32; area];
        let mut threshold_data = vec![false; area];

        for y in 0..out_height {
            for x in 0..out_width {
                let idx = y * out_width + x;
                let value = output_data[idx];
                pred_data[idx] = value;
                threshold_data[idx] = value > self.box_thresh;
            }
        }

        let pred_array =
            Array2::from_shape_vec((out_height, out_width), pred_data).map_err(|e| {
                InferenceError::PredictionError {
                    operation: "create prediction array".to_string(),
                    message: e.to_string(),
                }
            })?;

        let mut threshold_img: image::GrayImage =
            ImageBuffer::new(out_width as u32, out_height as u32);
        for y in 0..out_height {
            for x in 0..out_width {
                let idx = y * out_width + x;
                let pixel_value = if threshold_data[idx] { 255u8 } else { 0u8 };
                threshold_img.put_pixel(x as u32, y as u32, image::Luma([pixel_value]));
            }
        }

        let dilated_img = dilate(&threshold_img, Norm::LInf, 1);

        let text_boxes = self.get_text_boxes(&dilated_img, &pred_array, meta)?;

        Ok(text_boxes)
    }

    /// Extracts text bounding boxes from the thresholded detection map.
    ///
    /// Processes the binary threshold image to find contours, fits minimum
    /// area rectangles, and filters results based on size and confidence thresholds.
    ///
    /// # Arguments
    ///
    /// * `threshold_img` - Binary image from thresholding the probability map
    /// * `pred_array` - Original probability map for confidence scoring
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TextBox>)` - Filtered and rescaled text box detections
    /// * `Err(InferenceError)` - If processing fails
    fn get_text_boxes(
        &self,
        threshold_img: &image::GrayImage,
        pred_array: &Array2<f32>,
        meta: &DBNetPreprocessResult,
    ) -> Result<Vec<TextBox>, InferenceError> {
        const LONG_SIDE_THRESH: i32 = 3;
        const MAX_CANDIDATES: usize = 1000;

        let threshold_value = (self.box_thresh * 255.0) as u8;
        let contours: Vec<Contour<i32>> =
            find_contours_with_threshold(threshold_img, threshold_value);
        let num_contours = contours.len().min(MAX_CANDIDATES);
        let mut rs_boxes = Vec::with_capacity(num_contours);

        for contour in contours.iter().take(num_contours) {
            if contour.points.len() <= 2 {
                continue;
            }

            let contour_points: Vec<Point<i32>> = contour.points.clone();

            let min_area_points = min_area_rect(&contour_points);

            let min_area_coords: [Coord<i32>; 4] = [
                Coord {
                    x: min_area_points[0].x,
                    y: min_area_points[0].y,
                },
                Coord {
                    x: min_area_points[1].x,
                    y: min_area_points[1].y,
                },
                Coord {
                    x: min_area_points[2].x,
                    y: min_area_points[2].y,
                },
                Coord {
                    x: min_area_points[3].x,
                    y: min_area_points[3].y,
                },
            ];

            let (min_boxes, long_side) = box_utils::get_min_boxes(&min_area_coords);

            if long_side < LONG_SIDE_THRESH as f32 {
                continue;
            }

            let box_score = Self::box_score(&min_boxes, pred_array);

            if box_score < self.box_score_thresh {
                continue;
            }

            let unclipped_points = match box_utils::unclip_box(&min_boxes, self.unclip_ratio) {
                Ok(points) => points,
                Err(_) => continue,
            };

            let clip_points = min_area_rect(
                &unclipped_points
                    .iter()
                    .map(|p| Point::new(p.x, p.y))
                    .collect::<Vec<_>>(),
            );

            let clip_coords: [Coord<i32>; 4] = [
                Coord {
                    x: clip_points[0].x,
                    y: clip_points[0].y,
                }, // top-left
                Coord {
                    x: clip_points[1].x,
                    y: clip_points[1].y,
                }, // top-right
                Coord {
                    x: clip_points[2].x,
                    y: clip_points[2].y,
                }, // bottom-right
                Coord {
                    x: clip_points[3].x,
                    y: clip_points[3].y,
                }, // bottom-left
            ];

            let side1_len = (((clip_coords[1].x - clip_coords[0].x).pow(2) as f32)
                + ((clip_coords[1].y - clip_coords[0].y).pow(2) as f32))
                .sqrt();
            let side2_len = (((clip_coords[2].x - clip_coords[1].x).pow(2) as f32)
                + ((clip_coords[2].y - clip_coords[1].y).pow(2) as f32))
                .sqrt();

            if side1_len < 1.001 && side2_len < 1.001 {
                continue;
            }

            let (clip_min_boxes, clip_long_side) = box_utils::get_min_boxes(&clip_coords);

            if clip_long_side < (LONG_SIDE_THRESH + 2) as f32 {
                continue;
            }

            let int_clip_min_boxes = Self::rescale_box(
                meta,
                &clip_min_boxes,
                pred_array.ncols() as i32,
                pred_array.nrows() as i32,
            );

            rs_boxes.push(TextBox {
                bounds: Bounds::new(int_clip_min_boxes),
                angle: None,
                text: None,
                box_score,
                text_score: 0.0,
                span: None,
            });
        }

        Ok(rs_boxes)
    }

    /// Rescales a bounding box from model output coordinates to original image coordinates.
    ///
    /// Accounts for the preprocessing transformations (resize and padding) to map
    /// detected boxes back to their positions in the original input image.
    ///
    /// # Arguments
    ///
    /// * `box_points` - Four corner points in model output space
    /// * `pred_width` - Width of the prediction map
    /// * `pred_height` - Height of the prediction map
    ///
    /// # Returns
    ///
    /// Four corner points in original image coordinates, clamped to valid bounds.
    fn rescale_box(
        meta: &DBNetPreprocessResult,
        box_points: &[Coord<i32>; 4],
        pred_width: i32,
        pred_height: i32,
    ) -> [Coord<i32>; 4] {
        let mut result = [Coord { x: 0, y: 0 }; 4];

        for (i, p) in box_points.iter().enumerate() {
            let x_f = p.x as f32;
            let y_f = p.y as f32;

            let w_out = pred_width as f32;
            let h_out = pred_height as f32;

            if w_out < 1.0 || h_out < 1.0 {
                result[i] = Coord { x: 0, y: 0 };
                continue;
            }

            let x_dst = x_f * (meta.dst_dimensions[0] as f32 / w_out);
            let y_dst = y_f * (meta.dst_dimensions[1] as f32 / h_out);

            let w_scaled_content =
                (meta.dst_dimensions[0] - meta.padding[2] - meta.padding[3]) as f32;
            let h_scaled_content =
                (meta.dst_dimensions[1] - meta.padding[0] - meta.padding[1]) as f32;

            let x_on_scaled_content = x_dst - meta.padding[2] as f32;
            let y_on_scaled_content = y_dst - meta.padding[0] as f32;

            let mut x_orig = 0.0;
            let mut y_orig = 0.0;

            if w_scaled_content > 1e-6 {
                x_orig = x_on_scaled_content * (meta.src_dimensions[0] as f32 / w_scaled_content);
            }

            if h_scaled_content > 1e-6 {
                y_orig = y_on_scaled_content * (meta.src_dimensions[1] as f32 / h_scaled_content);
            }

            let max_x_coord = (meta.src_dimensions[0] - 1).max(0) as f32;
            let max_y_coord = (meta.src_dimensions[1] - 1).max(0) as f32;

            let pt_x = x_orig.clamp(0.0, max_x_coord) as i32;
            let pt_y = y_orig.clamp(0.0, max_y_coord) as i32;

            result[i] = Coord { x: pt_x, y: pt_y };
        }

        result
    }

    /// Calculates the average prediction score within a bounding box region.
    ///
    /// This function computes the mean value of a prediction map (typically a
    /// probability/confidence map) within the area defined by a bounding box.
    /// It uses a polygon mask to accurately include only pixels inside the box.
    ///
    /// # Arguments
    ///
    /// * `boxes` - Four corner points defining the bounding box
    /// * `pred` - 2D array of prediction values (e.g., text probability map)
    ///
    /// # Returns
    ///
    /// Average prediction value within the box (0.0 to 1.0 for probability maps).
    /// Returns 0.0 if the prediction array is empty or no pixels fall within the box.
    pub fn box_score(boxes: &[Coord<i32>; 4], pred: &Array2<f32>) -> f32 {
        let width = pred.ncols();
        let height = pred.nrows();

        if width == 0 || height == 0 {
            return 0.0;
        }

        let mut min_x = boxes[0].x;
        let mut max_x = boxes[0].x;
        let mut min_y = boxes[0].y;
        let mut max_y = boxes[0].y;

        for point in &boxes[1..] {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }

        let clamp_min_x = min_x.clamp(0, width as i32 - 1);
        let clamp_max_x = max_x.clamp(0, width as i32 - 1);
        let clamp_min_y = min_y.clamp(0, height as i32 - 1);
        let clamp_max_y = max_y.clamp(0, height as i32 - 1);

        let mask_width = (clamp_max_x - clamp_min_x + 1) as u32;
        let mask_height = (clamp_max_y - clamp_min_y + 1) as u32;

        let mut mask: image::GrayImage = ImageBuffer::new(mask_width, mask_height);

        let polygon_points: Vec<Point<i32>> = boxes
            .iter()
            .map(|point| Point::new(point.x - clamp_min_x, point.y - clamp_min_y))
            .collect();

        draw_polygon_mut(&mut mask, &polygon_points, Luma([255u8]));

        let mut sum = 0.0;
        let mut count = 0;

        for y in clamp_min_y..=clamp_max_y {
            for x in clamp_min_x..=clamp_max_x {
                let mask_x = (x - clamp_min_x) as u32;
                let mask_y = (y - clamp_min_y) as u32;

                if mask.get_pixel(mask_x, mask_y)[0] > 0 {
                    sum += pred[(y as usize, x as usize)];
                    count += 1;
                }
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Detects text regions in an image (main entry point).
    ///
    /// This is the primary method for text detection. It handles singleton
    /// initialization, preprocessing, inference, and post-processing in a
    /// single call.
    ///
    /// # Arguments
    ///
    /// * `image` - Input RGB image to analyze
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TextBox>)` - Detected text regions with bounding boxes and confidence scores
    /// * `Err(InferenceError)` - If any step of the detection pipeline fails
    pub fn run(image: &RgbImage) -> Result<Vec<TextBox>, InferenceError> {
        let pool = once_lock_try_init(&DBNET_POOL, || {
            let pool_size = AppConfig::get().inference_pool_size;
            SessionPool::new(pool_size, |w| Self::new(w))
        })?;
        pool.with(|model| {
            let (preprocessed, meta) = model.preprocess(image);
            model.detect(&preprocessed, &meta)
        })
    }
}
