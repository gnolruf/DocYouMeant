//! Utility functions for image processing operations.

use image::{imageops, ImageBuffer, Rgb, RgbImage};
use imageproc::geometric_transformations::{warp, Interpolation, Projection};
use ndarray::{Array, Array4, Axis};

use crate::document::text_box::Orientation;
use crate::document::text_box::TextBox;
use crate::utils::error::ImageError;

/// Adds padding around an image with a specified background color.
///
/// Creates a new image with the specified padding added to each side,
/// filling the padded areas with the specified color.
///
/// # Arguments
///
/// * `image` - The source RGB image to add padding to.
/// * `padding_top` - The number of pixels to add to the top edge.
/// * `padding_bottom` - The number of pixels to add to the bottom edge.
/// * `padding_left` - The number of pixels to add to the left edge.
/// * `padding_right` - The number of pixels to add to the right edge.
/// * `pad_color` - The RGB color to use for padding. Defaults to white (255, 255, 255) if `None`.
///
/// # Returns
///
/// A new [`RgbImage`] with the original image centered and surrounded by padding.
/// If all padding values are zero, returns a clone of the original image.
#[must_use]
pub fn add_image_padding(
    image: &RgbImage,
    padding_top: u32,
    padding_bottom: u32,
    padding_left: u32,
    padding_right: u32,
    pad_color: Option<Rgb<u8>>,
) -> RgbImage {
    if padding_top == 0 && padding_bottom == 0 && padding_left == 0 && padding_right == 0 {
        return image.clone();
    }

    let color = pad_color.unwrap_or(Rgb([255u8, 255u8, 255u8]));

    let original_width = image.width();
    let original_height = image.height();
    let new_width = original_width + padding_left + padding_right;
    let new_height = original_height + padding_top + padding_bottom;

    let mut padded_img: RgbImage = ImageBuffer::from_pixel(new_width, new_height, color);
    imageops::overlay(
        &mut padded_img,
        image,
        padding_left.into(),
        padding_top.into(),
    );

    padded_img
}

/// Extracts and perspective-corrects a quadrilateral region from an image.
///
/// Given four corner points defining a quadrilateral in the source image,
/// this function crops the bounding box, applies a perspective transformation
/// to rectify the region, and returns the corrected image.
///
/// # Arguments
///
/// * `src` - The source RGB image to extract the region from.
/// * `box_points` - A slice of exactly 4 `(i32, i32)` points representing the
///   corners of the quadrilateral in clockwise order starting from top-left:
///   `[top-left, top-right, bottom-right, bottom-left]`.
///
/// # Returns
///
/// Returns `Ok((image, is_vertical))` where:
/// * `image` - The perspective-corrected [`RgbImage`] of the extracted region.
/// * `is_vertical` - `true` if the output height is at least 1.5x the width,
///   indicating vertical text orientation.
///
/// # Errors
///
/// Returns an error if:
/// * `box_points` does not contain exactly 4 points.
/// * The bounding box has zero width or height.
/// * The output dimensions would be zero.
/// * The perspective transformation cannot be computed from the given points.
#[must_use]
pub fn get_rotate_crop_image(
    src: &RgbImage,
    box_points: &[(i32, i32)],
) -> Result<(RgbImage, bool), ImageError> {
    if box_points.len() != 4 {
        return Err(ImageError::InvalidInput {
            message: "Expected exactly 4 box points".to_string(),
        });
    }

    let mut min_x = box_points[0].0;
    let mut max_x = box_points[0].0;
    let mut min_y = box_points[0].1;
    let mut max_y = box_points[0].1;

    for point in box_points.iter().skip(1) {
        min_x = std::cmp::min(min_x, point.0);
        max_x = std::cmp::max(max_x, point.0);
        min_y = std::cmp::min(min_y, point.1);
        max_y = std::cmp::max(max_y, point.1);
    }

    let crop_width = (max_x - min_x) as u32;
    let crop_height = (max_y - min_y) as u32;

    if crop_width == 0 || crop_height == 0 {
        return Err(ImageError::InvalidInput {
            message: "Invalid bounding box dimensions".to_string(),
        });
    }

    let mut cropped_img: RgbImage = ImageBuffer::new(crop_width, crop_height);

    for y in 0..crop_height {
        for x in 0..crop_width {
            let src_x = (x as i32 + min_x).clamp(0, src.width() as i32 - 1) as u32;
            let src_y = (y as i32 + min_y).clamp(0, src.height() as i32 - 1) as u32;
            let pixel = src.get_pixel(src_x, src_y);
            cropped_img.put_pixel(x, y, *pixel);
        }
    }

    let adjusted_points: Vec<(f32, f32)> = box_points
        .iter()
        .map(|point| ((point.0 - min_x) as f32, (point.1 - min_y) as f32))
        .collect();

    let width_dist = ((adjusted_points[0].0 - adjusted_points[1].0).powi(2)
        + (adjusted_points[0].1 - adjusted_points[1].1).powi(2))
    .sqrt();
    let height_dist = ((adjusted_points[0].0 - adjusted_points[3].0).powi(2)
        + (adjusted_points[0].1 - adjusted_points[3].1).powi(2))
    .sqrt();

    let output_width = width_dist.round() as u32;
    let output_height = height_dist.round() as u32;

    if output_width == 0 || output_height == 0 {
        return Err(ImageError::InvalidInput {
            message: "Invalid output dimensions".to_string(),
        });
    }

    let from_points = [
        adjusted_points[0],
        adjusted_points[1],
        adjusted_points[2],
        adjusted_points[3],
    ];

    let to_points = [
        (0.0, 0.0),
        (output_width as f32, 0.0),
        (output_width as f32, output_height as f32),
        (0.0, output_height as f32),
    ];

    let projection = Projection::from_control_points(from_points, to_points)
        .ok_or(ImageError::ProjectionFailed)?;

    let warped = warp(
        &cropped_img,
        &projection,
        Interpolation::Bilinear,
        Rgb([255u8, 255u8, 255u8]),
    );

    let mut output_img: RgbImage = ImageBuffer::new(output_width, output_height);

    // Copy warped result to output (handling potential size differences)
    for y in 0..output_height.min(warped.height()) {
        for x in 0..output_width.min(warped.width()) {
            let pixel = warped.get_pixel(x, y);
            output_img.put_pixel(x, y, *pixel);
        }
    }

    let is_vertical = (output_height as f32) >= (output_width as f32) * 1.5;

    Ok((output_img, is_vertical))
}

/// Rotates an image based on the specified orientation.
///
/// Applies a rotation transformation to correct for detected text orientation.
/// The rotation is applied counter-clockwise to bring the text to standard
/// horizontal orientation.
///
/// # Arguments
///
/// * `img` - The RGB image to rotate.
/// * `angle_type` - The [`Orientation`] indicating the detected rotation of the content.
///
/// # Returns
///
/// A new [`RgbImage`] rotated to correct the orientation:
/// * [`Orientation::Oriented0`] - No rotation (returns a clone).
/// * [`Orientation::Oriented90`] - Rotates 270° (content was rotated 90° clockwise).
/// * [`Orientation::Oriented180`] - Rotates 180°.
/// * [`Orientation::Oriented270`] - Rotates 90° (content was rotated 270° clockwise).
#[must_use]
pub fn rotate_image(img: &RgbImage, angle_type: Orientation) -> RgbImage {
    match angle_type {
        Orientation::Oriented0 => img.clone(),
        Orientation::Oriented90 => imageops::rotate270(img),
        Orientation::Oriented180 => imageops::rotate180(img),
        Orientation::Oriented270 => imageops::rotate90(img),
    }
}

/// Rotates multiple images according to their corresponding text box orientations.
///
/// Processes a batch of images, rotating each one based on the detected angle
/// stored in its associated [`TextBox`]. Images without a detected angle are
/// cloned without rotation.
///
/// # Arguments
///
/// * `part_images` - A slice of RGB images to rotate.
/// * `text_boxes` - A slice of [`TextBox`] structs containing orientation information.
///   Must have the same length as `part_images`.
///
/// # Returns
///
/// A vector of rotated images in the same order as the input.
#[must_use]
pub fn rotate_images_by_angle(part_images: &[RgbImage], text_boxes: &[TextBox]) -> Vec<RgbImage> {
    let mut rotated_images = Vec::with_capacity(part_images.len());

    for (img, text_box) in part_images.iter().zip(text_boxes.iter()) {
        let rotated_img = if let Some(angle) = text_box.angle {
            rotate_image(img, angle)
        } else {
            img.clone()
        };

        rotated_images.push(rotated_img);
    }

    rotated_images
}

/// Normalizes an image by subtracting mean values and applying normalization factors.
///
/// Converts an RGB image to a 4D tensor suitable for neural network inference.
/// Each pixel channel is normalized using the formula:
/// `output = (pixel_value * norm) - (mean * norm)`
///
/// # Arguments
///
/// * `img` - The RGB image to normalize.
/// * `mean_values` - Array of 3 mean values for each RGB channel.
/// * `norm_values` - Array of 3 normalization factors for each RGB channel
///   (typically `1.0 / 255.0` or similar scaling factors).
///
/// # Returns
///
/// An `Array4<f32>` with shape `[1, 3, height, width]` in NCHW format
/// (batch, channels, height, width), suitable for ONNX model inference.
#[must_use]
pub fn subtract_mean_normalize(
    img: &RgbImage,
    mean_values: &[f32; 3],
    norm_values: &[f32; 3],
) -> Array4<f32> {
    let width = img.width() as usize;
    let height = img.height() as usize;

    let mut input = Array::zeros((1, 3, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x as u32, y as u32);

            for ch in 0..3 {
                let pixel_value = pixel.0[ch] as f32;

                let normalized =
                    (pixel_value * norm_values[ch]) - (mean_values[ch] * norm_values[ch]);

                input[[0, ch, y, x]] = normalized;
            }
        }
    }

    input
}

/// Extracts image regions for each detected text box.
///
/// Iterates through a collection of text boxes and extracts the corresponding
/// perspective-corrected image region from the source image using
/// [`get_rotate_crop_image`].
///
/// # Arguments
///
/// * `src` - The source RGB image containing all text regions.
/// * `text_boxes` - A slice of [`TextBox`] structs, each containing boundary
///   points that define a quadrilateral text region.
///
/// # Returns
///
/// Returns `Ok(Vec<RgbImage>)` containing the extracted and perspective-corrected
/// image for each text box, in the same order as the input `text_boxes`.
///
/// # Errors
///
/// Returns an error if any text box extraction fails (see [`get_rotate_crop_image`]
/// for specific error conditions).
#[must_use]
pub fn get_image_parts(
    src: &RgbImage,
    text_boxes: &[TextBox],
) -> Result<Vec<RgbImage>, ImageError> {
    let mut part_images = Vec::with_capacity(text_boxes.len());

    for text_box in text_boxes.iter() {
        let box_points: Vec<(i32, i32)> = text_box.bounds.iter().map(|p| (p.x, p.y)).collect();

        let (part_img, _) = get_rotate_crop_image(src, &box_points)?;

        part_images.push(part_img);
    }

    Ok(part_images)
}

/// Stacks multiple 4D arrays into a single 4D batch array.
///
/// Takes normalized image arrays of shape `[1, C, H, W]` and concatenates them
/// into a batch tensor of shape `[N, C, H, W]`.
///
/// # Arguments
///
/// * `arrays` - Slice of 4D arrays (each `[1, C, H, W]`) to concatenate
///
/// # Returns
///
/// * `Ok(Array4<f32>)` - Concatenated 4D array `[N, C, H, W]`
/// * `Err(ImageError)` - If arrays slice is empty or arrays have inconsistent shapes
#[must_use]
pub fn stack_arrays(arrays: &[Array4<f32>]) -> Result<Array4<f32>, ImageError> {
    if arrays.is_empty() {
        return Err(ImageError::InvalidInput {
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
