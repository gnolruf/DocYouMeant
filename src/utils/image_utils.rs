use image::{imageops, ImageBuffer, Rgb, RgbImage};
use imageproc::geometric_transformations::{warp, Interpolation, Projection};
use ndarray::{Array, Array4};

use crate::{document::text_box::Orientation, document::text_box::TextBox};

pub fn add_image_padding(
    image: &RgbImage,
    padding_top: u32,
    padding_bottom: u32,
    padding_left: u32,
    padding_right: u32,
) -> RgbImage {
    if padding_top == 0 && padding_bottom == 0 && padding_left == 0 && padding_right == 0 {
        return image.clone();
    }

    let original_width = image.width();
    let original_height = image.height();
    let new_width = original_width + padding_left + padding_right;
    let new_height = original_height + padding_top + padding_bottom;

    let mut padded_img: RgbImage = ImageBuffer::new(new_width, new_height);

    for pixel in padded_img.pixels_mut() {
        *pixel = Rgb([255u8, 255u8, 255u8]);
    }

    for y in 0..original_height {
        for x in 0..original_width {
            let src_pixel = image.get_pixel(x, y);
            padded_img.put_pixel(x + padding_left, y + padding_top, *src_pixel);
        }
    }

    padded_img
}

pub fn get_rotate_crop_image(
    src: &RgbImage,
    box_points: &[(i32, i32)],
) -> Result<(RgbImage, bool), Box<dyn std::error::Error + Send + Sync>> {
    if box_points.len() != 4 {
        return Err("Expected exactly 4 box points".into());
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
        return Err("Invalid bounding box dimensions".into());
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
        return Err("Invalid output dimensions".into());
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

    let projection = Projection::from_control_points(from_points, to_points).ok_or_else(
        || -> Box<dyn std::error::Error + Send + Sync> {
            "Failed to create perspective transformation".into()
        },
    )?;

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

pub fn rotate_image(img: &RgbImage, angle_type: Orientation) -> RgbImage {
    match angle_type {
        Orientation::Oriented0 => img.clone(),
        Orientation::Oriented90 => imageops::rotate270(img),
        Orientation::Oriented180 => imageops::rotate180(img),
        Orientation::Oriented270 => imageops::rotate90(img),
    }
}

pub fn rotate_images_by_angle(
    part_images: &[RgbImage],
    text_boxes: &[TextBox],
) -> Result<Vec<RgbImage>, Box<dyn std::error::Error + Send + Sync>> {
    let mut rotated_images = Vec::with_capacity(part_images.len());

    for (img, text_box) in part_images.iter().zip(text_boxes.iter()) {
        let rotated_img = if let Some(angle) = text_box.angle {
            rotate_image(img, angle)
        } else {
            img.clone()
        };

        rotated_images.push(rotated_img);
    }

    Ok(rotated_images)
}

pub fn subtract_mean_normalize(
    img: &RgbImage,
    mean_values: &[f32; 3],
    norm_values: &[f32; 3],
) -> Result<Array4<f32>, Box<dyn std::error::Error + Send + Sync>> {
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

    Ok(input)
}

pub fn get_image_parts(
    src: &RgbImage,
    text_boxes: &[TextBox],
) -> Result<Vec<RgbImage>, Box<dyn std::error::Error + Send + Sync>> {
    let mut part_images = Vec::with_capacity(text_boxes.len());

    for text_box in text_boxes.iter() {
        let box_points: Vec<(i32, i32)> = text_box.bounds.iter().map(|p| (p.x, p.y)).collect();

        let (part_img, _) = get_rotate_crop_image(src, &box_points)?;

        part_images.push(part_img);
    }

    Ok(part_images)
}
