use docyoumeant::document::text_box::{Orientation, TextBox};
use docyoumeant::utils::image_utils::{
    add_image_padding, get_image_parts, get_rotate_crop_image, rotate_image,
    rotate_images_by_angle, subtract_mean_normalize,
};
use geo::Coord;
use image::{ImageBuffer, Rgb, RgbImage};

// ============================================================================
// add_image_padding Tests
// ============================================================================

#[test]
fn test_add_image_padding() {
    let width = 10;
    let height = 10;
    let mut img: RgbImage = ImageBuffer::new(width, height);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]); // Black image
    }

    let padded = add_image_padding(&img, 2, 2, 2, 2);

    assert_eq!(padded.width(), 14);
    assert_eq!(padded.height(), 14);

    // Check padding color (should be white)
    assert_eq!(padded.get_pixel(0, 0), &Rgb([255, 255, 255]));
    assert_eq!(padded.get_pixel(13, 13), &Rgb([255, 255, 255]));

    // Check original image content (should be black) at offset (2, 2)
    assert_eq!(padded.get_pixel(2, 2), &Rgb([0, 0, 0]));
}

#[test]
fn test_add_image_padding_zero() {
    let width = 10;
    let height = 10;
    let img: RgbImage = ImageBuffer::new(width, height);
    let padded = add_image_padding(&img, 0, 0, 0, 0);

    assert_eq!(padded.width(), 10);
    assert_eq!(padded.height(), 10);
}

#[test]
fn test_add_image_padding_asymmetric() {
    let width = 10;
    let height = 10;
    let img: RgbImage = ImageBuffer::new(width, height);

    // Asymmetric padding: more on right and bottom
    let padded = add_image_padding(&img, 1, 5, 2, 8);

    assert_eq!(padded.width(), 10 + 2 + 8); // 20
    assert_eq!(padded.height(), 10 + 1 + 5); // 16
}

#[test]
fn test_add_image_padding_single_pixel() {
    let mut img: RgbImage = ImageBuffer::new(1, 1);
    img.put_pixel(0, 0, Rgb([128, 64, 32]));

    let padded = add_image_padding(&img, 1, 1, 1, 1);

    assert_eq!(padded.width(), 3);
    assert_eq!(padded.height(), 3);

    // Check center pixel preserved
    assert_eq!(padded.get_pixel(1, 1), &Rgb([128, 64, 32]));

    // Check corners are white
    assert_eq!(padded.get_pixel(0, 0), &Rgb([255, 255, 255]));
    assert_eq!(padded.get_pixel(2, 2), &Rgb([255, 255, 255]));
}

#[test]
fn test_add_image_padding_large_padding() {
    let img: RgbImage = ImageBuffer::new(5, 5);
    let padded = add_image_padding(&img, 100, 100, 100, 100);

    assert_eq!(padded.width(), 205);
    assert_eq!(padded.height(), 205);
}

#[test]
fn test_add_image_padding_one_side_only() {
    let mut img: RgbImage = ImageBuffer::new(5, 5);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([100, 100, 100]);
    }

    // Only pad top
    let padded = add_image_padding(&img, 10, 0, 0, 0);

    assert_eq!(padded.width(), 5);
    assert_eq!(padded.height(), 15);

    // Top should be white
    assert_eq!(padded.get_pixel(0, 0), &Rgb([255, 255, 255]));
    // Original content starts at y=10
    assert_eq!(padded.get_pixel(0, 10), &Rgb([100, 100, 100]));
}

// ============================================================================
// rotate_image Tests
// ============================================================================

#[test]
fn test_rotate_image() {
    let width = 2;
    let height = 1;
    let mut img: RgbImage = ImageBuffer::new(width, height);
    // Pixel 0,0 is Red, Pixel 1,0 is Blue
    img.put_pixel(0, 0, Rgb([255, 0, 0]));
    img.put_pixel(1, 0, Rgb([0, 0, 255]));

    // 0 degrees
    let rot0 = rotate_image(&img, Orientation::Oriented0);
    assert_eq!(rot0.width(), 2);
    assert_eq!(rot0.height(), 1);
    assert_eq!(rot0.get_pixel(0, 0), &Rgb([255, 0, 0]));

    // 90 degrees
    let rot90 = rotate_image(&img, Orientation::Oriented90);
    assert_eq!(rot90.width(), 1);
    assert_eq!(rot90.height(), 2);

    // 180 degrees
    let rot180 = rotate_image(&img, Orientation::Oriented180);
    assert_eq!(rot180.width(), 2);
    assert_eq!(rot180.height(), 1);
    // 180 rotation: [R, B] -> [B, R]
    assert_eq!(rot180.get_pixel(0, 0), &Rgb([0, 0, 255]));
    assert_eq!(rot180.get_pixel(1, 0), &Rgb([255, 0, 0]));

    // 270 degrees
    let rot270 = rotate_image(&img, Orientation::Oriented270);
    assert_eq!(rot270.width(), 1);
    assert_eq!(rot270.height(), 2);
}

#[test]
fn test_rotate_image_square() {
    let width = 10;
    let height = 10;
    let mut img: RgbImage = ImageBuffer::new(width, height);
    // Mark corner
    img.put_pixel(0, 0, Rgb([255, 0, 0]));
    img.put_pixel(9, 9, Rgb([0, 255, 0]));

    let rot90 = rotate_image(&img, Orientation::Oriented90);
    assert_eq!(rot90.width(), 10);
    assert_eq!(rot90.height(), 10);

    let rot180 = rotate_image(&img, Orientation::Oriented180);
    assert_eq!(rot180.width(), 10);
    assert_eq!(rot180.height(), 10);

    let rot270 = rotate_image(&img, Orientation::Oriented270);
    assert_eq!(rot270.width(), 10);
    assert_eq!(rot270.height(), 10);
}

#[test]
fn test_rotate_image_single_pixel() {
    let mut img: RgbImage = ImageBuffer::new(1, 1);
    img.put_pixel(0, 0, Rgb([42, 42, 42]));

    for orientation in [
        Orientation::Oriented0,
        Orientation::Oriented90,
        Orientation::Oriented180,
        Orientation::Oriented270,
    ] {
        let rotated = rotate_image(&img, orientation);
        assert_eq!(rotated.width(), 1);
        assert_eq!(rotated.height(), 1);
        assert_eq!(rotated.get_pixel(0, 0), &Rgb([42, 42, 42]));
    }
}

#[test]
fn test_rotate_image_tall_rectangle() {
    // 2x4 image (tall)
    let mut img: RgbImage = ImageBuffer::new(2, 4);
    img.put_pixel(0, 0, Rgb([255, 0, 0])); // Top-left red

    // 90 degrees: 2x4 -> 4x2
    let rot90 = rotate_image(&img, Orientation::Oriented90);
    assert_eq!(rot90.width(), 4);
    assert_eq!(rot90.height(), 2);

    // 270 degrees: 2x4 -> 4x2
    let rot270 = rotate_image(&img, Orientation::Oriented270);
    assert_eq!(rot270.width(), 4);
    assert_eq!(rot270.height(), 2);
}

// ============================================================================
// subtract_mean_normalize Tests
// ============================================================================

#[test]
fn test_subtract_mean_normalize() {
    let width = 2;
    let height = 2;
    let mut img: RgbImage = ImageBuffer::new(width, height);
    // Set all pixels to (100, 100, 100)
    for pixel in img.pixels_mut() {
        *pixel = Rgb([100, 100, 100]);
    }

    let mean = [50.0, 50.0, 50.0];
    let norm = [2.0, 2.0, 2.0];

    let result = subtract_mean_normalize(&img, &mean, &norm).unwrap();

    assert_eq!(result.shape(), &[1, 3, 2, 2]);

    // Check value at 0,0,0,0 (Batch 0, Channel 0, Y 0, X 0)
    // (100 * 2) - (50 * 2) = 200 - 100 = 100.
    assert!((result[[0, 0, 0, 0]] - 100.0).abs() < 1e-5);
}

#[test]
fn test_subtract_mean_normalize_zeros() {
    let img: RgbImage = ImageBuffer::new(2, 2);

    let mean = [0.0, 0.0, 0.0];
    let norm = [1.0, 1.0, 1.0];

    let result = subtract_mean_normalize(&img, &mean, &norm).unwrap();
    assert_eq!(result.shape(), &[1, 3, 2, 2]);

    // All zeros input with zero mean should produce zeros
    assert_eq!(result[[0, 0, 0, 0]], 0.0);
}

#[test]
fn test_subtract_mean_normalize_white_image() {
    let mut img: RgbImage = ImageBuffer::new(2, 2);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }

    let mean = [0.5, 0.5, 0.5];
    let norm = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0];

    let result = subtract_mean_normalize(&img, &mean, &norm).unwrap();

    // Formula: (pixel_value * norm) - (mean * norm)
    // (255 * (1/255)) - (0.5 * (1/255)) = 1.0 - 0.00196 ≈ 0.998
    assert!(result[[0, 0, 0, 0]] > 0.99 && result[[0, 0, 0, 0]] < 1.01);
}

#[test]
fn test_subtract_mean_normalize_single_pixel() {
    let mut img: RgbImage = ImageBuffer::new(1, 1);
    img.put_pixel(0, 0, Rgb([128, 64, 192]));

    let mean = [0.0, 0.0, 0.0];
    let norm = [1.0, 1.0, 1.0];

    let result = subtract_mean_normalize(&img, &mean, &norm).unwrap();

    assert_eq!(result.shape(), &[1, 3, 1, 1]);
    assert_eq!(result[[0, 0, 0, 0]], 128.0);
    assert_eq!(result[[0, 1, 0, 0]], 64.0);
    assert_eq!(result[[0, 2, 0, 0]], 192.0);
}

#[test]
fn test_subtract_mean_normalize_different_channel_params() {
    let mut img: RgbImage = ImageBuffer::new(1, 1);
    img.put_pixel(0, 0, Rgb([100, 100, 100]));

    // Different mean and norm per channel
    let mean = [50.0, 25.0, 0.0];
    let norm = [1.0, 2.0, 0.5];

    let result = subtract_mean_normalize(&img, &mean, &norm).unwrap();

    // R: (100 * 1.0) - (50.0 * 1.0) = 50.0
    // G: (100 * 2.0) - (25.0 * 2.0) = 200 - 50 = 150.0
    // B: (100 * 0.5) - (0.0 * 0.5) = 50.0
    assert!((result[[0, 0, 0, 0]] - 50.0).abs() < 1e-5);
    assert!((result[[0, 1, 0, 0]] - 150.0).abs() < 1e-5);
    assert!((result[[0, 2, 0, 0]] - 50.0).abs() < 1e-5);
}

// ============================================================================
// get_rotate_crop_image Tests
// ============================================================================

#[test]
fn test_get_rotate_crop_image_simple() {
    let width = 20;
    let height = 20;
    let mut img: RgbImage = ImageBuffer::new(width, height);
    // Fill with black
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]);
    }
    // Draw a white 5x5 square at 5,5
    for y in 5..10 {
        for x in 5..10 {
            img.put_pixel(x, y, Rgb([255, 255, 255]));
        }
    }

    // Box covering the white square
    let box_points = vec![(5, 5), (10, 5), (10, 10), (5, 10)];

    let (cropped, is_vertical) = get_rotate_crop_image(&img, &box_points).unwrap();

    // The box is 5x5.
    // Width distance: sqrt((5-10)^2 + (5-5)^2) = 5.
    // Height distance: sqrt((5-5)^2 + (5-10)^2) = 5.
    assert_eq!(cropped.width(), 5);
    assert_eq!(cropped.height(), 5);
    assert!(!is_vertical);

    // Check content - should be all white
    for pixel in cropped.pixels() {
        assert_eq!(*pixel, Rgb([255, 255, 255]));
    }
}

#[test]
fn test_get_rotate_crop_image_at_edge() {
    let mut img: RgbImage = ImageBuffer::new(20, 20);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([100, 100, 100]);
    }

    // Box at the edge of the image
    let box_points = vec![(0, 0), (10, 0), (10, 10), (0, 10)];

    let result = get_rotate_crop_image(&img, &box_points);
    assert!(result.is_ok());

    let (cropped, _) = result.unwrap();
    assert_eq!(cropped.width(), 10);
    assert_eq!(cropped.height(), 10);
}

#[test]
fn test_get_rotate_crop_image_extends_beyond() {
    let img: RgbImage = ImageBuffer::new(10, 10);

    // Box extends beyond image bounds
    let box_points = vec![(5, 5), (15, 5), (15, 15), (5, 15)];

    let result = get_rotate_crop_image(&img, &box_points);
    assert!(result.is_ok());

    let (cropped, _) = result.unwrap();
    // Should still produce output (with clamped coordinates)
    assert!(cropped.width() > 0);
    assert!(cropped.height() > 0);
}

#[test]
fn test_get_rotate_crop_image_vertical() {
    let img: RgbImage = ImageBuffer::new(100, 100);

    // Tall vertical box (height > 1.5 * width)
    let box_points = vec![(40, 10), (60, 10), (60, 90), (40, 90)];

    let result = get_rotate_crop_image(&img, &box_points);
    assert!(result.is_ok());

    let (cropped, is_vertical) = result.unwrap();
    // Height (80) >= Width (20) * 1.5 should be vertical
    assert!(is_vertical || cropped.height() as f32 >= cropped.width() as f32 * 1.5);
}

#[test]
fn test_get_rotate_crop_image_horizontal() {
    let img: RgbImage = ImageBuffer::new(100, 100);

    // Wide horizontal box
    let box_points = vec![(10, 40), (90, 40), (90, 60), (10, 60)];

    let result = get_rotate_crop_image(&img, &box_points);
    assert!(result.is_ok());

    let (_, is_vertical) = result.unwrap();
    assert!(!is_vertical);
}

#[test]
fn test_get_rotate_crop_image_wrong_point_count() {
    let img: RgbImage = ImageBuffer::new(20, 20);

    // Wrong number of points
    let box_points = vec![(0, 0), (10, 0), (10, 10)]; // Only 3 points

    let result = get_rotate_crop_image(&img, &box_points);
    assert!(result.is_err());
}

// ============================================================================
// get_image_parts Tests
// ============================================================================

#[test]
fn test_get_image_parts() {
    let width = 20;
    let height = 20;
    let mut img: RgbImage = ImageBuffer::new(width, height);
    // Fill with black
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]);
    }
    // Draw a white 5x5 square at 5,5
    for y in 5..10 {
        for x in 5..10 {
            img.put_pixel(x, y, Rgb([255, 255, 255]));
        }
    }

    let text_box = TextBox {
        bounds: [
            Coord { x: 5, y: 5 },
            Coord { x: 10, y: 5 },
            Coord { x: 10, y: 10 },
            Coord { x: 5, y: 10 },
        ],
        angle: None,
        text: None,
        box_score: 1.0,
        text_score: 1.0,
        span: None,
    };

    let parts = get_image_parts(&img, &[text_box]).unwrap();

    assert_eq!(parts.len(), 1);
    let part = &parts[0];
    assert_eq!(part.width(), 5);
    assert_eq!(part.height(), 5);

    // Check content - should be all white
    for pixel in part.pixels() {
        assert_eq!(*pixel, Rgb([255, 255, 255]));
    }
}

#[test]
fn test_get_image_parts_empty() {
    let img: RgbImage = ImageBuffer::new(100, 100);
    let text_boxes: Vec<TextBox> = vec![];

    let parts = get_image_parts(&img, &text_boxes).unwrap();
    assert!(parts.is_empty());
}

#[test]
fn test_get_image_parts_multiple() {
    let mut img: RgbImage = ImageBuffer::new(100, 100);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([128, 128, 128]);
    }

    let text_boxes = vec![
        TextBox {
            bounds: [
                Coord { x: 0, y: 0 },
                Coord { x: 20, y: 0 },
                Coord { x: 20, y: 10 },
                Coord { x: 0, y: 10 },
            ],
            angle: None,
            text: None,
            box_score: 1.0,
            text_score: 1.0,
            span: None,
        },
        TextBox {
            bounds: [
                Coord { x: 30, y: 30 },
                Coord { x: 80, y: 30 },
                Coord { x: 80, y: 50 },
                Coord { x: 30, y: 50 },
            ],
            angle: None,
            text: None,
            box_score: 1.0,
            text_score: 1.0,
            span: None,
        },
    ];

    let parts = get_image_parts(&img, &text_boxes).unwrap();
    assert_eq!(parts.len(), 2);
}

// ============================================================================
// rotate_images_by_angle Tests
// ============================================================================

#[test]
fn test_rotate_images_by_angle_no_rotation() {
    let mut img: RgbImage = ImageBuffer::new(10, 5);
    img.put_pixel(0, 0, Rgb([255, 0, 0]));

    let text_boxes = vec![TextBox {
        bounds: [Coord { x: 0, y: 0 }; 4],
        angle: None, // No angle specified
        text: None,
        box_score: 1.0,
        text_score: 1.0,
        span: None,
    }];

    let result = rotate_images_by_angle(&[img.clone()], &text_boxes).unwrap();
    assert_eq!(result.len(), 1);
    // Without angle, image should not be rotated
    assert_eq!(result[0].width(), 10);
    assert_eq!(result[0].height(), 5);
}

#[test]
fn test_rotate_images_by_angle_with_rotation() {
    let img: RgbImage = ImageBuffer::new(10, 5);

    let text_boxes = vec![TextBox {
        bounds: [Coord { x: 0, y: 0 }; 4],
        angle: Some(Orientation::Oriented90),
        text: None,
        box_score: 1.0,
        text_score: 1.0,
        span: None,
    }];

    let result = rotate_images_by_angle(&[img], &text_boxes).unwrap();
    assert_eq!(result.len(), 1);
    // 90 degree rotation swaps dimensions
    assert_eq!(result[0].width(), 5);
    assert_eq!(result[0].height(), 10);
}

#[test]
fn test_rotate_images_by_angle_mixed() {
    let img1: RgbImage = ImageBuffer::new(10, 5);
    let img2: RgbImage = ImageBuffer::new(8, 4);

    let text_boxes = vec![
        TextBox {
            bounds: [Coord { x: 0, y: 0 }; 4],
            angle: Some(Orientation::Oriented0),
            text: None,
            box_score: 1.0,
            text_score: 1.0,
            span: None,
        },
        TextBox {
            bounds: [Coord { x: 0, y: 0 }; 4],
            angle: Some(Orientation::Oriented180),
            text: None,
            box_score: 1.0,
            text_score: 1.0,
            span: None,
        },
    ];

    let result = rotate_images_by_angle(&[img1, img2], &text_boxes).unwrap();
    assert_eq!(result.len(), 2);

    // First image: no rotation (0 degrees)
    assert_eq!(result[0].width(), 10);
    assert_eq!(result[0].height(), 5);

    // Second image: 180 degree rotation (dimensions unchanged)
    assert_eq!(result[1].width(), 8);
    assert_eq!(result[1].height(), 4);
}
