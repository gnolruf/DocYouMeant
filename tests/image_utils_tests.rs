use docyoumeant::document::text_box::{Orientation, TextBox};
use docyoumeant::utils::image_utils::{
    add_image_padding, get_image_parts, get_rotate_crop_image, rotate_image,
    subtract_mean_normalize,
};
use geo::Coord;
use image::{ImageBuffer, Rgb, RgbImage};

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
