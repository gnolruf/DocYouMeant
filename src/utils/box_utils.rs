//! Utility functions for bounding box operations.

use crate::document::bounds::Bounds;
use geo::{Area, BooleanOps, Coord, LineString, Polygon};
use geo_clipper::Clipper;
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::drawing::draw_polygon_mut;
use imageproc::point::Point as ImageProcPoint;
use ndarray::Array2;

/// Calculates the Intersection over Union (IoU) between two polygons.
///
/// IoU is a measure of overlap between two bounding boxes, commonly used
/// in object detection to evaluate prediction accuracy or for non-maximum
/// suppression.
///
/// # Arguments
///
/// * `points1` - Slice of coordinates defining the first polygon's vertices
/// * `points2` - Slice of coordinates defining the second polygon's vertices
///
/// # Returns
///
/// A value between 0.0 and 1.0 where:
/// - 0.0 indicates no overlap
/// - 1.0 indicates perfect overlap
///
/// Returns 0.0 if either polygon has zero area.
#[inline]
pub fn calculate_iou(points1: &[Coord<i32>], points2: &[Coord<i32>]) -> f32 {
    let poly1 = Polygon::new(
        LineString::from_iter(points1.iter().map(|c| Coord {
            x: c.x as f32,
            y: c.y as f32,
        })),
        vec![],
    );
    let poly2 = Polygon::new(
        LineString::from_iter(points2.iter().map(|c| Coord {
            x: c.x as f32,
            y: c.y as f32,
        })),
        vec![],
    );

    let area1 = poly1.unsigned_area();
    let area2 = poly2.unsigned_area();

    if area1 == 0.0 || area2 == 0.0 {
        return 0.0;
    }

    let intersection = BooleanOps::intersection(&poly1, &poly2);
    let intersection_area = intersection.unsigned_area();

    let union = area1 + area2 - intersection_area;

    if union > 0.0 {
        intersection_area / union
    } else {
        0.0
    }
}

/// Calculates the overlap ratio of the first polygon covered by the second polygon.
///
/// Unlike IoU, this function measures how much of `poly1` is covered by `poly2`,
/// which is useful for determining if a smaller box is contained within a larger one.
///
/// # Arguments
///
/// * `poly1` - Slice of coordinates defining the first polygon (the reference polygon)
/// * `poly2` - Slice of coordinates defining the second polygon
///
/// # Returns
///
/// A value between 0.0 and 1.0 representing the fraction of `poly1`'s area
/// that overlaps with `poly2`. Returns 0.0 if `poly1` has zero area.
#[inline]
pub fn calculate_overlap(poly1: &[Coord<i32>], poly2: &[Coord<i32>]) -> f32 {
    let polygon1 = Polygon::new(
        LineString::from_iter(poly1.iter().map(|c| Coord {
            x: c.x as f64,
            y: c.y as f64,
        })),
        vec![],
    );
    let polygon2 = Polygon::new(
        LineString::from_iter(poly2.iter().map(|c| Coord {
            x: c.x as f64,
            y: c.y as f64,
        })),
        vec![],
    );

    let poly1_area = polygon1.unsigned_area();

    if poly1_area == 0.0 {
        return 0.0;
    }

    let intersection = BooleanOps::intersection(&polygon1, &polygon2);
    let intersection_area = intersection.unsigned_area();

    (intersection_area / poly1_area) as f32
}

/// Applies Non-Maximum Suppression (NMS) to filter overlapping detections.
///
/// NMS is a post-processing technique used in object detection to remove
/// redundant overlapping bounding boxes, keeping only the most confident
/// detection for each object.
///
/// # Algorithm
///
/// 1. Sort detections by confidence score in descending order
/// 2. For each detection (starting from highest confidence):
///    - Keep the detection if not suppressed
///    - Suppress all lower-confidence detections of the same class
///      that have IoU above the threshold
///
/// # Arguments
///
/// * `detections` - Vector of tuples containing:
///   - `[Coord<i32>; 4]`: Four corner points of the bounding box
///   - `usize`: Class label/index
///   - `f32`: Confidence score
/// * `nms_threshold` - IoU threshold above which overlapping boxes are suppressed
///
/// # Returns
///
/// Filtered vector of detections with redundant boxes removed.
///
/// # Note
///
/// Only boxes with the same class label are compared for suppression.
pub fn apply_nms(
    mut detections: Vec<([Coord<i32>; 4], usize, f32)>,
    nms_threshold: f32,
) -> Vec<([Coord<i32>; 4], usize, f32)> {
    if detections.is_empty() {
        return detections;
    }

    detections.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(detections[i]);

        for j in (i + 1)..detections.len() {
            if suppressed[j] || detections[i].1 != detections[j].1 {
                continue;
            }

            let iou = calculate_iou(&detections[i].0, &detections[j].0);
            if iou > nms_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// Expands a bounding box outward by a ratio proportional to its perimeter.
///
/// This function is commonly used in text detection to expand tight text boxes
/// to ensure complete text coverage, compensating for the shrinkage applied
/// during label generation in training.
///
/// # Arguments
///
/// * `box_points` - Four corner points of the original bounding box
/// * `unclip_ratio` - Expansion ratio (typically 1.5-2.0 for text detection)
///
/// # Returns
///
/// * `Ok(Vec<Coord<i32>>)` - Expanded polygon vertices
/// * `Err` - If the polygon offset operation fails
///
/// # Algorithm
///
/// The expansion distance is calculated as: `area * unclip_ratio / perimeter`
///
/// This ensures that boxes with different aspect ratios expand proportionally.
pub fn unclip_box(
    box_points: &[Coord<i32>; 4],
    unclip_ratio: f32,
) -> Result<Vec<Coord<i32>>, Box<dyn std::error::Error + Send + Sync>> {
    let mut area = 0.0;
    let mut dist = 0.0;

    for i in 0..box_points.len() {
        let next = (i + 1) % box_points.len();
        area += (box_points[i].x as f32) * (box_points[next].y as f32)
            - (box_points[i].y as f32) * (box_points[next].x as f32);
        dist += (((box_points[i].x - box_points[next].x).pow(2) as f32)
            + ((box_points[i].y - box_points[next].y).pow(2) as f32))
            .sqrt();
    }

    area = (area / 2.0).abs();
    let distance = area * unclip_ratio / dist;

    let line_string = LineString::from_iter(box_points.iter().map(|p| Coord {
        x: p.x as f64,
        y: p.y as f64,
    }));
    let polygon = Polygon::new(line_string, vec![]);

    let result = polygon.offset(
        distance as f64,
        geo_clipper::JoinType::Round(0.1),
        geo_clipper::EndType::ClosedPolygon,
        1000.0,
    );

    match result.0.first() {
        Some(poly) => {
            let exterior = &poly.exterior();
            let points: Vec<Coord<i32>> = exterior
                .0
                .iter()
                .map(|coord| Coord {
                    x: coord.x.round() as i32,
                    y: coord.y.round() as i32,
                })
                .collect();
            Ok(points)
        }
        None => Ok(box_points.to_vec()),
    }
}

/// Reorders corner points into a consistent top-left, top-right, bottom-right, bottom-left order.
///
/// This function normalizes the ordering of quadrilateral vertices to ensure
/// consistent processing regardless of the original point order. It also
/// calculates the maximum side length of the box.
///
/// # Arguments
///
/// * `corner_points` - Four corner points in arbitrary order
///
/// # Returns
///
/// A tuple containing:
/// - `[Coord<i32>; 4]`: Points reordered as [top-left, top-right, bottom-right, bottom-left]
/// - `f32`: Maximum side length of the box
///
/// # Ordering Algorithm
///
/// 1. Sort points by x-coordinate
/// 2. Assign left two points (index1, index4) based on y-coordinate
/// 3. Assign right two points (index2, index3) based on y-coordinate
#[inline]
pub fn get_min_boxes(corner_points: &[Coord<i32>; 4]) -> ([Coord<i32>; 4], f32) {
    let dx1 = (corner_points[1].x - corner_points[0].x) as f32;
    let dy1 = (corner_points[1].y - corner_points[0].y) as f32;
    let side1_len = (dx1 * dx1 + dy1 * dy1).sqrt();

    let dx2 = (corner_points[2].x - corner_points[1].x) as f32;
    let dy2 = (corner_points[2].y - corner_points[1].y) as f32;
    let side2_len = (dx2 * dx2 + dy2 * dy2).sqrt();

    let max_side_len = side1_len.max(side2_len);

    let mut box_points = *corner_points;

    box_points.sort_by(|a, b| {
        if (a.x - b.x).abs() == 0 {
            a.y.cmp(&b.y)
        } else {
            a.x.cmp(&b.x)
        }
    });

    let (index1, index4) = if box_points[1].y > box_points[0].y {
        (0, 1)
    } else {
        (1, 0)
    };

    let (index2, index3) = if box_points[3].y > box_points[2].y {
        (2, 3)
    } else {
        (3, 2)
    };

    let min_box = [
        box_points[index1],
        box_points[index2],
        box_points[index3],
        box_points[index4],
    ];

    (min_box, max_side_len)
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
/// * `Ok(f32)` - Average prediction value within the box (0.0 to 1.0 for probability maps)
/// * `Err` - If an error occurs during processing
///
/// Returns 0.0 if the prediction array is empty or no pixels fall within the box.
///
/// # Algorithm
///
/// 1. Compute the axis-aligned bounding rectangle of the box
/// 2. Create a binary mask by rasterizing the polygon
/// 3. Sum prediction values where the mask is non-zero
/// 4. Return the average
pub fn box_score(
    boxes: &[Coord<i32>; 4],
    pred: &Array2<f32>,
) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
    let width = pred.ncols();
    let height = pred.nrows();

    if width == 0 || height == 0 {
        return Ok(0.0);
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

    let mut mask: GrayImage = ImageBuffer::new(mask_width, mask_height);

    let polygon_points: Vec<ImageProcPoint<i32>> = boxes
        .iter()
        .map(|point| ImageProcPoint::new(point.x - clamp_min_x, point.y - clamp_min_y))
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
        Ok(sum / count as f32)
    } else {
        Ok(0.0)
    }
}

/// Determines the natural reading order of bounding boxes using a graph-based approach.
///
/// This function establishes a partial ordering of text boxes based on their
/// spatial relationships, then performs a topological sort to produce a linear
/// reading sequence that respects the natural left-to-right, top-to-bottom
/// reading flow.
///
/// # Type Parameters
///
/// * `T` - Any type implementing the [`HasBounds`] trait
///
/// # Arguments
///
/// * `boxes` - Slice of items with bounding box information
///
/// # Returns
///
/// A vector of 1-indexed positions indicating the reading order.
/// For example, `[2, 1, 3]` means the second box should be read first,
/// then the first box, then the third box.
///
/// # Algorithm
///
/// 1. Build a directed graph where edge (i, j) means box i should be read before box j
/// 2. Edges are added based on spatial relationships (above/left-of)
/// 3. Topological sort using Kahn's algorithm with tie-breaking by position
/// 4. Handle cycles by appending remaining nodes in spatial order
///
/// # Note
///
/// Returns 1-indexed positions to match common document labeling conventions.
pub fn graph_based_reading_order<T>(boxes: &[T]) -> Vec<usize>
where
    T: HasBounds,
{
    if boxes.is_empty() {
        return Vec::new();
    }

    let n = boxes.len();

    let bounds_list: Vec<&Bounds> = boxes.iter().map(|b| b.get_bounds()).collect();

    let mut graph: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }

            if should_come_before(bounds_list[i], bounds_list[j]) {
                graph[i].push(j);
                in_degree[j] += 1;
            }
        }
    }

    // Topological sort using Kahn's algorithm
    let mut queue: Vec<usize> = Vec::new();
    let mut result: Vec<usize> = Vec::new();

    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push(i);
        }
    }

    queue.sort_by(|&a, &b| {
        let bounds_a = bounds_list[a];
        let bounds_b = bounds_list[b];

        match bounds_a.center_y().cmp(&bounds_b.center_y()) {
            std::cmp::Ordering::Equal => bounds_a.center_x().cmp(&bounds_b.center_x()),
            other => other,
        }
    });

    while let Some(node) = queue.pop() {
        result.push(node);

        for &neighbor in &graph[node] {
            in_degree[neighbor] -= 1;
            if in_degree[neighbor] == 0 {
                queue.push(neighbor);

                queue.sort_by(|&a, &b| {
                    let bounds_a = bounds_list[a];
                    let bounds_b = bounds_list[b];

                    match bounds_a.center_y().cmp(&bounds_b.center_y()) {
                        std::cmp::Ordering::Equal => bounds_a.center_x().cmp(&bounds_b.center_x()),
                        other => other,
                    }
                });
            }
        }
    }

    if result.len() < n {
        let mut remaining: Vec<usize> = (0..n).filter(|i| !result.contains(i)).collect();
        remaining.sort_by(|&a, &b| {
            let bounds_a = bounds_list[a];
            let bounds_b = bounds_list[b];

            match bounds_a.center_y().cmp(&bounds_b.center_y()) {
                std::cmp::Ordering::Equal => bounds_a.center_x().cmp(&bounds_b.center_x()),
                other => other,
            }
        });
        result.extend(remaining);
    }

    result.into_iter().map(|i| i + 1).collect()
}

/// Trait for types that have bounding box coordinates.
///
/// Implement this trait for any type that has spatial bounds to enable
/// use with [`graph_based_reading_order`] and other spatial algorithms.
///
/// # Example
///
/// ```ignore
/// use crate::document::bounds::Bounds;
///
/// struct TextBox {
///     text: String,
///     bounds: Bounds,
/// }
///
/// impl HasBounds for TextBox {
///     fn get_bounds(&self) -> &Bounds {
///         &self.bounds
///     }
/// }
/// ```
pub trait HasBounds {
    /// Returns a reference to the Bounds.
    fn get_bounds(&self) -> &Bounds;
}

/// Determines if one box should be read before another based on spatial position.
///
/// This function implements the heuristics for establishing reading order:
/// - A box clearly above another comes first
/// - For boxes on roughly the same line, the leftmost comes first
///
/// # Arguments
///
/// * `bounds_i` - Bounds for the first box
/// * `bounds_j` - Bounds for the second box
///
/// # Returns
///
/// `true` if box i should be read before box j, `false` otherwise.
fn should_come_before(bounds_i: &Bounds, bounds_j: &Bounds) -> bool {
    let avg_height = (bounds_i.height() + bounds_j.height()) / 2;

    let vertical_separation = bounds_j.center_y() - bounds_i.center_y();

    if vertical_separation > (avg_height as f32 * 0.3) as i32 {
        return true;
    }

    let vertical_threshold = (avg_height as f32 * 0.5) as i32;
    if vertical_separation.abs() <= vertical_threshold
        && bounds_i.center_x() < bounds_j.center_x()
        && bounds_i.right() <= bounds_j.left() + (avg_height / 4)
    {
        return true;
    }

    false
}
