//! Utility functions for bounding box operations.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::document::bounds::Bounds;
use crate::utils::error::BoxError;
use crate::utils::lang_utils::Directionality;
use geo::{Area, BooleanOps, Coord, LineString, Polygon};
use geo_clipper::Clipper;

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
#[must_use]
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
#[must_use]
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
#[must_use]
pub fn apply_nms(
    mut detections: Vec<([Coord<i32>; 4], usize, f32)>,
    nms_threshold: f32,
) -> Vec<([Coord<i32>; 4], usize, f32)> {
    if detections.is_empty() {
        return detections;
    }

    detections.sort_by(|a, b| b.2.total_cmp(&a.2));

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
) -> Result<Vec<Coord<i32>>, BoxError> {
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
        None => Err(BoxError::OffsetFailed),
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
#[must_use]
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

/// Determines the natural reading order of bounding boxes using a graph-based approach.
///
/// This function establishes a partial ordering of text boxes based on their
/// spatial relationships, then performs a topological sort to produce a linear
/// reading sequence that respects the natural reading flow based on the
/// specified text directionality.
///
/// # Arguments
///
/// * `bounds_list` - Slice of bounding boxes to order
/// * `directionality` - Text direction (LTR or RTL) affecting horizontal ordering
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
/// 2. Edges are added based on spatial relationships (above/left-of for LTR, above/right-of for RTL)
/// 3. Topological sort using Kahn's algorithm with tie-breaking by position
/// 4. Handle cycles by appending remaining nodes in spatial order
///
/// # Note
///
/// Returns 1-indexed positions to match common document labeling conventions.
#[must_use]
pub fn graph_based_reading_order(
    bounds_list: &[Bounds],
    directionality: Directionality,
) -> Vec<usize> {
    if bounds_list.is_empty() {
        return Vec::new();
    }

    let n = bounds_list.len();

    let mut graph: Vec<Vec<usize>> = vec![Vec::with_capacity(n / 4); n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }

            if should_come_before(&bounds_list[i], &bounds_list[j], directionality) {
                graph[i].push(j);
                in_degree[j] += 1;
            }
        }
    }

    let mut heap: BinaryHeap<Reverse<(i32, i32, usize)>> = BinaryHeap::with_capacity(n);
    let mut result: Vec<usize> = Vec::with_capacity(n);

    let x_factor = match directionality {
        Directionality::Ltr => 1,
        Directionality::Rtl => -1,
    };

    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            let bounds = &bounds_list[i];
            heap.push(Reverse((
                bounds.center_y(),
                bounds.center_x() * x_factor,
                i,
            )));
        }
    }

    while let Some(Reverse((_, _, node))) = heap.pop() {
        result.push(node);

        for &neighbor in &graph[node] {
            in_degree[neighbor] -= 1;
            if in_degree[neighbor] == 0 {
                let bounds = &bounds_list[neighbor];
                heap.push(Reverse((
                    bounds.center_y(),
                    bounds.center_x() * x_factor,
                    neighbor,
                )));
            }
        }
    }

    if result.len() < n {
        let in_result: std::collections::HashSet<usize> = result.iter().copied().collect();
        let mut remaining: Vec<usize> = (0..n).filter(|i| !in_result.contains(i)).collect();
        remaining.sort_by(|&a, &b| {
            let bounds_a = &bounds_list[a];
            let bounds_b = &bounds_list[b];

            match bounds_a.center_y().cmp(&bounds_b.center_y()) {
                std::cmp::Ordering::Equal => match directionality {
                    Directionality::Ltr => bounds_a.center_x().cmp(&bounds_b.center_x()),
                    Directionality::Rtl => bounds_b.center_x().cmp(&bounds_a.center_x()),
                },
                other => other,
            }
        });
        result.extend(remaining);
    }

    result.into_iter().map(|i| i + 1).collect()
}

/// Determines if one box should be read before another based on spatial position.
///
/// This function implements the heuristics for establishing reading order:
/// - A box clearly above another comes first
/// - For boxes on roughly the same line:
///   - LTR: the leftmost comes first
///   - RTL: the rightmost comes first
///
/// # Arguments
///
/// * `bounds_i` - Bounds for the first box
/// * `bounds_j` - Bounds for the second box
/// * `directionality` - Text direction (LTR or RTL)
///
/// # Returns
///
/// `true` if box i should be read before box j, `false` otherwise.
fn should_come_before(
    bounds_i: &Bounds,
    bounds_j: &Bounds,
    directionality: Directionality,
) -> bool {
    let avg_height = (bounds_i.height() + bounds_j.height()) / 2;

    let vertical_separation = bounds_j.center_y() - bounds_i.center_y();

    if vertical_separation > (avg_height as f32 * 0.3) as i32 {
        return true;
    }

    let vertical_threshold = (avg_height as f32 * 0.5) as i32;
    if vertical_separation.abs() <= vertical_threshold {
        match directionality {
            Directionality::Ltr => {
                if bounds_i.center_x() < bounds_j.center_x()
                    && bounds_i.right() <= bounds_j.left() + (avg_height / 4)
                {
                    return true;
                }
            }
            Directionality::Rtl => {
                if bounds_i.center_x() > bounds_j.center_x()
                    && bounds_i.left() >= bounds_j.right() - (avg_height / 4)
                {
                    return true;
                }
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::text_box::TextBox;

    // calculate_iou Tests

    #[test]
    fn test_calculate_iou_identical() {
        let box1 = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let iou = calculate_iou(&box1, &box1);
        assert!((iou - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_iou_disjoint() {
        let box1 = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = vec![
            Coord { x: 20, y: 20 },
            Coord { x: 30, y: 20 },
            Coord { x: 30, y: 30 },
            Coord { x: 20, y: 30 },
        ];
        let iou = calculate_iou(&box1, &box2);
        assert!((iou - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_iou_partial() {
        let box1 = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = vec![
            Coord { x: 5, y: 0 },
            Coord { x: 15, y: 0 },
            Coord { x: 15, y: 10 },
            Coord { x: 5, y: 10 },
        ];
        let iou = calculate_iou(&box1, &box2);
        assert!((iou - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_iou_zero_area_box() {
        let box1 = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 0, y: 0 },
        ];
        let box2 = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let iou = calculate_iou(&box1, &box2);
        assert_eq!(iou, 0.0);
    }

    #[test]
    fn test_calculate_iou_negative_coordinates() {
        let box1 = vec![
            Coord { x: -10, y: -10 },
            Coord { x: 0, y: -10 },
            Coord { x: 0, y: 0 },
            Coord { x: -10, y: 0 },
        ];
        let box2 = vec![
            Coord { x: -5, y: -10 },
            Coord { x: 5, y: -10 },
            Coord { x: 5, y: 0 },
            Coord { x: -5, y: 0 },
        ];
        let iou = calculate_iou(&box1, &box2);
        assert!((iou - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_iou_one_inside_other() {
        let large_box = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 20, y: 0 },
            Coord { x: 20, y: 20 },
            Coord { x: 0, y: 20 },
        ];
        let small_box = vec![
            Coord { x: 5, y: 5 },
            Coord { x: 15, y: 5 },
            Coord { x: 15, y: 15 },
            Coord { x: 5, y: 15 },
        ];
        let iou = calculate_iou(&large_box, &small_box);
        assert!((iou - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_iou_touching_boxes() {
        let box1 = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = vec![
            Coord { x: 10, y: 0 },
            Coord { x: 20, y: 0 },
            Coord { x: 20, y: 10 },
            Coord { x: 10, y: 10 },
        ];
        let iou = calculate_iou(&box1, &box2);
        assert!(iou < 1e-6);
    }

    // calculate_overlap Tests

    #[test]
    fn test_calculate_overlap() {
        let box1 = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = vec![
            Coord { x: 5, y: 0 },
            Coord { x: 15, y: 0 },
            Coord { x: 15, y: 10 },
            Coord { x: 5, y: 10 },
        ];
        let overlap = calculate_overlap(&box1, &box2);
        assert!((overlap - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_overlap_complete_containment() {
        let small_box = vec![
            Coord { x: 5, y: 5 },
            Coord { x: 10, y: 5 },
            Coord { x: 10, y: 10 },
            Coord { x: 5, y: 10 },
        ];
        let large_box = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 20, y: 0 },
            Coord { x: 20, y: 20 },
            Coord { x: 0, y: 20 },
        ];
        let overlap = calculate_overlap(&small_box, &large_box);
        assert!((overlap - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_overlap_zero_area_box() {
        let zero_area = vec![
            Coord { x: 5, y: 5 },
            Coord { x: 5, y: 5 },
            Coord { x: 5, y: 5 },
            Coord { x: 5, y: 5 },
        ];
        let normal_box = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let overlap = calculate_overlap(&zero_area, &normal_box);
        assert_eq!(overlap, 0.0);
    }

    #[test]
    fn test_calculate_overlap_no_intersection() {
        let box1 = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = vec![
            Coord { x: 100, y: 100 },
            Coord { x: 110, y: 100 },
            Coord { x: 110, y: 110 },
            Coord { x: 100, y: 110 },
        ];
        let overlap = calculate_overlap(&box1, &box2);
        assert_eq!(overlap, 0.0);
    }

    // apply_nms Tests

    #[test]
    fn test_apply_nms() {
        let box1 = [
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = [
            Coord { x: 1, y: 1 },
            Coord { x: 11, y: 1 },
            Coord { x: 11, y: 11 },
            Coord { x: 1, y: 11 },
        ];
        let box3 = [
            Coord { x: 20, y: 20 },
            Coord { x: 30, y: 20 },
            Coord { x: 30, y: 30 },
            Coord { x: 20, y: 30 },
        ];

        let detections = vec![(box1, 0, 0.9), (box2, 0, 0.8), (box3, 0, 0.7)];
        let result = apply_nms(detections, 0.5);

        assert_eq!(result.len(), 2);
        assert!(result.iter().any(|d| d.2 == 0.9));
        assert!(result.iter().any(|d| d.2 == 0.7));
    }

    #[test]
    fn test_apply_nms_empty() {
        let detections: Vec<([Coord<i32>; 4], usize, f32)> = vec![];
        let result = apply_nms(detections, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_apply_nms_single_detection() {
        let box1 = [
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let detections = vec![(box1, 0, 0.9)];
        let result = apply_nms(detections, 0.5);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_apply_nms_different_classes() {
        let box1 = [
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = [
            Coord { x: 1, y: 1 },
            Coord { x: 11, y: 1 },
            Coord { x: 11, y: 11 },
            Coord { x: 1, y: 11 },
        ];
        let detections = vec![(box1, 0, 0.9), (box2, 1, 0.85)];
        let result = apply_nms(detections, 0.5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_apply_nms_same_scores() {
        let box1 = [
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = [
            Coord { x: 1, y: 1 },
            Coord { x: 11, y: 1 },
            Coord { x: 11, y: 11 },
            Coord { x: 1, y: 11 },
        ];
        let detections = vec![(box1, 0, 0.9), (box2, 0, 0.9)];
        let result = apply_nms(detections, 0.5);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_apply_nms_threshold_boundary() {
        let box1 = [
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = [
            Coord { x: 5, y: 0 },
            Coord { x: 15, y: 0 },
            Coord { x: 15, y: 10 },
            Coord { x: 5, y: 10 },
        ];
        let detections = vec![(box1, 0, 0.9), (box2, 0, 0.8)];
        let result = apply_nms(detections, 0.33);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_apply_nms_zero_threshold() {
        let box1 = [
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = [
            Coord { x: 9, y: 0 },
            Coord { x: 19, y: 0 },
            Coord { x: 19, y: 10 },
            Coord { x: 9, y: 10 },
        ];
        let detections = vec![(box1, 0, 0.9), (box2, 0, 0.8)];
        let result = apply_nms(detections, 0.0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_apply_nms_one_threshold() {
        let box1 = [
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let box2 = [
            Coord { x: 5, y: 0 },
            Coord { x: 15, y: 0 },
            Coord { x: 15, y: 10 },
            Coord { x: 5, y: 10 },
        ];
        let detections = vec![(box1, 0, 0.9), (box2, 0, 0.8)];
        let result = apply_nms(detections, 1.0);
        assert_eq!(result.len(), 2);
    }

    // get_min_boxes Tests

    #[test]
    fn test_get_min_boxes() {
        let points = [
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];

        let (sorted_box, max_side) = get_min_boxes(&points);
        assert!((max_side - 10.0).abs() < 1e-6);

        assert_eq!(sorted_box[0], Coord { x: 0, y: 0 });
        assert_eq!(sorted_box[1], Coord { x: 10, y: 0 });
        assert_eq!(sorted_box[2], Coord { x: 10, y: 10 });
        assert_eq!(sorted_box[3], Coord { x: 0, y: 10 });
    }

    #[test]
    fn test_get_min_boxes_rectangle() {
        let points = [
            Coord { x: 0, y: 0 },
            Coord { x: 20, y: 0 },
            Coord { x: 20, y: 10 },
            Coord { x: 0, y: 10 },
        ];
        let (_, max_side) = get_min_boxes(&points);
        assert!((max_side - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_min_boxes_single_point() {
        let points = [Coord { x: 5, y: 5 }; 4];
        let (_, max_side) = get_min_boxes(&points);
        assert!((max_side - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_min_boxes_negative_coords() {
        let points = [
            Coord { x: -10, y: -10 },
            Coord { x: 0, y: -10 },
            Coord { x: 0, y: 0 },
            Coord { x: -10, y: 0 },
        ];
        let (sorted_box, max_side) = get_min_boxes(&points);
        assert!((max_side - 10.0).abs() < 1e-6);
        assert_eq!(sorted_box.len(), 4);
    }

    // unclip_box Tests

    #[test]
    fn test_unclip_box() {
        let points = [
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];

        let result = unclip_box(&points, 2.0).unwrap();

        let min_x = result.iter().map(|p| p.x).min().unwrap();
        let min_y = result.iter().map(|p| p.y).min().unwrap();
        let max_x = result.iter().map(|p| p.x).max().unwrap();
        let max_y = result.iter().map(|p| p.y).max().unwrap();

        assert!(min_x < 0);
        assert!(min_y < 0);
        assert!(max_x > 10);
        assert!(max_y > 10);
    }

    #[test]
    fn test_unclip_box_zero_ratio() {
        let points = [
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ];

        let result = unclip_box(&points, 0.0).unwrap();

        let min_x = result.iter().map(|p| p.x).min().unwrap();
        let max_x = result.iter().map(|p| p.x).max().unwrap();
        let min_y = result.iter().map(|p| p.y).min().unwrap();
        let max_y = result.iter().map(|p| p.y).max().unwrap();

        assert!(min_x >= 0);
        assert!(min_y >= 0);
        assert!(max_x <= 10);
        assert!(max_y <= 10);
    }

    #[test]
    fn test_unclip_box_large_ratio() {
        let points = [
            Coord { x: 10, y: 10 },
            Coord { x: 20, y: 10 },
            Coord { x: 20, y: 20 },
            Coord { x: 10, y: 20 },
        ];

        let result = unclip_box(&points, 5.0).unwrap();

        let min_x = result.iter().map(|p| p.x).min().unwrap();
        let max_x = result.iter().map(|p| p.x).max().unwrap();

        assert!(min_x < 5);
        assert!(max_x > 25);
    }

    #[test]
    fn test_unclip_box_negative_coords() {
        let points = [
            Coord { x: -10, y: -10 },
            Coord { x: 0, y: -10 },
            Coord { x: 0, y: 0 },
            Coord { x: -10, y: 0 },
        ];

        let result = unclip_box(&points, 1.5).unwrap();

        let min_x = result.iter().map(|p| p.x).min().unwrap();
        let min_y = result.iter().map(|p| p.y).min().unwrap();

        assert!(min_x < -10);
        assert!(min_y < -10);
    }

    // graph_based_reading_order Tests

    fn make_bounds(coords: [Coord<i32>; 4]) -> Bounds {
        Bounds::new(coords)
    }

    #[test]
    fn test_reading_order_empty() {
        let boxes: Vec<Bounds> = vec![];
        let result = graph_based_reading_order(&boxes, Directionality::Ltr);
        assert!(result.is_empty());
    }

    #[test]
    fn test_reading_order_single() {
        let boxes = vec![make_bounds([
            Coord { x: 0, y: 0 },
            Coord { x: 10, y: 0 },
            Coord { x: 10, y: 10 },
            Coord { x: 0, y: 10 },
        ])];
        let result = graph_based_reading_order(&boxes, Directionality::Ltr);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_reading_order_left_to_right() {
        let boxes = vec![
            make_bounds([
                Coord { x: 200, y: 0 },
                Coord { x: 300, y: 0 },
                Coord { x: 300, y: 50 },
                Coord { x: 200, y: 50 },
            ]),
            make_bounds([
                Coord { x: 0, y: 0 },
                Coord { x: 100, y: 0 },
                Coord { x: 100, y: 50 },
                Coord { x: 0, y: 50 },
            ]),
            make_bounds([
                Coord { x: 100, y: 0 },
                Coord { x: 200, y: 0 },
                Coord { x: 200, y: 50 },
                Coord { x: 100, y: 50 },
            ]),
        ];
        let result = graph_based_reading_order(&boxes, Directionality::Ltr);
        assert_eq!(result, vec![2, 3, 1]);
    }

    #[test]
    fn test_reading_order_top_to_bottom() {
        let boxes = vec![
            make_bounds([
                Coord { x: 0, y: 100 },
                Coord { x: 100, y: 100 },
                Coord { x: 100, y: 150 },
                Coord { x: 0, y: 150 },
            ]),
            make_bounds([
                Coord { x: 0, y: 200 },
                Coord { x: 100, y: 200 },
                Coord { x: 100, y: 250 },
                Coord { x: 0, y: 250 },
            ]),
            make_bounds([
                Coord { x: 0, y: 0 },
                Coord { x: 100, y: 0 },
                Coord { x: 100, y: 50 },
                Coord { x: 0, y: 50 },
            ]),
        ];
        let result = graph_based_reading_order(&boxes, Directionality::Ltr);
        assert_eq!(result, vec![3, 1, 2]);
    }

    #[test]
    fn test_reading_order_two_columns() {
        let boxes = vec![
            make_bounds([
                Coord { x: 0, y: 0 },
                Coord { x: 100, y: 0 },
                Coord { x: 100, y: 30 },
                Coord { x: 0, y: 30 },
            ]),
            make_bounds([
                Coord { x: 0, y: 40 },
                Coord { x: 100, y: 40 },
                Coord { x: 100, y: 70 },
                Coord { x: 0, y: 70 },
            ]),
            make_bounds([
                Coord { x: 150, y: 0 },
                Coord { x: 250, y: 0 },
                Coord { x: 250, y: 30 },
                Coord { x: 150, y: 30 },
            ]),
            make_bounds([
                Coord { x: 150, y: 40 },
                Coord { x: 250, y: 40 },
                Coord { x: 250, y: 70 },
                Coord { x: 150, y: 70 },
            ]),
        ];
        let result = graph_based_reading_order(&boxes, Directionality::Ltr);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_reading_order_with_text_boxes() {
        let text_boxes: Vec<TextBox> = vec![
            TextBox {
                bounds: Bounds::new([
                    Coord { x: 100, y: 0 },
                    Coord { x: 200, y: 0 },
                    Coord { x: 200, y: 30 },
                    Coord { x: 100, y: 30 },
                ]),
                angle: None,
                text: Some("Second".into()),
                box_score: 0.9,
                text_score: 0.9,
                span: None,
            },
            TextBox {
                bounds: Bounds::new([
                    Coord { x: 0, y: 0 },
                    Coord { x: 100, y: 0 },
                    Coord { x: 100, y: 30 },
                    Coord { x: 0, y: 30 },
                ]),
                angle: None,
                text: Some("First".into()),
                box_score: 0.9,
                text_score: 0.9,
                span: None,
            },
        ];
        let bounds: Vec<_> = text_boxes.iter().map(|t| t.bounds).collect();
        let result = graph_based_reading_order(&bounds, Directionality::Ltr);
        assert_eq!(result, vec![2, 1]);
    }
}
