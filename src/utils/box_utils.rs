use geo::{Area, BooleanOps, Coord, LineString, Polygon};
use geo_clipper::Clipper;
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::drawing::draw_polygon_mut;
use imageproc::point::Point as ImageProcPoint;
use ndarray::Array2;

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

/// Helper struct to store bounding box metrics for efficient spatial comparisons
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct BoxMetrics {
    center_x: i32,
    center_y: i32,
    top: i32,
    bottom: i32,
    left: i32,
    right: i32,
    height: i32,
    width: i32,
}

impl BoxMetrics {
    fn from_bounds(bounds: &[Coord<i32>]) -> Self {
        let top = bounds.iter().map(|c| c.y).min().unwrap_or(0);
        let bottom = bounds.iter().map(|c| c.y).max().unwrap_or(0);
        let left = bounds.iter().map(|c| c.x).min().unwrap_or(0);
        let right = bounds.iter().map(|c| c.x).max().unwrap_or(0);

        let height = (bottom - top).max(1);
        let width = (right - left).max(1);
        let center_x = (left + right) / 2;
        let center_y = (top + bottom) / 2;

        Self {
            center_x,
            center_y,
            top,
            bottom,
            left,
            right,
            height,
            width,
        }
    }
}

pub fn graph_based_reading_order<T>(boxes: &[T]) -> Vec<usize>
where
    T: HasBounds,
{
    if boxes.is_empty() {
        return Vec::new();
    }

    let n = boxes.len();

    let metrics: Vec<BoxMetrics> = boxes
        .iter()
        .map(|b| BoxMetrics::from_bounds(b.get_bounds()))
        .collect();

    let mut graph: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }

            if should_come_before_metrics(&metrics[i], &metrics[j]) {
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
        let metrics_a = &metrics[a];
        let metrics_b = &metrics[b];

        match metrics_a.center_y.cmp(&metrics_b.center_y) {
            std::cmp::Ordering::Equal => metrics_a.center_x.cmp(&metrics_b.center_x),
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
                    let metrics_a = &metrics[a];
                    let metrics_b = &metrics[b];

                    match metrics_a.center_y.cmp(&metrics_b.center_y) {
                        std::cmp::Ordering::Equal => metrics_a.center_x.cmp(&metrics_b.center_x),
                        other => other,
                    }
                });
            }
        }
    }

    if result.len() < n {
        let mut remaining: Vec<usize> = (0..n).filter(|i| !result.contains(i)).collect();
        remaining.sort_by(|&a, &b| {
            let metrics_a = &metrics[a];
            let metrics_b = &metrics[b];

            match metrics_a.center_y.cmp(&metrics_b.center_y) {
                std::cmp::Ordering::Equal => metrics_a.center_x.cmp(&metrics_b.center_x),
                other => other,
            }
        });
        result.extend(remaining);
    }

    result.into_iter().map(|i| i + 1).collect()
}

pub trait HasBounds {
    fn get_bounds(&self) -> &[Coord<i32>];
}

fn should_come_before_metrics(metrics_i: &BoxMetrics, metrics_j: &BoxMetrics) -> bool {
    let avg_height = (metrics_i.height + metrics_j.height) / 2;

    let vertical_separation = metrics_j.center_y - metrics_i.center_y;

    if vertical_separation > (avg_height as f32 * 0.3) as i32 {
        return true;
    }

    let vertical_threshold = (avg_height as f32 * 0.5) as i32;
    if vertical_separation.abs() <= vertical_threshold
        && metrics_i.center_x < metrics_j.center_x
        && metrics_i.right <= metrics_j.left + (avg_height / 4)
    {
        return true;
    }

    false
}
