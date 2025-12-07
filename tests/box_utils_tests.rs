use docyoumeant::document::bounds::Bounds;
use docyoumeant::document::text_box::TextBox;
use docyoumeant::utils::box_utils::{
    apply_nms, calculate_iou, calculate_overlap, get_min_boxes, graph_based_reading_order,
    unclip_box,
};
use geo::Coord;

// ============================================================================
// calculate_iou Tests
// ============================================================================

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
    // Box 1: 0,0 to 10,10 (Area 100)
    let box1 = vec![
        Coord { x: 0, y: 0 },
        Coord { x: 10, y: 0 },
        Coord { x: 10, y: 10 },
        Coord { x: 0, y: 10 },
    ];
    // Box 2: 5,0 to 15,10 (Area 100)
    // Intersection: 5,0 to 10,10 (Width 5, Height 10 -> Area 50)
    // Union: 100 + 100 - 50 = 150
    // IoU: 50 / 150 = 1/3
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
    // Box with zero area (line)
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
    // Intersection: -5,-10 to 0,0 = 5x10 = 50
    // Union: 100 + 100 - 50 = 150
    // IoU: 50/150 = 1/3
    let iou = calculate_iou(&box1, &box2);
    assert!((iou - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn test_calculate_iou_one_inside_other() {
    // Small box completely inside larger box
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
    // Intersection: 10x10 = 100
    // Union: 400 + 100 - 100 = 400
    // IoU: 100/400 = 0.25
    let iou = calculate_iou(&large_box, &small_box);
    assert!((iou - 0.25).abs() < 1e-6);
}

#[test]
fn test_calculate_iou_touching_boxes() {
    // Two boxes that share an edge but don't overlap area
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
    // Touching boxes have zero overlap
    assert!(iou < 1e-6);
}

// ============================================================================
// calculate_overlap Tests
// ============================================================================

#[test]
fn test_calculate_overlap() {
    // Box 1: 0,0 to 10,10 (Area 100)
    let box1 = vec![
        Coord { x: 0, y: 0 },
        Coord { x: 10, y: 0 },
        Coord { x: 10, y: 10 },
        Coord { x: 0, y: 10 },
    ];
    // Box 2: 5,0 to 15,10
    // Intersection: 50
    // Overlap (Intersection / Area1) = 50 / 100 = 0.5
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
    // Small box completely inside larger box
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
    // 100% of small box is inside large box
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

// ============================================================================
// apply_nms Tests
// ============================================================================

#[test]
fn test_apply_nms() {
    // Box 1: 0,0 to 10,10. Score 0.9. Class 0.
    let box1 = [
        Coord { x: 0, y: 0 },
        Coord { x: 10, y: 0 },
        Coord { x: 10, y: 10 },
        Coord { x: 0, y: 10 },
    ];

    // Box 2: 1,1 to 11,11. Score 0.8. Class 0. High overlap with Box 1.
    let box2 = [
        Coord { x: 1, y: 1 },
        Coord { x: 11, y: 1 },
        Coord { x: 11, y: 11 },
        Coord { x: 1, y: 11 },
    ];

    // Box 3: 20,20 to 30,30. Score 0.7. Class 0. No overlap.
    let box3 = [
        Coord { x: 20, y: 20 },
        Coord { x: 30, y: 20 },
        Coord { x: 30, y: 30 },
        Coord { x: 20, y: 30 },
    ];

    let detections = vec![(box1, 0, 0.9), (box2, 0, 0.8), (box3, 0, 0.7)];

    // Threshold 0.5. Box 1 and Box 2 overlap > 0.5. Box 1 has higher score, so Box 2 should be suppressed.
    let result = apply_nms(detections, 0.5);

    assert_eq!(result.len(), 2);
    // Should contain box1 and box3
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
    // High overlap but different classes - both should be kept
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
    // Same scores - first one (after sort) should win
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
    // IoU exactly at threshold
    let box1 = [
        Coord { x: 0, y: 0 },
        Coord { x: 10, y: 0 },
        Coord { x: 10, y: 10 },
        Coord { x: 0, y: 10 },
    ];
    // Box with 50% overlap
    let box2 = [
        Coord { x: 5, y: 0 },
        Coord { x: 15, y: 0 },
        Coord { x: 15, y: 10 },
        Coord { x: 5, y: 10 },
    ];
    let detections = vec![(box1, 0, 0.9), (box2, 0, 0.8)];
    // With threshold 0.33, should suppress box2
    let result = apply_nms(detections, 0.33);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_apply_nms_zero_threshold() {
    // Zero threshold - all overlapping boxes should be suppressed
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
    // Threshold of 1.0 - only identical boxes suppressed
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

// ============================================================================
// get_min_boxes Tests
// ============================================================================

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
    // Sorted box should be ordered correctly
    assert_eq!(sorted_box.len(), 4);
}

// ============================================================================
// unclip_box Tests
// ============================================================================

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

    // With zero ratio, box should be nearly the same
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

    // Significantly expanded
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

    // Should expand in all directions
    let min_x = result.iter().map(|p| p.x).min().unwrap();
    let min_y = result.iter().map(|p| p.y).min().unwrap();

    assert!(min_x < -10);
    assert!(min_y < -10);
}

// ============================================================================
// graph_based_reading_order Tests
// ============================================================================

fn make_bounds(coords: [Coord<i32>; 4]) -> Bounds {
    Bounds::new(coords)
}

#[test]
fn test_reading_order_empty() {
    let boxes: Vec<Bounds> = vec![];
    let result = graph_based_reading_order(&boxes);
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
    let result = graph_based_reading_order(&boxes);
    assert_eq!(result, vec![1]);
}

#[test]
fn test_reading_order_left_to_right() {
    // Three boxes on the same row, left to right
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
    let result = graph_based_reading_order(&boxes);
    // Should be: 2 (left), 3 (middle), 1 (right)
    assert_eq!(result, vec![2, 3, 1]);
}

#[test]
fn test_reading_order_top_to_bottom() {
    // Three boxes in a column, top to bottom
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
    let result = graph_based_reading_order(&boxes);
    // Should be ordered by y: 3 (top), 1 (middle), 2 (bottom)
    assert_eq!(result, vec![3, 1, 2]);
}

#[test]
fn test_reading_order_two_columns() {
    // Two columns of text
    let boxes = vec![
        // Left column, row 1
        make_bounds([
            Coord { x: 0, y: 0 },
            Coord { x: 100, y: 0 },
            Coord { x: 100, y: 30 },
            Coord { x: 0, y: 30 },
        ]),
        // Left column, row 2
        make_bounds([
            Coord { x: 0, y: 40 },
            Coord { x: 100, y: 40 },
            Coord { x: 100, y: 70 },
            Coord { x: 0, y: 70 },
        ]),
        // Right column, row 1
        make_bounds([
            Coord { x: 150, y: 0 },
            Coord { x: 250, y: 0 },
            Coord { x: 250, y: 30 },
            Coord { x: 150, y: 30 },
        ]),
        // Right column, row 2
        make_bounds([
            Coord { x: 150, y: 40 },
            Coord { x: 250, y: 40 },
            Coord { x: 250, y: 70 },
            Coord { x: 150, y: 70 },
        ]),
    ];
    let result = graph_based_reading_order(&boxes);
    // Reading order should respect spatial layout
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
    let result = graph_based_reading_order(&bounds);
    // Second box (index 1) comes first spatially
    assert_eq!(result, vec![2, 1]);
}
