use docyoumeant::utils::box_utils::{
    apply_nms, calculate_iou, calculate_overlap, get_min_boxes, unclip_box,
};
use geo::Coord;

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
