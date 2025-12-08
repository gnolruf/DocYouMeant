use std::collections::HashMap;

use docyoumeant::document::bounds::Bounds;
use docyoumeant::document::content::{DocumentType, PageContent};
use docyoumeant::document::layout_box::{LayoutBox, LayoutClass};
use docyoumeant::document::text_box::{DocumentSpan, Orientation, TextBox};
use docyoumeant::document::{to_analyze_result, AnalysisResult};
use geo::Coord;

// ============================================================================
// ProcessMode Tests
// ============================================================================

#[test]
fn test_process_mode_from_str_general() {
    use docyoumeant::document::analysis::pipeline::ProcessMode;

    let mode: ProcessMode = "general".into();
    assert_eq!(mode, ProcessMode::General);
}

#[test]
fn test_process_mode_from_str_read() {
    use docyoumeant::document::analysis::pipeline::ProcessMode;

    let mode: ProcessMode = "read".into();
    assert_eq!(mode, ProcessMode::Read);
}

#[test]
fn test_process_mode_from_str_unknown() {
    use docyoumeant::document::analysis::pipeline::ProcessMode;

    let mode: ProcessMode = "unknown".into();
    assert_eq!(mode, ProcessMode::General);

    let mode2: ProcessMode = "".into();
    assert_eq!(mode2, ProcessMode::General);

    let mode3: ProcessMode = "analyze".into();
    assert_eq!(mode3, ProcessMode::General);
}

// ============================================================================
// Orientation Tests
// ============================================================================

#[test]
fn test_orientation_from_rotation_degrees_cardinal() {
    assert_eq!(
        Orientation::from_rotation_degrees(0.0),
        Some(Orientation::Oriented0)
    );
    assert_eq!(
        Orientation::from_rotation_degrees(90.0),
        Some(Orientation::Oriented90)
    );
    assert_eq!(
        Orientation::from_rotation_degrees(180.0),
        Some(Orientation::Oriented180)
    );
    assert_eq!(
        Orientation::from_rotation_degrees(270.0),
        Some(Orientation::Oriented270)
    );
}

#[test]
fn test_orientation_from_rotation_degrees_negative() {
    // -90 should normalize to 270
    assert_eq!(
        Orientation::from_rotation_degrees(-90.0),
        Some(Orientation::Oriented270)
    );
    // -180 should normalize to 180
    assert_eq!(
        Orientation::from_rotation_degrees(-180.0),
        Some(Orientation::Oriented180)
    );
    // -270 should normalize to 90
    assert_eq!(
        Orientation::from_rotation_degrees(-270.0),
        Some(Orientation::Oriented90)
    );
}

#[test]
fn test_orientation_from_rotation_degrees_over_360() {
    assert_eq!(
        Orientation::from_rotation_degrees(360.0),
        Some(Orientation::Oriented0)
    );
    assert_eq!(
        Orientation::from_rotation_degrees(450.0),
        Some(Orientation::Oriented90)
    );
    assert_eq!(
        Orientation::from_rotation_degrees(720.0),
        Some(Orientation::Oriented0)
    );
}

#[test]
fn test_orientation_from_rotation_degrees_invalid() {
    // 45 degrees is not a valid orientation
    assert_eq!(Orientation::from_rotation_degrees(45.0), None);
    assert_eq!(Orientation::from_rotation_degrees(135.0), None);
    assert_eq!(Orientation::from_rotation_degrees(225.0), None);
}

#[test]
fn test_orientation_most_common_empty() {
    let orientations: Vec<Orientation> = vec![];
    assert_eq!(Orientation::most_common(&orientations), None);
}

#[test]
fn test_orientation_most_common_single() {
    let orientations = vec![Orientation::Oriented90];
    assert_eq!(
        Orientation::most_common(&orientations),
        Some(Orientation::Oriented90)
    );
}

#[test]
fn test_orientation_most_common_majority() {
    let orientations = vec![
        Orientation::Oriented0,
        Orientation::Oriented0,
        Orientation::Oriented90,
        Orientation::Oriented0,
        Orientation::Oriented180,
    ];
    assert_eq!(
        Orientation::most_common(&orientations),
        Some(Orientation::Oriented0)
    );
}

#[test]
fn test_orientation_most_common_all_same() {
    let orientations = vec![
        Orientation::Oriented270,
        Orientation::Oriented270,
        Orientation::Oriented270,
    ];
    assert_eq!(
        Orientation::most_common(&orientations),
        Some(Orientation::Oriented270)
    );
}

// ============================================================================
// DocumentSpan Tests
// ============================================================================

#[test]
fn test_document_span_new() {
    let span = DocumentSpan::new(10, 20);
    assert_eq!(span.offset, 10);
    assert_eq!(span.length, 20);
}

#[test]
fn test_document_span_zero() {
    let span = DocumentSpan::new(0, 0);
    assert_eq!(span.offset, 0);
    assert_eq!(span.length, 0);
}

// ============================================================================
// LayoutClass Tests
// ============================================================================

#[test]
fn test_layout_class_from_id_valid() {
    assert_eq!(LayoutClass::from_id(0), Some(LayoutClass::ParagraphTitle));
    assert_eq!(LayoutClass::from_id(1), Some(LayoutClass::Image));
    assert_eq!(LayoutClass::from_id(2), Some(LayoutClass::Text));
    assert_eq!(LayoutClass::from_id(10), Some(LayoutClass::DocTitle));
    assert_eq!(LayoutClass::from_id(14), Some(LayoutClass::Footer));
    assert_eq!(
        LayoutClass::from_id(19),
        Some(LayoutClass::ReferenceContent)
    );
}

#[test]
fn test_layout_class_from_id_invalid() {
    assert_eq!(LayoutClass::from_id(20), None);
    assert_eq!(LayoutClass::from_id(100), None);
    assert_eq!(LayoutClass::from_id(usize::MAX), None);
}

// ============================================================================
// LayoutBox Region Tests
// ============================================================================

#[test]
fn test_layout_box_new() {
    let bounds = Bounds::new([
        Coord { x: 0, y: 0 },
        Coord { x: 100, y: 0 },
        Coord { x: 100, y: 50 },
        Coord { x: 0, y: 50 },
    ]);
    let region = LayoutBox::new(bounds, LayoutClass::DocTitle, 0.95)
        .with_page_number(1)
        .with_content("Test Title".into());

    assert_eq!(region.class, LayoutClass::DocTitle);
    assert_eq!(region.page_number, Some(1));
    assert_eq!(region.bounds, bounds);
    assert_eq!(region.content, Some("Test Title".into()));
}

#[test]
fn test_layout_box_is_region() {
    assert!(LayoutClass::DocTitle.is_region());
    assert!(LayoutClass::Header.is_region());
    assert!(LayoutClass::Footer.is_region());
    assert!(LayoutClass::Number.is_region());
    assert!(!LayoutClass::Text.is_region());
    assert!(!LayoutClass::Image.is_region());
}

// ============================================================================
// LayoutBox::build_regions Tests
// ============================================================================

fn create_text_box(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    text: Option<&str>,
    box_score: f32,
    text_score: f32,
) -> TextBox {
    TextBox {
        bounds: Bounds::new([
            Coord { x, y },
            Coord { x: x + w, y },
            Coord { x: x + w, y: y + h },
            Coord { x, y: y + h },
        ]),
        angle: None,
        text: text.map(String::from),
        box_score,
        text_score,
        span: None,
    }
}

fn create_layout_box(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    class: LayoutClass,
    confidence: f32,
) -> LayoutBox {
    LayoutBox::new(
        Bounds::new([
            Coord { x, y },
            Coord { x: x + w, y },
            Coord { x: x + w, y: y + h },
            Coord { x, y: y + h },
        ]),
        class,
        confidence,
    )
}

#[test]
fn test_build_regions_empty() {
    let layout_boxes: Vec<LayoutBox> = vec![];
    let text_boxes: Vec<TextBox> = vec![];

    let regions = LayoutBox::build_regions(1, &layout_boxes, &text_boxes);
    assert!(regions.is_empty());
}

#[test]
fn test_build_regions_no_matching_layout() {
    // Layout boxes with classes that don't generate regions
    let layout_boxes = vec![
        create_layout_box(0, 0, 100, 100, LayoutClass::Image, 0.9),
        create_layout_box(100, 0, 100, 100, LayoutClass::Text, 0.85),
    ];
    let text_boxes = vec![create_text_box(
        0,
        0,
        100,
        100,
        Some("Image caption"),
        0.9,
        0.9,
    )];

    let regions = LayoutBox::build_regions(1, &layout_boxes, &text_boxes);
    assert!(regions.is_empty());
}

#[test]
fn test_build_regions_title() {
    let layout_boxes = vec![create_layout_box(
        10,
        10,
        200,
        40,
        LayoutClass::DocTitle,
        0.95,
    )];
    let text_boxes = vec![create_text_box(
        10,
        10,
        200,
        40,
        Some("Document Title"),
        0.9,
        0.88,
    )];

    let regions = LayoutBox::build_regions(1, &layout_boxes, &text_boxes);

    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].class, LayoutClass::DocTitle);
    assert_eq!(regions[0].content, Some("Document Title".into()));
    assert_eq!(regions[0].page_number, Some(1));
}

#[test]
fn test_build_regions_footer() {
    let layout_boxes = vec![create_layout_box(0, 900, 600, 30, LayoutClass::Footer, 0.9)];
    let text_boxes = vec![create_text_box(
        0,
        900,
        600,
        30,
        Some("Page Footer Text"),
        0.85,
        0.8,
    )];

    let regions = LayoutBox::build_regions(1, &layout_boxes, &text_boxes);

    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].class, LayoutClass::Footer);
}

#[test]
fn test_build_regions_header() {
    let layout_boxes = vec![create_layout_box(0, 0, 600, 30, LayoutClass::Header, 0.92)];
    let text_boxes = vec![create_text_box(
        0,
        0,
        600,
        30,
        Some("Header Text"),
        0.9,
        0.85,
    )];

    let regions = LayoutBox::build_regions(1, &layout_boxes, &text_boxes);

    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].class, LayoutClass::Header);
}

#[test]
fn test_build_regions_page_number() {
    let layout_boxes = vec![create_layout_box(
        280,
        950,
        40,
        20,
        LayoutClass::Number,
        0.88,
    )];
    let text_boxes = vec![create_text_box(280, 950, 40, 20, Some("42"), 0.95, 0.92)];

    let regions = LayoutBox::build_regions(1, &layout_boxes, &text_boxes);

    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].class, LayoutClass::Number);
    assert_eq!(regions[0].content, Some("42".into()));
}

#[test]
fn test_build_regions_footnote() {
    let layout_boxes = vec![create_layout_box(
        50,
        800,
        500,
        60,
        LayoutClass::Footnote,
        0.87,
    )];
    let text_boxes = vec![create_text_box(
        50,
        800,
        500,
        60,
        Some("This is a footnote reference."),
        0.9,
        0.88,
    )];

    let regions = LayoutBox::build_regions(1, &layout_boxes, &text_boxes);

    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].class, LayoutClass::Footnote);
}

#[test]
fn test_build_regions_multiple_text_boxes_combined() {
    // Layout box that overlaps with multiple text boxes
    let layout_boxes = vec![create_layout_box(0, 0, 300, 50, LayoutClass::DocTitle, 0.9)];
    let text_boxes = vec![
        create_text_box(0, 0, 100, 50, Some("Hello"), 0.9, 0.85),
        create_text_box(100, 0, 100, 50, Some("World"), 0.88, 0.82),
        create_text_box(200, 0, 100, 50, Some("Title"), 0.92, 0.9),
    ];

    let regions = LayoutBox::build_regions(1, &layout_boxes, &text_boxes);

    assert_eq!(regions.len(), 1);
    let content = regions[0].content.as_ref().unwrap();
    assert!(content.contains("Hello"));
    assert!(content.contains("World"));
    assert!(content.contains("Title"));
}

#[test]
fn test_build_regions_text_box_not_overlapping() {
    // Layout box with no overlapping text boxes
    let layout_boxes = vec![create_layout_box(0, 0, 100, 50, LayoutClass::DocTitle, 0.9)];
    let text_boxes = vec![
        // Text box far away from layout box
        create_text_box(500, 500, 100, 50, Some("Distant Text"), 0.9, 0.85),
    ];

    let regions = LayoutBox::build_regions(1, &layout_boxes, &text_boxes);

    // No regions because text box doesn't overlap
    assert!(regions.is_empty());
}

#[test]
fn test_build_regions_mixed_layout_types() {
    let layout_boxes = vec![
        create_layout_box(0, 0, 600, 40, LayoutClass::Header, 0.92),
        create_layout_box(50, 50, 500, 60, LayoutClass::DocTitle, 0.95),
        create_layout_box(0, 500, 600, 400, LayoutClass::Text, 0.88),
        create_layout_box(0, 920, 600, 30, LayoutClass::Footer, 0.85),
    ];
    let text_boxes = vec![
        create_text_box(0, 0, 600, 40, Some("Company Name"), 0.9, 0.88),
        create_text_box(50, 50, 500, 60, Some("Annual Report 2024"), 0.95, 0.93),
        create_text_box(0, 500, 600, 400, Some("Body text content..."), 0.88, 0.85),
        create_text_box(0, 920, 600, 30, Some("Confidential"), 0.82, 0.8),
    ];

    let regions = LayoutBox::build_regions(1, &layout_boxes, &text_boxes);

    // Should have header, title, and footer (Text class doesn't create regions)
    assert_eq!(regions.len(), 3);

    let classes: Vec<_> = regions.iter().map(|r| r.class).collect();
    assert!(classes.contains(&LayoutClass::Header));
    assert!(classes.contains(&LayoutClass::DocTitle));
    assert!(classes.contains(&LayoutClass::Footer));
}

// ============================================================================
// AnalysisResult Tests
// ============================================================================

#[test]
fn test_analysis_result_new() {
    let result = AnalysisResult::new("test-process", "text");

    assert_eq!(result.process_id, "test-process");
    assert_eq!(result.content_format, "text");
    assert_eq!(result.api_version, env!("CARGO_PKG_VERSION").to_string());
    assert!(result.content.is_empty());
    assert!(result.pages.is_empty());
    assert!(result.regions.is_empty());
    assert!(result.question_answers.is_empty());
    assert!(result.metadata.is_none());
}

#[test]
fn test_analysis_result_with_content() {
    let result = AnalysisResult::new("test", "text").with_content("Hello World".into());

    assert_eq!(result.content, "Hello World");
}

#[test]
fn test_analysis_result_with_metadata() {
    let mut metadata = HashMap::new();
    metadata.insert("key".into(), serde_json::json!("value"));
    metadata.insert("count".into(), serde_json::json!(42));

    let result = AnalysisResult::new("test", "text").with_metadata(metadata);

    assert!(result.metadata.is_some());
    let meta = result.metadata.unwrap();
    assert_eq!(meta.get("key"), Some(&serde_json::json!("value")));
    assert_eq!(meta.get("count"), Some(&serde_json::json!(42)));
}

#[test]
fn test_analysis_result_add_page() {
    let mut result = AnalysisResult::new("test", "text");

    let page = PageContent::new(1);
    result.add_page(page);

    assert_eq!(result.pages.len(), 1);
    assert_eq!(result.pages[0].page_number, 1);
}

#[test]
fn test_analysis_result_add_regions() {
    let mut result = AnalysisResult::new("test", "text");

    let dummy_bounds = Bounds::new([Coord { x: 0, y: 0 }; 4]);
    let regions = vec![
        LayoutBox::new(dummy_bounds, LayoutClass::DocTitle, 0.9)
            .with_page_number(1)
            .with_content("Title".into()),
        LayoutBox::new(dummy_bounds, LayoutClass::Footer, 0.85)
            .with_page_number(1)
            .with_content("Footer".into()),
    ];
    result.add_regions(regions);

    assert_eq!(result.regions.len(), 2);
}

// ============================================================================
// to_analyze_result Tests
// ============================================================================

// Mock document content for testing
#[derive(Debug)]
struct MockDocumentContent {
    text: Option<String>,
    pages: Vec<PageContent>,
}

impl docyoumeant::document::content::DocumentContent for MockDocumentContent {
    fn page_count(&self) -> usize {
        self.pages.len()
    }

    fn get_text(&self) -> Option<String> {
        self.text.clone()
    }

    fn get_pages(&self) -> &Vec<PageContent> {
        &self.pages
    }

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
        &mut self.pages
    }
}

#[test]
fn test_to_analyze_result_empty_content() {
    let content = MockDocumentContent {
        text: None,
        pages: vec![],
    };

    let result = to_analyze_result(&DocumentType::Text, &content, "test-id");

    assert_eq!(result.process_id, "test-id");
    assert!(result.content.is_empty());
    assert!(result.pages.is_empty());
    assert!(result.metadata.is_some());
}

#[test]
fn test_to_analyze_result_with_text() {
    let content = MockDocumentContent {
        text: Some("Test content".into()),
        pages: vec![PageContent::new(1)],
    };

    let result = to_analyze_result(&DocumentType::Pdf, &content, "pdf-test");

    assert_eq!(result.content, "Test content");
    assert_eq!(result.pages.len(), 1);
}

#[test]
fn test_to_analyze_result_with_detected_language() {
    let mut page = PageContent::new(1);
    page.detected_language = Some("en".into());

    let content = MockDocumentContent {
        text: Some("Hello".into()),
        pages: vec![page],
    };

    let result = to_analyze_result(&DocumentType::Pdf, &content, "test");

    let metadata = result.metadata.unwrap();
    assert_eq!(metadata.get("language"), Some(&serde_json::json!("en")));
}

#[test]
fn test_to_analyze_result_with_regions() {
    let mut page = PageContent::new(1);
    page.regions = vec![LayoutBox::new(
        Bounds::new([Coord { x: 0, y: 0 }; 4]),
        LayoutClass::DocTitle,
        0.9,
    )
    .with_page_number(1)
    .with_content("Title".into())];

    let content = MockDocumentContent {
        text: None,
        pages: vec![page],
    };

    let result = to_analyze_result(&DocumentType::Pdf, &content, "test");

    assert_eq!(result.regions.len(), 1);
}

#[test]
fn test_to_analyze_result_metadata_document_type() {
    let content = MockDocumentContent {
        text: None,
        pages: vec![],
    };

    let result = to_analyze_result(&DocumentType::Word, &content, "test");

    let metadata = result.metadata.unwrap();
    assert!(metadata.contains_key("document_type"));
    assert!(metadata.contains_key("page_count"));
}

// ============================================================================
// TextBox Tests
// ============================================================================

#[test]
fn test_text_box_with_all_fields() {
    let text_box = TextBox {
        bounds: Bounds::new([
            Coord { x: 0, y: 0 },
            Coord { x: 100, y: 0 },
            Coord { x: 100, y: 50 },
            Coord { x: 0, y: 50 },
        ]),
        angle: Some(Orientation::Oriented0),
        text: Some("Hello World".into()),
        box_score: 0.95,
        text_score: 0.92,
        span: Some(DocumentSpan::new(0, 11)),
    };

    assert_eq!(text_box.bounds[0], Coord { x: 0, y: 0 });
    assert_eq!(text_box.angle, Some(Orientation::Oriented0));
    assert_eq!(text_box.text, Some("Hello World".into()));
    assert!((text_box.box_score - 0.95).abs() < 1e-6);
    assert!((text_box.text_score - 0.92).abs() < 1e-6);
    assert_eq!(text_box.span.unwrap().length, 11);
}

#[test]
fn test_text_box_minimal() {
    let text_box = TextBox {
        bounds: Bounds::new([Coord { x: 0, y: 0 }; 4]),
        angle: None,
        text: None,
        box_score: 0.0,
        text_score: 0.0,
        span: None,
    };

    assert!(text_box.angle.is_none());
    assert!(text_box.text.is_none());
    assert!(text_box.span.is_none());
}

// ============================================================================
// LayoutBox Tests
// ============================================================================

#[test]
fn test_layout_box_creation() {
    let layout_box = LayoutBox::new(
        Bounds::new([
            Coord { x: 10, y: 20 },
            Coord { x: 110, y: 20 },
            Coord { x: 110, y: 70 },
            Coord { x: 10, y: 70 },
        ]),
        LayoutClass::Table,
        0.87,
    );

    assert_eq!(layout_box.class, LayoutClass::Table);
    assert!((layout_box.confidence - 0.87).abs() < 1e-6);
    assert!(layout_box.page_number.is_none());
    assert!(layout_box.content.is_none());
}

// ============================================================================
// Integration-style Tests (without actual model inference)
// ============================================================================

#[test]
fn test_page_content_has_embedded_text_data() {
    let mut page = PageContent::new(1);

    assert!(!page.has_embedded_text_data());

    page.words = vec![create_text_box(0, 0, 50, 20, Some("Hello"), 1.0, 1.0)];
    assert!(!page.has_embedded_text_data());

    page.orientation = Some(Orientation::Oriented0);
    assert!(page.has_embedded_text_data());
}

#[test]
fn test_page_content_has_regions() {
    let mut page = PageContent::new(1);

    assert!(!page.has_regions());

    page.regions = vec![LayoutBox::new(
        Bounds::new([Coord { x: 0, y: 0 }; 4]),
        LayoutClass::DocTitle,
        0.9,
    )
    .with_page_number(1)
    .with_content("Title".into())];

    assert!(page.has_regions());
}
