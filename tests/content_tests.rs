use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use docyoumeant::document::content::{DocumentType, Modality};
use docyoumeant::document::Document;

// ============================================================================
// Test Helpers
// ============================================================================

fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_fixture(relative_path: &str) -> Vec<u8> {
    let path = fixtures_path().join(relative_path);
    fs::read(&path).unwrap_or_else(|_| panic!("Failed to read fixture: {}", path.display()))
}

// ============================================================================
// Document Content Tests
// ============================================================================

#[test]
fn test_word_document_content() {
    let bytes = load_fixture("docx/test.docx");
    let doc = Document::new(&bytes, "test.docx").unwrap();

    let content = doc.content().unwrap();
    assert_eq!(doc.doc_type(), &DocumentType::Word);
    assert!(content
        .get_text()
        .unwrap()
        .contains("This is a test document!"));
    assert!(HashSet::<Modality>::from(doc.doc_type().clone()).contains(&Modality::Text));
}

#[test]
fn test_pdf_document_content() {
    let bytes = load_fixture("pdf/test.pdf");
    let doc = Document::new(&bytes, "test.pdf").unwrap();

    let content = doc.content().unwrap();
    assert_eq!(doc.doc_type(), &DocumentType::Pdf);
    assert!(content
        .get_text()
        .unwrap()
        .contains("This is a test document!"));
    let modalities = HashSet::<Modality>::from(doc.doc_type().clone());
    assert!(modalities.contains(&Modality::Text));
    assert!(modalities.contains(&Modality::Image));
}

#[test]
fn test_excel_document_content() {
    let bytes = load_fixture("xlsx/test.xlsx");
    let doc = Document::new(&bytes, "test.xlsx").unwrap();

    let content = doc.content().unwrap();
    assert_eq!(doc.doc_type(), &DocumentType::Excel);
    let text = content.get_text().unwrap();
    assert!(text.contains("Sheet1"));
    assert!(text.contains("Sheet2"));
    assert!(HashSet::<Modality>::from(doc.doc_type().clone()).contains(&Modality::Text));
}

#[test]
fn test_jpg_image_content() {
    let bytes = load_fixture("jpg/test.jpg");
    let doc = Document::new(&bytes, "test.jpg").unwrap();

    assert!(matches!(doc.doc_type(), DocumentType::Jpeg));
    assert!(HashSet::<Modality>::from(doc.doc_type().clone()).contains(&Modality::Image));
}

// ============================================================================
// Text and CSV Content Tests
// ============================================================================

#[test]
fn test_text_content_loading() {
    let bytes = b"test content\n";
    let doc = Document::new(bytes, "test.txt").unwrap();

    assert!(doc.content().is_some());
    assert_eq!(doc.doc_type(), &DocumentType::Text);
}

#[test]
fn test_csv_content_loading() {
    let bytes = b"header1,header2\nvalue1,value2\n";
    let doc = Document::new(bytes, "test.csv").unwrap();

    assert!(doc.content().is_some());
    assert_eq!(doc.doc_type(), &DocumentType::Csv);
}

#[test]
fn test_text_content_extraction() {
    let content = "Hello\nWorld\nTest";
    let doc = Document::new(content.as_bytes(), "test.txt").unwrap();

    assert_eq!(doc.content().unwrap().get_text(), Some(content.to_string()));
}

#[test]
fn test_csv_with_single_column() {
    let content = "line1\nline2\nline3";
    let doc = Document::new(content.as_bytes(), "test.csv").unwrap();

    assert_eq!(doc.content().unwrap().get_text(), Some(content.to_string()));
}

#[test]
fn test_csv_with_multiple_columns() {
    let bytes = b"a,b,c\n1,2,3\n4,5,6";
    let doc = Document::new(bytes, "test.csv").unwrap();

    assert!(doc.content().unwrap().get_text().is_some());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_invalid_content() {
    // Test with corrupted/invalid bytes for a document type
    let invalid_bytes = b"not a valid docx file";
    let result = Document::new(invalid_bytes, "test.docx");
    assert!(result.is_err());
}

#[test]
fn test_unsupported_extension() {
    let bytes = b"content";
    let result = Document::new(bytes, "test.xyz");
    assert!(result.is_err());
}

#[test]
fn test_empty_text_file() {
    let bytes = b"";
    let doc = Document::new(bytes, "test.txt").unwrap();

    assert_eq!(doc.content().unwrap().get_text(), Some("".to_string()));
}

#[test]
fn test_text_modalities() {
    let bytes = b"test content";
    let doc = Document::new(bytes, "test.txt").unwrap();

    let modalities = HashSet::<Modality>::from(doc.doc_type().clone());
    assert!(modalities.contains(&Modality::Text));
}

#[test]
fn test_file_without_extension() {
    let bytes = b"test content";
    let result = Document::new(bytes, "");
    assert!(result.is_err());
}

#[test]
fn test_csv_quoted_fields() {
    let content = r#"header1,header2
"value,1",value2
value3,"value,4""#;
    let doc = Document::new(content.as_bytes(), "test.csv").unwrap();
    let text = doc.content().unwrap().get_text().unwrap();
    assert!(text.contains("value,1"));
    assert!(text.contains("value,4"));
}

#[test]
fn test_csv_empty() {
    let content = b"";
    let doc = Document::new(content, "test.csv").unwrap();
    assert_eq!(doc.content().unwrap().get_text(), Some("".to_string()));
}

#[test]
fn test_text_invalid_utf8() {
    let bytes = b"\xff\xfe\xfd"; // Invalid UTF-8
    let result = Document::new(bytes, "test.txt");
    assert!(result.is_err());
}

#[test]
fn test_png_image_content() {
    let bytes = load_fixture("png/test.png");
    let doc = Document::new(&bytes, "test.png").unwrap();

    assert!(matches!(doc.doc_type(), DocumentType::Png));
    assert!(HashSet::<Modality>::from(doc.doc_type().clone()).contains(&Modality::Image));
}

#[test]
fn test_invalid_image() {
    let bytes = b"not an image";
    let result = Document::new(bytes, "test.jpg");
    assert!(result.is_err());
}

// ============================================================================
// DocumentType Tests
// ============================================================================

#[test]
fn test_document_type_from_extension() {
    assert_eq!(
        DocumentType::from_extension("txt"),
        Some(DocumentType::Text)
    );
    assert_eq!(
        DocumentType::from_extension("docx"),
        Some(DocumentType::Word)
    );
    assert_eq!(DocumentType::from_extension("pdf"), Some(DocumentType::Pdf));
    assert_eq!(
        DocumentType::from_extension("xlsx"),
        Some(DocumentType::Excel)
    );
    assert_eq!(DocumentType::from_extension("csv"), Some(DocumentType::Csv));
    assert_eq!(DocumentType::from_extension("png"), Some(DocumentType::Png));
    assert_eq!(
        DocumentType::from_extension("jpg"),
        Some(DocumentType::Jpeg)
    );
    assert_eq!(
        DocumentType::from_extension("jpeg"),
        Some(DocumentType::Jpeg)
    );
    assert_eq!(
        DocumentType::from_extension("tiff"),
        Some(DocumentType::Tiff)
    );
    assert_eq!(
        DocumentType::from_extension("tif"),
        Some(DocumentType::Tiff)
    );
    assert_eq!(DocumentType::from_extension("unknown"), None);
}

#[test]
fn test_modality_for_all_types() {
    use std::collections::HashSet;

    let check_modality = |doc_type: DocumentType, expected: Vec<Modality>| {
        let modalities: HashSet<Modality> = doc_type.into();
        for m in expected {
            assert!(modalities.contains(&m));
        }
    };

    check_modality(DocumentType::Text, vec![Modality::Text]);
    check_modality(DocumentType::Word, vec![Modality::Text]);
    check_modality(DocumentType::Excel, vec![Modality::Text]);
    check_modality(DocumentType::Csv, vec![Modality::Text]);
    check_modality(DocumentType::Pdf, vec![Modality::Text, Modality::Image]);
    check_modality(DocumentType::Png, vec![Modality::Image]);
    check_modality(DocumentType::Jpeg, vec![Modality::Image]);
    check_modality(DocumentType::Tiff, vec![Modality::Image]);
}

// ============================================================================
// PDF Content Tests
// ============================================================================

#[test]
fn test_pdf_embedded_text_details() {
    let bytes = load_fixture("pdf/test.pdf");
    let doc = Document::new(&bytes, "test.pdf").unwrap();

    let content = doc.content().unwrap();
    let pages = content.get_pages();

    assert!(!pages.is_empty(), "PDF should have at least one page");

    let page = &pages[0];

    assert!(page.text.is_some());
    let page_text = page.text.as_ref().unwrap();
    assert!(page_text.contains("This is a test document!"));

    assert!(
        !page.words.is_empty(),
        "Should have extracted words from PDF"
    );

    let words_text: String = page
        .words
        .iter()
        .filter_map(|w| w.text.clone())
        .collect::<Vec<_>>()
        .join("");

    assert!(words_text.contains("This"), "Words should contain 'This'");
    assert!(words_text.contains("test"), "Words should contain 'test'");
    assert!(
        words_text.contains("document"),
        "Words should contain 'document'"
    );

    for word in &page.words {
        let has_dimensions = word.bounds.iter().any(|coord| coord.x != 0 || coord.y != 0);
        assert!(has_dimensions, "Word bounding box should not be all zeros");
        assert!(word.text.is_some(), "Word should have text");
        assert_eq!(
            word.box_score, 1.0,
            "Embedded text should have box_score 1.0"
        );
        assert_eq!(
            word.text_score, 1.0,
            "Embedded text should have text_score 1.0"
        );
        assert!(word.span.is_some(), "Word should have a span");
    }
}
