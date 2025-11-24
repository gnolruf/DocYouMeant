use docyoumeant::document::content::DocumentType;
use docyoumeant::document::{Document, DocumentError};

#[test]
fn test_new_document_with_valid_content() {
    let bytes = b"test content\n";
    let doc = Document::new(bytes, "test.txt").unwrap();

    assert_eq!(*doc.doc_type(), DocumentType::Text);
    assert!(doc.content().is_some());
}

#[test]
fn test_new_document_with_invalid_bytes() {
    let invalid_bytes = b"not a valid docx file";
    let result = Document::new(invalid_bytes, "test.docx");
    assert!(result.is_err());
}

#[test]
fn test_document_type_detection() {
    let fixtures_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    for (ext, expected_type) in DocumentType::supported_types() {
        let fixture_path = match ext {
            "txt" => fixtures_dir.join("txt/test.txt"),
            "docx" => fixtures_dir.join("docx/test.docx"),
            "pdf" => fixtures_dir.join("pdf/test.pdf"),
            "xlsx" => fixtures_dir.join("xlsx/test.xlsx"),
            "csv" => fixtures_dir.join("csv/test.csv"),
            "png" => fixtures_dir.join("png/test.png"),
            "jpg" | "jpeg" => fixtures_dir.join("jpg/test.jpg"),
            "tiff" | "tif" => fixtures_dir.join("tiff/test.tif"),
            _ => {
                continue;
            }
        };

        if !fixture_path.exists() {
            continue;
        }

        let bytes = std::fs::read(&fixture_path).expect("Failed to read fixture");
        let filename = format!("test.{}", ext);
        let doc = Document::new(&bytes, &filename)
            .unwrap_or_else(|_| panic!("Failed to create document for {}", ext));
        assert_eq!(*doc.doc_type(), expected_type, "Type mismatch for {}", ext);
    }
}

#[test]
fn test_content_access() {
    let bytes = b"test content\n";
    let doc = Document::new(bytes, "test.txt").unwrap();

    assert!(doc.content().is_some());
    assert!(doc.content().unwrap().get_text().is_some());
}

#[test]
fn test_invalid_extension() {
    let bytes = b"test content\n";
    let result = Document::new(bytes, "test.invalid");

    assert!(matches!(
        result,
        Err(DocumentError::UnsupportedFileType { .. })
    ));
}

#[test]
fn test_case_insensitive_extension() {
    let bytes = b"test content\n";
    let doc = Document::new(bytes, "test.TXT").unwrap();

    assert_eq!(*doc.doc_type(), DocumentType::Text);
}

#[test]
fn test_empty_extension() {
    let bytes = b"test content";
    let result = Document::new(bytes, "test");

    assert!(matches!(
        result,
        Err(DocumentError::UnsupportedFileType { .. })
    ));
}
