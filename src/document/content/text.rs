//! Plain text document content handling.
//!
//! This module provides content extraction for plain text files (`.txt`).

use super::super::error::DocumentError;
use super::{DocumentContent, PageContent};

/// Content container for plain text documents.
///
/// `TextContent` loads plain text files as single-page documents.
/// The entire file content is stored as the page's text without any
/// additional processing or structure extraction.
#[derive(Debug)]
pub struct TextContent {
    /// The pages extracted from the text file (always exactly one page).
    pages: Vec<PageContent>,
}

impl DocumentContent for TextContent {
    fn get_pages(&self) -> &[PageContent] {
        &self.pages
    }

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
        &mut self.pages
    }
}

impl TextContent {
    /// Loads a plain text file from raw bytes.
    ///
    /// This method decodes the bytes as UTF-8 text and creates a single-page
    /// document containing the full text content.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes of the text file (must be valid UTF-8).
    ///
    /// # Returns
    ///
    /// A boxed [`DocumentContent`] trait object containing the text as a
    /// single page, or a [`DocumentError::TextLoadError`] if decoding fails.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::TextLoadError`] if the bytes are not valid
    /// UTF-8 encoded text.
    pub fn load(bytes: &[u8]) -> Result<Box<dyn DocumentContent>, DocumentError> {
        let content =
            String::from_utf8(bytes.to_vec()).map_err(|e| DocumentError::TextLoadError {
                source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
            })?;

        let page = PageContent::with_text(1, content);

        Ok(Box::new(Self { pages: vec![page] }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::document::content::{DocumentType, Modality};
    use crate::document::Document;

    #[test]
    fn test_text_content_loading() {
        let bytes = b"test content\n";
        let doc = Document::new(bytes, "test.txt").unwrap();

        assert!(doc.content().is_some());
        assert_eq!(doc.doc_type(), &DocumentType::Text);
    }

    #[test]
    fn test_text_content_extraction() {
        let content = "Hello\nWorld\nTest";
        let doc = Document::new(content.as_bytes(), "test.txt").unwrap();

        assert_eq!(doc.content().unwrap().get_text(), Some(content.to_string()));
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
    fn test_text_invalid_utf8() {
        let bytes = b"\xff\xfe\xfd";
        let result = Document::new(bytes, "test.txt");
        assert!(result.is_err());
    }
}
