//! Microsoft Word document content handling.
//!
//! This module provides content extraction for Microsoft Word documents (`.docx`).
//! Word documents are parsed to extract paragraph text, with the entire document
//! represented as a single page.

use docx_rs::*;

use super::super::error::DocumentError;
use super::{DocumentContent, PageContent};

/// Content container for Microsoft Word documents.
///
/// `WordContent` parses `.docx` files and extracts paragraph text.
/// The document is represented as a single page regardless of actual
/// page breaks in the original document.
#[derive(Debug)]
pub struct WordContent {
    /// The pages extracted from the document (always exactly one page).
    pages: Vec<PageContent>,
}

impl DocumentContent for WordContent {
    fn get_pages(&self) -> &Vec<PageContent> {
        &self.pages
    }

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
        &mut self.pages
    }
}

impl WordContent {
    /// Loads and parses a Word document from raw bytes.
    ///
    /// This method parses the `.docx` file structure and extracts text
    /// from all paragraphs in the document body. The extracted text is
    /// stored as a single page.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes of the `.docx` file.
    ///
    /// # Returns
    ///
    /// A boxed [`DocumentContent`] trait object containing the extracted
    /// text as a single page, or a [`DocumentError::WordLoadError`] if
    /// parsing fails.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::WordLoadError`] if:
    /// - The file is not a valid `.docx` format
    /// - The document structure is corrupted
    /// - Required XML components are missing or malformed
    pub fn load(bytes: &[u8]) -> Result<Box<dyn DocumentContent>, DocumentError> {
        let docx = docx_rs::read_docx(bytes).map_err(|e| DocumentError::WordLoadError {
            message: e.to_string(),
        })?;

        let text_content = docx
            .document
            .children
            .iter()
            .fold(String::new(), |acc, c| match c {
                DocumentChild::Paragraph(p) => {
                    let mut new_acc = acc;
                    new_acc.push_str(&p.raw_text());
                    new_acc.push('\n');
                    new_acc
                }
                _ => acc,
            });

        let page = PageContent::with_text(1, text_content);

        Ok(Box::new(Self { pages: vec![page] }))
    }
}
