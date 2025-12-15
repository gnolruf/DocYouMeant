//! Plain text document content handling.
//!
//! This module provides content extraction for plain text files (`.txt`).
//! Text files are the simplest document type, containing only unformatted
//! text content without any structural information.

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
    fn get_pages(&self) -> &Vec<PageContent> {
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
