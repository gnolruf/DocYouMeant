//! Image document content handling.
//!
//! This module provides content loading for image files (PNG, JPEG, TIFF).
//! Images are loaded as single-page documents with the raw image data available
//! for visual analysis pipelines such as OCR and layout detection.
//!
//! # Supported Formats
//!
//! - **PNG**: Lossless compression, supports transparency
//! - **JPEG**: Lossy compression, common for photographs
//! - **TIFF**: High-quality format, common for scanned documents

use super::super::error::DocumentError;
use super::{DocumentContent, DocumentType, PageContent};

/// Content container for image-based documents.
///
/// `ImageContent` loads image files and prepares them for visual analysis.
/// Unlike text-based formats, images do not contain directly extractable text
/// and require OCR processing to obtain textual content.
#[derive(Debug)]
pub struct ImageContent {
    /// The original document type (PNG, JPEG, or TIFF).
    #[allow(dead_code)]
    doc_type: DocumentType,
    /// The pages extracted from the image (always exactly one page).
    pages: Vec<PageContent>,
}

impl DocumentContent for ImageContent {
    fn get_pages(&self) -> &Vec<PageContent> {
        &self.pages
    }

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
        &mut self.pages
    }
}

impl ImageContent {
    /// Loads an image file from raw bytes.
    ///
    /// This method decodes the image data and converts it to RGB format
    /// for use in visual analysis pipelines. The image is stored as a
    /// single-page document.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes of the image file.
    /// * `doc_type` - The document type indicating the image format
    ///   ([`DocumentType::Png`], [`DocumentType::Jpeg`], or [`DocumentType::Tiff`]).
    ///
    /// # Returns
    ///
    /// A boxed [`DocumentContent`] trait object containing the decoded image
    /// as a single page, or a [`DocumentError::ImageLoadError`] if decoding fails.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::ImageLoadError`] if:
    /// - The image format is not recognized or supported
    /// - The image data is corrupted or truncated
    /// - Memory allocation for the decoded image fails
    pub fn load(
        bytes: &[u8],
        doc_type: DocumentType,
    ) -> Result<Box<dyn DocumentContent>, DocumentError> {
        let img = image::load_from_memory(bytes)
            .map_err(|source| DocumentError::ImageLoadError { source })?
            .to_rgb8();

        let page = PageContent::with_image(1, img);

        Ok(Box::new(Self {
            doc_type,
            pages: vec![page],
        }))
    }
}
