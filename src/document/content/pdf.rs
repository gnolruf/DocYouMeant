//! PDF document content handling.
//!
//! This module provides content extraction for PDF files.
//! PDFs are unique in that they may contain both embedded text (from the
//! text layer) and visual content requiring OCR, making them a mixed-modality
//! document type.
//!
//! # Dependencies
//!
//! This module uses the `pdfium-render` crate, which requires the PDFium
//! library to be available at runtime (either bundled or system-installed).

use pdfium_render::prelude::*;

use super::super::bounds::Bounds;
use super::super::error::DocumentError;
use super::super::text_box::{Coord, Orientation, TextBox};
use super::{DocumentContent, PageContent};

/// Default rendering resolution in dots per inch for PDF pages.
const PDF_RENDER_DPI: f32 = 300.0;

/// Content container for PDF documents.
///
/// `PdfContent` provides rich extraction from PDF files, including both
/// embedded text with positional information and rendered page images.
#[derive(Debug)]
pub struct PdfContent {
    /// The pages extracted from the PDF document.
    pages: Vec<PageContent>,
}

impl DocumentContent for PdfContent {
    fn get_pages(&self) -> &[PageContent] {
        &self.pages
    }

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
        &mut self.pages
    }
}

impl PdfContent {
    /// Loads and parses a PDF document from raw bytes.
    ///
    /// This method initializes the PDFium library, parses the PDF structure,
    /// and extracts content from all pages including embedded text, word
    /// positions, and rendered page images.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes of the PDF file.
    ///
    /// # Returns
    ///
    /// A boxed [`DocumentContent`] trait object containing all extracted
    /// page content, or a [`DocumentError::PdfLoadError`] if parsing fails.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::PdfLoadError`] if:
    /// - The PDFium library cannot be loaded
    /// - The PDF file is corrupted or password-protected
    /// - Page rendering fails
    pub fn load(bytes: &[u8]) -> Result<Box<dyn DocumentContent>, DocumentError> {
        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name())
                .or_else(|_| Pdfium::bind_to_system_library())
                .map_err(|source| DocumentError::PdfLoadError { source })?,
        );

        let document = pdfium
            .load_pdf_from_byte_slice(bytes, None)
            .map_err(|source| DocumentError::PdfLoadError { source })?;

        Self::process_document(&document)
    }

    /// Processes all pages of a PDF document.
    ///
    /// Iterates through each page, extracting text content, word positions,
    /// and rendered images.
    ///
    /// # Arguments
    ///
    /// * `document` - Reference to the parsed PDF document.
    ///
    /// # Returns
    ///
    /// A boxed [`DocumentContent`] containing all processed pages.
    fn process_document(document: &PdfDocument) -> Result<Box<dyn DocumentContent>, DocumentError> {
        let total_pages = document.pages().len() as usize;
        let mut pages = Vec::new();

        let render_config = PdfRenderConfig::new().scale_page_by_factor(PDF_RENDER_DPI / 72.0);

        for page_index in 0..total_pages {
            let page_content =
                Self::process_pdf_page(document, page_index, PDF_RENDER_DPI, &render_config)?;
            pages.push(page_content);
        }

        Ok(Box::new(Self { pages }))
    }

    /// Processes a single PDF page.
    ///
    /// Extracts all content from a single page including:
    /// - Raw text content from the text layer
    /// - Individual word positions with bounding boxes
    /// - Rendered RGB image of the page
    /// - Text orientation information
    ///
    /// # Arguments
    ///
    /// * `document` - Reference to the PDF document.
    /// * `page_index` - Zero-based index of the page to process.
    /// * `dpi` - Rendering resolution in dots per inch.
    /// * `render_config` - PDFium render configuration.
    ///
    /// # Returns
    ///
    /// A [`PageContent`] struct populated with all extracted data.
    fn process_pdf_page(
        document: &PdfDocument,
        page_index: usize,
        dpi: f32,
        render_config: &PdfRenderConfig,
    ) -> Result<PageContent, DocumentError> {
        let page_index_u16 = page_index as u16;
        let page = document
            .pages()
            .get(page_index_u16)
            .map_err(|source| DocumentError::PdfLoadError { source })?;

        let text_page = page
            .text()
            .map_err(|source| DocumentError::PdfLoadError { source })?;

        let page_num: usize = page_index + 1;
        let mut page_content = PageContent::new(page_num);

        let page_text = text_page.all();
        let clean_text = page_text
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        if !clean_text.is_empty() {
            page_content.text = Some(clean_text);
        }

        let pdf_bitmap = page
            .render_with_config(render_config)
            .map_err(|source| DocumentError::PdfLoadError { source })?;

        let width = pdf_bitmap.width() as u32;
        let height = pdf_bitmap.height() as u32;
        let raw_bytes = pdf_bitmap.as_raw_bytes();

        let stride = width as usize * 4;
        let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            for x in 0..width {
                let pixel_index = y as usize * stride + x as usize * 4;
                if pixel_index + 2 < raw_bytes.len() {
                    let b = raw_bytes[pixel_index];
                    let g = raw_bytes[pixel_index + 1];
                    let r = raw_bytes[pixel_index + 2];
                    rgb_data.push(r);
                    rgb_data.push(g);
                    rgb_data.push(b);
                }
            }
        }

        if let Some(rgb_image) = image::RgbImage::from_raw(width, height, rgb_data) {
            page_content.image = Some(rgb_image);
        }

        let mut words = Vec::new();
        let mut orientations = Vec::new();
        Self::extract_orientations_from_objects(page.objects().iter(), &mut orientations)?;

        let page_orientation = Orientation::most_common(&orientations);
        if let Some(page_orientation) = page_orientation {
            page_content.orientation = Some(page_orientation);
        }

        Self::extract_words_from_text_page(&text_page, &mut words, page_orientation)?;

        let scale_factor = dpi / 72.0;
        let page_height = page.height().value;
        for word in &mut words {
            let coords = word.bounds.coords_mut();
            coords[0].x = (coords[0].x as f32 * scale_factor) as i32;
            coords[0].y = ((page_height - coords[0].y as f32) * scale_factor) as i32;
            coords[1].x = (coords[1].x as f32 * scale_factor) as i32;
            coords[1].y = ((page_height - coords[1].y as f32) * scale_factor) as i32;
            coords[2].x = (coords[2].x as f32 * scale_factor) as i32;
            coords[2].y = ((page_height - coords[2].y as f32) * scale_factor) as i32;
            coords[3].x = (coords[3].x as f32 * scale_factor) as i32;
            coords[3].y = ((page_height - coords[3].y as f32) * scale_factor) as i32;
        }

        page_content.words = words;
        Ok(page_content)
    }

    /// Extracts text orientation information from PDF page objects.
    ///
    /// Iterates through page objects to find text elements and records their
    /// rotation angles.
    ///
    /// # Arguments
    ///
    /// * `iterator` - Iterator over PDF page objects.
    /// * `orientations` - Vector to collect detected orientations.
    fn extract_orientations_from_objects(
        iterator: pdfium_render::prelude::PdfPageObjectsIterator,
        orientations: &mut Vec<Orientation>,
    ) -> Result<(), DocumentError> {
        for object in iterator {
            if let Some(text_object) = object.as_text_object() {
                let rotation_degrees = text_object.get_rotation_clockwise_degrees();
                let orientation = Orientation::from_rotation_degrees(rotation_degrees);
                if let Some(orient) = orientation {
                    orientations.push(orient);
                }
            }
        }
        Ok(())
    }

    /// Extracts individual words with bounding boxes from a PDF text page.
    ///
    /// Iterates through characters in the text layer, accumulating them into
    /// words based on whitespace boundaries. Each word is assigned a bounding
    /// box computed from the union of its character bounds.
    ///
    /// # Arguments
    ///
    /// * `text_page` - Reference to the PDF page text layer.
    /// * `words` - Vector to collect extracted words as [`TextBox`] instances.
    /// * `orientation` - Optional text orientation to apply to extracted words.
    fn extract_words_from_text_page(
        text_page: &PdfPageText,
        words: &mut Vec<TextBox>,
        orientation: Option<Orientation>,
    ) -> Result<(), DocumentError> {
        use std::collections::HashSet;

        let mut accumulated_text_length = 0usize;
        let mut current_word_text = String::new();
        let mut current_word_bounds: Option<(f32, f32, f32, f32)> = None;

        let mut seen_words: HashSet<(String, i32, i32, i32, i32)> = HashSet::new();

        for char_obj in text_page.chars().iter() {
            let char_text = char_obj.unicode_string().unwrap_or_default();

            if char_text.trim().is_empty() {
                if !current_word_text.is_empty() {
                    Self::push_word_if_unique(
                        &current_word_text,
                        current_word_bounds,
                        orientation,
                        accumulated_text_length,
                        words,
                        &mut seen_words,
                    );
                    accumulated_text_length += current_word_text.len() + 1;
                    current_word_text.clear();
                    current_word_bounds = None;
                }
                continue;
            }

            current_word_text.push_str(&char_text);

            if let Ok(bounds) = char_obj.loose_bounds() {
                let left = bounds.left().value;
                let bottom = bounds.bottom().value;
                let right = bounds.right().value;
                let top = bounds.top().value;

                if let Some((min_x, min_y, max_x, max_y)) = current_word_bounds {
                    current_word_bounds = Some((
                        min_x.min(left),
                        min_y.min(bottom),
                        max_x.max(right),
                        max_y.max(top),
                    ));
                } else {
                    current_word_bounds = Some((left, bottom, right, top));
                }
            }
        }

        if !current_word_text.is_empty() {
            Self::push_word_if_unique(
                &current_word_text,
                current_word_bounds,
                orientation,
                accumulated_text_length,
                words,
                &mut seen_words,
            );
        }

        Ok(())
    }

    /// Adds a word to the collection if it hasn't been seen before.
    ///
    /// Creates a [`TextBox`] for the word with its bounding box coordinates
    /// and adds it to the words vector only if an identical word (same text
    /// and position) hasn't already been added.
    ///
    /// # Arguments
    ///
    /// * `text` - The word text content.
    /// * `bounds` - Optional bounding box as `(min_x, min_y, max_x, max_y)`.
    /// * `orientation` - Optional text orientation.
    /// * `start_index` - Character offset of the word in the page text.
    /// * `words` - Vector to add the word to.
    /// * `seen_words` - Set tracking already-added words for deduplication.
    fn push_word_if_unique(
        text: &str,
        bounds: Option<(f32, f32, f32, f32)>,
        orientation: Option<Orientation>,
        start_index: usize,
        words: &mut Vec<TextBox>,
        seen_words: &mut std::collections::HashSet<(String, i32, i32, i32, i32)>,
    ) {
        if let Some((min_x, min_y, max_x, max_y)) = bounds {
            let word_bounds = [
                Coord {
                    x: min_x as i32,
                    y: max_y as i32,
                },
                Coord {
                    x: max_x as i32,
                    y: max_y as i32,
                },
                Coord {
                    x: max_x as i32,
                    y: min_y as i32,
                },
                Coord {
                    x: min_x as i32,
                    y: min_y as i32,
                },
            ];

            let key = (
                text.to_string(),
                min_x as i32,
                min_y as i32,
                max_x as i32,
                max_y as i32,
            );

            if seen_words.insert(key) {
                words.push(TextBox {
                    bounds: Bounds::new(word_bounds),
                    angle: orientation,
                    text: Some(text.to_string()),
                    box_score: 1.0,
                    text_score: 1.0,
                    span: Some(crate::document::text_box::DocumentSpan::new(
                        start_index,
                        text.len(),
                    )),
                });
            }
        }
    }
}
