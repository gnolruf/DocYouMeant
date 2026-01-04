//! Document content types and abstractions.
//!
//! This module provides content representations for supported document formats.
//! Each document type has a corresponding content struct that implements
//! the [`DocumentContent`] trait.

mod csv;
mod excel;
mod image;
mod pdf;
mod text;
mod word;

pub use csv::CsvContent;
pub use excel::ExcelContent;
pub use image::ImageContent;
pub use pdf::PdfContent;
pub use text::TextContent;
pub use word::WordContent;

use std::any::Any;
use std::collections::HashSet;

use ::image::RgbImage;

use serde::{Deserialize, Serialize};

use crate::document::layout_box::LayoutBox;
use crate::document::table::Table;
use crate::document::text_box::{Orientation, TextBox};
use crate::inference::tasks::question_and_answer_task::QuestionAndAnswerResult;

/// Supported document file types.
///
/// This enum represents all currently supported file formats.
/// Each variant corresponds to a specific file format with its own
/// content parsing and analysis capabilities.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum DocumentType {
    /// Plain text file (`.txt`).
    Text,
    /// Microsoft Word document (`.docx`).
    Word,
    /// Portable Document Format file (`.pdf`).
    Pdf,
    /// Microsoft Excel spreadsheet (`.xlsx`).
    Excel,
    /// Comma-separated values file (`.csv`).
    Csv,
    /// PNG image file (`.png`).
    Png,
    /// JPEG image file (`.jpg`, `.jpeg`).
    Jpeg,
    /// TIFF image file (`.tiff`, `.tif`).
    Tiff,
}

impl DocumentType {
    /// Creates a `DocumentType` from a file extension string.
    ///
    /// # Arguments
    ///
    /// * `ext` - The file extension without the leading dot (e.g., "pdf", "docx").
    ///
    /// # Returns
    ///
    /// `Some(DocumentType)` if the extension is recognized, `None` otherwise.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        Self::supported_types()
            .into_iter()
            .find(|(supported_ext, _)| supported_ext.eq_ignore_ascii_case(ext))
            .map(|(_, doc_type)| doc_type)
    }

    /// Returns all supported file extensions with their corresponding document types.
    ///
    /// This includes all recognized extensions, including aliases (e.g., both "jpg"
    /// and "jpeg" for JPEG images, both "tiff" and "tif" for TIFF images).
    ///
    /// # Returns
    ///
    /// A vector of tuples containing `(extension, DocumentType)` pairs.
    #[must_use]
    pub fn supported_types() -> Vec<(&'static str, DocumentType)> {
        vec![
            ("txt", DocumentType::Text),
            ("docx", DocumentType::Word),
            ("pdf", DocumentType::Pdf),
            ("xlsx", DocumentType::Excel),
            ("csv", DocumentType::Csv),
            ("png", DocumentType::Png),
            ("jpg", DocumentType::Jpeg),
            ("jpeg", DocumentType::Jpeg),
            ("tiff", DocumentType::Tiff),
            ("tif", DocumentType::Tiff),
        ]
    }

    /// Returns the canonical (preferred) file extension for this document type.
    ///
    /// For types with multiple recognized extensions (e.g., JPEG), this returns
    /// the primary extension.
    ///
    /// # Returns
    ///
    /// The canonical extension string without a leading dot.
    #[must_use]
    pub fn canonical_extension(&self) -> &'static str {
        match self {
            DocumentType::Text => "txt",
            DocumentType::Word => "docx",
            DocumentType::Pdf => "pdf",
            DocumentType::Excel => "xlsx",
            DocumentType::Csv => "csv",
            DocumentType::Png => "png",
            DocumentType::Jpeg => "jpg",
            DocumentType::Tiff => "tiff",
        }
    }
}

/// Content modality classification for documents.
///
/// Documents can contain different types of content that require different
/// processing approaches. This enum categorizes content into fundamental
/// modalities used to determine which analysis pipelines to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Textual content that can be directly extracted or parsed.
    ///
    /// Documents with this modality contain machine-readable text,
    /// such as embedded text in PDFs or structured document formats.
    Text,
    /// Visual content requiring image-based analysis.
    ///
    /// Documents with this modality require OCR
    /// and visual layout analysis to extract text.
    Image,
}

impl From<DocumentType> for HashSet<Modality> {
    /// Determines the content modalities present in a document type.
    fn from(doc_type: DocumentType) -> Self {
        let mut modalities = HashSet::new();
        match doc_type {
            DocumentType::Text | DocumentType::Word | DocumentType::Excel | DocumentType::Csv => {
                modalities.insert(Modality::Text);
            }
            DocumentType::Pdf => {
                modalities.insert(Modality::Text);
                modalities.insert(Modality::Image);
            }
            DocumentType::Png | DocumentType::Jpeg | DocumentType::Tiff => {
                modalities.insert(Modality::Image);
            }
        }
        modalities
    }
}

/// Content and analysis results for a single page within a document.
///
/// A `PageContent` represents one page of a document and serves as a container
/// for all extracted and analyzed content from that page. This includes the
/// raw image representation, detected text at various granularities, layout
/// information, and table structures.
///
/// # Content Hierarchy
///
/// The text content is organized in a hierarchy:
/// 1. **Image**: The visual representation used for analysis
/// 2. **Layout boxes**: Detected regions of interest (figures, text blocks, tables)
/// 3. **Text lines**: Lines of text detected by the text detection model
/// 4. **Words**: Individual words from OCR or embedded text extraction
/// 5. **Tables**: Structured table data with cell contents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageContent {
    /// The page number within the document (1-indexed).
    pub page_number: usize,
    /// The RGB image representation of the page for visual analysis.
    ///
    /// This is populated for image-based documents and PDF pages rendered
    /// to images. Skipped during serialization as it's an internal artifact.
    #[serde(skip)]
    pub image: Option<RgbImage>,
    /// Text lines detected on this page by the text detection model (DBNet).
    ///
    /// Each [`TextBox`] represents a contiguous line of text with its
    /// bounding coordinates and content.
    pub text_lines: Vec<TextBox>,
    /// Individual words extracted from this page.
    ///
    /// Words may come from embedded text (e.g., PDF text layer) or from
    /// OCR processing of the page image.
    pub words: Vec<TextBox>,
    /// Raw layout detection results for this page.
    ///
    /// These are the unprocessed layout boxes from the layout detection
    /// model, used internally during analysis. Skipped during serialization.
    #[serde(skip)]
    pub layout_boxes: Vec<LayoutBox>,
    /// Processed page regions with populated content.
    ///
    /// Regions are layout boxes that have been identified as meaningful
    /// areas of interest and have had their content populated. This is
    /// an internal representation skipped during serialization.
    #[serde(skip)]
    pub regions: Vec<LayoutBox>,
    /// Tables detected and parsed from this page.
    ///
    /// Each [`Table`] contains the cell structure and extracted text
    /// content for a detected table region.
    pub tables: Vec<Table>,
    /// The raw text content of this page as a single string.
    ///
    /// This may be extracted directly from text-based formats or
    /// assembled from OCR results for image-based documents.
    pub text: Option<String>,
    /// The detected text orientation for this page.
    ///
    /// Indicates if the page content is rotated.
    pub orientation: Option<Orientation>,
    /// The language detected or specified for OCR on this page.
    ///
    /// Contains a language code used for text
    /// recognition. Only serialized if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_language: Option<String>,
    /// Question and answer extraction results for this page.
    ///
    /// Contains results from document Q&A analysis. Skipped during
    /// serialization as it's processed separately.
    #[serde(skip)]
    pub question_answers: Vec<QuestionAndAnswerResult>,
}

impl PageContent {
    /// Creates a new empty `PageContent` for the given page number.
    ///
    /// # Arguments
    ///
    /// * `page_number` - The 1-indexed page number.
    ///
    /// # Returns
    ///
    /// A new `PageContent` with all fields initialized to empty/`None` values.
    #[must_use]
    pub fn new(page_number: usize) -> Self {
        Self {
            page_number,
            image: None,
            text_lines: Vec::new(),
            words: Vec::new(),
            layout_boxes: Vec::new(),
            regions: Vec::new(),
            tables: Vec::new(),
            text: None,
            orientation: None,
            detected_language: None,
            question_answers: Vec::new(),
        }
    }

    /// Creates a new `PageContent` initialized with an image.
    ///
    /// Use this constructor for image-based documents or PDF pages
    /// that have been rendered to images for visual analysis.
    ///
    /// # Arguments
    ///
    /// * `page_number` - The 1-indexed page number.
    /// * `image` - The RGB image representation of the page.
    ///
    /// # Returns
    ///
    /// A new `PageContent` with the image set and other fields empty.
    #[must_use]
    pub fn with_image(page_number: usize, image: RgbImage) -> Self {
        Self {
            page_number,
            image: Some(image),
            text_lines: Vec::new(),
            words: Vec::new(),
            layout_boxes: Vec::new(),
            regions: Vec::new(),
            tables: Vec::new(),
            text: None,
            orientation: None,
            detected_language: None,
            question_answers: Vec::new(),
        }
    }

    /// Creates a new `PageContent` initialized with text content.
    ///
    /// Use this constructor for text-based documents where the page
    /// content is already available as a string.
    ///
    /// # Arguments
    ///
    /// * `page_number` - The 1-indexed page number.
    /// * `text` - The text content of the page.
    ///
    /// # Returns
    ///
    /// A new `PageContent` with the text set and other fields empty.
    #[must_use]
    pub fn with_text(page_number: usize, text: String) -> Self {
        Self {
            page_number,
            image: None,
            text_lines: Vec::new(),
            words: Vec::new(),
            layout_boxes: Vec::new(),
            regions: Vec::new(),
            tables: Vec::new(),
            text: Some(text),
            orientation: None,
            detected_language: None,
            question_answers: Vec::new(),
        }
    }

    /// Returns `true` if this page has detected layout regions.
    ///
    /// Regions are populated during layout analysis and represent
    /// meaningful areas of the page (text blocks, figures, tables, etc.).
    #[inline]
    #[must_use]
    pub fn has_regions(&self) -> bool {
        !self.regions.is_empty()
    }

    /// Returns `true` if this page has embedded text data.
    ///
    /// Embedded text data indicates that words have been extracted
    /// (from PDF text layer or OCR) and the page orientation has been
    /// determined. This is used to decide if OCR processing is needed.
    #[inline]
    #[must_use]
    pub fn has_embedded_text_data(&self) -> bool {
        !self.words.is_empty() && self.orientation.is_some()
    }

    /// Returns `true` if this page contains detected tables.
    #[inline]
    #[must_use]
    pub fn has_tables(&self) -> bool {
        !self.tables.is_empty()
    }
}

/// Trait for document content implementations.
///
/// This trait defines the common interface for all document content types,
/// providing uniform access to pages and text extraction regardless of the
/// underlying document format.
///
/// # Required Methods
///
/// Implementors must provide:
/// - [`get_pages`](Self::get_pages): Immutable access to page contents
/// - [`get_pages_mut`](Self::get_pages_mut): Mutable access for analysis pipelines
///
/// # Provided Methods
///
/// The trait provides default implementations for:
/// - [`page_count`](Self::page_count): Returns the number of pages
/// - [`get_text`](Self::get_text): Extracts all text content from the document
pub trait DocumentContent: std::fmt::Debug + Any {
    /// Returns an immutable reference to the document's pages.
    ///
    /// Each [`PageContent`] contains the extracted content and analysis
    /// results for one page of the document.
    fn get_pages(&self) -> &Vec<PageContent>;

    /// Returns a mutable reference to the document's pages.
    ///
    /// This is used by analysis pipelines to populate extracted content
    /// (text lines, words, tables, etc.) into the page structures.
    fn get_pages_mut(&mut self) -> &mut Vec<PageContent>;

    /// Returns the total number of pages in the document.
    ///
    /// # Default Implementation
    ///
    /// Returns the length of the pages vector.
    fn page_count(&self) -> usize {
        self.get_pages().len()
    }

    /// Extracts and concatenates all text content from the document.
    ///
    /// Collects the `text` field from each page that has text content
    /// and joins them with newlines.
    ///
    /// # Returns
    ///
    /// - `Some(String)` containing the concatenated text if any pages have text
    /// - `None` if no pages contain text content
    ///
    /// # Default Implementation
    ///
    /// Iterates through all pages, filters for those with text, and joins
    /// the text content with newline separators.
    fn get_text(&self) -> Option<String> {
        let text_parts: Vec<String> = self
            .get_pages()
            .iter()
            .filter_map(|page| page.text.as_ref())
            .cloned()
            .collect();

        if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join("\n"))
        }
    }
}
