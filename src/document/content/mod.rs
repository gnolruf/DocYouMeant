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

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum DocumentType {
    Text,
    Word,
    Pdf,
    Excel,
    Csv,
    Png,
    Jpeg,
    Tiff,
}

impl DocumentType {
    pub fn from_extension(ext: &str) -> Option<Self> {
        Self::supported_types()
            .into_iter()
            .find(|(supported_ext, _)| supported_ext.eq_ignore_ascii_case(ext))
            .map(|(_, doc_type)| doc_type)
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
}

impl From<DocumentType> for HashSet<Modality> {
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

/// Represents content for a single page within a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageContent {
    /// The page number (1-indexed)
    pub page_number: usize,
    /// The image representation of the page (if applicable)
    #[serde(skip)]
    pub image: Option<RgbImage>,
    /// Text lines detected on this page (from DBNet text detection)
    pub text_lines: Vec<TextBox>,
    /// Individual words on this page (from embedded PDF text or OCR)
    pub words: Vec<TextBox>,
    /// Layout boxes found on this page (used internally, not serialized in final output)
    #[serde(skip)]
    pub layout_boxes: Vec<LayoutBox>,
    /// Page regions (layout boxes that are regions of interest, with content populated)
    #[serde(skip)]
    pub regions: Vec<LayoutBox>,
    /// Tables detected on this page
    pub tables: Vec<Table>,
    /// Raw text content for this page
    pub text: Option<String>,
    /// Document orientation detected for this page
    pub orientation: Option<Orientation>,
    /// The language detected or used for OCR on this page
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_language: Option<String>,
    /// Question and Answer results for this page
    #[serde(skip)]
    pub question_answers: Vec<QuestionAndAnswerResult>,
}

impl PageContent {
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

    pub fn has_regions(&self) -> bool {
        !self.regions.is_empty()
    }

    pub fn has_embedded_text_data(&self) -> bool {
        !self.words.is_empty() && self.orientation.is_some()
    }

    pub fn has_tables(&self) -> bool {
        !self.tables.is_empty()
    }
}

pub trait DocumentContent: std::fmt::Debug + Any {
    fn get_pages(&self) -> &Vec<PageContent>;

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent>;

    fn page_count(&self) -> usize {
        self.get_pages().len()
    }

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
