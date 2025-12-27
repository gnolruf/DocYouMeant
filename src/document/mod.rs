//! Document loading, parsing, and analysis module.
//!
//! This module provides the core document processing functionality for DocYouMeant,
//! including loading various file formats, extracting content, and performing
//! document analysis with OCR and layout detection.
//!
//! # Main Types
//!
//! - [`Document`]: The primary entry point for loading and analyzing documents
//! - [`DocumentType`]: Enum representing supported file formats
//! - [`DocumentContent`]: Trait for document content implementations
//! - [`AnalysisPipeline`]: Orchestrates the document analysis workflow
//! - [`AnalysisResult`]: Contains extracted text, layout, tables, and Q&A results

pub mod analysis;
pub mod bounds;
pub mod content;
pub mod error;
pub mod layout_box;
pub mod table;
pub mod text_box;

pub use analysis::{to_analyze_result, AnalysisPipeline, AnalysisResult};
pub use content::{DocumentContent, DocumentType, ImageContent};
pub use error::DocumentError;
pub use layout_box::{LayoutBox, LayoutClass};
pub use text_box::TextBox;

use content::{CsvContent, ExcelContent, PdfContent, TextContent, WordContent};
use lingua::Language;

/// Represents a loaded document that can be analyzed for content extraction.
///
/// `Document` is the primary entry point for document processing.
#[derive(Debug)]
pub struct Document {
    /// The type of document (PDF, Word, Image, etc.)
    doc_type: DocumentType,
    /// The loaded document content, if available
    content: Option<Box<dyn DocumentContent>>,
    /// Results from question-and-answer analysis
    question_answers:
        Vec<crate::inference::tasks::question_and_answer_task::QuestionAndAnswerResult>,
    /// Unique identifier for the analysis process
    process_id: String,
}

impl Document {
    /// Creates a new `Document` from raw bytes and a filename.
    ///
    /// The document type is inferred from the file extension. The content is
    /// immediately loaded and parsed based on the detected type.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes of the document file
    /// * `filename` - The filename including extension (used to determine document type)
    ///
    /// # Returns
    ///
    /// Returns `Ok(Document)` on success, or a [`DocumentError`] if:
    /// - The file has no extension
    /// - The file extension is not supported
    /// - The content cannot be parsed
    pub fn new(bytes: &[u8], filename: &str) -> Result<Self, DocumentError> {
        let extension = std::path::Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| DocumentError::UnsupportedFileType {
                extension: "(no extension)".to_string(),
            })?;

        let doc_type = DocumentType::from_extension(extension).ok_or_else(|| {
            DocumentError::UnsupportedFileType {
                extension: extension.to_string(),
            }
        })?;

        let content = match doc_type {
            DocumentType::Text => TextContent::load(bytes)?,
            DocumentType::Word => WordContent::load(bytes)?,
            DocumentType::Pdf => PdfContent::load(bytes)?,
            DocumentType::Excel => ExcelContent::load(bytes)?,
            DocumentType::Csv => CsvContent::load(bytes)?,
            DocumentType::Png | DocumentType::Jpeg | DocumentType::Tiff => {
                ImageContent::load(bytes, doc_type.clone())?
            }
        };

        Ok(Document {
            doc_type,
            content: Some(content),
            question_answers: Vec::new(),
            process_id: "general".to_string(),
        })
    }

    /// Returns a reference to the document type.
    pub fn doc_type(&self) -> &DocumentType {
        &self.doc_type
    }

    /// Returns a reference to the document content, if loaded.
    ///
    /// The content provides access to the underlying document data and any
    /// extracted information such as text, layout boxes, and tables.
    ///
    /// # Returns
    ///
    /// Returns `Some(&dyn DocumentContent)` if content is loaded, `None` otherwise.
    pub fn content(&self) -> Option<&dyn DocumentContent> {
        self.content.as_deref()
    }

    /// Returns a mutable reference to the document content, if loaded.
    ///
    /// This allows modifying the document content, such as adding extracted
    /// layout information during analysis.
    ///
    /// # Returns
    ///
    /// Returns `Some(&mut dyn DocumentContent)` if content is loaded, `None` otherwise.
    pub fn content_mut(&mut self) -> Option<&mut dyn DocumentContent> {
        self.content.as_deref_mut()
    }

    /// Analyzes the document content using the analysis pipeline.
    ///
    /// # Arguments
    ///
    /// * `questions` - Optional slice of questions to answer about the document
    /// * `process_id` - Unique identifier for this analysis process (used for logging/tracking)
    /// * `language` - Optional language hint for OCR
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or a [`DocumentError`] if:
    /// - The document content is not loaded
    /// - Analysis fails for any reason
    pub fn analyze(
        &mut self,
        questions: Option<&[String]>,
        process_id: &str,
        language: Option<Language>,
    ) -> Result<(), DocumentError> {
        self.process_id = process_id.to_string();
        let doc_type = self.doc_type.clone();
        let pipeline = AnalysisPipeline::new(doc_type, &self.process_id);

        let content = match self.content_mut() {
            Some(content) => content,
            None => {
                return Err(DocumentError::ContentNotLoaded);
            }
        };

        let questions = questions.unwrap_or(&[]);
        let question_answers = pipeline.analyze(content, questions, language)?;

        self.question_answers = question_answers;

        Ok(())
    }

    /// Converts the analyzed document into an [`AnalysisResult`].
    ///
    /// This method aggregates all analysis results including extracted text,
    /// layout information, tables, and question-answer pairs into a single
    /// result structure suitable for serialization or further processing.
    ///
    /// # Returns
    ///
    /// Returns `Ok(AnalysisResult)` containing all extracted information,
    /// or a [`DocumentError::ContentNotLoaded`] if the document content
    /// is not available.
    pub fn to_analyze_result(&self) -> Result<AnalysisResult, DocumentError> {
        let content = match self.content() {
            Some(content) => content,
            None => {
                return Err(DocumentError::ContentNotLoaded);
            }
        };

        let mut result = to_analyze_result(&self.doc_type, content, &self.process_id);

        if !self.question_answers.is_empty() {
            result.set_question_answers(self.question_answers.clone());
        }

        Ok(result)
    }
}
