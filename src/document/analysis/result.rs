//! Document analysis result types and serialization.
//!
//! This module provides the [`AnalysisResult`] struct which encapsulates all outputs
//! from document analysis, including extracted text content, page information,
//! layout regions, tables, and question-answering results.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::document::content::{DocumentContent, DocumentType, PageContent};
use crate::document::layout_box::LayoutBox;
use crate::document::table::Table;
use crate::inference::tasks::question_and_answer_task::QuestionAndAnswerResult;

/// The complete result of document analysis.
///
/// `AnalysisResult` aggregates all information extracted during document processing,
/// providing a comprehensive view of the document's content, structure, and any
/// question-answering results.
///
/// This struct is designed for serialization to JSON and is the primary output
/// format returned by the document analysis API.
///
/// # Serialization
///
/// Fields with empty collections (`tables`, `question_answers`) or `None` values
/// (`metadata`) are omitted from serialized output to reduce payload size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// The API version that produced this result.
    ///
    /// Automatically populated from `CARGO_PKG_VERSION` at build time.
    pub api_version: String,

    /// Unique identifier for this processing request.
    ///
    /// Used for tracking and correlating results with their originating requests.
    pub process_id: String,

    /// The format of the `content` field (e.g., "text", "markdown").
    pub content_format: String,

    /// The full extracted text content of the document.
    ///
    /// Contains concatenated text from all pages, typically separated by newlines.
    pub content: String,

    /// Per-page content and metadata.
    ///
    /// Each [`PageContent`] contains the text, layout information, and any
    /// question-answering results specific to that page.
    pub pages: Vec<PageContent>,

    /// Layout regions detected across all pages.
    ///
    /// Each [`LayoutBox`] represents a structural element such as a title,
    /// paragraph, figure, or table region.
    pub regions: Vec<LayoutBox>,

    /// Tables detected and parsed from the document.
    ///
    /// Omitted from serialization when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<Table>,

    /// Results from question-answering tasks.
    ///
    /// Contains answers to questions posed during analysis, with confidence
    /// scores and source spans. Omitted from serialization when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub question_answers: Vec<QuestionAndAnswerResult>,

    /// Additional metadata about the document and processing.
    ///
    /// Omitted from serialization when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl AnalysisResult {
    /// Creates a new empty analysis result.
    ///
    /// # Arguments
    ///
    /// * `process_id` - A unique identifier for this processing request
    /// * `content_format` - The format of the content (e.g., "text", "markdown")
    ///
    /// # Returns
    ///
    /// A new `AnalysisResult` with:
    /// - `api_version` set to the crate's package version
    /// - Empty `content` string
    /// - Empty collections for `pages`, `regions`, `tables`, and `question_answers`
    /// - `metadata` set to `None`
    pub fn new(process_id: &str, content_format: &str) -> Self {
        Self {
            api_version: env!("CARGO_PKG_VERSION").to_string(),
            process_id: process_id.to_string(),
            content_format: content_format.to_string(),
            content: String::new(),
            pages: Vec::new(),
            regions: Vec::new(),
            tables: Vec::new(),
            question_answers: Vec::new(),
            metadata: None,
        }
    }

    /// Sets the metadata for this result, consuming and returning `self`.
    ///
    /// # Arguments
    ///
    /// * `metadata` - A map of metadata key-value pairs
    ///
    /// # Returns
    ///
    /// The modified `AnalysisResult` with metadata set.
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Sets the text content for this result, consuming and returning `self`.
    ///
    /// # Arguments
    ///
    /// * `content` - The extracted text content of the document
    ///
    /// # Returns
    ///
    /// The modified `AnalysisResult` with content set.
    pub fn with_content(mut self, content: String) -> Self {
        self.content = content;
        self
    }

    /// Adds a page to this result.
    ///
    /// Pages are stored in the order they are added, which should correspond
    /// to their page numbers in the original document.
    ///
    /// # Arguments
    ///
    /// * `page` - The page content to add
    pub fn add_page(&mut self, page: PageContent) {
        self.pages.push(page);
    }

    /// Appends layout regions to this result.
    ///
    /// The provided regions are moved into this result's `regions` vector.
    /// This method is typically called once per page during analysis.
    ///
    /// # Arguments
    ///
    /// * `regions` - A vector of layout boxes to append (will be drained)
    pub fn add_regions(&mut self, mut regions: Vec<LayoutBox>) {
        self.regions.append(&mut regions);
    }

    /// Appends tables to this result.
    ///
    /// The provided tables are moved into this result's `tables` vector.
    /// This method is typically called once per page during analysis.
    ///
    /// # Arguments
    ///
    /// * `tables` - A vector of tables to append (will be drained)
    pub fn add_tables(&mut self, mut tables: Vec<Table>) {
        self.tables.append(&mut tables);
    }

    /// Sets the question-answering results.
    ///
    /// Replaces any existing question answers with the provided results.
    /// These results are aggregated from all pages in the document.
    ///
    /// # Arguments
    ///
    /// * `question_answers` - The complete list of question-answering results
    pub fn set_question_answers(&mut self, question_answers: Vec<QuestionAndAnswerResult>) {
        self.question_answers = question_answers;
    }
}

/// Converts document content into a serializable [`AnalysisResult`].
///
/// This function is the primary way to create an `AnalysisResult` from processed
/// document content. It aggregates information from all pages and populates
/// metadata based on the document type and detected properties.
///
/// # Arguments
///
/// * `doc_type` - The type of document that was processed (PDF, PNG, etc.)
/// * `content` - The processed document content containing pages, text, and regions
/// * `process_id` - A unique identifier for this processing request
///
/// # Returns
///
/// An `AnalysisResult` populated with:
/// - Combined text content from all pages
/// - Page information including text, layout, and tables
/// - Aggregated layout regions from all pages
/// - Aggregated tables from all pages
/// - Metadata including document type, page count, and detected language
pub fn to_analyze_result(
    doc_type: &DocumentType,
    content: &dyn DocumentContent,
    process_id: &str,
) -> AnalysisResult {
    let content_format = "text";

    let mut result = AnalysisResult::new(process_id, content_format);

    if let Some(text) = content.get_text() {
        result = result.with_content(text);
    }

    let mut detected_language: Option<String> = None;
    for page in content.get_pages() {
        if detected_language.is_none() && page.detected_language.is_some() {
            detected_language = page.detected_language.clone();
        }
        if page.has_regions() {
            result.add_regions(page.regions.clone());
        }
        if page.has_tables() {
            result.add_tables(page.tables.clone());
        }
        result.add_page(page.clone());
    }

    let mut metadata = HashMap::new();
    metadata.insert(
        "document_type".to_string(),
        serde_json::json!(format!("{:?}", doc_type)),
    );
    metadata.insert(
        "page_count".to_string(),
        serde_json::json!(content.page_count()),
    );

    if let Some(language) = detected_language {
        metadata.insert("language".to_string(), serde_json::json!(language));
    }

    result.with_metadata(metadata)
}
