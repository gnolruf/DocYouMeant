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
/// including the document's content, structure, and any
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
#[must_use]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::bounds::Bounds;
    use crate::document::layout_box::LayoutClass;
    use geo::Coord;

    #[test]
    fn test_analysis_result_new() {
        let result = AnalysisResult::new("test-process", "text");

        assert_eq!(result.process_id, "test-process");
        assert_eq!(result.content_format, "text");
        assert_eq!(result.api_version, env!("CARGO_PKG_VERSION").to_string());
        assert!(result.content.is_empty());
        assert!(result.pages.is_empty());
        assert!(result.regions.is_empty());
        assert!(result.question_answers.is_empty());
        assert!(result.metadata.is_none());
    }

    #[test]
    fn test_analysis_result_with_content() {
        let result = AnalysisResult::new("test", "text").with_content("Hello World".into());

        assert_eq!(result.content, "Hello World");
    }

    #[test]
    fn test_analysis_result_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("key".into(), serde_json::json!("value"));
        metadata.insert("count".into(), serde_json::json!(42));

        let result = AnalysisResult::new("test", "text").with_metadata(metadata);

        assert!(result.metadata.is_some());
        let meta = result.metadata.unwrap();
        assert_eq!(meta.get("key"), Some(&serde_json::json!("value")));
        assert_eq!(meta.get("count"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn test_analysis_result_add_page() {
        let mut result = AnalysisResult::new("test", "text");

        let page = PageContent::new(1);
        result.add_page(page);

        assert_eq!(result.pages.len(), 1);
        assert_eq!(result.pages[0].page_number, 1);
    }

    #[test]
    fn test_analysis_result_add_regions() {
        let mut result = AnalysisResult::new("test", "text");

        let dummy_bounds = Bounds::new([Coord { x: 0, y: 0 }; 4]);
        let regions = vec![
            LayoutBox::new(dummy_bounds, LayoutClass::DocTitle, 0.9)
                .with_page_number(1)
                .with_content("Title".into()),
            LayoutBox::new(dummy_bounds, LayoutClass::Footer, 0.85)
                .with_page_number(1)
                .with_content("Footer".into()),
        ];
        result.add_regions(regions);

        assert_eq!(result.regions.len(), 2);
    }

    // Mock document content for testing
    #[derive(Debug)]
    struct MockDocumentContent {
        text: Option<String>,
        pages: Vec<PageContent>,
    }

    impl DocumentContent for MockDocumentContent {
        fn page_count(&self) -> usize {
            self.pages.len()
        }

        fn get_text(&self) -> Option<String> {
            self.text.clone()
        }

        fn get_pages(&self) -> &Vec<PageContent> {
            &self.pages
        }

        fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
            &mut self.pages
        }
    }

    #[test]
    fn test_to_analyze_result_empty_content() {
        let content = MockDocumentContent {
            text: None,
            pages: vec![],
        };

        let result = to_analyze_result(&DocumentType::Text, &content, "test-id");

        assert_eq!(result.process_id, "test-id");
        assert!(result.content.is_empty());
        assert!(result.pages.is_empty());
        assert!(result.metadata.is_some());
    }

    #[test]
    fn test_to_analyze_result_with_text() {
        let content = MockDocumentContent {
            text: Some("Test content".into()),
            pages: vec![PageContent::new(1)],
        };

        let result = to_analyze_result(&DocumentType::Pdf, &content, "pdf-test");

        assert_eq!(result.content, "Test content");
        assert_eq!(result.pages.len(), 1);
    }

    #[test]
    fn test_to_analyze_result_with_detected_language() {
        let mut page = PageContent::new(1);
        page.detected_language = Some("en".into());

        let content = MockDocumentContent {
            text: Some("Hello".into()),
            pages: vec![page],
        };

        let result = to_analyze_result(&DocumentType::Pdf, &content, "test");

        let metadata = result.metadata.unwrap();
        assert_eq!(metadata.get("language"), Some(&serde_json::json!("en")));
    }

    #[test]
    fn test_to_analyze_result_with_regions() {
        let mut page = PageContent::new(1);
        page.regions = vec![LayoutBox::new(
            Bounds::new([Coord { x: 0, y: 0 }; 4]),
            LayoutClass::DocTitle,
            0.9,
        )
        .with_page_number(1)
        .with_content("Title".into())];

        let content = MockDocumentContent {
            text: None,
            pages: vec![page],
        };

        let result = to_analyze_result(&DocumentType::Pdf, &content, "test");

        assert_eq!(result.regions.len(), 1);
    }

    #[test]
    fn test_to_analyze_result_metadata_document_type() {
        let content = MockDocumentContent {
            text: None,
            pages: vec![],
        };

        let result = to_analyze_result(&DocumentType::Word, &content, "test");

        let metadata = result.metadata.unwrap();
        assert!(metadata.contains_key("document_type"));
        assert!(metadata.contains_key("page_count"));
    }
}
