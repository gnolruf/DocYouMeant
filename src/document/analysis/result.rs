use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::document::content::{DocumentContent, DocumentType, PageContent};
use crate::document::layout_box::LayoutBox;
use crate::document::table::Table;
use crate::inference::tasks::question_and_answer_task::QuestionAndAnswerResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub api_version: String,
    pub process_id: String,
    pub content_format: String,
    pub content: String,
    pub pages: Vec<PageContent>,
    pub regions: Vec<LayoutBox>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<Table>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub question_answers: Vec<QuestionAndAnswerResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl AnalysisResult {
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

    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_content(mut self, content: String) -> Self {
        self.content = content;
        self
    }

    pub fn add_page(&mut self, page: PageContent) {
        self.pages.push(page);
    }

    pub fn add_regions(&mut self, mut regions: Vec<LayoutBox>) {
        self.regions.append(&mut regions);
    }

    pub fn add_tables(&mut self, mut tables: Vec<Table>) {
        self.tables.append(&mut tables);
    }

    pub fn set_question_answers(&mut self, question_answers: Vec<QuestionAndAnswerResult>) {
        self.question_answers = question_answers;
    }
}

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
