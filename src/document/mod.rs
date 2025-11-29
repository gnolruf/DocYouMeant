pub mod analysis;
pub mod content;
pub mod error;
pub mod layout_box;
pub mod region;
pub mod text_box;

pub use analysis::{to_analyze_result, AnalysisPipeline, AnalysisResult};
pub use content::{DocumentContent, DocumentType, ImageContent};
pub use error::DocumentError;
pub use layout_box::{LayoutBox, LayoutClass};
pub use text_box::TextBox;

use content::{CsvContent, ExcelContent, PdfContent, TextContent, WordContent};

#[derive(Debug)]
pub struct Document {
    doc_type: DocumentType,
    content: Option<Box<dyn DocumentContent>>,
    question_answers:
        Vec<crate::inference::tasks::question_and_answer_task::QuestionAndAnswerResult>,
    process_id: String,
}

impl Document {
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

    pub fn doc_type(&self) -> &DocumentType {
        &self.doc_type
    }

    pub fn content(&self) -> Option<&dyn DocumentContent> {
        self.content.as_deref()
    }

    pub fn content_mut(&mut self) -> Option<&mut dyn DocumentContent> {
        self.content.as_deref_mut()
    }

    pub fn analyze(
        &mut self,
        questions: Option<&[String]>,
        process_id: &str,
        language: Option<&str>,
    ) -> Result<(), DocumentError> {
        self.process_id = process_id.to_string();
        let doc_type = self.doc_type.clone();
        let pipeline = AnalysisPipeline::new(
            doc_type,
            self.process_id.clone(),
            language.map(String::from),
        );

        let content = match self.content_mut() {
            Some(content) => content,
            None => {
                return Err(DocumentError::ContentNotLoaded);
            }
        };

        let questions = questions.unwrap_or(&[]);
        let question_answers = pipeline.analyze(content, questions)?;

        self.question_answers = question_answers;

        Ok(())
    }

    pub fn to_analyze_result(&self) -> Result<AnalysisResult, DocumentError> {
        let content = match self.content() {
            Some(content) => content,
            None => {
                return Err(DocumentError::ContentNotLoaded);
            }
        };

        let mut result = to_analyze_result(&self.doc_type, content, &self.process_id);

        // Set the analysis results from Document
        if !self.question_answers.is_empty() {
            result.set_question_answers(self.question_answers.clone());
        }

        Ok(result)
    }
}
