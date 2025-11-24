use serde::{Deserialize, Serialize};

use crate::inference::error::InferenceError;
use crate::inference::phi4mini::Phi4MiniInference;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionAndAnswerResult {
    pub question: String,
    pub answer: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl QuestionAndAnswerResult {
    pub fn new(question: String, answer: String) -> Self {
        Self {
            question,
            answer,
            confidence: None,
        }
    }
}

pub struct QuestionAndAnswerTask;

impl QuestionAndAnswerTask {
    pub fn get_or_init() -> Result<(), InferenceError> {
        Phi4MiniInference::get_or_init()
    }

    pub fn answer(
        context: &str,
        question: &str,
    ) -> Result<QuestionAndAnswerResult, InferenceError> {
        if context.trim().is_empty() {
            return Err(InferenceError::PreprocessingError {
                operation: "validate input".to_string(),
                message: "Context cannot be empty".to_string(),
            });
        }

        if question.trim().is_empty() {
            return Err(InferenceError::PreprocessingError {
                operation: "validate input".to_string(),
                message: "Question cannot be empty".to_string(),
            });
        }

        let tokenizer = Phi4MiniInference::get_tokenizer()?;
        let system_message = Self::build_system_message();
        let prompt = format!("Document Context:\n{context}\n\nQuestion: {question}");

        let response = Phi4MiniInference::with_instance(|model| {
            model.generate(&prompt, tokenizer, Some(&system_message))
        })?;

        Self::parse_response(&response, question)
    }

    fn build_system_message() -> String {
        r#"You are a document analysis expert. Answer questions based ONLY on the information provided in the document context.

INSTRUCTIONS:
1. Answer the question based ONLY on the information in the document context
2. If the answer is not in the document, respond with "I cannot find this information in the provided document."
3. Be concise and direct in your answer
4. Quote relevant parts of the document when appropriate
5. Do not make assumptions or add information not present in the document
6. If the question is ambiguous, provide the most reasonable interpretation based on the context

Provide your answer directly without additional formatting or explanation."#.to_string()
    }

    fn parse_response(
        response: &str,
        question: &str,
    ) -> Result<QuestionAndAnswerResult, InferenceError> {
        let answer = response.trim().to_string();

        if answer.is_empty() {
            return Err(InferenceError::PredictionError {
                operation: "parse response".to_string(),
                message: "Model returned an empty response".to_string(),
            });
        }

        Ok(QuestionAndAnswerResult::new(question.to_string(), answer))
    }
}
