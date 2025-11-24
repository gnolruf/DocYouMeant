use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::document::content::{KeyValuePair, PageContent};
use crate::document::text_box::TextBox;
use crate::inference::error::InferenceError;
use crate::inference::phi4mini::Phi4MiniInference;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValuePairExtractionResult {
    pub pairs: Vec<KeyValuePair>,
}

impl KeyValuePairExtractionResult {
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }
}

impl Default for KeyValuePairExtractionResult {
    fn default() -> Self {
        Self::new()
    }
}

pub struct KeyValuePairExtractionTask;

impl KeyValuePairExtractionTask {
    pub fn get_or_init() -> Result<(), InferenceError> {
        Phi4MiniInference::get_or_init()
    }

    pub fn extract(
        page_content: &PageContent,
    ) -> Result<KeyValuePairExtractionResult, InferenceError> {
        let text = page_content.text.clone().unwrap_or_default();
        if text.trim().is_empty() {
            return Ok(KeyValuePairExtractionResult::new());
        }

        let tokenizer = Phi4MiniInference::get_tokenizer()?;
        let system_message = Self::build_system_message();
        let prompt = format!("Document text:\n{text}");

        let response = Phi4MiniInference::with_instance(|model| {
            model.generate(&prompt, tokenizer, Some(&system_message))
        })?;

        let string_pairs = Self::parse_response(&response)?;

        let mut result_pairs = Vec::new();
        let mut used_indices = HashSet::new();
        let words = &page_content.words;

        for (key_str, value_str) in string_pairs {
            let key_box = Self::find_text_box(&key_str, words, &mut used_indices);
            if let Some(k_box) = key_box {
                let value_box = Self::find_text_box(&value_str, words, &mut used_indices);
                if let Some(v_box) = value_box {
                    result_pairs.push(KeyValuePair {
                        key: k_box,
                        value: v_box,
                    });
                }
            }
        }

        Ok(KeyValuePairExtractionResult {
            pairs: result_pairs,
        })
    }

    fn find_text_box(
        target_text: &str,
        words: &[TextBox],
        used_indices: &mut HashSet<usize>,
    ) -> Option<TextBox> {
        let target_clean = target_text.replace(|c: char| !c.is_alphanumeric(), "");
        if target_clean.is_empty() {
            return None;
        }

        for (i, _word) in words.iter().enumerate() {
            if used_indices.contains(&i) {
                continue;
            }

            let mut current_indices = Vec::new();
            let mut accumulated_text = String::new();
            let mut next_word_idx = i;

            while next_word_idx < words.len() {
                if used_indices.contains(&next_word_idx) {
                    next_word_idx += 1;
                    continue;
                }

                if let Some(text) = &words[next_word_idx].text {
                    let text_clean = text.replace(|c: char| !c.is_alphanumeric(), "");
                    accumulated_text.push_str(&text_clean);
                    current_indices.push(next_word_idx);

                    if accumulated_text == target_clean {
                        for idx in &current_indices {
                            used_indices.insert(*idx);
                        }
                        let matched_boxes: Vec<TextBox> = current_indices
                            .iter()
                            .map(|&idx| words[idx].clone())
                            .collect();
                        return TextBox::merge(&matched_boxes);
                    } else if accumulated_text.len() > target_clean.len() {
                        // Overshot, stop this path
                        break;
                    } else if !target_clean.starts_with(&accumulated_text) {
                        // Mismatch, stop this path
                        break;
                    }
                }
                next_word_idx += 1;
            }
        }
        None
    }

    fn build_system_message() -> String {
        r#"You are a document analysis expert. Extract key-value pairs from the document text provided by the user.

INSTRUCTIONS:
1. Extract ONLY explicit key-value pairs where a label/field name is followed by a value
2. Keys must be EXACT text from the document (the label/field name as written)
3. Values must be EXACT text from the document (preserve formatting, spacing, numbers)
4. Common patterns: "FIELD: value", "FIELD value", "FIELD\nvalue"
5. DO NOT extract addresses, phone numbers, or website URLs unless they have an explicit label
6. DO NOT extract instructional text, warnings, or footer information
7. DO NOT invent keys - only use labels that actually appear in the document
8. DO NOT include metadata about the document itself unless explicitly labeled
9. Return a flat JSON object (no nested arrays or objects)
10. Return ONLY the JSON object with no markdown formatting, explanation, or extra text

Example patterns to extract:
- "DATE: 12/10/98" → {"DATE": "12/10/98"}
- "TO: John Smith" → {"TO": "John Smith"}
- "Invoice Number: 12345" → {"Invoice Number: "12345"}

DO NOT extract:
- Unlabeled addresses or contact information
- Instructions or warnings
- Generic company information without labels
- The prompt instructions themselves

Output ONLY the JSON object:
{"key": "value", "key2": "value2"}"#.to_string()
    }

    fn parse_response(response: &str) -> Result<HashMap<String, String>, InferenceError> {
        let cleaned = response
            .trim()
            .strip_prefix("```json")
            .unwrap_or(response)
            .strip_prefix("```")
            .unwrap_or(response)
            .strip_suffix("```")
            .unwrap_or(response)
            .trim();

        let json_str = if let Some(start) = cleaned.find('{') {
            if let Some(end) = cleaned.rfind('}') {
                &cleaned[start..=end]
            } else {
                cleaned
            }
        } else {
            return Ok(HashMap::new());
        };

        let raw_pairs: HashMap<String, serde_json::Value> = serde_json::from_str(json_str)
            .map_err(|e| InferenceError::PredictionError {
                operation: "parse JSON response".to_string(),
                message: format!("Failed to parse JSON object: {e}\nResponse: {json_str}"),
            })?;

        let mut pairs = HashMap::new();
        for (key, value) in raw_pairs {
            if let Some(string_value) = value.as_str() {
                pairs.insert(key, string_value.to_string());
            } else {
                eprintln!("Warning: Skipping non-string value for key '{key}': {value:?}");
            }
        }

        Ok(pairs)
    }
}
