use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::document::content::{KeyValuePair, PageContent};
use crate::document::text_box::{Coord, DocumentSpan, Orientation, TextBox};
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

        let mut word_infos = Vec::with_capacity(words.len());
        let mut bigram_index: HashMap<String, Vec<usize>> = HashMap::new();

        for (i, word) in words.iter().enumerate() {
            let clean_text = if let Some(text) = &word.text {
                text.replace(|c: char| !c.is_alphanumeric(), "")
            } else {
                String::new()
            };

            if !clean_text.is_empty() {
                let prefix = if clean_text.len() >= 2 {
                    &clean_text[0..2]
                } else {
                    &clean_text[0..1]
                };
                bigram_index.entry(prefix.to_string()).or_default().push(i);
            }

            word_infos.push(WordInfo {
                clean_text,
                original: word.clone(),
            });
        }

        for (key_str, value_str) in string_pairs {
            let key_box =
                Self::find_text_box(&key_str, &word_infos, &bigram_index, &mut used_indices);
            if let Some(k_box) = key_box {
                let value_box =
                    Self::find_text_box(&value_str, &word_infos, &bigram_index, &mut used_indices);
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
        word_infos: &[WordInfo],
        bigram_index: &HashMap<String, Vec<usize>>,
        used_indices: &mut HashSet<usize>,
    ) -> Option<TextBox> {
        let target_clean = target_text.replace(|c: char| !c.is_alphanumeric(), "");
        if target_clean.is_empty() {
            return None;
        }

        let candidates = if target_clean.len() >= 2 {
            let mut c = Vec::new();
            if let Some(idxs) = bigram_index.get(&target_clean[0..2]) {
                c.extend(idxs);
            }
            if let Some(idxs) = bigram_index.get(&target_clean[0..1]) {
                c.extend(idxs);
            }
            c
        } else {
            bigram_index
                .get(&target_clean[0..1])
                .cloned()
                .unwrap_or_default()
        };

        if candidates.is_empty() {
            return None;
        }

        for &start_idx in &candidates {
            if used_indices.contains(&start_idx) {
                continue;
            }

            let mut current_indices = Vec::new();
            let mut accumulated_text = String::new();
            let mut next_word_idx = start_idx;

            let mut min_x = i32::MAX;
            let mut min_y = i32::MAX;
            let mut max_x = i32::MIN;
            let mut max_y = i32::MIN;
            let mut total_box_score = 0.0;
            let mut total_text_score = 0.0;
            let mut text_parts = Vec::new();
            let mut orientations = Vec::new();
            let mut min_offset = usize::MAX;
            let mut max_end = 0;
            let mut has_span = false;

            while next_word_idx < word_infos.len() {
                if used_indices.contains(&next_word_idx) {
                    next_word_idx += 1;
                    continue;
                }

                let info = &word_infos[next_word_idx];
                if info.clean_text.is_empty() {
                    next_word_idx += 1;
                    continue;
                }

                accumulated_text.push_str(&info.clean_text);
                current_indices.push(next_word_idx);

                let b = &info.original;
                for p in b.bounds {
                    min_x = min_x.min(p.x);
                    min_y = min_y.min(p.y);
                    max_x = max_x.max(p.x);
                    max_y = max_y.max(p.y);
                }
                total_box_score += b.box_score;
                total_text_score += b.text_score;
                if let Some(t) = &b.text {
                    text_parts.push(t.clone());
                }
                if let Some(o) = b.angle {
                    orientations.push(o);
                }
                if let Some(span) = b.span {
                    has_span = true;
                    min_offset = min_offset.min(span.offset);
                    max_end = max_end.max(span.offset + span.length);
                }

                if accumulated_text == target_clean {
                    for idx in &current_indices {
                        used_indices.insert(*idx);
                    }

                    let bounds = [
                        Coord { x: min_x, y: min_y },
                        Coord { x: max_x, y: min_y },
                        Coord { x: max_x, y: max_y },
                        Coord { x: min_x, y: max_y },
                    ];

                    let angle = Orientation::most_common(&orientations);
                    let text = if text_parts.is_empty() {
                        None
                    } else {
                        Some(text_parts.join(" "))
                    };

                    let count = current_indices.len() as f32;
                    let span = if has_span && min_offset < max_end {
                        Some(DocumentSpan::new(min_offset, max_end - min_offset))
                    } else {
                        None
                    };

                    return Some(TextBox {
                        bounds,
                        angle,
                        text,
                        box_score: total_box_score / count,
                        text_score: total_text_score / count,
                        span,
                    });
                } else if accumulated_text.len() > target_clean.len() {
                    // Overshot, stop this path
                    break;
                } else if !target_clean.starts_with(&accumulated_text) {
                    // Mismatch, stop this path
                    break;
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

struct WordInfo {
    clean_text: String,
    original: TextBox,
}
