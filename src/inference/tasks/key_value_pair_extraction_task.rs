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

struct WordInfo {
    clean_text: String,
    original: TextBox,
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

STRICT RULES:
1. Extract ONLY explicit key-value pairs where a label/field name has a CONCRETE value
2. The KEY is the field label (e.g., "Date", "Invoice Number", "Name")
3. The VALUE is the actual data (e.g., "12/10/98", "12345", "John Smith")
4. NEVER set a value to null - skip the pair entirely if there's no value
5. NEVER include the value inside the key - separate them properly
6. NEVER extract document titles, section headers, or form field names without values
7. NEVER hallucinate or invent data - ONLY EXTRACT WHAT IS EXPLICITLY PRESENT
8. NEVER arbitrarily repeat similar entries - THE DATA YOU PROVIDE MUST COME EXACTLY FROM THE TEXT
9. ONLY valid key value pairs with string keys and string values

CORRECT examples:
- "Date: 12/10/98" → "Date": "12/10/98"
- "Invoice Number 12345" → "Invoice Number": "12345"
- "Lic # 48000040179" → "Lic #": "48000040179"
- "Total Due: $1,234.56" → "Total Due": "$1,234.56"
- "Ship To: John Doe" → "Ship To": "John Doe"
- "Order Date: 2023-01-01" → "Order Date": "2023-01-01"

WRONG examples (DO NOT do these):
- "Lic #: 48000040179" → "Lic #": "48000040179" ← value is inside the key
- "Document Title": null ← null values are not allowed
- "Field Name": null ← skip fields without values
- "Item #1": "X", "Item #2": "X", "Item #3": "X", ... ← don't create repetitive entries
- "Invoice": "Invoice" ← key and value are identical/redundant
- "Date": "Unknown" ← do not invent "Unknown" values
- "Header": "Company Name" ← do not label general text as "Header"
- "Legal Description SECTION: 7 BLOCK : 1961": null ← value is inside the key, and value is set to null
- "Main File No. 1552474": null ← value is inside the key, and value is set to null
- "Gross Living Area": null ← null values are not allowed
- "Total Rooms": null ← null values are not allowed
- "Total Bedrooms": null ← null values are not allowed
- "Total Bathrooms": null ← null values are not allowed
- "Location": null ← null values are not allowed
- "View": null ← null values are not allowed
- "Site": null ← null values are not allowed
- "Quality": null ← null values are not allowed
- "Age": null ← null values are not allowed

Output ONLY valid key value pairs with string keys and string values:
{"key": "value", "key2": "value2"}"#.to_string()
    }

    fn parse_response(response: &str) -> Result<HashMap<String, String>, InferenceError> {
        let mut pairs = HashMap::new();

        let chars: Vec<char> = response.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            if chars[i] == '"' {
                let key_start = i;
                if let Some((key, key_end)) = Self::extract_quoted_string(&chars, i) {
                    i = key_end;

                    while i < len && chars[i].is_whitespace() {
                        i += 1;
                    }

                    if i < len && chars[i] == ':' {
                        i += 1;

                        while i < len && chars[i].is_whitespace() {
                            i += 1;
                        }

                        if i + 4 <= len {
                            let potential_null: String = chars[i..i + 4].iter().collect();
                            if potential_null == "null" {
                                i += 4;
                                continue;
                            }
                        }

                        if i < len && chars[i] == '"' {
                            if let Some((value, value_end)) = Self::extract_quoted_string(&chars, i)
                            {
                                i = value_end;

                                let key_trimmed = key.trim();
                                let value_trimmed = value.trim();

                                if !key_trimmed.is_empty()
                                    && !value_trimmed.is_empty()
                                    && !value_trimmed.eq_ignore_ascii_case("null")
                                {
                                    pairs
                                        .insert(key_trimmed.to_string(), value_trimmed.to_string());
                                }
                                continue;
                            }
                        }
                    }
                } else {
                    i = key_start + 1;
                    continue;
                }
            }
            i += 1;
        }

        Ok(pairs)
    }

    fn extract_quoted_string(chars: &[char], start: usize) -> Option<(String, usize)> {
        if start >= chars.len() || chars[start] != '"' {
            return None;
        }

        let mut result = String::new();
        let mut i = start + 1;
        let len = chars.len();

        while i < len {
            let c = chars[i];
            if c == '"' {
                return Some((result, i + 1));
            } else if c == '\\' && i + 1 < len {
                i += 1;
                let escaped = chars[i];
                match escaped {
                    '"' => result.push('"'),
                    '\\' => result.push('\\'),
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    _ => {
                        result.push('\\');
                        result.push(escaped);
                    }
                }
            } else {
                result.push(c);
            }
            i += 1;
        }
        None
    }
}
