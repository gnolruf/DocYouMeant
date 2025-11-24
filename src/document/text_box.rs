pub use geo::Coord;
use serde::{Deserialize, Serialize};

use crate::utils::serialization_utils::coord_array_i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentSpan {
    pub offset: usize,
    pub length: usize,
}

impl DocumentSpan {
    pub fn new(offset: usize, length: usize) -> Self {
        Self { offset, length }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Orientation {
    Oriented0,
    Oriented90,
    Oriented180,
    Oriented270,
}

impl Orientation {
    pub fn from_rotation_degrees(degrees: f32) -> Option<Self> {
        let normalized = ((degrees % 360.0) + 360.0) % 360.0;

        let rounded = normalized.round() as i32;

        match rounded {
            0 => Some(Orientation::Oriented0),
            90 => Some(Orientation::Oriented90),
            180 => Some(Orientation::Oriented180),
            270 => Some(Orientation::Oriented270),
            _ => None,
        }
    }

    pub fn most_common(orientations: &[Orientation]) -> Option<Orientation> {
        if orientations.is_empty() {
            return None;
        }

        let mut counts = std::collections::HashMap::new();
        for &orientation in orientations {
            *counts.entry(orientation).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(orientation, _)| orientation)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBox {
    #[serde(with = "coord_array_i32")]
    pub bounds: [Coord<i32>; 4],
    pub angle: Option<Orientation>,
    pub text: Option<String>,
    pub box_score: f32,
    pub text_score: f32,
    pub span: Option<DocumentSpan>,
}

impl TextBox {
    pub fn merge(boxes: &[TextBox]) -> Option<TextBox> {
        if boxes.is_empty() {
            return None;
        }

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

        for b in boxes {
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

        let count = boxes.len() as f32;
        let span = if has_span && min_offset < max_end {
            Some(DocumentSpan::new(min_offset, max_end - min_offset))
        } else {
            None
        };

        Some(TextBox {
            bounds,
            angle,
            text,
            box_score: total_box_score / count,
            text_score: total_text_score / count,
            span,
        })
    }
}
