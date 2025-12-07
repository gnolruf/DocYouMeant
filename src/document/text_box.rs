pub use geo::Coord;
use serde::{Deserialize, Serialize};

use crate::document::bounds::Bounds;

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
    pub bounds: Bounds,
    pub angle: Option<Orientation>,
    pub text: Option<String>,
    pub box_score: f32,
    pub text_score: f32,
    pub span: Option<DocumentSpan>,
}
