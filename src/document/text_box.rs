//! Text detection and recognition types.
//!
//! This module provides types for representing text regions in documents,
//! including the [`TextBox`] struct for individual text detections, [`Orientation`]
//! for text rotation, and [`DocumentSpan`] for tracking text positions within
//! the full document content.

pub use geo::Coord;
use serde::{Deserialize, Serialize};

use crate::document::bounds::Bounds;

/// A span representing a text region's position within the full document content.
///
/// `DocumentSpan` tracks where a piece of text appears in the concatenated
/// document text, enabling mapping between spatial text boxes and their
/// position in the linear text representation.
///
/// # Fields
///
/// * `offset` - The starting character index in the document text (0-indexed).
/// * `length` - The number of characters this span covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentSpan {
    /// The starting character index in the document text (0-indexed).
    pub offset: usize,
    /// The number of characters this span covers.
    pub length: usize,
}

impl DocumentSpan {
    /// Creates a new `DocumentSpan` with the specified offset and length.
    ///
    /// # Arguments
    ///
    /// * `offset` - The starting character index (0-indexed).
    /// * `length` - The number of characters in the span.
    ///
    /// # Returns
    ///
    /// A new `DocumentSpan` instance.
    #[must_use]
    pub fn new(offset: usize, length: usize) -> Self {
        Self { offset, length }
    }
}

/// Text orientation representing rotation in 90-degree increments.
///
/// This enum represents the four cardinal orientations that text can have
/// in a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Orientation {
    /// Normal horizontal text (0 degree rotation).
    Oriented0,
    /// Text rotated 90 degrees clockwise.
    Oriented90,
    /// Upside-down text (180 degrees rotation).
    Oriented180,
    /// Text rotated 270 degrees clockwise (90 degrees counter-clockwise).
    Oriented270,
}

impl Orientation {
    /// Converts a rotation angle in degrees to an [`Orientation`] variant.
    ///
    /// The input angle is normalized to the range [0, 360) and rounded to
    /// the nearest cardinal direction (0 degrees, 90 degrees, 180 degrees, or 270 degrees).
    ///
    /// # Arguments
    ///
    /// * `degrees` - The rotation angle in degrees. Can be any value,
    ///   including negative angles and angles > 360 degrees.
    ///
    /// # Returns
    ///
    /// Returns `Some(Orientation)` if the rounded angle matches a cardinal
    /// direction exactly, or `None` for non-cardinal angles.
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

    /// Finds the most frequently occurring orientation in a collection.
    ///
    /// This is useful for determining the dominant text orientation on a page
    /// or within a region by analyzing multiple text boxes.
    ///
    /// # Arguments
    ///
    /// * `orientations` - A slice of orientations to analyze.
    ///
    /// # Returns
    ///
    /// Returns `Some(Orientation)` with the most common orientation, or `None`
    /// if the input slice is empty. If there's a tie, one of the tied
    /// orientations is returned (not guaranteed which one).
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

/// A detected text region with recognized content and confidence scores.
///
/// `TextBox` represents a single text detection from the pipeline,
/// containing both spatial information (where the text is) and semantic
/// information (what the text says).
///
/// # Confidence Scores
///
/// Two separate confidence scores are provided:
/// - `box_score` - Confidence in the text detection (finding that text exists)
/// - `text_score` - Confidence in the text recognition (reading what it says)
///
/// Both scores range from 0.0 to 1.0, with higher values indicating
/// greater confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBox {
    /// Quadrilateral bounding coordinates of the text region.
    pub bounds: Bounds,
    /// Detected text orientation, if determined.
    pub angle: Option<Orientation>,
    /// Recognized text content from OCR.
    pub text: Option<String>,
    /// Confidence score for text detection (0.0 to 1.0).
    pub box_score: f32,
    /// Confidence score for text recognition (0.0 to 1.0).
    pub text_score: f32,
    /// Position of this text within the full document content.
    pub span: Option<DocumentSpan>,
}
