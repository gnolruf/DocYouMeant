//! Layout detection types and region building utilities.
//!
//! This module provides the [`LayoutBox`] type for representing detected layout regions
//! in a document, along with the [`LayoutClass`] enum for classifying different types
//! of document elements such as paragraphs, titles, tables, and images.

use geo::Coord;
use serde::{Deserialize, Serialize};

use crate::document::bounds::Bounds;
use crate::document::text_box::{Orientation, TextBox};
use crate::utils::box_utils;

/// Classification of document layout elements.
///
/// Each variant represents a distinct type of content region that can be
/// detected in a document during layout analysis. The numeric values correspond
/// to the model's output class IDs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutClass {
    ParagraphTitle = 0,
    Image = 1,
    Text = 2,
    Number = 3,
    Abstract = 4,
    Content = 5,
    FigureTitle = 6,
    Formula = 7,
    Table = 8,
    Reference = 9,
    DocTitle = 10,
    Footnote = 11,
    Header = 12,
    Algorithm = 13,
    Footer = 14,
    Seal = 15,
    Chart = 16,
    FormulaNumber = 17,
    AsideText = 18,
    ReferenceContent = 19,
}

impl LayoutClass {
    /// All layout classes in order of their numeric IDs (0-19).
    const ALL_CLASSES: [Self; 20] = [
        Self::ParagraphTitle,
        Self::Image,
        Self::Text,
        Self::Number,
        Self::Abstract,
        Self::Content,
        Self::FigureTitle,
        Self::Formula,
        Self::Table,
        Self::Reference,
        Self::DocTitle,
        Self::Footnote,
        Self::Header,
        Self::Algorithm,
        Self::Footer,
        Self::Seal,
        Self::Chart,
        Self::FormulaNumber,
        Self::AsideText,
        Self::ReferenceContent,
    ];

    /// Converts a numeric class ID to the corresponding [`LayoutClass`] variant.
    ///
    /// # Arguments
    ///
    /// * `id` - The numeric class ID from model output (0-19).
    ///
    /// # Returns
    ///
    /// Returns `Some(LayoutClass)` if the ID is valid (0-19), or `None` for invalid IDs.
    #[inline]
    #[must_use]
    pub fn from_id(id: usize) -> Option<Self> {
        Self::ALL_CLASSES.get(id).copied()
    }

    /// Checks if this layout class represents a text region.
    ///
    /// Text regions are layout elements that typically contain standalone text
    /// content which should be extracted and associated with the layout box.
    /// These include titles, headers, footers, and other labeled text areas.
    ///
    /// # Returns
    ///
    /// Returns `true` if this class is an applicable region class.
    ///
    /// Returns `false` for all other layout classes.
    pub fn is_region(&self) -> bool {
        matches!(
            self,
            Self::ParagraphTitle
                | Self::FigureTitle
                | Self::DocTitle
                | Self::Footer
                | Self::Footnote
                | Self::Header
                | Self::Number
        )
    }
}

/// A detected layout region within a document page.
///
/// `LayoutBox` represents a single detected element in the document's layout,
/// such as a paragraph, title, table, or image region. Each box contains
/// spatial bounds, a classification, and an optional extracted text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutBox {
    /// The quadrilateral bounding coordinates of the layout region.
    pub bounds: Bounds,
    /// The classification type of this layout element.
    pub class: LayoutClass,
    /// Model confidence score for this detection (0.0 to 1.0).
    pub confidence: f32,
    /// Page number this layout box belongs to (1-indexed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<usize>,
    /// Text content extracted from overlapping text boxes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl LayoutBox {
    /// Creates a new `LayoutBox` with the specified bounds, class, and confidence.
    ///
    /// The `page_number` and `content` fields are initialized to `None`.
    /// Use [`with_page_number`](Self::with_page_number) and [`with_content`](Self::with_content)
    /// to set these optional fields.
    ///
    /// # Arguments
    ///
    /// * `bounds` - The quadrilateral bounding coordinates of the layout region.
    /// * `class` - The classification type of this layout element.
    /// * `confidence` - Model confidence score (typically 0.0 to 1.0).
    #[must_use]
    pub fn new(bounds: Bounds, class: LayoutClass, confidence: f32) -> Self {
        Self {
            bounds,
            class,
            confidence,
            page_number: None,
            content: None,
        }
    }

    /// Sets the page number for this layout box.
    ///
    /// This method consumes the `LayoutBox` and returns it with the page number set,
    /// enabling method chaining in a builder pattern.
    ///
    /// # Arguments
    ///
    /// * `page_number` - The 1-indexed page number this layout box belongs to.
    ///
    /// # Returns
    ///
    /// Returns the modified `LayoutBox` with the page number set.
    #[must_use]
    pub fn with_page_number(mut self, page_number: usize) -> Self {
        self.page_number = Some(page_number);
        self
    }

    /// Sets the text content for this layout box.
    ///
    /// This method consumes the `LayoutBox` and returns it with the content set,
    /// enabling method chaining in a builder pattern.
    ///
    /// # Arguments
    ///
    /// * `content` - The extracted text content for this layout region.
    ///
    /// # Returns
    ///
    /// Returns the modified `LayoutBox` with the content set.
    #[must_use]
    pub fn with_content(mut self, content: String) -> Self {
        self.content = Some(content);
        self
    }

    /// Builds text regions by combining layout boxes with overlapping text boxes.
    ///
    /// This method processes layout boxes that are classified as regions (see
    /// [`LayoutClass::is_region`]) and associates them with overlapping text boxes
    /// to extract their text content. Each text box can only be used once.
    ///
    /// # Arguments
    ///
    /// * `page_number` - The 1-indexed page number to assign to the resulting regions.
    /// * `layout_boxes` - Slice of detected layout boxes from layout analysis.
    /// * `text_boxes` - Slice of detected text boxes from OCR.
    ///
    /// # Returns
    ///
    /// A vector of `LayoutBox` instances representing regions with their extracted
    /// text content. Only layout boxes where [`LayoutClass::is_region`] returns `true`
    /// are processed.
    ///
    /// # Algorithm
    ///
    /// For each region-type layout box:
    /// 1. Find all text boxes that overlap by more than 50%.
    /// 2. Mark those text boxes as used (preventing reuse).
    /// 3. Combine the text and bounds of overlapping text boxes.
    /// 4. Create a new `LayoutBox` with the combined content.
    pub fn build_regions(
        page_number: usize,
        layout_boxes: &[LayoutBox],
        text_boxes: &[TextBox],
    ) -> Vec<LayoutBox> {
        let mut regions = Vec::new();
        let mut used_text_boxes = vec![false; text_boxes.len()];

        for layout_box in layout_boxes {
            if !layout_box.class.is_region() {
                continue;
            }

            if let Some(combined_text_box) = Self::combine_overlapping_text_boxes(
                &layout_box.bounds,
                text_boxes,
                &mut used_text_boxes,
                0.5,
            ) {
                let avg_confidence = combined_text_box.text_score.max(layout_box.confidence);
                let content = combined_text_box.text.unwrap_or_default();

                let mut region =
                    LayoutBox::new(combined_text_box.bounds, layout_box.class, avg_confidence)
                        .with_page_number(page_number);

                if !content.is_empty() {
                    region = region.with_content(content);
                }

                regions.push(region);
            }
        }

        regions
    }

    /// Combines text boxes that overlap with a layout region's bounds.
    ///
    /// This internal method finds all text boxes that sufficiently overlap with
    /// the given layout bounds, marks them as used, and combines their content
    /// into a single `TextBox`.
    ///
    /// # Arguments
    ///
    /// * `layout_bounds` - The bounding region to find overlapping text boxes for.
    /// * `text_boxes` - Slice of all detected text boxes.
    /// * `used_text_boxes` - Mutable tracking array to mark text boxes as consumed.
    /// * `overlap_threshold` - Minimum overlap ratio (0.0 to 1.0) required for inclusion.
    ///
    /// # Returns
    ///
    /// Returns `Some(TextBox)` containing the combined text and bounds if any
    /// overlapping text boxes were found, or `None` if no text boxes overlap
    /// above the threshold.
    fn combine_overlapping_text_boxes(
        layout_bounds: &Bounds,
        text_boxes: &[TextBox],
        used_text_boxes: &mut [bool],
        overlap_threshold: f32,
    ) -> Option<TextBox> {
        let mut overlapping_texts = Vec::new();

        for (i, text_box) in text_boxes.iter().enumerate() {
            if used_text_boxes[i] {
                continue;
            }

            let overlap =
                box_utils::calculate_overlap(text_box.bounds.as_slice(), layout_bounds.as_slice());

            if overlap > overlap_threshold {
                overlapping_texts.push((i, text_box.clone()));
            }
        }

        if overlapping_texts.is_empty() {
            return None;
        }

        for (i, _) in &overlapping_texts {
            used_text_boxes[*i] = true;
        }

        overlapping_texts.sort_by(|a, b| {
            let a_y = a.1.bounds.top();
            let b_y = b.1.bounds.top();
            let a_x = a.1.bounds.left();
            let b_x = b.1.bounds.left();

            match a_y.cmp(&b_y) {
                std::cmp::Ordering::Equal => a_x.cmp(&b_x),
                other => other,
            }
        });

        let combined_text = overlapping_texts
            .iter()
            .filter_map(|(_, tb)| tb.text.as_ref())
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");

        let all_coords: Vec<Coord<i32>> = overlapping_texts
            .iter()
            .flat_map(|(_, tb)| tb.bounds.as_slice().iter().copied())
            .collect();

        let min_x = all_coords.iter().map(|c| c.x).min().unwrap_or(0);
        let max_x = all_coords.iter().map(|c| c.x).max().unwrap_or(0);
        let min_y = all_coords.iter().map(|c| c.y).min().unwrap_or(0);
        let max_y = all_coords.iter().map(|c| c.y).max().unwrap_or(0);

        let combined_bounds = Bounds::new([
            Coord { x: min_x, y: min_y },
            Coord { x: max_x, y: min_y },
            Coord { x: max_x, y: max_y },
            Coord { x: min_x, y: max_y },
        ]);

        let total_box_score: f32 = overlapping_texts.iter().map(|(_, tb)| tb.box_score).sum();
        let total_text_score: f32 = overlapping_texts.iter().map(|(_, tb)| tb.text_score).sum();
        let count = overlapping_texts.len() as f32;

        let avg_box_score = total_box_score / count;
        let avg_text_score = total_text_score / count;

        let orientations: Vec<Orientation> = overlapping_texts
            .iter()
            .filter_map(|(_, tb)| tb.angle)
            .collect();
        let combined_angle = Orientation::most_common(&orientations);

        Some(TextBox {
            bounds: combined_bounds,
            angle: combined_angle,
            text: Some(combined_text),
            box_score: avg_box_score,
            text_score: avg_text_score,
            span: None,
        })
    }
}
