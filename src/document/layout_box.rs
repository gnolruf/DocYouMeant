use geo::Coord;
use serde::{Deserialize, Serialize};

use crate::document::bounds::Bounds;
use crate::document::text_box::{Orientation, TextBox};
use crate::utils::box_utils;

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
    pub fn from_id(id: usize) -> Option<Self> {
        match id {
            0 => Some(Self::ParagraphTitle),
            1 => Some(Self::Image),
            2 => Some(Self::Text),
            3 => Some(Self::Number),
            4 => Some(Self::Abstract),
            5 => Some(Self::Content),
            6 => Some(Self::FigureTitle),
            7 => Some(Self::Formula),
            8 => Some(Self::Table),
            9 => Some(Self::Reference),
            10 => Some(Self::DocTitle),
            11 => Some(Self::Footnote),
            12 => Some(Self::Header),
            13 => Some(Self::Algorithm),
            14 => Some(Self::Footer),
            15 => Some(Self::Seal),
            16 => Some(Self::Chart),
            17 => Some(Self::FormulaNumber),
            18 => Some(Self::AsideText),
            19 => Some(Self::ReferenceContent),
            _ => None,
        }
    }

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutBox {
    pub bounds: Bounds,
    pub class: LayoutClass,
    pub confidence: f32,
    /// Page number this layout box belongs to (1-indexed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<usize>,
    /// Text content extracted from overlapping text boxes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl LayoutBox {
    pub fn new(bounds: Bounds, class: LayoutClass, confidence: f32) -> Self {
        Self {
            bounds,
            class,
            confidence,
            page_number: None,
            content: None,
        }
    }

    pub fn with_page_number(mut self, page_number: usize) -> Self {
        self.page_number = Some(page_number);
        self
    }

    pub fn with_content(mut self, content: String) -> Self {
        self.content = Some(content);
        self
    }

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
