use geo::Coord;
use serde::{Deserialize, Serialize};

use crate::document::bounds::Bounds;
use crate::document::layout_box::{LayoutBox, LayoutClass};
use crate::document::text_box::{Orientation, TextBox};
use crate::utils::box_utils::{self, HasBounds};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RegionRole {
    Title,
    SectionHeading,
    Footnote,
    PageHeader,
    PageFooter,
    PageNumber,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRegion {
    pub role: RegionRole,
    pub page_number: usize,
    pub bounds: Bounds,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl DocumentRegion {
    pub fn new(role: RegionRole, page_number: usize, bounds: Bounds, content: String) -> Self {
        Self {
            role,
            page_number,
            bounds,
            content,
            confidence: None,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

impl HasBounds for TextBox {
    fn get_bounds(&self) -> &Bounds {
        &self.bounds
    }
}

pub struct DocumentRegionBuilder;

impl DocumentRegionBuilder {
    pub fn build_regions(
        page_number: usize,
        layout_boxes: &[LayoutBox],
        text_boxes: &[TextBox],
    ) -> Vec<DocumentRegion> {
        let mut regions = Vec::new();
        let mut used_text_boxes = vec![false; text_boxes.len()];

        for layout_box in layout_boxes {
            let role = match layout_box.class {
                LayoutClass::ParagraphTitle | LayoutClass::FigureTitle | LayoutClass::DocTitle => {
                    Some(RegionRole::Title)
                }
                LayoutClass::Footer | LayoutClass::Footnote => Some(RegionRole::PageFooter),
                LayoutClass::Header => Some(RegionRole::PageHeader),
                LayoutClass::Number => Some(RegionRole::PageNumber),
                _ => None,
            };

            if let Some(role) = role {
                if let Some(combined_text_box) = Self::combine_overlapping_text_boxes(
                    &layout_box.bounds,
                    text_boxes,
                    &mut used_text_boxes,
                    0.5,
                ) {
                    let avg_confidence = combined_text_box.text_score.max(layout_box.confidence);

                    let content = combined_text_box.text.unwrap_or_default();
                    let bounds = combined_text_box.bounds;

                    regions.push(
                        DocumentRegion::new(role, page_number, bounds, content)
                            .with_confidence(avg_confidence),
                    );
                }
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
