use geo::Coord;
use serde::{Deserialize, Serialize};

use crate::utils::serialization_utils::coord_array_i32;

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutBox {
    #[serde(with = "coord_array_i32")]
    pub bounds: [Coord<i32>; 4],
    pub class: LayoutClass,
    pub confidence: f32,
}
