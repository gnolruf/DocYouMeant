use geo::Coord;
use serde::{Deserialize, Serialize};

use crate::utils::serialization_utils::coord_array_i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TableType {
    /// A table with visible grid lines/borders
    Wired,
    /// A table without visible grid lines/borders
    Wireless,
}

/// Represents a detected cell within a table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCell {
    /// The bounding box of the cell as 4 corner coordinates
    #[serde(with = "coord_array_i32")]
    pub bounds: [Coord<i32>; 4],
    /// The confidence score of the detection
    pub confidence: f32,
}
