//! Bounding box representation with utility methods.

use geo::Coord;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Represents a quadrilateral bounding box with four corner coordinates.
///
/// The corners are stored in order: top-left, top-right, bottom-right, bottom-left.
/// This struct provides utility methods for common spatial calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    coords: [Coord<i32>; 4],
}

impl Bounds {
    /// Creates a new `Bounds` from four corner coordinates.
    ///
    /// # Arguments
    ///
    /// * `coords` - Four corner points in order: [top-left, top-right, bottom-right, bottom-left]
    #[inline]
    pub fn new(coords: [Coord<i32>; 4]) -> Self {
        Self { coords }
    }

    /// Returns the underlying coordinate array.
    #[inline]
    pub fn coords(&self) -> &[Coord<i32>; 4] {
        &self.coords
    }

    /// Returns a mutable reference to the underlying coordinate array.
    #[inline]
    pub fn coords_mut(&mut self) -> &mut [Coord<i32>; 4] {
        &mut self.coords
    }

    /// Returns the coordinates as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[Coord<i32>] {
        &self.coords
    }

    /// Returns an iterator over the coordinates.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Coord<i32>> {
        self.coords.iter()
    }

    /// Returns the minimum x coordinate (left edge).
    #[inline]
    pub fn left(&self) -> i32 {
        self.coords.iter().map(|c| c.x).min().unwrap_or(0)
    }

    /// Returns the maximum x coordinate (right edge).
    #[inline]
    pub fn right(&self) -> i32 {
        self.coords.iter().map(|c| c.x).max().unwrap_or(0)
    }

    /// Returns the minimum y coordinate (top edge).
    #[inline]
    pub fn top(&self) -> i32 {
        self.coords.iter().map(|c| c.y).min().unwrap_or(0)
    }

    /// Returns the maximum y coordinate (bottom edge).
    #[inline]
    pub fn bottom(&self) -> i32 {
        self.coords.iter().map(|c| c.y).max().unwrap_or(0)
    }

    /// Returns the width of the bounding box (right - left).
    #[inline]
    pub fn width(&self) -> i32 {
        (self.right() - self.left()).max(1)
    }

    /// Returns the height of the bounding box (bottom - top).
    #[inline]
    pub fn height(&self) -> i32 {
        (self.bottom() - self.top()).max(1)
    }

    /// Returns the center point (centroid) of the bounding box.
    #[inline]
    pub fn centroid(&self) -> Coord<i32> {
        Coord {
            x: (self.left() + self.right()) / 2,
            y: (self.top() + self.bottom()) / 2,
        }
    }

    /// Returns the center x coordinate.
    #[inline]
    pub fn center_x(&self) -> i32 {
        (self.left() + self.right()) / 2
    }

    /// Returns the center y coordinate.
    #[inline]
    pub fn center_y(&self) -> i32 {
        (self.top() + self.bottom()) / 2
    }
}

impl std::ops::Index<usize> for Bounds {
    type Output = Coord<i32>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl std::ops::IndexMut<usize> for Bounds {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl From<[Coord<i32>; 4]> for Bounds {
    #[inline]
    fn from(coords: [Coord<i32>; 4]) -> Self {
        Self::new(coords)
    }
}

impl From<Bounds> for [Coord<i32>; 4] {
    #[inline]
    fn from(bounds: Bounds) -> Self {
        bounds.coords
    }
}

impl Serialize for Bounds {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;

        #[derive(Serialize)]
        struct CoordHelper {
            x: i32,
            y: i32,
        }

        let mut seq = serializer.serialize_seq(Some(4))?;
        for coord in &self.coords {
            seq.serialize_element(&CoordHelper {
                x: coord.x,
                y: coord.y,
            })?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for Bounds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct CoordHelper {
            x: i32,
            y: i32,
        }

        let helpers: Vec<CoordHelper> = Vec::deserialize(deserializer)?;
        if helpers.len() != 4 {
            return Err(serde::de::Error::custom(format!(
                "Expected 4 coordinates, got {}",
                helpers.len()
            )));
        }

        Ok(Self::new([
            Coord {
                x: helpers[0].x,
                y: helpers[0].y,
            },
            Coord {
                x: helpers[1].x,
                y: helpers[1].y,
            },
            Coord {
                x: helpers[2].x,
                y: helpers[2].y,
            },
            Coord {
                x: helpers[3].x,
                y: helpers[3].y,
            },
        ]))
    }
}
