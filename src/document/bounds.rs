//! Bounding box representation with utility methods.
//!
//! This module provides the [`Bounds`] type for representing quadrilateral bounding boxes.

use geo::Coord;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Represents a quadrilateral bounding box with four corner coordinates.
///
/// The corners are stored in order: top-left, top-right, bottom-right, bottom-left.
/// This struct provides utility methods for common spatial calculations such as
/// computing dimensions, center points, and edge positions.
///
/// # Coordinate System
///
/// The coordinate system assumes:
/// - **X-axis**: Increases from left to right
/// - **Y-axis**: Increases from top to bottom (standard image coordinates)
///
/// # Corner Order
///
/// ```text
/// [0] top-left -------- [1] top-right
///       |                      |
///       |                      |
/// [3] bottom-left ---- [2] bottom-right
/// ```
///
/// # Serialization
///
/// `Bounds` implements [`Serialize`] and [`Deserialize`], serializing as an array
/// of coordinate objects:
///
/// ```json
/// [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 50}, {"x": 0, "y": 50}]
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// The four corner coordinates in order: top-left, top-right, bottom-right, bottom-left.
    coords: [Coord<i32>; 4],
}

impl Bounds {
    /// Creates a new `Bounds` from four corner coordinates.
    ///
    /// # Arguments
    ///
    /// * `coords` - Four corner points in order: `[top-left, top-right, bottom-right, bottom-left]`
    #[inline]
    pub fn new(coords: [Coord<i32>; 4]) -> Self {
        Self { coords }
    }

    /// Creates a zero-sized `Bounds` with all coordinates at the origin.
    ///
    /// This is useful for document elements where geometric positioning is not
    /// available.
    ///
    /// # Returns
    ///
    /// A `Bounds` instance with all four corners at `(0, 0)`.
    #[inline]
    pub fn zero() -> Self {
        Self {
            coords: [
                Coord { x: 0, y: 0 },
                Coord { x: 0, y: 0 },
                Coord { x: 0, y: 0 },
                Coord { x: 0, y: 0 },
            ],
        }
    }

    /// Returns a reference to the underlying coordinate array.
    ///
    /// # Returns
    ///
    /// A reference to the array of four [`Coord<i32>`] points representing the corners.
    #[inline]
    pub fn coords(&self) -> &[Coord<i32>; 4] {
        &self.coords
    }

    /// Returns a mutable reference to the underlying coordinate array.
    ///
    /// This allows in-place modification of the bounding box coordinates.
    ///
    /// # Returns
    ///
    /// A mutable reference to the array of four [`Coord<i32>`] points.
    #[inline]
    pub fn coords_mut(&mut self) -> &mut [Coord<i32>; 4] {
        &mut self.coords
    }

    /// Returns the coordinates as a slice.
    ///
    /// This is useful when you need to pass the coordinates to functions that
    /// accept slices rather than fixed-size arrays.
    ///
    /// # Returns
    ///
    /// A slice containing all four corner coordinates.
    #[inline]
    pub fn as_slice(&self) -> &[Coord<i32>] {
        &self.coords
    }

    /// Returns an iterator over the coordinates.
    ///
    /// The iterator yields coordinates in order: top-left, top-right, bottom-right, bottom-left.
    ///
    /// # Returns
    ///
    /// An iterator yielding references to each [`Coord<i32>`].
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Coord<i32>> {
        self.coords.iter()
    }

    /// Returns the minimum x coordinate (left edge).
    ///
    /// This finds the leftmost point among all four corners, which is useful
    /// for non-axis-aligned quadrilaterals.
    ///
    /// # Returns
    ///
    /// The smallest x value among all coordinates, or `0` if the bounds are empty.
    #[inline]
    pub fn left(&self) -> i32 {
        self.coords.iter().map(|c| c.x).min().unwrap_or(0)
    }

    /// Returns the maximum x coordinate (right edge).
    ///
    /// This finds the rightmost point among all four corners.
    ///
    /// # Returns
    ///
    /// The largest x value among all coordinates, or `0` if the bounds are empty.
    #[inline]
    pub fn right(&self) -> i32 {
        self.coords.iter().map(|c| c.x).max().unwrap_or(0)
    }

    /// Returns the minimum y coordinate (top edge).
    ///
    /// In image coordinates where y increases downward, this returns the
    /// topmost (smallest y) point among all four corners.
    ///
    /// # Returns
    ///
    /// The smallest y value among all coordinates, or `0` if the bounds are empty.
    #[inline]
    pub fn top(&self) -> i32 {
        self.coords.iter().map(|c| c.y).min().unwrap_or(0)
    }

    /// Returns the maximum y coordinate (bottom edge).
    ///
    /// In image coordinates where y increases downward, this returns the
    /// bottommost (largest y) point among all four corners.
    ///
    /// # Returns
    ///
    /// The largest y value among all coordinates, or `0` if the bounds are empty.
    #[inline]
    pub fn bottom(&self) -> i32 {
        self.coords.iter().map(|c| c.y).max().unwrap_or(0)
    }

    /// Returns the width of the bounding box (right - left).
    ///
    /// The width is calculated as the difference between the rightmost and
    /// leftmost x coordinates. The result is guaranteed to be at least 1.
    ///
    /// # Returns
    ///
    /// The width of the axis-aligned bounding rectangle, minimum value of `1`.
    #[inline]
    pub fn width(&self) -> i32 {
        (self.right() - self.left()).max(1)
    }

    /// Returns the height of the bounding box (bottom - top).
    ///
    /// The height is calculated as the difference between the bottommost and
    /// topmost y coordinates. The result is guaranteed to be at least 1.
    ///
    /// # Returns
    ///
    /// The height of the axis-aligned bounding rectangle, minimum value of `1`.
    #[inline]
    pub fn height(&self) -> i32 {
        (self.bottom() - self.top()).max(1)
    }

    /// Returns the center point (centroid) of the bounding box.
    ///
    /// The centroid is calculated as the midpoint of the axis-aligned bounding
    /// rectangle, not the true centroid of the quadrilateral.
    ///
    /// # Returns
    ///
    /// A [`Coord<i32>`] representing the center point.
    ///
    /// # Note
    ///
    /// Due to integer division, the result may be slightly off-center for
    /// odd-dimensioned bounding boxes.
    #[inline]
    pub fn centroid(&self) -> Coord<i32> {
        Coord {
            x: (self.left() + self.right()) / 2,
            y: (self.top() + self.bottom()) / 2,
        }
    }

    /// Returns the center x coordinate.
    ///
    /// Equivalent to `self.centroid().x` but avoids computing the y coordinate.
    ///
    /// # Returns
    ///
    /// The x coordinate of the bounding box center.
    #[inline]
    pub fn center_x(&self) -> i32 {
        (self.left() + self.right()) / 2
    }

    /// Returns the center y coordinate.
    ///
    /// Equivalent to `self.centroid().y` but avoids computing the x coordinate.
    ///
    /// # Returns
    ///
    /// The y coordinate of the bounding box center.
    #[inline]
    pub fn center_y(&self) -> i32 {
        (self.top() + self.bottom()) / 2
    }
}

/// Allows indexing into the bounds to access individual corner coordinates.
///
/// # Panics
///
/// Panics if `index >= 4`.
impl std::ops::Index<usize> for Bounds {
    type Output = Coord<i32>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

/// Allows mutable indexing into the bounds to modify individual corner coordinates.
///
/// # Panics
///
/// Panics if `index >= 4`.
impl std::ops::IndexMut<usize> for Bounds {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

/// Creates a `Bounds` from an array of four coordinates.
///
/// This is equivalent to calling [`Bounds::new`].
impl From<[Coord<i32>; 4]> for Bounds {
    #[inline]
    fn from(coords: [Coord<i32>; 4]) -> Self {
        Self::new(coords)
    }
}

/// Converts a `Bounds` back into an array of four coordinates.
impl From<Bounds> for [Coord<i32>; 4] {
    #[inline]
    fn from(bounds: Bounds) -> Self {
        bounds.coords
    }
}

/// Serializes `Bounds` as an array of coordinate objects.
///
/// Each coordinate is serialized as `{"x": <value>, "y": <value>}`.
///
/// # Output Format
///
/// ```json
/// [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 50}, {"x": 0, "y": 50}]
/// ```
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

/// Deserializes `Bounds` from an array of coordinate objects.
///
/// # Errors
///
/// Returns an error if the input does not contain exactly 4 coordinate objects.
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
