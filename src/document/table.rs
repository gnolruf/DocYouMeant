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
    /// Row index of the cell (0-indexed)
    pub row_index: usize,
    /// Column index of the cell (0-indexed)
    pub column_index: usize,
    /// Number of rows spanned by this cell
    pub row_span: usize,
    /// Number of columns spanned by this cell
    pub column_span: usize,
}

impl TableCell {
    /// Creates a new TableCell with default row/column information
    pub fn new(bounds: [Coord<i32>; 4], confidence: f32) -> Self {
        Self {
            bounds,
            confidence,
            row_index: 0,
            column_index: 0,
            row_span: 1,
            column_span: 1,
        }
    }

    /// Returns the minimum x coordinate of the cell bounds
    pub fn min_x(&self) -> i32 {
        self.bounds.iter().map(|c| c.x).min().unwrap_or(0)
    }

    /// Returns the maximum x coordinate of the cell bounds
    pub fn max_x(&self) -> i32 {
        self.bounds.iter().map(|c| c.x).max().unwrap_or(0)
    }

    /// Returns the minimum y coordinate of the cell bounds
    pub fn min_y(&self) -> i32 {
        self.bounds.iter().map(|c| c.y).min().unwrap_or(0)
    }

    /// Returns the maximum y coordinate of the cell bounds
    pub fn max_y(&self) -> i32 {
        self.bounds.iter().map(|c| c.y).max().unwrap_or(0)
    }
}

/// Represents a detected table with its structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    /// The bounding box of the entire table as 4 corner coordinates
    #[serde(with = "coord_array_i32")]
    pub bounds: [Coord<i32>; 4],
    /// The type of table (wired or wireless)
    pub table_type: TableType,
    /// Number of rows in the table
    pub row_count: usize,
    /// Number of columns in the table
    pub column_count: usize,
    /// The cells within this table
    pub cells: Vec<TableCell>,
    /// The page number where this table was found
    pub page_number: usize,
    /// Confidence score for the table detection
    pub confidence: f32,
}

impl Table {
    /// Tolerance for clustering cell boundaries into rows/columns (in pixels)
    const CLUSTER_TOLERANCE: i32 = 10;

    /// Creates a new Table from detected cells, automatically determining row/column structure.
    /// Cell bounds are offset to be relative to the document coordinates based on the table bounds.
    pub fn from_cells(
        detected_cells: Vec<TableCell>,
        bounds: [Coord<i32>; 4],
        table_type: TableType,
        page_number: usize,
        confidence: f32,
    ) -> Self {
        if detected_cells.is_empty() {
            return Self {
                bounds,
                table_type,
                row_count: 0,
                column_count: 0,
                cells: Vec::new(),
                page_number,
                confidence,
            };
        }

        // Offset cell bounds to be relative to the document, not the cropped table image
        let offset_x = bounds.iter().map(|b| b.x).min().unwrap_or(0);
        let offset_y = bounds.iter().map(|b| b.y).min().unwrap_or(0);

        let mut cells: Vec<TableCell> = detected_cells
            .into_iter()
            .map(|c| {
                let offset_bounds = [
                    Coord {
                        x: c.bounds[0].x + offset_x,
                        y: c.bounds[0].y + offset_y,
                    },
                    Coord {
                        x: c.bounds[1].x + offset_x,
                        y: c.bounds[1].y + offset_y,
                    },
                    Coord {
                        x: c.bounds[2].x + offset_x,
                        y: c.bounds[2].y + offset_y,
                    },
                    Coord {
                        x: c.bounds[3].x + offset_x,
                        y: c.bounds[3].y + offset_y,
                    },
                ];
                TableCell::new(offset_bounds, c.confidence)
            })
            .collect();

        // Determine row and column structure from cell positions
        let (row_boundaries, col_boundaries) = Self::compute_grid_boundaries(&cells);
        let row_count = row_boundaries.len().saturating_sub(1);
        let column_count = col_boundaries.len().saturating_sub(1);

        // Assign row/column indices and spans to each cell
        for cell in &mut cells {
            Self::assign_cell_position(cell, &row_boundaries, &col_boundaries);
        }

        Self {
            bounds,
            table_type,
            row_count,
            column_count,
            cells,
            page_number,
            confidence,
        }
    }

    /// Computes the row and column boundaries from cell positions
    /// Returns (row_boundaries, column_boundaries) where boundaries are sorted y/x coordinates
    fn compute_grid_boundaries(cells: &[TableCell]) -> (Vec<i32>, Vec<i32>) {
        // Collect all unique y boundaries (top and bottom of each cell)
        let mut y_boundaries: Vec<i32> = cells
            .iter()
            .flat_map(|c| vec![c.min_y(), c.max_y()])
            .collect();

        // Collect all unique x boundaries (left and right of each cell)
        let mut x_boundaries: Vec<i32> = cells
            .iter()
            .flat_map(|c| vec![c.min_x(), c.max_x()])
            .collect();

        // Cluster and deduplicate boundaries
        let row_boundaries = Self::cluster_boundaries(&mut y_boundaries);
        let col_boundaries = Self::cluster_boundaries(&mut x_boundaries);

        (row_boundaries, col_boundaries)
    }

    /// Clusters nearby boundary values to handle slight misalignments
    fn cluster_boundaries(boundaries: &mut [i32]) -> Vec<i32> {
        if boundaries.is_empty() {
            return Vec::new();
        }

        boundaries.sort();

        let mut clustered = Vec::new();
        let mut current_cluster: Vec<i32> = vec![boundaries[0]];

        for &boundary in boundaries.iter().skip(1) {
            if boundary - current_cluster.last().unwrap_or(&0) <= Self::CLUSTER_TOLERANCE {
                current_cluster.push(boundary);
            } else {
                // Compute the mean of the current cluster
                let mean = current_cluster.iter().sum::<i32>() / current_cluster.len() as i32;
                clustered.push(mean);
                current_cluster = vec![boundary];
            }
        }

        // Don't forget the last cluster
        if !current_cluster.is_empty() {
            let mean = current_cluster.iter().sum::<i32>() / current_cluster.len() as i32;
            clustered.push(mean);
        }

        clustered
    }

    /// Assigns row/column index and span to a cell based on grid boundaries
    fn assign_cell_position(cell: &mut TableCell, row_boundaries: &[i32], col_boundaries: &[i32]) {
        let cell_min_y = cell.min_y();
        let cell_max_y = cell.max_y();
        let cell_min_x = cell.min_x();
        let cell_max_x = cell.max_x();

        // Find the row index (which row boundary interval contains the cell top)
        let row_start = Self::find_boundary_index(cell_min_y, row_boundaries);
        let row_end = Self::find_boundary_index(cell_max_y, row_boundaries);

        // Find the column index (which column boundary interval contains the cell left)
        let col_start = Self::find_boundary_index(cell_min_x, col_boundaries);
        let col_end = Self::find_boundary_index(cell_max_x, col_boundaries);

        cell.row_index = row_start;
        cell.column_index = col_start;
        cell.row_span = (row_end - row_start).max(1);
        cell.column_span = (col_end - col_start).max(1);
    }

    /// Finds which boundary interval a coordinate falls into
    fn find_boundary_index(coord: i32, boundaries: &[i32]) -> usize {
        if boundaries.is_empty() {
            return 0;
        }

        // Find the first boundary that is greater than coord
        for (i, &boundary) in boundaries.iter().enumerate() {
            if coord < boundary + Self::CLUSTER_TOLERANCE {
                return if i == 0 { 0 } else { i - 1 };
            }
        }

        // If coord is beyond all boundaries, return the last interval
        boundaries.len().saturating_sub(2)
    }
}
