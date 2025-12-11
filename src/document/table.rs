//! Table detection and cell structure representation.
//!
//! This module provides types for representing detected tables in documents,
//! including the [`Table`] struct for the overall table structure and [`TableCell`]
//! for individual cells within a table.

use geo::Coord;
use serde::{Deserialize, Serialize};

use crate::document::bounds::Bounds;
use crate::document::text_box::TextBox;

/// Classification of table structure based on visual appearance.
///
/// This enum distinguishes between tables that have visible borders/grid lines
/// and those that rely on spatial alignment without visible separators.
/// Different detection models may be used for each type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TableType {
    /// A table with visible grid lines/borders.
    ///
    /// These tables have explicit visual separators between cells,
    /// making cell boundaries easier to detect.
    Wired,
    /// A table without visible grid lines/borders.
    ///
    /// These tables rely on spatial alignment and whitespace to
    /// define cell boundaries, requiring different detection strategies.
    Wireless,
}

/// A single cell within a detected table.
///
/// Each `TableCell` represents one cell in the table grid, with its spatial
/// bounds, grid position (row/column), span information for merged cells,
/// and optional text content extracted from OCR.
///
/// # Grid Position
///
/// The `row_index` and `column_index` are 0-indexed positions in the table grid.
/// For merged cells, these represent the top-left position of the merged region,
/// with `row_span` and `column_span` indicating how many grid cells are covered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCell {
    /// The bounding box of the cell as 4 corner coordinates.
    pub bounds: Bounds,
    /// The confidence score of the cell detection (0.0 to 1.0).
    pub confidence: f32,
    /// Row index of the cell (0-indexed from top).
    pub row_index: usize,
    /// Column index of the cell (0-indexed from left).
    pub column_index: usize,
    /// Number of rows spanned by this cell (≥1, >1 for merged cells).
    pub row_span: usize,
    /// Number of columns spanned by this cell (≥1, >1 for merged cells).
    pub column_span: usize,
    /// The text content within this cell, created from matched OCR words.
    pub content: Option<TextBox>,
}

impl TableCell {
    /// Creates a new `TableCell` with the specified bounds and confidence.
    ///
    /// The cell is initialized with default grid position (0, 0) and no span
    /// (1x1). Use [`Table::from_cells`] to automatically compute proper grid
    /// positions based on cell geometry.
    ///
    /// # Arguments
    ///
    /// * `bounds` - The quadrilateral bounding coordinates of the cell.
    /// * `confidence` - Detection confidence score (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// A new `TableCell` with:
    /// - `row_index: 0`, `column_index: 0`
    /// - `row_span: 1`, `column_span: 1`
    /// - `content: None`
    pub fn new(bounds: Bounds, confidence: f32) -> Self {
        Self {
            bounds,
            confidence,
            row_index: 0,
            column_index: 0,
            row_span: 1,
            column_span: 1,
            content: None,
        }
    }

    /// Sets the cell's text content by combining multiple OCR word boxes.
    ///
    /// This method takes a slice of [`TextBox`] words that have been matched
    /// to this cell and combines them into a single content `TextBox`. The
    /// words are joined with spaces, and confidence scores are averaged.
    ///
    /// # Arguments
    ///
    /// * `words` - Slice of text boxes representing words within this cell.
    pub fn set_content_from_words(&mut self, words: &[TextBox]) {
        if words.is_empty() {
            return;
        }

        let mut texts = Vec::new();
        let mut total_text_score = 0.0;
        let mut total_box_score = 0.0;

        for word in words {
            if let Some(ref text) = word.text {
                texts.push(text.clone());
                total_text_score += word.text_score;
                total_box_score += word.box_score;
            }
        }

        if texts.is_empty() {
            return;
        }

        let word_count = texts.len() as f32;
        let combined_text = texts.join(" ");

        self.content = Some(TextBox {
            bounds: self.bounds,
            angle: None,
            text: Some(combined_text),
            box_score: total_box_score / word_count,
            text_score: total_text_score / word_count,
            span: None,
        });
    }

    /// Returns the minimum x coordinate (left edge) of the cell.
    #[inline]
    pub fn min_x(&self) -> i32 {
        self.bounds.left()
    }

    /// Returns the maximum x coordinate (right edge) of the cell.
    #[inline]
    pub fn max_x(&self) -> i32 {
        self.bounds.right()
    }

    /// Returns the minimum y coordinate (top edge) of the cell.
    #[inline]
    pub fn min_y(&self) -> i32 {
        self.bounds.top()
    }

    /// Returns the maximum y coordinate (bottom edge) of the cell.
    #[inline]
    pub fn max_y(&self) -> i32 {
        self.bounds.bottom()
    }
}

/// A detected table structure within a document page.
///
/// `Table` represents a complete table with its grid structure, cell contents,
/// and metadata. Tables are constructed from individually detected cells, which
/// are then organized into a coherent grid structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    /// The bounding box of the entire table as 4 corner coordinates.
    pub bounds: Bounds,
    /// The type of table (wired with borders or wireless without).
    pub table_type: TableType,
    /// Number of rows in the table grid.
    pub row_count: usize,
    /// Number of columns in the table grid.
    pub column_count: usize,
    /// The cells within this table, each with grid position and content.
    pub cells: Vec<TableCell>,
    /// The 1-indexed page number where this table was found.
    pub page_number: usize,
    /// Confidence score for the table detection (0.0 to 1.0).
    pub confidence: f32,
}

impl Table {
    /// Tolerance for clustering cell boundaries into rows/columns (in pixels).
    ///
    /// Cell edges within this distance of each other are considered to be
    /// on the same grid line. This accounts for minor detection inaccuracies.
    const CLUSTER_TOLERANCE: i32 = 10;

    /// Creates a `Table` from a collection of detected cells.
    ///
    /// This method takes raw cell detections and organizes them into a structured
    /// table by:
    /// 1. Offsetting cell coordinates to the table's position
    /// 2. Computing row and column grid lines by clustering cell edges
    /// 3. Assigning each cell to its grid position with appropriate spans
    ///
    /// # Arguments
    ///
    /// * `detected_cells` - Vector of detected cells (coordinates relative to table bounds).
    /// * `bounds` - The bounding box of the entire table in page coordinates.
    /// * `table_type` - Whether the table is wired (bordered) or wireless.
    /// * `page_number` - The 1-indexed page number containing this table.
    /// * `confidence` - Detection confidence score for the table (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// A fully constructed `Table` with computed grid structure and cell positions.
    ///
    /// # Note
    ///
    /// If `detected_cells` is empty, returns a table with `row_count: 0`,
    /// `column_count: 0`, and an empty `cells` vector.
    pub fn from_cells(
        detected_cells: Vec<TableCell>,
        bounds: Bounds,
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

        let offset_x = bounds.left();
        let offset_y = bounds.top();

        let mut cells: Vec<TableCell> = detected_cells
            .into_iter()
            .map(|c| {
                let offset_bounds = Bounds::new([
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
                ]);
                TableCell::new(offset_bounds, c.confidence)
            })
            .collect();

        let (row_lines, col_lines) = Self::compute_grid_from_edges(&cells);

        let row_count = if row_lines.len() > 1 {
            row_lines.len() - 1
        } else {
            1
        };
        let column_count = if col_lines.len() > 1 {
            col_lines.len() - 1
        } else {
            1
        };

        for cell in &mut cells {
            Self::assign_cell_to_grid(cell, &row_lines, &col_lines);
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

    /// Computes the table grid lines from cell edge positions.
    ///
    /// Extracts all horizontal (y) and vertical (x) edges from the cells,
    /// then clusters nearby edges to determine the canonical grid lines.
    ///
    /// # Arguments
    ///
    /// * `cells` - Slice of table cells to extract edges from.
    ///
    /// # Returns
    ///
    /// A tuple of `(row_lines, column_lines)` where:
    /// - `row_lines` are the y-coordinates of horizontal grid lines (sorted)
    /// - `column_lines` are the x-coordinates of vertical grid lines (sorted)
    fn compute_grid_from_edges(cells: &[TableCell]) -> (Vec<i32>, Vec<i32>) {
        let mut y_edges: Vec<i32> = cells.iter().flat_map(|c| [c.min_y(), c.max_y()]).collect();

        let mut x_edges: Vec<i32> = cells.iter().flat_map(|c| [c.min_x(), c.max_x()]).collect();

        let row_lines = Self::cluster_edges(&mut y_edges);
        let col_lines = Self::cluster_edges(&mut x_edges);

        (row_lines, col_lines)
    }

    /// Clusters nearby edge coordinates into canonical grid lines.
    ///
    /// Groups edges that are within [`CLUSTER_TOLERANCE`](Self::CLUSTER_TOLERANCE)
    /// pixels of each other and replaces each group with its mean value.
    /// This handles minor variations in cell boundary detection.
    ///
    /// # Arguments
    ///
    /// * `edges` - Mutable slice of edge coordinates to cluster (will be sorted).
    ///
    /// # Returns
    ///
    /// A sorted vector of clustered grid line positions.
    fn cluster_edges(edges: &mut [i32]) -> Vec<i32> {
        if edges.is_empty() {
            return Vec::new();
        }

        edges.sort();

        let mut clustered = Vec::new();
        let mut current_cluster: Vec<i32> = vec![edges[0]];

        for &edge in edges.iter().skip(1) {
            if edge - current_cluster.last().unwrap_or(&0) <= Self::CLUSTER_TOLERANCE {
                current_cluster.push(edge);
            } else {
                let mean = current_cluster.iter().sum::<i32>() / current_cluster.len() as i32;
                clustered.push(mean);
                current_cluster = vec![edge];
            }
        }

        if !current_cluster.is_empty() {
            let mean = current_cluster.iter().sum::<i32>() / current_cluster.len() as i32;
            clustered.push(mean);
        }

        clustered
    }

    /// Assigns a cell to its position in the table grid.
    ///
    /// Finds which grid lines the cell's edges align with and sets the cell's
    /// `row_index`, `column_index`, `row_span`, and `column_span` accordingly.
    ///
    /// # Arguments
    ///
    /// * `cell` - The cell to assign (modified in place).
    /// * `row_lines` - The y-coordinates of horizontal grid lines.
    /// * `col_lines` - The x-coordinates of vertical grid lines.
    fn assign_cell_to_grid(cell: &mut TableCell, row_lines: &[i32], col_lines: &[i32]) {
        let row_start = Self::find_aligned_line_index(cell.min_y(), row_lines);
        let row_end = Self::find_aligned_line_index(cell.max_y(), row_lines);

        let col_start = Self::find_aligned_line_index(cell.min_x(), col_lines);
        let col_end = Self::find_aligned_line_index(cell.max_x(), col_lines);

        cell.row_index = row_start;
        cell.column_index = col_start;

        cell.row_span = if row_end > row_start {
            row_end - row_start
        } else {
            1
        };
        cell.column_span = if col_end > col_start {
            col_end - col_start
        } else {
            1
        };
    }

    /// Finds the index of the grid line closest to a coordinate.
    ///
    /// Used to determine which grid line a cell edge aligns with.
    ///
    /// # Arguments
    ///
    /// * `coord` - The coordinate to find alignment for.
    /// * `lines` - The sorted grid line positions.
    ///
    /// # Returns
    ///
    /// The index of the closest grid line, or `0` if `lines` is empty.
    fn find_aligned_line_index(coord: i32, lines: &[i32]) -> usize {
        if lines.is_empty() {
            return 0;
        }

        let mut min_dist = i32::MAX;
        let mut best_idx = 0;

        for (i, &line) in lines.iter().enumerate() {
            let dist = (coord - line).abs();
            if dist < min_dist {
                min_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Matches OCR word boxes to table cells based on spatial overlap.
    ///
    /// For each cell, finds all words that overlap by at least the specified
    /// threshold and combines them into the cell's content. This populates
    /// the `content` field of each [`TableCell`].
    ///
    /// # Arguments
    ///
    /// * `words` - Slice of OCR-detected text boxes to match.
    /// * `overlap_threshold` - Minimum overlap ratio (0.0 to 1.0) for a word
    ///   to be considered part of a cell. A value of 0.5 means at least 50%
    ///   of the word must overlap with the cell.
    pub fn match_words_to_cells(&mut self, words: &[TextBox], overlap_threshold: f32) {
        use crate::utils::box_utils;

        for cell in &mut self.cells {
            let matched_words: Vec<TextBox> = words
                .iter()
                .filter(|word| {
                    box_utils::calculate_overlap(word.bounds.as_slice(), cell.bounds.as_slice())
                        >= overlap_threshold
                })
                .cloned()
                .collect();

            cell.set_content_from_words(&matched_words);
        }
    }
}
