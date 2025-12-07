use geo::Coord;
use serde::{Deserialize, Serialize};

use crate::document::text_box::TextBox;
use crate::utils::serialization_utils::coord_array_i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TableType {
    /// A table with visible grid lines/borders
    Wired,
    /// A table without visible grid lines/borders
    Wireless,
}

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
    /// The text content within this cell, created from matched words
    pub content: Option<TextBox>,
}

impl TableCell {
    pub fn new(bounds: [Coord<i32>; 4], confidence: f32) -> Self {
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

    pub fn min_x(&self) -> i32 {
        self.bounds.iter().map(|c| c.x).min().unwrap_or(0)
    }

    pub fn max_x(&self) -> i32 {
        self.bounds.iter().map(|c| c.x).max().unwrap_or(0)
    }

    pub fn min_y(&self) -> i32 {
        self.bounds.iter().map(|c| c.y).min().unwrap_or(0)
    }

    pub fn max_y(&self) -> i32 {
        self.bounds.iter().map(|c| c.y).max().unwrap_or(0)
    }
}

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

    fn compute_grid_from_edges(cells: &[TableCell]) -> (Vec<i32>, Vec<i32>) {
        let mut y_edges: Vec<i32> = cells.iter().flat_map(|c| [c.min_y(), c.max_y()]).collect();

        let mut x_edges: Vec<i32> = cells.iter().flat_map(|c| [c.min_x(), c.max_x()]).collect();

        let row_lines = Self::cluster_edges(&mut y_edges);
        let col_lines = Self::cluster_edges(&mut x_edges);

        (row_lines, col_lines)
    }

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

    pub fn match_words_to_cells(&mut self, words: &[TextBox], overlap_threshold: f32) {
        use crate::utils::box_utils;

        for cell in &mut self.cells {
            let matched_words: Vec<TextBox> = words
                .iter()
                .filter(|word| {
                    box_utils::calculate_overlap(&word.bounds, &cell.bounds) >= overlap_threshold
                })
                .cloned()
                .collect();

            cell.set_content_from_words(&matched_words);
        }
    }
}
