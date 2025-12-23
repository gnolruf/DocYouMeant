//! Microsoft Word document content handling.
//!
//! This module provides content extraction for Microsoft Word documents (`.docx`).

use docx_rs::*;

use super::super::error::DocumentError;
use super::{DocumentContent, PageContent};
use crate::document::bounds::Bounds;
use crate::document::table::{Table, TableCell, TableType};
use crate::document::text_box::TextBox;

/// Content container for Microsoft Word documents.
///
/// `WordContent` parses `.docx` files and extracts both paragraph text and tables.
#[derive(Debug)]
pub struct WordContent {
    /// The pages extracted from the document.
    pages: Vec<PageContent>,
}

impl DocumentContent for WordContent {
    fn get_pages(&self) -> &Vec<PageContent> {
        &self.pages
    }

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
        &mut self.pages
    }
}

/// Accumulates content for a single page during document parsing.
///
/// This helper struct collects text and tables as they are encountered
/// while iterating through document elements, then converts to a
/// [`PageContent`] when a page boundary is reached.
#[derive(Debug, Default)]
struct PageBuilder {
    /// Text content accumulated for this page.
    text_parts: Vec<String>,
    /// Tables found on this page.
    tables: Vec<Table>,
}

impl PageBuilder {
    /// Creates a new empty page builder.
    ///
    /// # Returns
    ///
    /// A `PageBuilder` with empty text parts and no tables.
    fn new() -> Self {
        Self::default()
    }

    /// Checks if this page has any content.
    ///
    /// # Returns
    ///
    /// `true` if this page has at least one text part or table.
    fn has_content(&self) -> bool {
        !self.text_parts.is_empty() || !self.tables.is_empty()
    }

    /// Converts the builder into a [`PageContent`] instance.
    ///
    /// Text parts are joined with newlines to form the page text.
    ///
    /// # Arguments
    ///
    /// * `page_number` - The 1-indexed page number to assign.
    ///
    /// # Returns
    ///
    /// A [`PageContent`] populated with the accumulated text and tables.
    fn into_page_content(self, page_number: usize) -> PageContent {
        let text = if self.text_parts.is_empty() {
            None
        } else {
            Some(self.text_parts.join("\n"))
        };

        let mut page = PageContent::new(page_number);
        page.text = text;
        page.tables = self.tables;
        page
    }
}

/// Checks if a paragraph contains a page break.
///
/// Examines all runs within the paragraph for `Break` elements with `Page` type.
///
/// # Arguments
///
/// * `paragraph` - The paragraph to examine for page breaks.
///
/// # Returns
///
/// `true` if the paragraph contains at least one page break element.
fn paragraph_has_page_break(paragraph: &Paragraph) -> bool {
    let page_break = Break::new(BreakType::Page);

    for child in &paragraph.children {
        if let ParagraphChild::Run(run) = child {
            for run_child in &run.children {
                if let RunChild::Break(br) = run_child {
                    if *br == page_break {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Extracts text content from a table cell.
///
/// Recursively extracts text from all content within the cell, including
/// paragraphs and nested tables. Multiple text elements are joined with spaces.
///
/// # Arguments
///
/// * `cell` - The table cell to extract text from.
///
/// # Returns
///
/// A string containing all text content from the cell, with elements
/// separated by spaces. Returns an empty string if the cell has no text.
fn extract_table_cell_text(cell: &docx_rs::TableCell) -> String {
    let mut texts = Vec::new();

    for content in &cell.children {
        match content {
            TableCellContent::Paragraph(p) => {
                let text = p.raw_text();
                if !text.is_empty() {
                    texts.push(text);
                }
            }
            TableCellContent::Table(nested_table) => {
                for row_child in &nested_table.rows {
                    let TableChild::TableRow(row) = row_child;
                    for cell_child in &row.cells {
                        let TableRowChild::TableCell(nested_cell) = cell_child;
                        let nested_text = extract_table_cell_text(nested_cell);
                        if !nested_text.is_empty() {
                            texts.push(nested_text);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    texts.join(" ")
}

/// Extracts the grid span (column span) from a table cell property.
///
/// Since the `grid_span` field is private in docx-rs, this function
/// serializes the property to JSON and extracts the `gridSpan` value.
///
/// # Arguments
///
/// * `property` - The table cell property containing grid span information.
///
/// # Returns
///
/// The column span value, or `1` if grid span is not set or extraction fails.
fn get_grid_span(property: &docx_rs::TableCellProperty) -> usize {
    // Serialize to JSON and extract gridSpan value
    if let Ok(json) = serde_json::to_value(property) {
        if let Some(grid_span) = json.get("gridSpan") {
            if let Some(span) = grid_span.as_u64() {
                return span as usize;
            }
        }
    }
    1
}

/// Converts a docx-rs table to the internal [`Table`] representation.
///
/// Creates table cells with proper row/column indices and span information
/// based on the document structure. All bounds are set to zero using
/// [`Bounds::zero()`] since Word documents store structural data rather
/// than geometric positioning.
///
/// # Arguments
///
/// * `docx_table` - The docx-rs table to convert.
/// * `page_number` - The 1-indexed page number where this table appears.
///
/// # Returns
///
/// A [`Table`] instance with:
/// - Cells populated with row/column indices and content
/// - `table_type` set to [`TableType::Wired`] (most Word tables have borders)
/// - `confidence` set to `1.0` (direct extraction)
/// - All bounds set to zero coordinates
fn convert_docx_table(docx_table: &docx_rs::Table, page_number: usize) -> Table {
    let mut cells = Vec::new();
    let mut row_count = 0;
    let mut max_columns = 0;

    for row_child in &docx_table.rows {
        let TableChild::TableRow(row) = row_child;
        let mut col_index = 0;

        for cell_child in &row.cells {
            let TableRowChild::TableCell(cell) = cell_child;
            let text = extract_table_cell_text(cell);

            let col_span = get_grid_span(&cell.property);

            let content = if text.is_empty() {
                None
            } else {
                Some(TextBox {
                    bounds: Bounds::zero(),
                    angle: None,
                    text: Some(text),
                    box_score: 1.0,
                    text_score: 1.0,
                    span: None,
                })
            };

            let mut table_cell = TableCell::new(Bounds::zero(), 1.0);
            table_cell.row_index = row_count;
            table_cell.column_index = col_index;
            table_cell.row_span = 1;
            table_cell.column_span = col_span;
            table_cell.content = content;

            cells.push(table_cell);
            col_index += col_span;
        }

        if col_index > max_columns {
            max_columns = col_index;
        }
        row_count += 1;
    }

    Table {
        bounds: Bounds::zero(),
        table_type: TableType::Wired,
        row_count,
        column_count: max_columns,
        cells,
        page_number,
        confidence: 1.0,
    }
}

/// Converts a table to a plain text representation.
///
/// Creates a text representation suitable for inclusion in the page text,
/// with cells separated by tabs and rows separated by newlines.
///
/// # Arguments
///
/// * `table` - The table to convert to text.
///
/// # Returns
///
/// A string with tab-separated cells and newline-separated rows.
/// Returns an empty string if the table has no cells.
fn table_to_text(table: &Table) -> String {
    if table.cells.is_empty() {
        return String::new();
    }

    let mut rows: Vec<Vec<&TableCell>> = vec![Vec::new(); table.row_count];
    for cell in &table.cells {
        if cell.row_index < table.row_count {
            rows[cell.row_index].push(cell);
        }
    }

    for row in &mut rows {
        row.sort_by_key(|c| c.column_index);
    }

    let row_texts: Vec<String> = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| {
                    cell.content
                        .as_ref()
                        .and_then(|c| c.text.as_ref())
                        .map(|s| s.as_str())
                        .unwrap_or("")
                })
                .collect::<Vec<_>>()
                .join("\t")
        })
        .collect();

    row_texts.join("\n")
}

impl WordContent {
    /// Loads and parses a Word document from raw bytes.
    ///
    /// This method parses the `.docx` file structure and extracts:
    /// - Text from all paragraphs, split into pages by page breaks
    /// - Tables with their cell structure and content
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes of the `.docx` file.
    ///
    /// # Returns
    ///
    /// A boxed [`DocumentContent`] trait object containing the extracted
    /// content organized into pages, or a [`DocumentError::WordLoadError`] if
    /// parsing fails.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::WordLoadError`] if:
    /// - The file is not a valid `.docx` format
    /// - The document structure is corrupted
    /// - Required XML components are missing or malformed
    pub fn load(bytes: &[u8]) -> Result<Box<dyn DocumentContent>, DocumentError> {
        let docx = docx_rs::read_docx(bytes).map_err(|e| DocumentError::WordLoadError {
            message: e.to_string(),
        })?;

        let mut pages: Vec<PageContent> = Vec::new();
        let mut current_page = PageBuilder::new();
        let mut page_number = 1;

        for child in &docx.document.children {
            match child {
                DocumentChild::Paragraph(p) => {
                    let has_page_break = paragraph_has_page_break(p);

                    let text = p.raw_text();
                    if !text.is_empty() {
                        current_page.text_parts.push(text);
                    }

                    if has_page_break && current_page.has_content() {
                        pages.push(current_page.into_page_content(page_number));
                        page_number += 1;
                        current_page = PageBuilder::new();
                    }
                }
                DocumentChild::Table(table) => {
                    let converted_table = convert_docx_table(table, page_number);

                    let table_text = table_to_text(&converted_table);
                    if !table_text.is_empty() {
                        current_page.text_parts.push(table_text);
                    }

                    current_page.tables.push(converted_table);
                }
                _ => {}
            }
        }

        if current_page.has_content() {
            pages.push(current_page.into_page_content(page_number));
        }

        if pages.is_empty() {
            pages.push(PageContent::new(1));
        }

        Ok(Box::new(Self { pages }))
    }
}
