//! Excel spreadsheet document content handling.
//!
//! This module provides content extraction for Microsoft Excel files (`.xlsx`).
//! Each worksheet in the workbook is represented as a separate page, with cell
//! contents extracted and formatted as tab-separated text.

use calamine::{Reader, Xlsx};

use super::super::error::DocumentError;
use super::{DocumentContent, PageContent};

/// Represents a single worksheet from an Excel workbook.
///
/// Contains the grid of cells extracted from one Excel sheet.
/// This is an intermediate representation used during parsing.
#[derive(Debug)]
pub struct ExcelSheet {
    /// The cell data organized as rows of cells.
    pub data: Vec<Vec<ExcelCell>>,
}

/// Represents a single cell from an Excel worksheet.
///
/// Contains the formatted string representation of the cell's value,
/// regardless of the original cell type (number, date, formula result, etc.).
#[derive(Debug)]
pub struct ExcelCell {
    /// The formatted string representation of the cell value.
    pub formatted: String,
}

/// Content container for Excel workbook documents.
///
/// `ExcelContent` parses `.xlsx` files and represents each worksheet as a
/// separate page. Cell contents are extracted and converted to text with
/// tab-separated columns and newline-separated rows.
///
/// # Multi-Sheet Handling
///
/// Each worksheet becomes a separate [`PageContent`] entry, with the sheet
/// name included as a header in the text content.
#[derive(Debug)]
pub struct ExcelContent {
    /// The pages extracted from the workbook (one per worksheet).
    pages: Vec<PageContent>,
}

impl DocumentContent for ExcelContent {
    fn get_pages(&self) -> &Vec<PageContent> {
        &self.pages
    }

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
        &mut self.pages
    }
}

impl ExcelContent {
    /// Loads and parses an Excel workbook from raw bytes.
    ///
    /// This method reads an `.xlsx` file and extracts text content from all
    /// worksheets. Each worksheet is converted to a separate page with its
    /// name as a header followed by tab-separated cell values.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes of the Excel file.
    ///
    /// # Returns
    ///
    /// A boxed [`DocumentContent`] trait object containing one page per
    /// worksheet, or a [`DocumentError::ExcelLoadError`] if parsing fails.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::ExcelLoadError`] if:
    /// - The file is not a valid `.xlsx` format
    /// - The workbook structure is corrupted
    /// - A worksheet cannot be read
    pub fn load(bytes: &[u8]) -> Result<Box<dyn DocumentContent>, DocumentError> {
        use std::io::Cursor;
        let cursor = Cursor::new(bytes);
        let mut workbook: Xlsx<_> = calamine::open_workbook_from_rs(cursor)
            .map_err(|source| DocumentError::ExcelLoadError { source })?;

        let mut pages = Vec::new();
        let mut page_num = 1;

        for name in workbook.sheet_names().to_owned() {
            let range = workbook
                .worksheet_range(&name)
                .map_err(|source| DocumentError::ExcelLoadError { source })?;

            let mut sheet_data = Vec::new();
            for row in range.rows() {
                let row_data = row
                    .iter()
                    .map(|cell| ExcelCell {
                        formatted: cell.to_string(),
                    })
                    .collect();
                sheet_data.push(row_data);
            }

            let sheet = ExcelSheet { data: sheet_data };

            let rows = sheet
                .data
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| cell.formatted.as_str())
                        .collect::<Vec<_>>()
                        .join("\t")
                })
                .collect::<Vec<_>>()
                .join("\n");
            let sheet_text = format!("Sheet: {name}\n{rows}");

            let page = PageContent::with_text(page_num, sheet_text);
            pages.push(page);
            page_num += 1;
        }

        Ok(Box::new(Self { pages }))
    }
}
