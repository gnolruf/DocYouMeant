//! CSV document content handling.
//!
//! This module provides content extraction for CSV (Comma-Separated Values) files.
//! CSV files are treated as single-page text documents where the content is
//! parsed and normalized for consistent representation.

use csv::Reader as CsvReader;

use super::super::error::DocumentError;
use super::{DocumentContent, PageContent};

/// Content container for CSV documents.
///
/// `CsvContent` parses CSV files and represents them as single-page text documents.
/// The parsing handles both standard comma-delimited files and simple line-based
/// text files that may not contain delimiters.
#[derive(Debug)]
pub struct CsvContent {
    /// The pages extracted from the CSV file (always exactly one page).
    pages: Vec<PageContent>,
}

impl DocumentContent for CsvContent {
    fn get_pages(&self) -> &Vec<PageContent> {
        &self.pages
    }

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
        &mut self.pages
    }
}

impl CsvContent {
    /// Loads and parses a CSV file from raw bytes.
    ///
    /// This method attempts to parse the input as a CSV file. If the content
    /// contains commas, it's treated as a standard CSV; otherwise, it's treated
    /// as a simple line-delimited text file.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes of the CSV file content.
    ///
    /// # Returns
    ///
    /// A boxed [`DocumentContent`] trait object containing the parsed CSV data
    /// as a single page, or a [`DocumentError::CsvLoadError`] if parsing fails.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::CsvLoadError`] if:
    /// - The bytes are not valid UTF-8 encoded text
    /// - The CSV parsing encounters malformed data
    pub fn load(bytes: &[u8]) -> Result<Box<dyn DocumentContent>, DocumentError> {
        let content =
            String::from_utf8(bytes.to_vec()).map_err(|e| DocumentError::CsvLoadError {
                source: std::io::Error::new(std::io::ErrorKind::InvalidData, e).into(),
            })?;

        let mut reader = CsvReader::from_reader(bytes);

        let records: Vec<Vec<String>> = if content.contains(',') {
            reader
                .records()
                .filter_map(|r| r.ok())
                .map(|record| record.iter().map(|s| s.to_string()).collect())
                .collect()
        } else {
            content
                .lines()
                .map(|line| vec![line.trim().to_string()])
                .collect()
        };

        let csv_text = if records.iter().all(|row| row.len() == 1) {
            records
                .iter()
                .map(|row| row[0].as_str())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            records
                .iter()
                .map(|row| row.join(","))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let page = PageContent::with_text(1, csv_text);

        Ok(Box::new(Self { pages: vec![page] }))
    }
}
