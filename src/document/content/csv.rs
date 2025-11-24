use csv::Reader as CsvReader;

use super::super::error::DocumentError;
use super::{DocumentContent, PageContent};

#[derive(Debug)]
pub struct CsvContent {
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
