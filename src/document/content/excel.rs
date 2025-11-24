use calamine::{Reader, Xlsx};

use super::super::error::DocumentError;
use super::{DocumentContent, PageContent};

#[derive(Debug)]
pub struct ExcelSheet {
    pub data: Vec<Vec<ExcelCell>>,
}

#[derive(Debug)]
pub struct ExcelCell {
    pub formatted: String,
}

#[derive(Debug)]
pub struct ExcelContent {
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
