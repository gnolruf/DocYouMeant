use docx_rs::*;

use super::super::error::DocumentError;
use super::{DocumentContent, PageContent};

#[derive(Debug)]
pub struct WordContent {
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

impl WordContent {
    pub fn load(bytes: &[u8]) -> Result<Box<dyn DocumentContent>, DocumentError> {
        let docx = docx_rs::read_docx(bytes).map_err(|e| DocumentError::WordLoadError {
            message: e.to_string(),
        })?;

        let text_content = docx
            .document
            .children
            .iter()
            .fold(String::new(), |acc, c| match c {
                DocumentChild::Paragraph(p) => {
                    let mut new_acc = acc;
                    new_acc.push_str(&p.raw_text());
                    new_acc.push('\n');
                    new_acc
                }
                _ => acc,
            });

        let page = PageContent::with_text(1, text_content);

        Ok(Box::new(Self { pages: vec![page] }))
    }
}
