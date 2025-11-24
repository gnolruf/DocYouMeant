use super::super::error::DocumentError;
use super::{DocumentContent, PageContent};

#[derive(Debug)]
pub struct TextContent {
    pages: Vec<PageContent>,
}

impl DocumentContent for TextContent {
    fn get_pages(&self) -> &Vec<PageContent> {
        &self.pages
    }

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
        &mut self.pages
    }
}

impl TextContent {
    pub fn load(bytes: &[u8]) -> Result<Box<dyn DocumentContent>, DocumentError> {
        let content =
            String::from_utf8(bytes.to_vec()).map_err(|e| DocumentError::TextLoadError {
                source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
            })?;

        let page = PageContent::with_text(1, content);

        Ok(Box::new(Self { pages: vec![page] }))
    }
}
