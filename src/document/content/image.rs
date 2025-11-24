use super::super::error::DocumentError;
use super::{DocumentContent, DocumentType, PageContent};

#[derive(Debug)]
pub struct ImageContent {
    #[allow(dead_code)]
    doc_type: DocumentType,
    pages: Vec<PageContent>,
}

impl DocumentContent for ImageContent {
    fn get_pages(&self) -> &Vec<PageContent> {
        &self.pages
    }

    fn get_pages_mut(&mut self) -> &mut Vec<PageContent> {
        &mut self.pages
    }
}

impl ImageContent {
    pub fn load(
        bytes: &[u8],
        doc_type: DocumentType,
    ) -> Result<Box<dyn DocumentContent>, DocumentError> {
        let img = image::load_from_memory(bytes)
            .map_err(|source| DocumentError::ImageLoadError { source })?
            .to_rgb8();

        let page = PageContent::with_image(1, img);

        Ok(Box::new(Self {
            doc_type,
            pages: vec![page],
        }))
    }
}
