use thiserror::Error;

#[derive(Error, Debug)]
pub enum DocumentError {
    #[error("Unsupported file type: {extension}")]
    UnsupportedFileType { extension: String },

    #[error("Failed to load image content")]
    ImageLoadError {
        #[source]
        source: image::ImageError,
    },

    #[error("Failed to load text content")]
    TextLoadError {
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to load CSV content")]
    CsvLoadError {
        #[source]
        source: csv::Error,
    },

    #[error("Failed to load Excel content")]
    ExcelLoadError {
        #[source]
        source: calamine::XlsxError,
    },

    #[error("Failed to load PDF content")]
    PdfLoadError {
        #[source]
        source: pdfium_render::prelude::PdfiumError,
    },

    #[error("Failed to load Word document content")]
    WordLoadError { message: String },

    #[error("Document content not loaded")]
    ContentNotLoaded,

    #[error("Model processing failed")]
    ModelProcessingError {
        #[from]
        source: crate::inference::InferenceError,
    },

    #[error("Processing error: {message}")]
    ProcessingError { message: String },
}

impl From<pdfium_render::prelude::PdfiumError> for DocumentError {
    fn from(source: pdfium_render::prelude::PdfiumError) -> Self {
        DocumentError::PdfLoadError { source }
    }
}
