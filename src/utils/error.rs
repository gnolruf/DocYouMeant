use thiserror::Error;

#[derive(Error, Debug)]
pub enum ImageError {
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Failed to create perspective transformation")]
    ProjectionFailed,
}

#[derive(Error, Debug)]
pub enum BoxError {
    #[error("Polygon offset operation failed")]
    OffsetFailed,
}
