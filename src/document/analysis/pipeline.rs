//! Document analysis pipeline for processing various document types.
//!
//! This module provides the core analysis pipeline that orchestrates document processing,
//! including text detection, layout analysis, table extraction, and question answering.

use image::RgbImage;
use tracing::{debug, info, instrument, warn};

use crate::document::content::{DocumentContent, DocumentType, PageContent};
use crate::document::error::DocumentError;
use crate::document::layout_box::{LayoutBox, LayoutClass};
use crate::document::table::{Table, TableType};
use crate::document::text_box::{Orientation, TextBox};
use crate::inference::tasks::language_detection_task::LanguageDetectionTask;
use crate::inference::tasks::question_and_answer_task::QuestionAndAnswerTask;
use crate::inference::{
    crnn::Crnn,
    dbnet::DBNet,
    error::InferenceError,
    lcnet::{LCNet, LCNetMode, LCNetResult},
    rtdetr::{RtDetr, RtDetrMode, RtDetrResult},
};
use crate::utils::{box_utils, image_utils};

/// Result type for image document processing.
///
/// Contains the extracted information from processing an image-based document:
/// - Text lines detected in the document
/// - Individual words extracted from text recognition
/// - Layout boxes identifying document regions
/// - Tables detected and parsed from the document
/// - Document orientation
/// - Detected language code
type ImageProcessingResult = (
    Vec<TextBox>,
    Vec<TextBox>,
    Vec<LayoutBox>,
    Vec<Table>,
    Orientation,
    String,
);

/// Result type for PDF document processing with embedded text.
///
/// Contains the extracted information from processing a PDF with embedded text:
/// - Text lines matched from embedded text data
/// - Layout boxes identifying document regions
/// - Tables detected and parsed from the document
/// - Document orientation
/// - Detected language code
type PdfProcessingResult = (
    Vec<TextBox>,
    Vec<LayoutBox>,
    Vec<Table>,
    Orientation,
    String,
);

/// Processing mode that determines the depth of document analysis.
///
/// The processing mode controls which analysis steps are performed during
/// document processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessMode {
    /// Full document analysis including layout detection, table extraction,
    /// and question answering capabilities.
    General,

    /// Simplified processing focused on text extraction only.
    Read,
}

impl From<&str> for ProcessMode {
    /// Converts a string identifier into a [`ProcessMode`].
    ///
    /// # Arguments
    ///
    /// * `s` - A string slice representing the desired mode
    ///
    /// # Returns
    ///
    /// - [`ProcessMode::Read`] if `s` equals "read"
    /// - [`ProcessMode::General`] for any other value (default)
    fn from(s: &str) -> Self {
        match s {
            "read" => ProcessMode::Read,
            _ => ProcessMode::General,
        }
    }
}

/// The main document analysis pipeline.
///
/// `AnalysisPipeline` orchestrates the complete document analysis workflow.
pub struct AnalysisPipeline {
    /// The type of document being processed (PDF, image, text, etc.)
    document_type: DocumentType,
    /// The processing mode controlling analysis depth
    process_mode: ProcessMode,
    /// Cached language for consistent processing across pages.
    language: std::cell::RefCell<Option<String>>,
}

impl AnalysisPipeline {
    /// Creates a new analysis pipeline for the specified document type.
    ///
    /// # Arguments
    ///
    /// * `document_type` - The type of document to be processed
    /// * `process_id` - A string identifier for the processing mode
    /// * `language` - Optional language code to use for text recognition.
    ///   If none provided, the language will be auto-detected.
    ///
    /// # Returns
    ///
    /// A new `AnalysisPipeline` instance configured for the specified document type
    /// and processing mode.
    pub fn new(document_type: DocumentType, process_id: String, language: Option<String>) -> Self {
        Self {
            document_type,
            process_mode: ProcessMode::from(process_id.as_str()),
            language: std::cell::RefCell::new(language),
        }
    }

    /// Processes a single page of a document.
    ///
    /// # Arguments
    ///
    /// * `page` - Mutable reference to the page content to process. Results are
    ///   written directly to this structure.
    /// * `questions` - Slice of questions to answer based on page content.
    ///   Ignored in Read mode.
    ///
    /// # Returns
    ///
    /// - `Ok(())` on successful processing
    /// - `Err(DocumentError)` if any processing step fails
    #[instrument(skip(self, page, questions), fields(page_number = page.page_number))]
    pub fn process_page(
        &self,
        page: &mut PageContent,
        questions: &[String],
    ) -> Result<(), DocumentError> {
        debug!(
            "Processing page {} with mode {:?}",
            page.page_number, self.process_mode
        );
        match self.document_type {
            DocumentType::Pdf => {
                if let Some(image) = page.image.clone() {
                    if page.has_embedded_text_data() {
                        debug!("Processing PDF with embedded text");
                        let (text_lines, layout_boxes, tables, orientation, language) =
                            self.process_pdf_with_embedded_text(&image, page)?;

                        page.orientation = Some(orientation);
                        page.layout_boxes = layout_boxes.clone();
                        page.text_lines = text_lines.clone();
                        page.detected_language = Some(language);
                        page.tables = tables;

                        self.update_page_text(page);

                        let regions =
                            LayoutBox::build_regions(page.page_number, &layout_boxes, &text_lines);
                        page.regions = regions;
                    } else {
                        debug!("Processing PDF as image");
                        let (text_lines, words, layout_boxes, tables, orientation, language) =
                            self.process_image_document(&image, page.page_number)?;

                        page.orientation = Some(orientation);
                        page.layout_boxes = layout_boxes.clone();
                        page.text_lines = text_lines.clone();
                        page.detected_language = Some(language);
                        page.tables = tables;
                        if !words.is_empty() {
                            page.words = words;
                        }

                        self.update_page_text(page);

                        let regions =
                            LayoutBox::build_regions(page.page_number, &layout_boxes, &text_lines);
                        page.regions = regions;
                    }
                }

                if self.process_mode != ProcessMode::Read {
                    let qa_results = self.answer_questions_for_page(page, questions)?;
                    page.question_answers = qa_results;
                }
            }
            DocumentType::Png | DocumentType::Jpeg | DocumentType::Tiff => {
                debug!("Processing image document");
                if let Some(image) = page.image.clone() {
                    let (text_lines, words, layout_boxes, tables, orientation, language) =
                        self.process_image_document(&image, page.page_number)?;

                    page.orientation = Some(orientation);
                    page.layout_boxes = layout_boxes.clone();
                    page.text_lines = text_lines.clone();
                    page.detected_language = Some(language);
                    page.tables = tables;
                    if !words.is_empty() {
                        page.words = words;
                    }

                    self.update_page_text(page);

                    let regions =
                        LayoutBox::build_regions(page.page_number, &layout_boxes, &text_lines);
                    page.regions = regions;
                }

                if self.process_mode != ProcessMode::Read {
                    let qa_results = self.answer_questions_for_page(page, questions)?;
                    page.question_answers = qa_results;
                }
            }
            DocumentType::Text | DocumentType::Word => {
                if self.process_mode != ProcessMode::Read {
                    let qa_results = self.answer_questions_for_page(page, questions)?;
                    page.question_answers = qa_results;
                }
            }
            DocumentType::Excel | DocumentType::Csv => {
                if self.process_mode != ProcessMode::Read {
                    let qa_results = self.answer_questions_for_page(page, questions)?;
                    page.question_answers = qa_results;
                }
            }
        }

        Ok(())
    }

    /// Updates the page's combined text content from detected text lines.
    ///
    /// Concatenates all text from `text_lines` into a single string.
    ///
    /// # Arguments
    ///
    /// * `page` - Mutable reference to the page content to update
    fn update_page_text(&self, page: &mut PageContent) {
        let texts: Vec<String> = page
            .text_lines
            .iter()
            .filter_map(|text_line| text_line.text.as_ref())
            .cloned()
            .collect();

        if !texts.is_empty() {
            page.text = Some(texts.join("\n"));
        }
    }

    /// Answers a list of questions based on the page's text content.
    ///
    /// Uses the question-answering model to find answers to each question
    /// within the page's extracted text. Questions that fail to process
    /// are logged as warnings but don't cause the method to fail.
    ///
    /// # Arguments
    ///
    /// * `page` - Reference to the page content containing the text to search
    /// * `questions` - Slice of questions to answer
    ///
    /// # Returns
    ///
    /// A vector of `QuestionAndAnswerResult` for each successfully processed question.
    /// Returns an empty vector if:
    /// - No questions are provided
    /// - The page has no text content
    /// - The page text is empty/whitespace only
    ///
    /// # Errors
    ///
    /// Returns `Ok` even if individual questions fail to process (failures are logged).
    fn answer_questions_for_page(
        &self,
        page: &PageContent,
        questions: &[String],
    ) -> Result<
        Vec<crate::inference::tasks::question_and_answer_task::QuestionAndAnswerResult>,
        DocumentError,
    > {
        if questions.is_empty() {
            return Ok(Vec::new());
        }

        let text = match page.text.as_ref() {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();

        for question in questions {
            match QuestionAndAnswerTask::answer(text, question) {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    warn!(
                        "Failed to answer question '{}' for page {}: {}",
                        question, page.page_number, e
                    );
                }
            }
        }

        Ok(results)
    }

    /// Determines the orientation of a document image.
    ///
    /// Uses the document orientation classification model to infer one of four
    /// categories (0 degrees, 90 degrees, 180 degrees, 270 degrees). If an embedded orientation is already
    /// known (e.g., from PDF metadata), it is returned directly without model inference.
    ///
    /// # Arguments
    ///
    /// * `image` - The document image to analyze
    /// * `embedded_orientation` - Optional pre-determined orientation from document metadata
    ///
    /// # Returns
    ///
    /// The detected or provided [`Orientation`] of the document.
    ///
    /// # Errors
    ///
    /// Returns an error if the orientation detection model fails.
    fn get_document_orientation(
        &self,
        image: &RgbImage,
        embedded_orientation: Option<Orientation>,
    ) -> Result<Orientation, DocumentError> {
        if let Some(orientation) = embedded_orientation {
            return Ok(orientation);
        }

        let orientations = match LCNet::run(
            std::slice::from_ref(image),
            LCNetMode::DocumentOrientation,
            false,
        )
        .map_err(|source| DocumentError::ModelProcessingError { source })?
        {
            LCNetResult::Orientations(orientations) => orientations,
            LCNetResult::TableTypes(_) => {
                return Err(DocumentError::ModelProcessingError {
                    source: InferenceError::ProcessingError {
                        message: "Unexpected table types result for document orientation"
                            .to_string(),
                    },
                });
            }
        };

        Ok(orientations
            .first()
            .copied()
            .unwrap_or(Orientation::Oriented0))
    }

    /// Detects text line regions in a document image.
    ///
    /// Identifies rectangular regions containing text and orders them in
    /// reading order using graph-based analysis.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to analyze for text regions
    ///
    /// # Returns
    ///
    /// A vector of [`TextBox`] instances representing detected text regions,
    /// ordered in natural reading order.
    ///
    /// # Errors
    ///
    /// Returns an error if text detection model inference fails.
    fn detect_text_lines(&self, image: &RgbImage) -> Result<Vec<TextBox>, DocumentError> {
        debug!("Starting text detection");
        let text_lines =
            DBNet::run(image).map_err(|source| DocumentError::ModelProcessingError { source })?;
        debug!("Detected {} raw text lines", text_lines.len());

        let bounds_list: Vec<_> = text_lines.iter().map(|t| t.bounds).collect();
        let ordered_indices = box_utils::graph_based_reading_order(&bounds_list);
        let ordered_text_lines: Vec<TextBox> = ordered_indices
            .iter()
            .filter_map(|&idx| {
                let box_idx = idx.saturating_sub(1);
                text_lines.get(box_idx).cloned()
            })
            .collect();

        Ok(ordered_text_lines)
    }

    /// Detects the document layout structure.
    ///
    /// # Arguments
    ///
    /// * `image` - The document image to analyze
    ///
    /// # Returns
    ///
    /// A vector of [`LayoutBox`] instances representing detected layout regions.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Layout detection model inference fails
    /// - The model returns an unexpected result type
    fn detect_layout(&self, image: &RgbImage) -> Result<Vec<LayoutBox>, DocumentError> {
        debug!("Starting layout detection");
        let result = RtDetr::run(image, RtDetrMode::Layout)
            .map_err(|source| DocumentError::ModelProcessingError { source })?;
        let layout = match result {
            RtDetrResult::LayoutBoxes(boxes) => boxes,
            _ => {
                return Err(DocumentError::ProcessingError {
                    message: "Unexpected result type from layout detection".to_string(),
                })
            }
        };
        debug!("Detected {} layout elements", layout.len());
        Ok(layout)
    }

    /// Detects and parses tables from the document image.
    ///
    /// # Arguments
    ///
    /// * `image` - The full document image
    /// * `layout_boxes` - Layout detection results to filter for table regions
    /// * `page_number` - The page number for table metadata
    ///
    /// # Returns
    ///
    /// A vector of [`Table`] instances representing detected tables.
    /// Returns an empty vector if no table regions are found in the layout.
    ///
    /// # Errors
    ///
    /// Returns an error if image cropping fails. Individual table cell detection
    /// failures are logged as warnings but don't cause the method to fail.
    #[instrument(skip(self, image, layout_boxes))]
    fn detect_tables(
        &self,
        image: &RgbImage,
        layout_boxes: &[LayoutBox],
        page_number: usize,
    ) -> Result<Vec<Table>, DocumentError> {
        debug!("Starting table detection");

        let table_boxes: Vec<&LayoutBox> = layout_boxes
            .iter()
            .filter(|lb| lb.class == LayoutClass::Table)
            .collect();

        if table_boxes.is_empty() {
            debug!("No table regions found in layout");
            return Ok(Vec::new());
        }

        debug!("Found {} table regions in layout", table_boxes.len());

        let mut table_images = Vec::with_capacity(table_boxes.len());
        for table_box in &table_boxes {
            let box_points: Vec<(i32, i32)> = table_box.bounds.iter().map(|p| (p.x, p.y)).collect();

            let (table_image, _) =
                image_utils::get_rotate_crop_image(image, &box_points).map_err(|e| {
                    DocumentError::ProcessingError {
                        message: format!("Failed to crop table image: {}", e),
                    }
                })?;

            table_images.push(table_image);
        }

        let table_types = self.classify_table_types(&table_images)?;

        let mut tables = Vec::with_capacity(table_boxes.len());
        for (i, ((table_box, table_image), table_type)) in table_boxes
            .iter()
            .zip(table_images.iter())
            .zip(table_types.iter())
            .enumerate()
        {
            debug!("Processing table {} of type {:?}", i + 1, table_type);

            match self.detect_table_cells(table_image, table_box, *table_type, page_number) {
                Ok(table) => {
                    debug!(
                        "Detected table with {} rows, {} columns, {} cells",
                        table.row_count,
                        table.column_count,
                        table.cells.len()
                    );
                    tables.push(table);
                }
                Err(e) => {
                    warn!("Failed to detect cells for table {}: {}", i + 1, e);
                    tables.push(Table::from_cells(
                        Vec::new(),
                        table_box.bounds,
                        *table_type,
                        page_number,
                        table_box.confidence,
                    ));
                }
            }
        }

        debug!("Completed table detection, found {} tables", tables.len());
        Ok(tables)
    }

    /// Classifies tables as wired (bordered) or wireless (borderless).
    ///
    /// # Arguments
    ///
    /// * `table_images` - Cropped images of table regions
    ///
    /// # Returns
    ///
    /// A vector of [`TableType`] values corresponding to each input image.
    /// Returns an empty vector if no images are provided.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Table classification model inference fails
    /// - The model returns an unexpected result type
    fn classify_table_types(
        &self,
        table_images: &[RgbImage],
    ) -> Result<Vec<TableType>, DocumentError> {
        if table_images.is_empty() {
            return Ok(Vec::new());
        }

        let result = LCNet::run(table_images, LCNetMode::TableType, false)
            .map_err(|source| DocumentError::ModelProcessingError { source })?;

        match result {
            LCNetResult::TableTypes(types) => Ok(types),
            LCNetResult::Orientations(_) => Err(DocumentError::ProcessingError {
                message: "Unexpected orientations result for table classification".to_string(),
            }),
        }
    }

    /// Detects individual cells within a table image.
    ///
    /// # Arguments
    ///
    /// * `table_image` - Cropped image of the table region
    /// * `table_box` - The layout box containing the table bounds and confidence
    /// * `table_type` - Whether the table is wired or wireless
    /// * `page_number` - The page number for table metadata
    ///
    /// # Returns
    ///
    /// A [`Table`] instance with detected cells and computed row/column structure.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Cell detection model inference fails
    /// - The model returns an unexpected result type
    fn detect_table_cells(
        &self,
        table_image: &RgbImage,
        table_box: &LayoutBox,
        table_type: TableType,
        page_number: usize,
    ) -> Result<Table, DocumentError> {
        let detection_mode = match table_type {
            TableType::Wired => RtDetrMode::WiredTableCell,
            TableType::Wireless => RtDetrMode::WirelessTableCell,
        };

        let result = RtDetr::run(table_image, detection_mode)
            .map_err(|source| DocumentError::ModelProcessingError { source })?;

        let detected_cells = match result {
            RtDetrResult::TableCells(cells) => cells,
            RtDetrResult::LayoutBoxes(_) => {
                return Err(DocumentError::ProcessingError {
                    message: "Unexpected layout boxes result for table cell detection".to_string(),
                });
            }
        };

        Ok(Table::from_cells(
            detected_cells,
            table_box.bounds,
            table_type,
            page_number,
            table_box.confidence,
        ))
    }

    /// Processes a PDF page that contains embedded text data.
    ///
    /// This method handles PDFs where text can be extracted directly from the
    /// document structure rather than requiring OCR.
    ///
    /// # Arguments
    ///
    /// * `image` - The rendered page image for visual analysis
    /// * `page` - The page content containing embedded text data in `words`
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - Text lines with matched embedded text
    /// - Layout boxes (empty in Read mode)
    /// - Detected tables with cell content (empty in Read mode)
    /// - Document orientation
    /// - Detected language code
    ///
    /// # Errors
    ///
    /// Returns an error if any model inference step fails.
    #[instrument(skip(self, image, page))]
    fn process_pdf_with_embedded_text(
        &self,
        image: &RgbImage,
        page: &PageContent,
    ) -> Result<PdfProcessingResult, DocumentError> {
        let document_orientation = self.get_document_orientation(image, page.orientation)?;
        debug!("Document orientation: {:?}", document_orientation);

        let oriented_image = if document_orientation != Orientation::Oriented0 {
            image_utils::rotate_image(image, document_orientation)
        } else {
            image.clone()
        };

        let mut text_lines = self.detect_text_lines(&oriented_image)?;
        debug!("Found {} text lines", text_lines.len());

        self.match_embedded_text_to_lines(&mut text_lines, &page.words);
        debug!("Matched embedded text to lines");

        let cached_language = self.language.borrow().clone();
        let language = match cached_language {
            Some(lang) => {
                debug!("Using provided/cached language: {}", lang);
                lang
            }
            None => {
                debug!("No language provided, running language detection on embedded text");
                let texts: Vec<String> =
                    text_lines.iter().filter_map(|tl| tl.text.clone()).collect();
                let detection_result = LanguageDetectionTask::detect_from_text(&texts)
                    .map_err(|source| DocumentError::ModelProcessingError { source })?;
                debug!(
                    "Detected language from embedded text: {}",
                    detection_result.language
                );
                *self.language.borrow_mut() = Some(detection_result.language.clone());
                detection_result.language
            }
        };

        let (layout_boxes, mut tables) = if self.process_mode == ProcessMode::Read {
            (Vec::new(), Vec::new())
        } else {
            let layout = self.detect_layout(&oriented_image)?;
            let tables = self.detect_tables(&oriented_image, &layout, page.page_number)?;
            (layout, tables)
        };

        for table in &mut tables {
            table.match_words_to_cells(&page.words, 0.5);
        }

        Ok((
            text_lines,
            layout_boxes,
            tables,
            document_orientation,
            language,
        ))
    }

    /// Matches embedded text words to detected text line regions.
    ///
    /// For each detected text line, finds embedded words that spatially overlap
    /// with the line region (>75% overlap) and concatenates them to form the
    /// line's text content.
    ///
    /// # Arguments
    ///
    /// * `text_lines` - Mutable slice of text line regions to populate with text
    /// * `embedded_words` - Word-level text boxes from embedded PDF data
    ///
    /// # Side Effects
    ///
    /// Updates each text line in place with:
    /// - `text`: Concatenated words separated by spaces
    /// - `text_score`: Average confidence of matched words
    /// - `span`: Document span for the text content
    /// - `angle`: Set to `Oriented0` if not already set
    fn match_embedded_text_to_lines(&self, text_lines: &mut [TextBox], embedded_words: &[TextBox]) {
        let mut current_offset = 0;
        for text_line in text_lines.iter_mut() {
            let mut matched_texts = Vec::new();
            let mut total_confidence = 0.0;
            let mut text_count = 0;

            for embedded_word in embedded_words {
                if box_utils::calculate_overlap(
                    embedded_word.bounds.as_slice(),
                    text_line.bounds.as_slice(),
                ) > 0.75
                {
                    if let Some(ref text) = embedded_word.text {
                        matched_texts.push(text.clone());
                        total_confidence += embedded_word.text_score;
                        text_count += 1;
                    }
                }
            }

            if !matched_texts.is_empty() {
                let text = matched_texts.join(" ");
                let length = text.len();
                text_line.text = Some(text);
                text_line.text_score = total_confidence / text_count as f32;
                text_line.span = Some(crate::document::text_box::DocumentSpan::new(
                    current_offset,
                    length,
                ));
                current_offset += length + 1;

                if text_line.angle.is_none() {
                    text_line.angle = Some(Orientation::Oriented0);
                }
            }
        }
    }

    /// Processes an image-based document using full OCR pipeline.
    ///
    /// This method handles documents where text must be extracted through OCR,
    /// including scanned PDFs and image files.
    ///
    /// # Arguments
    ///
    /// * `image` - The document image to process
    /// * `page_number` - The page number for metadata
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - Text lines with recognized text
    /// - Individual words from text recognition
    /// - Layout boxes (empty in Read mode)
    /// - Detected tables with cell content (empty in Read mode)
    /// - Document orientation
    /// - Detected language code
    ///
    /// # Errors
    ///
    /// Returns an error if any processing step fails, including:
    /// - Model inference failures
    /// - Image cropping/rotation errors
    /// - Text recognition failures
    #[instrument(skip(self, image))]
    fn process_image_document(
        &self,
        image: &RgbImage,
        page_number: usize,
    ) -> Result<ImageProcessingResult, DocumentError> {
        let document_orientation = self.get_document_orientation(image, None)?;
        debug!("Document orientation: {:?}", document_orientation);

        let oriented_image = if document_orientation != Orientation::Oriented0 {
            image_utils::rotate_image(image, document_orientation)
        } else {
            image.clone()
        };

        let mut text_lines = self.detect_text_lines(&oriented_image)?;
        debug!("Found {} text lines", text_lines.len());

        let image_parts =
            image_utils::get_image_parts(&oriented_image, &text_lines).map_err(|e| {
                DocumentError::ModelProcessingError {
                    source: InferenceError::ProcessingError {
                        message: format!("Failed to crop image parts: {e}"),
                    },
                }
            })?;

        let angles = match LCNet::run(&image_parts, LCNetMode::TextOrientation, true)
            .map_err(|source| DocumentError::ModelProcessingError { source })?
        {
            LCNetResult::Orientations(orientations) => orientations,
            LCNetResult::TableTypes(_) => {
                return Err(DocumentError::ModelProcessingError {
                    source: InferenceError::ProcessingError {
                        message: "Unexpected table types result for text orientation".to_string(),
                    },
                });
            }
        };

        for (i, angle) in angles.iter().enumerate() {
            if i < text_lines.len() {
                text_lines[i].angle = Some(*angle);
            }
        }

        let rotated_parts = image_utils::rotate_images_by_angle(&image_parts, &text_lines)
            .map_err(|e| DocumentError::ModelProcessingError {
                source: InferenceError::ProcessingError {
                    message: format!("Failed to rotate image parts: {e}"),
                },
            })?;

        let cached_language = self.language.borrow().clone();
        let language = match cached_language {
            Some(lang) => {
                debug!("Using provided/cached language: {}", lang);
                lang
            }
            None => {
                debug!("No language provided, running language detection");
                let detection_result =
                    LanguageDetectionTask::detect(&text_lines, &rotated_parts)
                        .map_err(|source| DocumentError::ModelProcessingError { source })?;
                debug!(
                    "Detected language: {} (confidence: {:.2})",
                    detection_result.language, detection_result.confidence
                );
                *self.language.borrow_mut() = Some(detection_result.language.clone());
                detection_result.language
            }
        };

        let mut text_recognizer = Crnn::new(&language)
            .map_err(|source| DocumentError::ModelProcessingError { source })?;

        let words = text_recognizer
            .get_texts(&rotated_parts, &mut text_lines)
            .map_err(|source| DocumentError::ModelProcessingError { source })?;
        debug!("Recognized text for {} lines", words.len());

        let (layout_boxes, mut tables) = if self.process_mode == ProcessMode::Read {
            (Vec::new(), Vec::new())
        } else {
            let layout = self.detect_layout(&oriented_image)?;
            let tables = self.detect_tables(&oriented_image, &layout, page_number)?;
            (layout, tables)
        };

        for table in &mut tables {
            table.match_words_to_cells(&words, 0.5);
        }

        Ok((
            text_lines,
            words,
            layout_boxes,
            tables,
            document_orientation,
            language,
        ))
    }

    /// Analyzes an entire document, processing all pages.
    ///
    /// This is the main entry point for document analysis. It iterates through
    /// all pages in the document, processes each one, and collects question
    /// answering results.
    ///
    /// # Arguments
    ///
    /// * `content` - Mutable reference to the document content to analyze.
    ///   Results are written directly to the page structures.
    /// * `questions` - Slice of questions to answer based on document content.
    ///   The same questions are asked for each page.
    ///
    /// # Returns
    ///
    /// A vector containing all `QuestionAndAnswerResult` instances from all pages.
    /// Results are returned in page order.
    ///
    /// # Errors
    ///
    /// Returns an error if processing any page fails. Pages are processed
    /// sequentially, so an error on one page stops processing of subsequent pages.
    #[instrument(skip(self, content, questions))]
    pub fn analyze(
        &self,
        content: &mut dyn DocumentContent,
        questions: &[String],
    ) -> Result<
        Vec<crate::inference::tasks::question_and_answer_task::QuestionAndAnswerResult>,
        DocumentError,
    > {
        info!("Starting document analysis");
        let mut all_qa_results = Vec::new();

        let pages = content.get_pages_mut();
        for page in pages {
            self.process_page(page, questions)?;

            all_qa_results.extend(page.question_answers.clone());
        }

        Ok(all_qa_results)
    }
}
