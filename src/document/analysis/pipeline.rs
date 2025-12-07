use image::RgbImage;
use tracing::{debug, info, instrument, warn};

use crate::document::content::{DocumentContent, DocumentType, PageContent};
use crate::document::error::DocumentError;
use crate::document::layout_box::{LayoutBox, LayoutClass};
use crate::document::region::DocumentRegionBuilder;
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

type ImageProcessingResult = (
    Vec<TextBox>,
    Vec<TextBox>,
    Vec<LayoutBox>,
    Vec<Table>,
    Orientation,
    String,
);

type PdfProcessingResult = (
    Vec<TextBox>,
    Vec<LayoutBox>,
    Vec<Table>,
    Orientation,
    String,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessMode {
    General,
    Read,
}

impl From<&str> for ProcessMode {
    fn from(s: &str) -> Self {
        match s {
            "read" => ProcessMode::Read,
            _ => ProcessMode::General,
        }
    }
}

pub struct AnalysisPipeline {
    document_type: DocumentType,
    process_mode: ProcessMode,
    language: std::cell::RefCell<Option<String>>,
}

impl AnalysisPipeline {
    pub fn new(document_type: DocumentType, process_id: String, language: Option<String>) -> Self {
        Self {
            document_type,
            process_mode: ProcessMode::from(process_id.as_str()),
            language: std::cell::RefCell::new(language),
        }
    }

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

                        let regions = DocumentRegionBuilder::build_regions(
                            page.page_number,
                            &layout_boxes,
                            &text_lines,
                        );
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

                        let regions = DocumentRegionBuilder::build_regions(
                            page.page_number,
                            &layout_boxes,
                            &text_lines,
                        );
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

                    let regions = DocumentRegionBuilder::build_regions(
                        page.page_number,
                        &layout_boxes,
                        &text_lines,
                    );
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

    #[instrument(skip(self, image, layout_boxes))]
    fn detect_tables(
        &self,
        image: &RgbImage,
        layout_boxes: &[LayoutBox],
        page_number: usize,
    ) -> Result<Vec<Table>, DocumentError> {
        debug!("Starting table detection");

        // Filter layout boxes to only include tables
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
