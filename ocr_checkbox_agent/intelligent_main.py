"""Intelligent OCR Checkbox Agent with advanced form understanding capabilities."""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger
import numpy as np

from .pdf_processor import PDFProcessor
from .ocr_engine import OCREngine
from .checkbox_detector import CheckboxDetector
from .data_structurer import DataStructurer, FormResponse, ExtractionResult
from .config import Config, FormTemplate, get_config, set_config

# Import intelligent components
from .nlp_engine import NLPEngine
from .form_analyzer import FormAnalyzer
from .response_detector import IntelligentResponseDetector
from .context_engine import ContextEngine, FormContext


class IntelligentOCRAgent:
    """Intelligent OCR Checkbox Agent with advanced form understanding."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the Intelligent OCR Agent."""
        self.config = config or get_config()
        set_config(self.config)
        
        # Initialize core components
        self.pdf_processor = PDFProcessor(self.config)
        self.ocr_engine = OCREngine(self.config)
        
        # Initialize intelligent components
        try:
            self.nlp_engine = NLPEngine()
            self.form_analyzer = FormAnalyzer(self.nlp_engine)
            self.response_detector = IntelligentResponseDetector(self.config)
            self.context_engine = ContextEngine(self.config)
            
            # Fallback components for compatibility
            self.checkbox_detector = CheckboxDetector(self.config)
            self.data_structurer = DataStructurer(self.config)
            
            self.intelligent_mode = True
            logger.info("Intelligent OCR Agent initialized with full NLP capabilities")
            
        except Exception as e:
            logger.warning(f"Failed to initialize intelligent components: {e}")
            logger.info("Falling back to basic mode")
            
            # Fallback to basic components
            self.checkbox_detector = CheckboxDetector(self.config)
            self.data_structurer = DataStructurer(self.config)
            self.intelligent_mode = False
        
        # Form template
        self.form_template: Optional[FormTemplate] = None
    
    def load_template(self, template_path: Path) -> None:
        """Load a form template."""
        self.form_template = FormTemplate.load(template_path)
        logger.info(f"Loaded form template: {self.form_template.name}")
    
    def process_pdf(self, pdf_path: Path, 
                   use_template: bool = False,
                   save_debug: bool = False,
                   detect_multiple_surveys: bool = True,
                   max_pages: Optional[int] = None,
                   max_surveys: Optional[int] = None,
                   enhanced_ocr: bool = True) -> ExtractionResult:
        """
        Process a single PDF file with intelligent analysis.
        
        Args:
            pdf_path: Path to PDF file
            use_template: Whether to use loaded template
            save_debug: Whether to save debug images
            detect_multiple_surveys: Whether to detect multiple surveys in one PDF
            max_pages: Maximum number of pages to process (for testing)
            max_surveys: Maximum number of surveys to process (for testing)
            enhanced_ocr: Whether to use enhanced OCR preprocessing
            
        Returns:
            Extraction result with intelligent analysis
        """
        start_time = time.time()
        errors = []
        warnings = []
        responses = []
        
        try:
            logger.info(f"Processing PDF with intelligent analysis: {pdf_path}")
            
            # Convert PDF to images
            pages = self.pdf_processor.process_pdf(pdf_path)
            
            # Limit pages if specified for testing
            if max_pages:
                pages = pages[:max_pages]
                logger.info(f"Limited processing to first {max_pages} pages for testing")
            
            if self.intelligent_mode:
                # Use intelligent processing
                responses = self._process_pdf_intelligent(
                    pages, use_template, save_debug, detect_multiple_surveys, 
                    max_surveys, enhanced_ocr
                )
            else:
                # Fallback to basic processing
                for page_idx, (image, metadata) in enumerate(pages):
                    try:
                        page_response = self._process_page_basic(
                            image, metadata, page_idx + 1, 
                            use_template, save_debug
                        )
                        if page_response:
                            responses.append(page_response)
                    except Exception as e:
                        error_msg = f"Error processing page {page_idx + 1}: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
            
            processing_time = time.time() - start_time
            
            result = ExtractionResult(
                document_path=str(pdf_path),
                total_pages=len(pages),
                responses=responses,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings
            )
            
            logger.info(f"Completed processing {pdf_path}: {len(responses)} responses in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Failed to process PDF {pdf_path}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return ExtractionResult(
                document_path=str(pdf_path),
                total_pages=0,
                responses=[],
                processing_time=time.time() - start_time,
                errors=errors,
                warnings=warnings
            )
    
    def _process_pdf_intelligent(self, pages: List[Tuple], 
                               use_template: bool,
                               save_debug: bool,
                               detect_multiple_surveys: bool,
                               max_surveys: Optional[int] = None,
                               enhanced_ocr: bool = True) -> List[FormResponse]:
        """Process PDF using intelligent analysis."""
        
        all_responses = []
        total_surveys_processed = 0
        
        for page_idx, (image, metadata) in enumerate(pages):
            try:
                logger.info(f"Processing page {page_idx + 1} with intelligent analysis")
                
                # Step 1: Enhanced OCR text extraction
                if enhanced_ocr:
                    text_result = self._enhanced_ocr_extraction(image)
                else:
                    text_result = self.ocr_engine.extract_text(image)
                
                text_blocks = text_result.get('text_blocks', [])
                
                # Step 2: Detect multiple surveys if requested
                if detect_multiple_surveys:
                    survey_boundaries = self.form_analyzer.detect_multiple_surveys(image, text_blocks)
                    if len(survey_boundaries) > 1:
                        logger.info(f"Detected {len(survey_boundaries)} surveys on page {page_idx + 1}")
                        
                        # Process each survey separately
                        for survey_idx, boundary in enumerate(survey_boundaries):
                            # Check if we've reached the survey limit
                            if max_surveys and total_surveys_processed >= max_surveys:
                                logger.info(f"Reached maximum surveys limit ({max_surveys}), stopping processing")
                                return all_responses
                            
                            survey_responses = self._process_survey_section(
                                image, text_blocks, boundary, metadata, 
                                page_idx + 1, survey_idx + 1, save_debug, enhanced_ocr
                            )
                            all_responses.extend(survey_responses)
                            total_surveys_processed += 1
                    else:
                        # Single survey
                        if max_surveys and total_surveys_processed >= max_surveys:
                            logger.info(f"Reached maximum surveys limit ({max_surveys}), stopping processing")
                            return all_responses
                        
                        survey_responses = self._process_single_survey(
                            image, text_blocks, metadata, page_idx + 1, save_debug, enhanced_ocr
                        )
                        all_responses.extend(survey_responses)
                        total_surveys_processed += 1
                else:
                    # Process as single survey
                    if max_surveys and total_surveys_processed >= max_surveys:
                        logger.info(f"Reached maximum surveys limit ({max_surveys}), stopping processing")
                        return all_responses
                    
                    survey_responses = self._process_single_survey(
                        image, text_blocks, metadata, page_idx + 1, save_debug, enhanced_ocr
                    )
                    all_responses.extend(survey_responses)
                    total_surveys_processed += 1
                
            except Exception as e:
                logger.error(f"Error in intelligent processing of page {page_idx + 1}: {e}")
                if save_debug:
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        return all_responses
    
    def _process_single_survey(self, image, text_blocks, metadata, 
                             page_number, save_debug, enhanced_ocr=True) -> List[FormResponse]:
        """Process a single survey using intelligent analysis."""
        
        try:
            # Step 1: Analyze form structure
            form_layout = self.form_analyzer.analyze_form_structure(image, text_blocks)
            
            # Step 2: Detect all responses using intelligent detector
            all_responses = []
            for response_area in form_layout.response_areas:
                for question in self._get_questions_from_sections(form_layout.sections):
                    question_responses = self.response_detector.detect_responses(
                        image, question, response_area, text_blocks
                    )
                    all_responses.extend(question_responses)
            
            # Step 3: Use context engine to validate and structure
            form_context = self.context_engine.analyze_form_context(
                form_layout, all_responses, metadata
            )
            
            # Step 4: Convert to FormResponse objects
            form_responses = self._convert_context_to_responses(
                form_context, metadata, page_number
            )
            
            # Step 5: Save debug information if requested
            if save_debug:
                self._save_intelligent_debug_info(
                    image, form_layout, form_context, page_number
                )
            
            logger.info(f"Intelligent analysis complete: {len(form_responses)} structured responses")
            return form_responses
            
        except Exception as e:
            logger.error(f"Error in single survey processing: {e}")
            return []
    
    def _process_survey_section(self, image, text_blocks, boundary, metadata,
                              page_number, survey_index, save_debug, enhanced_ocr=True) -> List[FormResponse]:
        """Process a specific survey section."""
        
        try:
            # Extract the survey section from the image
            start_y = boundary.start_y
            end_y = boundary.end_y if boundary.end_y != float('inf') else image.shape[0]
            
            section_image = image[start_y:end_y, :]
            
            # Filter text blocks to this section
            section_text_blocks = []
            for block in text_blocks:
                bbox = block.get('bbox', {})
                block_y = bbox.get('y', 0)
                if start_y <= block_y <= end_y:
                    # Adjust coordinates relative to section
                    adjusted_bbox = bbox.copy()
                    adjusted_bbox['y'] = block_y - start_y
                    block_copy = block.copy()
                    block_copy['bbox'] = adjusted_bbox
                    section_text_blocks.append(block_copy)
            
            # Process the section as a single survey
            section_responses = self._process_single_survey(
                section_image, section_text_blocks, metadata, page_number, save_debug, enhanced_ocr
            )
            
            # Add survey identification to responses
            for response in section_responses:
                response.metadata['survey_title'] = boundary.title
                response.metadata['survey_index'] = survey_index
                response.metadata['survey_confidence'] = boundary.confidence
            
            return section_responses
            
        except Exception as e:
            logger.error(f"Error processing survey section {survey_index}: {e}")
            return []
    
    def _get_questions_from_sections(self, sections) -> List:
        """Extract all questions from form sections."""
        all_questions = []
        for section in sections:
            all_questions.extend(section.questions)
        return all_questions
    
    def _convert_context_to_responses(self, form_context: FormContext, 
                                    metadata: Dict[str, Any],
                                    page_number: int) -> List[FormResponse]:
        """Convert FormContext to FormResponse objects."""
        
        form_responses = []
        
        # Create a response for each section or as a single response
        if len(form_context.sections) > 1:
            # Multiple sections - create response per section
            for section in form_context.sections:
                section_pairs = [pair for pair in form_context.question_response_pairs 
                               if any(q.text == pair.question.text for q in section.questions)]
                
                if section_pairs:
                    fields = {}
                    section_confidence = 0.0
                    
                    for pair in section_pairs:
                        field_name = self._sanitize_field_name(pair.question.text)
                        field_value = self._extract_response_value(pair.responses)
                        
                        fields[field_name] = {
                            'value': field_value,
                            'question_type': pair.question.question_type.value,
                            'response_type': pair.question.response_type.value,
                            'confidence': pair.confidence,
                            'validation_issues': [issue.message for issue in pair.validation_issues]
                        }
                        
                        section_confidence += pair.confidence
                    
                    if section_pairs:
                        section_confidence /= len(section_pairs)
                    
                    response = FormResponse(
                        document_id=metadata.get('filename', 'unknown'),
                        page_number=page_number,
                        timestamp=time.time(),
                        fields=fields,
                        metadata={
                            'section_title': section.title,
                            'section_type': section.section_type,
                            'intelligent_analysis': True,
                            'form_layout_type': form_context.metadata.get('layout_type', 'unknown'),
                            'completion_rate': form_context.completion_rate,
                            'quality_score': form_context.validation_summary.get('quality_score', 0.0)
                        },
                        confidence_score=section_confidence
                    )
                    
                    form_responses.append(response)
        else:
            # Single section or no sections - create single response
            fields = {}
            overall_confidence = form_context.overall_confidence
            
            for pair in form_context.question_response_pairs:
                field_name = self._sanitize_field_name(pair.question.text)
                field_value = self._extract_response_value(pair.responses)
                
                fields[field_name] = {
                    'value': field_value,
                    'question_type': pair.question.question_type.value,
                    'response_type': pair.question.response_type.value,
                    'confidence': pair.confidence,
                    'validation_issues': [issue.message for issue in pair.validation_issues]
                }
            
            response = FormResponse(
                document_id=metadata.get('filename', 'unknown'),
                page_number=page_number,
                timestamp=time.time(),
                fields=fields,
                metadata={
                    'intelligent_analysis': True,
                    'completion_rate': form_context.completion_rate,
                    'quality_score': form_context.validation_summary.get('quality_score', 0.0),
                    'total_validation_issues': sum(form_context.validation_summary.get('issue_counts', {}).values())
                },
                confidence_score=overall_confidence
            )
            
            form_responses.append(response)
        
        return form_responses
    
    def _extract_response_value(self, responses):
        """Extract the primary value from response list."""
        if not responses:
            return None
        
        # For single responses, return the value
        if len(responses) == 1:
            return responses[0].value
        
        # For multiple responses, return list of values
        return [r.value for r in responses if r.value is not None]
    
    def _sanitize_field_name(self, question_text: str) -> str:
        """Sanitize question text to create valid field name."""
        import re
        # Remove special characters and normalize
        sanitized = re.sub(r'[^\w\s-]', '', question_text)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        sanitized = sanitized.strip('_').lower()
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'unknown_field'
        
        # Limit length
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        return sanitized
    
    def _save_intelligent_debug_info(self, image, form_layout, form_context, page_number):
        """Save debug information for intelligent analysis."""
        
        try:
            debug_dir = self.config.temp_dir / "intelligent_debug"
            debug_dir.mkdir(exist_ok=True)
            
            import json
            import cv2
            
            # Save form layout info
            layout_info = {
                'layout_type': form_layout.layout_type,
                'total_questions': form_layout.total_questions,
                'sections': [
                    {
                        'title': section.title,
                        'type': section.section_type,
                        'question_count': len(section.questions)
                    } for section in form_layout.sections
                ],
                'response_areas': len(form_layout.response_areas)
            }
            
            with open(debug_dir / f"page_{page_number}_layout.json", 'w') as f:
                json.dump(layout_info, f, indent=2)
            
            # Save context analysis
            context_info = {
                'overall_confidence': form_context.overall_confidence,
                'completion_rate': form_context.completion_rate,
                'validation_summary': form_context.validation_summary
            }
            
            with open(debug_dir / f"page_{page_number}_context.json", 'w') as f:
                json.dump(context_info, f, indent=2)
            
            # Save original image
            cv2.imwrite(str(debug_dir / f"page_{page_number}_original.png"), 
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            logger.debug(f"Saved intelligent debug info for page {page_number}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug info: {e}")
    
    def _process_page_basic(self, image, metadata: Dict[str, Any], 
                          page_number: int, use_template: bool = False,
                          save_debug: bool = False):
        """Fallback to basic processing when intelligent mode is not available."""
        
        # Extract text using OCR
        text_result = self.ocr_engine.extract_text(image)
        
        # Detect checkboxes
        checkboxes = self.checkbox_detector.detect_checkboxes(
            image, text_result['text_blocks']
        )
        
        if not checkboxes:
            logger.warning(f"No checkboxes found on page {page_number}")
            return None
        
        # Save debug images if requested
        if save_debug:
            debug_dir = self.config.temp_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            
            import cv2
            cv2.imwrite(str(debug_dir / f"page_{page_number}_original.png"), image)
            
            vis_image = self.checkbox_detector.visualize_detections(image, checkboxes)
            cv2.imwrite(str(debug_dir / f"page_{page_number}_detected.png"), vis_image)
        
        # Structure the data
        response = self.data_structurer.structure_checkbox_data(
            checkboxes, metadata, page_number
        )
        
        return response
    
    def _enhanced_ocr_extraction(self, image):
        """Enhanced OCR extraction with better preprocessing for text and box identification."""
        
        try:
            import cv2
            import numpy as np
            
            # Create a copy for processing
            processed_image = image.copy()
            
            # Step 1: Convert to grayscale if needed
            if len(processed_image.shape) == 3:
                gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = processed_image
            
            # Step 2: Enhanced preprocessing for better OCR
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Contrast enhancement using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Deskewing (basic rotation correction)
            coords = np.column_stack(np.where(enhanced > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                
                if abs(angle) > 0.5:  # Only correct if significant skew
                    (h, w) = enhanced.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    enhanced = cv2.warpAffine(enhanced, M, (w, h), 
                                            flags=cv2.INTER_CUBIC, 
                                            borderMode=cv2.BORDER_REPLICATE)
            
            # Step 3: Multiple OCR strategies for better accuracy
            
            # Strategy 1: Standard OCR with enhanced image
            result1 = self.ocr_engine.extract_text(enhanced, preprocess=False)
            
            # Strategy 2: Binary threshold for crisp text
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result2 = self.ocr_engine.extract_text(binary, preprocess=False)
            
            # Strategy 3: Adaptive threshold for varied lighting
            adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            result3 = self.ocr_engine.extract_text(adaptive, preprocess=False)
            
            # Step 4: Merge results and choose best
            best_result = self._select_best_ocr_result([result1, result2, result3])
            
            # Step 5: Enhance text block detection
            enhanced_text_blocks = self._enhance_text_block_detection(enhanced, best_result.get('text_blocks', []))
            best_result['text_blocks'] = enhanced_text_blocks
            
            # Step 6: Detect and separate checkbox regions for better identification
            checkbox_regions = self._detect_enhanced_checkbox_regions(enhanced)
            best_result['checkbox_regions'] = checkbox_regions
            
            logger.debug(f"Enhanced OCR extracted {len(enhanced_text_blocks)} text blocks and {len(checkbox_regions)} checkbox regions")
            
            return best_result
            
        except Exception as e:
            logger.warning(f"Enhanced OCR failed, falling back to standard: {e}")
            return self.ocr_engine.extract_text(image)
    
    def _select_best_ocr_result(self, results):
        """Select the best OCR result based on confidence and text amount."""
        
        best_result = results[0]
        best_score = 0
        
        for result in results:
            # Calculate score based on confidence and text amount
            confidence = result.get('confidence', 0)
            text_blocks = result.get('text_blocks', [])
            text_amount = sum(len(block.get('text', '')) for block in text_blocks)
            
            # Score combines confidence and text detection
            score = confidence * 0.7 + min(text_amount / 1000, 1.0) * 0.3
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def _enhance_text_block_detection(self, image, text_blocks):
        """Enhance text block detection by merging nearby blocks and improving boundaries."""
        
        if not text_blocks:
            return text_blocks
        
        enhanced_blocks = []
        
        for block in text_blocks:
            bbox = block.get('bbox', {})
            if not bbox:
                continue
            
            # Expand bounding box slightly for better capture
            x = max(0, bbox.get('x', 0) - 2)
            y = max(0, bbox.get('y', 0) - 2)
            w = bbox.get('width', 0) + 4
            h = bbox.get('height', 0) + 4
            
            # Ensure within image bounds
            h_img, w_img = image.shape[:2]
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            enhanced_bbox = {
                'x': x, 'y': y, 'width': w, 'height': h
            }
            
            enhanced_block = block.copy()
            enhanced_block['bbox'] = enhanced_bbox
            enhanced_blocks.append(enhanced_block)
        
        # Merge nearby text blocks that likely belong together
        merged_blocks = self._merge_nearby_text_blocks(enhanced_blocks)
        
        return merged_blocks
    
    def _merge_nearby_text_blocks(self, blocks, distance_threshold=20):
        """Merge text blocks that are close to each other."""
        
        if len(blocks) <= 1:
            return blocks
        
        merged = []
        used = set()
        
        for i, block1 in enumerate(blocks):
            if i in used:
                continue
            
            bbox1 = block1.get('bbox', {})
            if not bbox1:
                merged.append(block1)
                continue
            
            # Find nearby blocks
            group = [block1]
            used.add(i)
            
            for j, block2 in enumerate(blocks[i+1:], i+1):
                if j in used:
                    continue
                
                bbox2 = block2.get('bbox', {})
                if not bbox2:
                    continue
                
                # Check if blocks are close
                distance = self._calculate_block_distance(bbox1, bbox2)
                
                if distance <= distance_threshold:
                    group.append(block2)
                    used.add(j)
            
            if len(group) > 1:
                # Merge the group
                merged_block = self._merge_text_block_group(group)
                merged.append(merged_block)
            else:
                merged.append(block1)
        
        return merged
    
    def _calculate_block_distance(self, bbox1, bbox2):
        """Calculate distance between two bounding boxes."""
        
        x1, y1 = bbox1.get('x', 0), bbox1.get('y', 0)
        w1, h1 = bbox1.get('width', 0), bbox1.get('height', 0)
        
        x2, y2 = bbox2.get('x', 0), bbox2.get('y', 0)
        w2, h2 = bbox2.get('width', 0), bbox2.get('height', 0)
        
        # Calculate center points
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        
        return ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
    
    def _merge_text_block_group(self, group):
        """Merge a group of text blocks into one."""
        
        # Combine text
        texts = [block.get('text', '') for block in group]
        combined_text = ' '.join(filter(None, texts))
        
        # Calculate combined bounding box
        bboxes = [block.get('bbox', {}) for block in group if block.get('bbox')]
        
        if bboxes:
            min_x = min(bbox.get('x', 0) for bbox in bboxes)
            min_y = min(bbox.get('y', 0) for bbox in bboxes)
            max_x = max(bbox.get('x', 0) + bbox.get('width', 0) for bbox in bboxes)
            max_y = max(bbox.get('y', 0) + bbox.get('height', 0) for bbox in bboxes)
            
            combined_bbox = {
                'x': min_x, 'y': min_y,
                'width': max_x - min_x, 'height': max_y - min_y
            }
        else:
            combined_bbox = {}
        
        # Use best confidence from group
        confidences = [block.get('confidence', 0) for block in group]
        best_confidence = max(confidences) if confidences else 0
        
        return {
            'text': combined_text,
            'bbox': combined_bbox,
            'confidence': best_confidence,
            'merged_from': len(group)
        }
    
    def _detect_enhanced_checkbox_regions(self, image):
        """Detect regions that likely contain checkboxes with enhanced methods."""
        
        try:
            import cv2
            
            checkbox_regions = []
            
            # Method 1: Detect rectangular patterns
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 1000:  # Reasonable checkbox size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Check if square-like
                    if 0.5 <= aspect_ratio <= 2.0:
                        checkbox_regions.append({
                            'bbox': (x, y, w, h),
                            'type': 'rectangular',
                            'confidence': 0.7
                        })
            
            # Method 2: Detect circular patterns
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                     param1=50, param2=30, minRadius=5, maxRadius=25)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    checkbox_regions.append({
                        'bbox': (x-r, y-r, 2*r, 2*r),
                        'type': 'circular',
                        'confidence': 0.6
                    })
            
            return checkbox_regions
            
        except Exception as e:
            logger.warning(f"Enhanced checkbox region detection failed: {e}")
            return []
    
    # Keep the same batch processing methods but update them to use intelligent processing
    def process_batch(self, input_paths: List[Path], 
                     output_file: Optional[Path] = None,
                     parallel: bool = True,
                     save_debug: bool = False,
                     detect_multiple_surveys: bool = True,
                     max_pages: Optional[int] = None,
                     max_surveys: Optional[int] = None,
                     enhanced_ocr: bool = True) -> List[ExtractionResult]:
        """Process multiple PDF files in batch with intelligent analysis."""
        
        logger.info(f"Starting intelligent batch processing of {len(input_paths)} files")
        
        results = []
        
        if parallel and len(input_paths) > 1:
            with ThreadPoolExecutor(max_workers=self.config.processing.parallel_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_pdf, path, False, save_debug, detect_multiple_surveys, max_pages, max_surveys, enhanced_ocr): path
                    for path in input_paths
                }
                
                for future in tqdm(as_completed(future_to_path), 
                                 total=len(input_paths), 
                                 desc="Processing PDFs"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        path = future_to_path[future]
                        logger.error(f"Failed to process {path}: {e}")
                        results.append(ExtractionResult(
                            document_path=str(path),
                            total_pages=0,
                            responses=[],
                            processing_time=0,
                            errors=[str(e)],
                            warnings=[]
                        ))
        else:
            for pdf_path in tqdm(input_paths, desc="Processing PDFs"):
                result = self.process_pdf(pdf_path, False, save_debug, detect_multiple_surveys, max_pages, max_surveys, enhanced_ocr)
                results.append(result)
        
        if output_file:
            self.save_results(results, output_file)
        
        logger.info(f"Intelligent batch processing completed: {len(results)} files processed")
        return results
    
    def save_results(self, results: List[ExtractionResult], output_path: Path) -> None:
        """Save extraction results with intelligent analysis metadata."""
        
        # Use the existing data structurer but enhance with intelligent metadata
        all_responses = []
        for result in results:
            for response in result.responses:
                # Add intelligent analysis indicators to metadata
                if not response.metadata.get('intelligent_analysis'):
                    response.metadata['processing_mode'] = 'basic'
                else:
                    response.metadata['processing_mode'] = 'intelligent'
                
            all_responses.extend(result.responses)
        
        if not all_responses:
            logger.warning("No responses to save")
            return
        
        # Use existing save methods
        suffix = output_path.suffix.lower()
        
        if suffix == '.csv':
            self.data_structurer.export_to_csv(all_responses, output_path)
        elif suffix in ['.xlsx', '.xls']:
            self.data_structurer.export_to_excel(all_responses, output_path)
        elif suffix == '.json':
            self.data_structurer.export_to_json(all_responses, output_path)
        else:
            csv_path = output_path.with_suffix('.csv')
            self.data_structurer.export_to_csv(all_responses, csv_path)
            logger.info(f"Unknown format, saved as CSV: {csv_path}")
        
        # Create extraction report
        report_path = output_path.parent / f"{output_path.stem}_intelligent_report.json"
        self.data_structurer.create_extraction_report(results, report_path)
        
        # Enhanced validation for intelligent results
        validation_results = self.data_structurer.validate_structured_data(all_responses)
        logger.info(f"Intelligent validation: {validation_results['quality_score']:.2f} quality score")
        
        if validation_results['issues']:
            logger.warning(f"Found {len(validation_results['issues'])} data quality issues")


# Backward compatibility - keep the original class name as alias
OCRCheckboxAgent = IntelligentOCRAgent


def main():
    """Enhanced main entry point with intelligent features."""
    parser = argparse.ArgumentParser(description="Intelligent OCR Checkbox Extraction Agent")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-c", "--config", help="Configuration file path")
    parser.add_argument("-t", "--template", help="Form template file path")
    parser.add_argument("-f", "--format", choices=["csv", "xlsx", "json"], 
                       default="csv", help="Output format")
    parser.add_argument("--parallel", action="store_true", 
                       help="Enable parallel processing")
    parser.add_argument("--debug", action="store_true", 
                       help="Save debug images and analysis")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--basic-mode", action="store_true",
                       help="Force basic mode (disable intelligent features)")
    parser.add_argument("--no-multi-survey", action="store_true",
                       help="Disable multiple survey detection")
    parser.add_argument("--max-pages", type=int,
                       help="Maximum number of pages to process (for testing)")
    parser.add_argument("--max-surveys", type=int,
                       help="Maximum number of surveys to process (for testing)")
    parser.add_argument("--no-enhanced-ocr", action="store_true",
                       help="Disable enhanced OCR preprocessing")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Load configuration
    config = Config()
    if args.config:
        config = Config.load(Path(args.config))
    
    # Override output format if specified
    if args.format:
        config.output.output_format = args.format
    
    # Initialize agent
    if args.basic_mode:
        from .main import OCRCheckboxAgent as BasicAgent
        agent = BasicAgent(config)
        logger.info("Using basic mode as requested")
    else:
        agent = IntelligentOCRAgent(config)
    
    # Load template if specified
    if args.template:
        agent.load_template(Path(args.template))
    
    # Process input
    input_path = Path(args.input)
    detect_multi_survey = not args.no_multi_survey
    enhanced_ocr = not args.no_enhanced_ocr
    
    if input_path.is_file():
        # Single file processing
        result = agent.process_pdf(
            input_path, 
            save_debug=args.debug,
            detect_multiple_surveys=detect_multi_survey,
            max_pages=args.max_pages,
            max_surveys=args.max_surveys,
            enhanced_ocr=enhanced_ocr
        )
        
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_intelligent_extracted.{args.format}"
        
        agent.save_results([result], output_path)
        
    elif input_path.is_dir():
        # Directory processing
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = input_path / "intelligent_extracted"
        
        # Find PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in {input_path}")
            sys.exit(1)
        
        results = agent.process_batch(
            pdf_files, 
            parallel=args.parallel,
            save_debug=args.debug,
            detect_multiple_surveys=detect_multi_survey,
            max_pages=args.max_pages,
            max_surveys=args.max_surveys,
            enhanced_ocr=enhanced_ocr
        )
        
        # Save consolidated results
        output_dir.mkdir(parents=True, exist_ok=True)
        consolidated_file = output_dir / f"consolidated_intelligent_results.{args.format}"
        agent.save_results(results, consolidated_file)
        
        logger.info(f"Processed {len(results)} files with intelligent analysis")
    
    else:
        logger.error(f"Input path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()