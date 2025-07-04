"""Form Structure Analyzer for intelligent layout understanding and question mapping."""

import cv2
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict
from loguru import logger

from .nlp_engine import NLPEngine, QuestionEntity, FormSection, QuestionType, ResponseType


class LayoutElement(Enum):
    """Types of layout elements that can be detected."""
    HEADER = "header"
    QUESTION = "question"
    RESPONSE_AREA = "response_area"
    INSTRUCTION = "instruction"
    SEPARATOR = "separator"
    FOOTER = "footer"
    TABLE = "table"
    LOGO = "logo"
    UNKNOWN = "unknown"


@dataclass
class ResponseArea:
    """Represents a detected response area."""
    bbox: Tuple[int, int, int, int]
    response_type: ResponseType
    expected_responses: int = 1
    orientation: str = "horizontal"  # horizontal, vertical, grid
    options: List[str] = field(default_factory=list)
    associated_question: Optional[QuestionEntity] = None
    confidence: float = 0.0


@dataclass
class FormLayout:
    """Represents the overall layout structure of a form."""
    sections: List[FormSection]
    response_areas: List[ResponseArea]
    layout_type: str = "unknown"  # single_column, multi_column, grid, mixed
    total_questions: int = 0
    completion_indicators: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurveyBoundary:
    """Represents boundaries of separate surveys within a document."""
    start_y: int
    end_y: int
    title: str
    confidence: float
    sections: List[FormSection] = field(default_factory=list)


class FormAnalyzer:
    """Analyzes form structure and layout for intelligent processing."""
    
    def __init__(self, nlp_engine: Optional[NLPEngine] = None):
        """Initialize form analyzer with NLP engine."""
        self.nlp_engine = nlp_engine or NLPEngine()
        
        # Layout analysis parameters
        self.min_question_spacing = 20
        self.response_area_proximity = 150
        self.section_break_threshold = 50
        
        logger.info("Form Analyzer initialized")
    
    def analyze_form_structure(self, image: np.ndarray, 
                             text_blocks: List[Dict[str, Any]]) -> FormLayout:
        """Analyze complete form structure including layout and content."""
        
        # Step 1: Detect layout elements
        layout_elements = self._detect_layout_elements(image, text_blocks)
        
        # Step 2: Identify questions using NLP
        questions = self.nlp_engine.extract_questions(text_blocks)
        
        # Step 3: Detect response areas
        response_areas = self._detect_response_areas(image, text_blocks, questions)
        
        # Step 4: Map questions to response areas
        self._map_questions_to_responses(questions, response_areas)
        
        # Step 5: Organize into sections
        sections = self._organize_into_sections(questions, layout_elements)
        
        # Step 6: Determine layout type
        layout_type = self._determine_layout_type(sections, response_areas)
        
        # Step 7: Analyze completion indicators
        completion_indicators = self._analyze_completion_indicators(image, text_blocks)
        
        form_layout = FormLayout(
            sections=sections,
            response_areas=response_areas,
            layout_type=layout_type,
            total_questions=len(questions),
            completion_indicators=completion_indicators
        )
        
        logger.info(f"Analyzed form: {len(sections)} sections, {len(questions)} questions, "
                   f"{len(response_areas)} response areas")
        
        return form_layout
    
    def detect_multiple_surveys(self, image: np.ndarray, 
                               text_blocks: List[Dict[str, Any]]) -> List[SurveyBoundary]:
        """Detect multiple surveys within a single document."""
        
        # Look for survey boundaries
        boundaries = []
        
        # Method 1: Look for title patterns
        title_boundaries = self._detect_title_boundaries(text_blocks)
        
        # Method 2: Look for visual separators
        visual_boundaries = self._detect_visual_separators(image)
        
        # Method 3: Look for content breaks
        content_boundaries = self._detect_content_breaks(text_blocks)
        
        # Combine and validate boundaries
        all_boundaries = title_boundaries + visual_boundaries + content_boundaries
        boundaries = self._merge_and_validate_boundaries(all_boundaries, text_blocks)
        
        logger.info(f"Detected {len(boundaries)} potential survey boundaries")
        return boundaries
    
    def _detect_layout_elements(self, image: np.ndarray, 
                               text_blocks: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Detect different types of layout elements."""
        
        elements = {
            'headers': [],
            'questions': [],
            'response_areas': [],
            'instructions': [],
            'separators': [],
            'tables': []
        }
        
        # Analyze text blocks for content type
        for block in text_blocks:
            element_type = self._classify_layout_element(block)
            
            if element_type == LayoutElement.HEADER:
                elements['headers'].append(block)
            elif element_type == LayoutElement.QUESTION:
                elements['questions'].append(block)
            elif element_type == LayoutElement.INSTRUCTION:
                elements['instructions'].append(block)
        
        # Detect visual elements
        elements['separators'] = self._detect_visual_separators(image)
        elements['tables'] = self._detect_table_structures(image)
        
        return elements
    
    def _classify_layout_element(self, block: Dict[str, Any]) -> LayoutElement:
        """Classify a text block into a layout element type."""
        text = block.get('text', '').strip()
        bbox = block.get('bbox', {})
        
        if not text:
            return LayoutElement.UNKNOWN
        
        # Check for headers (large, centered, capitalized)
        if self._is_header_text(text, bbox):
            return LayoutElement.HEADER
        
        # Check for questions
        if self.nlp_engine._is_question_like(text):
            return LayoutElement.QUESTION
        
        # Check for instructions
        if self._is_instruction_text(text):
            return LayoutElement.INSTRUCTION
        
        return LayoutElement.UNKNOWN
    
    def _is_header_text(self, text: str, bbox: Dict[str, Any]) -> bool:
        """Determine if text is a header."""
        # Headers are typically:
        # - Short (< 100 characters)
        # - Capitalized
        # - Larger font (estimated from bbox height)
        # - Centered or near top
        
        if len(text) > 100:
            return False
        
        # Check capitalization
        words = text.split()
        if len(words) > 1:
            cap_ratio = sum(1 for word in words if word[0].isupper()) / len(words)
            if cap_ratio > 0.6:
                return True
        
        # Check for header keywords
        header_keywords = ['survey', 'questionnaire', 'form', 'evaluation', 'feedback']
        if any(keyword in text.lower() for keyword in header_keywords):
            return True
        
        return False
    
    def _is_instruction_text(self, text: str) -> bool:
        """Determine if text is instructional."""
        instruction_indicators = [
            'please', 'instructions', 'directions', 'note', 'important',
            'complete', 'fill out', 'mark', 'select', 'choose'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in instruction_indicators)
    
    def _detect_response_areas(self, image: np.ndarray, 
                              text_blocks: List[Dict[str, Any]],
                              questions: List[QuestionEntity]) -> List[ResponseArea]:
        """Detect areas where responses are expected."""
        
        response_areas = []
        
        # Method 1: Detect checkbox/radio button patterns
        checkbox_areas = self._detect_checkbox_areas(image)
        
        # Method 2: Detect scale/rating areas
        scale_areas = self._detect_scale_areas(image, text_blocks)
        
        # Method 3: Detect text input areas
        text_areas = self._detect_text_input_areas(image)
        
        # Method 4: Detect smiley face or custom symbols
        symbol_areas = self._detect_symbol_areas(image)
        
        # Combine all detected areas
        all_areas = checkbox_areas + scale_areas + text_areas + symbol_areas
        
        # Convert to ResponseArea objects
        for area in all_areas:
            response_area = ResponseArea(
                bbox=area['bbox'],
                response_type=area['type'],
                expected_responses=area.get('count', 1),
                orientation=area.get('orientation', 'horizontal'),
                confidence=area.get('confidence', 0.5)
            )
            response_areas.append(response_area)
        
        return response_areas
    
    def _detect_checkbox_areas(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect checkbox-like patterns in the image."""
        areas = []
        
        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        checkbox_candidates = []
        
        for contour in contours:
            # Filter by area and aspect ratio
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  # Reasonable checkbox size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check if square-like
                if 0.7 <= aspect_ratio <= 1.3:
                    checkbox_candidates.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
        
        # Group nearby checkboxes
        grouped_checkboxes = self._group_nearby_elements(checkbox_candidates)
        
        for group in grouped_checkboxes:
            if len(group) >= 2:  # At least 2 checkboxes
                # Calculate bounding box for the group
                min_x = min(cb['bbox'][0] for cb in group)
                min_y = min(cb['bbox'][1] for cb in group)
                max_x = max(cb['bbox'][0] + cb['bbox'][2] for cb in group)
                max_y = max(cb['bbox'][1] + cb['bbox'][3] for cb in group)
                
                orientation = 'horizontal' if (max_x - min_x) > (max_y - min_y) else 'vertical'
                
                areas.append({
                    'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                    'type': ResponseType.CHECKBOX,
                    'count': len(group),
                    'orientation': orientation,
                    'confidence': 0.8
                })
        
        return areas
    
    def _detect_scale_areas(self, image: np.ndarray, 
                           text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect rating scale areas (numbers, lines, etc.)."""
        areas = []
        
        # Look for number sequences in text blocks
        for block in text_blocks:
            text = block.get('text', '')
            bbox = block.get('bbox', {})
            
            # Check for number sequences (1 2 3 4 5, 1-5, etc.)
            number_patterns = [
                r'1\s+2\s+3\s+4\s+5',
                r'1\s+2\s+3\s+4\s+5\s+6\s+7',
                r'1\s+2\s+3\s+4\s+5\s+6\s+7\s+8\s+9\s+10'
            ]
            
            for pattern in number_patterns:
                if re.search(pattern, text):
                    scale_size = len(text.split())
                    areas.append({
                        'bbox': (bbox.get('x', 0), bbox.get('y', 0), 
                               bbox.get('width', 0), bbox.get('height', 0)),
                        'type': ResponseType.NUMBER_SCALE,
                        'count': scale_size,
                        'orientation': 'horizontal',
                        'confidence': 0.9
                    })
        
        return areas
    
    def _detect_text_input_areas(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text input areas (lines, boxes)."""
        areas = []
        
        # Detect horizontal lines (text input lines)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect horizontal lines using morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Reasonable line size
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if line-like (wide and thin)
                if w > h * 5:
                    areas.append({
                        'bbox': (x, y, w, h),
                        'type': ResponseType.TEXT_BOX,
                        'count': 1,
                        'orientation': 'horizontal',
                        'confidence': 0.6
                    })
        
        return areas
    
    def _detect_symbol_areas(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect special symbols like smiley faces, stars, etc."""
        areas = []
        
        # This would require more sophisticated computer vision
        # For now, we'll implement basic circle detection for smiley faces
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=10, maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Group circles if there are multiple
            if len(circles) >= 3:  # At least 3 for a rating scale
                # Calculate bounding box
                min_x = min(circle[0] - circle[2] for circle in circles)
                min_y = min(circle[1] - circle[2] for circle in circles)
                max_x = max(circle[0] + circle[2] for circle in circles)
                max_y = max(circle[1] + circle[2] for circle in circles)
                
                areas.append({
                    'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                    'type': ResponseType.SMILEY_FACE,
                    'count': len(circles),
                    'orientation': 'horizontal',
                    'confidence': 0.7
                })
        
        return areas
    
    def _group_nearby_elements(self, elements: List[Dict[str, Any]], 
                              distance_threshold: int = 100) -> List[List[Dict[str, Any]]]:
        """Group nearby elements together."""
        if not elements:
            return []
        
        groups = []
        used = set()
        
        for i, element in enumerate(elements):
            if i in used:
                continue
            
            group = [element]
            used.add(i)
            
            # Find nearby elements
            for j, other in enumerate(elements[i+1:], i+1):
                if j in used:
                    continue
                
                # Calculate distance between centers
                center1 = (
                    element['bbox'][0] + element['bbox'][2] // 2,
                    element['bbox'][1] + element['bbox'][3] // 2
                )
                center2 = (
                    other['bbox'][0] + other['bbox'][2] // 2,
                    other['bbox'][1] + other['bbox'][3] // 2
                )
                
                distance = math.sqrt(
                    (center1[0] - center2[0]) ** 2 + 
                    (center1[1] - center2[1]) ** 2
                )
                
                if distance <= distance_threshold:
                    group.append(other)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _map_questions_to_responses(self, questions: List[QuestionEntity], 
                                   response_areas: List[ResponseArea]) -> None:
        """Map questions to their corresponding response areas."""
        
        for question in questions:
            if not question.bbox:
                continue
            
            best_area = None
            min_distance = float('inf')
            
            # Find closest response area
            q_center_y = question.bbox[1] + question.bbox[3] // 2
            
            for area in response_areas:
                if area.associated_question is not None:
                    continue  # Already assigned
                
                # Calculate vertical distance
                a_center_y = area.bbox[1] + area.bbox[3] // 2
                distance = abs(q_center_y - a_center_y)
                
                # Prefer areas that are to the right or below the question
                if area.bbox[1] >= question.bbox[1]:  # Below or at same level
                    distance *= 0.8  # Preference bonus
                
                if distance < min_distance and distance < self.response_area_proximity:
                    min_distance = distance
                    best_area = area
            
            # Assign the best matching area
            if best_area:
                best_area.associated_question = question
                
                # Update response type based on question type
                if question.response_type != ResponseType.UNKNOWN:
                    best_area.response_type = question.response_type
    
    def _organize_into_sections(self, questions: List[QuestionEntity], 
                               layout_elements: Dict[str, List[Dict]]) -> List[FormSection]:
        """Organize questions into logical sections."""
        
        # Use NLP engine to detect sections
        text_blocks = []
        for question in questions:
            if question.bbox:
                text_blocks.append({
                    'text': question.text,
                    'bbox': {
                        'x': question.bbox[0],
                        'y': question.bbox[1],
                        'width': question.bbox[2],
                        'height': question.bbox[3]
                    }
                })
        
        # Add headers to text blocks
        for header in layout_elements.get('headers', []):
            text_blocks.append(header)
        
        # Sort by vertical position
        text_blocks.sort(key=lambda x: x.get('bbox', {}).get('y', 0))
        
        sections = self.nlp_engine.detect_form_sections(text_blocks)
        
        # If no sections detected, create a default section
        if not sections and questions:
            sections = [FormSection(
                title="General Questions",
                questions=questions,
                section_type="general"
            )]
        
        return sections
    
    def _determine_layout_type(self, sections: List[FormSection], 
                              response_areas: List[ResponseArea]) -> str:
        """Determine the overall layout type of the form."""
        
        if not sections:
            return "unknown"
        
        # Analyze question positions
        all_questions = []
        for section in sections:
            all_questions.extend(section.questions)
        
        if not all_questions:
            return "unknown"
        
        # Check if questions are arranged in columns
        x_positions = [q.bbox[0] for q in all_questions if q.bbox]
        
        if len(x_positions) < 2:
            return "single_column"
        
        # Group by x-position to detect columns
        x_positions.sort()
        columns = []
        current_column = [x_positions[0]]
        
        for x in x_positions[1:]:
            if x - current_column[-1] < 100:  # Same column
                current_column.append(x)
            else:  # New column
                columns.append(current_column)
                current_column = [x]
        
        columns.append(current_column)
        
        if len(columns) == 1:
            return "single_column"
        elif len(columns) == 2:
            return "two_column"
        elif len(columns) > 2:
            return "multi_column"
        else:
            return "mixed"
    
    def _analyze_completion_indicators(self, image: np.ndarray, 
                                     text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze indicators of form completion status."""
        
        indicators = {
            'has_signature_area': False,
            'has_date_field': False,
            'completion_percentage': 0.0,
            'required_fields': 0,
            'filled_fields': 0
        }
        
        # Look for signature indicators
        signature_keywords = ['signature', 'sign here', 'signed', 'name']
        for block in text_blocks:
            text = block.get('text', '').lower()
            if any(keyword in text for keyword in signature_keywords):
                indicators['has_signature_area'] = True
                break
        
        # Look for date fields
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'date',
            r'mm/dd/yyyy'
        ]
        
        for block in text_blocks:
            text = block.get('text', '').lower()
            for pattern in date_patterns:
                if re.search(pattern, text):
                    indicators['has_date_field'] = True
                    break
        
        return indicators
    
    def _detect_title_boundaries(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect survey boundaries based on title patterns."""
        boundaries = []
        
        for block in text_blocks:
            text = block.get('text', '').strip()
            bbox = block.get('bbox', {})
            
            # Look for survey/form titles
            title_patterns = [
                r'(survey|questionnaire|form|evaluation)\s+\d+',
                r'part\s+[ivx\d]+',
                r'section\s+[ivx\d]+',
                r'^[A-Z][A-Z\s]{10,50}$'  # All caps titles
            ]
            
            for pattern in title_patterns:
                if re.search(pattern, text.lower()):
                    boundaries.append({
                        'y_position': bbox.get('y', 0),
                        'title': text,
                        'type': 'title',
                        'confidence': 0.8
                    })
                    break
        
        return boundaries
    
    def _detect_visual_separators(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect visual separators like lines, borders, etc."""
        separators = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Significant line
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's a separator line (wide and thin)
                if w > h * 10:
                    separators.append({
                        'y_position': y,
                        'type': 'line',
                        'confidence': 0.6,
                        'bbox': (x, y, w, h)
                    })
        
        return separators
    
    def _detect_content_breaks(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect content breaks based on spacing and content changes."""
        boundaries = []
        
        if len(text_blocks) < 2:
            return boundaries
        
        # Sort blocks by vertical position
        sorted_blocks = sorted(text_blocks, key=lambda x: x.get('bbox', {}).get('y', 0))
        
        # Look for large gaps between text blocks
        for i in range(len(sorted_blocks) - 1):
            current_block = sorted_blocks[i]
            next_block = sorted_blocks[i + 1]
            
            current_bottom = current_block.get('bbox', {}).get('y', 0) + current_block.get('bbox', {}).get('height', 0)
            next_top = next_block.get('bbox', {}).get('y', 0)
            
            gap = next_top - current_bottom
            
            # If gap is significantly larger than normal
            if gap > self.section_break_threshold:
                boundaries.append({
                    'y_position': current_bottom + gap // 2,
                    'type': 'content_break',
                    'confidence': 0.4,
                    'gap_size': gap
                })
        
        return boundaries
    
    def _merge_and_validate_boundaries(self, boundaries: List[Dict[str, Any]], 
                                     text_blocks: List[Dict[str, Any]]) -> List[SurveyBoundary]:
        """Merge and validate detected boundaries."""
        
        if not boundaries:
            return []
        
        # Sort boundaries by y position
        boundaries.sort(key=lambda x: x['y_position'])
        
        # Merge nearby boundaries
        merged_boundaries = []
        current_boundary = boundaries[0]
        
        for boundary in boundaries[1:]:
            if abs(boundary['y_position'] - current_boundary['y_position']) < 50:
                # Merge boundaries - keep the one with higher confidence
                if boundary['confidence'] > current_boundary['confidence']:
                    current_boundary = boundary
            else:
                merged_boundaries.append(current_boundary)
                current_boundary = boundary
        
        merged_boundaries.append(current_boundary)
        
        # Convert to SurveyBoundary objects
        survey_boundaries = []
        
        for i, boundary in enumerate(merged_boundaries):
            start_y = boundary['y_position']
            end_y = merged_boundaries[i + 1]['y_position'] if i + 1 < len(merged_boundaries) else float('inf')
            
            title = boundary.get('title', f"Survey {i + 1}")
            
            survey_boundary = SurveyBoundary(
                start_y=start_y,
                end_y=end_y,
                title=title,
                confidence=boundary['confidence']
            )
            
            survey_boundaries.append(survey_boundary)
        
        return survey_boundaries
    
    def _detect_table_structures(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect table-like structures in the form."""
        tables = []
        
        # This is a simplified table detection
        # In practice, you might want to use more sophisticated methods
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect both horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find intersections (potential table corners)
        intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
        
        contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) >= 4:  # At least 4 intersections for a table
            # Calculate bounding box of all intersections
            all_points = np.vstack([contour for contour in contours])
            x, y, w, h = cv2.boundingRect(all_points)
            
            tables.append({
                'bbox': (x, y, w, h),
                'type': 'table',
                'confidence': 0.7
            })
        
        return tables