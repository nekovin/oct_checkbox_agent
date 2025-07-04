"""Intelligent Response Detector for handling multiple response types and formats."""

import cv2
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from loguru import logger

from .nlp_engine import QuestionEntity, QuestionType, ResponseType
from .form_analyzer import ResponseArea
from .checkbox_detector import Checkbox, CheckboxState


class ResponseState(Enum):
    """Enhanced response states for various response types."""
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    SELECTED = "selected"
    UNSELECTED = "unselected"
    FILLED = "filled"
    EMPTY = "empty"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class ResponseValue:
    """Represents a detected response value with context."""
    value: Any  # The actual response value
    state: ResponseState
    confidence: float
    response_type: ResponseType
    bbox: Optional[Tuple[int, int, int, int]] = None
    raw_data: Optional[Dict[str, Any]] = None
    validation_status: str = "valid"  # valid, invalid, uncertain
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaleResponse:
    """Represents a response on a scale (numeric, smiley, stars, etc.)."""
    selected_value: Union[int, str]
    scale_range: Tuple[Union[int, str], Union[int, str]]
    scale_type: str  # numeric, smiley, star, custom
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class MultipleChoiceResponse:
    """Represents multiple choice response(s)."""
    selected_options: List[str]
    available_options: List[str]
    selection_type: str  # single, multiple
    confidence: float
    bboxes: List[Tuple[int, int, int, int]]


class IntelligentResponseDetector:
    """Advanced response detector that handles multiple response types intelligently."""
    
    def __init__(self, config=None):
        """Initialize the intelligent response detector."""
        self.config = config
        
        # Response type handlers
        self.type_handlers = {
            ResponseType.CHECKBOX: self._detect_checkbox_responses,
            ResponseType.RADIO_BUTTON: self._detect_radio_responses,
            ResponseType.SMILEY_FACE: self._detect_smiley_responses,
            ResponseType.STAR_RATING: self._detect_star_responses,
            ResponseType.NUMBER_SCALE: self._detect_number_scale_responses,
            ResponseType.TEXT_BOX: self._detect_text_responses,
            ResponseType.SLIDER: self._detect_slider_responses
        }
        
        # Pattern libraries for different response types
        self._setup_pattern_libraries()
        
        logger.info("Intelligent Response Detector initialized")
    
    def _setup_pattern_libraries(self):
        """Setup pattern libraries for various response types."""
        
        # Smiley face patterns (Unicode and ASCII)
        self.smiley_patterns = {
            'very_happy': ['😊', '😄', '😃', ':D', ':-D'],
            'happy': ['🙂', '😊', ':)', ':-)'],
            'neutral': ['😐', '😑', ':|', ':-|'],
            'sad': ['😞', '☹️', ':(', ':-('],
            'very_sad': ['😢', '😭', ':((', 'T_T']
        }
        
        # Star rating patterns
        self.star_patterns = {
            'filled': ['★', '⭐', '*'],
            'empty': ['☆', '✩'],
            'half': ['☆']
        }
        
        # Number scale indicators
        self.scale_indicators = [
            r'\b([1-9]|10)\b',  # Numbers 1-10
            r'([1-5])\s*/\s*5',  # X/5 format
            r'([1-9]|10)\s*/\s*10'  # X/10 format
        ]
        
        # Checkbox state indicators
        self.checkbox_indicators = {
            'checked': ['✓', '✔', 'X', 'x', '☑', '■'],
            'unchecked': ['☐', '□', '○', '◯']
        }
    
    def detect_responses(self, image: np.ndarray, 
                        question: QuestionEntity,
                        response_area: ResponseArea,
                        text_blocks: List[Dict[str, Any]]) -> List[ResponseValue]:
        """Detect responses for a given question and response area."""
        
        responses = []
        
        # Extract the response area from the image
        if response_area.bbox:
            x, y, w, h = response_area.bbox
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        # Get the appropriate handler for this response type
        handler = self.type_handlers.get(response_area.response_type, self._detect_generic_responses)
        
        try:
            # Detect responses using the specific handler
            detected_responses = handler(roi, question, response_area, text_blocks)
            responses.extend(detected_responses)
            
            # Validate responses against question context
            validated_responses = self._validate_responses(responses, question)
            
            logger.debug(f"Detected {len(validated_responses)} responses for question: {question.text[:50]}...")
            
            return validated_responses
            
        except Exception as e:
            logger.error(f"Error detecting responses: {e}")
            return []
    
    def _detect_checkbox_responses(self, roi: np.ndarray, 
                                  question: QuestionEntity,
                                  response_area: ResponseArea,
                                  text_blocks: List[Dict[str, Any]]) -> List[ResponseValue]:
        """Detect checkbox-style responses."""
        responses = []
        
        # Use existing checkbox detector as base
        from .checkbox_detector import CheckboxDetector
        detector = CheckboxDetector(self.config)
        
        # Convert response area text blocks to format expected by detector
        area_text_blocks = self._filter_text_blocks_to_area(text_blocks, response_area.bbox)
        
        checkboxes = detector.detect_checkboxes(roi, area_text_blocks)
        
        for checkbox in checkboxes:
            # Map checkbox state to response state
            if checkbox.state == CheckboxState.CHECKED:
                state = ResponseState.CHECKED
                value = True
            elif checkbox.state == CheckboxState.UNCHECKED:
                state = ResponseState.UNCHECKED
                value = False
            else:
                state = ResponseState.UNKNOWN
                value = None
            
            response = ResponseValue(
                value=value,
                state=state,
                confidence=checkbox.confidence,
                response_type=ResponseType.CHECKBOX,
                bbox=checkbox.bbox,
                raw_data={'checkbox': checkbox},
                metadata={
                    'fill_ratio': checkbox.fill_ratio,
                    'associated_text': checkbox.associated_text
                }
            )
            
            responses.append(response)
        
        return responses
    
    def _detect_radio_responses(self, roi: np.ndarray,
                               question: QuestionEntity,
                               response_area: ResponseArea,
                               text_blocks: List[Dict[str, Any]]) -> List[ResponseValue]:
        """Detect radio button style responses."""
        responses = []
        
        # Look for circular patterns
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Extract circle region
                circle_roi = gray[max(0, y-r):y+r, max(0, x-r):x+r]
                
                # Determine if filled (selected)
                if circle_roi.size > 0:
                    fill_ratio = np.sum(circle_roi < 128) / circle_roi.size
                    
                    if fill_ratio > 0.3:
                        state = ResponseState.SELECTED
                        value = True
                    else:
                        state = ResponseState.UNSELECTED
                        value = False
                    
                    # Find associated text
                    associated_text = self._find_nearest_text_to_point(
                        (x, y), text_blocks, response_area.bbox
                    )
                    
                    response = ResponseValue(
                        value=associated_text if associated_text else value,
                        state=state,
                        confidence=0.8,
                        response_type=ResponseType.RADIO_BUTTON,
                        bbox=(x-r, y-r, 2*r, 2*r),
                        metadata={
                            'fill_ratio': fill_ratio,
                            'associated_text': associated_text
                        }
                    )
                    
                    responses.append(response)
        
        return responses
    
    def _detect_smiley_responses(self, roi: np.ndarray,
                                question: QuestionEntity,
                                response_area: ResponseArea,
                                text_blocks: List[Dict[str, Any]]) -> List[ResponseValue]:
        """Detect smiley face responses."""
        responses = []
        
        # Method 1: Look for Unicode smiley faces in text
        area_text_blocks = self._filter_text_blocks_to_area(text_blocks, response_area.bbox)
        
        for block in area_text_blocks:
            text = block.get('text', '')
            
            for emotion, patterns in self.smiley_patterns.items():
                for pattern in patterns:
                    if pattern in text:
                        response = ResponseValue(
                            value=emotion,
                            state=ResponseState.SELECTED,
                            confidence=0.9,
                            response_type=ResponseType.SMILEY_FACE,
                            bbox=block.get('bbox'),
                            metadata={'emotion': emotion, 'pattern': pattern}
                        )
                        responses.append(response)
        
        # Method 2: Detect drawn smiley faces using computer vision
        cv_responses = self._detect_drawn_smileys(roi)
        responses.extend(cv_responses)
        
        return responses
    
    def _detect_drawn_smileys(self, roi: np.ndarray) -> List[ResponseValue]:
        """Detect hand-drawn or printed smiley faces."""
        responses = []
        
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
        
        # Detect circles (faces)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=15, maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Extract face region
                face_roi = gray[max(0, y-r):y+r, max(0, x-r):x+r]
                
                if face_roi.size > 0:
                    # Analyze face expression (simplified)
                    emotion = self._analyze_smiley_expression(face_roi)
                    
                    if emotion != 'unknown':
                        response = ResponseValue(
                            value=emotion,
                            state=ResponseState.SELECTED,
                            confidence=0.7,
                            response_type=ResponseType.SMILEY_FACE,
                            bbox=(x-r, y-r, 2*r, 2*r),
                            metadata={'emotion': emotion, 'detection_method': 'computer_vision'}
                        )
                        responses.append(response)
        
        return responses
    
    def _analyze_smiley_expression(self, face_roi: np.ndarray) -> str:
        """Analyze smiley face expression (simplified implementation)."""
        
        # This is a simplified analysis
        # In practice, you might use more sophisticated techniques
        
        h, w = face_roi.shape
        
        # Look for mouth curve in bottom half
        bottom_half = face_roi[h//2:, :]
        
        # Apply edge detection
        edges = cv2.Canny(bottom_half, 50, 150)
        
        # Count edge pixels in different regions
        left_third = edges[:, :w//3]
        middle_third = edges[:, w//3:2*w//3]
        right_third = edges[:, 2*w//3:]
        
        left_edges = np.sum(left_third > 0)
        middle_edges = np.sum(middle_third > 0)
        right_edges = np.sum(right_third > 0)
        
        # Simple heuristic: if more edges on sides than middle, likely smile
        if (left_edges + right_edges) > middle_edges * 1.5:
            return 'happy'
        elif middle_edges > (left_edges + right_edges) * 1.5:
            return 'sad'
        else:
            return 'neutral'
    
    def _detect_star_responses(self, roi: np.ndarray,
                              question: QuestionEntity,
                              response_area: ResponseArea,
                              text_blocks: List[Dict[str, Any]]) -> List[ResponseValue]:
        """Detect star rating responses."""
        responses = []
        
        # Method 1: Look for Unicode star characters
        area_text_blocks = self._filter_text_blocks_to_area(text_blocks, response_area.bbox)
        
        star_count = 0
        filled_stars = 0
        
        for block in area_text_blocks:
            text = block.get('text', '')
            
            # Count filled and empty stars
            for filled_pattern in self.star_patterns['filled']:
                filled_stars += text.count(filled_pattern)
            
            for empty_pattern in self.star_patterns['empty']:
                star_count += text.count(empty_pattern)
            
            star_count += filled_stars
        
        if star_count > 0:
            rating = filled_stars  # Assuming filled stars indicate rating
            
            response = ResponseValue(
                value=rating,
                state=ResponseState.SELECTED,
                confidence=0.8,
                response_type=ResponseType.STAR_RATING,
                metadata={
                    'total_stars': star_count,
                    'filled_stars': filled_stars,
                    'rating': rating
                }
            )
            responses.append(response)
        
        # Method 2: Detect drawn/printed stars using computer vision
        cv_responses = self._detect_drawn_stars(roi)
        responses.extend(cv_responses)
        
        return responses
    
    def _detect_drawn_stars(self, roi: np.ndarray) -> List[ResponseValue]:
        """Detect drawn or printed star shapes."""
        responses = []
        
        # This would require more sophisticated shape detection
        # For now, implement a basic version
        
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
        
        # Apply threshold and find contours
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        star_like_shapes = 0
        
        for contour in contours:
            # Simple heuristic: stars have multiple corners
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If it has 8-12 vertices, might be a star
            if 8 <= len(approx) <= 12:
                area = cv2.contourArea(contour)
                if 100 < area < 2000:  # Reasonable size
                    star_like_shapes += 1
        
        if star_like_shapes > 0:
            response = ResponseValue(
                value=star_like_shapes,
                state=ResponseState.SELECTED,
                confidence=0.6,
                response_type=ResponseType.STAR_RATING,
                metadata={
                    'detected_stars': star_like_shapes,
                    'detection_method': 'computer_vision'
                }
            )
            responses.append(response)
        
        return responses
    
    def _detect_number_scale_responses(self, roi: np.ndarray,
                                      question: QuestionEntity,
                                      response_area: ResponseArea,
                                      text_blocks: List[Dict[str, Any]]) -> List[ResponseValue]:
        """Detect numeric scale responses."""
        responses = []
        
        area_text_blocks = self._filter_text_blocks_to_area(text_blocks, response_area.bbox)
        
        for block in area_text_blocks:
            text = block.get('text', '')
            
            # Look for circled or marked numbers
            for pattern in self.scale_indicators:
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        value = int(match)
                        
                        # Check if this number appears to be selected
                        # (this would need more sophisticated analysis)
                        confidence = 0.7
                        
                        response = ResponseValue(
                            value=value,
                            state=ResponseState.SELECTED,
                            confidence=confidence,
                            response_type=ResponseType.NUMBER_SCALE,
                            bbox=block.get('bbox'),
                            metadata={'scale_value': value}
                        )
                        responses.append(response)
                        
                    except ValueError:
                        continue
        
        # Also detect visual indicators (circles around numbers, etc.)
        visual_responses = self._detect_visual_number_selection(roi, area_text_blocks)
        responses.extend(visual_responses)
        
        return responses
    
    def _detect_visual_number_selection(self, roi: np.ndarray, 
                                       text_blocks: List[Dict[str, Any]]) -> List[ResponseValue]:
        """Detect visual indicators of number selection (circles, highlighting, etc.)."""
        responses = []
        
        # Look for circles around numbers
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=30
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Find if there's a number near this circle
                for block in text_blocks:
                    bbox = block.get('bbox')
                    if bbox:
                        block_center_x = bbox.get('x', 0) + bbox.get('width', 0) // 2
                        block_center_y = bbox.get('y', 0) + bbox.get('height', 0) // 2
                        
                        distance = math.sqrt((x - block_center_x)**2 + (y - block_center_y)**2)
                        
                        if distance < r * 2:  # Close to the circle
                            text = block.get('text', '')
                            numbers = re.findall(r'\b([1-9]|10)\b', text)
                            
                            if numbers:
                                value = int(numbers[0])
                                
                                response = ResponseValue(
                                    value=value,
                                    state=ResponseState.SELECTED,
                                    confidence=0.8,
                                    response_type=ResponseType.NUMBER_SCALE,
                                    bbox=(x-r, y-r, 2*r, 2*r),
                                    metadata={
                                        'selection_method': 'circled',
                                        'scale_value': value
                                    }
                                )
                                responses.append(response)
        
        return responses
    
    def _detect_text_responses(self, roi: np.ndarray,
                              question: QuestionEntity,
                              response_area: ResponseArea,
                              text_blocks: List[Dict[str, Any]]) -> List[ResponseValue]:
        """Detect text input responses."""
        responses = []
        
        area_text_blocks = self._filter_text_blocks_to_area(text_blocks, response_area.bbox)
        
        for block in area_text_blocks:
            text = block.get('text', '').strip()
            
            if text and not self._is_question_or_instruction(text):
                # This appears to be a response
                response = ResponseValue(
                    value=text,
                    state=ResponseState.FILLED,
                    confidence=0.7,
                    response_type=ResponseType.TEXT_BOX,
                    bbox=block.get('bbox'),
                    metadata={'text_length': len(text)}
                )
                responses.append(response)
        
        return responses
    
    def _detect_slider_responses(self, roi: np.ndarray,
                                question: QuestionEntity,
                                response_area: ResponseArea,
                                text_blocks: List[Dict[str, Any]]) -> List[ResponseValue]:
        """Detect slider/gauge responses."""
        responses = []
        
        # Look for line-like structures with marks
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
        
        # Detect horizontal lines (slider tracks)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w > h * 5:  # Line-like
                # Look for marks on the line (slider position)
                line_roi = gray[y-5:y+h+5, x:x+w]
                
                # Find vertical marks
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
                marks = cv2.morphologyEx(line_roi, cv2.MORPH_OPEN, vertical_kernel)
                
                mark_contours, _ = cv2.findContours(marks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if mark_contours:
                    # Find the position of the mark relative to the line
                    for mark_contour in mark_contours:
                        mx, my, mw, mh = cv2.boundingRect(mark_contour)
                        
                        # Calculate position as percentage
                        position = (mx + mw//2) / w
                        
                        response = ResponseValue(
                            value=position,
                            state=ResponseState.SELECTED,
                            confidence=0.6,
                            response_type=ResponseType.SLIDER,
                            bbox=(x + mx, y + my, mw, mh),
                            metadata={
                                'position_percentage': position,
                                'slider_length': w
                            }
                        )
                        responses.append(response)
        
        return responses
    
    def _detect_generic_responses(self, roi: np.ndarray,
                                 question: QuestionEntity,
                                 response_area: ResponseArea,
                                 text_blocks: List[Dict[str, Any]]) -> List[ResponseValue]:
        """Generic response detection for unknown types."""
        responses = []
        
        # Try multiple detection methods
        checkbox_responses = self._detect_checkbox_responses(roi, question, response_area, text_blocks)
        text_responses = self._detect_text_responses(roi, question, response_area, text_blocks)
        
        responses.extend(checkbox_responses)
        responses.extend(text_responses)
        
        return responses
    
    def _filter_text_blocks_to_area(self, text_blocks: List[Dict[str, Any]], 
                                   area_bbox: Optional[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
        """Filter text blocks to those within the response area."""
        if not area_bbox:
            return text_blocks
        
        ax, ay, aw, ah = area_bbox
        filtered_blocks = []
        
        for block in text_blocks:
            bbox = block.get('bbox')
            if bbox:
                bx = bbox.get('x', 0)
                by = bbox.get('y', 0)
                bw = bbox.get('width', 0)
                bh = bbox.get('height', 0)
                
                # Check if block overlaps with area
                if (bx < ax + aw and bx + bw > ax and 
                    by < ay + ah and by + bh > ay):
                    filtered_blocks.append(block)
        
        return filtered_blocks
    
    def _find_nearest_text_to_point(self, point: Tuple[int, int], 
                                   text_blocks: List[Dict[str, Any]],
                                   area_bbox: Optional[Tuple[int, int, int, int]]) -> Optional[str]:
        """Find the nearest text to a given point."""
        
        filtered_blocks = self._filter_text_blocks_to_area(text_blocks, area_bbox)
        
        if not filtered_blocks:
            return None
        
        px, py = point
        nearest_text = None
        min_distance = float('inf')
        
        for block in filtered_blocks:
            bbox = block.get('bbox')
            text = block.get('text', '').strip()
            
            if bbox and text:
                # Calculate distance to block center
                bx = bbox.get('x', 0) + bbox.get('width', 0) // 2
                by = bbox.get('y', 0) + bbox.get('height', 0) // 2
                
                distance = math.sqrt((px - bx)**2 + (py - by)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_text = text
        
        return nearest_text
    
    def _is_question_or_instruction(self, text: str) -> bool:
        """Check if text is likely a question or instruction rather than a response."""
        text_lower = text.lower()
        
        # Common question/instruction indicators
        indicators = [
            '?', 'please', 'select', 'choose', 'rate', 'indicate', 
            'mark', 'check', 'how', 'what', 'when', 'where', 'why'
        ]
        
        return any(indicator in text_lower for indicator in indicators)
    
    def _validate_responses(self, responses: List[ResponseValue], 
                           question: QuestionEntity) -> List[ResponseValue]:
        """Validate responses against question context and expected format."""
        
        validated_responses = []
        
        for response in responses:
            # Basic validation
            validation_status = "valid"
            
            # Check response type compatibility
            if not self._is_response_compatible(response, question):
                validation_status = "invalid"
            
            # Check value range for scales
            if question.scale_range and response.response_type in [ResponseType.NUMBER_SCALE, ResponseType.STAR_RATING]:
                if isinstance(response.value, (int, float)):
                    min_val, max_val = question.scale_range
                    if not (min_val <= response.value <= max_val):
                        validation_status = "invalid"
            
            # Check for multiple selections when only one expected
            if (question.question_type == QuestionType.YES_NO and 
                len([r for r in responses if r.state == ResponseState.SELECTED]) > 1):
                validation_status = "invalid"
            
            response.validation_status = validation_status
            validated_responses.append(response)
        
        return validated_responses
    
    def _is_response_compatible(self, response: ResponseValue, question: QuestionEntity) -> bool:
        """Check if response type is compatible with question type."""
        
        compatible_pairs = {
            QuestionType.YES_NO: [ResponseType.CHECKBOX, ResponseType.RADIO_BUTTON],
            QuestionType.RATING_SCALE: [ResponseType.NUMBER_SCALE, ResponseType.STAR_RATING, ResponseType.SMILEY_FACE],
            QuestionType.LIKERT_SCALE: [ResponseType.RADIO_BUTTON, ResponseType.NUMBER_SCALE],
            QuestionType.MULTIPLE_CHOICE: [ResponseType.CHECKBOX, ResponseType.RADIO_BUTTON],
            QuestionType.TEXT_INPUT: [ResponseType.TEXT_BOX],
            QuestionType.NUMERIC: [ResponseType.TEXT_BOX, ResponseType.NUMBER_SCALE]
        }
        
        expected_types = compatible_pairs.get(question.question_type, [])
        return response.response_type in expected_types or not expected_types  # Allow if no specific expectation