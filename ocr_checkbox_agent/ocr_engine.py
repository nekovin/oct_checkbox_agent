"""OCR engine module for text extraction using pytesseract."""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
import pytesseract
from PIL import Image
from loguru import logger
import re

from .config import get_config
from .utils import (
    timing_decorator, ensure_grayscale, apply_threshold, 
    denoise_image, remove_lines, calculate_confidence_score,
    merge_nearby_boxes, normalize_text, save_debug_image
)


class OCREngine:
    """Handles text extraction from images using pytesseract."""
    
    def __init__(self, config=None):
        """Initialize OCR engine with configuration."""
        self.config = config or get_config()
        self.language = self.config.ocr.language
        self.psm = self.config.ocr.psm
        self.oem = self.config.ocr.oem
        self.confidence_threshold = self.config.ocr.confidence_threshold
        
        # Verify tesseract installation
        self._verify_tesseract()
        
    def _verify_tesseract(self):
        """Verify tesseract is installed and accessible."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise RuntimeError("Tesseract is not installed or not in PATH")
    
    @timing_decorator
    def extract_text(self, image: np.ndarray, 
                    preprocess: bool = True) -> Dict[str, Any]:
        """
        Extract text from image using OCR.
        
        Args:
            image: Input image
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary containing text and metadata
        """
        if preprocess:
            image = self._preprocess_for_ocr(image)
        
        # Get OCR data with bounding boxes
        ocr_data = self._get_ocr_data(image)
        
        # Extract structured text information
        text_blocks = self._extract_text_blocks(ocr_data)
        
        # Calculate overall confidence
        confidence = calculate_confidence_score(ocr_data)
        
        result = {
            'full_text': ' '.join([block['text'] for block in text_blocks]),
            'text_blocks': text_blocks,
            'confidence': confidence,
            'raw_data': ocr_data
        }
        
        return result
    
    def extract_text_regions(self, image: np.ndarray, 
                           regions: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
        """
        Extract text from specific regions of an image.
        
        Args:
            image: Input image
            regions: List of regions (x, y, w, h)
            
        Returns:
            List of extraction results for each region
        """
        results = []
        
        for idx, (x, y, w, h) in enumerate(regions):
            # Extract region
            roi = image[y:y+h, x:x+w]
            
            # Extract text from region
            result = self.extract_text(roi)
            result['region'] = {'x': x, 'y': y, 'width': w, 'height': h}
            result['region_index'] = idx
            
            results.append(result)
        
        return results
    
    def find_text_near_point(self, image: np.ndarray, 
                           point: Tuple[int, int], 
                           radius: int = 100) -> Optional[str]:
        """
        Find text near a specific point in the image.
        
        Args:
            image: Input image
            point: (x, y) coordinates
            radius: Search radius
            
        Returns:
            Nearest text or None
        """
        x, y = point
        h, w = image.shape[:2]
        
        # Define search region
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)
        
        # Extract region
        roi = image[y1:y2, x1:x2]
        
        # Extract text
        result = self.extract_text(roi, preprocess=True)
        
        if result['confidence'] >= self.confidence_threshold:
            return result['full_text']
        
        return None
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing optimized for OCR."""
        # Convert to grayscale
        gray = ensure_grayscale(image)
        
        # Denoise
        denoised = denoise_image(gray)
        
        # Remove lines (useful for forms)
        no_lines = remove_lines(denoised)
        
        # Apply adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            no_lines, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return adaptive
    
    def _get_ocr_data(self, image: np.ndarray) -> dict:
        """Get raw OCR data with bounding boxes."""
        custom_config = f'--oem {self.oem} --psm {self.psm}'
        
        try:
            data = pytesseract.image_to_data(
                image, 
                lang=self.language,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            return data
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return {'text': [], 'conf': [], 'left': [], 'top': [], 'width': [], 'height': []}
    
    def _extract_text_blocks(self, ocr_data: dict) -> List[Dict[str, Any]]:
        """Extract text blocks from OCR data."""
        text_blocks = []
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = str(ocr_data['text'][i]).strip()
            conf = int(ocr_data['conf'][i])
            
            # Skip empty text or low confidence
            if not text or conf < self.confidence_threshold * 100:
                continue
            
            block = {
                'text': text,
                'confidence': conf / 100.0,
                'bbox': {
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i]
                },
                'level': ocr_data.get('level', [0] * n_boxes)[i],
                'page_num': ocr_data.get('page_num', [1] * n_boxes)[i],
                'block_num': ocr_data.get('block_num', [0] * n_boxes)[i],
                'par_num': ocr_data.get('par_num', [0] * n_boxes)[i],
                'line_num': ocr_data.get('line_num', [0] * n_boxes)[i],
                'word_num': ocr_data.get('word_num', [0] * n_boxes)[i]
            }
            
            text_blocks.append(block)
        
        # Merge nearby text blocks
        text_blocks = self._merge_text_blocks(text_blocks)
        
        return text_blocks
    
    def _merge_text_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge nearby text blocks into lines."""
        if not blocks:
            return []
        
        # Group by line number
        lines = {}
        for block in blocks:
            key = (block['block_num'], block['par_num'], block['line_num'])
            if key not in lines:
                lines[key] = []
            lines[key].append(block)
        
        # Merge blocks in each line
        merged_blocks = []
        for line_blocks in lines.values():
            # Sort by x position
            line_blocks.sort(key=lambda b: b['bbox']['x'])
            
            # Merge text
            merged_text = ' '.join([b['text'] for b in line_blocks])
            avg_confidence = sum([b['confidence'] for b in line_blocks]) / len(line_blocks)
            
            # Calculate combined bbox
            min_x = min([b['bbox']['x'] for b in line_blocks])
            min_y = min([b['bbox']['y'] for b in line_blocks])
            max_x = max([b['bbox']['x'] + b['bbox']['width'] for b in line_blocks])
            max_y = max([b['bbox']['y'] + b['bbox']['height'] for b in line_blocks])
            
            merged_block = {
                'text': merged_text,
                'confidence': avg_confidence,
                'bbox': {
                    'x': min_x,
                    'y': min_y,
                    'width': max_x - min_x,
                    'height': max_y - min_y
                },
                'level': line_blocks[0]['level'],
                'page_num': line_blocks[0]['page_num'],
                'block_num': line_blocks[0]['block_num'],
                'par_num': line_blocks[0]['par_num'],
                'line_num': line_blocks[0]['line_num'],
                'word_count': len(line_blocks)
            }
            
            merged_blocks.append(merged_block)
        
        # Sort by position
        merged_blocks.sort(key=lambda b: (b['bbox']['y'], b['bbox']['x']))
        
        return merged_blocks
    
    def detect_form_fields(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect form fields (labels) in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected form fields
        """
        # Extract all text
        result = self.extract_text(image)
        
        form_fields = []
        
        # Common patterns for form fields
        patterns = [
            r'.*:$',  # Text ending with colon
            r'^\[.*\]$',  # Text in brackets
            r'^.*\?$',  # Questions
            r'^\d+\.',  # Numbered items
            r'^[A-Z][a-z]+.*:',  # Capitalized labels with colon
        ]
        
        for block in result['text_blocks']:
            text = block['text'].strip()
            
            # Check if text matches form field patterns
            is_field = False
            for pattern in patterns:
                if re.match(pattern, text):
                    is_field = True
                    break
            
            # Also check for common form keywords
            keywords = ['name', 'date', 'address', 'phone', 'email', 'yes', 'no', 
                       'signature', 'amount', 'number', 'select', 'choose']
            
            if not is_field:
                text_lower = text.lower()
                for keyword in keywords:
                    if keyword in text_lower:
                        is_field = True
                        break
            
            if is_field:
                form_fields.append({
                    'label': text,
                    'bbox': block['bbox'],
                    'confidence': block['confidence'],
                    'type': self._infer_field_type(text)
                })
        
        return form_fields
    
    def _infer_field_type(self, text: str) -> str:
        """Infer the type of form field from text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['yes', 'no', 'true', 'false']):
            return 'checkbox'
        elif any(word in text_lower for word in ['date', 'dob', 'birth']):
            return 'date'
        elif any(word in text_lower for word in ['signature', 'sign']):
            return 'signature'
        elif any(word in text_lower for word in ['email', 'e-mail']):
            return 'email'
        elif any(word in text_lower for word in ['phone', 'tel', 'mobile']):
            return 'phone'
        elif any(word in text_lower for word in ['amount', 'price', 'cost', '$']):
            return 'currency'
        elif any(word in text_lower for word in ['select', 'choose', 'option']):
            return 'selection'
        else:
            return 'text'
    
    def enhance_ocr_accuracy(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Apply multiple OCR strategies to enhance accuracy.
        
        Args:
            image: Input image
            
        Returns:
            Best OCR result
        """
        strategies = [
            {'psm': 6, 'preprocess': True},   # Uniform block
            {'psm': 11, 'preprocess': True},  # Sparse text
            {'psm': 3, 'preprocess': True},   # Automatic
            {'psm': 4, 'preprocess': False},  # Single column
        ]
        
        best_result = None
        best_confidence = 0
        
        for strategy in strategies:
            # Temporarily update config
            original_psm = self.psm
            self.psm = strategy['psm']
            
            try:
                result = self.extract_text(image, preprocess=strategy['preprocess'])
                
                if result['confidence'] > best_confidence:
                    best_confidence = result['confidence']
                    best_result = result
                    best_result['strategy'] = strategy
                    
            except Exception as e:
                logger.warning(f"OCR strategy {strategy} failed: {e}")
            finally:
                self.psm = original_psm
        
        return best_result or {'full_text': '', 'text_blocks': [], 'confidence': 0}