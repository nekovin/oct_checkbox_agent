"""Checkbox detection module using computer vision techniques."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from loguru import logger
from dataclasses import dataclass
from enum import Enum

from .config import get_config
from .utils import (
    timing_decorator, ensure_grayscale, apply_threshold,
    calculate_fill_ratio, is_square_like, extract_roi,
    save_debug_image, find_nearest_text
)


class CheckboxState(Enum):
    """Checkbox state enumeration."""
    UNCHECKED = "unchecked"
    CHECKED = "checked"
    UNKNOWN = "unknown"


@dataclass
class Checkbox:
    """Represents a detected checkbox."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    state: CheckboxState
    confidence: float
    fill_ratio: float
    associated_text: Optional[str] = None
    region_index: Optional[int] = None


class CheckboxDetector:
    """Detects and analyzes checkboxes in images."""
    
    def __init__(self, config=None):
        """Initialize checkbox detector with configuration."""
        self.config = config or get_config()
        self.min_size = self.config.checkbox_detection.min_checkbox_size
        self.max_size = self.config.checkbox_detection.max_checkbox_size
        self.aspect_ratio_threshold = self.config.checkbox_detection.checkbox_aspect_ratio_threshold
        self.binary_threshold = self.config.checkbox_detection.binary_threshold
        self.contour_area_threshold = self.config.checkbox_detection.contour_area_threshold
        self.fill_threshold = self.config.checkbox_detection.checkbox_fill_threshold
        
    @timing_decorator
    def detect_checkboxes(self, image: np.ndarray, 
                         text_blocks: Optional[List[dict]] = None) -> List[Checkbox]:
        """
        Detect checkboxes in image.
        
        Args:
            image: Input image
            text_blocks: Optional text blocks for associating labels
            
        Returns:
            List of detected checkboxes
        """
        # Preprocess image
        processed = self._preprocess_for_detection(image)
        
        # Detect checkbox candidates
        candidates = self._find_checkbox_candidates(processed)
        
        # Analyze each candidate
        checkboxes = []
        for bbox in candidates:
            checkbox = self._analyze_checkbox(image, bbox)
            
            # Associate with text if provided
            if text_blocks and checkbox:
                checkbox.associated_text = self._find_associated_text(
                    checkbox.bbox, text_blocks
                )
            
            if checkbox:
                checkboxes.append(checkbox)
        
        # Remove duplicates
        checkboxes = self._remove_duplicate_checkboxes(checkboxes)
        
        logger.info(f"Detected {len(checkboxes)} checkboxes")
        return checkboxes
    
    def detect_checkboxes_in_regions(self, image: np.ndarray, 
                                   regions: List[Tuple[int, int, int, int]],
                                   text_blocks: Optional[List[dict]] = None) -> Dict[int, List[Checkbox]]:
        """
        Detect checkboxes within specific regions.
        
        Args:
            image: Input image
            regions: List of regions (x, y, w, h)
            text_blocks: Optional text blocks for associating labels
            
        Returns:
            Dictionary mapping region index to checkboxes
        """
        results = {}
        
        for idx, (x, y, w, h) in enumerate(regions):
            # Extract region
            roi = image[y:y+h, x:x+w]
            
            # Detect checkboxes in region
            region_checkboxes = self.detect_checkboxes(roi, text_blocks)
            
            # Adjust coordinates to global image space
            for checkbox in region_checkboxes:
                cx, cy, cw, ch = checkbox.bbox
                checkbox.bbox = (cx + x, cy + y, cw, ch)
                checkbox.region_index = idx
            
            results[idx] = region_checkboxes
        
        return results
    
    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for checkbox detection."""
        # Convert to grayscale
        gray = ensure_grayscale(image)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, binary = cv2.threshold(blurred, self.binary_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _find_checkbox_candidates(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find potential checkbox regions."""
        # Find contours
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        candidates = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.contour_area_threshold:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if (self.min_size <= w <= self.max_size and 
                self.min_size <= h <= self.max_size):
                
                # Check if square-like
                if is_square_like(contour, self.aspect_ratio_threshold):
                    candidates.append((x, y, w, h))
        
        # Also try template matching for common checkbox patterns
        template_candidates = self._template_matching_detection(binary_image)
        candidates.extend(template_candidates)
        
        # Remove duplicates
        candidates = self._merge_overlapping_boxes(candidates)
        
        return candidates
    
    def _template_matching_detection(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Use template matching to find checkboxes."""
        candidates = []
        
        # Generate checkbox templates
        templates = self._generate_checkbox_templates()
        
        for template in templates:
            # Perform template matching
            result = cv2.matchTemplate(binary_image, template, cv2.TM_CCOEFF_NORMED)
            
            # Find locations with high correlation
            threshold = 0.7
            locations = np.where(result >= threshold)
            
            # Convert to bounding boxes
            h, w = template.shape
            for pt in zip(*locations[::-1]):
                candidates.append((pt[0], pt[1], w, h))
        
        return candidates
    
    def _generate_checkbox_templates(self) -> List[np.ndarray]:
        """Generate common checkbox templates."""
        templates = []
        
        # Square checkbox templates
        for size in [15, 20, 25, 30]:
            # Empty square
            template = np.ones((size, size), dtype=np.uint8) * 255
            cv2.rectangle(template, (2, 2), (size-3, size-3), 0, 2)
            templates.append(template)
            
            # Checked square (with X)
            template_checked = template.copy()
            cv2.line(template_checked, (4, 4), (size-5, size-5), 0, 2)
            cv2.line(template_checked, (size-5, 4), (4, size-5), 0, 2)
            templates.append(template_checked)
            
            # Checked square (filled)
            template_filled = template.copy()
            cv2.rectangle(template_filled, (5, 5), (size-6, size-6), 0, -1)
            templates.append(template_filled)
        
        return templates
    
    def _analyze_checkbox(self, image: np.ndarray, 
                         bbox: Tuple[int, int, int, int]) -> Optional[Checkbox]:
        """Analyze a checkbox candidate."""
        x, y, w, h = bbox
        
        # Extract checkbox region
        roi = extract_roi(image, bbox, padding=2)
        
        if roi.size == 0:
            return None
        
        # Calculate fill ratio
        fill_ratio = calculate_fill_ratio(roi)
        
        # Determine state
        if fill_ratio > self.fill_threshold:
            state = CheckboxState.CHECKED
            confidence = min(fill_ratio / self.fill_threshold, 1.0)
        elif fill_ratio < self.fill_threshold * 0.3:
            state = CheckboxState.UNCHECKED
            confidence = 1.0 - (fill_ratio / (self.fill_threshold * 0.3))
        else:
            state = CheckboxState.UNKNOWN
            confidence = 0.5
        
        # Additional validation using edge detection
        edges = cv2.Canny(roi, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Adjust confidence based on edge presence
        if edge_ratio > 0.1:  # Good edge definition
            confidence *= 1.1
        else:
            confidence *= 0.9
        
        confidence = min(max(confidence, 0.0), 1.0)
        
        return Checkbox(
            bbox=bbox,
            state=state,
            confidence=confidence,
            fill_ratio=fill_ratio
        )
    
    def _find_associated_text(self, checkbox_bbox: Tuple[int, int, int, int], 
                            text_blocks: List[dict]) -> Optional[str]:
        """Find text associated with a checkbox."""
        # Convert text blocks to expected format
        formatted_blocks = []
        for block in text_blocks:
            if 'bbox' in block:
                formatted_blocks.append({
                    'text': block.get('text', ''),
                    'left': block['bbox']['x'],
                    'top': block['bbox']['y'],
                    'width': block['bbox']['width'],
                    'height': block['bbox']['height']
                })
        
        return find_nearest_text(checkbox_bbox, formatted_blocks)
    
    def _merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]], 
                               overlap_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes."""
        if not boxes:
            return []
        
        # Sort boxes by area (larger first)
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        
        merged = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            
            x1, y1, w1, h1 = box1
            
            # Check for overlaps with other boxes
            group = [box1]
            for j, box2 in enumerate(boxes[i+1:], i+1):
                if j in used:
                    continue
                
                x2, y2, w2, h2 = box2
                
                # Calculate intersection
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                
                if xi2 > xi1 and yi2 > yi1:
                    # Calculate overlap ratio
                    intersection_area = (xi2 - xi1) * (yi2 - yi1)
                    box2_area = w2 * h2
                    overlap_ratio = intersection_area / box2_area
                    
                    if overlap_ratio > overlap_threshold:
                        group.append(box2)
                        used.add(j)
            
            # Merge group
            if len(group) > 1:
                min_x = min([b[0] for b in group])
                min_y = min([b[1] for b in group])
                max_x = max([b[0] + b[2] for b in group])
                max_y = max([b[1] + b[3] for b in group])
                merged.append((min_x, min_y, max_x - min_x, max_y - min_y))
            else:
                merged.append(box1)
        
        return merged
    
    def _remove_duplicate_checkboxes(self, checkboxes: List[Checkbox]) -> List[Checkbox]:
        """Remove duplicate checkboxes based on location."""
        if not checkboxes:
            return []
        
        # Sort by confidence (higher first)
        checkboxes = sorted(checkboxes, key=lambda c: c.confidence, reverse=True)
        
        unique = []
        
        for checkbox in checkboxes:
            x1, y1, w1, h1 = checkbox.bbox
            
            # Check if this overlaps with any existing checkbox
            is_duplicate = False
            for existing in unique:
                x2, y2, w2, h2 = existing.bbox
                
                # Calculate center distance
                c1_x, c1_y = x1 + w1//2, y1 + h1//2
                c2_x, c2_y = x2 + w2//2, y2 + h2//2
                distance = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
                
                # If centers are very close, consider as duplicate
                if distance < min(w1, h1, w2, h2) * 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(checkbox)
        
        return unique
    
    def visualize_detections(self, image: np.ndarray, 
                           checkboxes: List[Checkbox]) -> np.ndarray:
        """Visualize detected checkboxes on image."""
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        for checkbox in checkboxes:
            x, y, w, h = checkbox.bbox
            
            # Color based on state
            if checkbox.state == CheckboxState.CHECKED:
                color = (0, 255, 0)  # Green
            elif checkbox.state == CheckboxState.UNCHECKED:
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 0, 255)  # Red
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Add state text
            text = f"{checkbox.state.value} ({checkbox.confidence:.2f})"
            cv2.putText(vis_image, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add associated text if available
            if checkbox.associated_text:
                cv2.putText(vis_image, checkbox.associated_text[:30], 
                           (x + w + 5, y + h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return vis_image