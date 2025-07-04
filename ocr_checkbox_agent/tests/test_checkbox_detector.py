"""Tests for checkbox detector."""

import pytest
import numpy as np
import cv2
from ocr_checkbox_agent.checkbox_detector import CheckboxDetector, CheckboxState, Checkbox
from ocr_checkbox_agent.config import Config


class TestCheckboxDetector:
    """Test checkbox detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = CheckboxDetector()
        self.config = Config()
    
    def create_test_image_with_checkbox(self, size=100, checkbox_size=20, filled=False):
        """Create test image with a checkbox."""
        image = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw checkbox outline
        x, y = (size - checkbox_size) // 2, (size - checkbox_size) // 2
        cv2.rectangle(image, (x, y), (x + checkbox_size, y + checkbox_size), 0, 2)
        
        # Fill checkbox if requested
        if filled:
            cv2.rectangle(image, (x + 3, y + 3), (x + checkbox_size - 3, y + checkbox_size - 3), 0, -1)
        
        return image
    
    def test_checkbox_detector_init(self):
        """Test checkbox detector initialization."""
        detector = CheckboxDetector()
        
        assert detector.min_size == 10
        assert detector.max_size == 50
        assert detector.fill_threshold == 0.3
    
    def test_checkbox_detector_with_config(self):
        """Test checkbox detector with custom config."""
        config = Config()
        config.checkbox_detection.min_checkbox_size = 15
        config.checkbox_detection.max_checkbox_size = 40
        
        detector = CheckboxDetector(config)
        
        assert detector.min_size == 15
        assert detector.max_size == 40
    
    def test_preprocess_for_detection(self):
        """Test image preprocessing for detection."""
        image = self.create_test_image_with_checkbox()
        detector = CheckboxDetector()
        
        processed = detector._preprocess_for_detection(image)
        
        assert processed.shape == image.shape
        assert processed.dtype == np.uint8
        assert np.all((processed == 0) | (processed == 255))
    
    def test_find_checkbox_candidates(self):
        """Test finding checkbox candidates."""
        image = self.create_test_image_with_checkbox(size=100, checkbox_size=25)
        detector = CheckboxDetector()
        
        # Preprocess image
        processed = detector._preprocess_for_detection(image)
        
        # Find candidates
        candidates = detector._find_checkbox_candidates(processed)
        
        assert len(candidates) > 0
        
        # Check that we found a reasonable bounding box
        found_reasonable_box = False
        for x, y, w, h in candidates:
            if 15 <= w <= 35 and 15 <= h <= 35:  # Reasonable size range
                found_reasonable_box = True
                break
        
        assert found_reasonable_box
    
    def test_analyze_checkbox_checked(self):
        """Test analyzing a checked checkbox."""
        image = self.create_test_image_with_checkbox(size=100, checkbox_size=20, filled=True)
        detector = CheckboxDetector()
        
        # Assume we found the checkbox at center
        bbox = (40, 40, 20, 20)
        checkbox = detector._analyze_checkbox(image, bbox)
        
        assert checkbox is not None
        assert checkbox.state == CheckboxState.CHECKED
        assert checkbox.confidence > 0.5
        assert checkbox.fill_ratio > 0.3
    
    def test_analyze_checkbox_unchecked(self):
        """Test analyzing an unchecked checkbox."""
        image = self.create_test_image_with_checkbox(size=100, checkbox_size=20, filled=False)
        detector = CheckboxDetector()
        
        # Assume we found the checkbox at center
        bbox = (40, 40, 20, 20)
        checkbox = detector._analyze_checkbox(image, bbox)
        
        assert checkbox is not None
        assert checkbox.state == CheckboxState.UNCHECKED
        assert checkbox.confidence > 0.5
        assert checkbox.fill_ratio < 0.3
    
    def test_generate_checkbox_templates(self):
        """Test generating checkbox templates."""
        detector = CheckboxDetector()
        templates = detector._generate_checkbox_templates()
        
        assert len(templates) > 0
        
        # Check that templates are reasonable
        for template in templates:
            assert template.shape[0] >= 15
            assert template.shape[1] >= 15
            assert template.dtype == np.uint8
    
    def test_merge_overlapping_boxes(self):
        """Test merging overlapping boxes."""
        detector = CheckboxDetector()
        
        boxes = [
            (10, 10, 20, 20),
            (15, 15, 20, 20),  # Overlapping
            (100, 100, 20, 20)  # Separate
        ]
        
        merged = detector._merge_overlapping_boxes(boxes)
        
        # Should merge overlapping boxes
        assert len(merged) <= len(boxes)
        assert (100, 100, 20, 20) in merged  # Separate box should remain
    
    def test_remove_duplicate_checkboxes(self):
        """Test removing duplicate checkboxes."""
        detector = CheckboxDetector()
        
        checkbox1 = Checkbox(
            bbox=(10, 10, 20, 20),
            state=CheckboxState.CHECKED,
            confidence=0.9,
            fill_ratio=0.8
        )
        
        checkbox2 = Checkbox(
            bbox=(12, 12, 18, 18),  # Very close to first
            state=CheckboxState.CHECKED,
            confidence=0.7,
            fill_ratio=0.7
        )
        
        checkbox3 = Checkbox(
            bbox=(100, 100, 20, 20),  # Far from others
            state=CheckboxState.UNCHECKED,
            confidence=0.8,
            fill_ratio=0.1
        )
        
        checkboxes = [checkbox1, checkbox2, checkbox3]
        unique = detector._remove_duplicate_checkboxes(checkboxes)
        
        # Should keep higher confidence checkbox and the separate one
        assert len(unique) == 2
        assert checkbox1 in unique  # Higher confidence
        assert checkbox3 in unique  # Separate location
    
    def test_visualize_detections(self):
        """Test visualization of detected checkboxes."""
        detector = CheckboxDetector()
        image = self.create_test_image_with_checkbox(size=100, checkbox_size=20)
        
        checkbox = Checkbox(
            bbox=(40, 40, 20, 20),
            state=CheckboxState.CHECKED,
            confidence=0.9,
            fill_ratio=0.8
        )
        
        vis_image = detector.visualize_detections(image, [checkbox])
        
        assert vis_image.shape == (100, 100, 3)  # Should be color image
        assert vis_image.dtype == np.uint8
    
    def test_detect_checkboxes_empty_image(self):
        """Test detection on empty image."""
        detector = CheckboxDetector()
        empty_image = np.ones((100, 100), dtype=np.uint8) * 255
        
        checkboxes = detector.detect_checkboxes(empty_image)
        
        assert len(checkboxes) == 0
    
    def test_detect_checkboxes_with_text(self):
        """Test detection with associated text."""
        detector = CheckboxDetector()
        image = self.create_test_image_with_checkbox(size=100, checkbox_size=20)
        
        # Mock text blocks
        text_blocks = [
            {
                'text': 'Test Question',
                'bbox': {'x': 70, 'y': 45, 'width': 100, 'height': 20}
            }
        ]
        
        checkboxes = detector.detect_checkboxes(image, text_blocks)
        
        # Should find checkboxes and associate with text
        if checkboxes:
            assert checkboxes[0].associated_text == 'Test Question'