"""Tests for utility functions."""

import pytest
import numpy as np
import cv2
from ocr_checkbox_agent.utils import (
    ensure_grayscale, apply_threshold, calculate_fill_ratio,
    is_square_like, extract_roi, merge_nearby_boxes,
    normalize_text, find_nearest_text
)


class TestImageUtils:
    """Test image utility functions."""
    
    def test_ensure_grayscale(self):
        """Test grayscale conversion."""
        # Test with color image
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = ensure_grayscale(color_image)
        assert len(gray.shape) == 2
        assert gray.shape == (100, 100)
        
        # Test with already grayscale image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = ensure_grayscale(gray_image)
        assert len(result.shape) == 2
        np.testing.assert_array_equal(result, gray_image)
    
    def test_apply_threshold(self):
        """Test binary threshold application."""
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        binary = apply_threshold(image, threshold=127)
        
        assert binary.shape == image.shape
        assert np.all((binary == 0) | (binary == 255))
    
    def test_calculate_fill_ratio(self):
        """Test fill ratio calculation."""
        # Create test image with known fill ratio
        roi = np.ones((20, 20), dtype=np.uint8) * 255
        roi[5:15, 5:15] = 0  # 10x10 black square in 20x20 white square
        
        fill_ratio = calculate_fill_ratio(roi)
        expected_ratio = 100 / 400  # 100 black pixels out of 400 total
        assert abs(fill_ratio - expected_ratio) < 0.01
    
    def test_is_square_like(self):
        """Test square-like contour detection."""
        # Create square contour
        square_points = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.int32)
        square_contour = square_points.reshape(-1, 1, 2)
        
        assert is_square_like(square_contour, tolerance=0.1)
        
        # Create rectangle contour
        rect_points = np.array([[0, 0], [40, 0], [40, 10], [0, 10]], dtype=np.int32)
        rect_contour = rect_points.reshape(-1, 1, 2)
        
        assert not is_square_like(rect_contour, tolerance=0.1)
    
    def test_extract_roi(self):
        """Test ROI extraction."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        bbox = (10, 10, 20, 20)
        
        roi = extract_roi(image, bbox, padding=5)
        
        # With padding, ROI should be larger than bbox
        assert roi.shape[0] >= 20
        assert roi.shape[1] >= 20
        
        # Test boundary conditions
        bbox_edge = (90, 90, 20, 20)
        roi_edge = extract_roi(image, bbox_edge, padding=5)
        assert roi_edge.shape[0] > 0
        assert roi_edge.shape[1] > 0


class TestTextUtils:
    """Test text utility functions."""
    
    def test_normalize_text(self):
        """Test text normalization."""
        text = "  Hello,   World!  \n\t  "
        normalized = normalize_text(text)
        assert normalized == "Hello, World!"
        
        # Test with special characters
        text_special = "Hello@#$%^&*()World"
        normalized_special = normalize_text(text_special)
        assert normalized_special == "HelloWorld"
    
    def test_find_nearest_text(self):
        """Test finding nearest text to checkbox."""
        checkbox_bbox = (50, 50, 20, 20)
        
        text_boxes = [
            {'text': 'Question 1', 'left': 80, 'top': 50, 'width': 100, 'height': 20},
            {'text': 'Question 2', 'left': 80, 'top': 100, 'width': 100, 'height': 20},
            {'text': 'Far away text', 'left': 300, 'top': 300, 'width': 100, 'height': 20}
        ]
        
        nearest = find_nearest_text(checkbox_bbox, text_boxes)
        assert nearest == 'Question 1'
        
        # Test with no nearby text
        nearest_far = find_nearest_text(checkbox_bbox, text_boxes, max_distance=10)
        assert nearest_far is None


class TestBoxUtils:
    """Test bounding box utility functions."""
    
    def test_merge_nearby_boxes(self):
        """Test merging nearby bounding boxes."""
        boxes = [
            (10, 10, 20, 20),
            (15, 15, 20, 20),  # Overlapping
            (100, 100, 20, 20),  # Separate
            (12, 12, 15, 15)   # Overlapping with first
        ]
        
        merged = merge_nearby_boxes(boxes, threshold=10)
        
        # Should merge overlapping boxes
        assert len(merged) < len(boxes)
        assert (100, 100, 20, 20) in merged  # Separate box should remain
    
    def test_merge_nearby_boxes_empty(self):
        """Test merging with empty list."""
        merged = merge_nearby_boxes([])
        assert merged == []
    
    def test_merge_nearby_boxes_single(self):
        """Test merging with single box."""
        boxes = [(10, 10, 20, 20)]
        merged = merge_nearby_boxes(boxes)
        assert merged == boxes