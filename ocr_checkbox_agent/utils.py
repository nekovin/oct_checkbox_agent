"""Utility functions for OCR Checkbox Agent."""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Union, Any
import numpy as np
import cv2
from PIL import Image
from loguru import logger
import time
from functools import wraps


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Ensure image is in grayscale format."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Apply binary threshold to image."""
    gray = ensure_grayscale(image)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply denoising to image."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance image contrast using CLAHE."""
    gray = ensure_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Deskew image to correct rotation."""
    gray = ensure_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            if -45 <= angle <= 45:
                angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                rows, cols = image.shape[:2]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                return cv2.warpAffine(image, M, (cols, rows), 
                                    flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
    
    return image


def remove_lines(image: np.ndarray) -> np.ndarray:
    """Remove horizontal and vertical lines from image."""
    gray = ensure_grayscale(image)
    binary = apply_threshold(gray)
    
    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, 255, 2)
    
    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, 255, 2)
    
    return binary


def calculate_fill_ratio(roi: np.ndarray) -> float:
    """Calculate the fill ratio of a region of interest."""
    binary = apply_threshold(roi)
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return 0.0
    black_pixels = np.sum(binary == 0)
    return black_pixels / total_pixels


def is_square_like(contour: np.ndarray, tolerance: float = 0.2) -> bool:
    """Check if contour is square-like based on aspect ratio."""
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return False
    aspect_ratio = float(w) / h
    return abs(aspect_ratio - 1.0) <= tolerance


def extract_roi(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                padding: int = 5) -> np.ndarray:
    """Extract region of interest with padding."""
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    
    # Apply padding with bounds checking
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    
    return image[y1:y2, x1:x2]


def save_debug_image(image: np.ndarray, name: str, debug_dir: Path) -> None:
    """Save image for debugging purposes."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    filepath = debug_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(filepath), image)
    logger.debug(f"Debug image saved: {filepath}")


def calculate_confidence_score(ocr_data: dict) -> float:
    """Calculate average confidence score from OCR data."""
    confidences = [conf for conf in ocr_data.get('conf', []) if conf > 0]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def merge_nearby_boxes(boxes: List[Tuple[int, int, int, int]], 
                      threshold: int = 10) -> List[Tuple[int, int, int, int]]:
    """Merge nearby bounding boxes."""
    if not boxes:
        return []
    
    merged = []
    boxes = sorted(boxes, key=lambda x: (x[1], x[0]))  # Sort by y, then x
    
    current_box = list(boxes[0])
    for box in boxes[1:]:
        x, y, w, h = box
        cx, cy, cw, ch = current_box
        
        # Check if boxes are close enough to merge
        if (abs(y - cy) <= threshold and 
            abs(x - (cx + cw)) <= threshold):
            # Merge boxes
            new_x = min(x, cx)
            new_y = min(y, cy)
            new_w = max(x + w, cx + cw) - new_x
            new_h = max(y + h, cy + ch) - new_y
            current_box = [new_x, new_y, new_w, new_h]
        else:
            merged.append(tuple(current_box))
            current_box = list(box)
    
    merged.append(tuple(current_box))
    return merged


def get_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def create_temp_directory() -> Path:
    """Create a temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    logger.debug(f"Created temporary directory: {temp_dir}")
    return temp_dir


def cleanup_temp_directory(temp_dir: Path) -> None:
    """Clean up temporary directory."""
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
        logger.debug(f"Cleaned up temporary directory: {temp_dir}")


def validate_image(image: Union[np.ndarray, Image.Image]) -> bool:
    """Validate if image is valid and not empty."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if image is None or image.size == 0:
        return False
    
    if len(image.shape) not in [2, 3]:
        return False
    
    return True


def normalize_text(text: str) -> str:
    """Normalize text by removing extra spaces and special characters."""
    import re
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\-.,;:?!]', '', text)
    return text.strip()


def find_nearest_text(checkbox_bbox: Tuple[int, int, int, int], 
                     text_boxes: List[dict], 
                     max_distance: int = 100) -> Optional[str]:
    """Find nearest text to a checkbox."""
    cx, cy, cw, ch = checkbox_bbox
    checkbox_center_x = cx + cw // 2
    checkbox_center_y = cy + ch // 2
    
    nearest_text = None
    min_distance = float('inf')
    
    for text_box in text_boxes:
        if not text_box.get('text', '').strip():
            continue
            
        tx = text_box['left']
        ty = text_box['top']
        tw = text_box['width']
        th = text_box['height']
        
        # Calculate distance from checkbox center to text box center
        text_center_x = tx + tw // 2
        text_center_y = ty + th // 2
        
        distance = np.sqrt((checkbox_center_x - text_center_x)**2 + 
                          (checkbox_center_y - text_center_y)**2)
        
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            nearest_text = text_box['text']
    
    return nearest_text