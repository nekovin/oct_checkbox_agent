import cv2
import numpy as np
from pdf2image import convert_from_path
from boxdetect import config
from boxdetect.pipelines import get_boxes
from typing import List, Tuple, Dict, Any
import logging

class EnhancedCheckboxDetector:
    """
    Enhanced checkbox detector combining BoxDetect with OpenCV methods
    for robust checkbox detection and fill analysis
    """
    
    def __init__(self, dpi: int = 200):
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
    def load_pdf_page(self, pdf_path: str, page_num: int = 0) -> np.ndarray:
        """Convert PDF page to image array"""
        pages = convert_from_path(pdf_path, dpi=self.dpi)
        return np.array(pages[page_num])
    
    def configure_boxdetect(self, width_range: Tuple[int, int] = (20, 40), 
                          height_range: Tuple[int, int] = (20, 40)) -> config.PipelinesConfig:
        """Configure BoxDetect for checkbox detection"""
        cfg = config.PipelinesConfig()
        
        # Box size constraints
        cfg.width_range = width_range
        cfg.height_range = height_range
        
        # Multiple scaling factors for robustness
        cfg.scaling_factors = [0.8, 1.0, 1.2, 1.5]
        
        # For square/slightly rectangular checkboxes
        cfg.wh_ratio_range = (0.8, 1.2)
        
        # Individual checkboxes
        cfg.group_size_range = (1, 1)
        
        # Minimal preprocessing for clean PDFs
        cfg.dilation_iterations = 0
        cfg.blur_size = (1, 1)
        cfg.morph_kernels_type = 'rectangles'
        
        return cfg
    
    def detect_boxes_boxdetect(self, image: np.ndarray, cfg: config.PipelinesConfig) -> List[Tuple[int, int, int, int]]:
        """Detect boxes using BoxDetect"""
        try:
            rects, groups, output_image = get_boxes(image, cfg)
            return [(r[0], r[1], r[2], r[3]) for r in rects]  # (x, y, width, height)
        except Exception as e:
            self.logger.error(f"BoxDetect failed: {e}")
            return []
    
    def detect_boxes_opencv(self, image: np.ndarray, 
                          min_area: int = 300, 
                          max_area: int = 2000) -> List[Tuple[int, int, int, int]]:
        """Detect boxes using OpenCV contour analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter by size and aspect ratio
                if (min_area < area < max_area and 
                    0.8 < aspect_ratio < 1.2):
                    boxes.append((x, y, w, h))
        
        return boxes
    
    def combine_detections(self, boxes1: List[Tuple[int, int, int, int]], 
                          boxes2: List[Tuple[int, int, int, int]], 
                          overlap_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Combine detections from multiple methods, removing duplicates"""
        if not boxes1:
            return boxes2
        if not boxes2:
            return boxes1
        
        def boxes_overlap(box1, box2):
            """Check if two boxes overlap significantly"""
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # Calculate intersection
            left = max(x1, x2)
            top = max(y1, y2)
            right = min(x1 + w1, x2 + w2)
            bottom = min(y1 + h1, y2 + h2)
            
            if left < right and top < bottom:
                intersection = (right - left) * (bottom - top)
                union = w1 * h1 + w2 * h2 - intersection
                return intersection / union > overlap_threshold
            return False
        
        # Start with first set
        combined = list(boxes1)
        
        # Add non-overlapping boxes from second set
        for box2 in boxes2:
            if not any(boxes_overlap(box1, box2) for box1 in combined):
                combined.append(box2)
        
        return combined
    
    def detect_checkboxes(self, image: np.ndarray, 
                         width_range: Tuple[int, int] = (20, 40),
                         height_range: Tuple[int, int] = (20, 40)) -> List[Dict[str, Any]]:
        """
        Detect checkboxes using combined methods
        Returns list of checkbox dictionaries with position and metadata
        """
        # Method 1: BoxDetect
        cfg = self.configure_boxdetect(width_range, height_range)
        boxes_bd = self.detect_boxes_boxdetect(image, cfg)
        
        # Method 2: OpenCV
        min_area = width_range[0] * height_range[0]
        max_area = width_range[1] * height_range[1]
        boxes_cv = self.detect_boxes_opencv(image, min_area, max_area)
        
        # Combine results
        all_boxes = self.combine_detections(boxes_bd, boxes_cv)
        
        # Convert to dictionary format with metadata
        checkboxes = []
        for i, (x, y, w, h) in enumerate(all_boxes):
            checkbox = {
                'id': i,
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'area': w * h,
                'aspect_ratio': w / h if h > 0 else 0,
                'method': 'combined'
            }
            checkboxes.append(checkbox)
        
        # Sort by position (top-to-bottom, left-to-right)
        checkboxes.sort(key=lambda cb: (cb['center'][1], cb['center'][0]))
        
        self.logger.info(f"Detected {len(checkboxes)} checkboxes")
        return checkboxes
    
    def visualize_detections(self, image: np.ndarray, 
                           checkboxes: List[Dict[str, Any]], 
                           output_path: str = "detected_checkboxes.png"):
        """Visualize detected checkboxes"""
        vis_image = image.copy()
        
        for checkbox in checkboxes:
            x, y, w, h = checkbox['bbox']
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Add ID label
            cv2.putText(vis_image, str(checkbox['id']), 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Convert RGB to BGR for saving
        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_bgr)
        
        print(f"Visualization saved to: {output_path}")
        return vis_image

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = EnhancedCheckboxDetector(dpi=200)
    
    # Load PDF
    pdf_path = "output.pdf"  # Replace with your file
    image = detector.load_pdf_page(pdf_path, page_num=0)
    
    # Detect checkboxes
    checkboxes = detector.detect_checkboxes(image, width_range=(20, 40), height_range=(20, 40))
    
    # Visualize results
    detector.visualize_detections(image, checkboxes)
    
    # Print results
    print(f"Found {len(checkboxes)} checkboxes:")
    for cb in checkboxes:
        print(f"  ID {cb['id']}: {cb['bbox']} at center {cb['center']}")