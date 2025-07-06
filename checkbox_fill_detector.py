import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

class CheckboxFillDetector:
    """
    Detect whether checkboxes are filled/ticked using multiple analysis methods
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
    def extract_checkbox_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                              padding: int = 2) -> np.ndarray:
        """Extract checkbox region from image with optional padding"""
        x, y, w, h = bbox
        
        # Add padding but keep within image bounds
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        return image[y_start:y_end, x_start:x_end]
    
    def analyze_pixel_density(self, checkbox_region: np.ndarray, 
                            threshold: int = 127) -> Dict[str, float]:
        """Analyze pixel density in checkbox region"""
        # Convert to grayscale if needed
        if len(checkbox_region.shape) == 3:
            gray = cv2.cvtColor(checkbox_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = checkbox_region
        
        # Calculate pixel statistics
        total_pixels = gray.size
        dark_pixels = np.sum(gray < threshold)
        light_pixels = total_pixels - dark_pixels
        
        # Calculate density metrics
        dark_ratio = dark_pixels / total_pixels
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        return {
            'dark_ratio': dark_ratio,
            'light_ratio': light_pixels / total_pixels,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'total_pixels': total_pixels
        }
    
    def analyze_contours(self, checkbox_region: np.ndarray) -> Dict[str, Any]:
        """Analyze contours within checkbox region"""
        # Convert to grayscale if needed
        if len(checkbox_region.shape) == 3:
            gray = cv2.cvtColor(checkbox_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = checkbox_region
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours
        if contours:
            # Find largest contour (likely the check mark or fill)
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Calculate contour properties
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            return {
                'contour_count': len(contours),
                'largest_contour_area': contour_area,
                'hull_area': hull_area,
                'contour_area_ratio': contour_area / (checkbox_region.shape[0] * checkbox_region.shape[1]),
                'convexity': contour_area / hull_area if hull_area > 0 else 0
            }
        else:
            return {
                'contour_count': 0,
                'largest_contour_area': 0,
                'hull_area': 0,
                'contour_area_ratio': 0,
                'convexity': 0
            }
    
    def analyze_edges(self, checkbox_region: np.ndarray) -> Dict[str, float]:
        """Analyze edge patterns in checkbox region"""
        # Convert to grayscale if needed
        if len(checkbox_region.shape) == 3:
            gray = cv2.cvtColor(checkbox_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = checkbox_region
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge statistics
        total_pixels = edges.size
        edge_pixels = np.sum(edges > 0)
        
        return {
            'edge_density': edge_pixels / total_pixels,
            'edge_pixels': edge_pixels
        }
    
    def detect_cross_pattern(self, checkbox_region: np.ndarray) -> Dict[str, float]:
        """Detect X/cross pattern in checkbox using line detection"""
        # Convert to grayscale if needed
        if len(checkbox_region.shape) == 3:
            gray = cv2.cvtColor(checkbox_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = checkbox_region
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, 
                               minLineLength=5, maxLineGap=3)
        
        if lines is not None:
            # Analyze line angles to detect crossing patterns
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            # Check for crossing lines (roughly perpendicular)
            diagonal_lines = 0
            for angle in angles:
                if 30 <= abs(angle) <= 60 or 120 <= abs(angle) <= 150:
                    diagonal_lines += 1
            
            return {
                'line_count': len(lines),
                'diagonal_lines': diagonal_lines,
                'cross_pattern_score': diagonal_lines / len(lines) if lines is not None else 0
            }
        else:
            return {
                'line_count': 0,
                'diagonal_lines': 0,
                'cross_pattern_score': 0
            }
    
    def classify_checkbox_fill(self, checkbox_region: np.ndarray) -> Dict[str, Any]:
        """Classify checkbox as filled/empty using multiple methods"""
        
        # Analyze different aspects
        density = self.analyze_pixel_density(checkbox_region)
        contours = self.analyze_contours(checkbox_region)
        edges = self.analyze_edges(checkbox_region)
        cross = self.detect_cross_pattern(checkbox_region)
        
        # Combine metrics for classification
        features = {
            'dark_ratio': density['dark_ratio'],
            'mean_intensity': density['mean_intensity'],
            'std_intensity': density['std_intensity'],
            'contour_area_ratio': contours['contour_area_ratio'],
            'contour_count': contours['contour_count'],
            'edge_density': edges['edge_density'],
            'cross_pattern_score': cross['cross_pattern_score'],
            'line_count': cross['line_count']
        }
        
        # Classification rules (these thresholds may need tuning)
        filled_score = 0
        confidence = 0
        
        # Rule 1: High dark pixel ratio suggests filling
        if density['dark_ratio'] > 0.15:  # More than 15% dark pixels
            filled_score += 0.3
            confidence += 0.2
        
        # Rule 2: Significant contour area suggests marks
        if contours['contour_area_ratio'] > 0.1:  # Contour covers >10% of area
            filled_score += 0.25
            confidence += 0.2
        
        # Rule 3: Cross pattern detection
        if cross['cross_pattern_score'] > 0.3:  # Strong cross pattern
            filled_score += 0.3
            confidence += 0.3
        
        # Rule 4: Multiple small contours might indicate checkmarks
        if contours['contour_count'] > 1:
            filled_score += 0.1
            confidence += 0.1
        
        # Rule 5: High edge density suggests drawn content
        if edges['edge_density'] > 0.05:
            filled_score += 0.05
            confidence += 0.2
        
        # Final classification
        is_filled = filled_score > 0.4  # Threshold for "filled"
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        return {
            'is_filled': is_filled,
            'filled_score': filled_score,
            'confidence': confidence,
            'features': features,
            'analysis': {
                'density': density,
                'contours': contours,
                'edges': edges,
                'cross': cross
            }
        }
    
    def analyze_checkboxes(self, image: np.ndarray, 
                          checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze fill status for all detected checkboxes"""
        results = []
        
        for checkbox in checkboxes:
            # Extract checkbox region
            bbox = checkbox['bbox']
            region = self.extract_checkbox_region(image, bbox)
            
            # Analyze fill status
            fill_analysis = self.classify_checkbox_fill(region)
            
            # Combine with original checkbox data
            result = {
                **checkbox,
                'fill_analysis': fill_analysis,
                'is_filled': fill_analysis['is_filled'],
                'fill_confidence': fill_analysis['confidence']
            }
            
            results.append(result)
            
            if self.debug:
                self.logger.info(f"Checkbox {checkbox['id']}: "
                               f"filled={fill_analysis['is_filled']}, "
                               f"score={fill_analysis['filled_score']:.3f}, "
                               f"confidence={fill_analysis['confidence']:.3f}")
        
        return results
    
    def visualize_fill_analysis(self, image: np.ndarray, 
                              analyzed_checkboxes: List[Dict[str, Any]], 
                              output_path: str = "checkbox_fill_analysis.png"):
        """Visualize checkbox fill analysis results"""
        vis_image = image.copy()
        
        for checkbox in analyzed_checkboxes:
            x, y, w, h = checkbox['bbox']
            is_filled = checkbox['is_filled']
            confidence = checkbox['fill_confidence']
            
            # Choose color based on fill status
            color = (0, 255, 0) if is_filled else (255, 0, 0)  # Green for filled, red for empty
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"ID:{checkbox['id']} {'✓' if is_filled else '✗'} ({confidence:.2f})"
            cv2.putText(vis_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Convert RGB to BGR for saving
        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_bgr)
        
        print(f"Fill analysis visualization saved to: {output_path}")
        return vis_image

# Example usage
if __name__ == "__main__":
    # This would typically be used with the EnhancedCheckboxDetector
    from enhanced_checkbox_detector import EnhancedCheckboxDetector
    
    # Initialize detectors
    detector = EnhancedCheckboxDetector(dpi=200)
    fill_detector = CheckboxFillDetector(debug=True)
    
    # Load and analyze
    pdf_path = "output.pdf"  # Replace with your file
    image = detector.load_pdf_page(pdf_path)
    
    # Detect checkboxes
    checkboxes = detector.detect_checkboxes(image)
    
    # Analyze fill status
    analyzed_checkboxes = fill_detector.analyze_checkboxes(image, checkboxes)
    
    # Visualize results
    fill_detector.visualize_fill_analysis(image, analyzed_checkboxes)
    
    # Print summary
    filled_count = sum(1 for cb in analyzed_checkboxes if cb['is_filled'])
    print(f"\nSummary: {filled_count}/{len(analyzed_checkboxes)} checkboxes are filled")