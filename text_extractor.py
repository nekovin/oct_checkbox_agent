import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any, Tuple, Optional
import logging
import re

class TextExtractor:
    """
    Extract text from images using Tesseract OCR with bounding box information
    """
    
    def __init__(self, language: str = 'eng', debug: bool = False):
        self.language = language
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Test Tesseract installation
        try:
            pytesseract.get_tesseract_version()
            self.logger.info("Tesseract OCR is available")
        except Exception as e:
            self.logger.error(f"Tesseract OCR not available: {e}")
            raise
    
    def preprocess_for_ocr(self, image: np.ndarray, 
                          enhance_contrast: bool = True,
                          denoise: bool = True) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast
        if enhance_contrast:
            gray = cv2.equalizeHist(gray)
        
        # Denoise
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray)
        
        # Apply thresholding for better text recognition
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def extract_text_with_boxes(self, image: np.ndarray, 
                               config: str = '--psm 6') -> List[Dict[str, Any]]:
        """
        Extract text with bounding box information
        PSM modes:
        - 6: Uniform block of text (default)
        - 11: Sparse text
        - 12: Sparse text with OSD
        - 13: Raw line (treat as single text line)
        """
        
        # Preprocess image
        processed = self.preprocess_for_ocr(image)
        
        # Extract text data with boxes
        data = pytesseract.image_to_data(
            processed, 
            lang=self.language, 
            config=config, 
            output_type=pytesseract.Output.DICT
        )
        
        # Process results
        text_boxes = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            confidence = int(data['conf'][i])
            
            # Filter out low confidence and empty text
            if confidence > 0 and text:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                text_box = {
                    'text': text,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'confidence': confidence,
                    'level': data['level'][i],  # Word/line/paragraph level
                    'block_num': data['block_num'][i],
                    'par_num': data['par_num'][i],
                    'line_num': data['line_num'][i],
                    'word_num': data['word_num'][i]
                }
                text_boxes.append(text_box)
        
        self.logger.info(f"Extracted {len(text_boxes)} text elements")
        return text_boxes
    
    def extract_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text in larger regions (lines/paragraphs)"""
        
        # Use PSM 6 for block-based extraction
        text_data = self.extract_text_with_boxes(image, config='--psm 6')
        
        # Group by lines and blocks
        lines = {}
        for item in text_data:
            line_key = (item['block_num'], item['par_num'], item['line_num'])
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(item)
        
        # Combine words into lines
        text_regions = []
        for line_key, words in lines.items():
            if not words:
                continue
            
            # Sort words by position
            words.sort(key=lambda w: w['bbox'][0])  # Sort by x coordinate
            
            # Combine text
            combined_text = ' '.join(word['text'] for word in words)
            
            # Calculate bounding box for entire line
            min_x = min(w['bbox'][0] for w in words)
            min_y = min(w['bbox'][1] for w in words)
            max_x = max(w['bbox'][0] + w['bbox'][2] for w in words)
            max_y = max(w['bbox'][1] + w['bbox'][3] for w in words)
            
            text_region = {
                'text': combined_text,
                'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                'center': ((min_x + max_x)//2, (min_y + max_y)//2),
                'confidence': np.mean([w['confidence'] for w in words]),
                'word_count': len(words),
                'block_num': line_key[0],
                'par_num': line_key[1],
                'line_num': line_key[2],
                'words': words
            }
            text_regions.append(text_region)
        
        # Sort by position (top to bottom, left to right)
        text_regions.sort(key=lambda r: (r['center'][1], r['center'][0]))
        
        return text_regions
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\.\,\?\!\-\(\)\[\]]', '', text)
        
        return text
    
    def filter_checkbox_labels(self, text_regions: List[Dict[str, Any]], 
                             min_confidence: float = 50,
                             min_length: int = 2) -> List[Dict[str, Any]]:
        """Filter text regions that are likely checkbox labels"""
        
        labels = []
        for region in text_regions:
            text = self.clean_text(region['text'])
            
            # Filter by confidence and length
            if (region['confidence'] >= min_confidence and 
                len(text) >= min_length):
                
                # Update with cleaned text
                region['cleaned_text'] = text
                labels.append(region)
        
        return labels
    
    def visualize_text_extraction(self, image: np.ndarray, 
                                text_regions: List[Dict[str, Any]], 
                                output_path: str = "text_extraction.png"):
        """Visualize extracted text regions"""
        vis_image = image.copy()
        
        for i, region in enumerate(text_regions):
            x, y, w, h = region['bbox']
            confidence = region['confidence']
            
            # Color based on confidence
            if confidence >= 80:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence >= 50:
                color = (255, 255, 0)  # Yellow for medium confidence
            else:
                color = (255, 0, 0)  # Red for low confidence
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Add text label (truncated if too long)
            text = region.get('cleaned_text', region['text'])
            if len(text) > 20:
                text = text[:20] + "..."
            
            cv2.putText(vis_image, f"{i}: {text}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Convert RGB to BGR for saving
        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_bgr)
        
        print(f"Text extraction visualization saved to: {output_path}")
        return vis_image
    
    def extract_form_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract form structure with different text types"""
        
        # Extract all text
        all_text = self.extract_text_with_boxes(image)
        text_regions = self.extract_text_regions(image)
        
        # Categorize text
        questions = []
        options = []
        other_text = []
        
        for region in text_regions:
            text = region.get('cleaned_text', region['text'])
            
            # Heuristics for categorizing text
            if '?' in text or text.lower().startswith(('what', 'how', 'when', 'where', 'why', 'which')):
                questions.append(region)
            elif len(text) < 50 and region['confidence'] > 60:  # Short, confident text likely options
                options.append(region)
            else:
                other_text.append(region)
        
        return {
            'all_text': all_text,
            'text_regions': text_regions,
            'questions': questions,
            'options': options,
            'other_text': other_text,
            'statistics': {
                'total_text_elements': len(all_text),
                'total_regions': len(text_regions),
                'question_count': len(questions),
                'option_count': len(options)
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = TextExtractor(debug=True)
    
    # Load image (this would typically come from EnhancedCheckboxDetector)
    from enhanced_checkbox_detector import EnhancedCheckboxDetector
    detector = EnhancedCheckboxDetector()
    
    pdf_path = "output.pdf"  # Replace with your file
    image = detector.load_pdf_page(pdf_path)
    
    # Extract text
    form_structure = extractor.extract_form_structure(image)
    
    # Visualize
    extractor.visualize_text_extraction(image, form_structure['text_regions'])
    
    # Print results
    print(f"Form Structure Analysis:")
    print(f"  Total text regions: {form_structure['statistics']['total_regions']}")
    print(f"  Questions: {form_structure['statistics']['question_count']}")
    print(f"  Options: {form_structure['statistics']['option_count']}")
    
    print("\nExtracted Questions:")
    for q in form_structure['questions']:
        print(f"  - {q.get('cleaned_text', q['text'])}")
    
    print("\nExtracted Options:")
    for opt in form_structure['options'][:10]:  # Show first 10
        print(f"  - {opt.get('cleaned_text', opt['text'])}")