import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

class CheckboxLabelCorrelator:
    """
    Correlate checkboxes with their text labels using spatial analysis
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
    
    def calculate_distances(self, checkboxes: List[Dict[str, Any]], 
                          text_regions: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate distance matrix between checkboxes and text regions"""
        
        if not checkboxes or not text_regions:
            return np.array([])
        
        # Get centers of checkboxes and text regions
        checkbox_centers = np.array([cb['center'] for cb in checkboxes])
        text_centers = np.array([tr['center'] for tr in text_regions])
        
        # Calculate Euclidean distances
        distances = cdist(checkbox_centers, text_centers, metric='euclidean')
        
        return distances
    
    def find_nearest_text(self, checkbox: Dict[str, Any], 
                         text_regions: List[Dict[str, Any]], 
                         max_distance: float = 200) -> List[Dict[str, Any]]:
        """Find text regions near a checkbox"""
        
        checkbox_center = np.array(checkbox['center'])
        candidates = []
        
        for text_region in text_regions:
            text_center = np.array(text_region['center'])
            distance = np.linalg.norm(checkbox_center - text_center)
            
            if distance <= max_distance:
                candidates.append({
                    'text_region': text_region,
                    'distance': distance,
                    'direction': self._get_direction(checkbox_center, text_center)
                })
        
        # Sort by distance
        candidates.sort(key=lambda x: x['distance'])
        return candidates
    
    def _get_direction(self, checkbox_center: np.ndarray, 
                      text_center: np.ndarray) -> str:
        """Determine relative direction of text from checkbox"""
        dx = text_center[0] - checkbox_center[0]
        dy = text_center[1] - checkbox_center[1]
        
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'below' if dy > 0 else 'above'
    
    def analyze_form_layout(self, checkboxes: List[Dict[str, Any]], 
                           text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the overall layout of the form"""
        
        if not checkboxes or not text_regions:
            return {'layout_type': 'unknown', 'patterns': []}
        
        # Analyze checkbox positions
        checkbox_centers = np.array([cb['center'] for cb in checkboxes])
        
        # Cluster checkboxes by position to find groups
        if len(checkboxes) > 1:
            clustering = DBSCAN(eps=100, min_samples=2).fit(checkbox_centers)
            clusters = clustering.labels_
        else:
            clusters = [0]
        
        # Analyze text-checkbox relationships
        patterns = []
        
        for i, checkbox in enumerate(checkboxes):
            nearby_text = self.find_nearest_text(checkbox, text_regions, max_distance=150)
            
            if nearby_text:
                closest = nearby_text[0]
                pattern = {
                    'checkbox_id': checkbox['id'],
                    'cluster': clusters[i],
                    'nearest_text': closest['text_region']['text'],
                    'distance': closest['distance'],
                    'direction': closest['direction']
                }
                patterns.append(pattern)
        
        # Determine layout type
        directions = [p['direction'] for p in patterns]
        
        if directions.count('right') > len(directions) * 0.7:
            layout_type = 'checkbox_left_text_right'
        elif directions.count('left') > len(directions) * 0.7:
            layout_type = 'checkbox_right_text_left'
        elif directions.count('below') > len(directions) * 0.5:
            layout_type = 'checkbox_above_text_below'
        elif directions.count('above') > len(directions) * 0.5:
            layout_type = 'checkbox_below_text_above'
        else:
            layout_type = 'mixed'
        
        return {
            'layout_type': layout_type,
            'patterns': patterns,
            'clusters': clusters,
            'dominant_direction': max(set(directions), key=directions.count) if directions else 'unknown'
        }
    
    def correlate_by_proximity(self, checkboxes: List[Dict[str, Any]], 
                              text_regions: List[Dict[str, Any]], 
                              max_distance: float = 150) -> List[Dict[str, Any]]:
        """Correlate checkboxes with text using simple proximity"""
        
        correlations = []
        used_text_indices = set()
        
        # Sort checkboxes by position (top-to-bottom, left-to-right)
        sorted_checkboxes = sorted(checkboxes, key=lambda cb: (cb['center'][1], cb['center'][0]))
        
        for checkbox in sorted_checkboxes:
            nearby_text = self.find_nearest_text(checkbox, text_regions, max_distance)
            
            # Find the closest unused text
            best_match = None
            for candidate in nearby_text:
                text_idx = text_regions.index(candidate['text_region'])
                if text_idx not in used_text_indices:
                    best_match = candidate
                    used_text_indices.add(text_idx)
                    break
            
            correlation = {
                'checkbox': checkbox,
                'matched_text': best_match['text_region'] if best_match else None,
                'confidence': self._calculate_confidence(best_match) if best_match else 0,
                'distance': best_match['distance'] if best_match else float('inf'),
                'direction': best_match['direction'] if best_match else 'unknown'
            }
            
            correlations.append(correlation)
        
        return correlations
    
    def correlate_by_alignment(self, checkboxes: List[Dict[str, Any]], 
                              text_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate checkboxes with text using alignment analysis"""
        
        correlations = []
        
        for checkbox in checkboxes:
            cb_x, cb_y = checkbox['center']
            
            # Find text regions aligned horizontally or vertically
            aligned_text = []
            
            for text_region in text_regions:
                tx, ty = text_region['center']
                
                # Check horizontal alignment (same row)
                if abs(ty - cb_y) < 30:  # Within 30 pixels vertically
                    aligned_text.append({
                        'text_region': text_region,
                        'alignment': 'horizontal',
                        'distance': abs(tx - cb_x),
                        'direction': 'right' if tx > cb_x else 'left'
                    })
                
                # Check vertical alignment (same column)
                elif abs(tx - cb_x) < 30:  # Within 30 pixels horizontally
                    aligned_text.append({
                        'text_region': text_region,
                        'alignment': 'vertical',
                        'distance': abs(ty - cb_y),
                        'direction': 'below' if ty > cb_y else 'above'
                    })
            
            # Choose best match
            if aligned_text:
                # Prefer horizontal alignment, then closest distance
                aligned_text.sort(key=lambda x: (x['alignment'] != 'horizontal', x['distance']))
                best_match = aligned_text[0]
                
                correlation = {
                    'checkbox': checkbox,
                    'matched_text': best_match['text_region'],
                    'confidence': self._calculate_alignment_confidence(best_match),
                    'distance': best_match['distance'],
                    'direction': best_match['direction'],
                    'alignment': best_match['alignment']
                }
            else:
                correlation = {
                    'checkbox': checkbox,
                    'matched_text': None,
                    'confidence': 0,
                    'distance': float('inf'),
                    'direction': 'unknown',
                    'alignment': 'none'
                }
            
            correlations.append(correlation)
        
        return correlations
    
    def correlate_adaptive(self, checkboxes: List[Dict[str, Any]], 
                          text_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adaptive correlation using layout analysis"""
        
        # Analyze form layout
        layout_analysis = self.analyze_form_layout(checkboxes, text_regions)
        layout_type = layout_analysis['layout_type']
        
        self.logger.info(f"Detected layout type: {layout_type}")
        
        # Choose correlation strategy based on layout
        if layout_type in ['checkbox_left_text_right', 'checkbox_right_text_left']:
            # Horizontal layout - use alignment-based correlation
            correlations = self.correlate_by_alignment(checkboxes, text_regions)
        elif layout_type in ['checkbox_above_text_below', 'checkbox_below_text_above']:
            # Vertical layout - use alignment-based correlation
            correlations = self.correlate_by_alignment(checkboxes, text_regions)
        else:
            # Mixed or unknown layout - use proximity-based correlation
            correlations = self.correlate_by_proximity(checkboxes, text_regions)
        
        # Post-process to improve correlations
        correlations = self._post_process_correlations(correlations, layout_analysis)
        
        return correlations
    
    def _calculate_confidence(self, match: Dict[str, Any]) -> float:
        """Calculate confidence score for a match"""
        if not match:
            return 0
        
        distance = match['distance']
        direction = match['direction']
        
        # Distance-based confidence (closer = higher confidence)
        distance_confidence = max(0, 1 - distance / 200)
        
        # Direction-based confidence (right/below preferred for typical forms)
        direction_confidence = 0.9 if direction in ['right', 'below'] else 0.7
        
        return distance_confidence * direction_confidence
    
    def _calculate_alignment_confidence(self, match: Dict[str, Any]) -> float:
        """Calculate confidence score for alignment-based match"""
        distance = match['distance']
        alignment = match['alignment']
        direction = match['direction']
        
        # Alignment-based confidence
        alignment_confidence = 0.9 if alignment == 'horizontal' else 0.7
        
        # Distance-based confidence
        distance_confidence = max(0, 1 - distance / 100)
        
        # Direction preference
        direction_confidence = 0.9 if direction in ['right', 'below'] else 0.8
        
        return alignment_confidence * distance_confidence * direction_confidence
    
    def _post_process_correlations(self, correlations: List[Dict[str, Any]], 
                                  layout_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Post-process correlations to improve accuracy"""
        
        # Remove duplicate text assignments
        used_texts = set()
        processed = []
        
        # Sort by confidence (highest first)
        sorted_correlations = sorted(correlations, key=lambda x: x['confidence'], reverse=True)
        
        for correlation in sorted_correlations:
            if correlation['matched_text'] is None:
                processed.append(correlation)
                continue
            
            text_id = id(correlation['matched_text'])
            
            if text_id not in used_texts:
                used_texts.add(text_id)
                processed.append(correlation)
            else:
                # Text already used, set to None
                correlation['matched_text'] = None
                correlation['confidence'] = 0
                processed.append(correlation)
        
        # Restore original order
        processed.sort(key=lambda x: x['checkbox']['id'])
        
        return processed
    
    def create_survey_structure(self, correlations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create structured survey data from correlations"""
        
        survey_items = []
        
        for i, correlation in enumerate(correlations):
            checkbox = correlation['checkbox']
            text = correlation['matched_text']
            
            item = {
                'item_id': i,
                'checkbox_id': checkbox['id'],
                'checkbox_bbox': checkbox['bbox'],
                'is_filled': checkbox.get('is_filled', False),
                'fill_confidence': checkbox.get('fill_confidence', 0),
                'label': text.get('cleaned_text', text['text']) if text else f"Checkbox {checkbox['id']}",
                'label_confidence': text['confidence'] if text else 0,
                'correlation_confidence': correlation['confidence'],
                'spatial_relationship': {
                    'distance': correlation['distance'],
                    'direction': correlation['direction']
                }
            }
            
            survey_items.append(item)
        
        # Calculate overall quality metrics
        matched_count = sum(1 for item in survey_items if item['label_confidence'] > 0)
        avg_correlation_confidence = np.mean([item['correlation_confidence'] for item in survey_items])
        
        return {
            'survey_items': survey_items,
            'metadata': {
                'total_checkboxes': len(survey_items),
                'matched_labels': matched_count,
                'match_rate': matched_count / len(survey_items) if survey_items else 0,
                'avg_correlation_confidence': avg_correlation_confidence
            }
        }
    
    def visualize_correlations(self, image: np.ndarray, 
                             correlations: List[Dict[str, Any]], 
                             output_path: str = "checkbox_correlations.png"):
        """Visualize checkbox-label correlations"""
        import cv2
        
        vis_image = image.copy()
        
        for correlation in correlations:
            checkbox = correlation['checkbox']
            text_region = correlation['matched_text']
            confidence = correlation['confidence']
            
            # Draw checkbox
            x, y, w, h = checkbox['bbox']
            cb_color = (0, 255, 0) if confidence > 0.7 else (255, 255, 0) if confidence > 0.3 else (255, 0, 0)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), cb_color, 2)
            
            # Draw text region if matched
            if text_region:
                tx, ty, tw, th = text_region['bbox']
                cv2.rectangle(vis_image, (tx, ty), (tx + tw, ty + th), cb_color, 1)
                
                # Draw connection line
                cb_center = (x + w//2, y + h//2)
                text_center = (tx + tw//2, ty + th//2)
                cv2.line(vis_image, cb_center, text_center, cb_color, 1)
                
                # Add confidence label
                mid_point = ((cb_center[0] + text_center[0])//2, (cb_center[1] + text_center[1])//2)
                cv2.putText(vis_image, f"{confidence:.2f}", mid_point, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, cb_color, 1)
        
        # Convert RGB to BGR for saving
        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_bgr)
        
        print(f"Correlation visualization saved to: {output_path}")
        return vis_image

# Example usage
if __name__ == "__main__":
    # This would typically be used with other components
    from enhanced_checkbox_detector import EnhancedCheckboxDetector
    from checkbox_fill_detector import CheckboxFillDetector
    from text_extractor import TextExtractor
    
    # Initialize components
    detector = EnhancedCheckboxDetector()
    fill_detector = CheckboxFillDetector()
    text_extractor = TextExtractor()
    correlator = CheckboxLabelCorrelator(debug=True)
    
    # Load and process
    pdf_path = "output.pdf"  # Replace with your file
    image = detector.load_pdf_page(pdf_path)
    
    # Detect checkboxes and analyze fill
    checkboxes = detector.detect_checkboxes(image)
    analyzed_checkboxes = fill_detector.analyze_checkboxes(image, checkboxes)
    
    # Extract text
    form_structure = text_extractor.extract_form_structure(image)
    text_regions = form_structure['options']  # Use option text for correlation
    
    # Correlate checkboxes with labels
    correlations = correlator.correlate_adaptive(analyzed_checkboxes, text_regions)
    
    # Create survey structure
    survey = correlator.create_survey_structure(correlations)
    
    # Visualize
    correlator.visualize_correlations(image, correlations)
    
    # Print results
    print(f"\nSurvey Analysis Results:")
    print(f"Total checkboxes: {survey['metadata']['total_checkboxes']}")
    print(f"Matched labels: {survey['metadata']['matched_labels']}")
    print(f"Match rate: {survey['metadata']['match_rate']:.1%}")
    print(f"Avg correlation confidence: {survey['metadata']['avg_correlation_confidence']:.3f}")
    
    print(f"\nSurvey Items:")
    for item in survey['survey_items'][:5]:  # Show first 5
        status = "✓" if item['is_filled'] else "✗"
        print(f"  {status} {item['label']} (conf: {item['correlation_confidence']:.2f})")