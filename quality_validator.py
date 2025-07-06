import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import cv2

class QualityValidator:
    """
    Validate quality of survey scanning results and provide confidence scores
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.thresholds = {
            'min_checkbox_size': 100,  # Minimum checkbox area in pixels
            'max_checkbox_size': 2000,  # Maximum checkbox area in pixels
            'min_text_confidence': 30,  # Minimum OCR confidence
            'min_correlation_confidence': 0.3,  # Minimum correlation confidence
            'max_correlation_distance': 200,  # Maximum distance for correlation
            'min_fill_confidence': 0.4,  # Minimum fill detection confidence
            'aspect_ratio_tolerance': 0.3  # Tolerance for square checkboxes
        }
    
    def validate_checkbox_detection(self, checkboxes: List[Dict[str, Any]], 
                                  image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Validate quality of checkbox detection"""
        
        if not checkboxes:
            return {
                'valid': False,
                'score': 0.0,
                'issues': ['No checkboxes detected'],
                'statistics': {'count': 0}
            }
        
        issues = []
        valid_checkboxes = 0
        
        # Check individual checkboxes
        for checkbox in checkboxes:
            bbox = checkbox['bbox']
            x, y, w, h = bbox
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Size validation
            if area < self.thresholds['min_checkbox_size']:
                issues.append(f"Checkbox {checkbox['id']} too small: {area} pixels")
            elif area > self.thresholds['max_checkbox_size']:
                issues.append(f"Checkbox {checkbox['id']} too large: {area} pixels")
            else:
                valid_checkboxes += 1
            
            # Aspect ratio validation (should be roughly square)
            if not (0.5 <= aspect_ratio <= 2.0):
                issues.append(f"Checkbox {checkbox['id']} has unusual aspect ratio: {aspect_ratio:.2f}")
            
            # Position validation (should be within image bounds)
            if (x < 0 or y < 0 or 
                x + w > image_shape[1] or 
                y + h > image_shape[0]):
                issues.append(f"Checkbox {checkbox['id']} extends outside image bounds")
        
        # Overall statistics
        areas = [cb['bbox'][2] * cb['bbox'][3] for cb in checkboxes]
        aspect_ratios = [cb['aspect_ratio'] for cb in checkboxes]
        
        statistics = {
            'count': len(checkboxes),
            'valid_count': valid_checkboxes,
            'avg_area': np.mean(areas),
            'area_std': np.std(areas),
            'avg_aspect_ratio': np.mean(aspect_ratios),
            'aspect_ratio_std': np.std(aspect_ratios)
        }
        
        # Calculate quality score
        score = self._calculate_detection_score(checkboxes, statistics, len(issues))
        
        return {
            'valid': score >= 0.6,
            'score': score,
            'issues': issues,
            'statistics': statistics
        }
    
    def validate_fill_analysis(self, analyzed_checkboxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate quality of checkbox fill analysis"""
        
        if not analyzed_checkboxes:
            return {
                'valid': False,
                'score': 0.0,
                'issues': ['No analyzed checkboxes provided'],
                'statistics': {}
            }
        
        issues = []
        high_confidence_count = 0
        
        # Check fill analysis quality
        fill_confidences = []
        fill_scores = []
        
        for checkbox in analyzed_checkboxes:
            if 'fill_analysis' not in checkbox:
                issues.append(f"Checkbox {checkbox['id']} missing fill analysis")
                continue
            
            fill_analysis = checkbox['fill_analysis']
            confidence = fill_analysis.get('confidence', 0)
            filled_score = fill_analysis.get('filled_score', 0)
            
            fill_confidences.append(confidence)
            fill_scores.append(filled_score)
            
            # Check confidence levels
            if confidence >= self.thresholds['min_fill_confidence']:
                high_confidence_count += 1
            else:
                issues.append(f"Checkbox {checkbox['id']} has low fill confidence: {confidence:.3f}")
        
        if fill_confidences:
            statistics = {
                'total_analyzed': len(analyzed_checkboxes),
                'high_confidence_count': high_confidence_count,
                'avg_fill_confidence': np.mean(fill_confidences),
                'min_fill_confidence': np.min(fill_confidences),
                'max_fill_confidence': np.max(fill_confidences),
                'avg_fill_score': np.mean(fill_scores),
                'filled_count': sum(1 for cb in analyzed_checkboxes if cb.get('is_filled', False))
            }
            
            # Calculate quality score
            score = statistics['avg_fill_confidence'] * (high_confidence_count / len(analyzed_checkboxes))
        else:
            statistics = {}
            score = 0.0
        
        return {
            'valid': score >= 0.5,
            'score': score,
            'issues': issues,
            'statistics': statistics
        }
    
    def validate_text_extraction(self, form_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality of text extraction"""
        
        text_regions = form_structure.get('text_regions', [])
        
        if not text_regions:
            return {
                'valid': False,
                'score': 0.0,
                'issues': ['No text extracted'],
                'statistics': {}
            }
        
        issues = []
        high_confidence_text = 0
        
        # Check text quality
        confidences = []
        text_lengths = []
        
        for region in text_regions:
            confidence = region.get('confidence', 0)
            text = region.get('cleaned_text', region.get('text', ''))
            
            confidences.append(confidence)
            text_lengths.append(len(text))
            
            if confidence >= self.thresholds['min_text_confidence']:
                high_confidence_text += 1
            else:
                issues.append(f"Low confidence text: '{text}' ({confidence})")
        
        statistics = {
            'total_regions': len(text_regions),
            'high_confidence_count': high_confidence_text,
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'avg_text_length': np.mean(text_lengths),
            'questions_found': len(form_structure.get('questions', [])),
            'options_found': len(form_structure.get('options', []))
        }
        
        # Calculate quality score
        confidence_score = statistics['avg_confidence'] / 100.0  # Normalize to 0-1
        coverage_score = min(1.0, len(text_regions) / 10)  # Assume 10+ regions is good
        
        score = (confidence_score + coverage_score) / 2
        
        return {
            'valid': score >= 0.4,
            'score': score,
            'issues': issues,
            'statistics': statistics
        }
    
    def validate_correlations(self, correlations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate quality of checkbox-label correlations"""
        
        if not correlations:
            return {
                'valid': False,
                'score': 0.0,
                'issues': ['No correlations provided'],
                'statistics': {}
            }
        
        issues = []
        successful_matches = 0
        high_confidence_matches = 0
        
        # Check correlation quality
        correlation_confidences = []
        distances = []
        
        for corr in correlations:
            confidence = corr.get('confidence', 0)
            distance = corr.get('distance', float('inf'))
            matched_text = corr.get('matched_text')
            
            correlation_confidences.append(confidence)
            
            if matched_text is not None:
                successful_matches += 1
                distances.append(distance)
                
                if confidence >= self.thresholds['min_correlation_confidence']:
                    high_confidence_matches += 1
                else:
                    checkbox_id = corr.get('checkbox', {}).get('id', 'unknown')
                    issues.append(f"Low confidence correlation for checkbox {checkbox_id}: {confidence:.3f}")
                
                if distance > self.thresholds['max_correlation_distance']:
                    checkbox_id = corr.get('checkbox', {}).get('id', 'unknown')
                    issues.append(f"Large correlation distance for checkbox {checkbox_id}: {distance:.1f}px")
            else:
                distances.append(float('inf'))
                checkbox_id = corr.get('checkbox', {}).get('id', 'unknown')
                issues.append(f"No text match found for checkbox {checkbox_id}")
        
        statistics = {
            'total_correlations': len(correlations),
            'successful_matches': successful_matches,
            'high_confidence_matches': high_confidence_matches,
            'match_rate': successful_matches / len(correlations) if correlations else 0,
            'avg_correlation_confidence': np.mean(correlation_confidences),
            'avg_distance': np.mean([d for d in distances if d != float('inf')]) if distances else 0,
            'max_distance': max([d for d in distances if d != float('inf')]) if distances else 0
        }
        
        # Calculate quality score
        match_score = statistics['match_rate']
        confidence_score = statistics['avg_correlation_confidence']
        
        score = (match_score + confidence_score) / 2
        
        return {
            'valid': score >= 0.5,
            'score': score,
            'issues': issues,
            'statistics': statistics
        }
    
    def validate_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Validate input image quality"""
        
        issues = []
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Check image properties
        height, width = gray.shape
        
        # Resolution check
        if width < 1000 or height < 1000:
            issues.append(f"Low resolution: {width}x{height} (recommend >1000x1000)")
        
        # Contrast analysis
        contrast = gray.std()
        if contrast < 20:
            issues.append(f"Low contrast: {contrast:.1f} (recommend >20)")
        
        # Brightness analysis
        brightness = gray.mean()
        if brightness < 50 or brightness > 200:
            issues.append(f"Poor brightness: {brightness:.1f} (recommend 50-200)")
        
        # Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 100:
            issues.append(f"Image appears blurry: {blur_score:.1f} (recommend >100)")
        
        statistics = {
            'resolution': (width, height),
            'contrast': contrast,
            'brightness': brightness,
            'blur_score': blur_score,
            'file_size_mb': (image.nbytes / 1024 / 1024)
        }
        
        # Calculate quality score
        resolution_score = min(1.0, min(width, height) / 1000)
        contrast_score = min(1.0, contrast / 50)
        brightness_score = 1.0 if 50 <= brightness <= 200 else max(0, 1 - abs(brightness - 125) / 125)
        blur_score_norm = min(1.0, blur_score / 200)
        
        score = (resolution_score + contrast_score + brightness_score + blur_score_norm) / 4
        
        return {
            'valid': score >= 0.6,
            'score': score,
            'issues': issues,
            'statistics': statistics
        }
    
    def comprehensive_validation(self, 
                               image: np.ndarray,
                               checkboxes: List[Dict[str, Any]],
                               analyzed_checkboxes: List[Dict[str, Any]],
                               form_structure: Dict[str, Any],
                               correlations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive validation of all components"""
        
        validations = {}
        
        # Validate each component
        validations['image'] = self.validate_image_quality(image)
        validations['checkbox_detection'] = self.validate_checkbox_detection(checkboxes, image.shape)
        validations['fill_analysis'] = self.validate_fill_analysis(analyzed_checkboxes)
        validations['text_extraction'] = self.validate_text_extraction(form_structure)
        validations['correlations'] = self.validate_correlations(correlations)
        
        # Calculate overall scores
        overall_score = np.mean([v['score'] for v in validations.values()])
        overall_valid = all(v['valid'] for v in validations.values())
        
        # Collect all issues
        all_issues = []
        for component, validation in validations.items():
            for issue in validation['issues']:
                all_issues.append(f"[{component}] {issue}")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validations)
        
        return {
            'overall': {
                'valid': overall_valid,
                'score': overall_score,
                'grade': self._score_to_grade(overall_score)
            },
            'component_validations': validations,
            'all_issues': all_issues,
            'recommendations': recommendations,
            'summary': self._generate_summary(validations, overall_score)
        }
    
    def _calculate_detection_score(self, checkboxes: List[Dict[str, Any]], 
                                 statistics: Dict[str, Any], 
                                 issue_count: int) -> float:
        """Calculate detection quality score"""
        
        if not checkboxes:
            return 0.0
        
        # Base score from valid checkbox ratio
        valid_ratio = statistics['valid_count'] / statistics['count']
        
        # Penalty for issues
        issue_penalty = min(0.5, issue_count * 0.1)
        
        # Bonus for reasonable checkbox count
        count_bonus = min(0.2, statistics['count'] / 20)  # Bonus up to 20 checkboxes
        
        score = valid_ratio - issue_penalty + count_bonus
        return max(0.0, min(1.0, score))
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, validations: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Image quality recommendations
        image_val = validations['image']
        if image_val['score'] < 0.7:
            if image_val['statistics']['contrast'] < 20:
                recommendations.append("Improve image contrast using image enhancement tools")
            if image_val['statistics']['blur_score'] < 100:
                recommendations.append("Use higher quality scan or reduce motion blur")
            if min(image_val['statistics']['resolution']) < 1000:
                recommendations.append("Increase scan DPI for better resolution")
        
        # Checkbox detection recommendations
        checkbox_val = validations['checkbox_detection']
        if checkbox_val['score'] < 0.7:
            recommendations.append("Adjust checkbox size range parameters")
            recommendations.append("Check if form layout is suitable for detection")
        
        # Text extraction recommendations
        text_val = validations['text_extraction']
        if text_val['score'] < 0.6:
            recommendations.append("Improve image quality for better OCR results")
            recommendations.append("Consider using different OCR settings or preprocessing")
        
        # Correlation recommendations
        corr_val = validations['correlations']
        if corr_val['score'] < 0.6:
            recommendations.append("Review form layout - checkboxes may be too far from labels")
            recommendations.append("Consider manual verification of checkbox-label pairs")
        
        return recommendations
    
    def _generate_summary(self, validations: Dict[str, Any], overall_score: float) -> str:
        """Generate summary text"""
        
        grade = self._score_to_grade(overall_score)
        
        component_scores = {name: val['score'] for name, val in validations.items()}
        best_component = max(component_scores, key=component_scores.get)
        worst_component = min(component_scores, key=component_scores.get)
        
        summary = f"Overall quality grade: {grade} (score: {overall_score:.3f}). "
        summary += f"Best performing component: {best_component} "
        summary += f"({component_scores[best_component]:.3f}). "
        summary += f"Needs improvement: {worst_component} "
        summary += f"({component_scores[worst_component]:.3f})."
        
        return summary

# Example usage
if __name__ == "__main__":
    # This would typically be used with the full pipeline
    validator = QualityValidator(debug=True)
    
    # Example validation (would use real data in practice)
    print("Quality Validator initialized")
    print("Use with survey_scanner_pipeline.py for complete validation")