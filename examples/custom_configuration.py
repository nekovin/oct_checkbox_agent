#!/usr/bin/env python3
"""
Custom configuration example for Survey Scanner
"""

import sys
import os

# Add parent directory to path to import survey scanner modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survey_scanner_pipeline import SurveyScannerPipeline
from enhanced_checkbox_detector import EnhancedCheckboxDetector
from checkbox_fill_detector import CheckboxFillDetector
from text_extractor import TextExtractor
from checkbox_label_correlator import CheckboxLabelCorrelator
from quality_validator import QualityValidator

def high_resolution_config():
    """Configuration for high-resolution scans (300+ DPI)"""
    
    print("High Resolution Configuration Example")
    print("=" * 40)
    
    # For 300 DPI scans, checkboxes will be larger
    pipeline = SurveyScannerPipeline(
        dpi=300,
        checkbox_width_range=(30, 60),  # Scaled up from 200 DPI
        checkbox_height_range=(30, 60),
        output_dir="high_res_output",
        debug=True
    )
    
    # Process with high-resolution settings
    pdf_file = "high_res_survey.pdf"
    if os.path.exists(pdf_file):
        results = pipeline.process_pdf(pdf_file)
        print(f"Processed high-resolution scan: {results['survey_data']['metadata']['total_checkboxes']} checkboxes found")
    else:
        print(f"Sample file '{pdf_file}' not found")

def small_checkbox_config():
    """Configuration for forms with small checkboxes"""
    
    print("\nSmall Checkbox Configuration Example")
    print("=" * 40)
    
    # For forms with very small checkboxes
    pipeline = SurveyScannerPipeline(
        dpi=200,
        checkbox_width_range=(10, 25),  # Smaller range
        checkbox_height_range=(10, 25),
        output_dir="small_checkbox_output",
        debug=True
    )
    
    pdf_file = "small_checkbox_survey.pdf"
    if os.path.exists(pdf_file):
        results = pipeline.process_pdf(pdf_file)
        print(f"Processed small checkbox form: {results['survey_data']['metadata']['total_checkboxes']} checkboxes found")
    else:
        print(f"Sample file '{pdf_file}' not found")

def custom_component_config():
    """Example of using components with custom settings"""
    
    print("\nCustom Component Configuration Example")
    print("=" * 45)
    
    # Initialize components individually with custom settings
    detector = EnhancedCheckboxDetector(dpi=200)
    
    # Custom fill detector with adjusted thresholds
    fill_detector = CheckboxFillDetector(debug=True)
    
    # Custom text extractor
    text_extractor = TextExtractor(language='eng', debug=True)
    
    # Custom correlator
    correlator = CheckboxLabelCorrelator(debug=True)
    
    # Custom validator with modified thresholds
    validator = QualityValidator(debug=True)
    validator.thresholds.update({
        'min_checkbox_size': 50,  # Smaller minimum size
        'max_checkbox_size': 3000,  # Larger maximum size
        'min_text_confidence': 40,  # Lower text confidence threshold
        'min_correlation_confidence': 0.2,  # Lower correlation threshold
        'max_correlation_distance': 250,  # Larger correlation distance
    })
    
    pdf_file = "output.pdf"  # Replace with your file
    
    if os.path.exists(pdf_file):
        try:
            # Manual processing with custom settings
            print(f"Processing {pdf_file} with custom settings...")
            
            # Load image
            image = detector.load_pdf_page(pdf_file)
            
            # Detect checkboxes with custom range
            checkboxes = detector.detect_checkboxes(
                image, 
                width_range=(15, 50),  # Custom size range
                height_range=(15, 50)
            )
            print(f"Detected {len(checkboxes)} checkboxes")
            
            # Analyze fill with custom settings
            analyzed_checkboxes = fill_detector.analyze_checkboxes(image, checkboxes)
            filled_count = sum(1 for cb in analyzed_checkboxes if cb['is_filled'])
            print(f"Found {filled_count} filled checkboxes")
            
            # Extract text
            form_structure = text_extractor.extract_form_structure(image)
            print(f"Extracted {len(form_structure['text_regions'])} text regions")
            
            # Correlate with custom settings
            correlations = correlator.correlate_adaptive(analyzed_checkboxes, form_structure['options'])
            
            # Validate with custom thresholds
            validation = validator.comprehensive_validation(
                image, checkboxes, analyzed_checkboxes, form_structure, correlations
            )
            
            print(f"Quality assessment: {validation['overall']['grade']} (score: {validation['overall']['score']:.3f})")
            print(f"Recommendations: {len(validation['recommendations'])}")
            
            for rec in validation['recommendations'][:3]:  # Show first 3 recommendations
                print(f"  - {rec}")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Sample file '{pdf_file}' not found")

def language_specific_config():
    """Configuration for non-English forms"""
    
    print("\nLanguage-Specific Configuration Example")
    print("=" * 45)
    
    # For forms in other languages (requires appropriate Tesseract language packs)
    
    # Example configurations for different languages:
    language_configs = {
        'spanish': {
            'language': 'spa',
            'description': 'Spanish forms'
        },
        'french': {
            'language': 'fra', 
            'description': 'French forms'
        },
        'german': {
            'language': 'deu',
            'description': 'German forms'
        }
    }
    
    for lang_name, config in language_configs.items():
        print(f"\n{config['description']} configuration:")
        print(f"  Language code: {config['language']}")
        print(f"  Install with: sudo apt-get install tesseract-ocr-{config['language'][:3]}")
        
        # Example text extractor for this language
        try:
            extractor = TextExtractor(language=config['language'], debug=False)
            print(f"  ✓ {lang_name.title()} OCR available")
        except:
            print(f"  ✗ {lang_name.title()} OCR not available")

def quality_focused_config():
    """Configuration optimized for quality over speed"""
    
    print("\nQuality-Focused Configuration Example")
    print("=" * 45)
    
    # High-quality processing with multiple methods
    pipeline = SurveyScannerPipeline(
        dpi=300,  # Higher resolution
        checkbox_width_range=(25, 70),  # Wider range for 300 DPI
        checkbox_height_range=(25, 70),
        output_dir="quality_output",
        debug=True
    )
    
    # Additional quality settings would be applied in the individual components
    print("Quality-focused settings:")
    print("  - Higher DPI (300)")
    print("  - Wider checkbox size range")
    print("  - Multiple detection methods")
    print("  - Comprehensive validation")
    print("  - Detailed confidence scoring")

if __name__ == "__main__":
    high_resolution_config()
    small_checkbox_config()
    custom_component_config()
    language_specific_config()
    quality_focused_config()