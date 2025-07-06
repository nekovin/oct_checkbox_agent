#!/usr/bin/env python3
"""
Basic usage example for Survey Scanner
"""

import sys
import os

# Add parent directory to path to import survey scanner modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survey_scanner_pipeline import SurveyScannerPipeline

def basic_example():
    """Basic usage example"""
    
    print("Survey Scanner - Basic Usage Example")
    print("=" * 40)
    
    # Initialize pipeline with default settings
    pipeline = SurveyScannerPipeline(
        dpi=200,
        checkbox_width_range=(20, 40),  # Adjust based on your checkboxes
        checkbox_height_range=(20, 40),
        debug=True
    )
    
    # Example PDF file (replace with your actual file)
    pdf_file = "sample_survey.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"Sample file '{pdf_file}' not found.")
        print("Please place your PDF file in this directory and update the filename.")
        return
    
    try:
        # Process the PDF
        print(f"Processing {pdf_file}...")
        results = pipeline.process_pdf(pdf_file, page_num=0)
        
        # Extract survey data
        survey_data = results['survey_data']
        metadata = survey_data['metadata']
        
        # Print summary
        print(f"\nProcessing Results:")
        print(f"- Total checkboxes found: {metadata['total_checkboxes']}")
        print(f"- Labels matched: {metadata['matched_labels']}")
        print(f"- Match rate: {metadata['match_rate']:.1%}")
        print(f"- Average correlation confidence: {metadata['avg_correlation_confidence']:.3f}")
        
        # Show filled checkboxes
        filled_items = [item for item in survey_data['survey_items'] if item['is_filled']]
        print(f"\nFilled checkboxes ({len(filled_items)} total):")
        for item in filled_items:
            print(f"  ✓ {item['label']} (confidence: {item['fill_confidence']:.2f})")
        
        # Show empty checkboxes
        empty_items = [item for item in survey_data['survey_items'] if not item['is_filled']]
        print(f"\nEmpty checkboxes ({len(empty_items)} total):")
        for item in empty_items[:5]:  # Show first 5
            print(f"  ✗ {item['label']} (confidence: {item['fill_confidence']:.2f})")
        if len(empty_items) > 5:
            print(f"  ... and {len(empty_items) - 5} more")
        
        # Show output files
        print(f"\nOutput files saved in: {pipeline.output_dir}/")
        print("- JSON results with detailed data")
        print("- Visualization images showing detection results")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        print("Make sure the PDF file exists and is readable.")

if __name__ == "__main__":
    basic_example()