#!/usr/bin/env python3
"""
Batch processing example for Survey Scanner
"""

import sys
import os
import glob

# Add parent directory to path to import survey scanner modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survey_scanner_pipeline import SurveyScannerPipeline

def batch_processing_example():
    """Example of processing multiple PDF files"""
    
    print("Survey Scanner - Batch Processing Example")
    print("=" * 45)
    
    # Initialize pipeline
    pipeline = SurveyScannerPipeline(
        dpi=200,
        checkbox_width_range=(20, 40),
        checkbox_height_range=(20, 40),
        output_dir="batch_output",
        debug=False  # Less verbose for batch processing
    )
    
    # Find all PDF files in current directory
    pdf_files = glob.glob("*.pdf")
    
    if not pdf_files:
        print("No PDF files found in current directory.")
        print("Please add some PDF files to process.")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf_file}")
    
    try:
        # Process all PDFs (first page of each)
        print(f"\nProcessing {len(pdf_files)} files...")
        results = pipeline.batch_process(pdf_files, pages=[0])  # Process page 0 of each
        
        # Generate summary report
        summary = pipeline.create_summary_report(results)
        
        # Print summary
        print(f"\nBatch Processing Summary:")
        print(f"- Total files: {summary['summary']['total_files_processed']}")
        print(f"- Successful: {summary['summary']['successful']}")
        print(f"- Failed: {summary['summary']['failed']}")
        print(f"- Total checkboxes found: {summary['summary']['total_checkboxes_found']}")
        print(f"- Total checkboxes filled: {summary['summary']['total_checkboxes_filled']}")
        print(f"- Overall fill rate: {summary['summary']['overall_fill_rate']:.1%}")
        print(f"- Average match rate: {summary['summary']['average_match_rate']:.1%}")
        
        # Show details for each successful file
        print(f"\nDetailed Results:")
        for result in summary['file_results']:
            metadata = result['survey_data']['metadata']
            source_file = os.path.basename(result['metadata']['source_file'])
            filled_count = sum(1 for item in result['survey_data']['survey_items'] if item['is_filled'])
            
            print(f"  {source_file}:")
            print(f"    - Checkboxes: {metadata['total_checkboxes']}")
            print(f"    - Filled: {filled_count}")
            print(f"    - Match rate: {metadata['match_rate']:.1%}")
        
        # Show errors if any
        if summary['errors']:
            print(f"\nErrors:")
            for error in summary['errors']:
                print(f"  - {error['file']}: {error['error']}")
        
        print(f"\nOutput files saved in: {pipeline.output_dir}/")
        
    except Exception as e:
        print(f"Error during batch processing: {e}")

def selective_processing_example():
    """Example of processing specific pages from multiple PDFs"""
    
    print("\nSelective Processing Example")
    print("=" * 30)
    
    # Define specific files and pages to process
    file_page_pairs = [
        ("survey1.pdf", 0),  # First page of survey1.pdf
        ("survey2.pdf", 1),  # Second page of survey2.pdf
        ("survey3.pdf", 0),  # First page of survey3.pdf
    ]
    
    # Filter to only existing files
    existing_pairs = []
    for pdf_file, page_num in file_page_pairs:
        if os.path.exists(pdf_file):
            existing_pairs.append((pdf_file, page_num))
        else:
            print(f"File not found: {pdf_file}")
    
    if not existing_pairs:
        print("No specified files found.")
        return
    
    pipeline = SurveyScannerPipeline(output_dir="selective_output", debug=False)
    
    results = []
    for pdf_file, page_num in existing_pairs:
        try:
            print(f"Processing {pdf_file}, page {page_num}...")
            result = pipeline.process_pdf(pdf_file, page_num)
            results.append(result)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    # Generate summary
    if results:
        summary = pipeline.create_summary_report(results)
        print(f"\nProcessed {len(results)} files successfully")
        print(f"Average match rate: {summary['summary']['average_match_rate']:.1%}")

if __name__ == "__main__":
    batch_processing_example()
    selective_processing_example()