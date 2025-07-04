#!/usr/bin/env python3
"""
Test script for processing only the first survey (stopping after page 2).
This script demonstrates enhanced OCR preprocessing and intelligent survey detection.
"""

import sys
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from ocr_checkbox_agent.intelligent_main import IntelligentOCRAgent
from ocr_checkbox_agent.config import Config


def test_first_survey_only(pdf_path: str, output_dir: str = "test_results"):
    """
    Test processing with limitations for the first survey only.
    
    Args:
        pdf_path: Path to the PDF file to process
        output_dir: Directory to save results
    """
    
    print("🧪 Testing First Survey Processing")
    print("=" * 50)
    
    # Configure for enhanced processing
    config = Config()
    config.processing.dpi = 300  # High quality
    config.output.output_format = "xlsx"
    config.output.include_confidence_scores = True
    config.output.include_metadata = True
    
    # Initialize intelligent agent
    agent = IntelligentOCRAgent(config)
    
    # Process with restrictions for testing
    print(f"📄 Processing: {pdf_path}")
    print("📋 Configuration:")
    print(f"   - Max Pages: 2 (stop after page 2)")
    print(f"   - Max Surveys: 1 (only first survey)")
    print(f"   - Enhanced OCR: Enabled")
    print(f"   - Debug Mode: Enabled")
    print(f"   - Multi-Survey Detection: Enabled")
    print()
    
    try:
        result = agent.process_pdf(
            Path(pdf_path),
            save_debug=True,                    # Save debug images
            detect_multiple_surveys=True,       # Detect survey boundaries
            max_pages=2,                       # Stop after page 2
            max_surveys=1,                     # Only process first survey
            enhanced_ocr=True                  # Use enhanced OCR preprocessing
        )
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save results
        output_file = output_path / "first_survey_test_results.xlsx"
        agent.save_results([result], output_file)
        
        # Print summary
        print("✅ Processing Complete!")
        print(f"📊 Results Summary:")
        print(f"   - Document: {result.document_path}")
        print(f"   - Pages Processed: {result.total_pages}")
        print(f"   - Responses Found: {len(result.responses)}")
        print(f"   - Processing Time: {result.processing_time:.2f}s")
        print(f"   - Errors: {len(result.errors)}")
        print(f"   - Warnings: {len(result.warnings)}")
        
        if result.errors:
            print(f"❌ Errors encountered:")
            for error in result.errors:
                print(f"   - {error}")
        
        print(f"\n📁 Output saved to: {output_file}")
        
        # Check for debug files
        debug_dir = agent.config.temp_dir / "intelligent_debug"
        if debug_dir.exists():
            debug_files = list(debug_dir.glob("*"))
            print(f"🐛 Debug files created: {len(debug_files)}")
            print(f"   Location: {debug_dir}")
            
            for debug_file in debug_files:
                print(f"   - {debug_file.name}")
        
        # Show response details
        if result.responses:
            print(f"\n📋 Response Details:")
            for i, response in enumerate(result.responses, 1):
                print(f"   Response {i}:")
                print(f"   - Page: {response.page_number}")
                print(f"   - Fields: {len(response.fields)}")
                print(f"   - Confidence: {response.confidence_score:.2f}")
                
                if response.metadata.get('intelligent_analysis'):
                    print(f"   - Intelligent Analysis: ✅")
                    print(f"   - Completion Rate: {response.metadata.get('completion_rate', 0):.2f}")
                    print(f"   - Quality Score: {response.metadata.get('quality_score', 0):.2f}")
                
                # Show first few fields
                field_count = 0
                for field_name, field_data in response.fields.items():
                    if field_count >= 3:  # Show only first 3 fields
                        break
                    print(f"   - {field_name}: {field_data.get('value', 'N/A')}")
                    field_count += 1
                
                if len(response.fields) > 3:
                    print(f"   - ... and {len(response.fields) - 3} more fields")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def show_enhanced_ocr_info():
    """Show information about enhanced OCR features."""
    
    print("🔍 Enhanced OCR Features:")
    print("=" * 30)
    print("1. 📷 Advanced Preprocessing:")
    print("   - Noise reduction using Non-Local Means Denoising")
    print("   - Contrast enhancement with CLAHE")
    print("   - Automatic deskewing for rotated documents")
    print()
    print("2. 🎯 Multi-Strategy OCR:")
    print("   - Standard OCR on enhanced image")
    print("   - Binary threshold for crisp text")
    print("   - Adaptive threshold for varied lighting")
    print("   - Best result selection based on confidence")
    print()
    print("3. 📝 Text Block Enhancement:")
    print("   - Merge nearby text blocks")
    print("   - Expand bounding boxes for better capture")
    print("   - Improved spatial relationships")
    print()
    print("4. ☑️ Enhanced Checkbox Detection:")
    print("   - Rectangular pattern detection")
    print("   - Circular pattern detection") 
    print("   - Multiple response type support")
    print()


def main():
    """Main function for testing."""
    
    if len(sys.argv) < 2:
        print("Usage: python test_first_survey.py <pdf_path> [output_dir]")
        print()
        print("Example:")
        print("  python test_first_survey.py my_survey.pdf")
        print("  python test_first_survey.py my_survey.pdf custom_output/")
        print()
        show_enhanced_ocr_info()
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "test_results"
    
    # Verify PDF file exists
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Show enhanced OCR info
    show_enhanced_ocr_info()
    
    # Run the test
    success = test_first_survey_only(pdf_path, output_dir)
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("\nNext steps:")
        print("1. Check the Excel output file for structured data")
        print("2. Review debug images in temp/intelligent_debug/")
        print("3. Examine confidence scores and quality metrics")
        print("4. Adjust processing parameters if needed")
    else:
        print("\n💥 Test failed - check the error messages above")
        sys.exit(1)


if __name__ == "__main__":
    main()