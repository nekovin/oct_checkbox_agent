#!/usr/bin/env python3
"""
Basic usage examples for OCR Checkbox Agent.

This script demonstrates how to use the OCR Checkbox Agent to extract
checkbox data from PDF files and convert it to structured formats.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import cv2

# Add the parent directory to the path so we can import the agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_checkbox_agent import OCRCheckboxAgent, Config


def create_sample_checkbox_image():
    """Create a sample image with checkboxes for testing."""
    # Create a white background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(img, "Sample Survey Form", (250, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Create checkboxes with labels
    questions = [
        {"text": "Are you satisfied with our service?", "pos": (100, 150), "checked": True},
        {"text": "Would you recommend us to friends?", "pos": (100, 200), "checked": False},
        {"text": "Did our product meet your expectations?", "pos": (100, 250), "checked": True},
        {"text": "Would you purchase from us again?", "pos": (100, 300), "checked": True},
        {"text": "Was our customer support helpful?", "pos": (100, 350), "checked": False}
    ]
    
    for i, question in enumerate(questions):
        x, y = question["pos"]
        
        # Draw checkbox
        cv2.rectangle(img, (x, y), (x + 20, y + 20), (0, 0, 0), 2)
        
        # Fill checkbox if checked
        if question["checked"]:
            cv2.rectangle(img, (x + 3, y + 3), (x + 17, y + 17), (0, 0, 0), -1)
        
        # Add question text
        cv2.putText(img, question["text"], (x + 35, y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return img


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("=== Example 1: Basic Usage ===")
    
    # Initialize agent with default configuration
    agent = OCRCheckboxAgent()
    
    print("Agent initialized with default configuration")
    print(f"- DPI: {agent.config.processing.dpi}")
    print(f"- Output format: {agent.config.output.output_format}")
    print(f"- Include confidence scores: {agent.config.output.include_confidence_scores}")
    
    # Note: In real usage, you would process actual PDF files like this:
    # result = agent.process_pdf("path/to/your/form.pdf")
    # agent.save_results([result], "extracted_data.csv")
    
    print("✓ Example 1 completed\n")


def example_2_custom_configuration():
    """Example 2: Using custom configuration."""
    print("=== Example 2: Custom Configuration ===")
    
    # Create custom configuration
    config = Config()
    config.processing.dpi = 400  # Higher DPI for better quality
    config.output.output_format = "xlsx"  # Excel output
    config.checkbox_detection.min_checkbox_size = 15  # Larger minimum size
    config.checkbox_detection.checkbox_fill_threshold = 0.25  # More sensitive
    
    # Initialize agent with custom config
    agent = OCRCheckboxAgent(config)
    
    print("Agent initialized with custom configuration:")
    print(f"- DPI: {agent.config.processing.dpi}")
    print(f"- Output format: {agent.config.output.output_format}")
    print(f"- Min checkbox size: {agent.config.checkbox_detection.min_checkbox_size}")
    print(f"- Fill threshold: {agent.config.checkbox_detection.checkbox_fill_threshold}")
    
    print("✓ Example 2 completed\n")


def example_3_form_template():
    """Example 3: Using form templates."""
    print("=== Example 3: Form Templates ===")
    
    agent = OCRCheckboxAgent()
    
    # Check if template exists
    template_path = Path(__file__).parent.parent / "templates" / "survey_template.json"
    
    if template_path.exists():
        # Load template
        agent.load_template(template_path)
        print(f"Loaded template: {agent.form_template.name}")
        print(f"Template fields: {list(agent.form_template.fields.keys())}")
    else:
        print("Template file not found, skipping template demo")
    
    print("✓ Example 3 completed\n")


def example_4_batch_processing():
    """Example 4: Batch processing multiple files."""
    print("=== Example 4: Batch Processing ===")
    
    config = Config()
    config.processing.parallel_workers = 2  # Use 2 workers for parallel processing
    
    agent = OCRCheckboxAgent(config)
    
    # In real usage, you would have a list of PDF files:
    # pdf_files = [
    #     Path("form1.pdf"),
    #     Path("form2.pdf"),
    #     Path("form3.pdf")
    # ]
    # 
    # # Process all files
    # results = agent.process_batch(pdf_files, parallel=True)
    # 
    # # Save consolidated results
    # agent.save_results(results, "batch_results.xlsx")
    
    print("Batch processing configured for parallel execution")
    print(f"- Workers: {agent.config.processing.parallel_workers}")
    print("- In real usage, this would process multiple PDFs simultaneously")
    
    print("✓ Example 4 completed\n")


def example_5_quality_validation():
    """Example 5: Quality validation and confidence scoring."""
    print("=== Example 5: Quality Validation ===")
    
    agent = OCRCheckboxAgent()
    
    # Example of how validation works (using mock data)
    from ocr_checkbox_agent.data_structurer import FormResponse
    from datetime import datetime
    
    # Create sample responses with different confidence levels
    high_confidence_response = FormResponse(
        document_id="good_scan.pdf",
        page_number=1,
        timestamp=datetime.now(),
        fields={
            "question1": {"value": True, "confidence": 0.95, "state": "checked"},
            "question2": {"value": False, "confidence": 0.91, "state": "unchecked"}
        },
        metadata={"total_checkboxes": 2},
        confidence_score=0.93
    )
    
    low_confidence_response = FormResponse(
        document_id="poor_scan.pdf",
        page_number=1,
        timestamp=datetime.now(),
        fields={
            "question1": {"value": None, "confidence": 0.3, "state": "unknown"},
            "question2": {"value": False, "confidence": 0.4, "state": "unchecked"}
        },
        metadata={"total_checkboxes": 2},
        confidence_score=0.35
    )
    
    responses = [high_confidence_response, low_confidence_response]
    
    # Validate data quality
    validation_results = agent.data_structurer.validate_structured_data(responses)
    
    print("Quality validation results:")
    print(f"- Total responses: {validation_results['total_responses']}")
    print(f"- Valid responses: {validation_results['valid_responses']}")
    print(f"- Quality score: {validation_results['quality_score']:.2f}")
    print(f"- Issues found: {len(validation_results['issues'])}")
    
    if validation_results['issues']:
        print("\nIssues detected:")
        for issue in validation_results['issues']:
            print(f"  - Document: {issue['document_id']}, Issues: {issue['issues']}")
    
    print("✓ Example 5 completed\n")


def example_6_different_output_formats():
    """Example 6: Different output formats."""
    print("=== Example 6: Output Formats ===")
    
    # Mock data for demonstration
    from ocr_checkbox_agent.data_structurer import FormResponse, ExtractionResult
    from datetime import datetime
    
    response = FormResponse(
        document_id="sample.pdf",
        page_number=1,
        timestamp=datetime.now(),
        fields={
            "satisfaction": {"value": True, "confidence": 0.9, "state": "checked"},
            "recommend": {"value": False, "confidence": 0.85, "state": "unchecked"}
        },
        metadata={"total_checkboxes": 2},
        confidence_score=0.875
    )
    
    result = ExtractionResult(
        document_path="sample.pdf",
        total_pages=1,
        responses=[response],
        processing_time=2.5,
        errors=[],
        warnings=[]
    )
    
    # Show different output formats
    formats = ["csv", "xlsx", "json"]
    
    for fmt in formats:
        config = Config()
        config.output.output_format = fmt
        agent = OCRCheckboxAgent(config)
        
        print(f"Output format: {fmt}")
        print(f"- File extension: .{fmt}")
        print(f"- Includes confidence: {config.output.include_confidence_scores}")
        print(f"- Includes metadata: {config.output.include_metadata}")
        
        # In real usage:
        # agent.save_results([result], f"output.{fmt}")
    
    print("✓ Example 6 completed\n")


def main():
    """Run all examples."""
    print("OCR Checkbox Agent - Usage Examples")
    print("=" * 50)
    
    try:
        example_1_basic_usage()
        example_2_custom_configuration()
        example_3_form_template()
        example_4_batch_processing()
        example_5_quality_validation()
        example_6_different_output_formats()
        
        print("🎉 All examples completed successfully!")
        print("\nTo process real PDF files:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Install Tesseract OCR and Poppler")
        print("3. Run: python -m ocr_checkbox_agent.main your_form.pdf -o results.csv")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure all dependencies are installed.")


if __name__ == "__main__":
    main()