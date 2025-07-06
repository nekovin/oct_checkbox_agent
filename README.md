# Survey Scanner

A comprehensive Python pipeline for extracting checkbox states and labels from PDF survey forms using computer vision and OCR.

## Features

- **Enhanced Checkbox Detection**: Combines BoxDetect and OpenCV for robust checkbox identification
- **Fill State Analysis**: Determines whether checkboxes are checked using multiple analysis methods
- **Text Extraction**: Uses Tesseract OCR to extract labels and form text
- **Smart Correlation**: Automatically correlates checkboxes with their corresponding labels
- **Quality Validation**: Comprehensive validation with confidence scoring
- **Batch Processing**: Process multiple PDFs efficiently

## Installation

### Prerequisites

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install opencv-python numpy pillow pdf2image pytesseract boxdetect scikit-learn scipy matplotlib
```

## Quick Start

### Basic Usage

```python
from survey_scanner_pipeline import SurveyScannerPipeline

# Initialize pipeline
pipeline = SurveyScannerPipeline(
    dpi=200,
    checkbox_width_range=(20, 40),
    checkbox_height_range=(20, 40),
    debug=True
)

# Process a PDF
results = pipeline.process_pdf("your_survey.pdf", page_num=0)

# Print summary
survey_data = results['survey_data']
print(f"Found {survey_data['metadata']['total_checkboxes']} checkboxes")
print(f"Match rate: {survey_data['metadata']['match_rate']:.1%}")

# Show filled checkboxes
for item in survey_data['survey_items']:
    if item['is_filled']:
        print(f"✓ {item['label']}")
```

### Command Line Usage

```bash
python survey_scanner_pipeline.py
```

Make sure to update the `pdf_path` variable in the main function.

## Components

### 1. Enhanced Checkbox Detector (`enhanced_checkbox_detector.py`)

Detects checkboxes using multiple methods:
- BoxDetect library for structured detection
- OpenCV contour analysis for edge cases
- Configurable size and aspect ratio filtering

```python
from enhanced_checkbox_detector import EnhancedCheckboxDetector

detector = EnhancedCheckboxDetector(dpi=200)
image = detector.load_pdf_page("form.pdf")
checkboxes = detector.detect_checkboxes(image, width_range=(20, 40), height_range=(20, 40))
```

### 2. Checkbox Fill Detector (`checkbox_fill_detector.py`)

Analyzes whether checkboxes are filled:
- Pixel density analysis
- Contour detection for check marks
- Cross/X pattern recognition
- Confidence scoring

```python
from checkbox_fill_detector import CheckboxFillDetector

fill_detector = CheckboxFillDetector(debug=True)
analyzed_checkboxes = fill_detector.analyze_checkboxes(image, checkboxes)
```

### 3. Text Extractor (`text_extractor.py`)

Extracts text using Tesseract OCR:
- Form structure analysis
- Text region detection with bounding boxes
- Question and option categorization
- Confidence scoring

```python
from text_extractor import TextExtractor

extractor = TextExtractor(debug=True)
form_structure = extractor.extract_form_structure(image)
```

### 4. Checkbox Label Correlator (`checkbox_label_correlator.py`)

Correlates checkboxes with their labels:
- Spatial proximity analysis
- Layout pattern detection
- Adaptive correlation strategies
- Confidence scoring

```python
from checkbox_label_correlator import CheckboxLabelCorrelator

correlator = CheckboxLabelCorrelator(debug=True)
correlations = correlator.correlate_adaptive(checkboxes, text_regions)
```

### 5. Quality Validator (`quality_validator.py`)

Validates processing quality:
- Image quality assessment
- Component validation
- Confidence scoring
- Recommendations for improvement

```python
from quality_validator import QualityValidator

validator = QualityValidator(debug=True)
validation = validator.comprehensive_validation(image, checkboxes, analyzed_checkboxes, form_structure, correlations)
```

## Configuration

### Checkbox Detection Parameters

Adjust these based on your form's checkbox size at the chosen DPI:

```python
# For 200 DPI scans:
cfg.width_range = (20, 40)   # Checkbox width in pixels
cfg.height_range = (20, 40)  # Checkbox height in pixels

# For 300 DPI scans:
cfg.width_range = (30, 60)   # Scale up proportionally
cfg.height_range = (30, 60)
```

### Finding the Right Parameters

Use the measurement tools to determine checkbox sizes:

```python
from measure_for_boxdetect import measure_and_suggest_config
measure_and_suggest_config('your_survey.pdf')
```

Or use the quick measurement script:

```bash
python quick_measure.py your_survey.pdf
```

## Output

The pipeline generates several outputs:

### JSON Results
```json
{
  "metadata": {
    "source_file": "survey.pdf",
    "page_number": 0,
    "processed_at": "2025-01-15T10:30:00",
    "processing_parameters": {...}
  },
  "survey_data": {
    "survey_items": [
      {
        "item_id": 0,
        "checkbox_id": 0,
        "is_filled": true,
        "label": "Option A",
        "correlation_confidence": 0.85
      }
    ],
    "metadata": {
      "total_checkboxes": 10,
      "matched_labels": 9,
      "match_rate": 0.9
    }
  }
}
```

### Visualizations
- `*_checkboxes.png`: Detected checkboxes with IDs
- `*_fill_analysis.png`: Fill status visualization
- `*_text.png`: Extracted text regions
- `*_correlations.png`: Checkbox-label correlations

## Troubleshooting

### Common Issues

1. **No checkboxes detected**
   - Adjust `width_range` and `height_range`
   - Increase DPI for higher resolution
   - Check if form has unusual checkbox styles

2. **Poor text extraction**
   - Improve image quality/resolution
   - Adjust OCR preprocessing settings
   - Try different PSM modes in Tesseract

3. **Low correlation confidence**
   - Checkboxes may be too far from labels
   - Form layout may be unusual
   - Consider manual verification

4. **Fill detection errors**
   - Adjust fill detection thresholds
   - Check for unusual check mark styles
   - Verify image quality

### Quality Validation

The pipeline includes comprehensive quality validation:

```python
validation = validator.comprehensive_validation(...)
print(f"Overall quality: {validation['overall']['grade']}")
print(f"Recommendations: {validation['recommendations']}")
```

## Advanced Usage

### Batch Processing

```python
# Process multiple PDFs
pdf_files = ["survey1.pdf", "survey2.pdf", "survey3.pdf"]
results = pipeline.batch_process(pdf_files)

# Generate summary report
summary = pipeline.create_summary_report(results)
print(f"Success rate: {summary['summary']['successful']}/{summary['summary']['total_files_processed']}")
```

### Custom Configuration

```python
# Custom pipeline configuration
pipeline = SurveyScannerPipeline(
    dpi=300,
    checkbox_width_range=(30, 60),
    checkbox_height_range=(30, 60),
    output_dir="custom_output",
    debug=True
)

# Custom validation thresholds
validator = QualityValidator()
validator.thresholds['min_correlation_confidence'] = 0.5
validator.thresholds['max_correlation_distance'] = 150
```

## API Reference

### SurveyScannerPipeline

Main pipeline class for processing survey forms.

#### Methods

- `process_pdf(pdf_path, page_num=0)`: Process single PDF page
- `batch_process(pdf_files, pages=None)`: Process multiple PDFs
- `create_summary_report(results)`: Generate batch processing summary

#### Parameters

- `dpi`: Resolution for PDF conversion (default: 200)
- `checkbox_width_range`: Checkbox width range in pixels
- `checkbox_height_range`: Checkbox height range in pixels
- `output_dir`: Output directory for results
- `debug`: Enable debug logging

### Component Classes

Each component can be used independently:

- `EnhancedCheckboxDetector`: Checkbox detection
- `CheckboxFillDetector`: Fill state analysis
- `TextExtractor`: OCR text extraction
- `CheckboxLabelCorrelator`: Spatial correlation
- `QualityValidator`: Quality assessment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples
3. Open an issue with sample data if needed