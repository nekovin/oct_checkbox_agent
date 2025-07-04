# OCR Checkbox Extraction Agent

A robust, production-ready Python agent that extracts checkbox data from PDFs and converts it into structured tabular format using OCR and computer vision techniques.

## Features

- **PDF Processing**: Handle various PDF formats with automatic image conversion
- **OCR Integration**: Advanced text extraction using Tesseract OCR
- **Checkbox Detection**: Computer vision-based checkbox detection and state analysis
- **Data Structuring**: Convert extracted data to structured formats (CSV, Excel, JSON)
- **Batch Processing**: Process multiple PDFs with parallel processing support
- **Template Support**: Use form templates for consistent extraction
- **Quality Assurance**: Confidence scoring and validation for extracted data
- **Error Handling**: Comprehensive error handling and logging

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR**: Install Tesseract OCR for text extraction
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   
   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

3. **Poppler** (for PDF processing):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install poppler-utils
   
   # macOS
   brew install poppler
   
   # Windows
   # Download from: https://poppler.freedesktop.org/
   ```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from ocr_checkbox_agent import OCRCheckboxAgent

# Initialize agent
agent = OCRCheckboxAgent()

# Process a single PDF
result = agent.process_pdf("form.pdf")

# Save results to CSV
agent.save_results([result], "extracted_data.csv")
```

### Command Line Interface

```bash
# Process single PDF
python -m ocr_checkbox_agent.main input.pdf -o output.csv

# Process directory of PDFs
python -m ocr_checkbox_agent.main pdf_folder/ -o results/ --parallel

# With custom configuration
python -m ocr_checkbox_agent.main input.pdf -c config.json -o output.xlsx
```

### Batch Processing

```python
from pathlib import Path
from ocr_checkbox_agent import OCRCheckboxAgent

agent = OCRCheckboxAgent()

# Process multiple PDFs
pdf_files = list(Path("pdf_folder").glob("*.pdf"))
results = agent.process_batch(pdf_files, output_file="batch_results.xlsx")
```

## Configuration

### Basic Configuration

```python
from ocr_checkbox_agent import Config

config = Config()
config.processing.dpi = 300
config.processing.parallel_workers = 4
config.output.output_format = "xlsx"
config.output.include_confidence_scores = True

agent = OCRCheckboxAgent(config)
```

### Configuration File

Create a `config.json` file:

```json
{
  "ocr": {
    "language": "eng",
    "confidence_threshold": 0.6
  },
  "checkbox_detection": {
    "min_checkbox_size": 10,
    "max_checkbox_size": 50,
    "checkbox_fill_threshold": 0.3
  },
  "processing": {
    "dpi": 300,
    "parallel_workers": 4,
    "enable_preprocessing": true
  },
  "output": {
    "output_format": "csv",
    "include_confidence_scores": true,
    "include_metadata": true
  }
}
```

## Form Templates

Define form templates for consistent extraction:

```python
from ocr_checkbox_agent import FormTemplate

template = FormTemplate(
    name="Survey Form",
    fields={
        "satisfaction": {
            "type": "checkbox",
            "options": ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied"]
        },
        "recommendations": {
            "type": "checkbox",
            "options": ["Yes", "No"]
        }
    }
)

# Save template
template.save("survey_template.json")

# Load and use template
agent.load_template("survey_template.json")
result = agent.process_pdf("survey.pdf", use_template=True)
```

## Advanced Features

### Custom OCR Settings

```python
from ocr_checkbox_agent import Config, OCREngine

config = Config()
config.ocr.language = "eng+fra"  # Multiple languages
config.ocr.psm = 6  # Page segmentation mode
config.ocr.oem = 3  # OCR engine mode

ocr_engine = OCREngine(config)
```

### Checkbox Detection Tuning

```python
config = Config()
config.checkbox_detection.min_checkbox_size = 15
config.checkbox_detection.max_checkbox_size = 40
config.checkbox_detection.checkbox_fill_threshold = 0.25
config.checkbox_detection.binary_threshold = 130
```

### Debug Mode

```python
# Enable debug mode to save intermediate images
result = agent.process_pdf("form.pdf", save_debug=True)
# Check temp/debug/ folder for visualization images
```

## Output Formats

### CSV Output
```csv
document_id,page_number,timestamp,question_1,question_2,question_1_confidence,question_2_confidence
form.pdf,1,2024-01-15_10:30:00,True,False,0.95,0.88
```

### Excel Output
- **Checkbox_Data**: Main data with extracted checkbox values
- **Summary**: Document-level statistics
- **Metadata**: Processing metadata and confidence scores

### JSON Output
```json
{
  "export_info": {
    "timestamp": "2024-01-15T10:30:00",
    "total_responses": 1,
    "format_version": "1.0"
  },
  "responses": [
    {
      "document_id": "form.pdf",
      "page_number": 1,
      "fields": {
        "question_1": {
          "value": true,
          "confidence": 0.95,
          "bbox": [100, 200, 20, 20]
        }
      }
    }
  ]
}
```

## Performance Optimization

### Parallel Processing

```python
# Enable parallel processing for batch operations
config = Config()
config.processing.parallel_workers = 8
config.processing.batch_size = 20

agent = OCRCheckboxAgent(config)
results = agent.process_batch(pdf_files, parallel=True)
```

### Memory Management

```python
# Process large batches in chunks
from pathlib import Path

def process_large_batch(pdf_dir, chunk_size=50):
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    for i in range(0, len(pdf_files), chunk_size):
        chunk = pdf_files[i:i+chunk_size]
        results = agent.process_batch(chunk)
        
        # Process results immediately
        output_file = f"batch_{i//chunk_size}.csv"
        agent.save_results(results, output_file)
```

## Error Handling

```python
try:
    result = agent.process_pdf("problematic.pdf")
    
    if result.errors:
        print(f"Processing errors: {result.errors}")
    
    if result.warnings:
        print(f"Processing warnings: {result.warnings}")
        
except Exception as e:
    print(f"Fatal error: {e}")
```

## Quality Assurance

### Validation

```python
from ocr_checkbox_agent import DataStructurer

structurer = DataStructurer()
validation_results = structurer.validate_structured_data(responses)

print(f"Quality score: {validation_results['quality_score']}")
print(f"Issues found: {len(validation_results['issues'])}")
```

### Confidence Thresholds

```python
# Filter results by confidence
high_confidence_responses = [
    response for response in responses 
    if response.confidence_score >= 0.8
]
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**
   ```bash
   # Check if tesseract is in PATH
   tesseract --version
   
   # If not, install or add to PATH
   export PATH="/usr/local/bin:$PATH"
   ```

2. **Low detection accuracy**
   - Increase DPI: `config.processing.dpi = 400`
   - Adjust thresholds: `config.checkbox_detection.checkbox_fill_threshold = 0.2`
   - Enable preprocessing: `config.processing.enable_preprocessing = True`

3. **Memory issues with large PDFs**
   - Process pages individually
   - Reduce DPI: `config.processing.dpi = 200`
   - Use parallel processing: `config.processing.parallel_workers = 2`

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Save debug images
result = agent.process_pdf("form.pdf", save_debug=True)
```

## API Reference

### OCRCheckboxAgent

Main class for checkbox extraction.

#### Methods

- `process_pdf(pdf_path, use_template=False, save_debug=False)`: Process single PDF
- `process_batch(input_paths, output_file=None, parallel=True)`: Process multiple PDFs
- `save_results(results, output_path)`: Save extraction results
- `load_template(template_path)`: Load form template

### Config

Configuration management class.

#### Properties

- `ocr`: OCR engine settings
- `checkbox_detection`: Checkbox detection parameters
- `processing`: General processing settings
- `output`: Output format settings

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Submit an issue on the project repository