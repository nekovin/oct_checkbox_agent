# Contributing to OCR Checkbox Agent

Thank you for your interest in contributing to the OCR Checkbox Agent! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub Issues tab to report bugs or request features
- Provide detailed information including:
  - Operating system and Python version
  - Error messages and stack traces
  - Sample PDFs (if possible) that demonstrate the issue
  - Steps to reproduce the problem

### Submitting Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/ocr-checkbox-agent.git
   cd ocr-checkbox-agent
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding standards outlined below
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   python run_tests.py
   python examples/basic_usage.py
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description of your changes
   - Reference any related issues
   - Include screenshots or examples if applicable

## 📝 Coding Standards

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Keep functions focused and small (< 50 lines when possible)
- Add docstrings to all public functions and classes

### Type Hints
- Use type hints for function parameters and return values
- Import types from `typing` module when needed

```python
from typing import List, Dict, Optional, Union

def process_checkboxes(image: np.ndarray, 
                      config: Optional[Config] = None) -> List[Checkbox]:
    """Process checkboxes in an image."""
    pass
```

### Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings

```python
def detect_checkboxes(self, image: np.ndarray) -> List[Checkbox]:
    """
    Detect checkboxes in an image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of detected checkbox objects
        
    Raises:
        ValueError: If image is invalid
    """
```

### Error Handling
- Use appropriate exception types
- Provide meaningful error messages
- Log errors using the loguru logger

```python
from loguru import logger

try:
    result = process_image(image)
except ValueError as e:
    logger.error(f"Invalid image provided: {e}")
    raise
```

## 🧪 Testing Guidelines

### Writing Tests
- Write unit tests for all new functions
- Use pytest as the testing framework
- Aim for > 80% code coverage
- Include edge cases and error conditions

```python
def test_checkbox_detection_empty_image():
    """Test checkbox detection on empty image."""
    detector = CheckboxDetector()
    empty_image = np.ones((100, 100), dtype=np.uint8) * 255
    
    checkboxes = detector.detect_checkboxes(empty_image)
    
    assert len(checkboxes) == 0
```

### Test Organization
- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Group related tests in test classes
- Use descriptive test function names

### Mock Data
- Create realistic test data when possible
- Use fixtures for complex setup
- Avoid using large files in tests

## 🏗️ Architecture Guidelines

### Module Organization
- Keep modules focused on a single responsibility
- Use clear interfaces between modules
- Minimize circular dependencies

### Configuration
- Use the centralized config system for all settings
- Make new features configurable when appropriate
- Provide sensible defaults

### Performance
- Consider memory usage for large document processing
- Use lazy loading where possible
- Profile performance-critical code paths

## 📋 Development Setup

### Prerequisites
```bash
# Install system dependencies
sudo apt-get install tesseract-ocr poppler-utils  # Ubuntu/Debian
brew install tesseract poppler  # macOS

# Install Python dependencies
pip install -r ocr_checkbox_agent/requirements.txt
pip install pytest pytest-cov black flake8 mypy  # Development tools
```

### Development Tools
```bash
# Code formatting
black ocr_checkbox_agent/

# Linting
flake8 ocr_checkbox_agent/

# Type checking
mypy ocr_checkbox_agent/

# Running tests with coverage
pytest --cov=ocr_checkbox_agent tests/
```

## 🚀 Release Process

### Version Numbering
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `__init__.py` and `setup.py`

### Release Checklist
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version numbers are bumped
- [ ] CHANGELOG is updated
- [ ] Release notes are prepared

## 💡 Feature Ideas

### High Priority
- [ ] Improved checkbox detection algorithms
- [ ] Better template matching
- [ ] Enhanced error recovery
- [ ] Performance optimizations

### Medium Priority
- [ ] Web-based interface
- [ ] REST API server
- [ ] Cloud service integration
- [ ] Machine learning enhancements

### Low Priority
- [ ] GUI application
- [ ] Mobile app integration
- [ ] Advanced analytics
- [ ] Custom output formats

## 🔧 Debugging Tips

### Common Issues
1. **OCR not working**: Check Tesseract installation
2. **Poor detection**: Adjust DPI and thresholds
3. **Memory issues**: Reduce parallel workers
4. **Import errors**: Check Python path and dependencies

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Save debug images
agent.process_pdf("form.pdf", save_debug=True)
# Check temp/debug/ folder for images
```

### Profiling
```python
import cProfile
import pstats

# Profile performance
profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = agent.process_pdf("form.pdf")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

## 📚 Resources

### Documentation
- [OpenCV Documentation](https://docs.opencv.org/)
- [Tesseract OCR](https://tesseract-ocr.github.io/)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Computer Vision
- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

### OCR and Document Processing
- [Tesseract OCR Best Practices](https://tesseract-ocr.github.io/tessdoc/)
- [Document Image Analysis](https://en.wikipedia.org/wiki/Document_image_analysis)

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- CHANGELOG.md for their contributions
- Release notes for significant features

Thank you for helping make OCR Checkbox Agent better!