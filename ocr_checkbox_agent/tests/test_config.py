"""Tests for configuration management."""

import pytest
import json
import tempfile
from pathlib import Path
from ocr_checkbox_agent.config import Config, FormTemplate, OCRConfig, CheckboxDetectionConfig


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.ocr.language == "eng"
        assert config.ocr.confidence_threshold == 0.6
        assert config.checkbox_detection.min_checkbox_size == 10
        assert config.checkbox_detection.max_checkbox_size == 50
        assert config.processing.dpi == 300
        assert config.output.output_format == "csv"
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        config = Config()
        config.ocr.language = "fra"
        config.processing.dpi = 400
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save config
            config.save(temp_path)
            
            # Load config
            loaded_config = Config.load(temp_path)
            
            assert loaded_config.ocr.language == "fra"
            assert loaded_config.processing.dpi == 400
            
        finally:
            temp_path.unlink()
    
    def test_config_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.temp_dir = Path(temp_dir) / "temp"
            config.output_dir = Path(temp_dir) / "output"
            
            config.ensure_directories()
            
            assert config.temp_dir.exists()
            assert config.output_dir.exists()


class TestFormTemplate:
    """Test form template functionality."""
    
    def test_form_template_creation(self):
        """Test creating form templates."""
        template = FormTemplate(
            name="Test Form",
            fields={
                "question1": {"type": "checkbox", "options": ["Yes", "No"]},
                "question2": {"type": "text"}
            }
        )
        
        assert template.name == "Test Form"
        assert template.version == "1.0"
        assert len(template.fields) == 2
        assert template.fields["question1"]["type"] == "checkbox"
    
    def test_template_save_load(self):
        """Test saving and loading templates."""
        template = FormTemplate(
            name="Survey Form",
            fields={"satisfaction": {"type": "checkbox"}}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save template
            template.save(temp_path)
            
            # Load template
            loaded_template = FormTemplate.load(temp_path)
            
            assert loaded_template.name == "Survey Form"
            assert loaded_template.fields["satisfaction"]["type"] == "checkbox"
            
        finally:
            temp_path.unlink()


class TestConfigComponents:
    """Test individual configuration components."""
    
    def test_ocr_config(self):
        """Test OCR configuration."""
        ocr_config = OCRConfig(
            language="eng+fra",
            confidence_threshold=0.8
        )
        
        assert ocr_config.language == "eng+fra"
        assert ocr_config.confidence_threshold == 0.8
        assert ocr_config.psm == 6  # default
    
    def test_checkbox_detection_config(self):
        """Test checkbox detection configuration."""
        detect_config = CheckboxDetectionConfig(
            min_checkbox_size=15,
            max_checkbox_size=45,
            checkbox_fill_threshold=0.25
        )
        
        assert detect_config.min_checkbox_size == 15
        assert detect_config.max_checkbox_size == 45
        assert detect_config.checkbox_fill_threshold == 0.25