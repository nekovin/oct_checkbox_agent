"""Configuration management for OCR Checkbox Agent."""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from loguru import logger


class OCRConfig(BaseModel):
    """OCR engine configuration."""
    language: str = Field(default="eng", description="Tesseract language")
    psm: int = Field(default=6, description="Page segmentation mode")
    oem: int = Field(default=3, description="OCR Engine mode")
    confidence_threshold: float = Field(default=0.6, description="Minimum confidence score")
    
    
class CheckboxDetectionConfig(BaseModel):
    """Checkbox detection configuration."""
    min_checkbox_size: int = Field(default=10, description="Minimum checkbox size in pixels")
    max_checkbox_size: int = Field(default=50, description="Maximum checkbox size in pixels")
    checkbox_aspect_ratio_threshold: float = Field(default=0.2, description="Aspect ratio tolerance")
    binary_threshold: int = Field(default=127, description="Binary threshold for checkbox detection")
    contour_area_threshold: int = Field(default=100, description="Minimum contour area")
    checkbox_fill_threshold: float = Field(default=0.3, description="Percentage of filled pixels to consider checked")
    
    
class ProcessingConfig(BaseModel):
    """General processing configuration."""
    dpi: int = Field(default=300, description="DPI for PDF to image conversion")
    parallel_workers: int = Field(default=4, description="Number of parallel workers")
    batch_size: int = Field(default=10, description="Batch processing size")
    max_retries: int = Field(default=3, description="Maximum retries for failed operations")
    enable_preprocessing: bool = Field(default=True, description="Enable image preprocessing")
    
    
class OutputConfig(BaseModel):
    """Output configuration."""
    output_format: str = Field(default="csv", description="Output format (csv, xlsx, json)")
    include_confidence_scores: bool = Field(default=True, description="Include confidence scores in output")
    include_metadata: bool = Field(default=True, description="Include metadata in output")
    timestamp_format: str = Field(default="%Y-%m-%d_%H-%M-%S", description="Timestamp format")
    
    
class Config(BaseModel):
    """Main configuration class."""
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    checkbox_detection: CheckboxDetectionConfig = Field(default_factory=CheckboxDetectionConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # Paths
    temp_dir: Path = Field(default=Path("./temp"), description="Temporary directory")
    output_dir: Path = Field(default=Path("./output"), description="Output directory")
    template_dir: Path = Field(default=Path("./templates"), description="Template directory")
    
    def save(self, path: Path) -> None:
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from file."""
        if not path.exists():
            logger.warning(f"Config file {path} not found, using defaults")
            return cls()
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Configuration loaded from {path}")
        return cls(**data)
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for dir_path in [self.temp_dir, self.output_dir, self.template_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            

class FormTemplate(BaseModel):
    """Form template definition."""
    name: str
    version: str = "1.0"
    fields: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    checkbox_regions: Optional[Dict[str, Dict[str, Any]]] = None
    
    def save(self, path: Path) -> None:
        """Save template to file."""
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "FormTemplate":
        """Load template from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
        _config.ensure_directories()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    _config = config
    _config.ensure_directories()