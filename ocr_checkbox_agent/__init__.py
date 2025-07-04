"""OCR Checkbox Extraction Agent."""

from .main import OCRCheckboxAgent
from .config import Config, FormTemplate, get_config, set_config
from .pdf_processor import PDFProcessor
from .ocr_engine import OCREngine
from .checkbox_detector import CheckboxDetector, Checkbox, CheckboxState
from .data_structurer import DataStructurer, FormResponse, ExtractionResult

__version__ = "1.0.0"
__author__ = "OCR Checkbox Agent"
__description__ = "Extract checkbox data from PDFs and convert to structured format"

__all__ = [
    "OCRCheckboxAgent",
    "Config",
    "FormTemplate",
    "get_config",
    "set_config",
    "PDFProcessor",
    "OCREngine",
    "CheckboxDetector",
    "Checkbox",
    "CheckboxState",
    "DataStructurer",
    "FormResponse",
    "ExtractionResult",
]