"""Integration tests for OCR Checkbox Agent."""

import pytest
import numpy as np
import tempfile
import cv2
from pathlib import Path
from ocr_checkbox_agent import OCRCheckboxAgent, Config
from ocr_checkbox_agent.checkbox_detector import CheckboxState


class TestIntegration:
    """Integration tests for the complete system."""
    
    def create_test_pdf_image(self, size=(400, 600), checkboxes=None):
        """Create a test image that looks like a PDF page with checkboxes."""
        image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        
        if checkboxes is None:
            checkboxes = [
                {"pos": (50, 100), "size": 20, "filled": True, "text": "Yes"},
                {"pos": (50, 150), "size": 20, "filled": False, "text": "No"},
                {"pos": (50, 200), "size": 20, "filled": True, "text": "Maybe"}
            ]
        
        for checkbox in checkboxes:
            x, y = checkbox["pos"]
            size = checkbox["size"]
            
            # Draw checkbox outline
            cv2.rectangle(image, (x, y), (x + size, y + size), (0, 0, 0), 2)
            
            # Fill if checked
            if checkbox["filled"]:
                cv2.rectangle(image, (x + 3, y + 3), (x + size - 3, y + size - 3), (0, 0, 0), -1)
            
            # Add text label
            cv2.putText(image, checkbox["text"], (x + size + 10, y + size//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return image
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = OCRCheckboxAgent()
        
        assert agent.pdf_processor is not None
        assert agent.ocr_engine is not None
        assert agent.checkbox_detector is not None
        assert agent.data_structurer is not None
        assert agent.config is not None
    
    def test_agent_with_custom_config(self):
        """Test agent with custom configuration."""
        config = Config()
        config.processing.dpi = 200
        config.output.output_format = "json"
        
        agent = OCRCheckboxAgent(config)
        
        assert agent.config.processing.dpi == 200
        assert agent.config.output.output_format == "json"
    
    @pytest.mark.skip(reason="Requires actual PDF processing and OCR setup")
    def test_process_single_image(self):
        """Test processing a single image (simulating PDF page)."""
        agent = OCRCheckboxAgent()
        
        # Create test image
        image = self.create_test_pdf_image()
        
        # Mock metadata
        metadata = {
            "filename": "test.pdf",
            "page_dimensions": {"width": 400, "height": 600}
        }
        
        # Process the image directly (bypassing PDF processing)
        response = agent._process_page(image, metadata, 1)
        
        assert response is not None
        assert response.document_id == "test.pdf"
        assert response.page_number == 1
        assert len(response.fields) > 0
    
    def test_batch_processing_empty_list(self):
        """Test batch processing with empty list."""
        agent = OCRCheckboxAgent()
        
        results = agent.process_batch([])
        
        assert len(results) == 0
    
    def test_save_results_formats(self):
        """Test saving results in different formats."""
        agent = OCRCheckboxAgent()
        
        # Create mock extraction result
        from ocr_checkbox_agent.data_structurer import ExtractionResult, FormResponse
        from datetime import datetime
        
        response = FormResponse(
            document_id="test.pdf",
            page_number=1,
            timestamp=datetime.now(),
            fields={
                "question1": {"value": True, "confidence": 0.9, "state": "checked", "bbox": (10, 10, 20, 20), "fill_ratio": 0.8}
            },
            metadata={"total_checkboxes": 1},
            confidence_score=0.9
        )
        
        result = ExtractionResult(
            document_path="test.pdf",
            total_pages=1,
            responses=[response],
            processing_time=1.0,
            errors=[],
            warnings=[]
        )
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = Path(f.name)
        
        try:
            agent.save_results([result], csv_path)
            assert csv_path.exists()
            
            # Check report was created
            report_path = csv_path.parent / f"{csv_path.stem}_report.json"
            assert report_path.exists()
            
        finally:
            csv_path.unlink()
            if report_path.exists():
                report_path.unlink()
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        config = Config()
        config.processing.dpi = 300
        config.checkbox_detection.min_checkbox_size = 10
        
        agent = OCRCheckboxAgent(config)
        assert agent.config.processing.dpi == 300
        
        # Test config boundaries
        config.processing.dpi = 50  # Very low DPI
        config.checkbox_detection.min_checkbox_size = 5  # Very small
        
        agent = OCRCheckboxAgent(config)
        # Should still work but may affect quality
        assert agent.config.processing.dpi == 50
    
    def test_error_handling(self):
        """Test error handling in the system."""
        agent = OCRCheckboxAgent()
        
        # Test with non-existent file
        non_existent = Path("non_existent.pdf")
        result = agent.process_pdf(non_existent)
        
        assert len(result.errors) > 0
        assert len(result.responses) == 0
    
    def test_parallel_processing_config(self):
        """Test parallel processing configuration."""
        config = Config()
        config.processing.parallel_workers = 2
        
        agent = OCRCheckboxAgent(config)
        
        # Test with empty list (should not crash)
        results = agent.process_batch([], parallel=True)
        assert len(results) == 0
    
    def test_debug_mode(self):
        """Test debug mode functionality."""
        config = Config()
        config.temp_dir = Path(tempfile.mkdtemp())
        
        agent = OCRCheckboxAgent(config)
        
        # Create test image
        image = self.create_test_pdf_image()
        metadata = {"filename": "test.pdf"}
        
        # Process with debug mode (should not crash)
        try:
            response = agent._process_page(image, metadata, 1, save_debug=True)
            # Debug images should be saved to temp directory
            debug_dir = config.temp_dir / "debug"
            if debug_dir.exists():
                debug_files = list(debug_dir.glob("*.png"))
                # May or may not have debug files depending on detection results
        except Exception as e:
            # Debug mode should not cause crashes
            pytest.fail(f"Debug mode caused exception: {e}")
    
    def test_quality_validation(self):
        """Test quality validation across the system."""
        agent = OCRCheckboxAgent()
        
        # Create responses with different quality levels
        from ocr_checkbox_agent.data_structurer import FormResponse
        from datetime import datetime
        
        good_response = FormResponse(
            document_id="good.pdf",
            page_number=1,
            timestamp=datetime.now(),
            fields={"q1": {"value": True, "confidence": 0.95}},
            metadata={},
            confidence_score=0.95
        )
        
        bad_response = FormResponse(
            document_id="bad.pdf",
            page_number=1,
            timestamp=datetime.now(),
            fields={"q1": {"value": None, "confidence": 0.3}},
            metadata={},
            confidence_score=0.3
        )
        
        # Validate quality
        validation = agent.data_structurer.validate_structured_data([good_response, bad_response])
        
        assert validation["total_responses"] == 2
        assert validation["valid_responses"] == 1
        assert 0 <= validation["quality_score"] <= 1