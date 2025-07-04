"""Tests for data structurer."""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path
from datetime import datetime
from ocr_checkbox_agent.data_structurer import DataStructurer, FormResponse, ExtractionResult
from ocr_checkbox_agent.checkbox_detector import Checkbox, CheckboxState
from ocr_checkbox_agent.config import Config


class TestDataStructurer:
    """Test data structuring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.structurer = DataStructurer()
        self.config = Config()
    
    def create_test_checkbox(self, state=CheckboxState.CHECKED, confidence=0.9, text="Test Question"):
        """Create a test checkbox."""
        return Checkbox(
            bbox=(10, 10, 20, 20),
            state=state,
            confidence=confidence,
            fill_ratio=0.8 if state == CheckboxState.CHECKED else 0.1,
            associated_text=text
        )
    
    def create_test_form_response(self):
        """Create a test form response."""
        return FormResponse(
            document_id="test.pdf",
            page_number=1,
            timestamp=datetime.now(),
            fields={
                "question1": {
                    "value": True,
                    "state": "checked",
                    "confidence": 0.9,
                    "bbox": (10, 10, 20, 20),
                    "fill_ratio": 0.8
                },
                "question2": {
                    "value": False,
                    "state": "unchecked",
                    "confidence": 0.85,
                    "bbox": (10, 50, 20, 20),
                    "fill_ratio": 0.1
                }
            },
            metadata={
                "total_checkboxes": 2,
                "checked_count": 1,
                "unchecked_count": 1
            },
            confidence_score=0.875
        )
    
    def test_data_structurer_init(self):
        """Test data structurer initialization."""
        structurer = DataStructurer()
        
        assert structurer.output_format == "csv"
        assert structurer.include_confidence == True
        assert structurer.include_metadata == True
    
    def test_data_structurer_with_config(self):
        """Test data structurer with custom config."""
        config = Config()
        config.output.output_format = "xlsx"
        config.output.include_confidence_scores = False
        
        structurer = DataStructurer(config)
        
        assert structurer.output_format == "xlsx"
        assert structurer.include_confidence == False
    
    def test_structure_checkbox_data(self):
        """Test structuring checkbox data."""
        structurer = DataStructurer()
        
        checkboxes = [
            self.create_test_checkbox(CheckboxState.CHECKED, 0.9, "Question 1"),
            self.create_test_checkbox(CheckboxState.UNCHECKED, 0.8, "Question 2")
        ]
        
        metadata = {"filename": "test.pdf", "page_dimensions": {"width": 800, "height": 600}}
        
        response = structurer.structure_checkbox_data(checkboxes, metadata, 1)
        
        assert response.document_id == "test.pdf"
        assert response.page_number == 1
        assert len(response.fields) == 2
        assert response.fields["question_1"]["value"] == True
        assert response.fields["question_2"]["value"] == False
        assert response.confidence_score > 0.8
    
    def test_sanitize_field_name(self):
        """Test field name sanitization."""
        structurer = DataStructurer()
        
        # Test with special characters
        assert structurer._sanitize_field_name("Question #1: Yes/No?") == "question_1_yesno"
        
        # Test with spaces and dashes
        assert structurer._sanitize_field_name("First Name - Last Name") == "first_name_last_name"
        
        # Test with empty string
        assert structurer._sanitize_field_name("") == "unknown_field"
        
        # Test with only special characters
        assert structurer._sanitize_field_name("@#$%^&*()") == "unknown_field"
    
    def test_create_tabular_data(self):
        """Test creating tabular data."""
        structurer = DataStructurer()
        
        responses = [
            self.create_test_form_response(),
            self.create_test_form_response()
        ]
        
        df = structurer.create_tabular_data(responses)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "document_id" in df.columns
        assert "page_number" in df.columns
        assert "question1" in df.columns
        assert "question2" in df.columns
        
        # Check confidence columns if included
        if structurer.include_confidence:
            assert "question1_confidence" in df.columns
            assert "question2_confidence" in df.columns
    
    def test_create_tabular_data_empty(self):
        """Test creating tabular data with empty responses."""
        structurer = DataStructurer()
        
        df = structurer.create_tabular_data([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_export_to_csv(self):
        """Test exporting to CSV."""
        structurer = DataStructurer()
        responses = [self.create_test_form_response()]
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            structurer.export_to_csv(responses, temp_path)
            
            assert temp_path.exists()
            
            # Read back and verify
            df = pd.read_csv(temp_path)
            assert len(df) == 1
            assert "document_id" in df.columns
            
        finally:
            temp_path.unlink()
    
    def test_export_to_json(self):
        """Test exporting to JSON."""
        structurer = DataStructurer()
        responses = [self.create_test_form_response()]
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            structurer.export_to_json(responses, temp_path)
            
            assert temp_path.exists()
            
            # Read back and verify
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "export_info" in data
            assert "responses" in data
            assert len(data["responses"]) == 1
            assert data["responses"][0]["document_id"] == "test.pdf"
            
        finally:
            temp_path.unlink()
    
    def test_create_summary_data(self):
        """Test creating summary data."""
        structurer = DataStructurer()
        responses = [
            self.create_test_form_response(),
            self.create_test_form_response()
        ]
        
        summary_data = structurer._create_summary_data(responses)
        
        assert len(summary_data) == 1  # Both responses from same document
        assert summary_data[0]["document_id"] == "test.pdf"
        assert summary_data[0]["total_pages"] == 2
        assert summary_data[0]["total_fields"] == 4  # 2 fields per response
    
    def test_validate_structured_data(self):
        """Test data validation."""
        structurer = DataStructurer()
        
        # Create responses with different quality levels
        good_response = self.create_test_form_response()
        good_response.confidence_score = 0.9
        
        bad_response = self.create_test_form_response()
        bad_response.confidence_score = 0.3  # Low confidence
        bad_response.fields = {}  # No fields
        
        responses = [good_response, bad_response]
        
        validation_results = structurer.validate_structured_data(responses)
        
        assert validation_results["total_responses"] == 2
        assert validation_results["valid_responses"] == 1
        assert len(validation_results["issues"]) == 1
        assert validation_results["quality_score"] == 0.5
    
    def test_create_extraction_report(self):
        """Test creating extraction report."""
        structurer = DataStructurer()
        
        extraction_results = [
            ExtractionResult(
                document_path="test1.pdf",
                total_pages=2,
                responses=[self.create_test_form_response()],
                processing_time=5.0,
                errors=[],
                warnings=[]
            ),
            ExtractionResult(
                document_path="test2.pdf",
                total_pages=1,
                responses=[self.create_test_form_response()],
                processing_time=3.0,
                errors=["Test error"],
                warnings=["Test warning"]
            )
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            structurer.create_extraction_report(extraction_results, temp_path)
            
            assert temp_path.exists()
            
            # Read back and verify
            with open(temp_path, 'r') as f:
                report = json.load(f)
            
            assert "summary" in report
            assert "documents" in report
            assert report["summary"]["total_documents"] == 2
            assert report["summary"]["total_pages"] == 3
            assert len(report["documents"]) == 2
            
        finally:
            temp_path.unlink()