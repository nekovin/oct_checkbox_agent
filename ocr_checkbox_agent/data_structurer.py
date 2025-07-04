"""Data structuring module for formatting checkbox extraction results."""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, asdict
from loguru import logger

from .checkbox_detector import Checkbox, CheckboxState
from .config import get_config


@dataclass
class FormResponse:
    """Represents a single form response."""
    document_id: str
    page_number: int
    timestamp: datetime
    fields: Dict[str, Any]
    metadata: Dict[str, Any]
    confidence_score: float


@dataclass
class ExtractionResult:
    """Complete extraction result for a document."""
    document_path: str
    total_pages: int
    responses: List[FormResponse]
    processing_time: float
    errors: List[str]
    warnings: List[str]


class DataStructurer:
    """Handles structuring and formatting of extraction results."""
    
    def __init__(self, config=None):
        """Initialize data structurer with configuration."""
        self.config = config or get_config()
        self.output_format = self.config.output.output_format
        self.include_confidence = self.config.output.include_confidence_scores
        self.include_metadata = self.config.output.include_metadata
        self.timestamp_format = self.config.output.timestamp_format
        
    def structure_checkbox_data(self, 
                              checkboxes: List[Checkbox],
                              document_metadata: Dict[str, Any],
                              page_number: int = 1) -> FormResponse:
        """
        Structure checkbox data into a form response.
        
        Args:
            checkboxes: List of detected checkboxes
            document_metadata: Document metadata
            page_number: Page number
            
        Returns:
            Structured form response
        """
        # Group checkboxes by associated text
        fields = {}
        confidence_scores = []
        
        for checkbox in checkboxes:
            # Use associated text as field name, or generate one
            field_name = checkbox.associated_text or f"checkbox_{len(fields) + 1}"
            field_name = self._sanitize_field_name(field_name)
            
            # Convert checkbox state to boolean or text
            if checkbox.state == CheckboxState.CHECKED:
                value = True
            elif checkbox.state == CheckboxState.UNCHECKED:
                value = False
            else:
                value = None
            
            # Store field data
            field_data = {
                'value': value,
                'state': checkbox.state.value,
                'bbox': checkbox.bbox,
                'fill_ratio': checkbox.fill_ratio
            }
            
            if self.include_confidence:
                field_data['confidence'] = checkbox.confidence
            
            fields[field_name] = field_data
            confidence_scores.append(checkbox.confidence)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Prepare metadata
        response_metadata = {
            'document_name': document_metadata.get('filename', ''),
            'page_dimensions': document_metadata.get('page_dimensions', {}),
            'processing_method': 'checkbox_detection'
        }
        
        if self.include_metadata:
            response_metadata.update({
                'total_checkboxes': len(checkboxes),
                'checked_count': sum(1 for cb in checkboxes if cb.state == CheckboxState.CHECKED),
                'unchecked_count': sum(1 for cb in checkboxes if cb.state == CheckboxState.UNCHECKED),
                'unknown_count': sum(1 for cb in checkboxes if cb.state == CheckboxState.UNKNOWN),
                'avg_confidence': overall_confidence
            })
        
        return FormResponse(
            document_id=document_metadata.get('filename', 'unknown'),
            page_number=page_number,
            timestamp=datetime.now(),
            fields=fields,
            metadata=response_metadata,
            confidence_score=overall_confidence
        )
    
    def create_tabular_data(self, responses: List[FormResponse]) -> pd.DataFrame:
        """
        Create tabular data from form responses.
        
        Args:
            responses: List of form responses
            
        Returns:
            Pandas DataFrame with structured data
        """
        if not responses:
            return pd.DataFrame()
        
        # Collect all unique field names
        all_fields = set()
        for response in responses:
            all_fields.update(response.fields.keys())
        
        # Create rows
        rows = []
        for response in responses:
            row = {
                'document_id': response.document_id,
                'page_number': response.page_number,
                'timestamp': response.timestamp.strftime(self.timestamp_format)
            }
            
            # Add field values
            for field_name in sorted(all_fields):
                if field_name in response.fields:
                    field_data = response.fields[field_name]
                    row[field_name] = field_data.get('value', None)
                    
                    if self.include_confidence:
                        row[f"{field_name}_confidence"] = field_data.get('confidence', 0.0)
                else:
                    row[field_name] = None
                    if self.include_confidence:
                        row[f"{field_name}_confidence"] = 0.0
            
            # Add metadata if requested
            if self.include_metadata:
                row['overall_confidence'] = response.confidence_score
                row['total_checkboxes'] = response.metadata.get('total_checkboxes', 0)
                row['checked_count'] = response.metadata.get('checked_count', 0)
                row['unchecked_count'] = response.metadata.get('unchecked_count', 0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def export_to_csv(self, responses: List[FormResponse], 
                     output_path: Union[str, Path]) -> None:
        """Export responses to CSV file."""
        df = self.create_tabular_data(responses)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(responses)} responses to CSV: {output_path}")
    
    def export_to_excel(self, responses: List[FormResponse], 
                       output_path: Union[str, Path]) -> None:
        """Export responses to Excel file."""
        df = self.create_tabular_data(responses)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Checkbox_Data', index=False)
            
            # Summary sheet
            summary_data = self._create_summary_data(responses)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Metadata sheet if requested
            if self.include_metadata:
                metadata_df = self._create_metadata_sheet(responses)
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        logger.info(f"Exported {len(responses)} responses to Excel: {output_path}")
    
    def export_to_json(self, responses: List[FormResponse], 
                      output_path: Union[str, Path]) -> None:
        """Export responses to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_responses': len(responses),
                'format_version': '1.0'
            },
            'responses': []
        }
        
        for response in responses:
            response_data = asdict(response)
            response_data['timestamp'] = response.timestamp.isoformat()
            data['responses'].append(response_data)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(responses)} responses to JSON: {output_path}")
    
    def create_extraction_report(self, extraction_results: List[ExtractionResult], 
                               output_path: Union[str, Path]) -> None:
        """Create a comprehensive extraction report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect statistics
        total_documents = len(extraction_results)
        total_pages = sum(result.total_pages for result in extraction_results)
        total_responses = sum(len(result.responses) for result in extraction_results)
        total_processing_time = sum(result.processing_time for result in extraction_results)
        
        # Create report data
        report_data = {
            'summary': {
                'total_documents': total_documents,
                'total_pages': total_pages,
                'total_responses': total_responses,
                'total_processing_time': total_processing_time,
                'avg_processing_time_per_document': total_processing_time / total_documents if total_documents > 0 else 0,
                'avg_responses_per_document': total_responses / total_documents if total_documents > 0 else 0
            },
            'documents': []
        }
        
        for result in extraction_results:
            doc_data = {
                'path': result.document_path,
                'pages': result.total_pages,
                'responses': len(result.responses),
                'processing_time': result.processing_time,
                'errors': result.errors,
                'warnings': result.warnings,
                'avg_confidence': sum(r.confidence_score for r in result.responses) / len(result.responses) if result.responses else 0
            }
            report_data['documents'].append(doc_data)
        
        # Export as JSON
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Created extraction report: {output_path}")
    
    def _sanitize_field_name(self, field_name: str) -> str:
        """Sanitize field name for use as column header."""
        # Remove special characters and normalize
        import re
        sanitized = re.sub(r'[^\w\s-]', '', field_name)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        sanitized = sanitized.strip('_').lower()
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'unknown_field'
        
        return sanitized
    
    def _create_summary_data(self, responses: List[FormResponse]) -> List[Dict[str, Any]]:
        """Create summary statistics."""
        if not responses:
            return []
        
        # Group by document
        by_document = {}
        for response in responses:
            doc_id = response.document_id
            if doc_id not in by_document:
                by_document[doc_id] = []
            by_document[doc_id].append(response)
        
        summary_data = []
        for doc_id, doc_responses in by_document.items():
            total_fields = sum(len(r.fields) for r in doc_responses)
            avg_confidence = sum(r.confidence_score for r in doc_responses) / len(doc_responses)
            
            summary_data.append({
                'document_id': doc_id,
                'total_pages': len(doc_responses),
                'total_fields': total_fields,
                'avg_confidence': avg_confidence,
                'first_processed': min(r.timestamp for r in doc_responses).strftime(self.timestamp_format),
                'last_processed': max(r.timestamp for r in doc_responses).strftime(self.timestamp_format)
            })
        
        return summary_data
    
    def _create_metadata_sheet(self, responses: List[FormResponse]) -> pd.DataFrame:
        """Create metadata sheet for Excel export."""
        metadata_rows = []
        
        for response in responses:
            metadata_rows.append({
                'document_id': response.document_id,
                'page_number': response.page_number,
                'timestamp': response.timestamp.strftime(self.timestamp_format),
                'confidence_score': response.confidence_score,
                **response.metadata
            })
        
        return pd.DataFrame(metadata_rows)
    
    def validate_structured_data(self, responses: List[FormResponse]) -> Dict[str, Any]:
        """Validate structured data quality."""
        validation_results = {
            'total_responses': len(responses),
            'valid_responses': 0,
            'issues': []
        }
        
        for idx, response in enumerate(responses):
            issues = []
            
            # Check confidence scores
            if response.confidence_score < 0.5:
                issues.append(f"Low confidence score: {response.confidence_score:.2f}")
            
            # Check for empty fields
            if not response.fields:
                issues.append("No fields detected")
            
            # Check for suspicious patterns
            checkbox_values = [f.get('value') for f in response.fields.values() 
                             if isinstance(f, dict) and 'value' in f]
            
            if checkbox_values:
                all_none = all(v is None for v in checkbox_values)
                all_same = len(set(checkbox_values)) == 1
                
                if all_none:
                    issues.append("All checkboxes have unknown state")
                elif all_same and len(checkbox_values) > 3:
                    issues.append("All checkboxes have same state (suspicious)")
            
            if not issues:
                validation_results['valid_responses'] += 1
            else:
                validation_results['issues'].append({
                    'response_index': idx,
                    'document_id': response.document_id,
                    'page_number': response.page_number,
                    'issues': issues
                })
        
        validation_results['quality_score'] = (
            validation_results['valid_responses'] / validation_results['total_responses']
            if validation_results['total_responses'] > 0 else 0
        )
        
        return validation_results