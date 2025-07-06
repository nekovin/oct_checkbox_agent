#!/usr/bin/env python3
"""
Complete Survey Scanner Pipeline
Processes PDF forms to extract checkbox states and correlate with labels
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Import our custom modules
from enhanced_checkbox_detector import EnhancedCheckboxDetector
from checkbox_fill_detector import CheckboxFillDetector
from text_extractor import TextExtractor
from checkbox_label_correlator import CheckboxLabelCorrelator

class SurveyScannerPipeline:
    """
    Complete pipeline for processing survey forms
    """
    
    def __init__(self, 
                 dpi: int = 200,
                 checkbox_width_range: Tuple[int, int] = (20, 40),
                 checkbox_height_range: Tuple[int, int] = (20, 40),
                 output_dir: str = "output",
                 debug: bool = False):
        
        self.dpi = dpi
        self.checkbox_width_range = checkbox_width_range
        self.checkbox_height_range = checkbox_height_range
        self.output_dir = output_dir
        self.debug = debug
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.checkbox_detector = EnhancedCheckboxDetector(dpi=dpi)
        self.fill_detector = CheckboxFillDetector(debug=debug)
        self.text_extractor = TextExtractor(debug=debug)
        self.correlator = CheckboxLabelCorrelator(debug=debug)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("Survey Scanner Pipeline initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('survey_scanner')
        
        if not logger.handlers:
            level = logging.DEBUG if self.debug else logging.INFO
            logger.setLevel(level)
            
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_pdf(self, pdf_path: str, page_num: int = 0) -> Dict[str, Any]:
        """
        Process a PDF form and extract survey data
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number to process (0-indexed)
            
        Returns:
            Dictionary containing survey results and metadata
        """
        
        self.logger.info(f"Starting processing of {pdf_path}, page {page_num}")
        
        try:
            # Step 1: Load PDF page as image
            self.logger.info("Step 1: Loading PDF page")
            image = self.checkbox_detector.load_pdf_page(pdf_path, page_num)
            self.logger.info(f"Loaded image with shape: {image.shape}")
            
            # Step 2: Detect checkboxes
            self.logger.info("Step 2: Detecting checkboxes")
            checkboxes = self.checkbox_detector.detect_checkboxes(
                image, 
                width_range=self.checkbox_width_range,
                height_range=self.checkbox_height_range
            )
            self.logger.info(f"Detected {len(checkboxes)} checkboxes")
            
            # Step 3: Analyze checkbox fill status
            self.logger.info("Step 3: Analyzing checkbox fill status")
            analyzed_checkboxes = self.fill_detector.analyze_checkboxes(image, checkboxes)
            filled_count = sum(1 for cb in analyzed_checkboxes if cb['is_filled'])
            self.logger.info(f"Found {filled_count} filled checkboxes")
            
            # Step 4: Extract text
            self.logger.info("Step 4: Extracting text")
            form_structure = self.text_extractor.extract_form_structure(image)
            self.logger.info(f"Extracted {len(form_structure['text_regions'])} text regions")
            
            # Step 5: Correlate checkboxes with labels
            self.logger.info("Step 5: Correlating checkboxes with labels")
            
            # Use different text types based on what's available
            potential_labels = []
            potential_labels.extend(form_structure['options'])  # Option text
            potential_labels.extend(form_structure['other_text'])  # Other text
            
            if not potential_labels:
                potential_labels = form_structure['text_regions']  # All text as fallback
            
            correlations = self.correlator.correlate_adaptive(
                analyzed_checkboxes, 
                potential_labels
            )
            
            # Step 6: Create structured survey data
            self.logger.info("Step 6: Creating survey structure")
            survey_data = self.correlator.create_survey_structure(correlations)
            
            # Step 7: Save visualizations and results
            self.logger.info("Step 7: Saving results")
            results = self._save_results(
                pdf_path, 
                page_num, 
                image, 
                analyzed_checkboxes, 
                form_structure, 
                correlations, 
                survey_data
            )
            
            self.logger.info("Processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            raise
    
    def _save_results(self, 
                     pdf_path: str, 
                     page_num: int, 
                     image: np.ndarray,
                     checkboxes: List[Dict[str, Any]], 
                     form_structure: Dict[str, Any],
                     correlations: List[Dict[str, Any]], 
                     survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save processing results and visualizations"""
        
        # Create unique filename base
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"{base_name}_page{page_num}_{timestamp}"
        
        # Save visualizations
        viz_paths = {}
        
        # 1. Checkbox detection visualization
        checkbox_viz_path = os.path.join(self.output_dir, f"{output_base}_checkboxes.png")
        self.checkbox_detector.visualize_detections(image, checkboxes, checkbox_viz_path)
        viz_paths['checkboxes'] = checkbox_viz_path
        
        # 2. Fill analysis visualization
        fill_viz_path = os.path.join(self.output_dir, f"{output_base}_fill_analysis.png")
        self.fill_detector.visualize_fill_analysis(image, checkboxes, fill_viz_path)
        viz_paths['fill_analysis'] = fill_viz_path
        
        # 3. Text extraction visualization
        text_viz_path = os.path.join(self.output_dir, f"{output_base}_text.png")
        self.text_extractor.visualize_text_extraction(image, form_structure['text_regions'], text_viz_path)
        viz_paths['text_extraction'] = text_viz_path
        
        # 4. Correlation visualization
        correlation_viz_path = os.path.join(self.output_dir, f"{output_base}_correlations.png")
        self.correlator.visualize_correlations(image, correlations, correlation_viz_path)
        viz_paths['correlations'] = correlation_viz_path
        
        # Save JSON results
        json_path = os.path.join(self.output_dir, f"{output_base}_results.json")
        json_data = {
            'metadata': {
                'source_file': pdf_path,
                'page_number': page_num,
                'processed_at': datetime.now().isoformat(),
                'image_shape': image.shape,
                'processing_parameters': {
                    'dpi': self.dpi,
                    'checkbox_width_range': self.checkbox_width_range,
                    'checkbox_height_range': self.checkbox_height_range
                }
            },
            'survey_data': survey_data,
            'form_structure': {
                'statistics': form_structure['statistics'],
                'questions_found': len(form_structure['questions']),
                'options_found': len(form_structure['options'])
            },
            'visualizations': viz_paths
        }
        
        # Make JSON serializable
        json_data = self._make_json_serializable(json_data)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {json_path}")
        
        return json_data
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def batch_process(self, pdf_files: List[str], 
                     pages: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Process multiple PDF files"""
        
        if pages is None:
            pages = [0] * len(pdf_files)  # Process first page of each
        elif len(pages) == 1:
            pages = pages * len(pdf_files)  # Use same page for all files
        
        results = []
        
        for pdf_file, page_num in zip(pdf_files, pages):
            try:
                self.logger.info(f"Processing {pdf_file}, page {page_num}")
                result = self.process_pdf(pdf_file, page_num)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file}: {str(e)}")
                results.append({
                    'error': str(e),
                    'file': pdf_file,
                    'page': page_num
                })
        
        return results
    
    def create_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary report from batch processing results"""
        
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        if not successful_results:
            return {
                'summary': 'No successful processing results',
                'total_files': len(results),
                'successful': 0,
                'failed': len(failed_results),
                'errors': [r['error'] for r in failed_results]
            }
        
        # Aggregate statistics
        total_checkboxes = sum(r['survey_data']['metadata']['total_checkboxes'] for r in successful_results)
        total_filled = sum(sum(1 for item in r['survey_data']['survey_items'] if item['is_filled']) for r in successful_results)
        
        match_rates = [r['survey_data']['metadata']['match_rate'] for r in successful_results]
        avg_match_rate = np.mean(match_rates) if match_rates else 0
        
        return {
            'summary': {
                'total_files_processed': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'total_checkboxes_found': total_checkboxes,
                'total_checkboxes_filled': total_filled,
                'overall_fill_rate': total_filled / total_checkboxes if total_checkboxes > 0 else 0,
                'average_match_rate': avg_match_rate
            },
            'file_results': successful_results,
            'errors': [{'file': r['file'], 'error': r['error']} for r in failed_results]
        }

def main():
    """Example usage of the pipeline"""
    
    # Initialize pipeline
    pipeline = SurveyScannerPipeline(
        dpi=200,
        checkbox_width_range=(20, 40),
        checkbox_height_range=(20, 40),
        debug=True
    )
    
    # Process single PDF
    pdf_path = "output.pdf"  # Replace with your PDF
    
    if os.path.exists(pdf_path):
        results = pipeline.process_pdf(pdf_path, page_num=0)
        
        # Print summary
        survey_data = results['survey_data']
        metadata = survey_data['metadata']
        
        print(f"\n{'='*50}")
        print(f"SURVEY SCANNER RESULTS")
        print(f"{'='*50}")
        print(f"File: {pdf_path}")
        print(f"Total checkboxes: {metadata['total_checkboxes']}")
        print(f"Matched labels: {metadata['matched_labels']}")
        print(f"Match rate: {metadata['match_rate']:.1%}")
        print(f"Avg correlation confidence: {metadata['avg_correlation_confidence']:.3f}")
        
        print(f"\nFilled checkboxes:")
        for item in survey_data['survey_items']:
            if item['is_filled']:
                print(f"  ✓ {item['label']} (confidence: {item['fill_confidence']:.2f})")
        
        print(f"\nEmpty checkboxes:")
        for item in survey_data['survey_items']:
            if not item['is_filled']:
                print(f"  ✗ {item['label']} (confidence: {item['fill_confidence']:.2f})")
        
        print(f"\nVisualization files saved in: output/")
        
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please replace 'output.pdf' with your actual PDF file path")

if __name__ == "__main__":
    main()