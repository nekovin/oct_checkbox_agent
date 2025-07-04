"""Main OCR Checkbox Agent processing script."""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger

from .pdf_processor import PDFProcessor
from .ocr_engine import OCREngine
from .checkbox_detector import CheckboxDetector
from .data_structurer import DataStructurer, FormResponse, ExtractionResult
from .config import Config, FormTemplate, get_config, set_config


class OCRCheckboxAgent:
    """Main OCR Checkbox Agent class."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the OCR Checkbox Agent."""
        self.config = config or get_config()
        set_config(self.config)
        
        # Initialize components
        self.pdf_processor = PDFProcessor(self.config)
        self.ocr_engine = OCREngine(self.config)
        self.checkbox_detector = CheckboxDetector(self.config)
        self.data_structurer = DataStructurer(self.config)
        
        # Form template
        self.form_template: Optional[FormTemplate] = None
        
        logger.info("OCR Checkbox Agent initialized")
    
    def load_template(self, template_path: Path) -> None:
        """Load a form template."""
        self.form_template = FormTemplate.load(template_path)
        logger.info(f"Loaded form template: {self.form_template.name}")
    
    def process_pdf(self, pdf_path: Path, 
                   use_template: bool = False,
                   save_debug: bool = False) -> ExtractionResult:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            use_template: Whether to use loaded template
            save_debug: Whether to save debug images
            
        Returns:
            Extraction result
        """
        start_time = time.time()
        errors = []
        warnings = []
        responses = []
        
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Convert PDF to images
            pages = self.pdf_processor.process_pdf(pdf_path)
            
            for page_idx, (image, metadata) in enumerate(pages):
                try:
                    # Process single page
                    page_response = self._process_page(
                        image, metadata, page_idx + 1, 
                        use_template, save_debug
                    )
                    
                    if page_response:
                        responses.append(page_response)
                    
                except Exception as e:
                        error_msg = f"Error processing page {page_idx + 1}: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        
                        if save_debug:
                            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            processing_time = time.time() - start_time
            
            result = ExtractionResult(
                document_path=str(pdf_path),
                total_pages=len(pages),
                responses=responses,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings
            )
            
            logger.info(f"Completed processing {pdf_path}: {len(responses)} responses in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Failed to process PDF {pdf_path}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return ExtractionResult(
                document_path=str(pdf_path),
                total_pages=0,
                responses=[],
                processing_time=time.time() - start_time,
                errors=errors,
                warnings=warnings
            )
    
    def _process_page(self, image, metadata: Dict[str, Any], 
                     page_number: int, use_template: bool = False,
                     save_debug: bool = False) -> Optional[FormResponse]:
        """Process a single page."""
        # Extract text using OCR
        text_result = self.ocr_engine.extract_text(image)
        
        # Detect checkboxes
        checkboxes = self.checkbox_detector.detect_checkboxes(
            image, text_result['text_blocks']
        )
        
        if not checkboxes:
            logger.warning(f"No checkboxes found on page {page_number}")
            return None
        
        # Save debug images if requested
        if save_debug:
            debug_dir = self.config.temp_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            
            # Save original image
            import cv2
            cv2.imwrite(str(debug_dir / f"page_{page_number}_original.png"), image)
            
            # Save visualization
            vis_image = self.checkbox_detector.visualize_detections(image, checkboxes)
            cv2.imwrite(str(debug_dir / f"page_{page_number}_detected.png"), vis_image)
        
        # Structure the data
        response = self.data_structurer.structure_checkbox_data(
            checkboxes, metadata, page_number
        )
        
        return response
    
    def process_batch(self, input_paths: List[Path], 
                     output_file: Optional[Path] = None,
                     parallel: bool = True,
                     save_debug: bool = False) -> List[ExtractionResult]:
        """
        Process multiple PDF files in batch.
        
        Args:
            input_paths: List of PDF file paths
            output_file: Optional output file path
            parallel: Whether to process in parallel
            save_debug: Whether to save debug images
            
        Returns:
            List of extraction results
        """
        logger.info(f"Starting batch processing of {len(input_paths)} files")
        
        results = []
        
        if parallel and len(input_paths) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.processing.parallel_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_pdf, path, False, save_debug): path
                    for path in input_paths
                }
                
                for future in tqdm(as_completed(future_to_path), 
                                 total=len(input_paths), 
                                 desc="Processing PDFs"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        path = future_to_path[future]
                        logger.error(f"Failed to process {path}: {e}")
                        results.append(ExtractionResult(
                            document_path=str(path),
                            total_pages=0,
                            responses=[],
                            processing_time=0,
                            errors=[str(e)],
                            warnings=[]
                        ))
        else:
            # Sequential processing
            for pdf_path in tqdm(input_paths, desc="Processing PDFs"):
                result = self.process_pdf(pdf_path, False, save_debug)
                results.append(result)
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        logger.info(f"Batch processing completed: {len(results)} files processed")
        return results
    
    def save_results(self, results: List[ExtractionResult], 
                    output_path: Path) -> None:
        """Save extraction results to file."""
        # Collect all responses
        all_responses = []
        for result in results:
            all_responses.extend(result.responses)
        
        if not all_responses:
            logger.warning("No responses to save")
            return
        
        # Determine output format from file extension
        suffix = output_path.suffix.lower()
        
        if suffix == '.csv':
            self.data_structurer.export_to_csv(all_responses, output_path)
        elif suffix in ['.xlsx', '.xls']:
            self.data_structurer.export_to_excel(all_responses, output_path)
        elif suffix == '.json':
            self.data_structurer.export_to_json(all_responses, output_path)
        else:
            # Default to CSV
            csv_path = output_path.with_suffix('.csv')
            self.data_structurer.export_to_csv(all_responses, csv_path)
            logger.info(f"Unknown format, saved as CSV: {csv_path}")
        
        # Create extraction report
        report_path = output_path.parent / f"{output_path.stem}_report.json"
        self.data_structurer.create_extraction_report(results, report_path)
        
        # Validate data quality
        validation_results = self.data_structurer.validate_structured_data(all_responses)
        logger.info(f"Data validation: {validation_results['quality_score']:.2f} quality score")
        
        if validation_results['issues']:
            logger.warning(f"Found {len(validation_results['issues'])} data quality issues")
    
    def process_directory(self, input_dir: Path, 
                         output_dir: Path,
                         pattern: str = "*.pdf",
                         parallel: bool = True) -> List[ExtractionResult]:
        """Process all PDF files in a directory."""
        # Find all PDF files
        pdf_files = list(input_dir.glob(pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
        
        # Process files
        results = self.process_batch(pdf_files, None, parallel)
        
        # Save individual results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            if result.responses:
                filename = Path(result.document_path).stem
                output_file = output_dir / f"{filename}_extracted.{self.config.output.output_format}"
                
                self.data_structurer.export_to_csv([r for r in result.responses], output_file)
        
        # Save consolidated results
        consolidated_file = output_dir / f"consolidated_results.{self.config.output.output_format}"
        self.save_results(results, consolidated_file)
        
        return results


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(description="OCR Checkbox Extraction Agent")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-c", "--config", help="Configuration file path")
    parser.add_argument("-t", "--template", help="Form template file path")
    parser.add_argument("-f", "--format", choices=["csv", "xlsx", "json"], 
                       default="csv", help="Output format")
    parser.add_argument("--parallel", action="store_true", 
                       help="Enable parallel processing")
    parser.add_argument("--debug", action="store_true", 
                       help="Save debug images")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Load configuration
    config = Config()
    if args.config:
        config = Config.load(Path(args.config))
    
    # Override output format if specified
    if args.format:
        config.output.output_format = args.format
    
    # Initialize agent
    agent = OCRCheckboxAgent(config)
    
    # Load template if specified
    if args.template:
        agent.load_template(Path(args.template))
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file processing
        result = agent.process_pdf(input_path, save_debug=args.debug)
        
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_extracted.{args.format}"
        
        agent.save_results([result], output_path)
        
    elif input_path.is_dir():
        # Directory processing
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = input_path / "extracted"
        
        results = agent.process_directory(
            input_path, output_dir, parallel=args.parallel
        )
        
        logger.info(f"Processed {len(results)} files")
    
    else:
        logger.error(f"Input path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()