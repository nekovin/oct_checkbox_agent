"""PDF processing module for handling PDF files and image conversion."""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Union, Generator
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from pdf2image import convert_from_path, convert_from_bytes
import cv2
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import get_config
from .utils import timing_decorator, enhance_contrast, deskew_image, save_debug_image, validate_image


class PDFProcessor:
    """Handles PDF file processing and conversion to images."""
    
    def __init__(self, config=None):
        """Initialize PDF processor with configuration."""
        self.config = config or get_config()
        self.dpi = self.config.processing.dpi
        self.enable_preprocessing = self.config.processing.enable_preprocessing
        
    @timing_decorator
    def process_pdf(self, pdf_path: Union[str, Path], 
                   page_numbers: Optional[List[int]] = None) -> List[Tuple[np.ndarray, dict]]:
        """
        Process PDF file and convert to images.
        
        Args:
            pdf_path: Path to PDF file
            page_numbers: Specific pages to process (None for all)
            
        Returns:
            List of tuples containing (image, metadata)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract metadata
        metadata = self._extract_metadata(pdf_path)
        
        # Convert PDF to images
        images = self._pdf_to_images(pdf_path, page_numbers)
        
        # Process images
        processed_images = []
        for idx, image in enumerate(images):
            page_metadata = {
                **metadata,
                'page_number': idx + 1,
                'total_pages': len(images)
            }
            
            if self.enable_preprocessing:
                image = self._preprocess_image(image)
            
            processed_images.append((image, page_metadata))
        
        logger.info(f"Processed {len(processed_images)} pages from {pdf_path}")
        return processed_images
    
    def process_pdf_batch(self, pdf_paths: List[Union[str, Path]], 
                         parallel: bool = True) -> Generator[Tuple[Path, List[Tuple[np.ndarray, dict]]], None, None]:
        """
        Process multiple PDF files in batch.
        
        Args:
            pdf_paths: List of PDF file paths
            parallel: Whether to process in parallel
            
        Yields:
            Tuples of (pdf_path, processed_images)
        """
        if parallel and len(pdf_paths) > 1:
            with ThreadPoolExecutor(max_workers=self.config.processing.parallel_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_pdf, path): path 
                    for path in pdf_paths
                }
                
                for future in tqdm(as_completed(future_to_path), 
                                 total=len(pdf_paths), 
                                 desc="Processing PDFs"):
                    pdf_path = future_to_path[future]
                    try:
                        result = future.result()
                        yield (pdf_path, result)
                    except Exception as e:
                        logger.error(f"Error processing {pdf_path}: {e}")
                        yield (pdf_path, [])
        else:
            for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
                try:
                    result = self.process_pdf(pdf_path)
                    yield (pdf_path, result)
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {e}")
                    yield (pdf_path, [])
    
    def _pdf_to_images(self, pdf_path: Path, 
                      page_numbers: Optional[List[int]] = None) -> List[np.ndarray]:
        """Convert PDF pages to images."""
        try:
            # Try using pdf2image (requires poppler)
            if page_numbers:
                images = []
                for page_num in page_numbers:
                    page_images = convert_from_path(
                        pdf_path, 
                        dpi=self.dpi,
                        first_page=page_num,
                        last_page=page_num
                    )
                    images.extend(page_images)
            else:
                images = convert_from_path(pdf_path, dpi=self.dpi)
            
            # Convert PIL images to numpy arrays
            return [np.array(img) for img in images]
            
        except Exception as e:
            logger.warning(f"pdf2image failed, trying PyMuPDF: {e}")
            
            # Fallback to PyMuPDF
            return self._pdf_to_images_pymupdf(pdf_path, page_numbers)
    
    def _pdf_to_images_pymupdf(self, pdf_path: Path, 
                              page_numbers: Optional[List[int]] = None) -> List[np.ndarray]:
        """Convert PDF pages to images using PyMuPDF."""
        images = []
        
        with fitz.open(pdf_path) as pdf:
            pages_to_process = page_numbers or range(1, pdf.page_count + 1)
            
            for page_num in pages_to_process:
                page = pdf[page_num - 1]  # 0-indexed
                
                # Render page to image
                mat = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to numpy array
                img = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img.reshape(pix.h, pix.w, pix.n)
                
                # Convert RGBA to RGB if necessary
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                
                images.append(img)
        
        return images
    
    def _extract_metadata(self, pdf_path: Path) -> dict:
        """Extract metadata from PDF file."""
        metadata = {
            'filename': pdf_path.name,
            'filepath': str(pdf_path),
            'file_size': pdf_path.stat().st_size,
        }
        
        try:
            with fitz.open(pdf_path) as pdf:
                metadata.update({
                    'page_count': pdf.page_count,
                    'title': pdf.metadata.get('title', ''),
                    'author': pdf.metadata.get('author', ''),
                    'subject': pdf.metadata.get('subject', ''),
                    'creator': pdf.metadata.get('creator', ''),
                    'producer': pdf.metadata.get('producer', ''),
                    'creation_date': pdf.metadata.get('creationDate', ''),
                    'modification_date': pdf.metadata.get('modDate', ''),
                })
        except Exception as e:
            logger.warning(f"Could not extract metadata from {pdf_path}: {e}")
        
        return metadata
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to image."""
        if not validate_image(image):
            logger.warning("Invalid image provided for preprocessing")
            return image
        
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Enhance contrast
            image = enhance_contrast(image)
            
            # Deskew image
            image = deskew_image(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return image
    
    def extract_page_regions(self, image: np.ndarray, 
                           min_area: int = 1000) -> List[Tuple[int, int, int, int]]:
        """
        Extract distinct regions from a page (useful for form sections).
        
        Args:
            image: Page image
            min_area: Minimum area for a region to be considered
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))
        
        # Sort regions by position (top to bottom, left to right)
        regions.sort(key=lambda r: (r[1], r[0]))
        
        return regions
    
    def save_processed_images(self, images: List[Tuple[np.ndarray, dict]], 
                            output_dir: Union[str, Path]) -> List[Path]:
        """Save processed images to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for idx, (image, metadata) in enumerate(images):
            filename = f"{metadata['filename']}_page_{metadata.get('page_number', idx+1)}.png"
            filepath = output_dir / filename
            
            # Convert to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            cv2.imwrite(str(filepath), image_bgr)
            saved_paths.append(filepath)
            logger.debug(f"Saved processed image: {filepath}")
        
        return saved_paths