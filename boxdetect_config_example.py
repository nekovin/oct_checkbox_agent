from boxdetect import config
from boxdetect.pipelines import get_boxes
from pdf2image import convert_from_path
import numpy as np

def configure_boxdetect_for_pdf(pdf_path):
    """
    Configure boxdetect for PDF checkbox detection
    """
    
    # Convert PDF to image first
    # Note: boxdetect works on images, not PDFs directly
    pages = convert_from_path(pdf_path, dpi=200)  # 200 DPI is usually good
    img = np.array(pages[0])  # First page
    
    # Create config
    cfg = config.PipelinesConfig()
    
    # IMPORTANT: These values depend on your DPI and actual checkbox size
    # For 200 DPI PDFs, typical checkbox sizes:
    # - Small checkboxes: 15-25 pixels
    # - Medium checkboxes: 25-35 pixels  
    # - Large checkboxes: 35-45 pixels
    
    # Your current config (9-20) seems too small for 200 DPI
    # Try these instead:
    cfg.width_range = (20, 40)   # Adjust based on your checkboxes
    cfg.height_range = (20, 40)  # Adjust based on your checkboxes
    
    # Multiple scaling factors improve detection
    cfg.scaling_factors = [0.8, 1.0, 1.2, 1.5]  # More scales = better detection
    
    # For square checkboxes (w/h ratio close to 1)
    cfg.wh_ratio_range = (0.8, 1.2)  # Allow slightly rectangular boxes
    
    # Keep group_size_range = (1, 1) for individual checkboxes
    cfg.group_size_range = (1, 1)
    
    # For clean PDFs, minimal preprocessing is better
    cfg.dilation_iterations = 0  # No dilation for sharp PDFs
    cfg.blur_size = (1, 1)      # Minimal blur
    
    # Additional optimizations for PDFs
    cfg.morph_kernels_type = 'rectangles'  # Better for rectangular shapes
    cfg.horizontal_max_distance_multiplier = 2.0  # For grouped checkboxes
    
    return cfg, img

def test_detection(pdf_path):
    """Test boxdetect with different configurations"""
    
    # Get configured settings and image
    cfg, img = configure_boxdetect_for_pdf(pdf_path)
    
    # Run detection
    rects, groups, output_image = get_boxes(img, cfg)
    
    print(f"Detected {len(rects)} boxes with current config")
    
    if len(rects) == 0:
        print("\nNo boxes detected! Try adjusting:")
        print("1. Increase width_range and height_range")
        print("2. Add more scaling_factors")
        print("3. Check if PDF needs higher DPI conversion")
        
        # Try auto-adjustment
        print("\nTrying broader range...")
        cfg.width_range = (10, 50)
        cfg.height_range = (10, 50)
        rects, groups, output_image = get_boxes(img, cfg)
        print(f"Detected {len(rects)} boxes with broader range")
    
    # Show detected box sizes
    if rects:
        sizes = [(r[2], r[3]) for r in rects]  # (width, height)
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        
        print(f"\nDetected box sizes:")
        print(f"Width range: {min(widths)}-{max(widths)} pixels")
        print(f"Height range: {min(heights)}-{max(heights)} pixels")
        print(f"Most common size: {max(set(sizes), key=sizes.count)}")
    
    return rects, output_image

# Example usage with your PDF
if __name__ == "__main__":
    # Your PDF file
    pdf_file = "output.pdf"
    
    # Option 1: Use suggested config
    cfg, img = configure_boxdetect_for_pdf(pdf_file)
    rects, groups, output = get_boxes(img, cfg)
    print(f"Found {len(rects)} checkboxes")
    
    # Option 2: Test and auto-adjust
    # rects, output = test_detection(pdf_file)