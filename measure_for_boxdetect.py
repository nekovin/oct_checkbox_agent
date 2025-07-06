from pdf2image import convert_from_path
import cv2
import numpy as np
from boxdetect import config

def measure_and_suggest_config(pdf_path, sample_page=0):
    """Measure boxes and suggest boxdetect configuration"""
    
    print(f"Analyzing {pdf_path} to suggest boxdetect config...")
    
    # Convert PDF at different DPIs to understand scaling
    for dpi in [200, 300]:
        print(f"\n--- Testing at {dpi} DPI ---")
        
        pages = convert_from_path(pdf_path, dpi=dpi)
        img = np.array(pages[sample_page])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Collect square boxes (likely checkboxes)
        squares = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Look for squares
            if 0.85 < aspect_ratio < 1.15 and area > 50:
                avg_size = (w + h) / 2
                squares.append(avg_size)
        
        if squares:
            # Calculate statistics
            sizes = sorted(squares)
            min_size = sizes[0]
            max_size = sizes[-1]
            avg_size = np.mean(sizes)
            std_size = np.std(sizes)
            
            # Remove outliers (boxes too different from average)
            filtered_sizes = [s for s in sizes if abs(s - avg_size) < 2 * std_size]
            if filtered_sizes:
                min_filtered = min(filtered_sizes)
                max_filtered = max(filtered_sizes)
                avg_filtered = np.mean(filtered_sizes)
                
                print(f"Found {len(filtered_sizes)} checkbox-like squares")
                print(f"Size range: {min_filtered:.0f}-{max_filtered:.0f} pixels")
                print(f"Average size: {avg_filtered:.1f} pixels")
                
                # Suggest configuration
                margin = 3  # Add some margin
                suggested_min = max(1, int(min_filtered - margin))
                suggested_max = int(max_filtered + margin)
                
                print(f"\nSuggested boxdetect config for {dpi} DPI:")
                print(f"cfg.width_range = ({suggested_min}, {suggested_max})")
                print(f"cfg.height_range = ({suggested_min}, {suggested_max})")
    
    print("\n--- Recommended Configuration ---")
    print("""
from boxdetect import config

cfg = config.PipelinesConfig()

# Based on your PDF analysis
cfg.width_range = (15, 35)   # Adjust based on output above
cfg.height_range = (15, 35)  # Adjust based on output above

# For better accuracy with PDFs
cfg.scaling_factors = [1.0, 1.2, 1.5]  # Multiple scales for robustness

# For square checkboxes
cfg.wh_ratio_range = (0.8, 1.2)  # Allow slightly rectangular boxes

# For individual checkboxes
cfg.group_size_range = (1, 1)

# Minimal dilation for cleaner edges
cfg.dilation_iterations = 0

# Additional recommended settings
cfg.morph_kernels_type = 'rectangles'  # Better for checkboxes
cfg.blur_size = (1, 1)  # Minimal blur for sharp PDFs
    """)
    
    return True

def test_boxdetect_config(pdf_path, width_range, height_range):
    """Test specific boxdetect configuration"""
    
    from boxdetect import config
    from boxdetect.pipelines import get_boxes
    
    # Convert PDF to image
    pages = convert_from_path(pdf_path, dpi=200)
    img = np.array(pages[0])
    
    # Setup config
    cfg = config.PipelinesConfig()
    cfg.width_range = width_range
    cfg.height_range = height_range
    cfg.scaling_factors = [1.0, 1.2, 1.5]
    cfg.wh_ratio_range = (0.8, 1.2)
    cfg.group_size_range = (1, 1)
    cfg.dilation_iterations = 0
    
    # Detect boxes
    rects, groups, image = get_boxes(img, cfg)
    
    print(f"\nTesting config: width_range={width_range}, height_range={height_range}")
    print(f"Found {len(rects)} boxes")
    
    if rects:
        # Analyze detected boxes
        widths = [r[2] for r in rects]
        heights = [r[3] for r in rects]
        
        print(f"Detected box sizes:")
        print(f"  Width range: {min(widths)}-{max(widths)}")
        print(f"  Height range: {min(heights)}-{max(heights)}")
        print(f"  Average: {np.mean(widths):.1f} x {np.mean(heights):.1f}")
    
    return len(rects)

if __name__ == "__main__":
    # First, analyze your PDF
    # measure_and_suggest_config("your_survey.pdf")
    
    # Then test different configurations
    # test_boxdetect_config("your_survey.pdf", (15, 35), (15, 35))
    
    pass