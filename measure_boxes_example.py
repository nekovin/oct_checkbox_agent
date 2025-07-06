from measure_boxes import detect_boxes, visualize_boxes, get_box_statistics, measure_specific_region
from pdf2image import convert_from_path
import numpy as np

# Quick measurement for your PDF
def quick_measure(pdf_path):
    """Quick way to measure boxes in your PDF"""
    
    # Convert PDF to image
    print("Converting PDF to image...")
    pages = convert_from_path(pdf_path)
    img = np.array(pages[0])  # First page
    
    # Option 1: Automatic detection
    print("\nDetecting boxes automatically...")
    _, boxes = detect_boxes(pdf_path, is_pdf=True)
    
    if boxes:
        # Show statistics
        stats = get_box_statistics(boxes)
        
        # Show visualization
        visualize_boxes(img, boxes)
    else:
        print("No boxes detected automatically.")
    
    # Option 2: Manual measurement
    print("\nYou can also measure manually...")
    measure_specific_region(img)

# Example usage:
# quick_measure("your_survey.pdf")

# Or step by step:
if __name__ == "__main__":
    # Replace with your PDF file
    pdf_file = "your_survey.pdf"
    
    # Convert to image
    pages = convert_from_path(pdf_file)
    img = np.array(pages[0])
    
    # Detect boxes
    _, boxes = detect_boxes(pdf_file, is_pdf=True)
    
    # Print all box sizes
    print("\nDetected boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box['width']}x{box['height']} pixels at position ({box['x']}, {box['y']})")
    
    # Get average sizes (useful for checkboxes)
    if boxes:
        widths = [b['width'] for b in boxes]
        heights = [b['height'] for b in boxes]
        print(f"\nAverage box size: {np.mean(widths):.1f}x{np.mean(heights):.1f} pixels")