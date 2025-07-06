import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

def measure_boxes_cli(pdf_path, output_path="measured_boxes.png"):
    """Measure boxes in PDF and save annotated image"""
    
    print(f"Loading PDF: {pdf_path}")
    # Convert PDF to image
    pages = convert_from_path(pdf_path, dpi=200)
    img = np.array(pages[0])
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find rectangular boxes
    boxes = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter small noise (adjust threshold as needed)
            if area > 500:  # Minimum area
                boxes.append((x, y, w, h))
    
    # Sort boxes by area
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    
    # Draw boxes and labels on image
    annotated = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes[:50]):  # Show top 50 boxes
        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Add size label
        label = f"{w}x{h}"
        cv2.putText(annotated, label, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Save annotated image
    cv2.imwrite(output_path, annotated)
    print(f"Saved annotated image to: {output_path}")
    
    # Print statistics
    if boxes:
        print(f"\nFound {len(boxes)} boxes")
        print("\nTop 10 boxes by size:")
        for i, (x, y, w, h) in enumerate(boxes[:10]):
            print(f"  Box {i+1}: {w}x{h} pixels at position ({x}, {y})")
        
        # Find common sizes (likely checkboxes)
        widths = [b[2] for b in boxes]
        heights = [b[3] for b in boxes]
        
        # Group similar sizes
        from collections import Counter
        width_counts = Counter(widths)
        height_counts = Counter(heights)
        
        print("\nMost common widths:")
        for width, count in width_counts.most_common(5):
            if count > 2:  # At least 3 occurrences
                print(f"  {width} pixels: {count} occurrences")
        
        print("\nMost common heights:")
        for height, count in height_counts.most_common(5):
            if count > 2:
                print(f"  {height} pixels: {count} occurrences")
        
        # Look for square boxes (likely checkboxes)
        square_boxes = [(w, h) for x, y, w, h in boxes if abs(w - h) < 5]
        if square_boxes:
            square_sizes = [w for w, h in square_boxes]
            avg_size = np.mean(square_sizes)
            print(f"\nFound {len(square_boxes)} square boxes (likely checkboxes)")
            print(f"Average size: {avg_size:.1f}x{avg_size:.1f} pixels")
    
    return boxes

def measure_with_threshold(pdf_path, min_area=100, max_area=10000):
    """Measure boxes with specific size constraints"""
    
    pages = convert_from_path(pdf_path, dpi=200)
    img = np.array(pages[0])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Try different edge detection parameters
    results = []
    
    for threshold1 in [30, 50, 70]:
        for threshold2 in [100, 150, 200]:
            edges = cv2.Canny(gray, threshold1, threshold2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                if min_area < area < max_area:
                    aspect_ratio = w / h if h > 0 else 0
                    # Look for roughly square boxes
                    if 0.8 < aspect_ratio < 1.2:
                        boxes.append((w, h))
            
            if boxes:
                results.append({
                    'params': (threshold1, threshold2),
                    'count': len(boxes),
                    'avg_size': np.mean([w for w, h in boxes])
                })
    
    # Find most consistent result
    if results:
        best = max(results, key=lambda r: r['count'])
        print(f"\nBest parameters: Canny({best['params'][0]}, {best['params'][1]})")
        print(f"Found {best['count']} checkbox-like squares")
        print(f"Average size: {best['avg_size']:.1f} pixels")
    
    return results

# Simple coordinate picker for precise measurement
def pick_coordinates(image_path):
    """Simple method to get coordinates from image"""
    
    if image_path.endswith('.pdf'):
        pages = convert_from_path(image_path, dpi=200)
        img = np.array(pages[0])
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Save as temporary image for viewing
    Image.fromarray(img).save("temp_measure.png")
    
    print("\nImage saved as 'temp_measure.png'")
    print("Open the image in an image viewer to see pixel coordinates")
    print("Most image viewers show cursor position in the status bar")
    print("\nAlternatively, use this method:")
    print("1. Take a screenshot of a small area with known boxes")
    print("2. Open in any image editor (GIMP, Paint, etc.)")
    print("3. Use the selection tool to measure box dimensions")
    
    return img

if __name__ == "__main__":
    # Example usage
    pdf_file = "your_survey.pdf"  # Replace with your file
    
    # Method 1: Automatic detection with annotation
    # boxes = measure_boxes_cli(pdf_file)
    
    # Method 2: Find checkbox-sized squares
    # measure_with_threshold(pdf_file, min_area=100, max_area=2000)
    
    # Method 3: Manual coordinate picking
    # pick_coordinates(pdf_file)