#!/usr/bin/env python3
from pdf2image import convert_from_path
import cv2
import numpy as np
import sys

def quick_measure(pdf_path):
    """Quick measurement of boxes in PDF first page"""
    
    print(f"Analyzing: {pdf_path}")
    
    # Convert PDF
    pages = convert_from_path(pdf_path, dpi=200)
    img = np.array(pages[0])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Find edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Collect box sizes
    box_sizes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filter reasonable box sizes (adjust as needed)
        if 100 < area < 5000:  # Between 100 and 5000 pixels
            aspect_ratio = w / h if h > 0 else 0
            box_sizes.append({
                'width': w,
                'height': h,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'is_square': 0.9 < aspect_ratio < 1.1
            })
    
    # Group similar sizes
    if box_sizes:
        # Find square boxes (likely checkboxes)
        squares = [b for b in box_sizes if b['is_square']]
        
        if squares:
            sizes = [(b['width'] + b['height']) / 2 for b in squares]
            avg_size = np.mean(sizes)
            std_size = np.std(sizes)
            
            print(f"\nCheckbox Analysis:")
            print(f"Found {len(squares)} square boxes")
            print(f"Average size: {avg_size:.1f} ± {std_size:.1f} pixels")
            print(f"Size range: {min(sizes):.0f} - {max(sizes):.0f} pixels")
            
            # Show common sizes
            from collections import Counter
            size_counts = Counter([int(s) for s in sizes])
            print("\nMost common square sizes:")
            for size, count in size_counts.most_common(5):
                print(f"  {size} pixels: {count} boxes")
        
        # Find rectangular boxes
        rectangles = [b for b in box_sizes if not b['is_square']]
        if rectangles:
            print(f"\nRectangular boxes: {len(rectangles)}")
            # Group by similar dimensions
            similar_rects = {}
            for rect in rectangles:
                key = f"{rect['width']//10*10}x{rect['height']//10*10}"  # Round to nearest 10
                if key not in similar_rects:
                    similar_rects[key] = []
                similar_rects[key].append(rect)
            
            print("Common rectangular sizes:")
            for size, rects in sorted(similar_rects.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
                if len(rects) > 1:
                    print(f"  ~{size}: {len(rects)} boxes")
    
    # Save sample with boxes highlighted
    output = img.copy()
    for contour in contours[:100]:  # Draw first 100 contours
        x, y, w, h = cv2.boundingRect(contour)
        if 100 < w * h < 5000:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite("boxes_detected.png", output_bgr)
    print(f"\nSaved visualization to: boxes_detected.png")
    
    return box_sizes

if __name__ == "__main__":
    if len(sys.argv) > 1:
        quick_measure(sys.argv[1])
    else:
        print("Usage: python quick_measure.py your_survey.pdf")
        print("\nOr use in Python:")
        print("  from quick_measure import quick_measure")
        print("  boxes = quick_measure('your_survey.pdf')")