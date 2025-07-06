#!/usr/bin/env python3
"""
Survey Scanner - Process PDF surveys to extract checkbox states
"""

from boxdetect import config
from boxdetect.pipelines import get_boxes
from pdf2image import convert_from_path
import numpy as np
import cv2
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import os


def detect_filled_checkboxes(pdf_path, page_num=0, dpi=200):
    """Detect checkboxes and determine which are filled"""
    
    # Convert PDF to image
    pages = convert_from_path(pdf_path, dpi=dpi)
    img = np.array(pages[page_num])
    
    # Configure BoxDetect
    cfg = config.PipelinesConfig()
    cfg.width_range = (20, 40)
    cfg.height_range = (20, 40)
    cfg.scaling_factors = [0.8, 1.0, 1.2, 1.5]
    cfg.wh_ratio_range = (0.8, 1.2)
    cfg.group_size_range = (1, 1)
    cfg.dilation_iterations = 0
    cfg.blur_size = (1, 1)
    cfg.morph_kernels_type = 'rectangles'
    
    # Detect checkboxes
    rects, groups, output_image = get_boxes(img, cfg)
    print(f"Found {len(rects)} checkboxes")
    
    # Analyze each checkbox for fill
    filled_checkboxes = []
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        checkbox_region = img[y:y+h, x:x+w]
        is_filled, fill_score = is_checkbox_filled(checkbox_region)
        
        checkbox_info = {
            'id': i,
            'bbox': (x, y, w, h),
            'is_filled': is_filled,
            'fill_score': fill_score
        }
        filled_checkboxes.append(checkbox_info)
        print(f"Checkbox {i}: {'FILLED' if is_filled else 'EMPTY'} (score: {fill_score:.3f})")
    
    return filled_checkboxes, img


def is_checkbox_filled(checkbox_region):
    """Determine if a checkbox is filled"""
    
    if len(checkbox_region.shape) == 3:
        gray = cv2.cvtColor(checkbox_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = checkbox_region
    
    h, w = gray.shape
    margin = max(2, min(w, h) // 6)
    inner_region = gray[margin:h-margin, margin:w-margin]
    
    if inner_region.size == 0:
        return False, 0.0
    
    dark_pixels = np.sum(inner_region < 200)
    total_pixels = inner_region.size
    dark_ratio = dark_pixels / total_pixels
    
    variance = np.var(inner_region)
    mean_intensity = np.mean(inner_region)
    
    dark_score = min(1.0, dark_ratio * 5)
    variance_score = min(1.0, variance / 1000)
    intensity_score = max(0, 1 - (mean_intensity / 255))
    
    fill_score = (dark_score * 0.5 + variance_score * 0.3 + intensity_score * 0.2)
    is_filled = fill_score > 0.15
    
    return is_filled, fill_score


def apply_exclusion_regions(img, filled_checkboxes, top_percent=10, bottom_percent=5):
    """Apply exclusion regions to filter out unwanted checkboxes"""
    
    img_height, img_width = img.shape[:2]
    
    exclusion_regions = []
    
    # Top region
    top_height = int(img_height * top_percent / 100)
    exclusion_regions.append((0, 0, img_width, top_height))
    
    # Bottom region
    bottom_height = int(img_height * bottom_percent / 100)
    bottom_y_start = img_height - bottom_height
    exclusion_regions.append((0, bottom_y_start, img_width, img_height))
    
    print(f"Exclusion regions: top {top_percent}%, bottom {bottom_percent}%")
    
    # Filter checkboxes
    filtered_checkboxes = []
    for checkbox in filled_checkboxes:
        if not is_in_exclusion_region(checkbox['bbox'], exclusion_regions):
            filtered_checkboxes.append(checkbox)
    
    print(f"Filtered from {len(filled_checkboxes)} to {len(filtered_checkboxes)} checkboxes")
    return filtered_checkboxes, exclusion_regions


def is_in_exclusion_region(bbox, exclusion_regions):
    """Check if checkbox is in exclusion region"""
    x, y, w, h = bbox
    checkbox_center_x = x + w // 2
    checkbox_center_y = y + h // 2
    
    for ex_x1, ex_y1, ex_x2, ex_y2 in exclusion_regions:
        if (ex_x1 <= checkbox_center_x <= ex_x2 and 
            ex_y1 <= checkbox_center_y <= ex_y2):
            return True
    
    return False


def remove_overlapping_boxes(checkboxes, overlap_threshold=0.5):
    """Remove overlapping duplicate boxes"""
    
    if not checkboxes:
        return checkboxes, []
    
    print(f"Checking for overlaps in {len(checkboxes)} boxes")
    
    sorted_boxes = sorted(checkboxes, key=lambda x: x['fill_score'], reverse=True)
    filtered_boxes = []
    removed_boxes = []
    
    for current_box in sorted_boxes:
        is_duplicate = False
        
        for kept_box in filtered_boxes:
            overlap = calculate_overlap_ratio(current_box['bbox'], kept_box['bbox'])
            
            if overlap > overlap_threshold:
                removed_boxes.append(current_box)
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_boxes.append(current_box)
    
    print(f"Removed {len(removed_boxes)} overlapping boxes")
    return filtered_boxes, removed_boxes


def calculate_overlap_ratio(box1, box2):
    """Calculate overlap ratio between two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    
    if left < right and top < bottom:
        intersection = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        smaller_area = min(area1, area2)
        overlap_ratio = intersection / smaller_area if smaller_area > 0 else 0
        return overlap_ratio
    return 0


def sort_checkboxes_by_rows(checkboxes, row_tolerance=15):
    """Sort checkboxes by rows, then left to right"""
    
    if not checkboxes:
        return []
    
    rows = []
    sorted_by_y = sorted(checkboxes, key=lambda cb: cb['bbox'][1])
    
    current_row = [sorted_by_y[0]]
    current_y = sorted_by_y[0]['bbox'][1]
    
    for checkbox in sorted_by_y[1:]:
        y = checkbox['bbox'][1]
        
        if abs(y - current_y) <= row_tolerance:
            current_row.append(checkbox)
        else:
            rows.append(current_row)
            current_row = [checkbox]
            current_y = y
    
    rows.append(current_row)
    
    for row in rows:
        row.sort(key=lambda cb: cb['bbox'][0])
    
    sorted_checkboxes = []
    for row in rows:
        sorted_checkboxes.extend(row)
    
    print(f"Detected {len(rows)} rows of checkboxes")
    return sorted_checkboxes


def visualize_results(img, checkboxes, exclusion_regions, output_path="checkbox_results.png"):
    """Create visualization of results"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    labeled_img = img.copy()
    
    # Draw exclusion regions
    for ex_x1, ex_y1, ex_x2, ex_y2 in exclusion_regions:
        overlay = labeled_img.copy()
        cv2.rectangle(overlay, (ex_x1, ex_y1), (ex_x2, ex_y2), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.2, labeled_img, 0.8, 0, labeled_img)
    
    # Draw checkboxes
    for i, checkbox in enumerate(checkboxes):
        x, y, w, h = checkbox['bbox']
        is_filled = checkbox['is_filled']
        
        if is_filled:
            box_color = (0, 200, 0)
            text_color = (0, 150, 0)
        else:
            box_color = (200, 0, 0)
            text_color = (150, 0, 0)
        
        cv2.rectangle(labeled_img, (x, y), (x + w, y + h), box_color, 3)
        
        box_label = f"box{i+1}"
        font_scale = 0.6
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        label_x = x - 5
        label_y = y - 10
        
        cv2.rectangle(labeled_img, 
                      (label_x - 2, label_y - text_height - 2), 
                      (label_x + text_width + 2, label_y + 2), 
                      (255, 255, 255), -1)
        
        cv2.putText(labeled_img, box_label, (label_x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
        status_symbol = "✓" if is_filled else "✗"
        symbol_x = x + w//2 - 8
        symbol_y = y + h//2 + 8
        cv2.putText(labeled_img, status_symbol, (symbol_x, symbol_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 3)
    
    ax.imshow(labeled_img)
    filled_count = sum(1 for cb in checkboxes if cb['is_filled'])
    ax.set_title(f"Survey Results (Total: {len(checkboxes)}, Filled: {filled_count})", 
                 fontsize=16, pad=20)
    ax.axis('off')
    
    filled_patch = Rectangle((0, 0), 1, 1, fc=(0, 0.8, 0), alpha=0.8)
    empty_patch = Rectangle((0, 0), 1, 1, fc=(0.8, 0, 0), alpha=0.8)
    ax.legend([filled_patch, empty_patch], ['Filled', 'Empty'], 
              loc='upper right', fontsize=12, frameon=True, facecolor='white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")


def create_dataframe(checkboxes):
    """Create DataFrame with results"""
    
    custom_names = [
        "1_yes", "1_no",
        "1b_1", "1b_2", "1b_3", "1b_4", "1b_5", "1b_6", "1b_7", "1b_8", 
        "1b_9", "1b_10", "1b_11", "1b_12", "1b_13", "1b_14", "1b_15", "1b_16",
        "2_1", "2_2", "2_3", "2_4", "2_5",
        "3_1", "6_1", "3_2", "6_2", "3_3", "6_3", "3_4",
        "4_1", "7_1", "4_2", "4_3", "7_2", "4_4", "4_5", "7_3", "7_4", 
        "5_1", "7_5", "5_2"
    ]
    
    while len(custom_names) < len(checkboxes):
        custom_names.append(f"box{len(custom_names)+1}")
    
    checkbox_values = [checkbox['is_filled'] for checkbox in checkboxes]
    custom_df = pd.DataFrame([checkbox_values], columns=custom_names[:len(checkboxes)])
    
    # Order columns
    column_order = (["1_yes", "1_no"] + 
                   [f"1b_{i}" for i in range(1, 17)] + 
                   [f"2_{i}" for i in range(1, 6)] + 
                   [f"3_{i}" for i in range(1, 5)] + 
                   [f"4_{i}" for i in range(1, 6)] + 
                   [f"5_{i}" for i in range(1, 3)] + 
                   [f"6_{i}" for i in range(1, 4)] + 
                   [f"7_{i}" for i in range(1, 6)])
    
    ordered_columns = [col for col in column_order if col in custom_df.columns]
    ordered_df = custom_df[ordered_columns]
    
    return ordered_df


def main():
    parser = argparse.ArgumentParser(description='Process survey PDF to extract checkbox states')
    parser.add_argument('pdf_file', help='Path to the PDF file')
    parser.add_argument('--page', type=int, default=0, help='Page number to process (0-indexed)')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for PDF conversion')
    parser.add_argument('--top-exclude', type=int, default=10, help='Top exclusion percentage')
    parser.add_argument('--bottom-exclude', type=int, default=5, help='Bottom exclusion percentage')
    parser.add_argument('--output', default='survey_results.csv', help='Output CSV filename')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file '{args.pdf_file}' not found")
        return
    
    print(f"Processing {args.pdf_file}")
    
    # Step 1: Detect checkboxes
    checkboxes, img = detect_filled_checkboxes(args.pdf_file, args.page, args.dpi)
    
    # Step 2: Apply exclusions
    filtered_checkboxes, exclusion_regions = apply_exclusion_regions(
        img, checkboxes, args.top_exclude, args.bottom_exclude
    )
    
    # Step 3: Remove overlaps
    deduplicated_checkboxes, _ = remove_overlapping_boxes(filtered_checkboxes)
    
    # Step 4: Sort by rows
    final_checkboxes = sort_checkboxes_by_rows(deduplicated_checkboxes)
    
    # Step 5: Create DataFrame
    df = create_dataframe(final_checkboxes)
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(df)
    
    # Visualize if requested
    if args.visualize:
        visualize_results(img, final_checkboxes, exclusion_regions)
    
    # Summary
    filled_count = sum(1 for cb in final_checkboxes if cb['is_filled'])
    print(f"\nSummary:")
    print(f"Total checkboxes: {len(final_checkboxes)}")
    print(f"Filled: {filled_count}")
    print(f"Empty: {len(final_checkboxes) - filled_count}")


if __name__ == "__main__":
    main()