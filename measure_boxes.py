import cv2
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def detect_boxes(image_path, is_pdf=False):
    """Detect boxes in image and measure their sizes"""
    
    # Load image
    if is_pdf:
        pages = convert_from_path(image_path)
        img = np.array(pages[0])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for rectangular shapes
    boxes = []
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's a rectangle (4 corners)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by minimum size (adjust as needed)
            if area > 100:  # Minimum 100 pixels area
                boxes.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'aspect_ratio': w/h if h > 0 else 0
                })
    
    return img, boxes

def visualize_boxes(img, boxes, title="Detected Boxes"):
    """Visualize detected boxes with measurements"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Draw boxes and labels
    for i, box in enumerate(boxes):
        rect = patches.Rectangle(
            (box['x'], box['y']), 
            box['width'], 
            box['height'],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add size label
        label = f"{box['width']}x{box['height']}"
        ax.text(box['x'], box['y']-5, label, 
                color='red', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_title(f"{title} - Found {len(boxes)} boxes")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return fig

def measure_specific_region(img, interactive=True):
    """Manually measure a region by clicking"""
    if interactive:
        print("Click and drag to select a region. Close window when done.")
        
        class RegionSelector:
            def __init__(self):
                self.start_point = None
                self.end_point = None
                self.selecting = False
                self.rect = None
                
            def on_press(self, event):
                if event.inaxes:
                    self.start_point = (int(event.xdata), int(event.ydata))
                    self.selecting = True
                    
            def on_motion(self, event):
                if self.selecting and event.inaxes:
                    if self.rect:
                        self.rect.remove()
                    x0, y0 = self.start_point
                    x1, y1 = int(event.xdata), int(event.ydata)
                    width = abs(x1 - x0)
                    height = abs(y1 - y0)
                    self.rect = patches.Rectangle(
                        (min(x0, x1), min(y0, y1)), 
                        width, height,
                        linewidth=2, edgecolor='green', facecolor='none'
                    )
                    ax.add_patch(self.rect)
                    ax.set_title(f"Selection: {width}x{height} pixels")
                    fig.canvas.draw()
                    
            def on_release(self, event):
                if self.selecting and event.inaxes:
                    self.end_point = (int(event.xdata), int(event.ydata))
                    self.selecting = False
                    x0, y0 = self.start_point
                    x1, y1 = self.end_point
                    width = abs(x1 - x0)
                    height = abs(y1 - y0)
                    print(f"Selected region: {width}x{height} pixels")
                    print(f"Position: ({min(x0,x1)}, {min(y0,y1)})")
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_title("Click and drag to measure a region")
        
        selector = RegionSelector()
        fig.canvas.mpl_connect('button_press_event', selector.on_press)
        fig.canvas.mpl_connect('motion_notify_event', selector.on_motion)
        fig.canvas.mpl_connect('button_release_event', selector.on_release)
        
        plt.show()

def get_box_statistics(boxes):
    """Get statistics about detected boxes"""
    if not boxes:
        return "No boxes detected"
    
    widths = [b['width'] for b in boxes]
    heights = [b['height'] for b in boxes]
    areas = [b['area'] for b in boxes]
    
    stats = {
        'count': len(boxes),
        'width': {
            'min': min(widths),
            'max': max(widths),
            'mean': np.mean(widths),
            'std': np.std(widths)
        },
        'height': {
            'min': min(heights),
            'max': max(heights),
            'mean': np.mean(heights),
            'std': np.std(heights)
        },
        'area': {
            'min': min(areas),
            'max': max(areas),
            'mean': np.mean(areas),
            'std': np.std(areas)
        }
    }
    
    print(f"Box Statistics ({len(boxes)} boxes found):")
    print(f"Width:  min={stats['width']['min']:.0f}, max={stats['width']['max']:.0f}, mean={stats['width']['mean']:.1f}±{stats['width']['std']:.1f}")
    print(f"Height: min={stats['height']['min']:.0f}, max={stats['height']['max']:.0f}, mean={stats['height']['mean']:.1f}±{stats['height']['std']:.1f}")
    print(f"Area:   min={stats['area']['min']:.0f}, max={stats['area']['max']:.0f}, mean={stats['area']['mean']:.1f}")
    
    return stats

# Example usage
if __name__ == "__main__":
    # For PDF
    # img, boxes = detect_boxes("your_survey.pdf", is_pdf=True)
    
    # For image
    # img, boxes = detect_boxes("your_survey.png", is_pdf=False)
    
    # Visualize detected boxes
    # visualize_boxes(img, boxes)
    
    # Get statistics
    # stats = get_box_statistics(boxes)
    
    # Manually measure a specific region
    # measure_specific_region(img)
    
    pass