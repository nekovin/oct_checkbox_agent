import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import numpy as np

# Load your PDF
file_name = 'output.pdf'  # Replace with your file
pages = convert_from_path(file_name)
img = np.array(pages[0])

clicks = []

def onclick(event):
    if event.inaxes:
        clicks.append((event.xdata, event.ydata))
        if len(clicks) == 2:
            # Calculate box size
            width = abs(clicks[1][0] - clicks[0][0])
            height = abs(clicks[1][1] - clicks[0][1])
            print(f"Box size: {width:.0f} x {height:.0f} pixels")

            # Draw rectangle
            rect = Rectangle(
                (min(clicks[0][0], clicks[1][0]), min(clicks[0][1], clicks[1][1])),
                width, height,
                fill=False, edgecolor='red', linewidth=2
            )
            ax.add_patch(rect)
            plt.draw()
            clicks.clear()

fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(img)
ax.set_title("Click two opposite corners of a box to measure")
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()