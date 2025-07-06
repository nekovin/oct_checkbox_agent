from pdf2image import convert_from_path
from PIL import Image

# Convert PDF to PNG for easy measurement
pdf_file = "your_survey.pdf"  # Replace with your file
pages = convert_from_path(pdf_file, dpi=200)

# Save first page as PNG
pages[0].save("survey_page1.png")
print("Saved as survey_page1.png")
print("\nTo measure boxes:")
print("1. Open survey_page1.png in any image viewer/editor")
print("2. Use the selection tool or cursor coordinates")
print("3. Most viewers show pixel coordinates in status bar")
print("\nCommon checkbox sizes:")
print("- Small: 15x15 to 20x20 pixels")
print("- Medium: 25x25 to 30x30 pixels") 
print("- Large: 35x35 to 40x40 pixels")
print("(at 200 DPI resolution)")