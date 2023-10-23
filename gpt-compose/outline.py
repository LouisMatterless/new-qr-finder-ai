import cv2
import numpy as np

def extract_outlines(image_path, lower_threshold=50, upper_threshold=150):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    
    return edges

# Example usage:
edges = extract_outlines('2_debug.jpg')
cv2.imwrite('2_debug.jpg', edges)