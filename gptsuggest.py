import cv2
import numpy as np
from typing import List, Tuple

def detect_qr_code(image_path: str) -> List[Tuple]:
    # Read the input image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use the OpenCV QRCodeDetector
    detector = cv2.QRCodeDetector()

    return detector.detectAndDecodeMulti(gray)

# Test the function
if __name__ == "__main__":
    detected_qrs = detect_qr_code('qr_code_test.png')
    print(f"Detected QR codes: {detected_qrs}")
