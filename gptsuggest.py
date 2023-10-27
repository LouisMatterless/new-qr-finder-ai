import cv2
import numpy as np
from typing import List, Tuple

def detect_qr_code(image_path: str) -> List[Tuple]:
    """
    Detect a QR code in an image, which may be rotated, distorted, or colored differently.
    
    Parameters:
    - image_path (str): The path to the image file where the QR code is to be detected.
    
    Outputs:
    - List of detected QR codes information (List[Tuple]).
      Each tuple contains (decoded_info, points)
    """
    
    # Read the input image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use the OpenCV QRCodeDetector
    detector = cv2.QRCodeDetector()
    
    # Detect and decode the QR code
    detection_success, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(gray)
    
    detected_qrs = []
    
    if detection_success:
        print("QR code(s) detected!")
        for i in range(len(decoded_info)):
            print(f"Decoded Info {i+1}: {decoded_info[i]}")
            pts = np.array(points[i], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            detected_qrs.append((decoded_info[i], points[i]))
        
        # Display the output image with detected QR codes
        cv2.imshow("Detected QR codes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No QR code detected.")
    
    return detected_qrs

# Test the function
if __name__ == "__main__":
    detected_qrs = detect_qr_code('test_photos/bak/IMG_2340b.jpg')
    print(f"Detected QR codes: {detected_qrs}")