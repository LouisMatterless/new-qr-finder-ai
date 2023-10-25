import cv2
import numpy as np
from typing import List, Tuple

def detect_rotated_qr_code(image_path: str) -> List[Tuple]:
    """
    Detect a QR code in an image that may be rotated. The image is rotated degree by degree for 360 times.
    
    Parameters:
    - image_path (str): The path to the image file where the QR code is to be detected.
    
    Outputs:
    - List of detected QR codes information (List[Tuple]).
      Each tuple contains (decoded_info, points, angle)
    """
    
    # Read the input image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use the OpenCV QRCodeDetector
    detector = cv2.QRCodeDetector()
    
    detected_qrs = []

    for angle in range(0, 360, 10):
        # Rotate the image
        M = cv2.getRotationMatrix2D((gray.shape[1] // 2, gray.shape[0] // 2), angle, 1)
        rotated_gray = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
        
        # Detect and decode the QR code
        detection_success, decoded_info, points, _ = detector.detectAndDecodeMulti(rotated_gray)
        
        if detection_success:
            print(f"QR code(s) detected at {angle} degrees!")
            for i in range(len(decoded_info)):
                print(f"Decoded Info {i+1}: {decoded_info[i]}")
                pts = np.array(points[i], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(rotated_gray, [pts], True, (0, 255, 0), 2)
                detected_qrs.append((decoded_info[i], points[i], angle))

            # Uncomment the following lines if you want to display the rotated image where QR code is detected
            cv2.imshow(f"Detected QR codes at {angle} degrees", rotated_gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return detected_qrs
    
    if not detected_qrs:
        print("No QR code detected in any rotated images.")
    
    return detected_qrs

# Test the function
if __name__ == "__main__":
    detected_qrs = detect_rotated_qr_code('test_photos/bak/IMG_0169b.jpg')
    # detected_qrs = detect_rotated_qr_code('3_debug copy.jpg')
    print(f"Detected QR codes: {detected_qrs}")
