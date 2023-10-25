import cv2
import numpy as np
from typing import List, Tuple

def detect_qr_code_with_cone_effect(image_path: str) -> List[Tuple]:
    """
    Simulate the effect of viewing a QR code from different angles around a cone and detect it.
    
    Parameters:
    - image_path (str): The path to the image file where the QR code is to be detected.
    
    Outputs:
    - List of detected QR codes information (List[Tuple]).
      Each tuple contains (decoded_info, points, angle)
    """
    # Read the input image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use the OpenCV QRCodeDetector
    detector = cv2.QRCodeDetector()
    
    detected_qrs = []

    for angle in range(0, 360, 10):
        # Simulate the perspective effect due to the cone
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = 50 * np.sin(np.radians(angle))  # simulate cone effect
        pts2 = np.float32([[0+offset, 0+offset], [w-offset, 0+offset], [0+offset, h-offset], [w-offset, h-offset]])

        M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_img = cv2.warpPerspective(gray, M_perspective, (w, h))
        
        cv2.imshow(f"Detected QR codes at {angle} degrees", perspective_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Detect and decode the QR code
        detection_success, decoded_info, points, _ = detector.detectAndDecodeMulti(perspective_img)
        
        if detection_success:
            print(f"QR code(s) detected at {angle} degrees!")
            for i in range(len(decoded_info)):
                print(f"Decoded Info {i+1}: {decoded_info[i]}")
                pts = np.array(points[i], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(perspective_img, [pts], True, (0, 255, 0), 2)
                detected_qrs.append((decoded_info[i], points[i], angle))

            # Uncomment the following lines if you want to display the rotated image where QR code is detected
            # cv2.imshow(f"Detected QR codes at {angle} degrees", perspective_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    
    if not detected_qrs:
        print("No QR code detected in any rotated images.")
    
    return detected_qrs

# Test the function
if __name__ == "__main__":
    detected_qrs = detect_qr_code_with_cone_effect('test_photos/bak/IMG_0169b.jpg')
    print(f"Detected QR codes: {detected_qrs}")
