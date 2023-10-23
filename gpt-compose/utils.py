

import os
import random
import string

import time

import qrcode

#region string

def generate_random_string(min_length=10, max_length=50):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

#tests
test_string = generate_random_string()
print(test_string)

#endregion

#region qr code

def create_qr_code(data):
    """
    Generate a QR code image from a given string.
    
    Parameters:
    - data (str): The string to encode in the QR code.
    
    Returns:
    - PIL.Image: The generated QR code as a PIL Image.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

# Test the function with a sample string
sample_string = "Hello, QR Code!"
qr_img = create_qr_code(sample_string)
qr_img.show()

#endregion

# Create the directory path
output_dir = f"outputs/gpt-compose/dump-{int(time.time())}"
os.makedirs(output_dir, exist_ok=True)


from PIL import Image
import numpy as np
import cv2
import random

def shear_qr_image(image):
    """
    shear_qr_image
    
    Applies a random shear transformation to the provided QR code image.
    
    Parameters:
    - image (PIL.Image): The QR code image to be sheared.
    
    Returns:
    - PIL.Image: The sheared QR code image.
    """
    # Convert PIL Image to numpy array (with dtype uint8) for transformation
    np_image = np.array(image, dtype=np.uint8)

    # Define shear transformation parameters
    shear_x = random.uniform(-0.3, 0.3)  # Random shear in X direction
    shear_y = random.uniform(-0.3, 0.3)  # Random shear in Y direction
    
    # Define transformation matrix for shearing
    M = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])
    
    # Apply the shear transformation
    sheared_image = cv2.warpAffine(np_image, M, (image.width, image.height))
    
    # Convert numpy array back to PIL Image
    return Image.fromarray(sheared_image)

def export_qr_image(image, full_path):
    """
    export_qr_image
    
    Saves the provided QR code image to the specified path.
    
    Parameters:
    - image (PIL.Image): The QR code image to be saved.
    - path (str): The relative path where the image should be saved.
    
    Returns:
    - str: The absolute path where the image has been saved.
    """
    image.save(full_path)
    return full_path

# Test the function
test_path = "test_qr_code.png"
saved_path = export_qr_image(shear_qr_image(qr_img), os.path.join(output_dir, test_path))
saved_path