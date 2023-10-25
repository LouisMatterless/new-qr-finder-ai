import os
from PIL import Image

qr_folder = "outputs/output-transparentQR3/test_qr_codes"
qr_files = [f for f in os.listdir(qr_folder) if f != '.DS_Store']

for file in qr_files:
    # Construct the full path to the image file
    full_path = os.path.join(qr_folder, file)
    
    # Load the image
    img_6 = Image.open(full_path)

    # Create a new white image with the same size
    white_background = Image.new("RGBA", img_6.size, (255, 255, 255, 255))

    # Paste the image onto the white background
    white_background.paste(img_6, (0, 0), img_6)

    # Save the new image
    white_background.save(os.path.join(qr_folder, file + "_mod.png"))