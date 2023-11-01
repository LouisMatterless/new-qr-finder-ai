# Step 1: Generate folder of QR codes
# Step 2: Take a folder of photos and the QRs from previous step. Combine them into a new folder of photos with QRs randomly placed, in 1920x1080 resolution.
#         Randomize position, rotation, size and perspective on both the photo and QRs.
# Step 3: For each folder also output a text file with a list of center pixel coordinates, one per QR code, formatted as x|y one per line.
# Step 4: Also save the output pixel coordinates as a FullHD numpy array where each center pixel is 1 and others are 0. Saved as a .npy file.

import os
import random
import numpy as np
import cv2
import qrcode
import math
import sys
import shutil
from PIL import Image, ImageDraw
from tqdm import tqdm

import hashlib
from datetime import datetime
from qr_ai_helpers import adjusted_path, create_base_output_folder



def generate_qr_images(folder, num_images, qr_size):

# 10 words
    qr_random_words_1 = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "grey"]
    qr_random_words_2 = ["jumping", "running", "walking", "sitting", "standing", "sleeping", "eating", "drinking",
                        "playing", "flying"]
    qr_random_words_3 = ["cat", "dog", "bird", "fish", "mouse", "horse", "cow", "pig", "sheep", "goat"]

    print("Generating QR codes...")
    for i in tqdm(range(num_images)):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=qr_size,
            border=1,
        )

        qr_content = random.choice(qr_random_words_3)
        if random.choice([True, False]):
            qr_content = random.choice(qr_random_words_2) + " " + qr_content
        if random.choice([True, False]):
            qr_content = random.choice(qr_random_words_1) + " " + qr_content

        qr.add_data(qr_content)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Convert to RGBA and make white parts transparent
        img = img.convert("RGBA")
        datas = img.getdata()
        new_data = []
        for item in datas:
            # Check if the pixel is white
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        img.putdata(new_data)

        img.save(folder + "/" + str(i) + ".png", "PNG")

def generate_dataset(out_folder, photo_folder, qr_folder, num_images, image_size, max_qr_per_image):
    print("Generating dataset...")
    for i in tqdm(range(num_images)):
        while True:
            # Create new image
            img = Image.new('RGB', (image_size[0], image_size[1]), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)

            # Get random number of QR codes to place
            num_qr = random.randint(1, max_qr_per_image)

            # Get random QRs to place
            qr_files = os.listdir(qr_folder)
            qr_files = random.sample(qr_files, num_qr)

            # Get random photo
            photo_file = random.choice([f for f in os.listdir(photo_folder) if f != '.DS_Store'])
            photo = Image.open(photo_folder + "/" + photo_file)
            # Apply random perspective transformation and a random zoomed in crop (keeping same resolution)
            cropped_width = random.randint(photo.size[0] // 2, photo.size[0])
            cropped_height = cropped_width * image_size[1] // image_size[0] # keep aspect ratio
            cropped_pos_x = random.randint(0, photo.size[0] - cropped_width)
            cropped_pos_y = random.randint(0, photo.size[1] - cropped_height)
            photo = photo.crop((cropped_pos_x, cropped_pos_y, cropped_pos_x + cropped_width, cropped_pos_y + cropped_height))
            photo = photo.transform(photo.size, Image.PERSPECTIVE, data=(1, 0, 0, 0, 1, 0, 0, 0))
                                                                
            photo = photo.resize((image_size[0], image_size[1]))
            img.paste(photo, (0, 0))

            centers = []

            # Place QRs
            for qr_file in qr_files:
                # Get random QR
                qr_cv2, center_of_qr = load_and_transform_qr(qr_folder + "/" + qr_file)
                qr = Image.fromarray(qr_cv2)
                
                # Adjust transparency of QR code
                datas = qr.getdata()
                new_data = []
                for item in datas:
                    if isinstance(item, tuple):  # Ensure the item is a tuple
                        if item[3] in list(range(0, 1)): # transparent stays transparent
                            new_data.append((255, 255, 255, 0))  # fully transparent
                        elif item[0] in list(range(200, 256)):
                            new_data.append((255, 255, 255, 64))  # 1/4 transparent
                        else:
                            new_data.append((0, 0, 0, 128+32))  # more solid
                    else:  # not used
                        if item in list(range(200, 256)):
                            new_data.append(0)  # fully transparent
                        else:
                            new_data.append(128)  # half transparent
                qr.putdata(new_data)

                # for debug
                # qr.save(out_folder + "/" + str(generate_unique_hash()) + ".png")
                
                # Paste QR (don't paste close to edge, 20% of image size)
                w = img.size[0]
                h = img.size[1]
                random_pos = (random.randint(w // 5, w - w // 5 - qr.size[0]), random.randint(h // 5, h - h // 5 - qr.size[1]))
                img.paste(qr, random_pos, qr)

                # Save center
                center_x = random_pos[0] + center_of_qr[0]
                center_y = random_pos[1] + center_of_qr[1]
                centers.append((center_x, center_y))

            # make sure QRs don't overlap
            overlaps = False
            for a in range(len(centers)):
                for b in range(a + 1, len(centers)):
                    if math.sqrt((centers[a][0] - centers[b][0]) ** 2 + (centers[a][1] - centers[b][1]) ** 2) < 50:
                        overlaps = True
            if overlaps:
                print("Overlaps found, skipping image")
                continue

            # Save image
            img.save(out_folder + "/" + str(i) + ".jpg")

            # make center red for debug image
            img_debug = np.array(img)
            img_debug = cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB)
            for center in centers:
                img_debug = cv2.circle(img_debug, (round(center[0]), round(center[1])), 3, (0, 0, 255, 255), -1)
            cv2.imwrite(out_folder + "/" + str(i) + "_debug.jpg", img_debug)

            # Save centers
            with open(out_folder + "/" + str(i) + ".txt", "w") as f:
                for center in centers:
                    f.write(str(center[0]) + "|" + str(center[1]) + "\n")

            # Save centers as numpy array
            centers_np = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
            for center in centers:
                centers_np[round(center[0]), round(center[1])] = 1
            np.save(out_folder + "/" + str(i) + ".npy", centers_np)
            break

def random_transform(image, center):
    h, w, _ = image.shape

    # Random Scaling
    scale_factor = np.random.uniform(0.3, 1)

    M_scale = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
    # Update to scale around center
    M_scale[0, 2] = (1 - scale_factor) * center[0]
    M_scale[1, 2] = (1 - scale_factor) * center[1]

    # Random Rotation
    angle = np.random.uniform(0, 360)  # Random rotation angle between 0 and 360
    M_rot = cv2.getRotationMatrix2D(center, angle, 1)

    # Apply Scaling and Rotation
    M_transform = np.dot(M_rot, M_scale)  # Combine the two transformations
    transformed_image = cv2.warpAffine(image, M_transform, (w, h))

    # Update center
    new_center = np.dot(M_transform, [center[0], center[1], 1])

    # Random Perspective Transformation like a camera 3D view
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[np.random.uniform(0, 0.3) * w, np.random.uniform(0, 0.3) * h],
                       [np.random.uniform(0.7, 1) * w, np.random.uniform(0, 0.3) * h],
                       [np.random.uniform(0, 0.3) * w, np.random.uniform(0.7, 1) * h],
                       [np.random.uniform(0.7, 1) * w, np.random.uniform(0.7, 1) * h]])
    M_perspective = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply Perspective Transformation
    transformed_image = cv2.warpPerspective(transformed_image, M_perspective, (w, h))

    # Update center
    new_center = np.dot(M_perspective, [new_center[0], new_center[1], 1])
    new_center[0] /= new_center[2]
    new_center[1] /= new_center[2]

    return transformed_image, (new_center[0], new_center[1])

def load_and_transform_qr(imgpath):
    # print("Loading and transforming QR code: " + imgpath)
    # Example Usage
    image = cv2.imread(imgpath)  # Load an image

    # Add alpha channel if needed
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    image_w = image.shape[1]  # Get the width of the image
    image_h = image.shape[0]  # Get the height of the image
    center = (image_w / 2, image_h / 2)  # Get the center of the image

    # Add padding to avoid cropping after transformation
    padding = int(max(image_w, image_h) * 0.3)
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255, 0))
    center = (center[0] + padding, center[1] + padding)

    return random_transform(image, center)

base_output_folder = create_base_output_folder()

# Adjust paths for the test folders
test_qr_folder = adjusted_path("test_qr_codes", base_output_folder)
test_dataset_folder = adjusted_path("test_dataset", base_output_folder)

# Delete old folders
if os.path.exists(test_qr_folder):
    shutil.rmtree(test_qr_folder)
if os.path.exists(test_dataset_folder):
    shutil.rmtree(test_dataset_folder)

# Create the base output folder if it doesn't exist
if not os.path.exists("outputs"):
    os.mkdir("outputs")
if not os.path.exists(base_output_folder):
    os.mkdir(base_output_folder)

# Create output folders
os.mkdir(test_qr_folder)
os.mkdir(test_dataset_folder)

generate_qr_images(test_qr_folder, 100, 10//2)
generate_dataset(test_dataset_folder, "photos", test_qr_folder, 100, (1920//2, 1080//2), 3)

print("output created in: " + base_output_folder)
