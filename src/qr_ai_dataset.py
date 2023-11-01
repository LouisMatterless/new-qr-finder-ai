import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_dataset(dataset_folder, max_data_count=0, downsample=False):
    print("Loading dataset from " + dataset_folder + "...")
    inputs = []  # photos with QRs
    truths = []  # array of same size as photo, where each QR's center pixel is 1, rest is 0

    # loop over all jpg files in the folder
    for file in tqdm(os.listdir(dataset_folder)):
        if not file.endswith(".jpg"):
            continue
        if file.endswith("_debug.jpg"):
            continue

        name = os.path.splitext(file)[0]
        img = Image.open(dataset_folder + "/" + name + ".jpg")
        if downsample:
            img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.ANTIALIAS)

        img = np.array(img).transpose((1, 0, 2)) / 255.0
        inputs.append(img)

        npy_array_file = dataset_folder + "/" + name + ".npy"
        if not os.path.exists(npy_array_file):
            print("WARNING: No center pixel truth NPY file found for " + name + ". Skipping...")
            continue

        truth = np.load(npy_array_file)

        if downsample:
            # take 1 if any of the 4 pixels is 1
            truth = truth[::2, ::2] + truth[1::2, ::2] + truth[::2, 1::2] + truth[1::2, 1::2]
            truth = np.clip(truth, 0, 1)
        truths.append(truth)

        if len(inputs) >= max_data_count > 0:
            break

    print("Loaded " + str(len(inputs)) + " inputs and " + str(len(truths)) + " truths")
    return np.array(inputs), np.array(truths)
