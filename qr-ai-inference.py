import os

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from qr_ai_dataset import load_dataset
import tensorflow as tf
from qr_ai_helpers import weighted_mse, avg_pred

inputs, truths = load_dataset("test_dataset", max_data_count=25, downsample=False)

print("Inputs shape: " + str(inputs.shape))
print("Truths shape: " + str(truths.shape))

# load trained model
model = tf.keras.models.load_model("model.h5", custom_objects={"weighted_mse": weighted_mse, "avg_pred": avg_pred})


def fit_image_into_size(img, size):
    # Scale to fill without changing aspect ratio
    if img.size[0] > img.size[1]:
        # landscape
        scale = size[0] / img.size[0]
    else:
        # portrait
        scale = size[1] / img.size[1]
    if scale > 1:
        # don't scale up
        return img
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    img = img.crop((0, 0, size[0], size[1]))  # Crop to box
    return img


test_photo_folder = "test_photos"
test_photos = []
for file in os.listdir(test_photo_folder):
    if not file.endswith(".jpg"):
        continue
    img = Image.open(test_photo_folder + "/" + file)
    img = img.transpose(Image.ROTATE_270)
    img = fit_image_into_size(img, (540, 960))
    print("Test photo new size: " + str(img.size))
    img = np.array(img) / 255.0
    print("Test photo np shape: " + str(img.shape))
    test_photos.append(img)

test_photos = np.array(test_photos)
print("Test photos shape: " + str(test_photos.shape))


def predict_and_plot(inputs):
    predictions = model.predict(inputs, batch_size=1, verbose=1)

    # Plot predictions with detected centers overlaid in red, one subplot per image
    for i in range(0, len(inputs)):
        plt.subplot(1, len(inputs), i + 1)
        plt.imshow(inputs[i])
        plt.imshow(predictions[i], alpha=0.5, cmap="Reds")
        plt.axis("off")
    plt.show()


predict_and_plot(test_photos)
for interval in range(0, 5):
    predict_and_plot(inputs[interval * 5:(interval + 1) * 5])
