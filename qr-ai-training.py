import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, AUC
import matplotlib.pyplot as plt
from qr_ai_dataset import load_dataset
from qr_ai_helpers import weighted_mse, avg_pred
def build_model(input_shape=(1920, 1080, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    # Encoder
    #x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Bottleneck
    x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x)

    #x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)

    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return models.Model(inputs, outputs)


def train_model(model, inputs, truths, epochs=10):
    print("Training model...")
    history = model.fit(inputs, truths, epochs=epochs, batch_size=8, verbose=1, shuffle=True)
    print("Training complete. Final loss: " + str(history.history["loss"][-1]))
    return history


inputs, truths = load_dataset("dataset", max_data_count=0, downsample=False)

print("Inputs shape: " + str(inputs.shape))
print("Truths shape: " + str(truths.shape))

model = build_model(input_shape=inputs[0].shape)

# Make sure to check if it's on GPU
print("GPU Available: " + str(tf.test.is_gpu_available()))
print("CPU DEVICES:")
print(tf.config.list_physical_devices('CPU'))
print("GPU DEVICES:")
print(tf.config.list_physical_devices('GPU'))

# Compile the model with Mean Squared Error since it's a regression problem
# For classifiers precision and recall are good metrics.
# Simply speaking, high precision means low false positive rate, and high recall means low false negative rate

metrics = [Precision(), Recall(), avg_pred]
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00003)
#optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.00003)
#loss = tf.keras.losses.MeanSquaredError()

# Test the custom loss function on first sample and input just zeroes as prediction
print("Testing custom loss function...")
print("Truths[0] number of ones: " + str(np.sum(truths[0])))

# save as debug image
pixels = np.array(truths[0] * 255)
pixels = Image.fromarray(pixels.astype(np.uint8))
pixels.save("debug_truth.jpg")

pixels = np.array(inputs[0] * 255)
pixels = Image.fromarray(pixels.astype(np.uint8))
pixels.save("debug_input.jpg")

print("Truths[0] mean: " + str(np.mean(truths[0])))
print("Truths[0] std: " + str(np.std(truths[0])))
print("Truths[0] min: " + str(np.min(truths[0])))
print("Truths[0] max: " + str(np.max(truths[0])))
print("Test loss only zero: " + str(weighted_mse(truths[0], np.zeros(truths[0].shape))))

model.compile(optimizer=optimizer, loss=weighted_mse, metrics=metrics)

print("Model Summary:")
model.summary()

# Some debugging to figure out why both precision and recall go to 0 after some epochs
print("Inputs shape: " + str(inputs.shape))
print("Inputs mean: " + str(np.mean(inputs)))
print("Inputs std: " + str(np.std(inputs)))
print("Inputs max: " + str(np.max(inputs)))
print("Inputs min: " + str(np.min(inputs)))

print("Truths shape: " + str(truths.shape))
print("Truths mean: " + str(np.mean(truths)))
print("Truths std: " + str(np.std(truths)))
print("Truths max: " + str(np.max(truths)))
print("Truths min: " + str(np.min(truths)))

history = train_model(model, inputs, truths, epochs=500)

plt.plot(history.history['loss'], label='loss')
plt.savefig("loss.png")

# Save model
model.save("model.h5")
