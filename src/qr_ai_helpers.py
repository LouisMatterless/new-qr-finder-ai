import os
import tensorflow as tf
import time


def avg_pred(y_true, y_pred):
    return tf.keras.backend.mean(y_pred)


def weighted_mse(y_true, y_pred):
    y_true = tf.keras.backend.cast_to_floatx(y_true)
    return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred) * (0.001 + 99 * y_true))



def create_base_output_folder():
    timestamp = int(time.time())
    return get_base_output_folder(timestamp)
    

def get_base_output_folder(timestamp):
    return "outputs/output-" + str(timestamp)

# Modify the paths to be inside the new output folder
def adjusted_path(original_path, base_output_folder):
    return os.path.join(base_output_folder, original_path)