import os
import sys
import tensorflow as tf
import time
import shutil


def avg_pred(y_true, y_pred):
    return tf.keras.backend.mean(y_pred)


def weighted_mse(y_true, y_pred):
    y_true = tf.keras.backend.cast_to_floatx(y_true)
    return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred) * (0.001 + 99 * y_true))



def get_base_output_folder(timestamp):
    return "outputs/output-" + str(timestamp)

# Modify the paths to be inside the new output folder
# def adjusted_path(base_output_folder, original_path):
#     return os.path.join(base_output_folder, original_path)

def prepare_base_output_folder():
    if len(sys.argv) > 1:
        timestamp =   sys.argv[1]
    else:
        timestamp = input("Please enter the hash of the output folder, system will generate one if it is empty: ")
    
    if(timestamp is None):
        timestamp = int(time.time())
    
    base_output_folder = get_base_output_folder(timestamp)

    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    if not os.path.exists(base_output_folder):
        os.mkdir(base_output_folder)

    return timestamp

def prepare_folder(timestamp, folderName, reset = False):
    folderPath = os.path.join(get_base_output_folder(timestamp), folderName)

    if reset and os.path.exists(folderPath):
        shutil.rmtree(folderPath)

    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    
    return folderPath