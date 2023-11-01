import tensorflow as tf


def avg_pred(y_true, y_pred):
    return tf.keras.backend.mean(y_pred)


def weighted_mse(y_true, y_pred):
    y_true = tf.keras.backend.cast_to_floatx(y_true)
    return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred) * (0.001 + 99 * y_true))

def test_func():
    print("func ran")