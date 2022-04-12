"""functions to setup gpu on the server for machine learning"""
import tensorflow as tf


def setup_gpu():
    """find all gpus and set memory growth"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as error:
            # Memory growth must be set before GPUs have been initialized
            print(error)
