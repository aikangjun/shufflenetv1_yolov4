import tensorflow as tf


def swish(x):

    return x * tf.sigmoid(x)
