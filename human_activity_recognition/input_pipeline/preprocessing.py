import tensorflow as tf


def augment(features, labels):
    """Dataset augmentation"""
    # The data sample is already large enough

    return features, features


def preprocess(features, labels):
    """Dataset preprocessing"""

    # Format changed to int
    labels = tf.cast(labels, tf.int32)

    # get labels from 0 to 11
    labels = tf.subtract(labels, 1)

    return features, labels


