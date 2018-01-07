import tensorflow as tf


def manhattan_similarity(x1, x2):
    return tf.exp(-tf.norm(x1 - x2, ord=1, axis=1, keep_dims=True))


def cosine_similarity(x1, x2):
    counter = tf.reduce_sum(x1 * x2)
    denominator = tf.norm(x1, ord=2) * tf.norm(x2, ord=2)
    return counter/denominator
