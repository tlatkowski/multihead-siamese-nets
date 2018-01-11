import tensorflow as tf


def manhattan_similarity(x1, x2):
    return tf.exp(-tf.norm(x1 - x2, ord=1, axis=1, keep_dims=True))


def cosine_similarity(x1, x2):
    num = tf.reduce_sum(x1 * x2, axis=1)
    denom = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1)) * tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
    cos_sim = tf.expand_dims(tf.div(num, denom), -1)
    return cos_sim