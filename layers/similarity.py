import tensorflow as tf


def manhattan_similarity(x1, x2):
    """
    Similarity function based on manhattan distance and exponential function.
    Args:
        x1: x1 input vector
        x2: x2 input vector

    Returns: Similarity measure in range between 0...1,
    where 1 means full similarity and 0 means no similarity at all.

    """
    with tf.name_scope('manhattan_similarity'):
        manhattan_sim = tf.exp(-manhattan_distance(x1, x2))
    return manhattan_sim


def manhattan_distance(x1, x2):
    """
    Also known as l1 norm.
    Equation: sum(|x1 - x2|)
    Example:
        x1 = [1,2,3]
        x2 = [3,2,1]
        MD = (|1 - 3|) + (|2 - 2|) + (|3 - 1|) = 4
    Args:
        x1: x1 input vector
        x2: x2 input vector

    Returns: Manhattan distance between x1 and x2. Value grater than or equal to 0.

    """
    return tf.reduce_sum(tf.abs(x1 - x2), axis=1, keepdims=True)


def euclidean_distance(x1, x2):
    return tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), axis=1, keepdims=True))


def cosine_distance(x1, x2):
    # TODO consider adding for case when input vector contains only 0 values, eps = 1e-08
    num = tf.reduce_sum(x1 * x2, axis=1)
    denom = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1)) * tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
    cos_sim = tf.expand_dims(tf.div(num, denom), -1)
    return cos_sim
