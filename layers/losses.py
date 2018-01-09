import tensorflow as tf


def contrastive(predictions, labels):
    raise NotImplementedError


def _contrastive_plus(model_energy):
    return 0.25 * tf.square(1 - model_energy)


def _contrastive_minus(model_energy):
    pass


def cross_entropy(predictions, labels):
    # TODO add epsilon to cross entropy stability
    labels = tf.cast(labels, "float")
    predictions = tf.cast(predictions, "float")
    return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(predictions), axis=1))


def mse(predictions, labels):
    return tf.losses.mean_squared_error(labels, predictions)
