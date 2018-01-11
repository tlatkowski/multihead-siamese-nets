import tensorflow as tf


def contrastive(predictions, labels):
    contrastive_loss_minus = tf.to_float(labels) * _contrastive_plus(predictions)
    contrastive_loss_plus = (1.0 - tf.to_float(labels)) * _contrastive_minus(predictions)
    c_loss = tf.reduce_sum(contrastive_loss_plus + contrastive_loss_minus)
    return c_loss


def _contrastive_plus(model_energy):
    return 0.25 * tf.square(1.0 - tf.to_float(model_energy))


def _contrastive_minus(model_energy, margin=tf.constant(0.5)):
    mask = tf.to_float(tf.less(tf.to_float(model_energy), margin))
    return mask*tf.square(tf.to_float(model_energy))


def cross_entropy(predictions, labels):
    # TODO add epsilon to cross entropy stability
    labels = tf.cast(labels, "float")
    predictions = tf.cast(predictions, "float")
    return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(predictions), axis=1))


def mse(predictions, labels):
    return tf.losses.mean_squared_error(labels, predictions)
