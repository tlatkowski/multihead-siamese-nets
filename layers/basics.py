import tensorflow as tf


def feed_forward(x, num_hiddens, activation, reuse=False):
    return tf.layers.dense(x, num_hiddens, activation=activation, reuse=reuse)


def dropout(x, rate=0.2):
    return tf.layers.dropout(x, rate)


def residual(x_in, x_out):
    return x_in + x_out


def normalization(x):
    pass
