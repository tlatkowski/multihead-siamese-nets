import tensorflow as tf


def feed_forward(x, num_hiddens, activation=None, reuse=False):
    with tf.variable_scope('feed-forward2', reuse=reuse):
        ff = tf.layers.dense(x, num_hiddens, activation=activation, reuse=reuse)
    return ff


def linear(x, num_hiddens=None, reuse=False):
    if num_hiddens is None:
        num_hiddens = x.get_shape().as_list()[-1]
    # with tf.variable_scope('linear'):
    linear_layer = tf.layers.dense(x, num_hiddens)
    return linear_layer


def dropout(x, is_training, rate=0.2):
    return tf.layers.dropout(x, rate, training=tf.convert_to_tensor(is_training))


def residual(x_in, x_out, reuse=False):
    with tf.variable_scope('residual', reuse=reuse):
        res_con = x_in + x_out
    return res_con


def normalization(x, eps=1e-8):  # FIXME
    with tf.variable_scope('norm'):
        inputs_shape = x.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(x, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (x - mean) / tf.sqrt(variance + eps)
        outputs = gamma * normalized + beta

    return outputs


def optimize(loss, learning_rate=0.001):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)