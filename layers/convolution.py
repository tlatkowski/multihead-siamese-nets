import tensorflow as tf
from layers.basics import dropout


def cnn_layer(
        embedded_x,
        max_seq_len,
        num_filters,
        filter_size,
        is_training,
        dropout_rate,
        reuse=False,
):
    embedding_dim = embedded_x.get_shape().as_list()[-1]
    embedded_x_expanded = tf.expand_dims(embedded_x, -1)
    with tf.variable_scope('convolution', reuse=reuse):
        convoluted = tf.layers.conv2d(
            embedded_x_expanded,
            filters=num_filters,
            kernel_size=[filter_size, embedding_dim],
            activation=tf.nn.relu,
        )
    with tf.variable_scope('pooling', reuse=reuse):
        pooling = tf.layers.max_pooling2d(
            convoluted,
            pool_size=[max_seq_len - filter_size + 1, 1],
            strides=[1, 1],
        )
        pooling_flat = tf.reshape(pooling, [-1, num_filters])
    with tf.variable_scope('dropout', reuse=reuse):
        dropped = dropout(pooling_flat, is_training, rate=dropout_rate)
    return dropped


def cnn_layers(
        embedded_x,
        max_seq_len,
        num_filters,
        filter_sizes,
        is_training,
        dropout_rate,
        reuse=False,
):
    pooled_flats = []
    with tf.variable_scope('cnn_network', reuse=reuse):
        for i, (n, size) in enumerate(zip(num_filters, filter_sizes)):
            with tf.variable_scope('cnn_layer_{}'.format(i), reuse=reuse):
                pooled_flat = cnn_layer(
                    embedded_x,
                    max_seq_len,
                    num_filters=n,
                    filter_size=size,
                    is_training=is_training,
                    dropout_rate=dropout_rate,
                    reuse=reuse,
                )
                pooled_flats.append(pooled_flat)
        cnn_output = tf.concat(pooled_flats, axis=1)
    return cnn_output
