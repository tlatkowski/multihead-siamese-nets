import tensorflow as tf

from layers.basics import dropout


def cnn_layer(
        inputs,
        num_filters,
        filter_size,
        is_training,
        dropout_rate,
        reuse=False,
):
    sentence_length = inputs.shape.as_list()[1]
    with tf.variable_scope('convolution', reuse=reuse):
        convoluted = tf.layers.conv1d(
            inputs=inputs,
            filters=num_filters,
            kernel_size=[filter_size],
            activation=tf.nn.relu,
        )
    with tf.variable_scope('pooling', reuse=reuse):
        pooling = tf.layers.max_pooling1d(
            inputs=convoluted,
            pool_size=[sentence_length - filter_size + 1],
            strides=1,
        )
        pooling_flat = tf.reshape(pooling, [-1, num_filters])
    with tf.variable_scope('dropout', reuse=reuse):
        dropped = dropout(pooling_flat, is_training, rate=dropout_rate)
    return dropped


def cnn_layers(
        inputs,
        num_filters,
        filter_sizes,
        is_training,
        dropout_rate,
        reuse=False,
):
    pooled_flats = []
    with tf.variable_scope('cnn_network', reuse=reuse):
        # for i, (n, size) in enumerate(zip(num_filters, filter_sizes)):
        for i, filters in enumerate(num_filters):
            for filter_size in filter_sizes:
                with tf.variable_scope('cnn_layer_{}_{}'.format(i, filter_size), reuse=reuse):
                    pooled_flat = cnn_layer(
                        inputs=inputs,
                        num_filters=filters,
                        filter_size=filter_size,
                        is_training=is_training,
                        dropout_rate=dropout_rate,
                        reuse=reuse,
                    )
                    pooled_flats.append(pooled_flat)
        cnn_output = tf.concat(pooled_flats, axis=1)
    return cnn_output
