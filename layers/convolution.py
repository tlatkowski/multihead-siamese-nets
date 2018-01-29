import tensorflow as tf


def cnn_layer(embedded_x, max_seq_len, num_filters=200, filter_size=3, reuse=True):
    embedding_dim = embedded_x.get_shape().as_list()[-1]
    embedded_x_expanded = tf.expand_dims(embedded_x, -1)
    with tf.variable_scope('convolution', reuse=reuse):
        convoluted = tf.layers.conv2d(embedded_x_expanded,
                                      filters=num_filters,
                                      kernel_size=[filter_size, embedding_dim],
                                      activation=tf.nn.relu)
        pooling = tf.layers.max_pooling2d(convoluted,
                                          pool_size=[max_seq_len - filter_size + 1, 1],
                                          strides=[1, 1])
        pooling_flat = tf.reshape(pooling, [-1, num_filters])
    return pooling_flat


def cnn_layers(embedded_x, max_seq_len, num_filters=[50, 50, 50], filter_sizes=[2, 3, 4], reuse=True):
    pooled_flats = []
    for i, (n, size) in enumerate(zip(num_filters, filter_sizes)):
        with tf.variable_scope('cnn_layer_{}'.format(i), reuse=reuse):
            pooled_flat = cnn_layer(embedded_x, max_seq_len, num_filters=n, filter_size=size, reuse=reuse)
            pooled_flats.append(pooled_flat)
    return tf.concat(pooled_flats, axis=1)