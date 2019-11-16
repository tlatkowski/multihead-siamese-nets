import tensorflow as tf

from layers.basics import linear, dropout, feed_forward, residual


def stacked_multihead_attention(
        x,
        num_blocks,
        num_heads,
        use_residual,
        is_training,
        dropout_rate,
        reuse=False,
):
    num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope('stacked_multihead_attention', reuse=reuse):
        for i in range(num_blocks):
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=reuse):
                x, attentions = multihead_attention(x, x, x, use_residual, is_training,
                                                    dropout_rate, num_heads=num_heads, reuse=reuse)
                x = feed_forward(x, num_hiddens=num_hiddens, activation=tf.nn.relu, reuse=reuse)
    return x, attentions


def multihead_attention(
        queries,
        keys,
        values,
        use_residual,
        is_training,
        dropout_rate,
        num_units=None,
        num_heads=8,
        reuse=False,
):
    with tf.variable_scope('multihead-attention', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = linear(queries)
        K = linear(keys)
        V = linear(values)
        
        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
        
        Q_K_V, attentions = scaled_dot_product_attention(Q, K, V)
        Q_K_V = dropout(Q_K_V, is_training, rate=dropout_rate)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)
        
        output = feed_forward(Q_K_V_, num_units, reuse=reuse)
        
        if use_residual:
            output = residual(output, queries, reuse=reuse)
        # output = normalization(output)
    
    return output, attentions


def scaled_dot_product_attention(
        queries,
        keys,
        values,
        sequence_length=None,
        reuse=False,
):
    if sequence_length is None:
        sequence_length = tf.to_float(queries.get_shape().as_list()[-1])
    
    with tf.variable_scope('scaled_dot_product_attention', reuse=reuse):
        keys_T = tf.transpose(keys, [0, 2, 1])
        Q_K = tf.matmul(queries, keys_T) / tf.sqrt(sequence_length)
        attentions = tf.nn.softmax(Q_K)
        scaled_dprod_att = tf.matmul(attentions, values)
    return scaled_dprod_att, attentions
