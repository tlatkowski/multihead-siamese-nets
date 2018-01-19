import tensorflow as tf
from layers.similarity import manhattan_similarity
from layers.losses import mse
from layers.basics import dropout, feed_forward, residual, linear, normalization


class ScaledDotProductAttentionSiameseNet:

    def __init__(self, sequence_len, vocabulary_size, main_cfg, model_cfg):
        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        self.num_blocks = int(model_cfg['PARAMS']['num_blocks'])
        self.num_heads = int(model_cfg['PARAMS']['num_heads'])
        self.embedding_size = int(main_cfg['PARAMS']['embedding_size'])
        self.learning_rate = float(main_cfg['TRAINING']['learning_rate'])

        with tf.variable_scope('embeddings'):
            word_embeddings = tf.get_variable('word_embeddings', [vocabulary_size, self.embedding_size])
            embedded_x1 = tf.gather(word_embeddings, self.x1)
            embedded_x2 = tf.gather(word_embeddings, self.x2)

        with tf.variable_scope('siamese-multihead-attention'):
            self.out1 = stacked_multihead_attention(embedded_x1,
                                                    num_blocks=self.num_blocks,
                                                    num_heads=self.num_heads,
                                                    reuse=False)

            self.out2 = stacked_multihead_attention(embedded_x2,
                                                    num_blocks=self.num_blocks,
                                                    num_heads=self.num_heads,
                                                    reuse=True)

            self.out1 = tf.reduce_sum(self.out1, axis=1)
            self.out2 = tf.reduce_sum(self.out2, axis=1)

            self.predictions = manhattan_similarity(self.out1, self.out2)

        with tf.variable_scope('loss'):
            self.loss = mse(self.labels, self.predictions)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.variable_scope('metrics'):
            self.temp_sim = tf.rint(self.predictions)
            self.correct_predictions = tf.equal(self.temp_sim, tf.to_float(self.labels))
            self.accuracy = tf.reduce_mean(tf.to_float(self.correct_predictions))

            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()


def stacked_multihead_attention(x, num_blocks, num_heads, reuse):
    num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope('stacked_multihead_attention', reuse=reuse):
        for i in range(num_blocks):
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=reuse):
                x = multihead_attention(x, x, x, num_heads=num_heads, reuse=reuse)
                x = feed_forward(x, num_hiddens=num_hiddens, activation=tf.nn.relu, reuse=reuse)
    return x


def multihead_attention(queries, keys, values, num_units=None, num_heads=8, reuse=True):
    with tf.variable_scope('multihead-attention', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = linear(queries)
        K = linear(keys)
        V = linear(values)

        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        Q_K_V = scaled_dot_product_attention(Q, K, V)
        Q_K_V = dropout(Q_K_V)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)

        output = feed_forward(Q_K_V_, num_units, reuse=reuse)
        output = residual(output, queries, reuse=reuse)
        # output = normalization(output)

    return output


def scaled_dot_product_attention(queries, keys, values, model_size=None, reuse=False):
    if model_size is None:
        model_size = tf.to_float(queries.get_shape().as_list()[-1])

    with tf.variable_scope('scaled_dot_product_attention', reuse=reuse):
        keys_T = tf.transpose(keys, [0, 2, 1])
        Q_K = tf.matmul(queries, keys_T) / tf.sqrt(model_size)
        scaled_dprod_att = tf.matmul(tf.nn.softmax(Q_K), values)
    return scaled_dprod_att





