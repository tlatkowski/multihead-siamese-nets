import tensorflow as tf
from layers.similarity import manhattan_similarity
from layers.losses import mse


class ScaledDotProductAttentionSiameseNet:

    def __init__(self, sequence_len, vocabulary_size, embedding_size, hidden_size):
        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        with tf.variable_scope('embeddings'):
            word_embeddings = tf.get_variable('word_embeddings', [vocabulary_size, embedding_size])
            embedded_x1 = tf.gather(word_embeddings, self.x1)
            embedded_x2 = tf.gather(word_embeddings, self.x2)

        with tf.variable_scope('siamese-multihead-attention'):
            self.out1 = multihead_attention(embedded_x1, embedded_x1, embedded_x1, reuse=False)
            self.out2 = multihead_attention(embedded_x2, embedded_x2, embedded_x2)

            self.out1 = tf.reduce_sum(self.out1, axis=1)
            self.out2 = tf.reduce_sum(self.out2, axis=1)

            self.predictions = manhattan_similarity(self.out1, self.out2)

        with tf.variable_scope('loss'):
            self.loss = mse(self.labels, self.predictions)
            self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        with tf.variable_scope('metrics'):
            self.temp_sim = tf.rint(self.predictions)
            self.correct_predictions = tf.equal(self.temp_sim, tf.to_float(self.labels))
            self.accuracy = tf.reduce_mean(tf.to_float(self.correct_predictions))

            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()


def scaled_dot_product_attention(queries, keys, values, dk_size):
    q_k = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(dk_size)
    v_q_k = tf.matmul(q_k, values)
    return tf.nn.softmax(v_q_k)


def multihead_attention(queries, keys, values, num_units=None, num_heads=8, reuse=True):
    with tf.variable_scope('multihead-attention', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = tf.layers.dense(queries, num_units)
        K = tf.layers.dense(keys, num_units)
        V = tf.layers.dense(values, num_units)

        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        Q_K = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        model_size = tf.to_float(Q.get_shape().as_list()[-1])

        Q_K = tf.nn.softmax(Q_K / tf.sqrt(model_size))

        Q_K_V = tf.matmul(Q_K, V)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)

        logits = tf.layers.dense(Q_K_V_, num_units)
    return logits