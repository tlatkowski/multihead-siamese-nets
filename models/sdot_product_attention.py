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

            self.out1 = multihead_attention(embedded_x1, embedded_x1, embedded_x1, 512)
            tf.get_variable_scope().reuse_variables()
            self.out2 = multihead_attention(embedded_x2, embedded_x2, embedded_x2, 512)

            self.out1 = tf.reduce_mean(self.out1, axis=1)
            self.out2 = tf.reduce_mean(self.out2, axis=1)

            self.predictions = manhattan_similarity(self.out1, self.out2)

        with tf.variable_scope('loss'):
            self.loss = mse(self.labels, self.predictions)
            self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        with tf.variable_scope('metrics'):
            self.temp_sim = tf.rint(self.predictions)
            self.correct_predictions = tf.equal(self.temp_sim, tf.cast(self.labels, 'float'))
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, 'float'))


def scaled_dot_product_attention(queries, keys, values, dk_size):
    q_k = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(dk_size)
    v_q_k = tf.matmul(q_k, values)
    return tf.nn.softmax(v_q_k)


def multihead_attention(queries, keys, values, dk_size, num_heads=8):
    Q = tf.concat(tf.split(queries, num_heads, axis=2), axis=0)
    K = tf.concat(tf.split(keys, num_heads, axis=2), axis=0)
    V = tf.concat(tf.split(values, num_heads, axis=2), axis=0)

    Q_K = tf.nn.softmax(tf.matmul(Q, tf.transpose(K, [0, 2, 1])))
    Q_K_V = tf.matmul(Q_K, V)
    Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)

    logits = tf.layers.dense(Q_K_V_, 512)
    return logits