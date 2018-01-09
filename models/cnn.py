import tensorflow as tf
from layers.similarity import manhattan_similarity
from layers.losses import mse


class CNNbasedSiameseNet:

    def __init__(self, sequence_len, vocabulary_size, embedding_size, hidden_size):
        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        with tf.variable_scope('embeddings'):
            word_embeddings = tf.get_variable('word_embeddings', [vocabulary_size, embedding_size])
            embedded_x1 = tf.gather(word_embeddings, self.x1)
            embedded_x2 = tf.gather(word_embeddings, self.x2)

        with tf.variable_scope('siamese-cnn'):
            self.out1 = cnn_layer(embedded_x1, reuse=False)
            self.out2 = cnn_layer(embedded_x2)

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


def cnn_layer(embedded_x, reuse=True):
    with tf.variable_scope('convolution', reuse=reuse):
        convoluted = tf.layers.conv2d(embedded_x, filters=100, kernel_size=[3, 64], activation=tf.nn.relu)
    return convoluted

