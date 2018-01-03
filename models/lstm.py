import tensorflow as tf


class LSTMBasedSiameseNet:

    def __init__(self, sequence_len, vocabulary_size, embedding_size, hidden_size,
                 batch_size):
        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        with tf.variable_scope('embeddings'):
            word_embeddings = tf.get_variable('word_embeddings', [vocabulary_size, embedding_size])
            embedded_x1 = tf.gather(word_embeddings, self.x1)
            embedded_x2 = tf.gather(word_embeddings, self.x2)

        with tf.variable_scope('siamese-lstm'):
            fw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            bw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

            outputs_sen1, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                              bw_rnn_cell,
                                                              embedded_x1,
                                                              dtype=tf.float32)

            tf.get_variable_scope().reuse_variables()
            outputs_sen2, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                              bw_rnn_cell,
                                                              embedded_x2,
                                                              dtype=tf.float32)
            self.out1 = tf.concat([outputs_sen1[0], outputs_sen1[1]], axis=2)
            self.out2 = tf.concat([outputs_sen2[0], outputs_sen2[1]], axis=2)

            self.out1 = tf.reduce_mean(self.out1, axis=1)
            self.out2 = tf.reduce_mean(self.out2, axis=1)

            self.predictions = similarity(self.out1, self.out2)

        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)

        with tf.variable_scope('metrics'):
            # self.predictions = tf.cast(self.predictions, 'float')
            temp_sim = tf.subtract(tf.ones_like(self.predictions), tf.rint(self.predictions))

            correct_predictions = tf.equal(temp_sim, tf.cast(self.labels, 'float'))
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, 'float'))


def contrastive_loss(self, predicted, actual):
    pass


def mse_loss(self, predicted, actual):
    pass


def similarity(sentence1, sentence2):
    return tf.exp(-tf.norm(sentence1 - sentence2, ord=1, axis=1, keep_dims=True))
