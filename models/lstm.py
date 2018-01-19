import tensorflow as tf
from layers.similarity import manhattan_similarity
from layers.losses import mse


class LSTMBasedSiameseNet:

    def __init__(self, sequence_len, vocabulary_size, main_cfg, model_cfg):
        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        self.hidden_size = int(model_cfg['PARAMS']['hidden_size'])
        self.embedding_size = int(main_cfg['PARAMS']['embedding_size'])
        self.learning_rate = float(main_cfg['TRAINING']['learning_rate'])

        with tf.variable_scope('embeddings'):
            word_embeddings = tf.get_variable('word_embeddings', [vocabulary_size, self.embedding_size])
            embedded_x1 = tf.gather(word_embeddings, self.x1)
            embedded_x2 = tf.gather(word_embeddings, self.x2)

        with tf.variable_scope('siamese-lstm'):
            fw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

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






