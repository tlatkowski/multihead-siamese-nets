import tensorflow as tf

from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import SiameseNet


class LSTMBasedSiameseNet(SiameseNet):

    def __init__(self, sequence_len, vocabulary_size, main_cfg, model_cfg):
        SiameseNet.__init__(self, sequence_len, vocabulary_size, main_cfg)

        self.hidden_size = int(model_cfg['PARAMS']['hidden_size'])

        with tf.variable_scope('siamese-lstm'):
            fw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

            outputs_sen1, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                              bw_rnn_cell,
                                                              self.embedded_x1,
                                                              dtype=tf.float32)

            tf.get_variable_scope().reuse_variables()
            outputs_sen2, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                              bw_rnn_cell,
                                                              self.embedded_x2,
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






