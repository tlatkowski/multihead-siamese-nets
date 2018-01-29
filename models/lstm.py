import tensorflow as tf

from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import SiameseNet


class LSTMBasedSiameseNet(SiameseNet):

    def __init__(self, sequence_len, vocabulary_size, main_cfg, model_cfg):
        SiameseNet.__init__(self, sequence_len, vocabulary_size, main_cfg, model_cfg, mse)

    def siamese_layer(self, sequence_len, model_cfg):
        hidden_size = int(model_cfg['PARAMS']['hidden_size'])

        fw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        bw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

        outputs_sen1, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                          bw_rnn_cell,
                                                          self.embedded_x1,
                                                          dtype=tf.float32)

        tf.get_variable_scope().reuse_variables()
        outputs_sen2, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                          bw_rnn_cell,
                                                          self.embedded_x2,
                                                          dtype=tf.float32)
        out1 = tf.concat([outputs_sen1[0], outputs_sen1[1]], axis=2)
        out2 = tf.concat([outputs_sen2[0], outputs_sen2[1]], axis=2)

        out1 = tf.reduce_mean(out1, axis=1)
        out2 = tf.reduce_mean(out2, axis=1)

        return manhattan_similarity(out1, out2)








