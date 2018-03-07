import tensorflow as tf

from layers.losses import mse
from layers.recurrent import rnn_layer
from layers.similarity import manhattan_similarity
from models.base_model import SiameseNet


class LSTMBasedSiameseNet(SiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        SiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)

    def siamese_layer(self, sequence_len, model_cfg):
        hidden_size = model_cfg['PARAMS'].getint('hidden_size')
        cell_type = model_cfg['PARAMS'].get('cell_type')

        outputs_sen1 = rnn_layer(self.embedded_x1, hidden_size, cell_type)
        outputs_sen2 = rnn_layer(self.embedded_x2, hidden_size, cell_type, reuse=True)

        out1 = tf.reduce_mean(outputs_sen1, axis=1)
        out2 = tf.reduce_mean(outputs_sen2, axis=1)

        return manhattan_similarity(out1, out2)
