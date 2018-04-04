import tensorflow as tf

from layers.attention import stacked_multihead_attention
from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import BaseSiameseNet


class MultiheadAttentionSiameseNet(BaseSiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        BaseSiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)

    def siamese_layer(self, sequence_len, model_cfg):
        num_blocks = model_cfg['PARAMS'].getint('num_blocks')
        num_heads = model_cfg['PARAMS'].getint('num_heads')
        use_residual = model_cfg['PARAMS'].getboolean('use_residual')

        out1, self.debug = stacked_multihead_attention(self.embedded_x1,
                                                       num_blocks=num_blocks,
                                                       num_heads=num_heads,
                                                       use_residual=use_residual,
                                                       is_training=self.is_training)

        out2, _ = stacked_multihead_attention(self.embedded_x2,
                                              num_blocks=num_blocks,
                                              num_heads=num_heads,
                                              use_residual=use_residual,
                                              is_training=self.is_training,
                                              reuse=True)

        out1 = tf.reduce_sum(out1, axis=1)
        out2 = tf.reduce_sum(out2, axis=1)

        return manhattan_similarity(out1, out2)
