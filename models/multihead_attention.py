import tensorflow as tf

from layers.attention import stacked_multihead_attention
from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import SiameseNet


class MultiheadAttentionSiameseNet(SiameseNet):

    def __init__(self, sequence_len, vocabulary_size, main_cfg, model_cfg):
        SiameseNet.__init__(self, sequence_len, vocabulary_size, main_cfg, model_cfg, mse)

    def siamese_layer(self, sequence_len, model_cfg):
        num_blocks = int(model_cfg['PARAMS']['num_blocks'])
        num_heads = int(model_cfg['PARAMS']['num_heads'])
        use_residual = bool(model_cfg['PARAMS']['use_residual'])

        out1, self.debug = stacked_multihead_attention(self.embedded_x1,
                                                            num_blocks=num_blocks,
                                                            num_heads=num_heads,
                                                            use_residual=use_residual)

        out2, _ = stacked_multihead_attention(self.embedded_x2,
                                           num_blocks=num_blocks,
                                           num_heads=num_heads,
                                           use_residual=use_residual,
                                           reuse=True)

        out1 = tf.reduce_sum(out1, axis=1)
        out2 = tf.reduce_sum(out2, axis=1)

        return manhattan_similarity(out1, out2)





