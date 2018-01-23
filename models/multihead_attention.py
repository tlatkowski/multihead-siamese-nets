import tensorflow as tf

from layers.attention import stacked_multihead_attention
from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import SiameseNet


class MultiheadAttentionSiameseNet(SiameseNet):

    def __init__(self, sequence_len, vocabulary_size, main_cfg, model_cfg):
        SiameseNet.__init__(self, sequence_len, vocabulary_size, main_cfg)

        self.num_blocks = int(model_cfg['PARAMS']['num_blocks'])
        self.num_heads = int(model_cfg['PARAMS']['num_heads'])
        self.use_residual = bool(model_cfg['PARAMS']['use_residual'])

        with tf.variable_scope('siamese-multihead-attention'):
            self.out1 = stacked_multihead_attention(self.embedded_x1,
                                                    num_blocks=self.num_blocks,
                                                    num_heads=self.num_heads,
                                                    reuse=False,
                                                    use_residual=self.use_residual)

            self.out2 = stacked_multihead_attention(self.embedded_x2,
                                                    num_blocks=self.num_blocks,
                                                    num_heads=self.num_heads,
                                                    reuse=True,
                                                    use_residual=self.use_residual)

            self.out1 = tf.reduce_sum(self.out1, axis=1)
            self.out2 = tf.reduce_sum(self.out2, axis=1)

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




