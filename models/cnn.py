import tensorflow as tf

from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import SiameseNet


class CNNbasedSiameseNet(SiameseNet):

    def __init__(self, sequence_len, vocabulary_size, main_cfg, model_cfg):
        SiameseNet.__init__(self, sequence_len, vocabulary_size, main_cfg)

        self.num_filters = _parse_list(model_cfg['PARAMS']['num_filters'])
        self.filter_sizes = _parse_list(model_cfg['PARAMS']['filter_sizes'])

        with tf.variable_scope('siamese-cnn'):
            self.out1 = cnn_layers(self.embedded_x1,
                                   sequence_len,
                                   num_filters=self.num_filters,
                                   filter_sizes=self.filter_sizes,
                                   reuse=False)

            self.out2 = cnn_layers(self.embedded_x2,
                                   sequence_len,
                                   num_filters=self.num_filters,
                                   filter_sizes=self.filter_sizes)

            # self.out1 = feed_forward(dropout(self.out1), num_hiddens=128, activation=tf.nn.relu, reuse=False)
            # self.out2 = feed_forward(dropout(self.out2), num_hiddens=128, activation=tf.nn.relu)

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


def cnn_layer(embedded_x, max_seq_len, num_filters=200, filter_size=3, reuse=True):
    embedding_dim = embedded_x.get_shape().as_list()[-1]
    embedded_x_expanded = tf.expand_dims(embedded_x, -1)
    with tf.variable_scope('convolution', reuse=reuse):
        convoluted = tf.layers.conv2d(embedded_x_expanded,
                                      filters=num_filters,
                                      kernel_size=[filter_size, embedding_dim],
                                      activation=tf.nn.relu)
        pooling = tf.layers.max_pooling2d(convoluted,
                                          pool_size=[max_seq_len - filter_size + 1, 1],
                                          strides=[1, 1])
        pooling_flat = tf.reshape(pooling, [-1, num_filters])
    return pooling_flat


def cnn_layers(embedded_x, max_seq_len, num_filters=[50, 50, 50], filter_sizes=[2, 3, 4], reuse=True):
    pooled_flats = []
    for i, (n, size) in enumerate(zip(num_filters, filter_sizes)):
        with tf.variable_scope('cnn_layer_{}'.format(i), reuse=reuse):
            pooled_flat = cnn_layer(embedded_x, max_seq_len, num_filters=n, filter_size=size, reuse=reuse)
            pooled_flats.append(pooled_flat)
    return tf.concat(pooled_flats, axis=1)


def _parse_list(x):
    return [int(i.strip()) for i in x.split(',')]
