import tensorflow as tf
from layers.similarity import cosine_similarity
from layers.losses import contrastive


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

            self.predictions = cosine_similarity(self.out1, self.out2)

        with tf.variable_scope('loss'):
            self.loss = contrastive(self.labels, self.predictions)
            self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        with tf.variable_scope('metrics'):
            self.temp_sim = tf.rint(self.predictions)
            self.correct_predictions = tf.equal(self.temp_sim, tf.to_float(self.labels))
            self.accuracy = tf.reduce_mean(tf.to_float(self.correct_predictions))

            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()


def cnn_layer(embedded_x, num_filters=200, filter_size=3, reuse=True):
    embedding_dim = embedded_x.get_shape().as_list()[-1]
    embedded_x_expanded = tf.expand_dims(embedded_x, -1)
    with tf.variable_scope('convolution', reuse=reuse):
        convoluted = tf.layers.conv2d(embedded_x_expanded,
                                      filters=num_filters,
                                      kernel_size=[filter_size, embedding_dim],
                                      activation=tf.nn.relu)
        pooling = tf.layers.max_pooling2d(convoluted,
                                          pool_size=[76, 1],
                                          strides=[1, 1])
        pooling_flat = tf.reshape(pooling, [-1, num_filters])
    return pooling_flat

