import tensorflow as tf
import abc


class SiameseNet:
    __metaclass__ = abc.ABCMeta

    def __init__(self, sequence_len, vocabulary_size, main_cfg, model_cfg, loss_function):
        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        self.debug = None

        self.embedding_size = int(main_cfg['PARAMS']['embedding_size'])
        self.learning_rate = float(main_cfg['TRAINING']['learning_rate'])

        with tf.variable_scope('embeddings'):
            word_embeddings = tf.get_variable('word_embeddings', [vocabulary_size, self.embedding_size])
            self.embedded_x1 = tf.gather(word_embeddings, self.x1)
            self.embedded_x2 = tf.gather(word_embeddings, self.x2)

        with tf.variable_scope('siamese'):
            self.predictions = self.siamese_layer(sequence_len, model_cfg)

        with tf.variable_scope('loss'):
            self.loss = loss_function(self.labels, self.predictions)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.variable_scope('metrics'):
            self.temp_sim = tf.rint(self.predictions)
            self.correct_predictions = tf.equal(self.temp_sim, tf.to_float(self.labels))
            self.accuracy = tf.reduce_mean(tf.to_float(self.correct_predictions))

            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()

    @abc.abstractmethod
    def siamese_layer(self, sequence_len, model_cfg):
        """Implementation of specific siamese layer"""
