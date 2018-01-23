import tensorflow as tf


class SiameseNet:

    def __init__(self, sequence_len, vocabulary_size, main_cfg):
        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_len])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        self.embedding_size = int(main_cfg['PARAMS']['embedding_size'])
        self.learning_rate = float(main_cfg['TRAINING']['learning_rate'])

        with tf.variable_scope('embeddings'):
            word_embeddings = tf.get_variable('word_embeddings', [vocabulary_size, self.embedding_size])
            self.embedded_x1 = tf.gather(word_embeddings, self.x1)
            self.embedded_x2 = tf.gather(word_embeddings, self.x2)
