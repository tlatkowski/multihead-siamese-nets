import tensorflow as tf


class ScaledDotProductAttentionSiameseNet:

    def __init__(self, sequence_len, vocabulary_size, embedding_size, hidden_size,
                 batch_size):
        pass


def scaled_dot_product_attention(queries, keys, values, dk_size):
    q_k = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(dk_size)
    v_q_k = tf.matmul(q_k, values)
    return tf.nn.softmax(v_q_k)