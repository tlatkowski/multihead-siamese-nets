import tensorflow as tf

from layers import attention


class TestAttentionLayer(tf.test.TestCase):
    
    def test_scaled_dot_product_attention_output_shape(self):
        batch_size = 4
        split_embedding_size = 8
        sequence_length = 20
        queries = tf.random_normal(shape=[batch_size, sequence_length, split_embedding_size])
        keys = tf.random_normal(shape=[batch_size, sequence_length, split_embedding_size])
        values = tf.random_normal(shape=[batch_size, sequence_length, split_embedding_size])
    
        outputs, _ = attention.scaled_dot_product_attention(queries, keys, values)
        actual_attentions_shape = outputs.shape
        expected_attentions_shape = [batch_size, sequence_length, split_embedding_size]
    
        self.assertEqual(actual_attentions_shape, expected_attentions_shape)
    
    def test_scaled_dot_product_attention_attentions_shape(self):
        batch_size = 4
        split_embedding_size = 8
        sequence_length = 20
        queries = tf.random_normal(shape=[batch_size, sequence_length, split_embedding_size])
        keys = tf.random_normal(shape=[batch_size, sequence_length, split_embedding_size])
        values = tf.random_normal(shape=[batch_size, sequence_length, split_embedding_size])
        
        _, attentions = attention.scaled_dot_product_attention(queries, keys, values)
        actual_attentions_shape = attentions.shape
        expected_attentions_shape = [batch_size, sequence_length, sequence_length]
        
        self.assertEqual(actual_attentions_shape, expected_attentions_shape)
