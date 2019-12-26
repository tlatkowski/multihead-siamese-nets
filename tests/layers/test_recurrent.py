import tensorflow as tf

from layers.recurrent import rnn_layer


class TestRNN(tf.test.TestCase):
    
    def testRNNUnidirectionalNetwork(self):
        embedded_x = tf.random_normal([1, 2, 5])  # batch x sentence length x embedding size
        hidden_size = 10
        rnn_output = rnn_layer(
            embedded_x=embedded_x,
            hidden_size=hidden_size,
            bidirectional=False,
            cell_type='GRU',
        )
        
        actual_output = rnn_output.get_shape().as_list()[-1]
        self.assertEqual(actual_output, hidden_size)
    
    def testRNNBidirectionalNetwork(self):
        embedded_x = tf.random_normal([1, 2, 5])  # batch x sentence length x embedding size
        hidden_size = 10
        rnn_output = rnn_layer(
            embedded_x=embedded_x,
            hidden_size=hidden_size,
            bidirectional=True,
            cell_type='GRU',
        )
        
        actual_output = rnn_output.get_shape().as_list()[-1]
        self.assertEqual(actual_output, 2 * hidden_size)
