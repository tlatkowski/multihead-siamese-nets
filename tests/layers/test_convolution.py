import tensorflow as tf

from layers import convolution


class TestConvolutionLayer(tf.test.TestCase):
    
    def test_cnn_layer_output_shape(self):
        inputs = tf.ones(shape=(4, 20, 32))
        conv_layer_output = convolution.cnn_layer(
            embedded_x=inputs,
            max_seq_len=20,
            num_filters=64,
            filter_size=2,
            is_training=False,
            dropout_rate=0.0,
        )
        expected_shape = (4, 64)
        actual_shape = conv_layer_output.shape
        self.assertAllEqual(expected_shape, actual_shape)
