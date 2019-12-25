import tensorflow as tf

from models import cnn


class TestCNNModel(tf.test.TestCase):
    
    def test_cnn_model_output_shape(self):
        raise NotImplementedError