import numpy as np
import tensorflow as tf

from models import cnn


class TestCNNModel(tf.test.TestCase):
    
    def test_cnn_model_output_shape(self):
        with self.cached_session() as session:
            model_cfg = {
                'PARAMS': {
                    'num_filters': '50, 50, 50',
                    'filter_sizes': '2, 3, 4',
                    'dropout_rate': 0.0,
                },
            }
            
            cnn_model = cnn.CNNSiameseNet(
                max_sequence_len=20,
                vocabulary_size=100,
                loss_function='mse',
                embedding_size=32,
                learning_rate=0.001,
                model_cfg=model_cfg,
            )
            sentence1 = np.ones(shape=(4, 20), dtype=np.float32)
            sentence2 = np.ones(shape=(4, 20), dtype=np.float32)
            feed_dict = {
                cnn_model.x1: sentence1,
                cnn_model.x2: sentence2,
                cnn_model.is_training: False,
            }
            session.run(tf.initialize_all_variables())
            prediction = session.run([cnn_model.temp_sim], feed_dict=feed_dict)
            print(prediction)
