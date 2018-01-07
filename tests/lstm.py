import tensorflow as tf
import numpy as np
from models.lstm import LSTMBasedSiameseNet, manhattan_similarity


class LSTMBasedSiameseNetTest(tf.test.TestCase):

    def testNetwork(self):
        with self.test_session() as test_session:
            x1 = np.array([2., 3., 4.])
            x2 = np.array([3., 1., 5.])
            siamese_lstm_model = LSTMBasedSiameseNet(3, vocabulary_size=150, embedding_size=32,
                                                     hidden_size=64, batch_size=10)
            siamese_lstm_model.x1 = x1
            siamese_lstm_model.x2 = x2
            siamese_lstm_model.labels = [1.]

            actual_pearson_coefficient = test_session.run(siamese_lstm_model.predictions)
            correct_pearson_coefficient = tf.constant([.5])
            self.assertEqual(actual_pearson_coefficient, correct_pearson_coefficient.eval())

    def testSimilarity(self):
        with self.test_session() as test_session:
            x1 = np.array([2., 3., 4.])
            x2 = np.array([3., 1., 5.])
            siamese_lstm_model = manhattan_similarity(x1, x2)

            actual_pearson_coefficient = test_session.run(siamese_lstm_model)
            correct_pearson_coefficient = tf.constant([.5])
            self.assertEqual(actual_pearson_coefficient, correct_pearson_coefficient.eval())

    def testSimilarity2D(self):
        with self.test_session() as test_session:
            x1 = np.array([[1., 1., 1.], [2., 2., 2.]])
            x2 = np.array([[2., 2., 2.], [1., 1., 1.]])
            siamese_lstm_model = manhattan_similarity(x1, x2)

            actual_pearson_coefficient = test_session.run(siamese_lstm_model)
            correct_pearson_coefficient = tf.constant([.5])
            self.assertEqual(actual_pearson_coefficient, correct_pearson_coefficient.eval())