import numpy as np
import tensorflow as tf

from models.rnn import manhattan_similarity


class TestSimilarity(tf.test.TestCase):

    def testManhattanSimilaritySame(self):
        with self.test_session() as test_session:
            x1 = np.array([[1., 1.]])
            x2 = np.array([[1., 1.]])
            siamese_lstm_model = manhattan_similarity(x1, x2)

            actual_output = test_session.run(siamese_lstm_model)
            correct_output = [1.]
            self.assertEqual(actual_output, correct_output)

    def testSimilarity2D(self):
        with self.test_session() as test_session:
            x1 = np.array([[1., 1.], [1., 1.]])
            x2 = np.array([[1., 1.], [1., 1.]])
            siamese_lstm_model = manhattan_similarity(x1, x2)

            actual_output = test_session.run(siamese_lstm_model)
            correct_output = [[1.], [1.]]
            self.assertAllEqual(actual_output, correct_output)
