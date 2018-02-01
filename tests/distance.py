import numpy as np
import tensorflow as tf

from layers.similarity import manhattan_distance, cosine_distance


class DistanceTest(tf.test.TestCase):

    def testManhattanDistance(self):
        with self.test_session() as test_session:
            x1 = np.array([[1., 2., 3.]])
            x2 = np.array([[3., 2., 1.]])

            actual_output = test_session.run(manhattan_distance(x1, x2))
            correct_output = [4.]

            self.assertEqual(actual_output, correct_output)

    def testManhattanDistanceSame(self):
        with self.test_session() as test_session:
            x1 = np.array([[1., 1.]])
            x2 = np.array([[1., 1.]])

            actual_output = test_session.run(manhattan_distance(x1, x2))
            correct_output = [0.]

            self.assertEqual(actual_output, correct_output)

    def testManhattanDistance2D(self):
        with self.test_session() as test_session:
            x1 = np.array([[1., 1., 1.], [2., 2., 2.]])
            x2 = np.array([[2., 2., 2.], [1., 1., 1.]])

            actual_output = test_session.run(manhattan_distance(x1, x2))
            correct_output = [[3.], [3.]]

            self.assertAllEqual(actual_output, correct_output)

    def testCosineDistanceOpposite(self):
        with self.test_session() as test_session:
            x1 = np.array([[-1., -1.]])
            x2 = np.array([[1., 1.]])

            actual_output = test_session.run(cosine_distance(x1, x2))
            correct_output = [[-1.]]

            self.assertAllClose(actual_output, correct_output)

    def testCosineDistanceSame(self):
        with self.test_session() as test_session:
            x1 = np.array([[1., 1.]])
            x2 = np.array([[1., 1.]])

            actual_output = list(test_session.run(cosine_distance(x1, x2)))
            correct_output = [[1.]]

            self.assertAllClose(actual_output, correct_output)

    def testCosineDistanceOrthogonal(self):
        with self.test_session() as test_session:
            x1 = np.array([[-1., 1.]])
            x2 = np.array([[1., 1.]])

            actual_output = test_session.run(cosine_distance(x1, x2))
            correct_output = [0.]

            self.assertEqual(actual_output, correct_output)

    def testCosineDistance2D(self):
        with self.test_session() as test_session:
            x1 = np.array([[0., 1.], [2., 3.]])
            x2 = np.array([[0., 1.], [-2., -3.]])

            actual_output = test_session.run(cosine_distance(x1, x2))
            correct_output = [[1.], [-1.]]

            self.assertAllClose(actual_output, correct_output)