import unittest

import numpy as np

from deeptracking.utils.transform import Transform


class TestTransformMethods(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_it_should_initialise_as_identity(self):
        transform = Transform()
        truth = np.eye(4)
        np.testing.assert_almost_equal(transform.matrix, truth)

    def test_it_should_init_from_parameters(self):
        transform = Transform.from_parameters(10, 1, 2.2, 1, 0.707, 3.1)
        truth = np.array([[-0.7597, -0.0316, 0.6496, 10],
                          [-0.5236, -0.5626, -0.6398, 1],
                          [0.3856, -0.8262, 0.4108, 2.2],
                          [0, 0, 0, 1]])
        np.testing.assert_almost_equal(transform.matrix, truth, 4)

    def test_it_should_convert_to_parameters(self):
        matrix = np.array([[-0.7597, -0.0316, 0.6496, 10],
                          [-0.5236, -0.5626, -0.6398, 1],
                          [0.3856, -0.8262, 0.4108, 2.2],
                          [0, 0, 0, 1]])
        transform = Transform.from_matrix(matrix)
        np.testing.assert_almost_equal(transform.to_parameters(), np.array([10, 1, 2.2, 1, 0.707, 3.1]), 4)

    def test_it_should_set_translation(self):
        transform = Transform()
        transform.set_translation(10, 1, 2.2)
        truth = np.array([[1, 0, 0, 10],
                          [0, 1, 0, 1],
                          [0, 0, 1, 2.2],
                          [0, 0, 0, 1]])
        np.testing.assert_almost_equal(transform.matrix, truth)

    def test_it_should_set_rotation_from_euler(self):
        transform = Transform()
        transform.set_rotation(1.0, 0.707, 3.1)
        truth = np.array([[-0.7597, -0.0316, 0.6496, 0],
                          [-0.5236, -0.5626, -0.6398, 0],
                          [0.3856, -0.8262, 0.4108, 0],
                          [0, 0, 0, 1]])
        np.testing.assert_almost_equal(transform.matrix, truth, 4)

    def test_it_should_translate_from_other_transform(self):
        transform = Transform.from_parameters(10, 1, 2.2, 1, 0.707, 3.1)
        translation = Transform.from_parameters(10, -1, 2.2, 0, 0, 0)
        transform.translate(transform=translation)
        truth = np.array([[-0.7597, -0.0316, 0.6496, 20],
                          [-0.5236, -0.5626, -0.6398, 0],
                          [0.3856, -0.8262, 0.4108, 4.4],
                          [0, 0, 0, 1]])
        np.testing.assert_almost_equal(transform.matrix, truth, 4)

    def test_it_should_translate_from_parameters(self):
        transform = Transform.from_parameters(10, 1, 2.2, 1, 0.707, 3.1)
        transform.translate(10, -1, 2.2)
        truth = np.array([[-0.7597, -0.0316, 0.6496, 20],
                          [-0.5236, -0.5626, -0.6398, 0],
                          [0.3856, -0.8262, 0.4108, 4.4],
                          [0, 0, 0, 1]])
        np.testing.assert_almost_equal(transform.matrix, truth, 4)

    def test_it_should_rotate_from_other_transform(self):
        transform = Transform.from_parameters(0, 0, 0, 1, 0.707, 3.1)
        rotation = Transform.from_parameters(0, 0, 0, -1, -0.707, -3.1)
        transform.rotate(transform=rotation)
        truth = np.array([[0.39, 0.5479, 0.74, 0.],
                          [0.9196, -0.2729, -0.2826, 0.],
                          [0.0471, 0.7908, -0.6103, 0.],
                          [0., 0., 0., 1.]])
        np.testing.assert_almost_equal(transform.matrix, truth, 4)

    def test_it_should_rotate_from_parameters(self):
        transform = Transform.from_parameters(0, 0, 0, 1, 0.707, 3.1)
        transform.rotate(-1, -0.707, -3.1)
        truth = np.array([[0.39, 0.5479, 0.74, 0.],
                          [0.9196, -0.2729, -0.2826, 0.],
                          [0.0471, 0.7908, -0.6103, 0.],
                          [0., 0., 0., 1.]])
        np.testing.assert_almost_equal(transform.matrix, truth, 4)

    def test_two_identity_should_be_equal(self):
        transform1 = Transform()
        transform2 = Transform()
        self.assertEqual(transform1, transform2)
        transform3 = Transform.from_parameters(1, 0, 0, 0, 0, 0)
        self.assertNotEqual(transform1, transform3)

    def test_two_similar_transform_should_be_equal(self):
        transform1 = Transform.from_parameters(1.1, 1.2, 1.3, 1.1, 1.2, 1.3)
        transform2 = Transform.from_parameters(1.10001, 1.2, 1.30001, 1.1, 1.200001, 1.3)
        self.assertEqual(transform1, transform2)
