import unittest

import numpy as np

from deeptracking.data.dataset import Dataset
from deeptracking.utils.transform import Transform


class TestDatasetMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # run these once as they take time
        cls.dummy_rgb = np.zeros((150, 150, 3), dtype=np.float32)
        cls.dummy_depth = np.zeros((150, 150), dtype=np.float32)
        cls.dummy_pose = Transform()

    def setUp(self):
        self.populated_dataset = Dataset("data")
        for i in range(10):
            self.populated_dataset.add_pose(self.dummy_rgb, self.dummy_depth, self.dummy_pose)


    def tearDown(self):
        pass

    def test_it_should_have_size_0_at_init(self):
        dataset = Dataset("data")
        self.assertEqual(dataset.size(), 0)

    def test_it_should_add_sample(self):
        dataset = Dataset("data")
        dataset.add_pose(self.dummy_rgb, self.dummy_depth, self.dummy_pose)
        self.assertEqual(dataset.size(), 1)

    def test_it_should_return_index_after_adding_pose(self):
        dataset = Dataset("data")
        index0 = dataset.add_pose(self.dummy_rgb, self.dummy_depth, self.dummy_pose)
        self.assertEqual(index0, 0)
        index1 = dataset.add_pose(self.dummy_rgb, self.dummy_depth, self.dummy_pose)
        self.assertEqual(index1, 1)

    def test_it_return_0_if_no_pair(self):
        self.assertEqual(self.populated_dataset.pair_size(1), 0)

    def test_it_should_add_pair(self):
        self.populated_dataset.add_pair(self.dummy_rgb, self.dummy_depth, self.dummy_pose, 1)
        self.assertEqual(self.populated_dataset.pair_size(1), 1)
        self.populated_dataset.add_pair(self.dummy_rgb, self.dummy_depth, self.dummy_pose, 1)
        self.assertEqual(self.populated_dataset.pair_size(1), 2)

    def test_it_should_raise_indexerror_if_pose_id_does_not_exists(self):
        self.assertRaises(IndexError, self.populated_dataset.add_pair, self.dummy_rgb, self.dummy_depth, self.dummy_pose, 20)
        self.assertRaises(IndexError, self.populated_dataset.add_pair, self.dummy_rgb, self.dummy_depth, self.dummy_pose, 10)

