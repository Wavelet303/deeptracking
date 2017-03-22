import unittest

import numpy as np

from deeptracking.utils.plyparser import PlyParser


class TestTransformMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # run these once as they take time
        cls.basic_preload = PlyParser("data/basic.ply")
        cls.basic_color_preload = PlyParser("data/basic_color.ply")

    def setUp(self):
        self.basic = TestTransformMethods.basic_preload
        self.basic_color = TestTransformMethods.basic_color_preload

    def tearDown(self):
        pass

    def test_it_should_return_none_if_no_texture(self):
        self.assertIsNone(self.basic.get_texture())

    def test_it_should_return_texture_if_texture(self):
        tex = self.basic_color_preload.get_texture()
        self.assertEqual(tex.shape, (4096, 4096, 3))

    def test_it_should_return_vertex(self):
        vertex = self.basic.get_vertex()
        truth = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 1, 1],
                          [0, 1, 0],
                          [1, 0, 0],
                          [1, 0, 1],
                          [1, 1, 1],
                          [1, 1, 0]], dtype=np.float32)

        np.testing.assert_almost_equal(vertex, truth, 4)

    def test_it_should_return_color(self):
        colors = self.basic_color.get_vertex_color()
        truth = np.zeros((8, 3), dtype=np.uint8)
        truth.fill(255)

        np.testing.assert_almost_equal(colors, truth, 4)

    def test_it_should_return_none_if_no_color(self):
        self.assertIsNone(self.basic.get_vertex_color())

    def test_it_should_return_texture_coords(self):
        coords = self.basic_color.get_texture_coord()
        truth = np.zeros((8, 2), dtype=np.float32)
        truth.fill(0.1)

        np.testing.assert_almost_equal(coords, truth, 4)

    def test_it_should_return_none_if_no_texture_coord(self):
        self.assertIsNone(self.basic.get_texture_coord())

    def test_it_should_return_faces(self):
        faces = self.basic_color.get_faces()
        truth = np.array([[0, 1, 2, 3],
                          [7, 6, 5, 4],
                          [0, 4, 5, 1],
                          [1, 5, 6, 2],
                          [2, 6, 7, 3],
                          [3, 7, 4, 0]], dtype=np.int32)
        np.testing.assert_almost_equal(faces, truth, 4)

    def test_it_should_return_none_if_no_faces(self):
        self.assertIsNone(self.basic.get_faces())