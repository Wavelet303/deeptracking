"""
    Transforms contain utility functions to manipulate pointclouds

    date : 2017-20-03
"""

__author__ = "Mathieu Garon"
__version__ = "0.0.1"

from plyfile import PlyData, PlyElement
import os
import numpy as np
from PIL import Image


class PlyParser:
    def __init__(self, path):
        self.path = path
        self.data = PlyData.read(path)

    def get_texture(self):
        for comment in self.data.comments:
            if "TextureFile" in comment:
                tex_path = os.path.join(os.path.dirname(self.path), comment.split(" ")[-1])
                return np.array(Image.open(tex_path)).astype(np.uint8)
        return None

    def get_vertex(self):
        for element in self.data.elements:
            if element.name == "vertex":
                return PlyParser.recarray_to_array(element.data[["x", "y", "z"]], np.float32)

    def get_vertex_color(self):
        for element in self.data.elements:
            if element.name == "vertex":
                try:
                    return PlyParser.recarray_to_array(element.data[["red", "green", "blue"]], np.uint8)
                except ValueError:
                    break
        return None

    def get_vertex_normals(self):
        for element in self.data.elements:
            if element.name == "vertex":
                try:
                    return PlyParser.recarray_to_array(element.data[["nx", "ny", "nz"]], np.float32)
                except ValueError:
                    break
        return None

    def get_texture_coord(self):
        for element in self.data.elements:
            if element.name == "vertex":
                try:
                    return PlyParser.recarray_to_array(element.data[["texture_u", "texture_v"]], np.float32)
                except ValueError:
                    break
        return None

    def get_faces(self):
        for element in self.data.elements:
            if element.name == "face":
                try:
                    faces_object = element.data["vertex_indices"]
                except ValueError:
                    break
                # take for granted that all faces are of same lenght
                faces = np.ndarray((len(faces_object), len(faces_object[0])), dtype=np.uint32)
                for i, face in enumerate(faces_object):
                    faces[i, :] = face
                return faces
        return None

    @staticmethod
    def recarray_to_array(array, type):
        return array.view(type).reshape(array.shape + (-1,))
