"""
    Visual tests of modelrenderer

"""
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.utils.uniform_sphere_sampler import UniformSphereSampler
from deeptracking.utils.camera import Camera
import cv2
import numpy as np

if __name__ == '__main__':
    camera = Camera.load_from_json("../data/camera.json")
    window = InitOpenGL(camera.width, camera.height)
    renderers = []

    sampler = UniformSphereSampler(0.5, 0.7)
    dict = {}
    id = 0
    renderer = ModelRenderer("../data/test.ply", "../../data/shaders", camera, window)
    renderer.load_ambiant_occlusion_map("../data/test_ao.ply")
    for sample in sampler:
        rgb, depth = renderer.render(sample.transpose())
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.imshow("depth", (depth / np.max(depth) * 255).astype(np.uint8))
        cv2.waitKey()
