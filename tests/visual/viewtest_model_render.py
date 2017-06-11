"""
    Visual tests of modelrenderer

"""
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.utils.uniform_sphere_sampler import UniformSphereSampler
from deeptracking.utils.camera import Camera
import cv2
import numpy as np
import random
import time


def project_crop_box(transform, box_size):
    obj_x = transform.matrix[0, 3] * 1000
    obj_y = transform.matrix[1, 3] * 1000
    obj_z = transform.matrix[2, 3] * 1000
    offset = box_size / 2
    points = np.ndarray((5, 3), dtype=np.float)
    points[0] = [obj_x - offset, obj_y - offset, -obj_z]
    points[1] = [obj_x - offset, obj_y + offset, -obj_z]
    points[2] = [obj_x + offset, obj_y - offset, -obj_z]
    points[3] = [obj_x + offset, obj_y + offset, -obj_z]
    camera_points = camera.project_points(points).astype(np.int32)
    return camera_points

if __name__ == '__main__':
    camera = Camera.load_from_json("../data/camera.json")
    window_size = (150, 150)
    window = InitOpenGL(*window_size)

    renderers = []

    sampler = UniformSphereSampler(0.5, 2)
    dict = {}
    id = 0
    renderer = ModelRenderer("../data/test.ply", "../../deeptracking/data/shaders", camera, window, window_size)
    renderer.load_ambiant_occlusion_map("../data/test_ao.ply")

    for sample in sampler:
        sample.translate(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(0.01, 0.1))

        start_time = time.time()
        camera_points = project_crop_box(sample, 250)
        renderer.setup_camera(camera, camera_points[0, 1],
                              camera_points[2, 1],
                              camera_points[1, 0],
                              camera_points[0, 0])
        rgb, depth = renderer.render(sample.transpose())
        print("Basic Render time : {}".format(time.time() - start_time))
        cv2.imshow("rgb", rgb[:, :, ::-1])
        # cv2.imshow("depth", (depth / np.max(depth) * 255).astype(np.uint8))
        cv2.waitKey()
