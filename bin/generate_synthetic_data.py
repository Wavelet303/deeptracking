from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.utils.camera import Camera
from deeptracking.utils.transform import Transform
from deeptracking.data.dataset_utils import combine_view_transform
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.data.dataset_utils import normalize_scale, numpy_int16_to_uint8, save_rgbd_as_numpy
from deeptracking.utils.uniform_sphere_sampler import UniformSphereSampler
import sys
import json
import os
import math
import numpy as np


def merge_image(color, depth):
    depth8 = numpy_int16_to_uint8(depth)
    frame = np.concatenate((color, depth8), axis=2)
    return frame

def save(frame, transform, id, vp_dict, path):
    np.save(os.path.join(path, id), frame)
    params = {}
    for i, param in enumerate(transform.to_parameters()):
        params[str(i)] = str(param)
    vp_dict[id] = {"vector": params}

if __name__ == '__main__':

    args = ArgumentParser(sys.argv[1:])
    if args.help:
        args.print_help()
        sys.exit(1)

    with open(args.config_file) as data_file:
        data = json.load(data_file)

    MODELS = data["models"]
    SHADER_PATH = data["shader_path"]
    OUTPUT_PATH = data["output_path"]
    SAMPLE_QUANTITY = int(data["sample_quantity"])
    RANDOM_LIGHT_DIRECTION = data["random_light_direction"] == "True"
    RANDOM_LIGHT_POWER = data["random_light_power"] == "True"
    TRANSLATION_RANGE = float(data["translation_range"])
    ROTATION_RANGE = math.radians(float(data["rotation_range"]))
    SPHERE_MIN_RADIUS = float(data["sphere_min_radius"])
    SPHERE_MAX_RADIUS = float(data["sphere_max_radius"])
    IMAGE_SIZE = (int(data["image_size"]), int(data["image_size"]))

    camera = Camera.load_from_json(data["camera_path"])
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    window = InitOpenGL(camera.width, camera.height)
    sampler = UniformSphereSampler(SPHERE_MIN_RADIUS, SPHERE_MAX_RADIUS)
    dict = {}
    id = 0
    # Iterate over all models from config files
    for model in MODELS:
        vpRender = ModelRenderer(model["model_path"], SHADER_PATH, camera, window)
        vpRender.load_ambiant_occlusion_map(model["ambiant_occlusion_model"])
        OBJECT_WIDTH = int(model["object_width"])
        count = 0
        for sample in sampler:
            random_transform = Transform.random((-TRANSLATION_RANGE, TRANSLATION_RANGE),
                                                (-ROTATION_RANGE, ROTATION_RANGE))
            pair = combine_view_transform(sample, random_transform)

            rgbA, depthA = vpRender.render(sample.transpose())
            rgbB, depthB = vpRender.render(pair.transpose(), sampler.random_direction())

            rgbA, depthA = normalize_scale(rgbA, depthA, sample.inverse(), vpRender.camera, IMAGE_SIZE, OBJECT_WIDTH)
            rgbB, depthB = normalize_scale(rgbB, depthB, sample.inverse(), vpRender.camera, IMAGE_SIZE, OBJECT_WIDTH)

            if args.verbose:
                import cv2
                cv2.imshow("testA", rgbA[:, :, ::-1])
                cv2.imshow("testB", rgbB[:, :, ::-1])
                cv2.waitKey()

            save_rgbd_as_numpy(rgbA, depthA, OUTPUT_PATH, str(id))
            save_rgbd_as_numpy(rgbB, depthB, OUTPUT_PATH, "{}n0".format(id))

            frameA = merge_image(rgbA, depthA)
            frameB = merge_image(rgbB, depthB)

            save(frameA, sample, str(id), dict, OUTPUT_PATH)
            save(frameB, random_transform, "{}n0".format(id), dict, OUTPUT_PATH)

            count += 1
            id += 1
            if count > SAMPLE_QUANTITY:
                break
            sys.stdout.write("Progress: %d%%   \r" % (int(count / SAMPLE_QUANTITY * 100)))
            sys.stdout.flush()
