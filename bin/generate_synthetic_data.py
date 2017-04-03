from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.utils.camera import Camera
from deeptracking.utils.transform import Transform
from deeptracking.data.dataset import Dataset
from deeptracking.data.dataset_utils import combine_view_transform, show_frames
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.data.dataset_utils import normalize_scale
from deeptracking.utils.uniform_sphere_sampler import UniformSphereSampler
import sys
import json
import os
import math
import cv2
ESCAPE_KEY = 1048603


if __name__ == '__main__':

    args = ArgumentParser(sys.argv[1:])
    if args.help:
        args.print_help()
        sys.exit(1)

    with open(args.config_file) as data_file:
        data = json.load(data_file)

    # Populate important data from config file
    MODELS = data["models"]
    SHADER_PATH = data["shader_path"]
    OUTPUT_PATH = data["output_path"]
    SAMPLE_QUANTITY = int(data["sample_quantity"])
    TRANSLATION_RANGE = float(data["translation_range"])
    ROTATION_RANGE = math.radians(float(data["rotation_range"]))
    SPHERE_MIN_RADIUS = float(data["sphere_min_radius"])
    SPHERE_MAX_RADIUS = float(data["sphere_max_radius"])
    IMAGE_SIZE = (int(data["image_size"]), int(data["image_size"]))
    PRELOAD = data["preload"] == "True"

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # Write important misc data to file
    metadata = {}
    metadata["translation_range"] = str(TRANSLATION_RANGE)
    metadata["rotation_range"] = str(ROTATION_RANGE)
    metadata["image_size"] = str(IMAGE_SIZE[0])
    metadata["object_width"] = {}
    for model in MODELS:
        metadata["object_width"][model["name"]] = str(model["object_width"])
    metadata["min_radius"] = str(SPHERE_MIN_RADIUS)
    metadata["max_radius"] = str(SPHERE_MAX_RADIUS)

    camera = Camera.load_from_json(data["camera_path"])
    dataset = Dataset(OUTPUT_PATH)
    dataset.camera = camera
    window = InitOpenGL(camera.width, camera.height)
    sphere_sampler = UniformSphereSampler(SPHERE_MIN_RADIUS, SPHERE_MAX_RADIUS)
    preload_count = 0
    if PRELOAD:
        if dataset.load(viewpoint_file_only=True):
            preload_count = dataset.size()
            print("This Dataset already contains {} samples".format(preload_count))
    # Iterate over all models from config files
    for model in MODELS:
        vpRender = ModelRenderer(model["model_path"], SHADER_PATH, dataset.camera, window)
        vpRender.load_ambiant_occlusion_map(model["ambiant_occlusion_model"])
        OBJECT_WIDTH = int(model["object_width"])
        for i in range(SAMPLE_QUANTITY - preload_count):
            random_pose = sphere_sampler.get_random()
            random_transform = Transform.random((-TRANSLATION_RANGE, TRANSLATION_RANGE),
                                                (-ROTATION_RANGE, ROTATION_RANGE))
            pair = combine_view_transform(random_pose, random_transform)

            rgbA, depthA = vpRender.render(random_pose.transpose())
            rgbB, depthB = vpRender.render(pair.transpose(), sphere_sampler.random_direction())
            rgbA, depthA = normalize_scale(rgbA, depthA, random_pose.inverse(), dataset.camera, IMAGE_SIZE, OBJECT_WIDTH)
            rgbB, depthB = normalize_scale(rgbB, depthB, random_pose.inverse(), dataset.camera, IMAGE_SIZE, OBJECT_WIDTH)

            index = dataset.add_pose(rgbA, depthA, random_pose)
            dataset.add_pair(rgbB, depthB, random_transform, index)

            sys.stdout.write("Progress: %d%%   \r" % (int((i + preload_count) / SAMPLE_QUANTITY * 100)))
            sys.stdout.flush()

            if i % 50 == 0:
                dataset.dump_on_disk(metadata)

            if args.verbose:
                show_frames(rgbA, depthA, rgbB, depthB)
            cv2.imshow("testB", rgbB[:, :, ::-1])
            k = cv2.waitKey(1)
            if k == ESCAPE_KEY:
                break

    dataset.dump_on_disk(metadata)
