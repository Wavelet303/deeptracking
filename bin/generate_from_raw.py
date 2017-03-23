from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.utils.camera import Camera
from deeptracking.utils.transform import Transform
from deeptracking.data.dataset import Dataset
from deeptracking.data.frame import Frame, FrameNumpy
from deeptracking.data.dataset_utils import combine_view_transform, center_pixel
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.data.dataset_utils import normalize_scale
from deeptracking.utils.uniform_sphere_sampler import UniformSphereSampler
from scipy import ndimage
import sys
import json
import os
import math
import cv2
import numpy as np
import random


def mask_real_image(color, depth, depth_render):
    mask = (depth_render != 0).astype(np.uint8)[:, :, np.newaxis]
    masked_rgb = color * mask

    masked_hsv = cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2HSV)
    saturation_mask = (masked_hsv[:, :, 2] <= 255)[:, :, np.newaxis].astype(np.uint8)
    total_mask = np.bitwise_and(mask, saturation_mask)

    kernel = np.ones((3, 3), np.uint8)
    total_mask = cv2.erode(total_mask, kernel)

    masked_color = color * total_mask[:, :, np.newaxis]
    # hack
    if depth.shape[0] > total_mask.shape[0]:
        depth = depth[:-2, :]
    masked_depth = depth * total_mask
    return masked_color, masked_depth


def random_z_rotation(rgb, depth, pose):
    rotation = random.uniform(-180, 180)
    rotation_matrix = Transform()
    rotation_matrix.set_rotation(0, 0, math.radians(rotation))

    pixel = center_pixel(pose, dataset.camera)
    new_rgb = rotate_image(rgb, rotation, pixel[0])
    new_depth = rotate_image(depth, rotation, pixel[0])
    new_pose = combine_view_transform(pose, rotation_matrix)
    return new_rgb, new_depth, new_pose

def rotate_image(img, angle, pivot):
    pivot = pivot.astype(np.int32)
    # double size of image while centering object
    pads = [[img.shape[0] - pivot[0], pivot[0]], [img.shape[1] - pivot[1], pivot[1]]]
    if len(img.shape) > 2:
        pads.append([0, 0])
    imgP = np.pad(img, pads, 'constant')
    # reduce size of matrix to rotate around the object
    if len(img.shape) > 2:
        total_y = np.sum(imgP.any(axis=(0, 2))) * 2.4
        total_x = np.sum(imgP.any(axis=(1, 2))) * 2.4
    else:
        total_y = np.sum(imgP.any(axis=0)) * 2.4
        total_x = np.sum(imgP.any(axis=1)) * 2.4
    cropy = (imgP.shape[0] - total_y)/2
    cropx = (imgP.shape[1] - total_x)/2
    imgP[cropy:-cropy, cropx:-cropx] = ndimage.rotate(imgP[cropy:-cropy, cropx:-cropx], angle, reshape=False)
    return imgP[pads[0][0]: -pads[0][1], pads[1][0]: -pads[1][1]]

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

    dataset = Dataset(OUTPUT_PATH)
    dataset.load()
    window = InitOpenGL(dataset.camera.width, dataset.camera.height)

    model = MODELS[0]
    vpRender = ModelRenderer(model["model_path"], SHADER_PATH, dataset.camera, window)
    vpRender.load_ambiant_occlusion_map(model["ambiant_occlusion_model"])
    OBJECT_WIDTH = int(model["object_width"])

    for i in range(dataset.size()):
        frame, pose = dataset.data_pose[i]

        rgb_render, depth_render = vpRender.render(pose.transpose())
        rgb_render = cv2.resize(rgb_render, (dataset.camera.width, dataset.camera.height))
        depth_render = cv2.resize(depth_render, (dataset.camera.width, dataset.camera.height))

        rgb, depth = frame.get_rgb_depth(dataset.path)
        masked_rgb, masked_depth = mask_real_image(rgb, depth, depth_render)

        for j in range(SAMPLE_QUANTITY):
            masked_rgb, masked_depth, pose = random_z_rotation(masked_rgb, masked_depth, pose)
            random_transform = Transform.random((-TRANSLATION_RANGE, TRANSLATION_RANGE),
                                                (-ROTATION_RANGE, ROTATION_RANGE))
            inverted_random_transform = Transform.from_parameters(*(-random_transform.to_parameters()))
            previous_pose = pose.copy()
            previous_pose = combine_view_transform(previous_pose, inverted_random_transform)

            rgbA, depthA = vpRender.render(previous_pose.transpose())
            rgbA, depthA = normalize_scale(rgbA, depthA, previous_pose.inverse(), dataset.camera, IMAGE_SIZE, OBJECT_WIDTH)
            rgbB, depthB = normalize_scale(masked_rgb, masked_depth, previous_pose.inverse(), dataset.camera, IMAGE_SIZE, OBJECT_WIDTH)

            index = dataset.add_pose(rgbA, depthA, previous_pose)
            dataset.add_pair(rgbB, depthB, random_transform, index)

            sys.stdout.write("Progress: %d%%   \r" % (int(i / SAMPLE_QUANTITY * 100)))
            sys.stdout.flush()

            if args.verbose:
                import cv2

                cv2.imshow("testA", rgbA[:, :, ::-1])
                cv2.imshow("testB", rgbB[:, :, ::-1])
                cv2.waitKey()
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

    dataset.dump_on_disk(metadata)
