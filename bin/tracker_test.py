import PyTorchHelpers

from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.utils.camera import Camera
from deeptracking.utils.transform import Transform
from deeptracking.data.dataset import Dataset
from deeptracking.data.dataset_utils import combine_view_transform
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.data.dataset_utils import normalize_scale, normalize_image, unnormalize_label
from deeptracking.utils.uniform_sphere_sampler import UniformSphereSampler
import sys
import json
import numpy as np
import os
import math
import cv2

if __name__ == '__main__':
    args = ArgumentParser(sys.argv[1:])
    if args.help:
        args.print_help()
        sys.exit(1)

    with open(args.config_file) as data_file:
        data = json.load(data_file)

    # Populate important data from config file
    VIDEO_PATH = data["video_path"]
    MODEL_PATH = data["model_path"]
    MODELS_3D = data["models"]
    SHADER_PATH = data["shader_path"]
    MEAN_STD_PATH = data["mean_std_path"]

    mean = np.load(os.path.join(MEAN_STD_PATH, "mean.npy"))
    std = np.load(os.path.join(MEAN_STD_PATH, "std.npy"))

    video_data = Dataset(VIDEO_PATH)
    if video_data.load() == -1:
        print("Error Loading video data...")
        sys.exit(-1)
    window = InitOpenGL(video_data.camera.width, video_data.camera.height)

    # load torch model
    ModelClass = PyTorchHelpers.load_lua_class("deeptracking/model/rgbd_tracker.lua", 'RGBDTracker')
    tracker_model = ModelClass('cuda')
    tracker_model.load(MODEL_PATH)
    tracker_model.show_model()
    IMAGE_SIZE = (int(tracker_model.get_configs("inputSize")), int(tracker_model.get_configs("inputSize")))

    vpRender = ModelRenderer(MODELS_3D[0]["model_path"], SHADER_PATH, video_data.camera, window)
    vpRender.load_ambiant_occlusion_map(MODELS_3D[0]["ambiant_occlusion_model"])
    OBJECT_WIDTH = int(MODELS_3D[0]["object_width"])

    net_input = np.ndarray((1, 8, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.float32)
    net_prior = np.ndarray((1, 7), dtype=np.float32)

    current_frame, current_pose = video_data.data_pose[0]
    current_pose = current_pose.inverse()

    for i in range(1, video_data.size()):
        render_rgb, render_depth = vpRender.render(current_pose.inverse().transpose())
        previous_frame = current_frame
        previous_pose = current_pose
        current_frame, current_pose = video_data.data_pose[i]
        rgbB, depthB = current_frame.get_rgb_depth(video_data.path)

        if args.verbose:
            cv2.imshow("Estimation", rgbB[:, :, ::-1])

        rgbA, depthA = normalize_scale(render_rgb, render_depth, previous_pose, video_data.camera, IMAGE_SIZE,
                                       OBJECT_WIDTH)
        rgbB, depthB = normalize_scale(rgbB, depthB, previous_pose, video_data.camera, IMAGE_SIZE, OBJECT_WIDTH)

        if args.verbose:
            cv2.imshow("rgbA", rgbA[:, :, ::-1])
            cv2.imshow("rgbB", rgbB[:, :, ::-1])
            cv2.waitKey()

        rgbA, depthA = normalize_image(rgbA, depthA, mean[:4], std[:4])
        rgbB, depthB = normalize_image(rgbB, depthB, mean[4:], std[4:])

        net_input[0, 0:3, :, :] = rgbA
        net_input[0, 3, :, :] = depthA
        net_input[0, 4:7, :, :] = rgbB
        net_input[0, 7, :, :] = depthB
        net_prior[0] = np.array(previous_pose.to_parameters(isQuaternion=True))
        prediction = tracker_model.test([net_input, net_prior]).asNumpyTensor()
        prediction = unnormalize_label(prediction,
                                       float(tracker_model.get_configs("translation_range")),
                                       float(tracker_model.get_configs("rotation_range")))
        prediction = Transform.from_parameters(*prediction[0], is_degree=True)
        print(prediction)
        current_pose = combine_view_transform(previous_pose.inverse(), prediction).inverse()
