from deeptracking.detector.detector_aruco import ArucoDetector
from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.data.sensors.kinect2 import Kinect2
from deeptracking.utils.camera import Camera
from deeptracking.utils.transform import Transform
from deeptracking.data.dataset import Dataset
from deeptracking.data.frame import Frame, FrameNumpy
from deeptracking.data.dataset_utils import combine_view_transform
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.data.dataset_utils import normalize_scale
from deeptracking.utils.uniform_sphere_sampler import UniformSphereSampler
import sys
import json
import os
import cv2
import math
import numpy as np


save_next = False


def on_click(event, x, y, flags, param):
    global save_next
    if event == cv2.EVENT_LBUTTONDBLCLK:
        save_next = True

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
    IMAGE_SIZE = (int(data["image_size"]), int(data["image_size"]))
    CAMERA_PATH = data["camera_path"]
    DETECTOR_PATH = data["detector_layout_path"]
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    sensor = Kinect2(CAMERA_PATH)
    camera = sensor.intrinsics()
    ratio = 2
    camera.set_ratio(ratio)
    sensor.start()

    window = InitOpenGL(camera.width, camera.height)
    detector = ArucoDetector(camera, DETECTOR_PATH)
    vpRender = ModelRenderer(MODELS[0]["model_path"], SHADER_PATH, camera, window)
    vpRender.load_ambiant_occlusion_map(MODELS[0]["ambiant_occlusion_model"])

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_click)

    T = Transform.from_parameters(-0.0017330130795, 0.00853765942156, -0.102324359119,
                                  0.242059546511, -1.22307961834, 2.01838164219,
                                  True)

    while True:
        bgr, depth = sensor.get_frame()
        bgr = cv2.resize(bgr, (int(1920 / ratio), int(1080 / ratio)))
        depth = cv2.resize(depth, (int(1920 / ratio), int(1080 / ratio)))

        if args.verbose:
            cv2.imshow("image", bgr[:, :, ::-1])
            #print("Frame Grab time :{}".format(sensor_time))
            #print("Processing time : {}".format(processing_time))

        key = cv2.waitKey(1)
        if key != -1:
            print(key)
        if key == 1048603:  # escape key
            break
        #elif key == 1048608:  # space key
            #record = not record
        elif key == 1114033:  # 1
            T.rotate(z=math.radians(-1))
        elif key == 1114034:  # 2
            T.translate(z=0.001)
        elif key == 1114035:  # 3
            T.rotate(x=math.radians(-1))
        elif key == 1114036:  # 4
            T.translate(x=-0.001)
        #elif key == 1114037:  # 5
            #show_debug = not show_debug
        elif key == 1114038:  # 6
            T.translate(x=0.001)
        elif key == 1114039:  # 7
            T.rotate(z=math.radians(1))
        elif key == 1114040:  # 8
            T.translate(z=-0.001)
        elif key == 1114041:  # 9
            T.rotate(x=math.radians(1))
        elif key == 1113938:  # up
            T.translate(y=-0.001)
        elif key == 1113940:  # down
            T.translate(y=0.001)
        elif key == 1113937:  # left
            T.rotate(y=math.radians(-1))
        elif key == 1113939:  # right
            T.rotate(y=math.radians(1))
        elif key == 1048688:  # p
            print(T)
    sensor.stop()
