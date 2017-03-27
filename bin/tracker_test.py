from deeptracking.data.dataset_utils import image_blend
from deeptracking.data.sensors.kinect2 import Kinect2
from deeptracking.data.sensors.viewpointgenerator import ViewpointGenerator
from deeptracking.detector.detector_aruco import ArucoDetector
from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.data.dataset import Dataset
from deeptracking.tracker.deeptracker import DeepTracker
import sys
import json
import time
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
    VIDEO_PATH = data["video_path"]
    MODEL_PATH = data["model_path"]
    MODELS_3D = data["models"]
    SHADER_PATH = data["shader_path"]
    MEAN_STD_PATH = data["mean_std_path"]

    OBJECT_WIDTH = int(MODELS_3D[0]["object_width"])
    MODEL_3D_PATH = MODELS_3D[0]["model_path"]
    MODEL_3D_AO_PATH = MODELS_3D[0]["ambiant_occlusion_model"]
    USE_SENSOR = data["use_sensor"] == "True"
    frame_download_path = None

    if USE_SENSOR:
        sensor = Kinect2(data["sensor_camera_path"])
        detector = ArucoDetector(sensor.camera, data["detector_layout_path"])
        frame_generator = ViewpointGenerator(sensor, detector)
        camera = sensor.camera
        sensor.start()
    else:
        video_data = Dataset(VIDEO_PATH)
        if video_data.load() == -1:
            print("Error Loading video data...")
            sys.exit(-1)
        frame_download_path = video_data.path
        gen = lambda alist: [(yield i) for i in alist]  #simply make the list a generator for compatibility with sensor's generator
        frame_generator = gen(video_data.data_pose)
        camera = video_data.camera

    tracker = DeepTracker(MODEL_PATH,
                          MODEL_3D_PATH,
                          MODEL_3D_AO_PATH,
                          SHADER_PATH,
                          camera,
                          MEAN_STD_PATH,
                          OBJECT_WIDTH)

    previous_frame, previous_pose = next(frame_generator)
    previous_rgb, previous_depth = previous_frame.get_rgb_depth(frame_download_path)
    previous_pose = previous_pose.inverse()
    use_ground_truth_pose = False

    for current_frame, current_pose in frame_generator:
        # get actual frame
        current_rgb, current_depth = current_frame.get_rgb_depth(frame_download_path)

        start_time = time.time()
        if use_ground_truth_pose:
            previous_pose = current_pose.inverse()
            if args.verbose:
                if previous_pose is not None:
                    rgb, depth = tracker.renderer.render(previous_pose.inverse().transpose())
                    screen = image_blend(rgb, current_rgb)
                else:
                    screen = current_rgb
                cv2.imshow("Debug", screen[:, :, ::-1])
        else:
            # process pose estimation of current frame given last pose
            previous_pose = tracker.estimate_current_pose(previous_pose, current_rgb, current_depth)
            if args.verbose:
                screen = tracker.get_debug_screen(previous_rgb)
                cv2.imshow("Debug", screen[:, :, ::-1])
        print("Estimation processing time : {}".format(time.time() - start_time))
        previous_rgb = current_rgb

        key = cv2.waitKey()
        key_chr = chr(key & 255)
        if key != -1:
            print("pressed key id : {}, char : [{}]".format(key, key_chr))
        if key == ESCAPE_KEY:
            break
        elif key_chr == ' ':
            use_ground_truth_pose = not use_ground_truth_pose

