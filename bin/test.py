from deeptracking.data.dataset_utils import image_blend, angle_distance
from deeptracking.data.sensors.kinect2 import Kinect2
from deeptracking.data.sensors.viewpointgenerator import ViewpointGenerator
from deeptracking.detector.detector_aruco import ArucoDetector
from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.data.dataset import Dataset
from deeptracking.tracker.deeptracker import DeepTracker
from deeptracking.utils.filters import MeanFilter
import sys
import json
import time
import cv2
import numpy as np

from deeptracking.utils.data_logger import DataLogger
import os
ESCAPE_KEY = 1048603
UNITY_DEMO = False


def log_pose_difference(prediction, ground_truth, logger):
    prediction_params = prediction.to_parameters(isDegree=True)
    ground_truth_params = ground_truth.to_parameters(isDegree=True)
    difference = np.zeros(6)
    for j in range(3):
        difference[j] = abs(prediction_params[j] - ground_truth_params[j])
        difference[j + 3] = abs(angle_distance(prediction_params[j + 3], ground_truth_params[j + 3]))
    logger.add_row(logger.get_dataframes_id()[0], difference)

if __name__ == '__main__':

    if UNITY_DEMO:
        TCP_IP = "0.0.0.0"
        TCP_PORT = 9050
        print("Activating Unity server on {}:{}".format(TCP_IP, TCP_PORT))
        sys.path.insert(0, '/home/mathieu/source/Camera_streamer_for_Unity')
        import pycam_server.server as server
        from pycam_server.frame import ExampleMetadata

        meta = ExampleMetadata()
        unity_server = server.Server(TCP_IP, TCP_PORT)
        while not unity_server.has_connection():
            time.sleep(1)
        output_rot_filter = MeanFilter(2)
        output_trans_filter = MeanFilter(2)

    args = ArgumentParser(sys.argv[1:])
    if args.help:
        args.print_help()
        sys.exit(1)

    with open(args.config_file) as data_file:
        data = json.load(data_file)

    # Populate important data from config file
    OUTPUT_PATH = data["output_path"]
    VIDEO_PATH = data["video_path"]
    MODEL_PATH = data["model_path"]
    model_split_path = MODEL_PATH.split(os.sep)
    model_name = model_split_path[-1]
    model_folder = os.sep.join(model_split_path[:-1])
    MODELS_3D = data["models"]
    SHADER_PATH = data["shader_path"]

    OBJECT_WIDTH = int(MODELS_3D[0]["object_width"])
    MODEL_3D_PATH = MODELS_3D[0]["model_path"]
    MODEL_3D_AO_PATH = MODELS_3D[0]["ambiant_occlusion_model"]
    USE_SENSOR = data["use_sensor"] == "True"
    RESET_FREQUENCY = int(data["reset_frequency"])
    frame_download_path = None
    use_ground_truth_pose = True

    if USE_SENSOR:
        sensor = Kinect2(data["sensor_camera_path"])
        detector = ArucoDetector(sensor.camera, data["detector_layout_path"])
        frame_generator = ViewpointGenerator(sensor, detector)
        camera = sensor.camera
    else:
        video_data = Dataset(VIDEO_PATH)
        if not video_data.load():
            print("[ERROR] Error while loading video...")
            sys.exit(-1)
        frame_download_path = video_data.path
        # Makes the list a generator for compatibility with sensor's generator
        gen = lambda alist: [(yield i) for i in alist]
        frame_generator = gen(video_data.data_pose)
        camera = video_data.camera
        use_ground_truth_pose = False

    tracker = DeepTracker(camera,
                          data["model_file"],
                          OBJECT_WIDTH,
                          MODEL_3D_PATH,
                          MODEL_3D_AO_PATH,
                          SHADER_PATH)
    tracker.load(MODEL_PATH)
    tracker.print()
    previous_frame, previous_pose = next(frame_generator)
    previous_rgb, previous_depth = previous_frame.get_rgb_depth(frame_download_path)
    previous_pose = previous_pose.inverse()
    data_logger = DataLogger()
    data_logger.create_dataframe("{}_eval".format(model_name), ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz"))
    for i, (current_frame, ground_truth_pose) in enumerate(frame_generator):
        # get actual frame
        current_rgb, current_depth = current_frame.get_rgb_depth(frame_download_path)
        screen = current_rgb

        if use_ground_truth_pose:
            previous_pose = ground_truth_pose.inverse()
            if previous_pose is not None:
                rgb, depth = tracker.renderer.render(previous_pose.inverse().transpose())
                screen = image_blend(rgb, current_rgb)
        else:
            if RESET_FREQUENCY != 0 and i % RESET_FREQUENCY == 0:
                previous_pose = ground_truth_pose.inverse()
            # process pose estimation of current frame given last pose
            start_time = time.time()
            for i in range(2):
                predicted_pose = tracker.estimate_current_pose(previous_pose, current_rgb, current_depth, debug=args.verbose)

            print("Estimation processing time : {}".format(time.time() - start_time))
            screen = tracker.get_debug_screen(previous_rgb)
            if not USE_SENSOR:
                log_pose_difference(predicted_pose.inverse(), ground_truth_pose, data_logger)
            previous_pose = predicted_pose

        previous_rgb = current_rgb
        if UNITY_DEMO:
            if meta.camera_parameters is None:
                meta.camera_parameters = camera.copy()
                meta.camera_parameters.distortion = meta.camera_parameters.distortion.tolist()
            meta.object_pose = []
            if previous_pose:
                params = previous_pose.inverse().to_parameters()
                params[3:] = output_rot_filter.compute_mean(params[3:])
                params[:3] = output_trans_filter.compute_mean(params[:3])
                meta.add_object_pose(*params)
            unity_server.send_data_to_clients(current_rgb[:, :, ::-1], meta)

        cv2.imshow("Debug", screen[:, :, ::-1])
        key = cv2.waitKey(1)
        key_chr = chr(key & 255)
        if key != -1:
            print("pressed key id : {}, char : [{}]".format(key, key_chr))
        if key == ESCAPE_KEY:
            break
        elif key_chr == ' ':
            use_ground_truth_pose = not use_ground_truth_pose
            frame_generator.compute_detection(use_ground_truth_pose)
    log_folder = os.path.join(model_folder, "scores")
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    data_logger.save(log_folder)