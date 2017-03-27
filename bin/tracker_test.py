
from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.data.dataset import Dataset
from deeptracking.tracker.deeptracker import DeepTracker
import sys
import json
import time
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

    video_data = Dataset(VIDEO_PATH)
    if video_data.load() == -1:
        print("Error Loading video data...")
        sys.exit(-1)
    OBJECT_WIDTH = int(MODELS_3D[0]["object_width"])
    MODEL_3D_PATH = MODELS_3D[0]["model_path"]
    MODEL_3D_AO_PATH = MODELS_3D[0]["ambiant_occlusion_model"]

    tracker = DeepTracker(MODEL_PATH,
                          MODEL_3D_PATH,
                          MODEL_3D_AO_PATH,
                          SHADER_PATH,
                          video_data.camera,
                          MEAN_STD_PATH,
                          OBJECT_WIDTH)

    previous_frame, previous_pose = video_data.data_pose[0]
    previous_rgb, previous_depth = previous_frame.get_rgb_depth(video_data.path)
    previous_pose = previous_pose.inverse()

    for i in range(1, video_data.size()):
        # get actual frame
        current_frame, current_pose = video_data.data_pose[i]
        current_rgb, current_depth = current_frame.get_rgb_depth(video_data.path)

        # process pose estimation of current frame given last pose
        start_time = time.time()
        previous_pose = tracker.estimate_current_pose(previous_pose, current_rgb, current_depth)
        print("Estimation processing time : {}".format(time.time() - start_time))
        if args.verbose:
            debug = tracker.get_debug_screen(previous_rgb)
            cv2.imshow("Debug", debug[:, :, ::-1])
            cv2.waitKey()
        previous_rgb = current_rgb

