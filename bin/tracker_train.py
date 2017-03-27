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


if __name__ == '__main__':
    args = ArgumentParser(sys.argv[1:])
    if args.help:
        args.print_help()
        sys.exit(1)

    with open(args.config_file) as data_file:
        data = json.load(data_file)

    # Populate important data from config file
    TRAIN_PATH = data["train_path"]
    VALID_PATH = data["valid_path"]
    MEAN_STD_PATH = data["mean_std_path"]

    train_dataset = Dataset(TRAIN_PATH)
    train_dataset.load()
    valid_dataset = Dataset(VALID_PATH)
    valid_dataset.load()
    # dataset setup

    tracker = DeepTracker(train_dataset.camera,
                          MEAN_STD_PATH)
    tracker.set_configs_({
        "learningRate" : int(data["training_param"]["learning_rate"]),
        "inputSize": int(data["training_param"]["input_size"]),
        "weightDecay": float(data["training_param"]["weight_decay"]),
        "learningRateDecay": float(data["training_param"]["learning_rate_decay"]),
        "linearSize": int(data["training_param"]["linear_size"]),
        "convo1Filters": int(data["training_param"]["convo1_filters"]),
        "convo2Filters": int(data["training_param"]["convo2_filters"]),
    })
    tracker.print()