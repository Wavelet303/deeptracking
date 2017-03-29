import PyTorchHelpers

from deeptracking.data.dataaugmentation import DataAugmentation
from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.data.dataset import Dataset
import sys
import json
import logging
import logging.config
from datetime import datetime
import os

from deeptracking.utils.slack_logger import SlackLogger


def config_logging(data):
    logging_filename = "{}.log".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logging_path = data["logging"]["path"]
    path = os.path.join(logging_path, logging_filename)
    if not os.path.exists(logging_path):
        os.mkdir(logging_path)
    dictLogConfig = {
        "version": 1,
        "handlers": {
            "fileHandler": {
                "class": "logging.FileHandler",
                "formatter": "basic_formatter",
                "filename": path
            }
        },
        "loggers": {
            "Model Training": {
                "handlers": ["fileHandler"],
                "level": data["logging"]["level"],
            }
        },

        "formatters": {
            "basic_formatter": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    }
    logging.setLoggerClass(SlackLogger)
    logging.config.dictConfig(dictLogConfig)

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

    RGB_NOISE = float(data["data_augmentation"]["rgb_noise"])
    DEPTH_NOISE = float(data["data_augmentation"]["depth_noise"])
    OCCLUDER_PATH = data["data_augmentation"]["occluder_path"]
    BACKGROUND_PATH = data["data_augmentation"]["background_path"]
    BLUR_NOISE = int(data["data_augmentation"]["blur_noise"])
    HUE_NOISE = float(data["data_augmentation"]["hue_noise"])

    LEARNING_RATE = float(data["training_param"]["learning_rate"])
    LEARNING_RATE_DECAY = float(data["training_param"]["learning_rate_decay"])
    WEIGHT_DECAY = float(data["training_param"]["weight_decay"])
    INPUT_SIZE = int(data["training_param"]["input_size"])
    LINEAR_SIZE = int(data["training_param"]["linear_size"])
    CONVO1_SIZE = int(data["training_param"]["convo1_size"])
    CONVO2_SIZE = int(data["training_param"]["convo2_size"])

    config_logging(data)
    logger = logging.getLogger("Model Training")
    print(logger)

    logger.info("Setup Datasets")
    data_augmentation = DataAugmentation()
    data_augmentation.set_rgb_noise(RGB_NOISE)
    data_augmentation.set_depth_noise(DEPTH_NOISE)
    if OCCLUDER_PATH != "":
        data_augmentation.set_occluder(OCCLUDER_PATH)
    if BACKGROUND_PATH != "":
        data_augmentation.set_background(BACKGROUND_PATH)
    data_augmentation.set_blur(BLUR_NOISE)
    data_augmentation.set_hue_noise(HUE_NOISE)

    train_dataset = Dataset(TRAIN_PATH)
    train_dataset.load()
    train_dataset.set_data_augmentation(data_augmentation)
    valid_dataset = Dataset(VALID_PATH)
    valid_dataset.load()
    valid_dataset.set_data_augmentation(data_augmentation)

    logger.info("Setup Model")
    model_class = PyTorchHelpers.load_lua_class("deeptracking/model/rgbd_tracker.lua", 'RGBDTracker')
    tracker_model = model_class('cuda')
    tracker_model.set_configs({
        "learningRate": LEARNING_RATE,
        "learningRateDecay": LEARNING_RATE_DECAY,
        "weightDecay": WEIGHT_DECAY,
        "input_size": INPUT_SIZE,
        "linear_size": LINEAR_SIZE,
        "convo1_size": CONVO1_SIZE,
        "convo2_size": CONVO2_SIZE
    })
    tracker_model.build_model()
    tracker_model.init_model()
    logger.debug(tracker_model.model_string())
    logger.slack("test")
    logger.data("oui", 3)
