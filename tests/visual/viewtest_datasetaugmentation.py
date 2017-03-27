import cv2
import numpy as np
import matplotlib.pyplot as plt
from deeptracking.data.dataset import Dataset
from deeptracking.data.dataaugmentation import DataAugmentation

if __name__ == '__main__':

    object_path = "../../synthetic_dataset_test"
    occluder_path = "/home/mathieu/Dataset/DeepTrack/hand_150k"
    background_path = "/home/mathieu/Dataset/RGBD/Realsense_backgrounds"
    object_dataset = Dataset(object_path)
    object_dataset.load()

    data_augmentation = DataAugmentation()
    data_augmentation.set_rgb_noise(2)
    #data_augmentation.set_depth_noise(2)
    #data_augmentation.set_hue_noise(0.05)
    #data_augmentation.set_occluder(occluder_path)
    #data_augmentation.set_background(background_path)
    #data_augmentation.set_blur(3)
    # data_augmentation.set_jitter(20, 20)

    #object_dataset.set_data_augmenter(data_augmentation)

    for i in range(10000):
        rgb, depth, pose = object_dataset.load_image(i)
        rgb_augmented, depth_augmented = data_augmentation.augment(rgb, depth, pose, True)

        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.imshow("rgb_augmented", rgb_augmented[:, :, ::-1])

        plt.figure(0)
        plt.imshow(rgb)
        plt.figure(1)
        plt.imshow(rgb_augmented)
        plt.figure(2)
        plt.imshow(rgb - rgb_augmented)
        plt.show()

        plt.figure(0)
        plt.imshow(depth, cmap="gray")
        plt.figure(1)
        plt.imshow(depth_augmented, cmap="gray")
        plt.figure(2)
        plt.imshow(depth.astype(np.int32) - depth_augmented.astype(np.int32))
        plt.show()

        cv2.waitKey()
