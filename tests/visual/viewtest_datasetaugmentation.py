import numpy as np
import matplotlib.pyplot as plt
from deeptracking.data.dataset import Dataset
from deeptracking.data.dataaugmentation import DataAugmentation

if __name__ == '__main__':

    object_path = "/home/mathieu/Dataset/DeepTrack/skull/train_cyclegan"
    occluder_path = "/home/mathieu/Dataset/DeepTrack/mixed/test"
    background_path = "/home/mathieu/Dataset/RGBD/SUN3D"

    object_dataset = Dataset(object_path)
    object_dataset.load()

    data_augmentation = DataAugmentation()
    data_augmentation.set_rgb_noise(2)
    data_augmentation.set_depth_noise(2)
    data_augmentation.set_hue_noise(0.07)
    data_augmentation.set_occluder(occluder_path)
    data_augmentation.set_background(background_path)
    data_augmentation.set_blur(5)
    # data_augmentation.set_jitter(20, 20)

    for i in range(object_dataset.size()):
        rgb, depth, pose = object_dataset.load_image(i)
        rgb, depth, label = object_dataset.load_pair(i, 0)
        rgb_augmented, depth_augmented = data_augmentation.augment(rgb, depth, pose, True)

        plt.figure(0)
        plt.imshow(rgb)
        plt.figure(1)
        plt.imshow(rgb_augmented)
        plt.figure(2)
        plt.imshow(rgb - rgb_augmented)
        plt.show()

        plt.figure(0)
        plt.imshow(depth)
        plt.figure(1)
        plt.imshow(depth_augmented)
        plt.figure(2)
        plt.imshow(depth.astype(np.int32) - depth_augmented.astype(np.int32))
        plt.show()
