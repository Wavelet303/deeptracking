import os
import numpy as np
import json

from deeptracking.data.parallelminibatch import ParallelMinibatch
from deeptracking.utils.transform import Transform
from deeptracking.utils.camera import Camera
from deeptracking.data.frame import Frame


class Dataset(ParallelMinibatch):
    def __init__(self, folder_path, frame_class=Frame, minibatch_size=64, max_parallel_buffer_size=0):
        ParallelMinibatch.__init__(self, max_parallel_buffer_size)
        self.path = folder_path
        self.data_pose = []
        self.data_pair = {}
        self.metadata = {}
        self.camera = None
        self.frame_class = frame_class
        self.mean = None
        self.std = None
        self.data_augmentation = None
        self.minibatch_size = minibatch_size
        #if self.normalize:
        #    try:
        #        self.mean = np.load(os.path.join(normalize_param_folder, "mean.npy"))
        #        self.std = np.load(os.path.join(normalize_param_folder, "std.npy"))
        #    except Exception:
        #        raise IOError("Folder {} does not contain mean.npy and std.npy".format(normalize_param_folder))

    def add_pose(self, rgb, depth, pose):
        index = self.size()
        frame = self.frame_class(rgb, depth, str(index))
        self.data_pose.append((frame, pose))
        return index

    def pair_size(self, id):
        if id not in self.data_pair:
            return 0
        else:
            return len(self.data_pair[id])

    def add_pair(self, rgb, depth, pose, id):
        if id >= len(self.data_pose):
            raise IndexError("impossible to add pair if pose does not exists")
        if id in self.data_pair:
            frame = self.frame_class(rgb, depth, "{}n{}".format(id, len(self.data_pair[id]) - 1))
            self.data_pair[id].append((frame, pose))
        else:
            frame = self.frame_class(rgb, depth, "{}n0".format(id))
            self.data_pair[id] = [(frame, pose)]

    def dump_on_disk(self, metadata={}):
        """
        Unload all images data from ram and save them to the dataset's path ( can be reloaded with load_from_disk())
        :return:
        """
        viewpoints_data = {}
        for frame, pose in self.data_pose:
            self.insert_pose_in_dict(viewpoints_data, frame.id, pose)
            if int(frame.id) in self.data_pair:
                viewpoints_data[frame.id]["pairs"] = len(self.data_pair[int(frame.id)])
                for pair_frame, pair_pose in self.data_pair[int(frame.id)]:
                    self.insert_pose_in_dict(viewpoints_data, pair_frame.id, pair_pose)
                    pair_frame.dump(self.path)
            else:
                viewpoints_data[frame.id]["pairs"] = 0
            frame.dump(self.path)
        viewpoints_data["metaData"] = metadata
        with open(os.path.join(self.path, "viewpoints.json"), 'w') as outfile:
            json.dump(viewpoints_data, outfile)
        self.camera.save(self.path)

    def load(self):
        """
        Load a viewpoints.json to dataset's structure
        Todo: datastructure should be more similar to json structure...
        :return:
        """
        # Load viewpoints file and camera file
        with open(os.path.join(self.path, "viewpoints.json")) as data_file:
            data = json.load(data_file)
        self.camera = Camera.load_from_json(self.path)
        self.metadata = data["metaData"]

        count = 0
        while True:
            try:
                id = str(count)
                pose = Transform.from_parameters(*[float(data[id]["vector"][str(x)]) for x in range(6)])
                self.add_pose(None, None, pose)
                if "pairs" in data[id]:
                    for i in range(int(data[id]["pairs"])):
                        pair_id = "{}n{}".format(id, i)
                        pose = Transform.from_parameters(*[float(data[pair_id]["vector"][str(x)]) for x in range(6)])
                        self.add_pair(None, None, pose, count)
                count += 1

            except KeyError:
                return

    @staticmethod
    def insert_pose_in_dict(dict, key, item):
        params = {}
        for i, param in enumerate(item.to_parameters()):
            params[str(i)] = str(param)
        dict[key] = {"vector": params}

    def size(self):
        return len(self.data_pose)

    def get_valid_index(self):
        return np.arange(0, self.size())

    def set_data_augmentation(self, data_augmentation):
        self.data_augmentation = data_augmentation

    def unnormalize_image(self, rgb, depth, type):
        if type == 'viewpoint':
            mean = self.mean[:4]
            std = self.std[:4]
        else:
            mean = self.mean[4:]
            std = self.std[4:]

        rgb *= std[:3, np.newaxis, np.newaxis]
        rgb += mean[:3, np.newaxis, np.newaxis]
        rgb = rgb.astype(np.uint8)
        depth *= std[3, np.newaxis, np.newaxis]
        depth += mean[3, np.newaxis, np.newaxis]
        depth = depth.astype(np.uint16)
        return rgb.T, depth.T

    def get_image_pair(self, index):
        frame, pose = self.data_pose[index]
        frame_pair, pose_pair = self.data_pair[index][0]
        rgb, depth = frame.get_rgb_depth(self.path)
        rgb_pair, depth_pair = frame_pair.get_rgb_depth(self.path)
        return rgb, depth, pose, rgb_pair, depth_pair

    def load_image(self, index):
        frame, pose = self.data_pose[index]
        rgb, depth = frame.get_rgb_depth(self.path)
        return rgb, depth, pose

    def load_pair(self, index, pair_id):
        frame, pose = self.data_pair[index][pair_id]
        rgb, depth = frame.get_rgb_depth(self.path)
        return rgb, depth, pose

    def get_sample(self, index, image_buffer, prior_buffer, label_buffer, buffer_index):
        rgbA, depthA, poseA = self.load_image(index)
        rgbB, depthB, poseB = self.load_pair(index, 0)
        if self.data_augmentation is not None:
            rgbA, depthA = self.data_augmentation.augment(rgbA, depthA, poseA, real=False)
            rgbB, depthB = self.data_augmentation.augment(rgbB, depthB, poseB, real=True)
        # normalize data
        #image_buffer[buffer_index] = None
        prior_buffer[buffer_index] = poseA.to_parameters(isQuaternion=True)
        label_buffer[buffer_index] = poseB.to_parameters()

    def get_permutations(self, minibatch_size):
        permutations = np.random.permutation(self.get_valid_index())
        return [permutations[x:x + minibatch_size] for x in range(0, len(permutations), minibatch_size)]

    """
        PARALLEL MINIBATCH METHODS
    """
    def compute_minibatches_permutations_(self):
        permutations = np.random.permutation(self.get_valid_index())
        return [permutations[x:x + self.minibatch_size] for x in range(0, len(permutations), self.minibatch_size)]

    def load_minibatch(self, task):
        image_buffer = np.ndarray((len(task), 8, int(self.metadata["image_size"]), int(self.metadata["image_size"])), dtype=np.float32)
        prior_buffer = np.ndarray((len(task), 7), dtype=np.float32)
        label_buffer = np.ndarray((len(task), 6), dtype=np.float32)
        for buffer_index, permutation in enumerate(task):
            self.get_sample(permutation, image_buffer, prior_buffer, label_buffer, buffer_index)
        return image_buffer, prior_buffer, label_buffer
