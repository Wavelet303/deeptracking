import os
import numpy as np
import json

from deeptracking.utils.camera import Camera
from deeptracking.data.frame import Frame

class Dataset:
    def __init__(self, folder_path, frame_class=Frame, normalize=True, normalize_param_folder=""):
        self.path = folder_path
        self.data_pose = []
        self.data_pair = {}
        self.metadata = {}
        self.frame_class = frame_class
        #self.header = Dataset.load_viewpoint_header(self.path)
        #self.camera = Camera.load_from_json(self.path)
        #self.viewpoint_size, self.pair_size, self.total_size = Dataset.extract_viewpoint_sizes(self.header)
        #self.normalize = normalize
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

    def dump_on_disk(self):
        viewpoints_data = {}
        for frame, pose in self.data_pose:
            frame.dump(self.path)
            self.insert_pose_in_dict(viewpoints_data, frame.id, pose)
        with open(os.path.join(self.path, "viewpoints.json"), 'w') as outfile:
            json.dump(dict, outfile)

    @staticmethod
    def insert_pose_in_dict(dict, key, item):
        params = {}
        for i, param in enumerate(item.to_parameters()):
            params[str(i)] = str(param)
        dict[key] = {"vector": params}

    def get_labels(self):
        return

    def get_priors(self):
        return

    def load_input(self, dataset_index, tensor, tensor_index):
        return

    def size(self):
        return len(self.data_pose)

    def get_pair_index(self, dataset_index):
        return dataset_index % self.pair_size

    def get_origin_index(self, dataset_index):
        if self.pair_size:
            out = dataset_index / self.pair_size
        else:
            out = dataset_index
        return int(out)

    def extract_viewpoint_poses(self):
        viewpoint_size = int(self.header["metaData"]["frameQty"])
        viewpoint_sphere_pose = []
        for i in range(viewpoint_size):
            id = str(i)
            pose = Dataset.load_pose(self.header, id).inverse()
            viewpoint_sphere_pose.append(pose.to_parameters(isQuaternion=False)[:3])
        return viewpoint_sphere_pose

    def get_permutations(self, minibatch_size):
        permutations = np.random.permutation(self.get_valid_index())
        return [permutations[x:x + minibatch_size] for x in range(0, len(permutations), minibatch_size)]

    def get_valid_index(self):
        return [x for x in range(self.size())]

    def normalize_image(self, rgb, depth, type):
        rgb = rgb.T
        depth = depth.T
        if self.normalize:
            rgb, depth = Dataset.normalize_image_(rgb, depth, type, self.mean, self.std)
        return rgb, depth

    def extract_image_size(self):
        try:  # compatibility with older versions
            size = int(self.header["metaData"]["image_size"])
        except KeyError:
            size = 100
        return size

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