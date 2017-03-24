import os
import numpy as np
import json

from deeptracking.utils.transform import Transform
from deeptracking.utils.camera import Camera
from deeptracking.data.frame import Frame


class Dataset:
    def __init__(self, folder_path, frame_class=Frame, normalize_param_folder=""):
        self.path = folder_path
        self.data_pose = []
        self.data_pair = {}
        self.metadata = {}
        self.camera = None
        self.frame_class = frame_class
        self.mean = None
        self.std = None
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
        try:
            # Load viewpoints file and camera file
            with open(os.path.join(self.path, "viewpoints.json")) as data_file:
                data = json.load(data_file)
            self.camera = Camera.load_from_json(self.path)
        except FileNotFoundError:
            return -1
        count = 0
        while True:
            try:
                id = str(count)
                pose = Transform.from_parameters(*[float(data[id]["vector"][str(x)]) for x in range(6)])
                self.data_pose.append((Frame(None, None, id), pose))
                if "pairs" in data[id]:
                    for i in range(int(data[id]["pairs"])):
                        pair_id = "{}n{}".format(id, i)
                        self.data_pair[pair_id] = data[pair_id]
                count += 1

            except KeyError:
                return 1

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
