from os import path
import numpy as np

from deeptracking.utils.camera import Camera


class Dataset:
    def __init__(self, folder_path, normalize=True, normalize_param_folder=""):
        self.path = folder_path
        self.header = Dataset.load_viewpoint_header(self.path)
        self.camera = Camera.load_from_json(self.path)
        self.viewpoint_size, self.pair_size, self.total_size = Dataset.extract_viewpoint_sizes(self.header)
        self.normalize = normalize
        if self.normalize:
            try:
                self.mean = np.load(path.join(normalize_param_folder, "mean.npy"))
                self.std = np.load(path.join(normalize_param_folder, "std.npy"))
            except Exception:
                raise IOError("Folder {} does not contain mean.npy and std.npy".format(normalize_param_folder))

    def get_labels(self):
        return

    def get_priors(self):
        return

    def load_input(self, dataset_index, tensor, tensor_index):
        return

    def size(self):
        return self.total_size

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
