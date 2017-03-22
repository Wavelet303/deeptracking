import numpngw
import os
import numpy as np
from PIL import Image


class Frame:
    def __init__(self, rgb, depth, id):
        self.rgb = rgb
        self.depth = depth
        self.id = id

    def is_on_disk(self):
        return self.rgb is None and self.depth is None

    def dump(self, path):
        if not self.is_on_disk():
            numpngw.write_png(os.path.join(path, '{}.png').format(self.id), self.rgb)
            numpngw.write_png(os.path.join(path, '{}d.png').format(self.id), self.depth.astype(np.uint16))
            self.rgb = None
            self.depth = None

    def load(self, path):
        self.rgb = np.array(Image.open(os.path.join(path, self.id + ".png")))
        self.depth = np.array(Image.open(os.path.join(path, self.id + "d.png"))).astype(np.uint16)


class FrameNumpy:
    def __init__(self, rgb, depth, id):
        depth8 = self.numpy_int16_to_uint8(depth)
        self.frame = np.concatenate((rgb, depth8), axis=2)
        self.id = id

    def is_on_disk(self):
        return self.frame is None

    def dump(self, path):
        if not self.is_on_disk():
            np.save(os.path.join(path, self.id), self.frame)
            self.frame = None

    def load(self, path):
        self.frame = np.load(os.path.join(path, self.id))

    @staticmethod
    def numpy_int16_to_uint8(depth):
        x, y = depth.shape
        out = np.ndarray((x, y, 2), dtype=np.uint8)
        out[:, :, 0] = np.right_shift(depth, 8)
        out[:, :, 1] = depth.astype(np.uint8)
        return out

    @staticmethod
    def numpy_uint8_to_int16(depth8):
        x, y, c = depth8.shape
        out = np.ndarray((x, y), dtype=np.int16)
        out[:, :] = depth8[:, :, 0]
        out = np.left_shift(out, 8)
        out[:, :] += depth8[:, :, 1]
        return out