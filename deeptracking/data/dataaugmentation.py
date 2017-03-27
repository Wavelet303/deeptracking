from deeptracking.data.dataset import Dataset

from scipy import ndimage
from skimage.color import rgb2hsv, hsv2rgb
from scipy.misc import imresize
import numpy as np
import scipy.signal
import scipy.stats as st
import random


class DataAugmentation:
    def __init__(self):
        self.occluder = None
        self.background = None
        self.rgb_noise = None
        self.depth_noise = None
        self.blur_kernel = None
        self.jitter = None
        self.hue_noise = None

    def set_background(self, path):
        self.background = Dataset(path)

    def set_occluder(self, path):
        self.occluder = Dataset(path)

    def set_rgb_noise(self, gaussian_std):
        self.rgb_noise = gaussian_std

    def set_depth_noise(self, gaussian_std):
        self.depth_noise = gaussian_std

    def set_hue_noise(self, offset):
        """
        offset is the % of random hue offset distribution
        :param offset:
        :return:
        """
        self.hue_noise = offset

    def set_blur(self, size):
        self.blur_kernel = self.gkern(size)

    def set_jitter(self, max_x, max_y):
        self.jitter = (max_x, max_y)

    def augment(self, rgb, depth, prior, real=False):
        ret_rgb = rgb
        ret_depth = depth

        if real and self.occluder:
            if random.uniform(0, 1) < 0.6:
                id, type = self.occluder.get_random_id()
                occluder_rgb, occluder_depth = self.occluder.load_image(id, type)
                occluder_rgb = self.add_hue_noise(occluder_rgb.T, 1).T
                occluder_rgb = imresize(occluder_rgb, ret_depth.shape, interp='nearest')
                occluder_depth = imresize(occluder_depth, ret_depth.shape, interp='nearest', mode="F").astype(np.int16)
                ret_rgb, ret_depth = self.depth_blend(ret_rgb, ret_depth, occluder_rgb, occluder_depth)

        if real and self.hue_noise:
            if random.uniform(0, 1) > 0.05:
                ret_rgb = self.add_hue_noise(ret_rgb, self.hue_noise)

        if self.jitter:
            self.x_jitter = random.randint(-self.jitter[0], self.jitter[0])
            self.y_jitter = random.randint(-self.jitter[1], self.jitter[1])
            if self.x_jitter > 0:
                ret_rgb = np.pad(ret_rgb, ((self.x_jitter, 0), (0, 0), (0, 0)), mode='constant')[:-self.x_jitter, :, :]
                ret_depth = np.pad(ret_depth, ((self.x_jitter, 0), (0, 0)), mode='constant')[:-self.x_jitter, :]
            else:
                ret_rgb = np.pad(ret_rgb, ((0, abs(self.x_jitter)), (0, 0), (0, 0)), mode='constant')[
                          abs(self.x_jitter):, :, :]
                ret_depth = np.pad(ret_depth, ((0, abs(self.x_jitter)), (0, 0)), mode='constant')[abs(self.x_jitter):,
                            :]
            if self.y_jitter > 0:
                ret_rgb = np.pad(ret_rgb, ((0, 0), (self.y_jitter, 0), (0, 0)), mode='constant')[:, :-self.y_jitter, :]
                ret_depth = np.pad(ret_depth, ((0, 0), (self.y_jitter, 0)), mode='constant')[:, :-self.y_jitter]
            else:
                ret_rgb = np.pad(ret_rgb, ((0, 0), (0, abs(self.y_jitter)), (0, 0)), mode='constant')[:,
                          abs(self.y_jitter):, :]
                ret_depth = np.pad(ret_depth, ((0, 0), (0, abs(self.y_jitter))), mode='constant')[:,
                            abs(self.y_jitter):]

        if real and self.background:
            color_background, depth_background = self.background.load_random_image(ret_rgb.shape[1])
            depth_background = depth_background.astype(np.int32)
            depth_background += int(prior[2] * 1000)
            ret_rgb, ret_depth = self.color_blend(ret_rgb, ret_depth, color_background, depth_background)

        if real and self.rgb_noise:
            if random.uniform(0, 1) > 0.05:
                noise = random.uniform(0, self.rgb_noise)
                ret_rgb = self.add_noise(ret_rgb, noise)
        if real and self.depth_noise:
            if random.uniform(0, 1) > 0.05:
                noise = random.uniform(0, self.depth_noise)
                ret_depth = self.add_noise(ret_depth, noise)

        if real and self.blur_kernel is not None:
            if random.uniform(0, 1) < 0.4:
                ret_rgb[0, :, :] = scipy.signal.convolve2d(ret_rgb[0, :, :], self.blur_kernel, mode='same')
                ret_rgb[1, :, :] = scipy.signal.convolve2d(ret_rgb[1, :, :], self.blur_kernel, mode='same')
                ret_rgb[2, :, :] = scipy.signal.convolve2d(ret_rgb[2, :, :], self.blur_kernel, mode='same')
            if random.uniform(0, 1) < 0.4:
                ret_depth[:, :] = scipy.signal.convolve2d(ret_depth[:, :], self.blur_kernel, mode='same')

        return ret_rgb.astype(np.uint8), ret_depth

    @staticmethod
    def add_noise(img, gaussian_std):
        type = img.dtype
        copy = img.astype(np.float)
        gaussian_noise = np.random.normal(0, gaussian_std, img.shape)
        copy = (gaussian_noise + copy)
        if type == np.uint8:
            copy[copy < 0] = 0
            copy[copy > 255] = 255
        return copy.astype(type)

    @staticmethod
    def add_hue_noise(rgb, hue_offset):
        hsv = rgb2hsv(rgb)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue_offset, hue_offset)) % 1
        rgb = hsv2rgb(hsv) * 255
        return rgb.astype(np.uint8)

    @staticmethod
    def color_blend(rgb1, depth1, rgb2, depth2):
        mask = np.all(rgb1 == 0, axis=2)
        mask = ndimage.binary_dilation(mask).astype(mask.dtype)
        depth1[mask] = 0
        rgb1[mask, :] = 0
        mask = mask.astype(np.uint8)
        new_depth = depth2 * mask + depth1
        new_color = rgb2 * mask[:, :, np.newaxis] + rgb1
        return new_color.astype(np.uint8), new_depth

    @staticmethod
    def depth_blend(rgb1, depth1, rgb2, depth2):

        new_depth2 = depth2.copy()
        new_depth1 = depth1.copy()

        new_depth2[np.all(rgb2 == 0, axis=2)] = -100000
        new_depth1[np.all(rgb1 == 0, axis=2)] = -100000

        mask = (new_depth1 < new_depth2)
        pos_mask = mask.astype(np.uint8)
        neg_mask = (mask == False).astype(np.uint8)

        masked_occluder = rgb1 * pos_mask[:, :, np.newaxis]
        masked_object = rgb2 * neg_mask[:, :, np.newaxis]

        blend_rgb = masked_occluder + masked_object

        masked_occluder = depth1 * pos_mask
        masked_object = depth2 * neg_mask
        blend_depth = masked_occluder + masked_object

        return blend_rgb, blend_depth

    @staticmethod
    def gkern(kernlen=21, nsig=3.5):
        """Returns a 2D Gaussian kernel array."""

        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel