import numpy as np
import math
from deeptracking.utils.transform import Transform
from scipy.misc import imresize

try:
    import cv2
    import matplotlib.pyplot as plt
except ImportError:
    pass


def noise_image(img, gaussian_std=6, sp_proba=0.01, sp_strenght=50):
    type = img.dtype
    if len(img.shape) == 2:
        mask = img[:, :] != 0
    else:
        mask = img[:, :, 0] != 0
        mask = mask[:, :, np.newaxis]
    gaussian_noise = np.random.normal(0, gaussian_std, img.shape)
    salt_pepper = np.random.uniform(0, 1, img.shape)
    gaussian_noise[salt_pepper > 1 - sp_proba] = sp_strenght
    gaussian_noise[salt_pepper < sp_proba] = -sp_strenght
    return ((gaussian_noise + img) * mask).astype(type)


def load_pose(header, id):
    return Transform.from_parameters(float(header[id]["vector"]["0"]),
                                     float(header[id]["vector"]["1"]),
                                     float(header[id]["vector"]["2"]),
                                     float(header[id]["vector"]["3"]),
                                     float(header[id]["vector"]["4"]),
                                     float(header[id]["vector"]["5"]))


def extract_viewpoint_sizes(header):
    viewpoint_size = int(header["metaData"]["frameQty"])
    pair_size = int(header["metaData"]["PairsQty"])
    total_size = viewpoint_size * pair_size
    if total_size == 0:
        total_size = viewpoint_size
    return viewpoint_size, pair_size, total_size


def crop_viewpoint(viewpoint):
    viewpoint.frame.color = crop_image(viewpoint.frame.color)
    viewpoint.frame.depth = crop_image(viewpoint.frame.depth)


def crop_image(image, crop_center, size=100):
    y = int(crop_center[1] - size / 2)
    x = int(crop_center[0] - size / 2)
    image = image[y: y + size, x: x + size]
    return image


def normalize_label(params, max_translation, max_rotation_rad):
    params[0] /= max_translation
    params[1] /= max_translation
    params[2] /= max_translation
    params[3] /= max_rotation_rad
    params[4] /= max_rotation_rad
    params[5] /= max_rotation_rad
    return params


def unnormalize_label(params, max_translation, max_rotation_rad):
    params[:, :3] *= max_translation
    params[:, 3:] *= math.degrees(max_rotation_rad)
    return params


def load_prior(header, prior_tensor, isQuaternion=True, inverse=False):
    viewpoint_size, pair_size, total_size = extract_viewpoint_sizes(header)
    for i in range(total_size):
        if pair_size:
            id = str(int(i / pair_size))
        else:
            id = str(i)
        pose = load_pose(header, id)
        if inverse:
            pose = pose.inverse()
        prior_tensor[i, :] = np.array(pose.to_parameters(isQuaternion=isQuaternion))


def load_label(header, label_tensor):
    viewpoint_size, pair_size, total_size = extract_viewpoint_sizes(header)
    for i in range(viewpoint_size):
        for j in range(pair_size):
            id = str(i) + "n" + str(j)
            pose = load_pose(header, str(id))
            label = np.array(pose.to_parameters(isQuaternion=False))
            normalize_label(label,
                            float(header["metaData"]["translation_range"]),
                            float(header["metaData"]["rotation_range"]))
            label_tensor[i * pair_size + j, :] = label


def combine_view_transform(vp, view_transform):
    """
    combines a camera space transform with a camera axis dependent transform.
    Whats important here is that view transform's translation represent the displacement from
    each axis, and rotation from each axis. The rotation is applied around the translation point of view_transform.
    :param vp:
    :param view_transform:
    :return:
    """
    camera_pose = vp.copy()
    R = camera_pose.rotation
    T = camera_pose.translation
    rand_R = view_transform.rotation
    rand_T = view_transform.translation

    rand_R.combine(R)
    T.combine(rand_R)
    rand_T.combine(T)
    return rand_T


def normalize_scale(color, depth, pose, camera, output_size=(100, 100), scale_size=230):
    pose = pose.inverse()
    pixels = rect_from_pose(pose, camera, scale_size)

    # pad zeros if the crop happens outside of original image
    lower_x = 0
    lower_y = 0
    higher_x = 0
    higher_y = 0
    if pixels[0, 0] < 0:
        lower_x = -pixels[0, 0]
        pixels[:, 0] += lower_x
    if pixels[0, 1] < 0:
        lower_y = -pixels[0, 1]
        pixels[:, 1] += lower_y
    if pixels[1, 0] > camera.width:
        higher_x = pixels[1, 0] - camera.width
    if pixels[1, 1] > camera.height:
        higher_y = pixels[1, 1] - camera.height

    color = np.pad(color, ((lower_y, higher_y), (lower_x, higher_x), (0, 0)), mode="constant", constant_values=0)
    depth = np.pad(depth, ((lower_y, higher_y), (lower_x, higher_x)), mode="constant", constant_values=0)
    color_crop = color[pixels[0, 0]:pixels[1, 0], pixels[0, 1]:pixels[2, 1], :]
    depth_crop = depth[pixels[0, 0]:pixels[1, 0], pixels[0, 1]:pixels[2, 1]].astype(np.float)
    mask_depth = imresize(depth_crop, output_size, interp='nearest', mode="F") != 0
    mask_rgb = imresize(color_crop, output_size, interp='nearest') != 0
    resized_color_crop = imresize(color_crop, output_size, interp='nearest')
    resized_depth_crop = imresize(depth_crop, output_size, interp='nearest', mode="F").astype(np.int16)
    return resized_color_crop * mask_rgb, resized_depth_crop * mask_depth


def cv_normalize_scale(color, depth, pose, camera, output_size=(100, 100), scale_size=230):
    pose = pose.inverse()
    pixels = rect_from_pose(pose, camera, scale_size)

    # pad zeros if the crop happens outside of original image
    lower_x = 0
    lower_y = 0
    higher_x = 0
    higher_y = 0
    if pixels[0, 0] < 0:
        lower_x = -pixels[0, 0]
        pixels[:, 0] += lower_x
    if pixels[0, 1] < 0:
        lower_y = -pixels[0, 1]
        pixels[:, 1] += lower_y
    if pixels[1, 0] > camera.width:
        higher_x = pixels[1, 0] - camera.width
    if pixels[1, 1] > camera.height:
        higher_y = pixels[1, 1] - camera.height

    color = np.pad(color, ((lower_y, higher_y), (lower_x, higher_x), (0, 0)), mode="constant", constant_values=0)
    depth = np.pad(depth, ((lower_y, higher_y), (lower_x, higher_x)), mode="constant", constant_values=0)
    color_crop = color[pixels[0, 0]:pixels[1, 0], pixels[0, 1]:pixels[2, 1], :]
    depth_crop = depth[pixels[0, 0]:pixels[1, 0], pixels[0, 1]:pixels[2, 1]].astype(np.float)
    mask_depth = cv2.resize(depth_crop, output_size, interpolation=cv2.INTER_NEAREST) != 0
    mask_rgb = cv2.resize(color_crop, output_size, interpolation=cv2.INTER_NEAREST) != 0
    resized_depth_crop = cv2.resize(depth_crop, output_size, interpolation=cv2.INTER_NEAREST)
    resized_color_crop = cv2.resize(color_crop, output_size, interpolation=cv2.INTER_NEAREST)
    return resized_color_crop * mask_rgb, resized_depth_crop * mask_depth


def rect_from_pose(pose, camera, scale_size=230):
    obj_x = pose.matrix[0, 3] * 1000
    obj_y = pose.matrix[1, 3] * 1000
    obj_z = pose.matrix[2, 3] * 1000
    offset = scale_size / 2
    points = np.ndarray((4, 3), dtype=np.float)
    points[0] = [obj_x - offset, -obj_y - offset, -obj_z]
    points[1] = [obj_x - offset, -obj_y + offset, -obj_z]
    points[2] = [obj_x + offset, -obj_y - offset, -obj_z]
    points[3] = [obj_x + offset, -obj_y + offset, -obj_z]
    return camera.project_points(points).astype(np.int32)


def center_pixel(pose, camera):
    obj_x = pose.matrix[0, 3] * 1000
    obj_y = pose.matrix[1, 3] * 1000
    obj_z = pose.matrix[2, 3] * 1000
    point = [obj_x, -obj_y, -obj_z]
    return camera.project_points(np.array([point])).astype(np.uint32)


def image_blend(foreground, background):
    """
    Uses pixel 0 to compute blending mask
    :param foreground:
    :param background:
    :return:
    """
    if len(foreground.shape) == 2:
        mask = foreground[:, :] == 0
    else:
        mask = foreground[:, :, 0] == 0
        mask = mask[:, :, np.newaxis]
    return background * mask + foreground


def normalize_channels(rgb, depth, mean, std):
    """
    Normalize image by negating mean and dividing by std (precomputed)
    :param self:
    :param rgb:
    :param depth:
    :param type:
    :return:
    """
    rgb = rgb.astype(np.float32)
    rgb -= mean[np.newaxis, np.newaxis, :3]
    rgb /= std[np.newaxis, np.newaxis, :3]
    depth = depth.astype(np.float32)
    depth -= mean[np.newaxis, np.newaxis, 3]
    depth /= std[np.newaxis, np.newaxis, 3]
    return rgb.T, depth.T

def unormalize_channels(rgb, depth, mean, std):
    rgb = rgb.T
    depth = depth.T
    rgb *= std[np.newaxis, np.newaxis, :3]
    rgb += mean[np.newaxis, np.newaxis, :3]
    depth *= std[np.newaxis, np.newaxis, 3]
    depth += mean[np.newaxis, np.newaxis, 3]
    return rgb.astype(np.uint8), depth


def show_frames(rgbA, depthA, rgbB, depthB):
    fig, axis = plt.subplots(2, 2)
    ax1, ax2 = axis[0, :]
    ax3, ax4 = axis[1, :]
    ax1.imshow(rgbA)
    ax2.imshow(rgbB)
    ax3.imshow(depthA)
    ax4.imshow(depthB)
    plt.show()


def show_frames_from_buffer(image_buffer, mean, std):
    rgbA = image_buffer[0, 0:3, :, :]
    depthA = image_buffer[0, 3, :, :]
    rgbB = image_buffer[0, 4:7, :, :]
    depthB = image_buffer[0, 7, :, :]
    rgbA, depthA = unormalize_channels(rgbA, depthA, mean[:4], std[:4])
    rgbB, depthB = unormalize_channels(rgbB, depthB, mean[4:], std[4:])
    show_frames(rgbA, depthA, rgbB, depthB)


def normalize_depth(depth, pose):
    depth = depth.astype(np.float32)
    zero_mask = depth == 0
    depth += pose.matrix[2, 3] * 1000
    depth[zero_mask] = 5000
    return depth