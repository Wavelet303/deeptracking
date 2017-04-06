"""
    use a pose detector (aruco, checkboard) and compute the pose on the whole dataset
"""

from deeptracking.data.dataset import Dataset
from deeptracking.data.dataset_utils import image_blend
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.utils.camera import Camera
import cv2
import os

from deeptracking.detector.detector_aruco import ArucoDetector

if __name__ == '__main__':
    dataset_path = "/home/mathieu/Dataset/DeepTrack/sequence/skull/1"
    detector_path = "../deeptracking/detector/aruco_layout.xml"
    model_path = "/home/mathieu/Dataset/3D_models/skull/skull.ply"
    model_ao_path = "/home/mathieu/Dataset/3D_models/skull/skull_ao.ply"
    shader_path = "../deeptracking/data/shaders"

    dataset = Dataset(dataset_path)
    camera = Camera.load_from_json(dataset_path)
    files = [f for f in os.listdir(dataset_path) if os.path.splitext(f)[-1] == ".png" and 'd' not in os.path.splitext(f)[0]]
    detector = ArucoDetector(camera, detector_path)
    window = InitOpenGL(camera.width, camera.height)
    vpRender = ModelRenderer(model_path, shader_path, camera, window)
    vpRender.load_ambiant_occlusion_map(model_ao_path)

    for i in range(len(files)):
        img = cv2.imread(os.path.join(dataset.path, "{}.png".format(i)))
        detection = detector.detect(img)
        dataset.add_pose(None, None, detection)
        rgb_render, depth_render = vpRender.render(detection.transpose())
        bgr_render = rgb_render[:, :, ::-1].copy()
        img = image_blend(bgr_render, img)
        
        cv2.imshow("view", img)
        cv2.waitKey(30)
    dataset.save_json_files({"save_type": "png"})
