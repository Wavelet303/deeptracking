from deeptracking.data.dataset import Dataset
from tqdm import tqdm
import os


if __name__ == '__main__':
    datasets_path = ["/home/mathieu/Dataset/DeepTrack/dragon/valid_real",
                     "/home/mathieu/Dataset/DeepTrack/dragon/valid_real"]
    output_path = "/home/mathieu/Dataset/DeepTrack/dragon/valid_mixed"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print("Merging datasets :")
    for path in datasets_path:
        print(path)
    print("Into : {}".format(output_path))

    # load datasets
    datasets = [Dataset(x) for x in datasets_path]
    for dataset in datasets:
        dataset.load()

    # metadata sanity check
    for dataset_check in datasets:
        for other_dataset in datasets:
            if dataset_check.metadata != other_dataset.metadata:
                raise RuntimeError("Dataset {} have different metadata than {}".format(dataset_check.path,
                                                                                       other_dataset.path))

    metadata = datasets[0].metadata
    camera = datasets[0].camera
    output_dataset = Dataset(output_path, frame_class=metadata["save_type"])
    output_dataset.camera = camera
    output_dataset.metadata = metadata

    # transfer data
    for dataset in datasets:
        print("Process dataset {}".format(dataset.path))
        for i in tqdm(range(dataset.size())):
            pair_id = 0
            rgbA, depthA, initial_pose = dataset.load_image(i)
            rgbB, depthB, transformed_pose = dataset.load_pair(i, pair_id)

            output_dataset.add_pose(rgbA, depthA, initial_pose)
            output_dataset.add_pair(rgbB, depthB, transformed_pose, pair_id)

            if i % 500 == 0:
                output_dataset.dump_images_on_disk()
            if i % 5000 == 0:
                output_dataset.save_json_files(metadata)

    output_dataset.dump_images_on_disk()
    output_dataset.save_json_files(metadata)