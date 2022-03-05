import glob
import pprint

from torch.utils.data import Dataset, DataLoader, sampler
from pathlib import Path, PosixPath
from PIL import Image
import matplotlib.pyplot as plt
import os


class CloudDataset(Dataset):
    def __init__(self, path_to_images_dir: PosixPath, path_to_labels_dir: PosixPath, pytorch=True):
        super().__init__()
        path_to_images_dir = Path(path_to_images_dir)
        path_to_labels_dir = Path(path_to_labels_dir)

        # Loop through the files in folders and combine, into a dictionary, all bands

        paths_to_red = sorted(filename for filename in path_to_images_dir.glob("*") if "band=r" in filename.name)
        paths_to_green = sorted(filename for filename in path_to_images_dir.glob("*") if "band=g" in filename.name)
        paths_to_blue = sorted(filename for filename in path_to_images_dir.glob("*") if "band=b" in filename.name)
        paths_to_nir = sorted(filename for filename in path_to_images_dir.glob("*") if "band=nir" in filename.name)
        paths_to_ground_truth = sorted(filename for filename in path_to_labels_dir.glob("*"))

        self.files = [
            {
                'red': item[0],
                'green': item[1],
                'blue': item[2],
                'nir': item[3],
                'gt': item[4],
            }
            for item in
            list(zip(
                paths_to_red,
                paths_to_green,
                paths_to_blue,
                paths_to_nir,
                paths_to_ground_truth,
            ))

        ]
        self.pytorch = pytorch

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    data = CloudDataset(
        path_to_images_dir="/tmp/overstory/tiled/images",
        path_to_labels_dir="/tmp/overstory/tiled/labels",
    )
    print(len(data))
    pprint.pprint(data.files)
