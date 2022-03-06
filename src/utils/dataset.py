import pprint
from typing import Optional

import numpy as np
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from pathlib import Path, PosixPath
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose

from definitions import DEBUG_PLOT


class CloudDataset(Dataset):
    def __init__(self, path_to_images_dir: PosixPath, path_to_labels_dir: PosixPath,
                 transform: Optional[Compose],
                 pytorch=True):
        self.transform = transform
        path_to_images_dir = Path(path_to_images_dir)
        path_to_labels_dir = Path(path_to_labels_dir)

        # Loop through the files in folders and combine, into a dictionary, all bands

        paths_to_red = sorted(
            filename for filename in path_to_images_dir.glob("*") if "band=r" in filename.name)
        paths_to_green = sorted(
            filename for filename in path_to_images_dir.glob("*") if "band=g" in filename.name)
        paths_to_blue = sorted(
            filename for filename in path_to_images_dir.glob("*") if "band=b" in filename.name)
        paths_to_nir = sorted(
            filename for filename in path_to_images_dir.glob("*") if "band=nir" in filename.name)
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

    def open_mask(self, idx, add_dims=False):
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        # TODO: check correctness of line below
        clipped_mask = np.where(raw_mask > 0.0, 1, 0)

        # debug plotting
        if DEBUG_PLOT:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(raw_mask)
            ax[1].imshow(clipped_mask)
            plt.show()

        return np.expand_dims(clipped_mask, 0) if add_dims else raw_mask

    def open_as_array(self, idx, invert=False, include_nir=False):
        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                            ], axis=2)
        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)

        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))

        normalized = (raw_rgb - raw_rgb.min()) / (raw_rgb.max() - raw_rgb.min())

        # debug plotting
        if DEBUG_PLOT:
            fig, ax = plt.subplots(nrows=1, ncols=4)
            ax[0].imshow(normalized[0, :, :])
            ax[1].imshow(normalized[1, :, :])
            ax[2].imshow(normalized[2, :, :])
            ax[3].imshow(normalized[3, :, :])
            plt.show()

        return normalized

    def __getitem__(self, idx):
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=True), dtype=torch.torch.int64)
        return x, y

    def open_as_pil(self, idx):
        arr = 256 * self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')

    def __repr__(self):
        return 'Cloud dataset class with {} files'.format(self.__len__())

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    data = CloudDataset(
        path_to_images_dir="/tmp/overstory/tiled/images",
        path_to_labels_dir="/tmp/overstory/tiled/labels",
    )
    print(len(data))
    pprint.pprint(data.files)
    x, y = data[10]
    x.shape

    fig, ax = plt.subplots(1, 2, figsize=(10, 9))
    ax[0].imshow(data.open_as_array(11))
    ax[1].imshow(data.open_mask(11))

    train_ds, valid_ds = torch.utils.data.random_split(data, (80, 20))

    train_dl = DataLoader(train_ds, batch_size=12, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=12, shuffle=True)

    xb, yb = next(iter(train_dl))
    xb.shape, yb.shape

    plt.show()
