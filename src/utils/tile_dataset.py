from typing import Tuple
import os

from definitions import DATA_DIR
from src.utils.utils2 import ImageTiler
import glob


def tile_dataset(tile_shape: Tuple[int, int] = (1024, 1024)):
    # tile the images
    path_to_raw_images = os.path.join(DATA_DIR, "raw", "images")
    path_to_tiled_images = os.path.join(DATA_DIR, "tiled", "images")
    paths_to_raw_images = glob.glob(os.path.join(path_to_raw_images, "*.tif"))
    image_tiler = ImageTiler(paths_to_raw_images)
    image_tiler.extract_tile(paths_to_raw_images, path_to_tiled_images, tile_shape=tile_shape,
                             bands=["r", "g", "b", "nir"])
    # tile the labels
    path_to_raw_images = os.path.join(DATA_DIR, "raw", "labels")
    path_to_tiled_images = os.path.join(DATA_DIR, "tiled", "labels")
    paths_to_raw_images = glob.glob(os.path.join(path_to_raw_images, "*.tif"))
    image_tiler = ImageTiler(paths_to_raw_images)
    image_tiler.extract_tile(paths_to_raw_images, path_to_tiled_images, tile_shape=tile_shape,
                             bands=["r", ])


if __name__ == '__main__':
    tile_dataset(tile_shape=(256, 256))
