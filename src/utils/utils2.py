import os
import numpy as np
import rasterio
from fastai.vision.data import ImageDataLoaders, SegmentationDataLoaders
from typing import Collection, List, Tuple

from PIL import Image
from fastai.data.transforms import get_files, get_image_files
from fastai.vision import *
from fastai.data import *
import pathlib

from collections import namedtuple

from torchvision.models.detection.image_list import ImageList

ImageTile = namedtuple('ImageTile', 'path idx rows cols')

from matplotlib import pyplot as plt
import os
import rasterio
import numpy as np


def _ensure_opened(ds):
    "Ensure that `ds` is an opened Rasterio dataset and not a str/pathlike object."
    return ds if type(ds) == rasterio.io.DatasetReader else rasterio.open(str(ds), "r")


def read_crop(ds, crop, bands=None, pad=False):
    """
    Read rasterio `crop` for the given `bands`..
    Args:
        ds: Rasterio dataset.
        crop: Tuple or list containing the area to be cropped (px, py, w, h).
        bands: List of `bands` to read from the dataset.
    Returns:
        A numpy array containing the read image `crop` (bands * h * w).
    """
    ds = _ensure_opened(ds)
    if pad: raise ValueError('padding not implemented yet.')
    if bands is None:
        bands = [i for i in range(1, ds.count + 1)]

    # assert len(bands) <= ds.count, "`bands` cannot contain more bands than the number of bands in the dataset."
    # assert max(bands) <= ds.count, "The maximum value in `bands` should be smaller or equal to the band count."
    window = None
    if crop is not None:
        assert len(crop) == 4, "`crop` should be a tuple or list of shape (px, py, w, h)."
        px, py, w, h = crop
        w = ds.width - px if (px + w) > ds.width else w
        h = ds.height - py if (py + h) > ds.height else h
        assert (px + w) <= ds.width, "The crop (px + w) is larger than the dataset width."
        assert (py + h) <= ds.height, "The crop (py + h) is larger than the dataset height."
        window = rasterio.windows.Window(px, py, w, h)
    meta = ds.meta
    meta.update(count=len(bands))
    if crop is not None:
        meta.update({
            'height': window.height,  # make the aoi more smooth so data is easier correctly downloaded
            'width': window.width,
            'transform': rasterio.windows.transform(window, ds.transform)})
    return ds.read(bands, window=window), meta


def plot_rgb(img, clip_percentile=(2, 98), clip_values=None, bands=[3, 2, 1], figsize=(20, 20), nodata=None,
             figtitle=None, crop=None, ax=None):
    """
    Plot clipped (and optionally cropped) RGB image.
    Args:
        img: Path to image, rasterio dataset or numpy array of shape (bands, height, width).
        clip_percentile: (min percentile, max percentile) to use for clippping.
        clip_values: (min value, max value) to use for clipping (if set clip_percentile is ignored).
        bands: Bands to use as RGB values (starting at 1).
        figsize: Size of the matplotlib figure.
        figtitle: Title to use for the figure (if None and img is a path we will use the image filename).
        crop: Window to use to crop the image (px, py, w, h).
        ax: If not None, use this Matplotlib axis for plotting.
    Returns:
        A matplotlib figure.
    """
    meta = None
    if isinstance(img, str):
        assert os.path.exists(img), "{} does not exist!".format(img)
        figtitle = os.path.basename(img) if figtitle is None else figtitle
        img = rasterio.open(img)
        img, meta = read_crop(img, crop, bands)
    elif isinstance(img, rasterio.io.DatasetReader):
        img, meta = read_crop(img, crop, bands)
    elif isinstance(img, np.ndarray):
        assert len(img.shape) <= 3, "Array should have no more than 3 dimensions."
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        elif img.shape[0] > 3:
            img = img[np.array(bands) - 1, :, :]
        if crop is not None:
            img = img[:, py:py + h, px:px + w]
    else:
        raise ValueError("img should be str, rasterio dataset or numpy array. (got {})".format(type(img)))
    img = img.astype(float)
    nodata = nodata if nodata is not None else (meta['nodata'] if meta is not None else None)
    if nodata is not None:
        img[img == nodata] = np.nan
    if clip_values is not None:
        assert len(clip_values) == 2, "Clip values should have the shape (min value, max value)"
        assert clip_values[0] < clip_values[1], "clip_values[0] should be smaller than clip_values[1]"
    elif clip_percentile is not None:
        assert len(
            clip_percentile) == 2, "Clip_percentile should have the shape (min percentile, max percentile)"
        assert clip_percentile[0] < clip_percentile[
            1], "clip_percentile[0] should be smaller than clip_percentile[1]"
        clip_values = None if clip_percentile == (0, 100) else [np.nanpercentile(img, clip_percentile[i]) for
                                                                i in range(2)]
    if clip_values is not None:
        img[~np.isnan(img)] = np.clip(img[~np.isnan(img)], *clip_values)
    clip_values = (np.nanmin(img), np.nanmax(img)) if clip_values is None else clip_values
    img[~np.isnan(img)] = (img[~np.isnan(img)] - clip_values[0]) / (clip_values[1] - clip_values[0])
    if img.shape[0] <= 3:
        img = np.transpose(img, (1, 2, 0))
    alpha = np.all(~np.isnan(img), axis=2)[:, :, np.newaxis].astype(float)
    img = np.concatenate((img, alpha), axis=2)

    if not ax:
        figure, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(figtitle) if figtitle is not None else None
        ax.imshow(img)
        plt.close()
        return figure
    else:
        ax.imshow(img)


def get_labels_tiles(fn):
    path, *tile = fn
    path = path_lbl / path.name
    return ImageTile(path, *tile)


def get_tiles(images: str, rows: int, cols: int) -> List[ImageTile]:
    images_tiles = []
    for img in images:
        for i in range(rows * cols):
            images_tiles.append(ImageTile(img, i, rows, cols))
    return images_tiles


def open_image_tile(img_t: ImageTile, mask=False, **kwargs) -> Image:
    """given and ImageTile it returns and Image with the tile,
    set mask to True for masks"""
    path, idx, rows, cols = img_t
    img = open_image(path, **kwargs) if not mask else open_mask(path, **kwargs)
    row = idx // cols
    col = idx % cols
    tile_x = img.size[0] // cols
    tile_y = img.size[1] // rows
    return Image(img.data[:, col * tile_x:(col + 1) * tile_x, row * tile_y:(row + 1) * tile_y])


class Bounds:
    def __init__(self,
                 x_min: int,
                 y_min: int,
                 x_max: int,
                 y_max: int,
                 path_to_parent_image: pathlib.PosixPath,
                 ):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.path_to_parent_image = path_to_parent_image


class ImageTiler:
    def __init__(self, image_tiles: List[ImageTile]):
        self.image_tiles = image_tiles

    def _compute_bounds(self, image_shape: Tuple[int, int], tile_shape: Tuple[int, int], path_to_image: str):
        image_y = image_shape[0]
        image_x = image_shape[1]
        tile_y = tile_shape[0]
        tile_x = tile_shape[1]

        rows = image_y // tile_y
        cols = image_x // tile_x
        bounds = []
        for i in range(rows):
            for j in range(cols):
                bounds.append(Bounds(
                    x_min=j * tile_shape[1],
                    y_min=i * tile_shape[0],
                    x_max=(j + 1) * tile_shape[1],
                    y_max=(i + 1) * tile_shape[0],
                    path_to_parent_image=path_to_image
                ))

        return bounds

    def extract_tile(self, path_to_image: str, tile_shape: Tuple[int, int]):
        with rasterio.open(path_to_image, "r") as infile:
            band_r = infile.read(1)
            band_g = infile.read(2)
            band_b = infile.read(3)
            # band_nir = infile.read(8)

        list_of_bounds = self._compute_bounds(band_r.shape, tile_shape, path_to_image)

        for index, bounds in enumerate(list_of_bounds):
            out_r = band_r[bounds.y_min:bounds.y_max, bounds.x_min:bounds.x_max]
            out_g = band_g[bounds.y_min:bounds.y_max, bounds.x_min:bounds.x_max]
            out_b = band_b[bounds.y_min:bounds.y_max, bounds.x_min:bounds.x_max]
            # out_nir = band_nir[bounds.y_min:bounds.y_max, bounds.x_min:bounds.x_max]

            im = Image.fromarray(out_r)
            im.save(os.path.splitext(path_to_image)[0] + "_r_tile_{}.png".format(index),
                format="png")
            im = Image.fromarray(out_g)

            im.save(os.path.splitext(path_to_image)[0] + "_g_tile_{}.png".format(index),
                format="png")
            im = Image.fromarray(out_b)

            im.save(os.path.splitext(path_to_image)[0] + "_b_tile_{}.png".format(index),
                format="png")


class SegmentationTileItemList:

    def __init__(self, paths_to_tiled_images: List[pathlib.PosixPath]):
        self.paths_to_tiled_images = paths_to_tiled_images

    def open(self, fn: ImageTile) -> Image:
        return open_image_tile(fn, convert_mode=self.convert_mode, after_open=self.after_open)

    @classmethod
    def from_folder(cls, path: str = '.', rows=1, cols=1, extensions: Collection[str] = None,
                    **kwargs):
        """patchs the from_folder method, generating list of ImageTile
        with all the possible tiles for all the images in folder"""
        files = get_image_files(path)
        paths_to_tiled_images = get_tiles(files, rows, cols)
        return paths_to_tiled_images


def label_func(fn): return path / "labels" / f"{fn.stem}_P{fn.suffix}"


if __name__ == '__main__':
    # get the images to be tiled
    list_of_tiles = SegmentationTileItemList.from_folder(path="/tmp/overstory/images", rows=4, cols=4)
    image_tiler = ImageTiler(list_of_tiles)
    image_tiler.extract_tile(path_to_image=list_of_tiles[0].path, tile_shape=(1024, 1024))
    #
    # path = pathlib.Path("/tmp/overstory/")
    # fnames = get_image_files(path / "images")
    # codes = np.loadtxt(path / 'codes.txt', dtype=str)
    # dls = SegmentationDataLoaders.from_label_func(
    #     path, bs=8, fnames=fnames, label_func=label_func, codes=codes
    # )
    # dls.show_batch(max_n=6)
