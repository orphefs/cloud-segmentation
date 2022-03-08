from definitions import DATA_DIR
from src.model.unet import UNET
from src.utils.utils2 import get_dataloaders
import os

if __name__ == '__main__':
    # TODO: change to pytest format

    # not a formal pytest, but does the job for now
    unet = UNET(4, 2)
    train_dl, valid_dl = get_dataloaders(
        path_to_tiled_img_dir=os.path.join(DATA_DIR, "tiled", "images"),
        path_to_tiled_label_dir=os.path.join(DATA_DIR, "tiled", "images"),
        batch_size=4,
        split=(0.8, 0.2)
    )
    # test one pass
    xb, yb = next(iter(train_dl))
    print(xb.shape, yb.shape)
    pred = unet(xb)
    print(pred.shape)
