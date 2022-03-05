from src.model.unet import UNET
from src.utils.utils2 import get_dataloaders

if __name__ == '__main__':
    # TODO: change to pytest format

    # not a formal pytest, but does the job for now
    unet = UNET(4, 2)
    train_dl, valid_dl = get_dataloaders(
        path_to_tiled_img_dir="/tmp/overstory/tiled/images",
        path_to_tiled_label_dir="/tmp/overstory/tiled/labels",
        batch_size=2,
        split=(80, 20)
    )
    # test one pass
    xb, yb = next(iter(train_dl))
    print(xb.shape, yb.shape)
    pred = unet(xb)
    print(pred.shape)
