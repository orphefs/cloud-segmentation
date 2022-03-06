import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import torch

from definitions import DATA_DIR
from src.model.unet import UNET
from src.utils.utils2 import get_dataloaders

if __name__ == '__main__':
    # load data
    train_dl, valid_dl = get_dataloaders(
        path_to_tiled_img_dir=os.path.join(DATA_DIR, "tiled", "images"),
        path_to_tiled_label_dir=os.path.join(DATA_DIR, "tiled", "labels"),
        batch_size=1,
        split=(80, 20)

    )

    # prepare model
    path_to_checkpoints_dir = Path(DATA_DIR) / "checkpoints"
    path_to_model_checkpoint = path_to_checkpoints_dir / "model.pt"
    checkpoint = torch.load(path_to_model_checkpoint)
    model = UNET(4, 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()

    x, y = list(train_dl)[19]
    out_data = model(x)

    # plotting
    fig, ax = plt.subplots(nrows=3, ncols=4)
    ax[0, 0].imshow(x[0, 0, :, :])
    ax[0, 1].imshow(x[0, 1, :, :])
    ax[0, 2].imshow(x[0, 2, :, :])
    ax[0, 3].imshow(x[0, 3, :, :])
    ax[1, 0].imshow(out_data.detach().numpy()[0, 0, :, :])
    ax[1, 1].imshow(y.detach().numpy()[0, 0, :, :])
    ax[2, 0].hist(out_data.detach().numpy()[0, 0, :, :].flatten())
    ax[2, 1].hist(y.detach().numpy()[0, 0, :, :].flatten())

    plt.show()
