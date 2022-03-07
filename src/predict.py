import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import torch

from definitions import DATA_DIR
from src.model.unet import UNET
from src.utils.utils2 import get_dataloaders

def accuracy_metric(inp, targ):
    mask = targ == 1

    accuracy = (inp[mask] == targ[mask]).mean()
    return accuracy

def plot_results(x, y):
    out_data = model(x)

    # plotting
    fig, ax = plt.subplots(nrows=3, ncols=4)
    ax[0, 0].imshow(x[0, 0, :, :])
    ax[0, 0].set_title("R")
    ax[0, 1].imshow(x[0, 1, :, :])
    ax[0, 1].set_title("G")
    ax[0, 2].imshow(x[0, 2, :, :])
    ax[0, 2].set_title("B")
    ax[0, 3].imshow(x[0, 3, :, :])
    ax[0, 3].set_title("NIR")

    model_output = out_data.detach().numpy()[0, 0, :, :]
    ax[1, 0].imshow(model_output)
    ax[1, 0].set_title("Model output")

    true_mask = y.detach().numpy()[0, 0, :, :]
    ax[1, 1].imshow(true_mask)
    ax[1, 1].set_title("True Mask")

    super_threshold_indices = model_output > 0.0
    model_output[super_threshold_indices] = 1
    model_output[~super_threshold_indices] = 0

    ax[1, 2].imshow(model_output)
    ax[1, 2].set_title("Thresholded model output")

    ax[2, 0].hist(model_output.flatten(), bins=100)
    ax[2, 1].hist(true_mask.flatten(),  bins=100)
    ax[2, 2].hist(model_output.flatten(), bins=100)

    accuracy = accuracy_metric(inp=model_output, targ=true_mask)
    fig.suptitle("Accuracy: {}".format(accuracy), fontsize=16)

    plt.show()


if __name__ == '__main__':
    # load data
    train_dl, valid_dl = get_dataloaders(
        path_to_tiled_img_dir=os.path.join(DATA_DIR, "tiled", "images"),
        path_to_tiled_label_dir=os.path.join(DATA_DIR, "tiled", "labels"),
        batch_size=1,
        split=(1700, 64)

    )

    # prepare model
    path_to_checkpoints_dir = Path(DATA_DIR) / "checkpoints"
    path_to_model_checkpoint = path_to_checkpoints_dir / "model.pt"
    checkpoint = torch.load(path_to_model_checkpoint)
    model = UNET(n_channels=4, n_classes=1, bilinear=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    # loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()

    batches = list(train_dl)
    batch_no = 319
    x, y = batches[batch_no]
    plot_results(x, y)
