import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader
from typing import List
from sklearn.metrics import confusion_matrix
from definitions import DATA_DIR
from src.model.unet import UNET
from src.utils.utils2 import get_dataloaders

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def accuracy_metric(inp, targ):
    mask = targ == inp
    accuracy = mask.mean()
    return accuracy


def plot_results(x, y, output, pred_probs, threshold_value):
    # Plot inputs (R,G,B,NIR)
    fig, ax = plt.subplots(nrows=3, ncols=4)
    ax[0, 0].imshow(x[0, 0, :, :])
    ax[0, 0].set_title("R")
    ax[0, 1].imshow(x[0, 1, :, :])
    ax[0, 1].set_title("G")
    ax[0, 2].imshow(x[0, 2, :, :])
    ax[0, 2].set_title("B")
    ax[0, 3].imshow(x[0, 3, :, :])
    ax[0, 3].set_title("NIR")

    # Plot true mask and histogram
    true_mask = y.detach().numpy()[0, 0, :, :]
    ax[1, 0].imshow(true_mask)
    ax[1, 0].set_title("True Mask")
    ax[2, 0].hist(true_mask.flatten(), bins=50)

    # Plot raw model output
    output = output.detach().numpy().squeeze()
    ax[1, 1].imshow(output)
    ax[1, 1].set_title("Model output")
    ax[2, 1].hist(output.flatten(), bins=50)

    # Plot probabilities (sigmoid output)
    pprobs = pred_probs.detach().numpy().squeeze()
    ax[1, 2].imshow(pprobs)
    ax[1, 2].set_title("Probabilities (sigmoid out)")
    ax[2, 2].hist(pprobs.flatten(), bins=50)

    # threshold in order to calculate accuracy metric
    pprobs[pprobs >= threshold_value] = 1
    pprobs[pprobs < threshold_value] = 0

    # Plot thresholded output
    ax[1, 3].imshow(pprobs)
    ax[1, 3].set_title("Thresholded at {}".format(threshold_value))
    ax[2, 3].hist(pprobs.flatten(), bins=50)

    # pprobs = 1 - pprobs

    # accuracy = accuracy_metric(inp=pprobs, targ=true_mask)
    accuracy = sklearn.metrics.accuracy_score(y_true=true_mask.flatten(),
        y_pred=pprobs.flatten())

    print(confusion_matrix(true_mask.flatten(), pprobs.flatten()))

    fig.suptitle("Accuracy: {}".format(accuracy), fontsize=16)

    plt.show()


def predict(path_to_model_checkpoint: str, inputs: List[torch.Tensor], true_masks: List[torch.Tensor],
            threshold_value: float):
    model = load_model(path_to_model_checkpoint)
    output = model(inputs)
    pred_probs = torch.sigmoid(output)[0]
    plot_results(inputs, true_masks, output, pred_probs, threshold_value)


def load_model(path_to_model_checkpoint: str):
    checkpoint = torch.load(path_to_model_checkpoint, map_location=torch.device(device))
    model = UNET(n_channels=4, n_classes=1, bilinear=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    return model


def evaluate(path_to_model_checkpoint: str, dataloader: DataLoader, threshold_value: float):
    model = load_model(path_to_model_checkpoint)

    total_confusion_matrix = np.zeros([2,2], dtype=np.uint)
    for batch in dataloader:
        inputs, y = batch
        true_mask = y.detach().numpy()[0, 0, :, :]
        output = model(inputs)
        pred_probs = torch.sigmoid(output)[0]
        pprobs = pred_probs.detach().numpy().squeeze()

        # threshold in order to calculate accuracy metric
        pprobs[pprobs >= threshold_value] = 1
        pprobs[pprobs < threshold_value] = 0
        total_confusion_matrix = total_confusion_matrix+ confusion_matrix(true_mask.flatten(), pprobs.flatten().astype(dtype=np.uint))
    return total_confusion_matrix


if __name__ == '__main__':
    # Example usage
    # load data
    _, valid_dl = get_dataloaders(
        path_to_tiled_img_dir=os.path.join(DATA_DIR, "tiled", "images"),
        path_to_tiled_label_dir=os.path.join(DATA_DIR, "tiled", "labels"),
        batch_size=1,
        split=(3300, 228)
    )

    batches = list(valid_dl)
    # no_batch = random.randrange(0,len(batches))
    no_batch = 30
    print(no_batch)
    x, y = batches[no_batch]

    predict(path_to_model_checkpoint="/home/orphefs/Dropbox/job_applications/overstory/model.pt",
        inputs=x, true_masks=y, threshold_value=0.995)

    total_confusion_matrx = evaluate(path_to_model_checkpoint="/home/orphefs/Dropbox/job_applications/overstory/model.pt",
    dataloader=valid_dl, threshold_value=0.995)
