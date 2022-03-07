import os
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Callable
from definitions import DATA_DIR, DEBUG_PRINT, DEBUG_PLOT

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.model.unet import UNET
from src.utils.utils2 import get_dataloaders


def train(model: nn.Module, train_dl: DataLoader, validation_dl: DataLoader,
          loss_fn: CrossEntropyLoss, optimizer: Optimizer, accuracy_fn: Callable,
          path_to_model_checkpoint: str, epochs: int = 1):
    start = time.time()
    if device.type != "cpu":
        model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = validation_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                if device.type != "cpu":
                    x = x.cuda()
                    y = y.cuda()
                step += 1

                # convert to datatype according to loss fn

                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.long)

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)

                    if DEBUG_PRINT:
                        print("x: ", x.shape)
                        print("outputs: ", outputs.shape)
                        print("y: ", y.shape)
                        print("target min", y.min())
                        print("target max", y.max())

                    y = torch.argmax(y, dim=1)
                    loss = loss_fn(outputs, y)

                    if DEBUG_PRINT:
                        print(loss)

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y)

                acc = accuracy_fn(outputs, y)

                running_acc += acc * dataloader.batch_size
                running_loss += loss * dataloader.batch_size

                if step % 10 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc,
                        torch.cuda.memory_allocated() / 1024 / 1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, path_to_model_checkpoint)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss


def accuracy_metric(inp, targ):
    targ = targ.squeeze(1)
    mask = targ == 1

    accuracy = (inp.argmax(dim=1)[mask] == targ[mask]).float().mean()

    if DEBUG_PLOT:
        inp_np = inp[0, 0, :, :].cpu().detach().numpy()
        targ_np = targ[0, :, :].cpu().detach().numpy()
        mask_np = mask[0, :, :].cpu().detach().numpy()
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(inp_np)
        ax[0, 1].imshow(targ_np)
        ax[0, 2].imshow(mask_np)
        ax[1, 0].hist(inp_np.flatten(), bins=100)
        ax[1, 1].hist(targ_np.flatten(), bins=100)
        ax[1, 2].hist(np.uint8(mask_np.flatten()), bins=100)
        fig.suptitle('Accuracy: {}'.format(accuracy), fontsize=16)
        plt.show()

    return accuracy if not accuracy.isnan() else 0.0


if __name__ == '__main__':
    path_to_checkpoints_dir = Path(DATA_DIR) / "checkpoints"
    if not os.path.exists(path_to_checkpoints_dir):
        os.mkdir(path_to_checkpoints_dir)

    unet = UNET(n_channels=4, n_classes=2, bilinear=True)
    # test one pass
    train_dl, valid_dl = get_dataloaders(
        path_to_tiled_img_dir=os.path.join(DATA_DIR, "tiled", "images"),
        path_to_tiled_label_dir=os.path.join(DATA_DIR, "tiled", "labels"),
        batch_size=1,
        split=(180,20),
        normalize_dataset=False,
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(unet.parameters(), lr=0.01)
    train_loss, validation_loss = train(
        model=unet,
        train_dl=train_dl,
        validation_dl=valid_dl,
        loss_fn=loss_function,
        optimizer=optimizer,
        accuracy_fn=accuracy_metric,
        path_to_model_checkpoint=path_to_checkpoints_dir / "model.pt",
        epochs=10,
    )
