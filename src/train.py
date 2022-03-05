import os
import time

import torch
from typing import Callable

from definitions import DATA_DIR

device = torch.device("cuda")

from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.model.unet import UNET
from src.utils.utils2 import get_dataloaders


def train(model: nn.Module, train_dl: DataLoader, validation_dl: DataLoader,
          loss_fn: CrossEntropyLoss, optimizer: Optimizer, accuracy_fn: Callable, epochs: int = 1):
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

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    print("x: ", x.shape)
                    print("outputs: ", outputs.shape)
                    print("y: ", y.shape)
                    loss = loss_fn(outputs, y)
                    print(loss)
                    # TODO: added np.squeeze() to match dimensions
                    # loss = loss_fn(np.squeeze(outputs), np.squeeze(y))

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
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

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss


def accuracy_metric(predb, yb):
    # accuracy metric is defined as the mean value of number of matched pixels between
    # prediction and mask
    if device.type != "cpu":
        return (predb.argmax(dim=1) == yb.cuda()).float().mean()
    else:
        return (predb.argmax(dim=1) == yb).float().mean()


if __name__ == '__main__':
    unet = UNET(4, 2)
    # test one pass
    train_dl, valid_dl = get_dataloaders(
        path_to_tiled_img_dir=os.path.join(DATA_DIR, "tiled", "images"),
        path_to_tiled_label_dir=os.path.join(DATA_DIR, "tiled", "images"),
        batch_size=1,
        split=(80, 20)

    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.01)
    train_loss, validation_loss = train(
        model=unet,
        train_dl=train_dl,
        validation_dl=valid_dl,
        loss_fn=loss_function,
        optimizer=optimizer,
        accuracy_fn=accuracy_metric,
        epochs=1,
    )
