from src.model.unet import UNET

if __name__ == '__main__':
    unet = UNET(4,2)

    # test one pass
    xb, yb = next(iter(train_dl))
