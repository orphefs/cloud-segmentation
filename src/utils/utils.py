# from fastai2.torch_basics import *
# from fastai2.test import *
from fastai.data.all import *
from fastai.vision.core import *
# from fastai2.notebook.showdoc import *
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resized_image(fn: Path, sz=128):
    x = PILImage.create(fn).convert('RGB').resize((sz, sz))
    return tensor(array(x)).permute(2, 0, 1).float() / 255.


class TitledImage(tuple):
    def show(self, ctx=None, **kwargs):
        show_titled_image(self, ctx=ctx, **kwargs)


if __name__ == '__main__':
    source = Path('/tmp/overstory/masks')
    items = get_image_files(source)
    split_idx = RandomSplitter()(items)
    img = resized_image(items[0])
    TitledImage((img, 'test')).show()
