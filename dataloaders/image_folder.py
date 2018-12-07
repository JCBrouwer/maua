###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for data_path, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(data_path, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    def __init__(self, data_path, transform=None, return_paths=False, loader=default_loader):
        """
        constructor for the ImageFolder dataset class
        :param data_path: path to folder containing images
        :param transform: transforms to preprocess images with
        :param return_paths: whether to return paths along with image
        :param loader: image loader
        """
        
        imgs = make_dataset(data_path)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + data_path + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.data_path = data_path
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
