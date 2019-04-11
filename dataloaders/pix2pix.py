import os.path
import torch as th
import torch.utils.data
import torchvision.transforms.functional as F
import torchvision.transforms as tn
from .base import BaseDataLoader
from .image_folder import make_dataset
from PIL import Image
import random

class Pix2PixDataLoader(BaseDataLoader):
    def __init__(self, batch_size=16, shuffle=True, num_workers=3, **kwargs):
        """
        constructor for the class Pix2PixDataLoader
        :param batch_size: number of images per batch
        :param shuffle: whether to shuffle order of data from folders
        :param num_workers: number of processes to use
        """
        self.batch_size = batch_size
        self.dataset = Pix2PixDataset(**kwargs)
        self.dataloader = th.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=num_workers)


class Pix2PixDataset():
    def __init__(self, data_path, direction='AtoB', input_nc=3, output_nc=3, serial_batches=True,
                 loadSize=1024, fineSize=512, resize=True, crop=True, vflip=False, hflip=False):
        """
        constructor for the class Pix2PixDataset
        :param data_path: path to folder containing images to train on
        :param direction: AtoB or BtoA
        :param input_nc: number of channels of input images
        :param output_nc: number of channels of output images
        :param serial_batches: if true, takes images in order to make batches, otherwise takes them randomly
        :param loadSize: size to scale loaded images to (can be largest side or tuple) if resize set
        :param fineSize: size to crop scaled images to (can be largest side or tuple) if crop set
        :param resize: whether to resize loaded images
        :param crop: whether to crop scaled images
        :param vflip: whether to flip vertically
        :param hflip: whether to flip horizontally
        """
        super(Pix2PixDataset, self).__init__()
        self.data_path = data_path
        self.serial_batches = serial_batches
        self.direction = direction
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.resize = resize
        self.crop = crop
        self.vflip = vflip
        self.hflip = hflip

        self.dir_A = os.path.join(data_path, 'A')
        self.dir_B = os.path.join(data_path, 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if self.resize:
            A_img = tn.Resize(self.loadSize)(A_img)
            B_img = tn.Resize(self.loadSize)(B_img)

        if self.crop:
            i, j, h, w = tn.RandomCrop.get_params(A_img, output_size=(self.fineSize, self.fineSize))
            A_img = F.crop(A_img, i, j, h, w)
            B_img = F.crop(B_img, i, j, h, w)

        if self.hflip:
            if random.random() > 0.5:
                A_img = F.hflip(A_img)
                B_img = F.hflip(B_img)

        if self.vflip:
            if random.random() > 0.5:
                A_img = F.vflip(A_img)
                B_img = F.vflip(B_img)

        A = tn.ToTensor()(A_img)
        B = tn.ToTensor()(B_img)

        if self.direction == 'BtoA':
            input_nc = self.output_nc
            output_nc = self.input_nc
        else:
            input_nc = self.input_nc
            output_nc = self.output_nc

        # normalize = tn.Normalize(mean=[0.5, 0.5, 0.5],
        #                          std=[0.5, 0.5, 0.5])
        # normalize(A)
        # normalize(B)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'Pix2PixDataset'


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = th.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = th.cat(return_images, 0)
        return return_images
