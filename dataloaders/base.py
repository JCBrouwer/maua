import torch as th
import torchvision as tv
from .image_folder import ImageFolder

class BaseDataLoader():
    def __init__(self, data_path, transforms, batch_size=8):
        """
        constructor for the class BaseDataLoader
        :param data_path: path to data
        :param transforms: transforms to preprocess with
        :param batch_size: number of images to return at a time
        """
        self.data_path  = data_path
        self.transforms = transforms
        self.batch_size = batch_size
        self.dataset    = ImageFolder(data_path=data_path, transform=transforms)
        self.dataloader = th.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    def batches(self):
        return len(iter(self.dataloader))

    def set_batch_size(self, new_val):
        self.dataloader = th.utils.data.DataLoader(self.dataset, batch_size=new_val, shuffle=True, num_workers=3)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataloader)
