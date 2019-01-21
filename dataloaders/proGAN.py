import os.path, math
import torch as th
import torchvision.transforms as tn
import torch.utils.data
from .base import BaseDataLoader
from ..dataloaders import ImageFolder
from .image_folder import make_dataset
from torchvision.utils import save_image

class ProGANDataLoader(BaseDataLoader):
    def __init__(self, prescaled_data=True, prescaled_data_path=None, **kwargs):
        """
        constructor for the class ProGANDataloader
        :param prescaled_data: whether dataset should be pre-resized to each size on disk.
                               trades large performance boost for smaller sizes for more disk space.
        :param prescaled_data_path: path to save prescaled dataset to
        """
        super(ProGANDataLoader).__init__(**kwargs)
        self.prescaled_data = prescaled_data
        self.prescaled_data_path = prescaled_data_path

    
    def get_batch_sizes(self, model):
        batch_sizes = dict()
        batch_size = 512
        for depth, image_size in enumerate(map(lambda x: 2^(x+3), range(model.depth-2))):
            too_big = True
            while too_big:
                noise = th.randn(batch_size, model.latent_size).to(model.device)
                images = th.randn(batch_size, model.latent_size).to(model.device)
                try:
                    model.optimize_D(noise, images, depth+3, 1.0)
                    model.optimize_G(noise, images, depth+3, 1.0)
                    batch_sizes[image_size] = batch_size
                    too_big = False
                except RuntimeError:
                    import gc
                    gc.collect()
                    th.cuda.empty_cache()
                    batch_size /= math.sqrt(2)
        return batch_sizes


    def set_batch_size(self, current_res, new_val):
        if self.prescaled_data:
            self.dataset = ImageFolder(data_path=self.data_path+"/%s"%current_res, transform=tn.ToTensor())
        self.dataloader = th.utils.data.DataLoader(self.dataset, batch_size=new_val, shuffle=True, num_workers=3)


    def generate_prescaled_dataset(self, sizes):
        if not self.prescaled_data: return
        print("Generating prescaled dataset...")
        data_path = 'maua/datasets/%s_prescaled'%self.data_path.split('/')[-1] if self.prescaled_data_path is None else self.prescaled_data_path
        if not os.path.isdir(data_path) or \
           not len(self.dataloader)*len(sizes) == len(ProGANDataLoader(data_path=self.data_path)):
            # create a copy of the dataset on disk for each size
            from pathos.multiprocessing import ProcessingPool as Pool
            pool = Pool(len(sizes))

            def prescale_dataset(size):
                os.makedirs(data_path+"/%s"%size, exist_ok=True)
                transforms = tn.Compose([self.transforms, tn.Resize(size), tn.ToTensor()])
                t_data = BaseDataLoader(self.data_path, transforms=transforms, batch_size=1)
                for i, sample in enumerate(t_data, 1):
                    sample = th.clamp(sample, min=0, max=1)
                    save_image(sample, data_path+"/%s/%s.png"%(size,i))
                return len(os.listdir(data_path+"/%s"%size))

            results = pool.map(prescale_dataset, sizes)
            pool.close()
            pool.join()
            assert sum(results) == len(self.dataloader)*len(sizes)
        self.dataloader.data_path = data_path