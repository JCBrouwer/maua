import os.path, math, time, itertools, tqdm
import torch as th
import torchvision.transforms as tn
import torch.utils.data
from PIL import Image
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
        super(ProGANDataLoader, self).__init__(**kwargs)
        self.prescaled_data = prescaled_data
        self.prescaled_data_path = prescaled_data_path

    
    def get_batch_sizes(self, model):
        print("Calculating maximum batch sizes...")
        batch_sizes = dict()
        batch_size = 512
        for depth, image_size in enumerate(map(lambda x: 2**(x+3), range(model.depth-1))):
            too_big = True
            while too_big:
                if batch_size < 1:
                    batch_sizes.setdefault(image_size, 0)
                    return batch_sizes
                noise = th.randn(round(batch_size / 2.)*2, model.latent_size).to(model.device)
                images = th.randn(round(batch_size / 2.)*2, 3, image_size, image_size).to(model.device)
                try:
                    model.optimize_D(noise, images, depth+1, 1.0)
                    model.optimize_G(noise, images, depth+1, 1.0)
                    batch_sizes[image_size] = round(batch_size / math.sqrt(2) / 2.)*2
                    too_big = False
                except RuntimeError:
                    import gc
                    gc.collect()
                    th.cuda.empty_cache()
                    batch_size /= math.sqrt(math.sqrt(2))
        print(batch_sizes)
        return batch_sizes


    def set_batch_size(self, current_res, new_val):
        if self.prescaled_data:
            self.dataset = ImageFolder(data_path=self.data_path+"/%s"%current_res, transform=tn.ToTensor())
        self.dataloader = th.utils.data.DataLoader(self.dataset, batch_size=new_val, shuffle=True, num_workers=3)


    def generate_prescaled_dataset(self, sizes):
        if not self.prescaled_data: return
        print("Generating prescaled dataset...")
        data_path = self.prescaled_data_path
        if data_path is None: data_path = 'maua/datasets/%s_prescaled'%self.data_path.split('/')[-1]
        if not os.path.isdir(data_path) or \
           not len(self.dataloader)*len(sizes) == len(ProGANDataLoader(data_path=data_path)):
            # create a copy of the dataset on disk for each size
            from pathos.multiprocessing import ProcessingPool
            pool = ProcessingPool()

            def prescale_dataset(tup):
                image_file, size = tup
                try:
                    Image.open(data_path+"/%s/%s"%(size,image_file.split("/")[-1]))
                    return 1
                except:
                    os.makedirs(data_path+"/%s"%size, exist_ok=True)
                    image = Image.open(self.data_path+"/"+image_file)
                    transforms = tn.Compose([self.transforms, tn.Resize(size), tn.ToTensor()])
                    processed = th.clamp(transforms(image), min=0, max=1)
                    save_image(processed, data_path+"/%s/%s"%(size,image_file.split("/")[-1]))
                    return 1

            jobs = list(itertools.product(filter(lambda im: not im.startswith("."), os.listdir(self.data_path)), sizes))
            results = pool.amap(prescale_dataset, jobs)
            time.sleep(1)
            pbar = tqdm.tqdm(total=len(self.dataloader)*len(sizes))
            pbar.set_description("Images processed")
            while not results.ready():
                num_files = sum([len(os.listdir(data_path+"/%s"%size)) for size in sizes])
                pbar.update(num_files - pbar.n)
                time.sleep(1)
            pbar.close()
            pool.close()
            pool.join()
            assert sum(results.get()) == len(self.dataloader)*len(sizes)
        else:
            print("Dataset already generated.")
        self.data_path = data_path