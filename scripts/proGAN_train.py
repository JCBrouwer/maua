import torch as th
import torchvision.transforms as tn
from torchvision.utils import save_image
from ..models import ProGAN
from ..dataloaders import BaseDataLoader

depth = 8
model = ProGAN(
    name = 'flowerGAN',
    save_dir = 'maua/modelzoo/flower_progan',
    depth = depth,
    latent_size = 128,
    gpu = 0,
    seed = 27
)

dataloader = BaseDataLoader(
    data_path = 'maua/datasets/flower_progan',
    transforms = tn.Compose([tn.Resize(2**depth),
                             tn.RandomHorizontalFlip(),
                             tn.RandomVerticalFlip(),
                             tn.RandomChoice([
                                tn.RandomRotation([0,0]),
                                tn.RandomRotation([90,90]),
                                tn.RandomRotation([270,270])]),
                             tn.ToTensor()])
)

model.train(
    dataloader = dataloader,
    fade_in = 0.75,
    save_freq = 25,
    log_freq = 5,
    epochs_dict = {8: 50, 16: 50, 32: 50, 64: 50, 128: 75, 256: 75},
    batches_dict = {8: 512, 16: 128, 32: 48, 64: 24, 128: 12, 256: 6}
)

result = model(th.randn(1, 128)*3) # messing with the latent vector has a big effect on output image
save_image(result, 'maua/output/progan_flower1.png')

result = model(th.randn(1, 128)*2+0.5)
save_image(result, 'maua/output/progan_flower2.png')

result = model(th.randn(1, 128)**2)
save_image(result, 'maua/output/progan_flower3.png')

result = model(th.randn(1, 128))
save_image(result, 'maua/output/progan_flower4.png')

result = model(th.cos(th.randn(1, 128)))
save_image(result, 'maua/output/progan_flower5.png')

result = model(th.randn(1, 128)/2+1)
save_image(result, 'maua/output/progan_flower6.png')