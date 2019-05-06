import torch as th
import torchvision.transforms as tn
from torchvision.utils import save_image
from ..models import ProGAN
from ..dataloaders.proGAN import ProGANDataLoader

depth = 9
model = ProGAN(
    name = 'flowerGAN',
    save_dir = 'maua/modelzoo/flowerGAN',
    depth = depth,
    latent_size = 128,
    gpu = 0,
    seed = 27
)

dataloader = ProGANDataLoader(
    data_path = 'maua/datasets/flower_pix2pix/B',
    prescaled_data = True,
    prescaled_data_path = 'maua/datasets/flowerGAN_prescaled',
    transforms = tn.Compose([tn.Resize(2**depth),
                             tn.RandomHorizontalFlip(),
                             tn.RandomVerticalFlip(),
                             tn.RandomChoice([
                                tn.RandomRotation([0,0]),
                                tn.RandomRotation([90,90]),
                                tn.RandomRotation([270,270])
                             ]))
)

model.train(
    dataloader = dataloader,
    fade_in = 0.75,
    save_freq = 25,
    log_freq = 5,
    loss = "r1-reg",
    num_epochs = 75
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