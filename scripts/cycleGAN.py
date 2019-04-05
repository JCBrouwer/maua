import torch as th
import torchvision.transforms as tn
from torchvision.utils import save_image
from ..models import CycleGAN
from PIL import Image

model = CycleGAN(
    name = 'maps',
    save_dir = 'maua/modelzoo/maps_cyclegan',
    input_nc = 1,
    output_nc = 1,
    lambda_A = 10.0,
    lambda_B = 10.0,
    lambda_identity = 0.5,
    lambda_feat = 10.0,
    no_vgg = True,
    n_enhancers = 1,
    resnet_blocks = 9,
    n_scales = 3,
    n_layers_D = 3,
    gpu = 0,
    seed = 27
)

model.train(
    data_path = 'maua/datasets/maps_cyclegan',
    num_epochs = 30,
    epochs_decay = 20,
    save_freq = 5,
    log_freq = 1,
    batch_size = 5,
    resize = True,
    loadSize = 256,
    crop = False,
    vflip = False,
    hflip = True
)

result = model(tn.ToTensor()(Image.open('maua/datasets/flower_pix2pix/test/1.jpg').convert('RGB')).unsqueeze(0))
save_image(result, 'maua/output/pix2pix_flower1.png')

result = model(tn.ToTensor()(Image.open('maua/datasets/flower_pix2pix/test/2.jpg').convert('RGB')).unsqueeze(0))
save_image(result, 'maua/output/pix2pix_flower2.png')