import torch as th
import torchvision.transforms as tn
from torchvision.utils import save_image
from ..models import Pix2Pix
from PIL import Image

model = Pix2Pix(
    name = 'facades',
    save_dir = 'maua/modelzoo/facades_pix2pix',
    direction = 'BtoA',
    norm='batch',
    lambda_L1 = 100.0,
    lambda_feat = 10.0,
    lambda_vgg = 10.0,
    no_feat = True,
    no_vgg = True,
    no_dropout = False,
    no_lsgan = True,
    n_enhancers = 0,
    subnet_G='unet',
    unet_downs=8,
    # resnet_blocks = 9,
    n_scales = 1,
    n_layers_D = 3,
    gpu = 0,
    seed = 27
)


model.train(
    continue_train=True,
    data_path = 'maua/datasets/facades_pix2pix',
    num_epochs = 200,
    epochs_decay = 100,
    save_freq = 5,
    log_freq = 1,
    batch_size = 1,
    shuffle = True,
    resize = True,
    fineSize = 256,
    loadSize = 286,
    crop = True,
    vflip = False,
    hflip = True
)