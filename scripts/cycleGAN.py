import os
import torch as th
import torchvision.transforms as tn
from torchvision.utils import save_image
from ..models import CycleGAN
from PIL import Image

model = CycleGAN(
    name = 'maps',
    save_dir = 'maua/modelzoo/maps_cyclegan',
    lambda_A = 10.0,
    lambda_B = 10.0,
    lambda_identity = 0.05,
    lambda_feat = 1.0,
    no_feat = False,
    no_vgg = True,
    n_enhancers = 2,
    resnet_blocks = 6,
    n_scales = 3,
    n_layers_D = 3,
    gpu = 0,
    seed = 27
)

model.train(
    data_path = 'maua/datasets/maps_cyclegan',
    num_epochs = 100,
    epochs_decay = 100,
    save_freq = 10,
    log_freq = 1,
    batch_size = 1,
    resize = True,
    loadSize = 600,
    fineSize = 256,
    crop = True,
    vflip = True,
    hflip = True
)

del model.D_A
del model.D_B
model.model_names = ['G_A','G_B']

for name in model.model_names:
    if isinstance(name, str):
        net = getattr(model, name)
        net.eval()
        
os.makedirs('maua/output/maps_cyclegan/', exist_ok=True)
with th.no_grad():
    test_dir = "maua/datasets/maps_cyclegan/etc/val/"
    for i,path in enumerate(os.listdir(test_dir)):
        result = [model.G_A(tn.ToTensor()(tn.Resize(600)(Image.open(test_dir+path).convert('RGB'))).unsqueeze(0)),
                  model.G_B(tn.ToTensor()(tn.Resize(600)(Image.open(test_dir+path).convert('RGB'))).unsqueeze(0))]
        save_image(result, 'maua/output/maps_cyclegan/%i.png'%i)
