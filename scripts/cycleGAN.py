import os, itertools
import torch as th
import torchvision.transforms as tn
from torchvision.utils import save_image
from ..models import CycleGAN
from PIL import Image

model_name = 'horse2zebra'

model = CycleGAN(
    name = model_name,
    save_dir = 'maua/modelzoo/'+model_name,
    gpu = 0,
    seed = 27
)

model.train(
    data_path = 'maua/datasets/'+model_name,
    num_epochs = 100,
    epochs_decay = 100,
    save_freq = 10,
    log_freq = 1,
    batch_size = 1,
    resize = True,
    loadSize = 286,
    fineSize = 256,
    crop = True,
    vflip = False,
    hflip = True
)

del model.D_A
del model.D_B
model.model_names = ['G_A','G_B']

for name in model.model_names:
    if isinstance(name, str):
        net = getattr(model, name)
        net.eval()
        
os.makedirs('maua/output/'+model_name, exist_ok=True)
with th.no_grad():
    test_dir = "maua/datasets/%s/testA/"%model_name
    for i,path in enumerate(os.listdir(test_dir)):
        result = model.G_A(tn.ToTensor()(tn.Resize(512)(Image.open(test_dir+path).convert('RGB'))).unsqueeze(0))
        save_image(result, 'maua/output/%s/A_%i.png'%(model_name, i))

    test_dir = "maua/datasets/%s/testB/"%model_name
    for i,path in enumerate(os.listdir(test_dir)):
        result = model.G_B(tn.ToTensor()(tn.Resize(512)(Image.open(test_dir+path).convert('RGB'))).unsqueeze(0))
        save_image(result, 'maua/output/%s/B_%i.png'%(model_name, i))
