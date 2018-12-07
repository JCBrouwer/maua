import scipy.ndimage as ndi
from skimage import feature
import torch as th
import torch.nn.functional as F
import numpy as np
from ..video import GLWindow
from ..models import ProGAN, Pix2Pix


with th.no_grad():
    win_size = 512

    # load pix2pix model
    pix = Pix2Pix(name="default", save_dir="maua/modelzoo/flower_pix2pix", gpu=0, model_names=["G"])
    pix.load_networks('final')

    # load proGAN model
    pg = ProGAN(name="default", depth=8, latent_size=128, save_dir="maua/modelzoo/flower_progan/", model_names = ["G"], gpu=0, use_ema=False)
    pg.load_networks('final')

    # generate looping noise for latent variabels
    len_loop = 900
    latents = th.randn(len_loop, 128)*4
    latents = ndi.gaussian_filter(latents.numpy(), [15, 0], mode='wrap')
    latents = th.from_numpy(latents).float().to(th.device('cuda'))

    idx = -1

    # define per frame operation
    def frame(state):
        global idx
        idx = (idx + 1) % len_loop

        # feed latent variable to ProGAN
        state = pg(latents[idx].unsqueeze(0))
        state = th.clamp(state, min=0, max=1)

        # convert back to numpy and run canny edge detection (should do this in pytorch for speed)
        state = state.squeeze().permute(1,2,0).cpu()
        state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
        state = feature.canny(state).astype(np.float)

        # convert back, upscale the result, and run pix2pix
        state = th.tensor(state).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()
        state = F.interpolate(state, win_size, mode='bilinear')
        state = pix(state)
        return state

    # create GL window and run it
    glw = GLWindow(init_f=frame, per_frame_f=frame, width=win_size, height=win_size)
    glw.run()