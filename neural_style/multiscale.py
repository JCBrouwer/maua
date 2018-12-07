from .base import NeuralStyle
import os
import torch as th
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from PIL import Image
from .utils import *


class MultiscaleStyle(NeuralStyle):
    def __init__(self, start_size=256, steps=5, **kwargs):
        """
        constructor for the class MultiscaleStyle, extends NeuralStyle
        :param start_size: size to start styling from
        :param steps: scaling steps between start and final sizes (inclusive)
        """
        super(MultiscaleStyle, self).__init__(**kwargs)
        self.start_size = start_size
        self.steps = steps


    def maybe_save(self, num_calls, current_size, img, output):
        if (self.save_iter > 0 and num_calls[0] % self.save_iter == 0) \
            or (num_calls[0] == self.num_iterations and current_size == self.image_size):
            output_filename, file_extension = os.path.splitext(output)
            if current_size == self.image_size:
                filename = "%s%s"%(output_filename, file_extension)
            else:
                filename = "%s_%s%s"%(output_filename, current_size, file_extension)
            disp = deprocess(img)
            # Maybe perform postprocessing for color independent style transfer
            if self.original_colors:
                disp = original_colors(deprocess(self.content_image), disp)
            disp.save(str(filename))


    def run(self, content, style, init=None, output=None):
        """
        run the multiscale neural style algorithm on supplied images
        :param content: content image file path
        :param style: style image file path
        :param init: init image file path
        :param output: output image file path
        """
        if self.seed >= 0:
            th.manual_seed(self.seed)
            th.cuda.manual_seed(self.seed)
            th.backends.cudnn.deterministic=True

        styles = style.split(',')
        if output is None:
            style_names = map(lambda s: os.path.splitext(os.path.basename(s))[0], styles)
            content_name, _ = os.path.splitext(os.path.basename(content))
            output = 'maua/output/%s_%s.png'%(content_name, "_".join(style_names))

        self.style_images, style_blend_weights = self.handle_style_images(styles)

        content_final = preprocess(content, self.image_size)
        content_final = match_color(content_final, self.style_images[0]).type(self.dtype)

        if init is not None:
            init_image = preprocess(init, self.start_size)
        else:
            _, C, H, W = content_final.size()
            init_image = th.rand(C, H, W).mul(255).unsqueeze(0)
        init_image = match_color(init_image, self.style_images[0]).type(self.dtype)

        scale_factor = (self.image_size / self.start_size)**(1.0/(self.steps - 1))
        for scale in range(self.steps):
            current_size = self.image_size
            for s in range(self.steps - 1 - scale): current_size /= scale_factor
            current_size = round(current_size)
            print("Styling at size %d x %d" % (current_size, current_size))

            init_image = nn.functional.interpolate(init_image, size=current_size)
            self.content_image = nn.functional.interpolate(content_final, size=current_size)

            for i in self.content_losses:
                i.mode = 'capture'
            print("Capturing content targets")
            self.net(self.content_image)
            for i in self.content_losses:
                i.mode = 'None'

            for i, image in enumerate(self.style_images):
                print("Capturing style target " + str(i+1))
                for j in self.style_losses:
                    j.mode = 'capture'
                    j.blend_weight = style_blend_weights[i]
                self.net(image)

            for i in self.content_losses:
                i.mode = 'loss'
            for i in self.style_losses:
                i.mode = 'loss'

            img = init_image.requires_grad_()

            num_calls = [0]
            def feval():
                num_calls[0] += 1
                optimizer.zero_grad()
                self.net(img)
                loss = 0

                for mod in self.content_losses:
                    loss += mod.loss * self.content_weight
                for mod in self.style_losses:
                    loss += mod.loss * self.style_weight
                if self.tv_weight > 0:
                    for mod in self.tv_losses:
                        loss += mod.loss * self.tv_weight
                loss.backward()

                self.maybe_print(num_calls, loss)
                self.maybe_save(num_calls, current_size, img, output)
                return loss

            optimizer, loopVal = self.setup_optimizer(img)
            while num_calls[0] <= loopVal:
                optimizer.step(feval)

            init_image = match_color(img, self.style_images[0]).type(self.dtype)

        ret = deprocess(img)
        if self.original_colors:
            ret = original_colors(deprocess(self.content_image), ret)

        return ret
