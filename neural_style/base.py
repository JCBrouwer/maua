import os
import torch as th
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from .utils import *

class NeuralStyle(nn.Module):
    def __init__(self, style_blend_weights=None, image_size=1024, content_weight=5,
                 style_weight=100, tv_weight=1e-2, style_scale=1, original_colors=False,
                 pooling='max', model_type='vgg19', content_layers='relu4_2',
                 style_layers='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', num_iterations=1000,
                 optimizer='lbfgs', learning_rate=1, lbfgs_num_correction=0, gpu=0, backend='cudnn',
                 cudnn_autotune=True, seed=-1, print_iter=250, save_iter=1000):
        """
        constructor for the class NeuralStyle object
        :param style_blend_weights: weights for style images
        :param image_size: size of output image
        :param content_weight: strength of content image in output
        :param style_weight: strength of style image in output
        :param tv_weight: strength of total variation penalty in output image (higher value is "smoother" output)
        :param style_scale: scale of style image
        :param original_colors: whether to convert the final styled image to the original content's colors
        :param pooling: type of pooling to use between convolutional layers, options [avg, max]
        :param model_type: model type to use for optimization, options: [vgg19, vgg16, nin]
        :param content_layers: layers in model at which to optimize output image for content
        :param style_layers: layers in model at which to optimize output image for style
        :param num_iterations: number of iterations to optimize for
        :param optimizer: options: [lbfgs, adam]
        :param learning_rate: learning rate for optimizer
        :param lbfgs_num_correction: correction parameter for lbgfs
        :param gpu: ID of GPU to use
        :param backend: backend to use
        :param cudnn_autotune: whether to enable cudnn autobenchmarking
        :param seed: random seed
        :param print_iter: number of iterations to print after
        :param save_iter: number of iterations to save partial output
        """

        super(NeuralStyle, self).__init__()

        self.image_size = image_size
        self.num_iterations = num_iterations
        self.print_iter = print_iter
        self.save_iter = save_iter

        self.style_blend_weights = style_blend_weights
        self.style_scale = style_scale
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight

        self.original_colors = original_colors

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lbfgs_num_correction = lbfgs_num_correction

        self.gpu = gpu
        self.backend = backend
        self.cudnn_autotune = cudnn_autotune
        self.seed = seed

        if self.gpu > -1:
            if self.backend == 'cudnn':
                th.backends.cudnn.enabled = True
                if self.cudnn_autotune:
                    th.backends.cudnn.benchmark = True
            else:
                th.backends.cudnn.enabled = False
            th.cuda.set_device(self.gpu)
            self.dtype = th.cuda.FloatTensor
        elif self.gpu == -1:
           if self.backend =='mkl':
               th.backends.mkl.enabled = True
           self.dtype = th.FloatTensor

        Image.MAX_IMAGE_PIXELS = 1000000000 # Support gigapixel images

        if model_type == 'vgg19':
            from ..models.imagenet import VGG
            imagenet = VGG(model_file='maua/modelzoo/vgg19_imagenet.pth', layer_num=19, pooling=pooling,
                           tv_weight=tv_weight, content_layers=content_layers, style_layers=style_layers,
                           gpu=gpu, layer_depth_weighting=False)
        elif model_type == 'vgg16':
            from ..models.imagenet import VGG
            imagenet = VGG(layer_num=16, pooling=pooling, tv_weight=tv_weight, content_layers=content_layers,
                           style_layers=style_layers, gpu=gpu, layer_depth_weighting=False)
        elif model_type == 'nin':
            from ..models.imagenet import NIN
            imagenet = NIN(pooling=pooling, tv_weight=tv_weight, content_layers=content_layers,
                           style_layers=style_layers, gpu=gpu, layer_depth_weighting=False)
        else:
            print('Model type %s not supported'%(model_type))

        self.net = imagenet.net
        self.content_losses = imagenet.content_losses
        self.style_losses = imagenet.style_losses
        self.tv_losses = imagenet.tv_losses
        for param in self.net.parameters():
            param.requires_grad = False
        del imagenet


    def handle_style_images(self, style_image_list):
        style_images = []
        for image in style_image_list:
            style_size = int(self.image_size * self.style_scale)
            img = preprocess(image, style_size).type(self.dtype)
            style_images.append(img)

        # Handle style blending weights for multiple style inputs
        style_blend_weights = []
        if self.style_blend_weights == None:
            # Style blending not specified, so use equal weighting
            for i in style_image_list:
                style_blend_weights.append(1.0)
            for i, blend_weights in enumerate(style_blend_weights):
                style_blend_weights[i] = int(style_blend_weights[i])
        else:
            style_blend_weights = self.style_blend_weights.split(',')
            assert len(style_blend_weights) == len(style_image_list), \
              "-style_blend_weights and -style_images must have the same number of elements!"

        # Normalize the style blending weights so they sum to 1
        style_blend_sum = 0
        for i, blend_weights in enumerate(style_blend_weights):
            style_blend_weights[i] = float(style_blend_weights[i])
            style_blend_sum = float(style_blend_sum) + style_blend_weights[i]
        for i, blend_weights in enumerate(style_blend_weights):
            style_blend_weights[i] = float(style_blend_weights[i]) / float(style_blend_sum)

        return style_images, style_blend_weights


    # Configure the optimizer
    def setup_optimizer(self, img):
        if self.optimizer == 'lbfgs':
            print("Running optimization with L-BFGS")
            optim_state = {
                'max_iter': self.num_iterations,
                'tolerance_change': -1,
                'tolerance_grad': -1,
            }
            if self.lbfgs_num_correction > 0:
                optim_state['history_size'] = self.lbfgs_num_correction
            optimizer = optim.LBFGS([img], **optim_state)
            loopVal = 1
        elif self.optimizer == 'adam':
            print("Running optimization with ADAM")
            optimizer = optim.Adam([img], lr = self.learning_rate)
            loopVal = self.num_iterations - 1
        return optimizer, loopVal


    def maybe_print(self, num_calls, loss):
        if self.print_iter > 0 and num_calls[0] % self.print_iter == 0:
            print("Iteration " + str(num_calls[0]) + " / "+ str(self.num_iterations))
            for i, loss_module in enumerate(self.content_losses):
                print("  Content " + str(i+1) + " loss: " + str(loss_module.loss.item() * self.content_weight))
            for i, loss_module in enumerate(self.style_losses):
                print("  Style " + str(i+1) + " loss: " + str(loss_module.loss.item() * self.style_weight))
            print("  Total loss: " + str(loss.item()))


    def maybe_save(self, num_calls, img, output):
        if (self.save_iter > 0 and num_calls[0] % self.save_iter == 0) or \
            self.save_iter == 0 and num_calls[0] == self.num_iterations:
            output_filename, file_extension = os.path.splitext(output)
            if num_calls[0] == self.num_iterations:
                filename = "%s%s"%(output_filename, file_extension)
            else:
                filename = "%s_%s%s"%(output_filename, num_calls[0], file_extension)
            disp = deprocess(img)
            # Maybe perform postprocessing for color independent style transfer
            if self.original_colors:
                disp = original_colors(deprocess(self.content_image), disp)
            disp.save(str(filename))


    def run(self, content, style, init=None, output=None):
        """
        run the neural style algorithm on supplied images
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

        self.content_image = preprocess(content, self.image_size)
        self.content_image = match_color(self.content_image, self.style_images[0]).type(self.dtype)

        if init is not None:
            init_image = preprocess(init, self.image_size).type(self.dtype)
        else:
            _, C, H, W = self.content_image.size()
            init_image = th.rand(C, H, W).mul(255).unsqueeze(0)
        init_image = match_color(init_image, self.style_images[0]).type(self.dtype)

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
            self.maybe_save(num_calls, img, output)
            return loss

        optimizer, loopVal = self.setup_optimizer(img)
        while num_calls[0] <= loopVal:
            optimizer.step(feval)

        ret = deprocess(img)
        if self.original_colors:
            ret = original_colors(deprocess(self.content_image), ret)

        return ret