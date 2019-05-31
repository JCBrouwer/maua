import torch as th
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np
import functools, itertools


# TODO add n_critic for training just like ProGAN allowing multiple D steps per G step
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, n_scales=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        # creates nlayerD with each layer/scale accessible by name
        # append n_scales discriminators to model
        super(MultiscaleDiscriminator, self).__init__()
        self.n_scales = n_scales
        self.use_sigmoid = use_sigmoid

        for scale in range(self.n_scales):
            discriminator = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, True)
            setattr(self, 'discriminator_%s'%(scale), discriminator)

    def get_features(self, input):
        # forward with input downsampled for each scale
        # return list of lists of activations at each layer for each discriminator scale
        results = []
        for scale in range(self.n_scales):
            # scale input for given discrim scale
            prev_output = interpolate(input, scale_factor=2**(scale - self.n_scales + 1))

            # get separate layers of discriminator
            discriminator = getattr(self, 'discriminator_%s'%(scale)).model
            layers = {k:v for (k,v) in discriminator.named_children() if 'layer' in k}

            # forward through the network storing discriminator features by layer
            per_scale_results = []
            for idx in range(len(layers)):
                layer = layers['layer_%s'%idx]
                prev_output = layer(prev_output)
                per_scale_results.append(prev_output)

            final_preds = getattr(discriminator, 'final_conv')(per_scale_results[-1])
            per_scale_results.append(final_preds)

            if self.use_sigmoid:
                final_preds = getattr(discriminator, 'sigmoid')(final_preds)
                per_scale_results.append(final_preds)

            results.append(per_scale_results)
        return results

    def forward(self, input):
        # forward with input downsampled for each scale
        # return list of predictions for each discriminator scale
        results = []
        for scale in range(self.n_scales):
            discriminator = getattr(self, 'discriminator_%s'%(scale))
            downsampled_input = interpolate(input, scale_factor=2**(scale - self.n_scales + 1))
            predictions = discriminator(downsampled_input)
            results.append(predictions)
        return results
                

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, multiscale=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.model = nn.Sequential()
        self.model.add_module("layer_0", nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ))

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            self.model.add_module("layer_%s"%n, nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        self.model.add_module("layer_%s"%n_layers, nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))

        self.model.add_module("final_conv", nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))

        if use_sigmoid:
            self.model.add_module("sigmoid", nn.Sigmoid())

    def forward(self, input):
        return self.model(input)


class ProGrowDiscriminator(nn.Module):
    """ Discriminator of the progressive growing GAN """
    def __init__(self, depth=7, feature_size=512, use_eql=True):
        """
        constructor for the class
        :param depth: total depth of the discriminator (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use equalized learning rate
        """
        from .layers import equalized_conv2d

        super(ProGrowDiscriminator, self).__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), "feature size cannot be produced"

        # create state of the object
        self.use_eql = use_eql
        self.depth = depth
        self.feature_size = feature_size

        self.final_block = ProGrowFinalBlock(self.feature_size, use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = nn.ModuleList([])  # initialize to empty list

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            self.fromRGB = lambda out_channels: \
                equalized_conv2d(3, out_channels, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.fromRGB = lambda out_channels: Conv2d(3, out_channels, (1, 1), bias=True)

        self.rgb_to_features = nn.ModuleList([self.fromRGB(self.feature_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = ProGrowConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 3)),
                    use_eql=self.use_eql
                )
                rgb = self.fromRGB(int(self.feature_size // np.power(2, i - 2)))
            else:
                layer = ProGrowConvBlock(self.feature_size,
                                            self.feature_size, use_eql=self.use_eql)
                rgb = self.fromRGB(self.feature_size)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(self, x, depth, alpha):
        """
        forward pass of the discriminator
        :param x: input to the network
        :param depth: current depth of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return: out => raw prediction values (WGAN-GP)
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if depth > 0:
            residual = self.rgb_to_features[depth - 1](self.temporaryDownsampler(x))

            straight = self.layers[depth - 1](
                self.rgb_to_features[depth](x)
            )

            y = (alpha * straight) + ((1 - alpha) * residual)

            for block in reversed(self.layers[:depth - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)

        out = self.final_block(y)

        return out


class ProGrowFinalBlock(nn.Module):
    """ Final block for the Discriminator """
    def __init__(self, in_channels, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """
        from .layers import equalized_conv2d, MinibatchStdDev

        super(ProGrowFinalBlock, self).__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()
        if use_eql:
            self.conv_1 = equalized_conv2d(in_channels + 1, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = equalized_conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_3 = equalized_conv2d(in_channels, 1, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class ProGrowConvBlock(nn.Module):
    """ General block in the discriminator  """
    def __init__(self, in_channels, out_channels, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """
        from .layers import equalized_conv2d

        super(ProGrowConvBlock, self).__init__()

        if use_eql:
            self.conv_1 = equalized_conv2d(in_channels, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = equalized_conv2d(in_channels, out_channels, (3, 3), pad=1, bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)

        self.downSampler = nn.AvgPool2d(2)

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y
