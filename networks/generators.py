import torch as th
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np
import functools

# TODO figure out how to make compatible with unet
class MultiscaleGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=9, n_downsampling=3, n_enhancers=1, n_blocks_enhancer=3,
                 padding_type='reflect', use_deconvolution=True, subnet='resnet'):
        # main generator (resnet or unet) first, without last couple conv layers
        # then n_enhancers that double resolution (1/2 ngf each time)
        # accessible by downsample & upsampling blocks (setattr)
        super(MultiscaleGenerator, self).__init__()

        self.n_enhancers = n_enhancers

        if subnet is 'resnet':
            mainG = ResnetGenerator(input_nc, output_nc, ngf * 2**n_enhancers, norm_layer, use_dropout,
                                    n_blocks, n_downsampling, padding_type, use_deconvolution)
            # remove last couple conv layers
            mainG = nn.Sequential(*mainG.model[:-3])
        # elif subnet is 'unet':
        #     mainG = UnetGenerator(input_nc, output_nc, num_downs=n_downsampling, ngf * 2**n_enhancers,
        #                           norm_layer, use_dropout)
        else:
            print("Unknown sub network type")

        self.mainG = mainG
        
        for n in range(1, n_enhancers + 1):
            e_ngf = ngf * 2**(n_enhancers - n)

            if subnet is 'resnet':
                enhanceG = ResnetGenerator(input_nc=input_nc, output_nc=output_nc, ngf=e_ngf,
                                           norm_layer=norm_layer, use_dropout=use_dropout,
                                           n_blocks=n_blocks_enhancer, n_downsampling=1,
                                           padding_type=padding_type, use_deconvolution=use_deconvolution)

                downsample_block = nn.Sequential(*enhanceG.model[:7])
                upsample_block = nn.Sequential(*enhanceG.model[7:])

                # don't need final output conv layers except on last enhancer
                if not n == n_enhancers:
                    upsample_block = nn.Sequential(*upsample_block[:-3])

                setattr(self, 'downsample_%s'%(n), downsample_block)
                setattr(self, 'upsample_%s'%(n), upsample_block)
            # elif subnet is 'unet':
            #     enhanceG = UnetGenerator(input_nc, output_nc, n_downsampling, e_ngf, norm_layer, use_dropout)

            #     downsample_block = enhanceG.model[]
            #     upsample_block = enhanceG.model[]

            #     if not n is n_enhancers:
            #         enhanceG = 
            else:
                print("Unknown sub network type")
    
    def forward(self, input):
        # for each scale (main G + n_enhancers)
        # forward the downsampled input through the downsample block
        # add to the output from the previous scale
        # forward sum through upsampling block
        result = self.mainG(interpolate(input, scale_factor=2**(-self.n_enhancers)))
        for n in range(1, self.n_enhancers + 1):
            downsampled_input = interpolate(input, scale_factor=2**(n - self.n_enhancers))

            downsample_block = getattr(self, 'downsample_%s'%(n))
            upsample_block = getattr(self, 'upsample_%s'%(n))

            mixin = downsample_block(downsampled_input)
            result = upsample_block(result + mixin)
        return result


class ResnetGenerator(nn.Module):
    """Defines the generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    Code and idea originally from Justin Johnson's architecture. https://github.com/jcjohnson/fast-neural-style/"""
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=9, n_downsampling=3, padding_type='reflect', use_deconvolution=True):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2**i
            if use_deconvolution:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)]
            else:
                model += [nn.Upsample(scale_factor = 2, mode='bilinear',align_corners=True),
                          nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0)]
            model += [norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    """Defines the submodule for the resnet generator."""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UnetGenerator(nn.Module):
    """Defines the Unet generator.
    num_downs: number of downsamplings in UNet.
    For example, if num_downs == 7, image of size 128x128 will be size 1x1 at the bottleneck"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# wrap upsample and downsample sections with settatr to allow use in MultiscaleGenerator?
class UnetSkipConnectionBlock(nn.Module):
    """Defines the submodule with skip connection.
    X -------------------identity------------------ X
    |-- downsampling -- |submodule| -- upsampling --|"""
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
                 innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return th.cat([x, self.model(x)], 1)


class ProGrowGenerator(nn.Module):
    """ Generator of the progressive growing GAN network """
    def __init__(self, depth=7, latent_size=512, use_eql=True):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, Upsample

        super(ProGrowGenerator, self).__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size

        # register the modules required for the GAN
        self.initial_block = ProGrowInitialBlock(self.latent_size, use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the ToRGB layers for various outputs:
        if self.use_eql:
            from .layers import equalized_conv2d
            self.toRGB = lambda in_channels: \
                equalized_conv2d(in_channels, 3, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.toRGB = lambda in_channels: Conv2d(in_channels, 3, (1, 1), bias=True)

        self.rgb_converters = ModuleList([self.toRGB(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = ProGrowConvBlock(self.latent_size,
                                            self.latent_size, use_eql=self.use_eql)
                rgb = self.toRGB(self.latent_size)
            else:
                layer = ProGrowConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = self.toRGB(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

        # register the temporary upsampler
        self.temporaryUpsampler = Upsample(scale_factor=2)

    def forward(self, x, depth, alpha):
        """
        forward pass of the Generator
        :param x: input noise
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :return: y => output
        """

        # assert depth < self.depth, "Requested output depth cannot be produced"

        y = self.initial_block(x)

        if depth > 0:
            for block in self.layers[:depth - 1]:
                y = block(y)

            residual = self.rgb_converters[depth - 1](self.temporaryUpsampler(y))
            straight = self.rgb_converters[depth](self.layers[depth - 1](y))

            out = (alpha * straight) + ((1 - alpha) * residual)

        else:
            out = self.rgb_converters[0](y)

        return out


class ProGrowInitialBlock(nn.Module):
    """ Module implementing the initial block of the input """
    def __init__(self, in_channels, use_eql):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        :param use_eql: whether to use equalized learning rate
        """
        from .layers import equalized_conv2d, equalized_deconv2d, PixelwiseNorm

        super(ProGrowInitialBlock, self).__init__()

        if use_eql:
            self.conv_1 = equalized_deconv2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = equalized_conv2d(in_channels, in_channels, (3, 3),
                                            pad=1, bias=True)

        else:
            from torch.nn import Conv2d, ConvTranspose2d
            self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)

        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = th.unsqueeze(th.unsqueeze(x, -1), -1)

        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # apply pixel norm
        y = self.pixNorm(y)

        return y


class ProGrowConvBlock(nn.Module):
    """ Module implementing a general convolutional block """
    def __init__(self, in_channels, out_channels, use_eql):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        :param use_eql: whether to use equalized learning rate
        """
        from .layers import equalized_conv2d, PixelwiseNorm

        super(ProGrowConvBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2)

        if use_eql:
            self.conv_1 = equalized_conv2d(in_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = equalized_conv2d(out_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels, out_channels, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = Conv2d(out_channels, out_channels, (3, 3),
                                 padding=1, bias=True)

        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.upsample(x)
        y = self.pixNorm(self.lrelu(self.conv_1(y)))
        y = self.pixNorm(self.lrelu(self.conv_2(y)))

        return y