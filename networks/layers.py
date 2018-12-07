import torch as th
import torch.nn as nn


# extending Conv2D and Deconv2D layers for equalized learning rate logic
class equalized_conv2d(nn.Module):
    """ conv2d with the concept of equalized learning rate """
    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, initializer='kaiming', bias=True):
        """
        constructor for the class
        :param c_in: input channels
        :param c_out:  output channels
        :param k_size: kernel size (h, w) should be a tuple or a single integer
        :param stride: stride for conv
        :param pad: padding
        :param initializer: initializer. one of kaiming or xavier
        :param bias: whether to use bias or not
        """
        super(equalized_conv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':
            nn.init.kaiming_normal_(self.conv.weight, a=nn.init.calculate_gain('conv2d'))
        elif initializer == 'xavier':
            nn.init.xavier_normal_(self.conv.weight)

        self.use_bias = bias

        if self.use_bias:
            self.bias = nn.Parameter(th.FloatTensor(c_out).fill_(0))
        self.scale = (th.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        try:
            dev_scale = self.scale.to(x.get_device())
        except RuntimeError:
            dev_scale = self.scale
        x = self.conv(x.mul(dev_scale))
        if self.use_bias:
            return x + self.bias.view(1, -1, 1, 1).expand_as(x)
        return x


class equalized_deconv2d(nn.Module):
    """ Transpose convolution using the equalized learning rate """
    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, initializer='kaiming', bias=True):
        """
        constructor for the class
        :param c_in: input channels
        :param c_out: output channels
        :param k_size: kernel size
        :param stride: stride for convolution transpose
        :param pad: padding
        :param initializer: initializer. one of kaiming or xavier
        :param bias: whether to use bias or not
        """
        super(equalized_deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':
            nn.init.kaiming_normal_(self.deconv.weight, a=nn.init.calculate_gain('conv2d'))
        elif initializer == 'xavier':
            nn.init.xavier_normal_(self.deconv.weight)

        self.use_bias = bias

        if self.use_bias:
            self.bias = nn.Parameter(th.FloatTensor(c_out).fill_(0))
        self.scale = (th.mean(self.deconv.weight.data ** 2)) ** 0.5
        self.deconv.weight.data.copy_(self.deconv.weight.data / self.scale)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        try:
            dev_scale = self.scale.to(x.get_device())
        except RuntimeError:
            dev_scale = self.scale

        x = self.deconv(x.mul(dev_scale))
        if self.use_bias:
            return x + self.bias.view(1, -1, 1, 1).expand_as(x)
        return x


class equalized_linear(nn.Module):
    """ Linear layer using equalized learning rate """
    def __init__(self, c_in, c_out, initializer='kaiming', bias=True):
        """
        Linear layer from pytorch extended to include equalized learning rate
        :param c_in: number of input channels
        :param c_out: number of output channels
        :param initializer: initializer to be used: one of "kaiming" or "xavier"
        :param bias: whether to use bias with the linear layer
        """
        super(equalized_linear, self).__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        if initializer == 'kaiming':
            nn.init.kaiming_normal_(self.linear.weight,
                                       a=nn.init.calculate_gain('linear'))
        elif initializer == 'xavier':
            nn.init.xavier_normal_(self.linear.weight)

        self.use_bias = bias

        if self.use_bias:
            self.bias = nn.Parameter(th.FloatTensor(c_out).fill_(0))
        self.scale = (th.mean(self.linear.weight.data ** 2)) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data / self.scale)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        try:
            dev_scale = self.scale.to(x.get_device())
        except RuntimeError:
            dev_scale = self.scale
        x = self.linear(x.mul(dev_scale))
        if self.use_bias:
            return x + self.bias.view(1, -1).expand_as(x)
        return x


class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class MinibatchStdDev(nn.Module):
    """ Minibatch standard deviation layer for the discriminator """
    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = th.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = th.cat([x, y], 1)

        # return the computed values:
        return y
