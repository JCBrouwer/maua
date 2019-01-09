import torch as th
import torch.autograd
import numpy as np


class NormalizeGradients(th.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.div(th.norm(grad_out, 1) + 1e-12)


class ContentLoss(th.nn.Module):

    def __init__(self, strength=1.0, normalize=False):
        super(ContentLoss, self).__init__()
        self.crit = th.nn.MSELoss()
        self.mode = 'None'
        self.strength = strength
        self.normalize = normalize

    def forward(self, input):
        if self.mode == 'capture':
            self.target = input.detach()
        elif self.mode == 'loss':
            self.loss = self.crit(input, self.target) * self.strength
        if self.normalize:
            input = NormalizeGradients.apply(input)
        return input


class GramMatrix(th.nn.Module):

    def forward(self, input):
        B, C, H, W = input.size()
        x_flat = input.view(C, H * W)
        return th.mm(x_flat, x_flat.t())


class StyleLoss(th.nn.Module):

    def __init__(self, strength=1.0, normalize=False):
        super(StyleLoss, self).__init__()
        self.target = th.Tensor()
        self.gram = GramMatrix()
        self.crit = th.nn.MSELoss()
        self.mode = 'None'
        self.blend_weight = None
        self.strength = strength
        self.normalize = normalize

    def forward(self, input):
        self.G = self.gram(input)
        self.G = self.G.div(input.nelement())
        if self.mode == 'capture':
            if self.blend_weight == None:
                self.target = self.G.detach()
            elif self.target.nelement() == 0:
                self.target = self.G.detach().mul(self.blend_weight)
            else:
                self.target = self.target.add(self.blend_weight, self.G.detach())
        elif self.mode == 'loss':
            self.loss = self.crit(self.G, self.target) * self.strength
        if self.normalize:
            input = NormalizeGradients.apply(input)
        return input


class TVLoss(th.nn.Module):

    def __init__(self, strength=1.0):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.x_diff = input[:,:,1:,:] - input[:,:,:-1,:]
        self.y_diff = input[:,:,:,1:] - input[:,:,:,:-1]
        self.loss = (th.sum(th.abs(self.x_diff)) + th.sum(th.abs(self.y_diff))) * self.strength
        return input


### NOTE **kwargs allow for extra arguments to discriminator (e.g. height and alpha for ProGANs)
class GANLoss:
    """ Base class for all losses """
    def __init__(self, D):
        self.D = D

    def loss_D(self, real_samps, fake_samps, **kwargs):
        raise NotImplementedError("loss_D method has not been implemented")

    def loss_G(self, real_samps, fake_samps, **kwargs):
        raise NotImplementedError("loss_G method has not been implemented")


class WGAN_GP(GANLoss):

    def __init__(self, device, D, drift=0.001, use_gp=False):
        super().__init__(D)
        self.device = device
        self.drift = drift
        self.use_gp = use_gp

    def gradient_penalty(self, real_samps, fake_samps, reg_lambda=10, **kwargs):
        """
        helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param height: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        from torch.autograd import grad

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(self.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)

        # forward pass
        op = self.D.forward(merged, **kwargs)

        # obtain gradient of op wrt. merged
        gradient = grad(outputs=op, inputs=merged, create_graph=True,
                        grad_outputs=th.ones_like(op),
                        retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def loss_D(self, real_samps, fake_samps, **kwargs):
        # define the (Wasserstein) loss
        fake_out = self.D(fake_samps, **kwargs)
        real_out = self.D(real_samps, **kwargs)

        loss = (th.mean(fake_out) - th.mean(real_out) + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            fake_samps.requires_grad = True  # turn on gradients for penalty calculation
            gp = self.gradient_penalty(real_samps, fake_samps, **kwargs)
            loss += gp

        return loss

    def loss_G(self, _, fake_samps, **kwargs):
        # calculate the WGAN loss for generator
        loss = -th.mean(self.D(fake_samps, **kwargs))

        return loss


class LSGAN(GANLoss):

    def __init__(self, D):
        super().__init__(D)

    def loss_D(self, real_samps, fake_samps, **kwargs):
        return 0.5 * (((th.mean(self.D(real_samps, **kwargs)) - 1) ** 2)
                      + (th.mean(self.D(fake_samps, **kwargs))) ** 2)

    def loss_G(self, _, fake_samps, **kwargs):
        return 0.5 * ((th.mean(self.D(fake_samps, **kwargs)) - 1) ** 2)


class LSGAN_SIGMOID(GANLoss):

    def __init__(self, D):
        super().__init__(D)

    def loss_D(self, real_samps, fake_samps, **kwargs):
        from torch.nn.functional import sigmoid
        real_scores = th.mean(sigmoid(self.D(real_samps, **kwargs)))
        fake_scores = th.mean(sigmoid(self.D(fake_samps, **kwargs)))
        return 0.5 * (((real_scores - 1) ** 2) + (fake_scores ** 2))

    def loss_G(self, _, fake_samps, **kwargs):
        from torch.nn.functional import sigmoid
        scores = th.mean(sigmoid(self.D(fake_samps, **kwargs)))
        return 0.5 * ((scores - 1) ** 2)


class HingeLoss(GANLoss):

    def __init__(self, D):
        super().__init__(D)

    def loss_D(self, real_samps, fake_samps, **kwargs):
        r_preds = self.D(real_samps, **kwargs)
        f_preds = self.D(fake_samps, **kwargs)

        loss = th.mean(th.nn.ReLU()(1 - r_preds)) + th.mean(th.nn.ReLU()(1 + f_preds))

        return loss

    def loss_G(self, _, fake_samps, **kwargs):
        return -th.mean(self.D(fake_samps, **kwargs))


class RelativisticAverageHinge(GANLoss):

    def __init__(self, D):
        super().__init__(D)

    def loss_D(self, real_samps, fake_samps, **kwargs):
        # Obtain predictions
        r_preds = self.D(real_samps, **kwargs)
        f_preds = self.D(fake_samps, **kwargs)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        loss = th.mean(th.nn.ReLU()(1 - r_f_diff)) + th.mean(th.nn.ReLU()(1 + f_r_diff))

        return loss

    def loss_G(self, real_samps, fake_samps, **kwargs):
        # Obtain predictions
        r_preds = self.D(real_samps, **kwargs)
        f_preds = self.D(fake_samps, **kwargs)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        return th.mean(th.nn.ReLU()(1 + r_f_diff)) + th.mean(th.nn.ReLU()(1 - f_r_diff))

class R1Regularized(GANLoss):

    def __init__(self, D):
        super().__init__(D)
        from torch.nn import BCEWithLogitsLoss
        self.criterion = BCEWithLogitsLoss()
        self.reg_param = 10
        self.reg = None

    def loss_D(self, real_samps, fake_samps, height, alpha):
        # predictions for real images and fake images separately :
        r_preds = self.D(real_samps, height, alpha)
        f_preds = self.D(fake_samps, height, alpha)
        self.reg = self.reg_param * self.compute_grad2(r_preds, real_samps).mean()

        # calculate the real loss:
        real_loss = self.criterion(th.squeeze(r_preds), th.ones(real_samps.shape[0]).to(self.D.device))

        # calculate the fake loss:
        fake_loss = self.criterion(th.squeeze(f_preds), th.zeros(fake_samps.shape[0]).to(self.D.device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def loss_G(self, _, fake_samps, height, alpha):
        preds, _, _ = self.D(fake_samps, height, alpha)
        return self.criterion(th.squeeze(preds), th.ones(fake_samps.shape[0]).to(self.D.device))

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = th.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg