import time, datetime, os, math
import torch as th
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
from torchvision.utils import save_image
import torchvision.transforms as tn
from ..dataloaders.pix2pix import *
from .base import BaseModel
from ..networks.generators import MultiscaleGenerator
from ..networks.discriminators import MultiscaleDiscriminator

class Pix2Pix(BaseModel):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, ndf=32, subnet_G='resnet', unet_downs=8,
                 resnet_blocks=9, n_layers_D=3, norm='batch', pool_size=0, no_dropout=False, no_lsgan=True,
                 no_feat=False, no_vgg=True, init_type='normal', init_gain=0.02, lr=0.0002, beta1=0.5, beta2=0.999,
                 direction='AtoB', lambda_L1=100.0, n_enhancers=2, n_scales=3, lambda_feat=10.0,
                 lambda_vgg=1.0, vgg_type='vgg19', pooling='max', style_layers="",
                 content_layers="relu1_1,relu2_1,relu3_1,relu4_1,relu5_1", **kwargs):
        """
        constructor for the class Pix2Pix, extends BaseModel
        :param input_nc: number of channels of input images
        :param output_nc: number of channels of output images
        :param ngf: number of channels of generator (at largest spatial resolution)
        :param ndf: number of channels of discriminator (at largest spatial resolution)
        :param direction: AtoB or BtoA
        :param lambda_L1: weight for L1 loss
        :param lambda_feat: weight for discriminator feature loss
        :param lambda_vgg: weight for vgg feature loss
        :param n_enhancers: number of enhancer networks in the generator
        :param n_scales: number of scales in the discriminator (if type is multiscale)
        :param n_layers_D: number of layers in the discriminator (if type is n_layers)
        :param subnet_G: type of network to use in the generator, options: [resnet, unet]
        :param resnet_blocks: number of blocks in Resnet generator (if type is resnet)
        :param unet_downs: number of downsamplings in Unet generator (if type is unet)
                           e.g. if unet_downs is 7, 128x128 image will be size 1x1 at the bottleneck
        :param norm: instance normalization or batch normalization
        :param vgg_type: type of model to use for vgg feature normalization, options [vgg19, vgg16, nin]
        :param pooling: type of pooling to use in vgg model, options [max, avg]
        :param content_layers: layers to insert vgg feature losses on
        :param style_layers: same as content_layers but will match gram matrices instead
        :param pool_size: the size of image buffer that stores previously generated images
        :param no_dropout: no dropout for the generator
        :param no_lsgan: do *not* use least square GAN, if false, use vanilla GAN
        :param no_feat: do *not* use discriminator feature loss
        :param no_vgg: do *not* use vgg feature loss. VGG LOSS IS BROKEN AT THE MOMENT, LEAVE THIS True
        :param init_type: network initialization, options: [normal, xavier, kaiming, orthogonal]
        :param init_gain: scaling factor for normal, xavier and orthogonal
        :param lr: initial learning rate for adam
        :param beta1: beta1 parameter for adam
        :param beta2: beta2 parameter for adam
        """
        super(Pix2Pix,self).__init__(**kwargs)

        self.direction = direction
        self.input_nc = input_nc
        self.output_nc = output_nc

        # load/define networks
        norm_layer = self.get_norm_layer(norm_type=norm)

        net = MultiscaleGenerator(input_nc=input_nc, output_nc=output_nc, ngf=ngf, norm_layer=norm_layer,
                                  use_dropout=not no_dropout, n_blocks=resnet_blocks, n_enhancers=n_enhancers,
                                  subnet=subnet_G,  n_blocks_enhancer=3, padding_type='reflect',
                                  use_deconvolution=True, n_downsampling=3)
        self.G = self.init_net(net, init_type, init_gain)

        use_sigmoid = no_lsgan
        net = MultiscaleDiscriminator(input_nc=input_nc + output_nc, ndf=ndf, n_scales=n_scales,
                                      n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        self.D = self.init_net(net, init_type, init_gain)

        self.fake_AB_pool = ImagePool(pool_size)
        self.fake_label = th.tensor(0.0).to(self.device)
        self.real_label = th.tensor(1.0).to(self.device)

        # define loss functions
        if not no_lsgan:
            self.loss_GAN = nn.MSELoss()
        else:
            self.loss_GAN = nn.BCELoss()
        self.loss_L1 = nn.L1Loss()
        self.lambda_L1 = lambda_L1
        self.lambda_feat = lambda_feat
        self.lambda_vgg = lambda_vgg

        # initialize optimizers
        self.optimizers = []
        self.optimizer_G = th.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = th.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.no_vgg = no_vgg
        self.no_feat = no_feat
        if not self.no_vgg:
            self.vgg, self.vgg_features = self.get_vgg(vgg_type, pooling, content_layers, style_layers)


    def get_vgg(self, vgg_type, pooling, content_layers, style_layers):
        if vgg_type == 'vgg19':
            from .imagenet import VGG
            imagenet = VGG(layer_num=19, pooling=pooling, tv_weight=0, content_layers=content_layers,
                           style_layers=style_layers, gpu=self.gpu, layer_depth_weighting=True)
        elif vgg_type == 'vgg16':
            from .imagenet import VGG
            imagenet = VGG(layer_num=16, pooling=pooling, tv_weight=0, content_layers=content_layers,
                           style_layers=style_layers, gpu=self.gpu, layer_depth_weighting=True)
        elif vgg_type == 'nin':
            from .imagenet import NIN
            imagenet = NIN(pooling=pooling, tv_weight=0, content_layers=content_layers,
                           style_layers=style_layers, gpu=self.gpu, layer_depth_weighting=True)
        else:
            print('Model type %s not supported'%(vgg_type))

        vgg = imagenet.net
        for param in vgg.parameters():
            param.requires_grad = False
        vgg_features = imagenet.content_losses + imagenet.style_losses + imagenet.tv_losses

        del imagenet

        return vgg, vgg_features


    def get_norm_layer(self, norm_type='instance'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        return norm_layer


    def get_scheduler(self, optimizer, start_epoch, num_epochs, epochs_decay, lr_policy, lr_decay):
        if lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + start_epoch - num_epochs) / float(epochs_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=0.1)
        elif lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                    threshold=0.01, patience=5)
        elif lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
        return scheduler


    def init_weights(self, net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented'%init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)


    def init_net(self, net, init_type='normal', init_gain=0.02):
        if self.device == th.device('cuda'):
            assert(torch.cuda.is_available())
            net = torch.nn.DataParallel(net).to(self.device)
        self.init_weights(net, init_type, gain=init_gain)
        return net


    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']


    def optimize_D(self, real_A, fake_B, real_B):
        real_AB = th.cat((real_A, real_B), 1)
        fake_AB = self.fake_AB_pool.query(th.cat((real_A, fake_B), 1))

        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()

        pred_fake = self.D(fake_AB.detach()) # detach to prevent gradient propagating to G
        loss_D_fake = 0
        for preds in pred_fake:
            loss_D_fake += self.loss_GAN(preds, self.fake_label.expand_as(preds))

        if not self.no_feat:
            (self.D.module if self.gpu is not -1 else self.D).capture_feature_targets() 
        pred_real = self.D(real_AB)
        loss_D_real = 0
        for preds in pred_real:
            loss_D_real += self.loss_GAN(preds, self.real_label.expand_as(preds))

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        self.optimizer_D.step()

        return loss_D.item()


    def optimize_G(self, real_A, fake_B, real_B):
        fake_AB = th.cat((real_A, fake_B), 1)

        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        
        if not self.no_feat:
            (self.D.module if self.gpu is not -1 else self.D).capture_features()
        pred_fake = self.D(fake_AB)
        loss_G_GAN = 0
        for preds in pred_fake:
            loss_G_GAN += self.loss_GAN(preds, self.fake_label.expand_as(preds))

        loss_G_L1 = self.loss_L1(fake_B, real_B) * self.lambda_L1

        loss_G = loss_G_GAN + loss_G_L1

        if not self.no_feat:
            loss_G_feat = (self.D.module if self.gpu is not -1 else self.D).feature_loss()
            loss_G += loss_G_feat * self.lambda_feat

        if not self.no_vgg:
            for mod in self.vgg_features:
                mod.mode = 'capture'
            self.vgg(real_B)
            
            for mod in self.vgg_features:
                mod.mode = 'loss'
            self.vgg(fake_B)

            loss_G_VGG = 0
            for i,mod in enumerate(self.vgg_features):
                loss_G_VGG += mod.loss * self.lambda_vgg

            loss_G += loss_G_VGG

        loss_G.backward()
        self.optimizer_G.step()

        return loss_G.item()


    def forward(self, real_A):
        return self.G(real_A)


    def train(self, continue_train=False, data_path='maua/datasets/default_pix2pix',
        dataloader=None, start_epoch=1, num_epochs=100, epochs_decay=100, save_freq=10, log_freq=1,
        loadSize=256, fineSize=256, serial_batches=True, lr_policy='lambda', lr_decay_iters=50,
        shuffle=True, num_workers=3, batch_size=16, resize=True, crop=True, vflip=True, hflip=True):
        """
        Training function for Pix2Pix object
        :param continue_train: whether to continue training or not
        :param data_path: path to folder containing images to train on
        :param dataloader: custom dataloader to use, otherwise images will be resized to max resolution
                           and normalized to ImageNet mean and standard deviation
        :param batch_size: number of images per batch
        :param shuffle: whether to shuffle images or enumerate dataset in order
        :param num_workers: number of worker processes for data loading
        :param start_epoch: epoch to continue training from (defaults to most recent, if continuing training)
        :param num_epochs: number of epochs to train with full learning rate
        :param epochs_decay: number of epochs to train with decaying learning rate
        :param save_freq: frequency to save checkpoints in number of epochs
        :param log_freq: frequency to log / save images in number of or fraction of epochs
        :param resize: whether to resize loaded images
        :param crop: whether to crop scaled images
        :param vflip: whether to flip vertically
        :param hflip: whether to flip horizontally
        :param loadSize: size to scale loaded images to (can be largest side or tuple) if resize set
        :param fineSize: size to crop scaled images to (can be largest side or tuple) if crop set
        :param serial_batches: if true, takes image pairs, otherwise takes them randomly
        :param lr_policy: learning rate policy, options: [lambda, step, plateau, cosine]
        :param lr_decay_iters: multiply by a gamma every lr_decay_iters iterations
        """
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        self.model_names = ["G", "D"]

        if continue_train:
            start_epoch = self.get_latest_network(start_epoch, max_epoch=(num_epochs+epochs_decay))

        self.schedulers = [self.get_scheduler(optim, start_epoch, num_epochs, epochs_decay, lr_policy, lr_decay_iters)
                           for optim in self.optimizers]

        print("Starting training on "+str(self.device))

        # create dataloader
        if dataloader is None:
            dataloader = Pix2PixDataLoader(batch_size=batch_size, shuffle=True, num_workers=3,
                            data_path=data_path, direction=self.direction, serial_batches=serial_batches,
                            resize=resize, crop=crop, vflip=vflip, hflip=hflip, input_nc=self.input_nc,
                            output_nc=self.output_nc, loadSize=loadSize, fineSize=fineSize)

        dataset_size = len(dataloader)
        total_batches = dataloader.batches()
        print('%d training images in %d batches' % (dataset_size, total_batches))
        start_time = time.time()

        for epoch in range(start_epoch, num_epochs + epochs_decay + 1):
            for i, data in enumerate(dataloader):
                AtoB = self.direction == 'AtoB'
                real_A = data['A' if AtoB else 'B'].to(self.device)
                real_B = data['B' if AtoB else 'A'].to(self.device)
                fake_B = self.G(real_A)

                loss_D = self.optimize_D(real_A, fake_B, real_B)
                loss_G = self.optimize_G(real_A, fake_B, real_B)

                samples = th.stack([real_A[0],fake_B[0],real_B[0]]).detach()

                if i % math.ceil(total_batches * log_freq) == 0 and not (i == 0 or i == total_batches):
                    img_file = os.path.join(self.save_dir, "images", "sample_%d_%d.png"%(epoch, i))
                    save_image(samples, img_file, nrow=3)

                    elapsed = datetime.timedelta(seconds=time.time() - start_time)
                    print('Finished batch %d / %d \t [Elapsed: %s] \t loss_D: %.4f \t loss_G: %.4f'%
                          (i, total_batches, elapsed, loss_D, loss_G))

            if epoch % log_freq == 0:
                img_file = os.path.join(self.save_dir, "images", "sample_%d.png" % epoch)
                save_image(samples, img_file, nrow=3)

                elapsed = datetime.timedelta(seconds=time.time() - start_time)
                print('End of epoch %d / %d \t [Elapsed: %s] \t loss_D %.4f \t loss_G: %.4f'%
                      (epoch, num_epochs+epochs_decay, elapsed, loss_D, loss_G))

            if epoch % save_freq == 0:
                print('Saving model')
                self.save_networks(epoch)

            self.update_learning_rate()

        print("Training finished, took: ", datetime.timedelta(seconds=time.time() - start_time))
        self.save_networks("final")