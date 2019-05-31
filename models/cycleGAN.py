import time, datetime, os, math, itertools
import torch as th
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
from torchvision.utils import save_image
import torchvision.transforms as tn
from ..dataloaders.pix2pix import *
from .base import BaseModel
from ..networks.generators import MultiscaleGenerator, ResnetGenerator
from ..networks.discriminators import MultiscaleDiscriminator


class CycleGAN(BaseModel):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, ndf=64, subnet_G='resnet', unet_downs=8,
                resnet_blocks=9, n_layers_D=3, norm='instance', pool_size=50, no_dropout=True, no_lsgan=False,
                no_feat=True, no_vgg=True, init_type='normal', init_gain=0.02, lr=0.0002, beta1=0.5, beta2=0.999,
                n_enhancers=0, n_scales=1, lambda_feat=0.5, lambda_A=10.0, lambda_B=10.0,
                lambda_identity=0.5, lambda_vgg=0.0, vgg_type='vgg19', vgg_pooling='max',
                style_layers="relu1_1,relu2_1,relu3_1,relu4_1,relu5_1", content_layers="", **kwargs):
        """
        constructor for the class CycleGAN, extends BaseModel
        :param input_nc: number of channels of input images
        :param output_nc: number of channels of output images
        :param ngf: number of channels of generator (at largest spatial resolution)
        :param ndf: number of channels of discriminator (at largest spatial resolution)
        :param lambda_feat: weight for discriminator feature loss
        :param lambda_vgg: weight for vgg feature loss
        :param lambda_A: weight for cycle loss (A -> B -> A)
        :param lambda_B: weight for cycle loss (B -> A -> B)
        :param lambda_identity: use identity mapping. Setting lambda_identity other than 0 has an effect
                                of scaling the weight of the identity mapping loss. For example, if the
                                weight of the identity loss should be 10 times smaller than the weight
                                of the reconstruction loss, set lambda_identity = 0.1
        :param n_enhancers: number of enhancer networks in the generator
        :param n_scales: number of scales in the discriminator
        :param n_layers_D: number of layers in each discriminator
        :param subnet_G: type of network to use in the generator, options: [resnet, unet]
        :param resnet_blocks: number of blocks in Resnet generator (if type is resnet)
        :param unet_downs: number of downsamplings in Unet generator (if type is unet)
                            e.g. if unet_downs is 7, 128x128 image will be size 1x1 at the bottleneck
        :param norm: instance normalization or batch normalization
        :param vgg_type: type of model to use for vgg feature normalization, options [vgg19, vgg16, nin]
        :param vgg_pooling: type of pooling to use in vgg model, options [max, avg]
        :param content_layers: layers to insert vgg feature losses on
        :param style_layers: same as content_layers but will match gram matrices instead
        :param pool_size: the size of image buffer that stores previously generated images
        :param no_dropout: no dropout for the generator
        :param no_lsgan: do *not* use least square GAN, use vanilla GAN instead
        :param no_feat: do *not* use discriminator feature loss
        :param no_vgg: do *not* use vgg feature loss. VGG LOSS IS BROKEN AT THE MOMENT, LEAVE THIS True
        :param init_type: network initialization, options: [normal, xavier, kaiming, orthogonal]
        :param init_gain: scaling factor for normal, xavier and orthogonal
        :param lr: initial learning rate for adam
        :param beta1: beta1 parameter for adam
        :param beta2: beta2 parameter for adam
        """
        super(CycleGAN, self).__init__(**kwargs)

        self.input_nc = input_nc
        self.output_nc = output_nc

        # load/define networks
        norm_layer = self.get_norm_layer(norm_type=norm)

        net = MultiscaleGenerator(input_nc=input_nc, output_nc=output_nc, ngf=ngf, norm_layer=norm_layer,
                                    use_dropout=not no_dropout, n_blocks=resnet_blocks, n_enhancers=n_enhancers,
                                    subnet=subnet_G,  n_blocks_enhancer=3, padding_type='reflect',
                                    use_deconvolution=True, n_downsampling=4)
        self.G_A = self.init_net(net, init_type, init_gain)

        net = MultiscaleGenerator(input_nc=output_nc, output_nc=input_nc, ngf=ngf, norm_layer=norm_layer,
                                    use_dropout=not no_dropout, n_blocks=resnet_blocks, n_enhancers=n_enhancers,
                                    subnet=subnet_G,  n_blocks_enhancer=3, padding_type='reflect',
                                    use_deconvolution=True, n_downsampling=4)
        self.G_B = self.init_net(net, init_type, init_gain)

        use_sigmoid = no_lsgan
        net = MultiscaleDiscriminator(input_nc=output_nc, ndf=ndf, n_scales=n_scales, n_layers=n_layers_D,
                                      norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        self.D_A = self.init_net(net, init_type, init_gain)

        net = MultiscaleDiscriminator(input_nc=input_nc, ndf=ndf, n_scales=n_scales, n_layers=n_layers_D,
                                      norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        self.D_B = self.init_net(net, init_type, init_gain)

        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)
        self.fake_label = th.tensor(0.0).to(self.device)
        self.real_label = th.tensor(1.0).to(self.device)

        # define loss functions
        if not no_lsgan:
            self.loss_GAN = nn.MSELoss()
        else:
            self.loss_GAN = nn.BCELoss()
        self.loss_cycle = nn.L1Loss()
        self.loss_identity = nn.L1Loss()
        self.lambda_feat = lambda_feat
        self.lambda_vgg = lambda_vgg
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_identity

        # initialize optimizers
        self.optimizers = []
        self.optimizer_G = th.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = th.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=lr, betas=(beta1, beta2))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.no_vgg = no_vgg
        self.no_feat = no_feat
        if not self.no_vgg:
            self.vgg, self.vgg_features = self.get_vgg(vgg_type, vgg_pooling, content_layers, style_layers)


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


    def forward(self, real_A, real_B):
        fake_B = self.G_A(real_A)
        rec_A = self.G_B(fake_B)
        fake_A = self.G_B(real_B)
        rec_B = self.G_A(fake_A)
        return fake_A, fake_B, rec_A, rec_B


    def backward_D_basic(self, D, real, fake):
        # Real
        D.pred_real = (D.module if self.gpu is not -1 else D).get_features(real)
        # pred_real = D(real)
        loss_D_real = 0
        for preds in D.pred_real:
            loss_D_real += self.loss_GAN(preds[-1], self.real_label.expand_as(preds[-1]))
            # loss_D_real += self.loss_GAN(preds, self.real_label.expand_as(preds))

        # Fake
        pred_fake = D(fake.detach())
        loss_D_fake = 0
        for preds in pred_fake:
            loss_D_fake += self.loss_GAN(preds, self.fake_label.expand_as(preds))

        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D.item()


    def backward_D_A(self, fake_B, real_B):
        fake_B = self.fake_B_pool.query(fake_B)
        return self.backward_D_basic(self.D_A, real_B, fake_B)


    def backward_D_B(self, fake_A, real_A):
        fake_A = self.fake_A_pool.query(fake_A)
        return self.backward_D_basic(self.D_B, real_A, fake_A)


    def backward_G(self, fake_A, fake_B, rec_A, rec_B, real_A, real_B):
        # Identity loss
        if self.lambda_identity > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.G_A(real_B)
            loss_idt_A = self.loss_identity(idt_A, real_B) * self.lambda_B * self.lambda_identity
            # G_B should be identity if real_A is fed.
            idt_B = self.G_B(real_A)
            loss_idt_B = self.loss_identity(idt_B, real_A) * self.lambda_A * self.lambda_identity
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        pred_fake_A = (self.D_A.module if self.gpu is not -1 else self.D_A).get_features(fake_B)
        # pred_fake_A = self.D_A(fake_B)
        loss_G_A = 0
        for preds in pred_fake_A:
            loss_G_A += self.loss_GAN(preds[-1], self.fake_label.expand_as(preds[-1]))
            # loss_G_A += self.loss_GAN(preds, self.fake_label.expand_as(preds))

        # GAN loss D_B(G_B(B))
        pred_fake_B = (self.D_B.module if self.gpu is not -1 else self.D_B).get_features(fake_A)
        # pred_fake_B = self.D_B(fake_A)
        loss_G_B = 0
        for preds in pred_fake_B:
            loss_G_B += self.loss_GAN(preds[-1], self.fake_label.expand_as(preds[-1]))
            # loss_G_B += self.loss_GAN(preds, self.fake_label.expand_as(preds))
            
        # cycle losses
        loss_cycle_A = self.loss_cycle(rec_A, real_A) * self.lambda_A
        loss_cycle_B = self.loss_cycle(rec_B, real_B) * self.lambda_B

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        # discriminator mean feature losses
        if not self.no_feat:
            num_scales = len(pred_fake_A)
            num_layers = len(pred_fake_A[0])

            loss_G_feat = 0
            for scale in range(num_scales):
                for layer in range(num_layers):
                    loss_G_feat += nn.L1Loss()(pred_fake_A[scale][layer].mean(0), self.D_A.pred_real[scale][layer].mean(0).detach())
            loss_G += loss_G_feat * self.lambda_feat / num_scales / num_layers
            
            loss_G_feat = 0
            for scale in range(num_scales):
                for layer in range(num_layers):
                    loss_G_feat += nn.L1Loss()(pred_fake_B[scale][layer].mean(0), self.D_B.pred_real[scale][layer].mean(0).detach())
            loss_G += loss_G_feat * self.lambda_feat / num_scales / num_layers

        # if not self.no_vgg:
        #     loss_G_VGG = 0

        #     for mod in self.vgg_features:
        #         mod.mode = 'capture'
        #     self.vgg(real_A)
            
        #     for mod in self.vgg_features:
        #         mod.mode = 'loss'
        #     self.vgg(fake_A)

        #     for i,mod in enumerate(self.vgg_features):
        #         loss_G_VGG += mod.loss * self.lambda_vgg

        #     for mod in self.vgg_features:
        #         mod.mode = 'capture'
        #     self.vgg(real_B)
            
        #     for mod in self.vgg_features:
        #         mod.mode = 'loss'
        #     self.vgg(fake_B)

        #     for i,mod in enumerate(self.vgg_features):
        #         loss_G_VGG += mod.loss * self.lambda_vgg

        #     loss_G += loss_G_VGG

        loss_G.backward()
        return loss_G.item()

    def train(self, continue_train=False, data_path='maua/datasets/default_pix2pix',
        dataloader=None, start_epoch=1, num_epochs=100, epochs_decay=100, save_freq=10, log_freq=1,
        loadSize=256, fineSize=256, lr_policy='lambda', lr_decay_iters=50, num_workers=3,
        batch_size=16, resize=True, crop=True, vflip=True, hflip=True):
        """
        Training function for Pix2Pix object
        :param continue_train: whether to continue training or not
        :param data_path: path to folder containing images to train on
        :param dataloader: custom dataloader to use, otherwise images will be resized to max resolution
                           and normalized to ImageNet mean and standard deviation
        :param batch_size: number of images per batch
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
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        self.print_networks(verbose=True)

        if continue_train:
            start_epoch = self.get_latest_network(start_epoch, max_epoch=(num_epochs+epochs_decay))

        self.schedulers = [self.get_scheduler(optim, start_epoch, num_epochs, epochs_decay, lr_policy, lr_decay_iters)
                           for optim in self.optimizers]

        print("Starting training on "+str(self.device))

        # create dataloader
        if dataloader is None:
            dataloader = Pix2PixDataLoader(batch_size=batch_size, shuffle=True, num_workers=3,
                            data_path=data_path, direction=None, serial_batches=False,
                            resize=resize, crop=crop, vflip=vflip, hflip=hflip, input_nc=self.input_nc,
                            output_nc=self.output_nc, loadSize=loadSize, fineSize=fineSize)

        dataset_size = len(dataloader)
        total_batches = dataloader.batches()
        print('%d training images in %d batches' % (dataset_size, total_batches))
        start_time = time.time()

        for epoch in range(start_epoch, num_epochs + epochs_decay + 1):
            for i, data in enumerate(dataloader):
                real_A = data['A'].to(self.device)
                real_B = data['B'].to(self.device)
                fake_A, fake_B, rec_A, rec_B = self.forward(real_A, real_B)

                self.set_requires_grad([self.D_A, self.D_B], True)
                self.optimizer_D.zero_grad()
                loss_D = self.backward_D_A(fake_B, real_B)
                loss_D += self.backward_D_B(fake_A, real_A)
                self.optimizer_D.step()

                self.set_requires_grad([self.D_A, self.D_B], False)
                self.optimizer_G.zero_grad()        
                loss_G = self.backward_G(fake_A, fake_B, rec_A, rec_B, real_A, real_B)
                self.optimizer_G.step()

                if i % math.ceil(total_batches * log_freq) == 0 and not (i == 0 or i == total_batches):
                    img_file = os.path.join(self.save_dir, "images", "sample_%d_%d.png"%(epoch, i))

                    samples = th.stack([real_A[0],fake_B[0],rec_A[0],real_B[0],fake_A[0],rec_B[0]]).detach()
                    samples = (samples + 1) / 2.0
                    
                    save_image(samples, img_file, nrow=3)

                    elapsed = datetime.timedelta(seconds=time.time() - start_time)
                    print('Finished batch %d / %d \t [Elapsed: %s] \t loss_D: %.4f \t loss_G: %.4f'%
                          (i, total_batches, elapsed, loss_D, loss_G))

            if epoch % log_freq == 0:
                img_file = os.path.join(self.save_dir, "images", "sample_%d.png" % epoch)

                samples = th.stack([real_A[0],fake_B[0],rec_A[0],real_B[0],fake_A[0],rec_B[0]]).detach()
                samples = (samples + 1) / 2.0

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