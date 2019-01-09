import time, timeit, datetime, os, math, copy
from itertools import accumulate
import numpy as np
import torch as th
import torchvision as tv
from torchvision.utils import save_image
import torchvision.transforms as tn
from torch.nn.functional import interpolate
from .base import BaseModel
from ..dataloaders.base import BaseDataLoader
from ..networks.generators import ProGrowGenerator as Generator
from ..networks.discriminators import ProGrowDiscriminator as Discriminator


class ProGAN(BaseModel):
    """ Wrapper around the Generator and the Discriminator """

    def __init__(self, depth=7, latent_size=256, num_channels=3, learning_rate=1e-3, beta_1=0,
                 beta_2=0.99, eps=1e-8, drift=0.001, n_critic=1, use_eql=True, loss="wgan-gp",
                 use_ema=True, ema_decay=0.999, checkpoint=None, **kwargs):
        """
        constructor for the class ProGAN, extends BaseModel
        :param depth: depth of the GAN, 2^depth is the final size of generated images
        :param latent_size: latent size of the manifold used by the GAN
        :param num_channels: *NOT YET IMPLEMENTED* will control number of channels of in/outputs
        :param n_critic: number of times to update discriminator
                         (Used only if loss is wgan or wgan-gp)
        :param drift: drift penalty for the discriminator
                      (Used only if loss is wgan or wgan-gp)
        :param use_eql: whether to use equalized learning rate
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-sig", "hinge", "rel-avg", "r1-reg"]
                     Or an instance of GANLoss
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param checkpoint: generator checkpoint to load for inference
        :param learning_rate: base learning rate for Adam
        :param beta_1: beta_1 parameter for Adam
        :param beta_2: beta_2 parameter for Adam
        :param eps: epsilon parameter for Adam
        """
        from torch.optim import Adam
        from torch.nn import DataParallel
        import os

        super(ProGAN, self).__init__(**kwargs)

        # state of the object
        self.latent_size = latent_size
        self.num_channels = num_channels
        self.depth = depth - 1 # makes sure depth entered is equal to power of 2 of images generated
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.n_critic = n_critic
        self.use_eql = use_eql
        self.drift = drift

        # Create the Generator and the Discriminator
        self.G = Generator(self.depth, self.latent_size, use_eql=self.use_eql).to(self.device)
        self.D = Discriminator(self.depth, self.latent_size, use_eql=self.use_eql).to(self.device)

        # if code is to be run on GPU, we can use DataParallel:
        if self.device == th.device("cuda"):
            self.G = DataParallel(self.G)
            self.D = DataParallel(self.D)

        # define the optimizers for the discriminator and generator
        self.default_rate = learning_rate
        self.G_optim = Adam(self.G.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)
        self.D_optim = Adam(self.D.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

        # define the loss function used for training the GAN
        self.loss = self.setup_loss(loss)

        # setup the ema for the generator
        if self.use_ema:
            # create a shadow copy of the generator
            self.G_shadow = copy.deepcopy(self.G)

            # initialize the G_shadow weights equal to the weights of G
            self.update_average(self.G_shadow, self.G, beta=0)

        if checkpoint is not None:
            self.model_names = ['G']
            self.load_networks(checkpoint)
            self.set_requires_grad(self.G, requires_grad=False)


    def setup_loss(self, loss):
        if isinstance(loss, str):
            from ..networks.losses import GANLoss, WGAN_GP, LSGAN, LSGAN_SIGMOID, HingeLoss,
                                          RelativisticAverageHinge, R1Regularized
            loss = loss.lower()  # lowercase the string
            if loss == "wgan":
                loss = WGAN_GP(self.device, self.D, self.drift, use_gp=False)
                # note if you use just wgan, you will have to use weight clipping
                # in order to prevent gradient exploding
            elif loss == "wgan-gp":
                loss = WGAN_GP(self.device, self.D, self.drift, use_gp=True)
            elif loss == "lsgan":
                loss = LSGAN(self.D)
            elif loss == "lsgan-sig":
                loss = LSGAN_SIGMOID(self.D)
            elif loss == "hinge":
                loss = HingeLoss(self.D)
            elif loss == "rel-avg":
                loss = RelativisticAverageHingeLoss(self.D)
            elif loss == "r1-reg":
                loss = R1Regularized(self.D)
            else:
                raise ValueError("Unknown loss function requested")
        elif not isinstance(loss, GANLoss):
            raise ValueError("loss is neither an instance of GANLoss nor a string")
        return loss

    
    # This function updates the exponential average weights based on the current training
    def update_average(self, model_tgt, model_src, beta):
        """
        update the target model using exponential moving averages
        :param model_src: target model
        :param model_src: source model
        :param beta: value of decay beta
        :return: None (updates the target model)
        """

        # utility function for toggling the gradient requirements of the models
        def toggle_grad(model, requires_grad):
            for p in model.parameters():
                p.requires_grad_(requires_grad)

        # turn off gradient calculation
        toggle_grad(model_tgt, False)
        toggle_grad(model_src, False)

        param_dict_src = dict(model_src.named_parameters())

        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

        # turn back on the gradient calculation
        toggle_grad(model_tgt, True)
        toggle_grad(model_src, True)


    def generate_prescaled_dataset(self, dataloader, sizes):
        print("Generating prescaled dataset...")
        self.prescaled_data = True
        data_path = 'maua/datasets/%s_prescaled'%self.name
        if not len(dataloader)*len(sizes) == len(BaseDataLoader(data_path,tn.ToTensor())):
            # create a copy of the dataset on disk for each size
            from pathos.multiprocessing import ProcessingPool as Pool
            pool = Pool(len(sizes))

            def prescale_dataset(size):
                os.makedirs(data_path+"/%s"%size, exist_ok=True)
                transforms = tn.Compose([dataloader.transforms, tn.Resize(size), tn.ToTensor()])
                t_data = BaseDataLoader(dataloader.data_path, transforms=transforms, batch_size=1)
                for i, sample in enumerate(t_data, 1):
                    sample = th.clamp(sample, min=0, max=1)
                    save_image(sample, data_path+"/%s/%s.png"%(size,i))
                return len(os.listdir(data_path+"/%s"%size))

            results = pool.map(prescale_dataset, sizes)
            pool.close()
            pool.join()
            assert sum(results) == len(dataloader)*len(sizes)
        dataloader.data_path = data_path


    def forward(self, real_A):
        return self.G(real_A, self.depth-1, alpha=1)


    def optimize_D(self, noise, real_batch, depth, alpha):
        from torch.nn import AvgPool2d
        from ..networks.losses import R1Regularized
        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1)) if not self.prescaled_data else 1
        prior_downsample_factor = max(int(np.power(2, self.depth - depth)), 0) if not self.prescaled_data else 2

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch), scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        loss_val = 0
        for _ in range(self.n_critic):
            # generate a batch of samples
            fake_samples = self.G(noise, depth, alpha).detach()

            loss = self.loss.loss_D(real_samples, fake_samples, depth=depth, alpha=alpha)

            # optimize discriminator
            self.D_optim.zero_grad()

            # TODO add WGAN regularization and self.regularize param
            if type(self.loss, R1Regularized):
                loss.backward(retain_graph=True)
                self.loss.reg.backward()
            else:
                loss.backward()

            self.D_optim.step()

            loss_val += loss.item()

        return loss_val / self.n_critic


    def optimize_G(self, noise, depth, alpha):
        # generate fake samples:
        fake_samples = self.G(noise, depth, alpha)

        loss = self.loss.loss_G(None, fake_samples, depth=depth, alpha=alpha)

        # optimize the generator
        self.G_optim.zero_grad()
        loss.backward()
        self.G_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.update_average(self.G_shadow, self.G, self.ema_decay)

        # return the loss value
        return loss.item()


    def train(self, continue_train=False, data_path='maua/datasets/default_progan',
        dataloader=None, start_epoch=1, start_depth=1, fade_in=0.5, save_freq=10, log_freq=5,
        epochs_dict={8: 50, 16: 50, 32: 75, 64: 75, 128: 100, 256: 100, 512: 150, 1024: 150},
        batches_dict={8: 512, 16: 128, 32: 48, 64: 24, 128: 12, 256: 6, 512: 3, 1024: 1},
        learning_rates_dict={256: 5e-4, 512: 2.5e-4, 1024: 1e-4}, prescaled_data=True):
        """
        Training function for ProGAN object
        :param continue_train: whether to continue training or not
        :param data_path: path to folder containing images to train on
        :param dataloader: custom dataloader to use, otherwise images will be resized to max resolution
                           and normalized to ImageNet mean and standard deviation
        :param start_epoch: epoch to continue training from (defaults to most recent, if continuing training)
        :param fade_in: fraction of epochs per depth to fade into the new resolution
        :param save_freq: frequency to save checkpoints in number of epochs
        :param log_freq: frequency to log / save images in number of or fraction of epochs
        :param epochs_dict: dictionary of number of epochs to train per resolution
        :param batches_dict: dictionary of batch sizes per resolution
        :param learning_rates_dict: dictionary of learning rates per resolution (defaults to self.learning_rate)
        :param prescaled_data: whether dataset should be pre-resized to each size on disk.
                               trades large performance boost for smaller sizes for more disk space.
        """
        self.model_names = ["G", "D"]
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        start_epoch = epoch = 1
        num_epochs = sum(list(epochs_dict.values())[:self.depth])

        if continue_train:
            start_epoch = self.get_latest_network(start_epoch, max_epoch=num_epochs)
            start_depth = next((i for i,v in enumerate(accumulate(epochs_dict.values())) if v > start_epoch), -1) + 1
            epoch = start_epoch
            start_epoch -= sum(list(epochs_dict.values())[:start_depth-1])

        # create fixed_input for debugging
        fixed_input = th.randn(12, self.latent_size).to(self.device)

        # create dataloader
        if dataloader is None:
            transforms = tv.transforms.Compose([tn.Resize(2**(self.depth + 1)),
                                                tn.ToTensor()])
            dataloader = BaseDataLoader(data_path, transforms=transforms, batch_size=1)
        if prescaled_data: self.generate_prescaled_dataset(dataloader, sizes=epochs_dict.keys())

        dataset_size = len(dataloader)
        print('# training images = %d' % dataset_size)

        print("Starting training on "+str(self.device))
        # create a global time counter
        global_time = time.time()
        for depth in range(start_depth, self.depth):
            current_res = 2**(depth + 2)
            print("Current resolution: %d x %d" % (current_res, current_res))

            if self.prescaled_data:
                from ..dataloaders import ImageFolder
                dataloader.dataset = ImageFolder(data_path=dataloader.data_path+"/%s"%current_res, transform=tn.ToTensor())
            dataloader.set_batch_size(batches_dict[current_res])
            total_batches = dataloader.batches()

            learning_rate = learning_rates_dict.get(current_res, self.default_rate)
            self.D_optim.lr = self.G_optim.lr = learning_rate

            for e in range(start_epoch if depth == start_depth else 1, epochs_dict[current_res] + 1):
                start = time.time()
                
                # calculate the value of alpha for fade-in effect
                alpha = min(e / (epochs_dict[current_res] * fade_in), 1)
                if log_freq < 1: print("Start of epoch: %s / %s \t Fade in: %s"%(epoch, num_epochs, alpha))

                loss_D, loss_G = 0, 0
                # iterate over the dataset in batches:
                for i, batch in enumerate(dataloader, 1):
                    images = batch.to(self.device)
                    noise = th.randn(images.shape[0], self.latent_size).to(self.device)

                    loss_D += self.optimize_D(noise, images, depth, alpha)
                    loss_G += self.optimize_G(noise, depth, alpha)

                    # provide feedback
                    if i % math.ceil(total_batches * log_freq) == 0 and not (i == 0 or i == total_batches):
                        elapsed = str(datetime.timedelta(seconds=time.time() - global_time))
                        print("Elapsed: [%s] Batch: %d / %d d_loss: %f  g_loss: %f" %
                                (elapsed, i, total_batches, loss_D / math.ceil(total_batches*log_freq),
                                loss_G / math.ceil(total_batches*log_freq)))
                        loss_D, loss_G = 0, 0

                        # create a grid of samples and save it
                        gen_img_file = os.path.join(self.save_dir, "images", "sample_res%d_e%d_b%d" %
                                                    (current_res, epoch, i) + ".png")
                        with th.no_grad():
                            self.create_grid(
                                samples=self.G(fixed_input, depth, alpha),
                                scale_factor=int(np.power(2, self.depth - depth - 2)),
                                img_file=gen_img_file,
                            )

                if log_freq < 1: print("End of epoch:", epoch, "Took: ", time.time() - start, "sec")

                if log_freq >= 1 and epoch % log_freq == 0 or epoch == num_epochs:
                    elapsed = str(datetime.timedelta(seconds=time.time() - global_time))
                    print("Elapsed: [%s] Epoch: %d / %d Fade in: %.02f d_loss: %f  g_loss: %f" %
                          (elapsed, epoch, num_epochs, alpha, loss_D, loss_G))
                    # create a grid of samples and save it
                    gen_img_file = os.path.join(self.save_dir, "images", "sample_res%d_e%d" %
                                                (current_res, epoch) + ".png")
                    with th.no_grad():
                        self.create_grid(
                            samples=self.G(fixed_input, depth, alpha),
                            scale_factor=int(np.power(2, self.depth - depth)/4),
                            img_file=gen_img_file,
                        )

                if epoch % save_freq == 0 or epoch == num_epochs:
                    self.save_networks(epoch)

                epoch += 1

        print("Training finished, took: ", datetime.timedelta(seconds=time.time() - global_time))
        self.save_networks("final")


    # used to create grid of training images
    def create_grid(self, samples, scale_factor, img_file, real_imgs=False):
        samples = th.clamp(samples, min=0, max=1)

        # upsample the image
        if scale_factor > 1 and not real_imgs:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))+1))

