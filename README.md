# Maua
## v0.1.1

Maua is a framework that allows for easy training of and interoperability between multiple state of the art image processing networks. Maua provides a modular object oriented API that allows uses ranging from piecing together low level components into a new architecture to training a GAN with a single command.

Those with little experience can use Maua without needing to know all of the details of what's running under the hood (just point at a folder of images and train), while those confident in their skills can dive deep into the building blocks of the various networks and put them together in new and interesting ways.

Maua seeks to integrate many different approaches under a single umbrella to streamline pipelines which would otherwise span multiple frameworks/languages/repositories/etc. with manual intermediate steps into a couple short and sweet python scripts. This way it is easier to push the creative boundaries of machine learning approaches without having to spend so much time installing and figuring out new, badly documented research repositories.

NOTE: Maua is still at an early stage of development, weird bugs/behaviors/expectations still lurk in the code that will slowly be ironed out as the framework matures.


## Instalation

Maua requires Python 3.7 and PyTorch 1.0 (which can be installed [here](https://pytorch.org/get-started/locally/)). Training neural networks takes a lot of compute so most of the included networks require a decent GPU to get results in a reasonable amount of time.

Furthermore Maua relies on [Pathos](https://github.com/uqfoundation/pathos)(`pip install pathos`) for multithreading.

There are also some optional dependencies required for running a GLWindow with network outputs, namely [PyCuda](https://wiki.tiker.net/PyCuda) and [Glumpy](https://glumpy.github.io/). These can be installed with pip or conda.

After installing the needed dependencies this repository can be cloned and you can get to work!


## Usage

The best way to use Maua at the moment is by creating a script in the `scripts/` folder. Then it can be run from command line (from the folder with the repository in it) with `python3 -m maua.scripts.my_script`.

There are some example scripts included that show how to train and test all the currently implemented networks.

Pretrained example checkpoints and the datasets they were trained on can be downloaded from [here](https://drive.google.com/open?id=1ZuGi6o2cxvgeu3M5NotmfI7kWB-oxulL), they should be placed in the folders with the corresponding name in the repository. The datasets are based on the [102 Category Oxford Flowers Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/). The images have been cropped to squares and prepared in the folder structure that Maua expects. You can replicate any checkpoint by simply running its respective training script.

Here are some results from Maua v0.1:

### ProGAN
![Latent Bouquet](output/progan_samples.jpg?raw=true "Latent Bouquet")

![Training FlowerGAN](output/progan_train.gif?raw=true "Training FlowerGAN")

Training took about 12 hours on a GTX 1080 Ti.

### Pix2Pix
![Edges2Flowers](output/pix2pix_samples.jpg?raw=true "Edges2Flowers")

Training took about 10 hours on a GTX 1080 Ti.

From left to right: the input to the network, the result for pix2pix without multiscale GAN, pix2pix with no_vgg=False, and pix2pix with no_vgg=True.

The implementation of pix2pixHD is still a work in progress. Expect an improvement in results when using the VGG feature loss in future versions.

### Style Transfer
![Stylish Rose](output/style_transfer_samples.jpg?raw=true "Stylish Rose")

From left to right: the content image, style image, output using NeuralStyle, and MultiscaleStyle.

Also included is a script, voltaStyle.py, that uses progressively smaller ImageNet classifiers to achieve up to 36 megapixel style transfer images (6000x6000 px) on a graphics card with 11 GB VRAM.

Note that using NeuralStyle without specifying a model_file will download the respective model to your modelzoo.

Normalizing gradients is not yet fully functioning, this feature should be left False for the time being.


## Roadmap

Below is a list of planned features and improvements:

- Style Transfer
    - Tiling
    - Optic flow weighting
    - Segmentation maps
	- Deep painterly harmonization
- Pix2Pix
    - Easy frame prediction models
        - Frechet video distance loss (pretrained I3D feature loss)
        - Scheduled sampling
    - Segmentation & instance maps
- ProGAN
    - Conditional ProGAN
	- Self attention loss
	- Relativistic hinge loss
	- Ganstability loss
- Video / GLWindow
    - Video writer
    - MIDI/keyboard interactivity
    - Music responsiveness
    - GL filters & effects
- More Networks
    - CycleGAN
    - CPPNs
    - DeepDream / Lucid
    - FAMos
	- SAVP
	- SRGAN
	- vid2vid / RecycleGAN
- YAML Config System
- Gradient Checkpointing
- Options for 32, 16, or 8 bit float/int computation


## Acknowledgments

Maua contains partial/full implementations of and/or uses code from the papers and repositories listed below:

ProGAN: [paper](https://arxiv.org/abs/1710.10196), [code](https://github.com/akanimax/pro_gan_pytorch)

Pix2Pix: [paper](https://arxiv.org/abs/1611.07004), [paper](https://arxiv.org/abs/1711.11585), [code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

Neural Style: [paper](https://arxiv.org/abs/1508.06576), [code](https://github.com/ProGamerGov/neural-style-pt)

GLWindow: [code](https://gist.github.com/victor-shepardson/5b3d3087dc2b4817b9bffdb8e87a57c4)


## License

This repository is licensed under the GPL v3 license, you can find more information in LICENSE and gpl3.txt

The pretrained network checkpoints and their datasets are not for commercial use, they are only included to illustrate functionality and verify the included example scripts.
