# Immediate

- Integrate new and improved style transfer implementation
- Break out pix2pix/cyclegan/proGAN into submodules
	- refresh pix2pix/cyclegan implementation
		- add CUT?
	- deprecate proGAN
- Integrate stylegan2 implementation
- Add ml4a bindings

# Long Term

- Style Transfer
    - Tiling
    - Optic flow weighting / video style transfer
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
        - ~Relativistic hinge loss~
        - ~Ganstability loss (R1 regularization)~
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
- Gradient accumulation
- Options for 32, 16, or 8 bit float/int computation
