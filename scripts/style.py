from ..neural_style import MultiscaleStyle, NeuralStyle

content = 'maua/datasets/content/rose.jpg'
style = 'maua/datasets/style/yellow.jpg'

ns = NeuralStyle(
    content_image = content,
    style_images = style,
    output_image = "maua/output/styled_rose_basic.png",
    image_size = 512,
    num_iterations = 1200,
    normalize_gradients = False,
    save_iter = 400,
    print_iter = 400,
    gpu = 0,
    seed = 27
)
ns.run()
del ns # clear up VRAM for next model

ms = MultiscaleStyle(
    content_image = content,
    style_images = style,
    output_image = "maua/output/styled_rose_multiscale.png",
    start_size = 128,
    image_size = 512,
    steps = 3,
    num_iterations = 400,
    normalize_gradients = False,
    save_iter = 400,
    print_iter = 400,
    gpu = 0,
    seed = 27
)
ms.run()
