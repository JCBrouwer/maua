from ..neural_style import MultiscaleStyle, NeuralStyle

content = 'maua/datasets/content/rose.jpg'
style = 'maua/datasets/style/yellow.jpg'

ns = NeuralStyle(
    image_size = 512,
    num_iterations = 1200,
    save_iter = 400,
    print_iter = 400,
    gpu = 0,
    seed = 27
)
ns.run(
    content = content,
    style = style,
    output="maua/output/styled_rose_basic.png"
)
del ns # clear up VRAM for next model

ms = MultiscaleStyle(
    start_size = 128,
    image_size = 512,
    steps = 3,
    num_iterations = 400,
    style_weight = 50,
    save_iter = 400,
    print_iter = 400,
    gpu = 0,
    seed = 27
)
ms.run(
    content = content,
    style = style,
    output = "maua/output/styled_rose_multiscale.png"
)
