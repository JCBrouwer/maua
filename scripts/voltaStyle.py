import os, sys
from maua.neural_style import NeuralStyle

# based on volta-x3 script by /u/vic8760 on Reddit
# https://www.reddit.com/r/deepdream/comments/954h2w/voltax3_script_release_the_best_hq_neuralstyle/

content = sys.argv[1]
content_name, _ = os.path.splitext(os.path.basename(content))

style = sys.argv[2]
style_names = list(map(lambda s: os.path.splitext(os.path.basename(s))[0], style.split(',')))

output = 'maua/output/%s_%s.png'%(content_name, "_".join(style_names))

style_factor = 1

ns = NeuralStyle(
    content_image = content,
    style_images = style,
    image_size = 256,
    num_iterations = 1200,
    style_weight = style_factor * 50,
    content_weight = 5,
    tv_weight = 1e-3,
    normalize_gradients = False,
    save_iter = 0,
    print_iter = 0,
    gpu = 0,
    seed = 27,
    output_image = output.replace(".png","1.png")
)
img = ns.run()

img = ns.run(
    init_image = img,
    image_size = 512,
    num_iterations = 800,
    style_weight = style_factor * 1000,
    content_weight = 1,
    tv_weight = 1e-4,
    output_image = output.replace(".png","2.png")
)

img = ns.run(
    init_image = img,
    image_size = 724,
    style_weight = style_factor * 2000,
    num_iterations = 400,
    output_image = output.replace(".png","3.png")
)

img = ns.run(
    init_image = img,
    image_size = 1024,
    num_iterations = 200,
    style_weight = style_factor * 500,
    model_type = 'nyud',
    style_layers = 'relu1_2,relu2_2,relu3_3,relu4_3,relu5_3,relu6_1',
    output_image = output.replace(".png","4.png")
)

img = ns.run(
    init_image = img,
    image_size = 1448,
    tv_weight = 1e-5,
    style_layers = 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1',
    output_image = output.replace(".png","5.png")
)

img = ns.run(
    init_image = img,
    tv_weight = 1e-4,
    style_weight = 5,
    image_size = 2000,
    num_iterations = 50,
    model_type = 'prune',
    style_layers = 'relu1_2,relu2_2,relu3_3,relu4_3,relu5_3',
    output_image = output.replace(".png","6.png")
)

img = ns.run(
    init_image = img,
    image_size = 3000,
    optimizer = 'adam',
    style_layers = 'relu1_1,relu2_1,relu3_1,relu4_1',
    output_image = output.replace(".png","7.png")
)

img = ns.run(
    init_image = img,
    image_size = 4000,
    num_iterations = 10,
    content_weight = 0,
    style_weight = 10,
    model_type = 'nin',
    style_layers = 'relu1_1,relu2_2,relu3_1,relu3_3,relu4_1,relu4_3,relu5_2',
    output_image = output.replace(".png","8.png")
)

img = ns.run(
    init_image = img,
    image_size = 5000,
    output_image = output.replace(".png","9.png")
)

ns.run(
    init_image = img,
    image_size = 5300,
    style_layers = 'relu1_1,relu2_1,relu3_1',
    output_image = output
)