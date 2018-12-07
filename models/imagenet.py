import torch, os, copy
import torch.nn as nn
from torch.utils.model_zoo import load_url
import urllib.request
from ..networks.losses import ContentLoss, StyleLoss, TVLoss

# TODO make this an abstract factory?
# TODO allow multiple TVLosses at different layers?

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()

    def insert_losses(self, content_layers, style_layers, tv_weight, pooling, layer_depth_weighting=False):
        content_layers = content_layers.split(',')
        style_layers = style_layers.split(',')

        # Set up the self.network, inserting style and content loss modules
        cnn = copy.deepcopy(self.features)
        self.content_losses, self.style_losses, self.tv_losses = [], [], []
        next_content_idx, next_style_idx = 1, 1
        self.net = nn.Sequential()
        c, r, p, n = 0, 0, 0, 0
        if tv_weight > 0:
            tv_mod = TVLoss()
            self.net.add_module('tv_loss', tv_mod)
            self.tv_losses.append(tv_mod)

        for i, layer in enumerate(list(cnn), 1):
            if next_content_idx <= len(content_layers) or next_style_idx <= len(style_layers):
                if isinstance(layer, nn.Conv2d):
                    self.net.add_module(str(n), layer)
                    n+=1

                    if self.layer_list['C'][c] in content_layers:
                        print("Setting up content layer " + str(i) + ": " + str(self.layer_list['C'][c]))
                        loss_module = ContentLoss(strength=2**(c-5) if layer_depth_weighting else 1)
                        self.net.add_module('c_'+self.layer_list['C'][c], loss_module)
                        self.content_losses.append(loss_module)

                    if self.layer_list['C'][c] in style_layers:
                        print("Setting up style layer " + str(i) + ": " + str(self.layer_list['C'][c]))
                        loss_module = StyleLoss(strength=2**(c-5) if layer_depth_weighting else 1)
                        self.net.add_module('s_'+self.layer_list['C'][c], loss_module)
                        self.style_losses.append(loss_module)
                    c+=1

                if isinstance(layer, nn.ReLU):
                    self.net.add_module(str(n), layer)
                    n+=1

                    if self.layer_list['R'][r] in content_layers:
                        print("Setting up content layer " + str(i) + ": " + str(self.layer_list['R'][r]))
                        loss_module = ContentLoss(strength=2**(r-5) if layer_depth_weighting else 1)
                        self.net.add_module('c_'+self.layer_list['R'][r], loss_module)
                        self.content_losses.append(loss_module)
                        next_content_idx += 1

                    if self.layer_list['R'][r] in style_layers:
                        print("Setting up style layer " + str(i) + ": " + str(self.layer_list['R'][r]))
                        loss_module = StyleLoss(strength=2**(r-5) if layer_depth_weighting else 1)
                        self.net.add_module('s_'+self.layer_list['R'][r], loss_module)
                        self.style_losses.append(loss_module)
                        next_style_idx += 1
                    r+=1

                if isinstance(layer, nn.MaxPool2d):
                    if pooling == 'avg':
                        self.net.add_module(str(n), nn.AvgPool2d((3, 3),(2, 2),(0, 0), ceil_mode=True))
                    else:
                        self.net.add_module(str(n), layer)
                    n+=1
                    p+=1


# TODO readd build_sequential to clean up this double checkpoint download?
class VGG(ImageNet):
    def __init__(self, model_file=None, layer_num=19, pooling='max', gpu=-1, tv_weight=0,
                 content_layers="", style_layers="", num_classes=1000, layer_depth_weighting=False):
        super(VGG, self).__init__()

        if layer_num == 19:
            if model_file is not None:
                if not os.path.exists(model_file):
                    sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth")
                    torch.save(sd, model_file)
            from torchvision.models import vgg19
            self.features = vgg19(pretrained=True).features
            self.layer_list = {
                'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'],
                'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4', 'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4', 'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4'],
                'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
            }
        elif layer_num == 16:
            if model_file is not None:
                if not os.path.exists(model_file):
                    sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
                    torch.save(sd, model_file)
            from torchvision.models import vgg16
            self.layer_list = {
                'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'],
                'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3'],
                'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
            }
            self.features = vgg16(pretrained=True).features
        else:
            print("vgg%s not available, options [vgg19, vgg16]"%(layer_num))

        if not (tv_weight==0 and content_layers=="" and style_layers==""):
            self.insert_losses(content_layers, style_layers, tv_weight, pooling, layer_depth_weighting)

        if model_file is not None:
            caffe_net = torch.load(model_file)
            new_dict = {k.replace("features.",""):v for k,v in caffe_net.items() if "features" in k}
            self.net.load_state_dict(new_dict, strict=False)
            print("Successfully loaded " + str(model_file))

        if gpu > -1:
            self.net = self.net.cuda()


# TODO check if this works in neural style / pix2pix without utils.preprocess() input first
class NIN(ImageNet):
    def __init__(self, model_file='maua/modelzoo/nin_imagenet.pth', pooling='max', gpu=-1,
                 tv_weight=0, content_layers="", style_layers="", layer_depth_weighting=False):
        super(NIN, self).__init__()

        self.layer_list = {
            'C': ['conv1', 'cccp1', 'cccp2', 'conv2', 'cccp3', 'cccp4', 'conv3', 'cccp5', 'cccp6', 'conv4-1024', 'cccp7-1024', 'cccp8-1024'],
            'R': ['relu0', 'relu1', 'relu2', 'relu3', 'relu5', 'relu6', 'relu7', 'relu8', 'relu9', 'relu10', 'relu11', 'relu12'],
            'P': ['pool1', 'pool2', 'pool3', 'pool4'],
            'D': ['drop'],
        }

        if pooling == 'max':
            pool2d = nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        elif pooling == 'avg':
            pool2d = nn.AvgPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)

        self.features = nn.Sequential(
            nn.Conv2d(3,96,(11, 11),(4, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,(1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Conv2d(96,256,(5, 5),(1, 1),(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,(1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,(1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Dropout(0.5),
            nn.Conv2d(384,1024,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1024,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1000,(1, 1)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((6, 6),(1, 1),(0, 0),ceil_mode=True),
            nn.Softmax(),
        )
        if not os.path.isfile(model_file):
            print("Model file not found. Downloading...")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", model_file)
        self.load_state_dict(torch.load(model_file), False)
        print("Successfully loaded " + str(model_file))

        if not (tv_weight==0 and content_layers=="" and style_layers==""):
            self.insert_losses(content_layers, style_layers, tv_weight, pooling, layer_depth_weighting)

        if gpu > -1:
            self.net = self.net.cuda()