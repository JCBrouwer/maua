import torch, os, copy
import torch.nn as nn
from torch.utils.model_zoo import load_url
import urllib.request
from ..networks.losses import ContentLoss, StyleLoss, TVLoss

# TODO make this an abstract factory
# TODO allow multiple TVLosses at different layers
# TODO readd build_sequential to clean up this double checkpoint download

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()

    def insert_losses(self, content_layers, style_layers, tv_weight, pooling, normalize_gradients=False,
                      layer_depth_weighting=False, content_weight=1.0, style_weight=1.0):
        content_layers = content_layers.split(',')
        style_layers = style_layers.split(',')

        # Set up the self.network, inserting style and content loss modules
        cnn = copy.deepcopy(self.features)
        self.content_losses, self.style_losses, self.tv_losses = [], [], []
        next_content_idx, next_style_idx = 1, 1
        self.net = nn.Sequential()
        c, r, p, n = 0, 0, 0, 0
        if tv_weight > 0:
            tv_mod = TVLoss(tv_weight)
            self.net.add_module('tv_loss', tv_mod)
            self.tv_losses.append(tv_mod)

        for i, layer in enumerate(list(cnn), 1):
            if next_content_idx <= len(content_layers) or next_style_idx <= len(style_layers):
                if isinstance(layer, nn.Conv2d):
                    self.net.add_module(str(n), layer)
                    n+=1

                    if self.layer_list['C'][c] in content_layers:
                        print("Setting up content layer " + str(i) + ": " + str(self.layer_list['C'][c]))
                        loss_module = ContentLoss(strength=content_weight*(2**(c-5) if layer_depth_weighting else content_weight),
                                                  normalize=normalize_gradients)
                        self.net.add_module('c_'+self.layer_list['C'][c], loss_module)
                        self.content_losses.append(loss_module)

                    if self.layer_list['C'][c] in style_layers:
                        print("Setting up style layer " + str(i) + ": " + str(self.layer_list['C'][c]))
                        loss_module = StyleLoss(strength=style_weight*(2**(c-5) if layer_depth_weighting else style_weight),
                                                  normalize=normalize_gradients)
                        self.net.add_module('s_'+self.layer_list['C'][c], loss_module)
                        self.style_losses.append(loss_module)
                    c+=1

                if isinstance(layer, nn.ReLU):
                    self.net.add_module(str(n), layer)
                    n+=1

                    if self.layer_list['R'][r] in content_layers:
                        print("Setting up content layer " + str(i) + ": " + str(self.layer_list['R'][r]))
                        loss_module = ContentLoss(strength=content_weight*(2**(c-5) if layer_depth_weighting else content_weight),
                                                  normalize=normalize_gradients)
                        self.net.add_module('c_'+self.layer_list['R'][r], loss_module)
                        self.content_losses.append(loss_module)
                        next_content_idx += 1

                    if self.layer_list['R'][r] in style_layers:
                        print("Setting up style layer " + str(i) + ": " + str(self.layer_list['R'][r]))
                        loss_module = StyleLoss(strength=style_weight*(2**(c-5) if layer_depth_weighting else style_weight),
                                                normalize=normalize_gradients)
                        self.net.add_module('s_'+self.layer_list['R'][r], loss_module)
                        self.style_losses.append(loss_module)
                        next_style_idx += 1
                    r+=1

                if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                    if pooling == 'avg':
                        self.net.add_module(str(n), nn.AvgPool2d((3, 3),(2, 2),(0, 0), ceil_mode=True))
                    if pooling == 'max':
                        self.net.add_module(str(n), nn.AvgPool2d((3, 3),(2, 2),(0, 0), ceil_mode=True))
                    n+=1
                    p+=1


class VGG(ImageNet):
    def __init__(self, model_file=None, layer_num=19, pooling='max', gpu=-1, tv_weight=0,
                 content_layers="", style_layers="", num_classes=1000, layer_depth_weighting=False,
                 content_weight=1.0, style_weight=1.0, normalize_gradients=False):
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
            self.insert_losses(content_layers=content_layers, style_layers=style_layers, tv_weight=tv_weight,
                               content_weight=content_weight, style_weight=style_weight, pooling=pooling,
                               normalize_gradients=normalize_gradients, layer_depth_weighting=layer_depth_weighting)

        if model_file is not None:
            caffe_net = torch.load(model_file)
            new_dict = {k.replace("features.",""):v for k,v in caffe_net.items() if "features" in k}
            self.net.load_state_dict(new_dict, strict=False)
            print("Successfully loaded " + str(model_file))

        if gpu > -1:
            self.net = self.net.cuda()


class NIN(ImageNet):
    def __init__(self, model_file='maua/modelzoo/nin_imagenet.pth', pooling='max', gpu=-1,
                 tv_weight=0, content_layers="", style_layers="", layer_depth_weighting=False,
                 content_weight=1.0, style_weight=1.0, normalize_gradients=False):
        super(NIN, self).__init__()

        self.layer_list = {
            'C': ['conv1_1', 'conv1_2', 'conv1_3', 'conv2_1', 'conv2_2', 'conv2_3', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3'],
            'R': ['relu1_1', 'relu1_2', 'relu1_3', 'relu2_1', 'relu2_2', 'relu2_3', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3'],
            'P': ['pool1', 'pool2', 'pool3', 'pool4']
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
            nn.Softmax()
        )

        if not os.path.isfile(model_file):
            print("Model file not found. Downloading...")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", model_file)
        self.load_state_dict(torch.load(model_file), False)
        print("Successfully loaded " + str(model_file))

        if not (tv_weight==0 and content_layers=="" and style_layers==""):
            self.insert_losses(content_layers=content_layers, style_layers=style_layers, tv_weight=tv_weight,
                               content_weight=content_weight, style_weight=style_weight, pooling=pooling,
                               normalize_gradients=normalize_gradients, layer_depth_weighting=layer_depth_weighting)

        if gpu > -1:
            self.net = self.net.cuda()


class ChannelPruning(ImageNet):
    def __init__(self, model_file='maua/modelzoo/channel_pruning.pth', pooling='max', gpu=-1,
                 tv_weight=0, content_layers="", style_layers="", layer_depth_weighting=False,
                 content_weight=1.0, style_weight=1.0, normalize_gradients=False):
        super(ChannelPruning, self).__init__()

        self.layer_list = {
            'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'],
            'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_2'],
            'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']
        }

        self.features = nn.Sequential(
            nn.Conv2d(3,24,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(24,22,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(22,41,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(41,51,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(51,108,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(108,89,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(89,111,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(111,184,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(184,276,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(276,228,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(228,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Softmax()
        )

        if not os.path.isfile(model_file):
            print("Channel Pruning model file not found.")
            exit()
        self.features.load_state_dict(torch.load(model_file), False)
        print("Successfully loaded " + str(model_file))

        if not (tv_weight==0 and content_layers=="" and style_layers==""):
            self.insert_losses(content_layers=content_layers, style_layers=style_layers, tv_weight=tv_weight,
                               content_weight=content_weight, style_weight=style_weight, pooling=pooling,
                               normalize_gradients=normalize_gradients, layer_depth_weighting=layer_depth_weighting)

        if gpu > -1:
            self.net = self.net.cuda()


class NyudFcn32s(ImageNet):
    def __init__(self, model_file='maua/modelzoo/nyud_fcn32s.pth', pooling='max', gpu=-1,
                 tv_weight=0, content_layers="", style_layers="", layer_depth_weighting=False,
                 content_weight=1.0, style_weight=1.0, normalize_gradients=False):
        super(NyudFcn32s, self).__init__()

        self.layer_list = {
            'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'conv6_1', 'conv6_2', 'conv6_3'],
            'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3', 'relu6_1', 'relu6_2', 'relu6_3'],
            'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']
        }

        self.features = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(1, 1),(100, 100)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(512,4096,(7, 7)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(4096,4096,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(4096,40,(1, 1)),
            nn.ReLU(inplace=True)
        )

        if not os.path.isfile(model_file):
            print("Nyud-Fcn32s model file not found.")
            exit()
        self.features.load_state_dict(torch.load(model_file))
        print("Successfully loaded " + str(model_file))

        if not (tv_weight==0 and content_layers=="" and style_layers==""):
            self.insert_losses(content_layers=content_layers, style_layers=style_layers, tv_weight=tv_weight,
                               content_weight=content_weight, style_weight=style_weight, pooling=pooling,
                               normalize_gradients=normalize_gradients, layer_depth_weighting=layer_depth_weighting)

        if gpu > -1:
            self.net = self.net.cuda()