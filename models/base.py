import os, fnmatch
from collections import OrderedDict
import torch as th

class BaseModel(th.nn.Module):
    def __init__(self, name, save_dir='maua/modelzoo', model_names=[], seed=-1, gpu=-1, auto_benchmark=True):
        """
        constructor for the class BaseModel
        :param name: name for model
        :param save_dir: directory to save model to
        :param model_names: names of models to load and save e.g. [G, D]
        :param gpu: index of device to train on, -1 for cpu
        :param seed: random seed
        :param auto_benchmark: whether to use cudnn benchmarking
        """
        
        super(BaseModel, self).__init__()
        self.name = name
        self.save_dir = save_dir
        self.model_names = model_names
        self.device = th.device("cpu")
        self.seed = seed
        if self.seed >= 0:
            th.manual_seed(self.seed)
            if gpu != -1:
                th.cuda.manual_seed(self.seed)
                th.backends.cudnn.deterministic = True
        if gpu != -1:
            self.device = th.device("cuda")
            th.backends.cudnn.benchmark = auto_benchmark
        self.to(self.device)
        self.gpu = gpu


    # save network to file
    def save_networks(self, epoch):
        for model_name in self.model_names:
            if isinstance(model_name, str):
                save_filename = '%s_%s_net_%s.pth' % (self.name, epoch, model_name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, model_name)
                if th.cuda.is_available() and not 'optim' in model_name:
                    th.save(net.module.cpu().state_dict(), save_path)
                    net.module.to(self.device)
                else:
                    th.save(net.state_dict(), save_path)


    # load network from file
    def load_networks(self, epoch):
        for model_name in self.model_names:
            if isinstance(epoch, str):
                load_filename = epoch
            else:
                load_filename = '%s_%s_net_%s.pth' % (self.name, epoch, model_name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, model_name)
            if isinstance(net, th.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            state_dict = th.load(load_path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            if state_dict.keys() != net.state_dict().keys():
                print('checkpoint state dictionaries are not identical, some parameters may not be initialized correctly')
            net.load_state_dict(state_dict, strict=False)


    # print network information
    def print_networks(self, verbose=False):
        print('---------- %s initialized -------------' % (self.name))
        for model_name in self.model_names:
            if isinstance(model_name, str) and not 'optim' in model_name:
                net = getattr(self, model_name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (model_name, num_params / 1e6))
        print('-----------------------------------------------')


    def get_latest_network(self, start_epoch, max_epoch):
        latest_epoch = None
        if start_epoch is 1:
            for e in range(max_epoch, -1, -1):
                for file in os.listdir(self.save_dir):
                    if fnmatch.fnmatch(file, '%s_%s_net_G.pth' % (self.name, e)):
                        latest_epoch = e
                        break
                else:
                    continue
                break
        else:
            latest_epoch = start_epoch
        self.load_networks(latest_epoch)
        self.print_networks()
        start_epoch = latest_epoch + 1
        return start_epoch


    # allows modular en/disabling of gradients
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
