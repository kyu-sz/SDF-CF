from collections import OrderedDict

import scipy.io
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import numpy as np


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x ** 2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq, pad, pad, pad, pad), 2),
                            torch.cat((pad, x_sq, pad, pad, pad), 2),
                            torch.cat((pad, pad, x_sq, pad, pad), 2),
                            torch.cat((pad, pad, pad, x_sq, pad), 2),
                            torch.cat((pad, pad, pad, pad, x_sq), 2)), 1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:, 2:-2, :, :]
        x = x / ((2. + 0.0001 * x_sumsq) ** 0.75)
        return x


class VGGM(nn.Module):
    def __init__(self, model_path=None, model_url=None):
        super(VGGM, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    nn.CrossMapLRN2d(5, alpha=5e-4, beta=0.0001, k=2),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),
                                    nn.CrossMapLRN2d(5, alpha=5e-4, beta=0.0001, k=2),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU()))]))

        if model_path is not None:
            if model_path.endswith('pth'):
                states = torch.load(model_path)
                shared_layers = states['shared_layers']
                self.layers.load_state_dict(shared_layers, strict=False)
            elif model_path.endswith('mat'):
                mat = scipy.io.loadmat(model_path)
                mat_layers = list(mat['layers'])[0]

                # copy conv weights
                for i in range(3):
                    weight, bias = mat_layers[i * 4]['weights'].item()[0]
                    self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
                    self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])
        elif model_url is not None:
            self.layers.load_state_dict(model_zoo.load_url(model_url))

    def forward(self, x):
        for name, module in self.layers.named_children():
            x = module(x)
            if name == 'conv3':
                x = x.view(x.size(0), -1)
        return x
