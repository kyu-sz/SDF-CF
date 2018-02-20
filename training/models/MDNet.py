from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable


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


class MDNet(nn.Module):
    def __init__(self, model_path=None):
        super(MDNet, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU()))]))

        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers, strict=False)

    def forward(self, x):
        for name, module in self.layers.named_children():
            x = module(x)
            if name == 'conv3':
                x = x.view(x.size(0), -1)
        return x
