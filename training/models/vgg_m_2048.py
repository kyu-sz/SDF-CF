from collections import OrderedDict
from itertools import chain

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class VGG_M_2048(nn.Module):
    def __init__(self, model_path=None, model_url=None):
        super(VGG_M_2048, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 96, kernel_size=7, stride=2)),
            ('relu1', nn.ReLU()),
            ('norm1', nn.CrossMapLRN2d(5, alpha=1e-4, beta=0.75, k=2)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            ('conv2', nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1)),
            ('relu2', nn.ReLU()),
            ('norm2', nn.CrossMapLRN2d(5, alpha=1e-4, beta=0.75, k=2)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            ('conv3', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            ('relu5', nn.ReLU()),
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('fc6', nn.Conv2d(512, 4096, kernel_size=6, stride=1, padding=0)),
            ('relu6', nn.ReLU()),
            ('fc7', nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding=0)),
            ('relu7', nn.ReLU())]))

        self.fc8 = nn.Sequential(OrderedDict([
            ('fc8ext', nn.Conv2d(2048, 3624, kernel_size=1, stride=1, padding=0)),
            ('prob', nn.Softmax(dim=1))]))

        self.bbox_reg = nn.Conv2d(2048, 4, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.bbox_reg.weight, 0)
        nn.init.constant(self.bbox_reg.bias, 0)

        if model_path is not None:
            if model_path.endswith('mat'):
                mat = scipy.io.loadmat(model_path)
                mat_layers = list(mat['layers'])[0]

                # copy conv weights
                for i in range(len(mat_layers)):
                    if mat_layers[i]['type'] == 'conv':
                        for name, module in self.features.named_children():
                            if name == mat_layers[i]['name']:
                                weight, bias = mat_layers[i]['weights'].item()[0]
                                module.weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
                                module.bias.data = torch.from_numpy(bias[:, 0])
                                print('Weights of layer {} loaded!'.format(name))
                                break
            else:
                raise NotImplementedError
        elif model_url is not None:
            self.features.load_state_dict(model_zoo.load_url(model_url))

    def save(self, model_path):
        self.cpu()
        if model_path.endswith('mat'):
            model_dict = {'layers': []}
            for name, module in chain(self.features.named_children(),
                                      self.classifier.named_children(),
                                      self.bbox_reg.named_children()):
                layer = {'name': name,
                         'type': 'conv',
                         'weights': [module.weight.data.numpy(), module.bias.data.numpy()],
                         'size': [module.kernel_size[0],
                                  module.kernel_size[1],
                                  module.in_channels,
                                  module.out_channels],
                         'pad': [0, 0, 0, 0],
                         'stride': [module.stride[0], module.stride[0], module.stride[1], module.stride[1]],
                         'dilation': [module.dilation[0], module.dilation[1]]}
                model_dict['layers'].append(layer)
            scipy.io.savemat(model_path, mdict=model_dict)
        elif model_path.endswith('pth'):
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.cuda()

    def forward(self, x, output_layers):
        output_dict = {}
        num_outputs_left = len(output_layers)
        for name, module in self.features.named_children():
            x = module(x)
            if name in output_layers:
                if num_outputs_left > 1:
                    output_dict[name] = x.clone()
                    num_outputs_left -= 1
                else:
                    output_dict[name] = x
                    return output_dict

        if 'bbox_reg' in output_layers:
            output_dict['bbox_reg'] = self.bbox_reg(x)
            if num_outputs_left > 1:
                num_outputs_left -= 1
            else:
                return output_dict

        for name, module in self.classifier.named_children():
            x = module(x)
            if name in output_layers:
                if num_outputs_left > 1:
                    output_dict[name] = x.clone()
                    num_outputs_left -= 1
                else:
                    output_dict[name] = x
                    return output_dict
