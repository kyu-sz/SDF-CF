from collections import OrderedDict
from itertools import chain

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class VGG_M_2048(nn.Module):
    INPUT_MEAN = [123.6591, 116.7663, 103.9318]
    INPUT_STD = [1, 1, 1]

    def __init__(self,
                 num_classes: int = 1000,
                 model_path: str = None,
                 model_url: str = None,
                 class_names: list = None,
                 class_desc: list = None):
        """
        Construct VGG-M-2048 model with specified number of prediction classes.
        :param num_classes: Number of prediction classes.
        :param model_path: Path of weights of model stored locally.
        :param model_url: URL of weights of model available via the web.
        :param class_names: Name of prediction classes.
        :param class_desc: Description of prediction classes.
        """
        super(VGG_M_2048, self).__init__()

        self.class_names = None
        self.class_desc = None
        if class_names is not None and class_desc is not None:
            if num_classes != len(class_names) or num_classes != len(class_desc):
                print('Warning: mismatch between number of classes and number of class names or descriptions: {}:{}:{}'
                      .format(num_classes, len(class_names), len(class_desc)))
            else:
                self.class_names = class_names
                self.class_desc = class_desc

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
            ('dropout6', nn.Dropout()),
            ('fc7', nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding=0)),
            ('relu7', nn.ReLU()),
            ('dropout7', nn.Dropout())]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc8ext', nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, padding=0)),
            ('dropout8', nn.Dropout()),
            ('prob', nn.Softmax(dim=1))]))

        self.bbox_reg = nn.Conv2d(2048, 4, kernel_size=1, stride=1, padding=0)

        # Initialize the FC layers with all 0 parameters to prevent messing up the other parameters.
        nn.init.constant_(self.bbox_reg.weight, 0)
        nn.init.constant_(self.bbox_reg.bias, 0)
        nn.init.constant_(self.classifier.fc8ext.weight, 0)
        nn.init.constant_(self.classifier.fc8ext.bias, 0)

        if model_path is not None:
            self.load(model_path)
        elif model_url is not None:
            self.features.load_state_dict(model_zoo.load_url(model_url))

    def load(self, model_path):
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
        elif 'pth' in model_path:
            state = torch.load(model_path)
            self.load_state_dict(state['state_dict'])
        else:
            raise NotImplementedError

    def save(self, model_path):
        if model_path.endswith('mat'):
            if self.class_names is None or self.class_desc is None:
                raise RuntimeError('Class names and descriptions not available for model saving to Matlab!')

            model_dict = {'layers': [[]],
                          'meta':   {'inputs':        {'name': 'data',
                                                       'size': [224, 224, 3, 10]},
                                     'classes':       {'name':        self.class_names,
                                                       'description': self.class_desc},
                                     'normalization': {'imageSize':     [224, 224, 3, 10],
                                                       'averageImage':  self.INPUT_MEAN,
                                                       'keepAspect':    1,
                                                       'boarder':       [32, 32],
                                                       'cropSize':      [0.875, 0.875],
                                                       'interpolation': 'bilinear'}}}
            for name, module in chain(self.features.named_children(),
                                      self.classifier.named_children(),
                                      self.bbox_reg.named_children()):
                if type(module) is nn.Conv2d:
                    layer = {'name':     name,
                             'type':     'conv',
                             'weights':  [np.transpose(module.weight.data.cpu().numpy(), [2, 3, 1, 0]),
                                          module.bias.data.cpu().numpy()],
                             'size':     [[float(module.kernel_size[0]),
                                          float(module.kernel_size[1]),
                                          float(module.in_channels),
                                          float(module.out_channels)]],
                             'pad':      [[float(module.padding[0])] * 4],
                             'stride':   [[float(module.stride[0]), float(module.stride[1])]],
                             'dilation': [[float(module.dilation[0]), float(module.dilation[1])]]}
                elif type(module) is nn.ReLU:
                    layer = {'name':     name,
                             'type':     'relu',
                             'leak':     0.,
                             'weights':  [],
                             'precious': 0}
                elif type(module) is nn.CrossMapLRN2d:
                    layer = {'name':     name,
                             'type':     'lrn',
                             'param':    [[float(module.size),
                                           float(module.k),
                                           float(module.alpha),
                                           float(module.beta)]],
                             'weights':  [],
                             'precious': 0}
                elif type(module) is nn.MaxPool2d:
                    layer = {'name':     name,
                             'type':     'pool',
                             'method':   'max',
                             'pool':     [[float(module.kernel_size), float(module.kernel_size)]],
                             'stride':   [[float(module.stride), float(module.stride)]],
                             'pad':      [[0., 1., 0., 1.]],
                             'weights':  [],
                             'precious': 0,
                             'opts':     {}}
                elif type(module) is nn.Softmax:
                    layer = {'name':     name,
                             'type':     'softmax',
                             'weights':  [],
                             'precious': 0}
                elif type(module) is nn.Dropout:
                    # Do nothing for dropout.
                    continue
                else:
                    print('Matlab conversion for {} is not implemented!'.format(type(module)))
                    raise NotImplementedError
                model_dict['layers'][0].append(layer)
            scipy.io.savemat(model_path, mdict=model_dict, oned_as='column')
        elif 'pth' in model_path:
            torch.save({'state_dict': self.state_dict()}, model_path)
        else:
            raise NotImplementedError

    def forward(self, x, output_layers):
        output_dict = {}
        num_outputs_left = len(output_layers)
        for name, module in self.features.named_children():
            x = module(x)
            if name in output_layers:
                output_dict[name] = x
                if num_outputs_left > 1:
                    num_outputs_left -= 1
                else:
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
                output_dict[name] = x
                if num_outputs_left > 1:
                    num_outputs_left -= 1
                else:
                    return output_dict
