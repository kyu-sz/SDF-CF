import os
import cv2
import numpy as np

import torch.utils.data as data

class VOTLoader(data.Dataset):
    def __init__(self, data_dir):
        self._data_dir = data_dir
        list_fn = data_dir + '/list.txt'
        with open(list_fn, 'r') as f:
            self._seq_names = [l.split()[0] for l in f]

    def __getitem__(self, index):
        name = self._seq_names[index]
        sub_dir = self._data_dir + '/' + name
        seq_data = {'name': name,
                    'images': [cv2.imread(sub_dir + '/' + fn)
                               for fn
                               in sorted(os.listdir(sub_dir))
                               if fn.endswith('jpg')]}
        with open(sub_dir + '/groundtruth.txt', 'r') as f:
            seq_data['bbox'] = [np.array([float(s) for s in line.split(',')])
                                for line in f]
        return seq_data

    def __len__(self):
        return len(self._seq_names)
