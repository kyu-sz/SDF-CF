import os
import cv2


class VOTLoader:
    def __init__(self, data_dir):
        self._data_dir = data_dir
        list_fn = data_dir + '/list.txt'
        with open(list_fn, 'r') as f:
            self._seq_names = [l.split()[0] for l in f]

    def __iter__(self):
        for name in self._seq_names:
            sub_dir = self._data_dir + '/' + name
            seq_data = {'name': name,
                        'images': [cv2.imread(sub_dir + '/' + fn)
                                   for fn
                                   in sorted(os.listdir(sub_dir))
                                   if fn.endswith('jpg')]}
            yield seq_data
