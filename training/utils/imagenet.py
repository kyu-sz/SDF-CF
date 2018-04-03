import os
import subprocess

import numpy as np
import torch.utils.data as data

from .img_loader import default_loader
from .utils import read_annotation


def _check_or_extract(dir, pack_suffix, output_suffix=''):
    if not os.path.exists(dir):
        if os.path.exists(dir + pack_suffix):
            os.makedirs(dir, exist_ok=True)
            subprocess.Popen(['tar', '-xf', dir + pack_suffix, '-C', os.path.join(dir, output_suffix)])
            print('Extracting', dir + pack_suffix, 'to', os.path.join(dir, output_suffix))
        else:
            print('Warning: cannot find', dir + pack_suffix)


class ImageNetDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, loader=default_loader):
        self._data_dir = data_dir
        print('Loading ImageNet from {}...'.format(self._data_dir))
        self._transform = transform
        self._loader = loader

        self._annotation_dir = data_dir + '/Annotation'
        _check_or_extract(self._annotation_dir, '.tar.gz')
        self._image_dir = data_dir + '/fall11_whole'
        _check_or_extract(self._image_dir, '.tar')

        annotations = sorted(os.listdir(data_dir + '/Annotation'))
        image_data = os.listdir(data_dir + '/fall11_whole')

        self.idx2cls = {}
        self.num_classes = 0
        wnid_set = set()
        self.num_images_per_class = []
        # read classes that are in the widely-used 1000 classes.
        with open(os.path.dirname(os.path.realpath(__file__)) + '/imagenet_cls.txt', 'r') as f:
            for line in f:
                wnid, cid, name = line.split()
                cid = int(cid)
                wnid_set.add(wnid)
                self.idx2cls[cid] = wnid
                self.num_classes += 1
                _check_or_extract(os.path.join(self._annotation_dir, wnid), '.tar.gz', '../..')
                _check_or_extract(os.path.join(self._image_dir, wnid), '.tar')
                self.num_images_per_class.append(
                    len([fn for fn in os.listdir(self.image_dir(wnid)) if fn.endswith('JPEG')])
                    if os.path.exists(self.image_dir(wnid)) else 0)

        # find classes that are newly available in the ImageNet dataset, other than the 1000 classes.
        for wnid in annotations:
            wnid = wnid[:-7] if wnid.endswith('.tar.gz') else wnid
            if wnid not in wnid_set and (wnid in image_data or wnid + '.tar' in image_data):
                self.num_classes += 1
                self.idx2cls[self.num_classes] = wnid
                wnid_set.add(wnid)
                _check_or_extract(os.path.join(self._annotation_dir, wnid), '.tar.gz', '../..')
                _check_or_extract(os.path.join(self._image_dir, wnid), '.tar')
                self.num_images_per_class.append(
                    len([fn for fn in os.listdir(self.image_dir(wnid)) if fn.endswith('JPEG')])
                    if os.path.exists(self.image_dir(wnid)) else 0)

        self._idx_end = np.cumsum(self.num_images_per_class)

        print('Found {} classes!'.format(self.num_classes))

    def image_dir(self, wnid=''):
        return os.path.join(self._image_dir, wnid)

    def annotation_dir(self, wnid=''):
        return os.path.join(self._annotation_dir, wnid)

    def __getitem__(self, index):
        cid = np.searchsorted(self._idx_end, 3) + 1
        wnid = self.idx2cls[cid]
        idx_in_class = index - (self._idx_end[cid - 2] if cid >= 2 else 0)

        img_fn = sorted(os.listdir(self.image_dir(wnid)))[idx_in_class]
        img = self._loader(os.path.join(self.image_dir(wnid), img_fn))

        annotation_fn = img_fn[:-4] + '.'
        annotation = read_annotation(os.path.join(self.annotation_dir(wnid), annotation_fn))

        return img, annotation

    def __len__(self):
        return self._idx_end[-1]


if __name__ == "__main__":
    dataset = ImageNetDataset(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'datasets', 'ImageNet'))
    print('Loaded {} images from {} classes in ImageNet'.format(len(dataset), dataset.num_classes))
