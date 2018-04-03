import os
import subprocess

import torch.utils.data as data
from .img_loader import default_loader


def _check_or_extract(dir, pack_suffix):
    if not os.path.exists(dir):
        os.makedirs(dir)
        subprocess.Popen(['tar', '-xf', dir + pack_suffix, '-C', dir])
        print('Extracted', dir + pack_suffix, 'to', dir)


class ImageNetDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, loader=default_loader):
        self._data_dir = data_dir
        self._transform = transform
        self._loader = loader

        self._annotation_dir = data_dir + '/Annotation'
        _check_or_extract(self._annotation_dir, '.tar.gz')
        self._image_dir = data_dir + '/fall11_whole'
        _check_or_extract(self._image_dir, '.tar')

        annotations = os.listdir(data_dir + '/Annotation')
        image_data = os.listdir(data_dir + '/fall11_whole')

        self.classes = {}
        self.num_classes = 0
        wnid_set = set()
        # read classes that are in the widely-used 1000 classes.
        with open(os.path.dirname(os.path.realpath(__file__)) + '/imagenet_cls.txt', 'r') as f:
            for line in f:
                wnid, cid, name = line.split()
                cid = int(cid)
                self.num_classes += 1
                if wnid not in annotations:
                    if wnid + '.tar.gz' in annotations:
                        _check_or_extract(os.path.join(self._annotation_dir, wnid), '.tar.gz')
                    else:
                        print(wnid, cid, name, 'has no annotation!')
                        continue
                if wnid not in image_data:
                    if wnid + '.tar' in image_data:
                        _check_or_extract(os.path.join(self._image_dir, wnid), '.tar')
                    else:
                        print(wnid, cid, name, 'has no images!')
                        continue
                self.classes[cid] = wnid
                wnid_set.add(wnid)

        # find classes that are newly available in the ImageNet dataset, other than the 1000 classes.
        for wnid in annotations:
            wnid = wnid[:-7] if wnid.endswith('.tar.gz') else wnid
            if wnid not in wnid_set and (wnid in image_data or wnid + '.tar' in image_data):
                self.num_classes += 1
                self.classes[self.num_classes] = wnid
                wnid_set.add(wnid)
                _check_or_extract(os.path.join(self._annotation_dir, wnid), '.tar.gz')
                _check_or_extract(os.path.join(self._image_dir, wnid), '.tar')

        print('Found {} classes!'.format(self.num_classes))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


if __name__ == "__main__":
    _ = ImageNetDataset(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'datasets', 'ImageNet'))
