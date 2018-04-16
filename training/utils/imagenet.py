import os
import os.path as osp
import subprocess
import asyncio
import threading

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F

from .img_loader import default_loader
from .utils import read_annotation
import requests


def _check_or_extract(dir, pack_suffix, output_suffix='', async=True, force_extract=False):
    if not osp.exists(dir) or force_extract:
        if osp.exists(dir + pack_suffix):
            os.makedirs(dir, exist_ok=True)
            args = ['tar', '-xf', dir + pack_suffix, '-C', osp.join(dir, output_suffix)]
            if async:
                subprocess.Popen(args)
                print('Extracting', dir + pack_suffix, 'to', osp.join(dir, output_suffix))
            else:
                subprocess.call(args)
                print('Extracted', dir + pack_suffix, 'to', osp.join(dir, output_suffix))
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
        self._image_dir = data_dir + '/Image'
        _check_or_extract(self._image_dir, '.tar')

        annotations = sorted(os.listdir(self._annotation_dir))
        image_data = os.listdir(self._image_dir)

        self.idx2cls = {}
        self.num_classes = 0
        wnid_set = set()
        self._num_samples_per_class = []
        # read classes that are in the widely-used 1000 classes.
        with open(osp.dirname(osp.realpath(__file__)) + '/imagenet_cls.txt', 'r') as f:
            for line in f:
                wnid, cid, name = line.split()
                cid = int(cid) - 1
                wnid_set.add(wnid)
                self.idx2cls[cid] = wnid
                self.num_classes += 1
                _check_or_extract(self.annotation_dir(wnid), '.tar.gz', '../..', async=False)
                _check_or_extract(self.image_dir(wnid), '.tar', async=True)
                self._num_samples_per_class.append(
                    len([fn for fn in os.listdir(self.annotation_dir(wnid)) if fn.endswith('xml')])
                    if osp.exists(self.annotation_dir(wnid)) else 0)

        # find classes that are newly available in the ImageNet dataset, other than the 1000 classes.
        for wnid in annotations:
            wnid = wnid[:-7] if wnid.endswith('.tar.gz') else wnid
            if wnid not in wnid_set and (wnid in image_data or wnid + '.tar' in image_data):
                self.num_classes += 1
                self.idx2cls[self.num_classes] = wnid
                wnid_set.add(wnid)
                _check_or_extract(osp.join(self._annotation_dir, wnid), '.tar.gz', '../..', async=False)
                _check_or_extract(osp.join(self._image_dir, wnid), '.tar', async=True)
                self._num_samples_per_class.append(
                    len([fn for fn in os.listdir(self.annotation_dir(wnid)) if fn.endswith('xml')])
                    if osp.exists(self.annotation_dir(wnid)) else 0)

        self._idx_end = np.cumsum(self._num_samples_per_class)

        self._url_dict = {}
        self._url_dict_reading_thread = threading.Thread(target=self._read_urls,
                                                         name="URLs Reading Thread")
        self._url_dict_reading_thread.start()

        print('Found {} classes!'.format(self.num_classes))

    def _read_urls(self):
        for url_fn in ['fall11_urls.txt', 'spring10_urls.txt', 'winter11_urls.txt', 'urls.txt']:
            with open(osp.join(self._data_dir, url_fn), 'r', errors='ignore') as f:
                for line in f:
                    name, url = line.split()[:2]
                    self._url_dict[name] = url

    def _download_img(self, folder, name):
        self._url_dict_reading_thread.join()

        with open(osp.join(self.image_dir(folder), name + '.JPEG'), 'wb') as handle:
            response = requests.get(self._url_dict[name], stream=True)
            if not response.ok:
                print(response)
                return False
            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)
        return True

    def image_dir(self, wnid=''):
        return osp.join(self._image_dir, wnid)

    def annotation_dir(self, wnid=''):
        return osp.join(self._annotation_dir, wnid)

    def __getitem__(self, index):
        # Read the annotation for the frame.
        cid = np.searchsorted(self._idx_end, index + 1)
        wnid = self.idx2cls[cid]
        idx_in_class = index - (self._idx_end[cid - 1] if cid >= 1 else 0)
        annotation_fn = sorted(os.listdir(self.annotation_dir(wnid)))[idx_in_class]
        img_annotation = read_annotation(osp.join(self.annotation_dir(wnid), annotation_fn))

        # Find the annotation for the object.
        obj_annotation = None
        for obj in img_annotation['objects']:
            if obj['name'] == wnid:
                obj_annotation = obj
                break
        if obj_annotation is None:
            return self[np.random.randint(len(self))]

        # Load image. Download it if the image file does not exist.
        img_path = osp.join(self.image_dir(img_annotation['folder']), img_annotation['filename'] + '.JPEG')
        if not os.path.exists(img_path):
            if img_annotation['filename'] + '.JPEG' not in self._url_dict \
                    or not self._download_img(img_annotation['folder'], img_annotation['filename']):
                return self[np.random.randint(len(self))]
        try:
            img = self._loader(img_path)
        except OSError:
            if img_annotation['filename'] + '.JPEG' not in self._url_dict \
                    or not self._download_img(img_annotation['folder'], img_annotation['filename']):
                return self[np.random.randint(len(self))]
            img = self._loader(img_path)

        # Crop the square patch of the object.
        scale = np.random.uniform(0.5, 1.1)

        x_mid = (obj_annotation['xmin'] + obj_annotation['xmax']) * 0.5
        y_mid = (obj_annotation['ymin'] + obj_annotation['ymax']) * 0.5
        patch_size = min(max(obj_annotation['xmax'] - obj_annotation['xmin'],
                             obj_annotation['ymax'] - obj_annotation['ymin']) * scale,
                         min(img_annotation['width'], img_annotation['height']))
        xmin = min(img_annotation['width'] - patch_size,
                   max(0, int(x_mid - (patch_size * 0.5 + np.random.uniform(-0.1, 0.1)))))
        ymin = min(img_annotation['height'] - patch_size,
                   max(0, int(y_mid - (patch_size * 0.5 + np.random.uniform(-0.1, 0.1)))))
        xmax = xmin + patch_size
        ymax = ymin + patch_size
        img = img.crop((xmin, ymin, xmax, ymax))

        # Calculate bounding box regression target.
        bbox_x = (obj_annotation['xmin'] + obj_annotation['xmax'] - xmin - xmax) * 0.5 / patch_size
        bbox_y = (obj_annotation['ymin'] + obj_annotation['ymax'] - ymin - ymax) * 0.5 / patch_size
        bbox_width = (obj_annotation['xmax'] - obj_annotation['xmin']) / patch_size - 1
        bbox_height = (obj_annotation['ymax'] - obj_annotation['ymin']) / patch_size - 1

        if np.random.uniform(-1, 1) > 0:
            img = F.hflip(img)
            bbox_x = -bbox_x
            bbox_y = -bbox_y

        return self._transform(img), \
               torch.LongTensor([[int(cid)]]), \
               torch.FloatTensor([bbox_x, bbox_y, bbox_width, bbox_height])

    def __len__(self):
        return self._idx_end[-1]


if __name__ == "__main__":
    dataset = ImageNetDataset(
        osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'datasets', 'ImageNet'))
    print('Loaded {} images from {} classes in ImageNet'.format(len(dataset), dataset.num_classes))
