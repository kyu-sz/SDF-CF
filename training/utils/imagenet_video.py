import os
import cv2
import numpy as np

import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import math
from itertools import groupby
import xml.etree.ElementTree
import torchvision.transforms.functional as TF


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def _get_element(xml_block, name):
    """
    Get a unique element from a XML block by name.
    Args:
        xml_block (xml.etree.ElementTree.Element): XML block.
        name (str): element name.
    Returns:
        xml.etree.ElementTree.Element: the corresponding element block.
    """
    return xml_block.iter(name).__next__()


def _read_annotation(annotation_fn):
    e = xml.etree.ElementTree.parse(annotation_fn).getroot()

    size_block = _get_element(e, 'size')
    width = int(_get_element(size_block, 'width').text)
    height = int(_get_element(size_block, 'height').text)

    obj_block = _get_element(e, 'object')
    bndbox_block = _get_element(obj_block, 'bndbox')
    xmax = int(_get_element(bndbox_block, 'xmax').text)
    xmin = int(_get_element(bndbox_block, 'xmin').text)
    ymax = int(_get_element(bndbox_block, 'ymax').text)
    ymin = int(_get_element(bndbox_block, 'ymin').text)

    return {'width': width, 'height': height, 'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin': ymin}


class ImageNetVideoDataset(data.Dataset):
    def __init__(self, data_dir, subset='train', transform=None, loader=default_loader):
        self._data_dir = data_dir
        self._subset = subset

        list_fn = data_dir + '/ImageSets/VID/' + subset + '.txt'
        with open(list_fn, 'r') as f:
            self._frames = [l.split()[0] for l in f]

        self._transform = transform
        self._loader = loader

    def __getitem__(self, index):
        cur_frame = self._frames[index]

        # Read information of the current target.
        cur_img = self._loader(self._data_dir + '/Data/VID/' + self._subset + '/' + cur_frame + '.JPEG')
        cur_annotation_fn = self._loader(self._data_dir + '/Annotations/VID/' + self._subset + '/' + cur_frame + '.xml')
        cur_annotation = _read_annotation(cur_annotation_fn)
        cur_target = cur_img.crop(cur_annotation['xmin'],
                                  cur_annotation['ymin'],
                                  cur_annotation['xmax'],
                                  cur_annotation['ymax'])
        target_width = cur_annotation['xmax'] - cur_annotation['xmin']
        target_height = cur_annotation['ymax'] - cur_annotation['ymin']

        # Pick positive peer.
        frame_idx = int(cur_frame[-6:])
        if frame_idx == 0:
            # Pick a random shifted and scaled version of the target as the positive peer.
            pos_peer = cur_img.crop(cur_annotation['xmin'] + target_width * np.random.uniform(-0.1, 0.1),
                                    cur_annotation['ymin'] + target_height * np.random.uniform(-0.1, 0.1),
                                    cur_annotation['xmax'] + target_width * np.random.uniform(-0.1, 0.1),
                                    cur_annotation['ymax'] + target_height * np.random.uniform(-0.1, 0.1))
        else:
            # Use the target from the previous frame as the positive peer.
            prev_frame = self._frames[index - 1]
            prev_img = self._loader(self._data_dir + '/Data/VID/' + self._subset + '/' + prev_frame + '.JPEG')
            prev_annotation_fn = self._loader(self._data_dir + '/Annotations/VID/'
                                              + self._subset + '/' + prev_frame + '.xml')
            prev_annotation = _read_annotation(prev_annotation_fn)
            pos_peer = prev_img.crop(prev_annotation['xmin'],
                                     prev_annotation['ymin'],
                                     prev_annotation['xmax'],
                                     prev_annotation['ymax'])

        # Pick negative peer.
        if self._subset == 'train':
            # For training, pick a neighboring area as the negative peer.
            neg_xmin = cur_annotation['xmin'] \
                       + target_width * np.random.uniform(0.5, 0.8) * np.sign(np.random.uniform(-1, 1))
            neg_ymin = cur_annotation['ymin'] \
                       + target_height * np.random.uniform(0.5, 0.8) * np.sign(np.random.uniform(-1, 1))
            neg_xmax = neg_xmin + target_width * np.random.uniform(0.5, 1.5)
            neg_ymax = neg_ymin + target_height * np.random.uniform(0.5, 1.5)
        else:
            # For validation, use a fixed area as the negative peer.
            if frame_idx == 0:
                # There is no previous frame. Use a fixed neighbor area.
                neg_xmin = max(cur_annotation['xmin'] - target_width / 2, 0)
                neg_ymin = max(cur_annotation['ymin'] - target_height / 2, 0)
                neg_xmax = neg_xmin + target_width
                neg_ymax = neg_ymin + target_height
            else:
                # Use the target area in the previous frame.
                neg_xmin = prev_annotation['xmin']
                neg_xmax = prev_annotation['xmax']
                neg_ymin = prev_annotation['ymin']
                neg_ymax = prev_annotation['ymax']

        neg_peer = cur_img.crop(neg_xmin, neg_ymin, neg_xmax, neg_ymax)

        if self._subset == 'train':
            if np.random.uniform(-1, 1) > 0:
                cur_target = TF.hflip(cur_target)
                pos_peer = TF.hflip(pos_peer)
                neg_peer = TF.hflip(neg_peer)

        return self._transform(cur_target), \
               self._transform(pos_peer), \
               self._transform(neg_peer)

    def __len__(self):
        return len(self._frames)
