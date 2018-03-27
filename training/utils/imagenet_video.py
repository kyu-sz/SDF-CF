import xml.etree.ElementTree

import numpy as np
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image


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

    try:
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
    except StopIteration:
        return None


class ImageNetVideoDataset(data.Dataset):
    def __init__(self, data_dir, subset='train', transform=None, loader=default_loader):
        self._data_dir = data_dir
        self._subset = subset

        list_fn = data_dir + '/ImageSets/VID/' + subset + '.txt'
        with open(list_fn, 'r') as f:
            # Skip first frames of the sequences.
            self._frames = [l.split()[0] for l in f if int(l.split()[0][-6:]) > 0]

        self._transform = transform
        self._loader = loader

    def __getitem__(self, index):
        cur_frame = self._frames[index]
        prev_frame = cur_frame[:-6] + str(int(cur_frame[-6:]) - 1).zfill(6)

        # Read annotation of the target in the current frame and the previous frame.

        cur_annotation_fn = self._data_dir + '/Annotations/VID/' + self._subset + '/' + cur_frame + '.xml'
        cur_annotation = _read_annotation(cur_annotation_fn)
        if cur_annotation is None:
            return self.__getitem__((index + 1) % len(self))
        prev_annotation_fn = self._data_dir + '/Annotations/VID/' + self._subset + '/' + prev_frame + '.xml'
        prev_annotation = _read_annotation(prev_annotation_fn)
        if prev_annotation is None:
            return self.__getitem__((index + 1) % len(self))

        cur_img = self._loader(self._data_dir + '/Data/VID/' + self._subset + '/' + cur_frame + '.JPEG')
        x_mid = (cur_annotation['xmin'] + cur_annotation['xmax']) / 2
        y_mid = (cur_annotation['ymin'] + cur_annotation['ymax']) / 2
        patch_size = min(max(cur_annotation['xmax'] - cur_annotation['xmin'],
                             cur_annotation['ymax'] - cur_annotation['ymin']),
                         min(cur_annotation['width'], cur_annotation['height']))
        xmin = max(0, int(x_mid - patch_size / 2))
        ymin = max(0, int(y_mid - patch_size / 2))
        xmax = xmin + patch_size
        ymax = ymin + patch_size
        cur_target = cur_img.crop((xmin, ymin, xmax, ymax)).copy()

        # Use the target from the previous frame as the positive peer.
        prev_img = self._loader(self._data_dir + '/Data/VID/' + self._subset + '/' + prev_frame + '.JPEG')
        pos_x_mid = (prev_annotation['xmin'] + prev_annotation['xmax']) / 2
        pos_y_mid = (prev_annotation['ymin'] + prev_annotation['ymax']) / 2
        pos_patch_size = min(max(prev_annotation['xmax'] - prev_annotation['xmin'],
                                 prev_annotation['ymax'] - prev_annotation['ymin']),
                             min(prev_annotation['width'], prev_annotation['height']))
        pos_xmin = max(0, int(pos_x_mid - patch_size / 2))
        pos_xmax = max(0, int(pos_y_mid - patch_size / 2))
        pos_ymin = pos_xmin + pos_patch_size
        pos_ymax = pos_xmax + pos_patch_size
        pos_peer = prev_img.crop((pos_xmin,
                                  pos_ymin,
                                  pos_xmax,
                                  pos_ymax)).copy()

        # Pick negative peer.
        if self._subset == 'train':
            # For training, pick a neighboring area as the negative peer.
            if patch_size == min(cur_annotation['width'], cur_annotation['height']):
                neg_patch_size = patch_size * np.random.uniform(0.5, 1)
            else:
                neg_patch_size = min(min(cur_annotation['width'], cur_annotation['height']),
                                     patch_size * np.random.uniform(0.5, 1.5))
            neg_xmin = min(max(xmin + patch_size * np.random.uniform(-0.2, 0.2), 0),
                           cur_annotation['width'] - neg_patch_size)
            neg_ymin = min(max(ymin + patch_size * np.random.uniform(-0.2, 0.2), 0),
                           cur_annotation['height'] - neg_patch_size)
            neg_xmax = neg_xmin + neg_patch_size
            neg_ymax = neg_ymin + neg_patch_size
        else:
            # For validation, use the target area in the previous frame as the negative peer.
            neg_x_mid = (prev_annotation['xmin'] + prev_annotation['xmax']) / 2
            neg_y_mid = (prev_annotation['ymin'] + prev_annotation['ymax']) / 2
            neg_patch_size = min(max(prev_annotation['xmax'] - prev_annotation['xmin'],
                                     prev_annotation['ymax'] - prev_annotation['ymin']),
                                 min(cur_annotation['width'], cur_annotation['height']))
            neg_xmin = max(0, int(neg_x_mid - neg_patch_size / 2))
            neg_xmax = max(0, int(neg_y_mid - neg_patch_size / 2))
            neg_ymin = xmin + patch_size
            neg_ymax = ymin + patch_size

        neg_peer = cur_img.crop((neg_xmin, neg_ymin, neg_xmax, neg_ymax)).copy()

        if self._subset == 'train':
            if np.random.uniform(-1, 1) > 0:
                cur_target = TF.hflip(cur_target)
                pos_peer = TF.hflip(pos_peer)
                neg_peer = TF.hflip(neg_peer)

        if self._transform is not None:
            return self._transform(cur_target), \
                   self._transform(pos_peer), \
                   self._transform(neg_peer)
        else:
            return cur_target, pos_peer, neg_peer

    def __len__(self):
        return len(self._frames)
