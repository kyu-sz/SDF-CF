import numpy as np
import torch
import torch.utils.data as data
from utils.img_loader import default_loader
import torchvision.transforms.functional as F
from utils.utils import read_annotation


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

        scale = np.random.uniform(0.5, 1.1) if self._subset == 'train' else 1

        # Read annotation of the target in the current frame and the previous frame.
        cur_annotation_fn = self._data_dir + '/Annotations/VID/' + self._subset + '/' + cur_frame + '.xml'
        cur_annotation = read_annotation(cur_annotation_fn)
        if cur_annotation is None:
            return self.__getitem__((index + 1) % len(self))
        prev_annotation_fn = self._data_dir + '/Annotations/VID/' + self._subset + '/' + prev_frame + '.xml'
        prev_annotation = read_annotation(prev_annotation_fn)
        if prev_annotation is None:
            return self.__getitem__((index + 1) % len(self))

        # Crop the patch of the target in the current frame.
        cur_img = self._loader(self._data_dir + '/Data/VID/' + self._subset + '/' + cur_frame + '.JPEG')
        x_mid = (cur_annotation['xmin'] + cur_annotation['xmax']) * 0.5
        y_mid = (cur_annotation['ymin'] + cur_annotation['ymax']) * 0.5
        patch_size = min(max(cur_annotation['xmax'] - cur_annotation['xmin'],
                             cur_annotation['ymax'] - cur_annotation['ymin']) * scale,
                         min(cur_annotation['width'], cur_annotation['height']))
        xmin = max(0, int(x_mid - patch_size * 0.5))
        ymin = max(0, int(y_mid - patch_size * 0.5))
        xmax = xmin + patch_size
        ymax = ymin + patch_size
        cur_target = cur_img.crop((xmin, ymin, xmax, ymax))

        # Calculate bounding box regression target.
        bbox_x = (cur_annotation['xmin'] + cur_annotation['xmax'] - xmin - xmax) * 0.5 / patch_size
        bbox_y = (cur_annotation['ymin'] + cur_annotation['ymax'] - ymin - ymax) * 0.5 / patch_size
        bbox_width = (cur_annotation['xmax'] - cur_annotation['xmin']) / patch_size
        bbox_height = (cur_annotation['ymax'] - cur_annotation['ymin']) / patch_size

        # Use the target from the previous frame as the positive peer.
        prev_img = self._loader(self._data_dir + '/Data/VID/' + self._subset + '/' + prev_frame + '.JPEG')
        pos_x_mid = (prev_annotation['xmin'] + prev_annotation['xmax']) * 0.5
        pos_y_mid = (prev_annotation['ymin'] + prev_annotation['ymax']) * 0.5
        pos_patch_size = min(max(prev_annotation['xmax'] - prev_annotation['xmin'],
                                 prev_annotation['ymax'] - prev_annotation['ymin']) * scale,
                             min(prev_annotation['width'], prev_annotation['height']))
        pos_xmin = max(0, int(pos_x_mid - patch_size * 0.5))
        pos_ymin = max(0, int(pos_y_mid - patch_size * 0.5))
        pos_xmax = pos_xmin + pos_patch_size
        pos_ymax = pos_ymin + pos_patch_size
        pos_peer = prev_img.crop((pos_xmin,
                                  pos_ymin,
                                  pos_xmax,
                                  pos_ymax))
        pos_bbox_x = (prev_annotation['xmin'] + prev_annotation['xmax'] - pos_xmin - pos_xmax) * 0.5 / pos_patch_size
        pos_bbox_y = (prev_annotation['ymin'] + prev_annotation['ymax'] - pos_ymin - pos_ymax) * 0.5 / pos_patch_size
        pos_bbox_width = (prev_annotation['xmax'] - prev_annotation['xmin']) / pos_patch_size
        pos_bbox_height = (prev_annotation['ymax'] - prev_annotation['ymin']) / pos_patch_size

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
            neg_x_mid = (prev_annotation['xmin'] + prev_annotation['xmax']) * 0.5
            neg_y_mid = (prev_annotation['ymin'] + prev_annotation['ymax']) * 0.5
            neg_patch_size = min(max(prev_annotation['xmax'] - prev_annotation['xmin'],
                                     prev_annotation['ymax'] - prev_annotation['ymin']) * scale,
                                 min(cur_annotation['width'], cur_annotation['height']))
            neg_xmin = max(0, int(neg_x_mid - neg_patch_size * 0.5))
            neg_ymin = max(0, int(neg_y_mid - neg_patch_size * 0.5))
            neg_xmax = xmin + patch_size
            neg_ymax = ymin + patch_size

        neg_peer = cur_img.crop((neg_xmin, neg_ymin, neg_xmax, neg_ymax))

        if self._subset == 'train':
            if np.random.uniform(-1, 1) > 0:
                cur_target = F.hflip(cur_target)
                pos_peer = F.hflip(pos_peer)
                neg_peer = F.hflip(neg_peer)

        if self._transform is not None:
            return self._transform(cur_target), \
                   self._transform(pos_peer), \
                   self._transform(neg_peer), \
                   torch.FloatTensor([bbox_x, bbox_y, bbox_width, bbox_height]), \
                   torch.FloatTensor([pos_bbox_x, pos_bbox_y, pos_bbox_width, pos_bbox_height])
        else:
            # noinspection PyArgumentList
            return cur_target, \
                   pos_peer, \
                   neg_peer, \
                   torch.FloatTensor([bbox_x, bbox_y, bbox_width, bbox_height]), \
                   torch.FloatTensor([pos_bbox_x, pos_bbox_y, pos_bbox_width, pos_bbox_height])

    def __len__(self):
        return len(self._frames)
