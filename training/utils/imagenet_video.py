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

        scale = np.random.uniform(0.7, 1.1) if self._subset == 'train' else 1

        # Read annotation of the target in the current frame and the previous frame.
        cur_annotation_fn = self._data_dir + '/Annotations/VID/' + self._subset + '/' + cur_frame + '.xml'
        cur_annotation = read_annotation(cur_annotation_fn)
        if len(cur_annotation['objects']) == 0:
            return self.__getitem__((index + 1) % len(self))
        prev_annotation_fn = self._data_dir + '/Annotations/VID/' + self._subset + '/' + prev_frame + '.xml'
        prev_annotation = read_annotation(prev_annotation_fn)
        if len(prev_annotation['objects']) == 0:
            return self.__getitem__((index + 1) % len(self))

        # Crop the patch of the target in the current frame.
        xmin = cur_annotation['objects'][0]['xmin']
        xmax = cur_annotation['objects'][0]['xmax']
        ymin = cur_annotation['objects'][0]['ymin']
        ymax = cur_annotation['objects'][0]['ymax']
        x_mid = (xmin + xmax) * 0.5
        y_mid = (ymin + ymax) * 0.5
        patch_size = min(max(xmax - xmin,
                             ymax - ymin) * scale,
                         min(cur_annotation['width'], cur_annotation['height']))
        patch_xmin = max(0, int(x_mid - patch_size * 0.5))
        patch_ymin = max(0, int(y_mid - patch_size * 0.5))
        patch_xmax = patch_xmin + patch_size
        patch_ymax = patch_ymin + patch_size
        cur_img = self._loader(self._data_dir + '/Data/VID/' + self._subset + '/' + cur_frame + '.JPEG')
        cur_target = cur_img.crop((patch_xmin, patch_ymin, patch_xmax, patch_ymax))

        # Calculate bounding box regression target.
        bbox_x = (xmin + xmax - patch_xmin - patch_xmax) * 0.5 / patch_size
        bbox_y = (ymin + ymax - patch_ymin - patch_ymax) * 0.5 / patch_size
        bbox_width = (xmax - xmin) / patch_size - 1
        bbox_height = (ymax - ymin) / patch_size - 1

        # Use the target from the previous frame as the positive peer.
        prev_img = self._loader(self._data_dir + '/Data/VID/' + self._subset + '/' + prev_frame + '.JPEG')
        prev_xmin = prev_annotation['objects'][0]['xmin']
        prev_xmax = prev_annotation['objects'][0]['xmax']
        prev_ymin = prev_annotation['objects'][0]['ymin']
        prev_ymax = prev_annotation['objects'][0]['ymax']
        pos_x_mid = (prev_xmin + prev_xmax) * 0.5
        pos_y_mid = (prev_ymin + prev_ymax) * 0.5
        pos_patch_size = min(max(prev_xmax - prev_xmin,
                                 prev_ymax - prev_ymin) * scale,
                             min(prev_annotation['width'], prev_annotation['height']))
        pos_patch_xmin = max(0, int(pos_x_mid - patch_size * 0.5))
        pos_patch_ymin = max(0, int(pos_y_mid - patch_size * 0.5))
        pos_patch_xmax = pos_patch_xmin + pos_patch_size
        pos_patch_ymax = pos_patch_ymin + pos_patch_size
        pos_sample = prev_img.crop((pos_patch_xmin,
                                    pos_patch_ymin,
                                    pos_patch_xmax,
                                    pos_patch_ymax))
        pos_bbox_x = (prev_xmin + prev_xmax - pos_patch_xmin - pos_patch_xmax) * 0.5 / pos_patch_size
        pos_bbox_y = (prev_ymin + prev_ymax - pos_patch_ymin - pos_patch_ymax) * 0.5 / pos_patch_size
        pos_bbox_width = (prev_xmax - prev_xmin) / pos_patch_size - 1
        pos_bbox_height = (prev_ymax - prev_ymin) / pos_patch_size - 1

        # Pick negative peer.
        if self._subset == 'train':
            # For training, pick a neighboring area as the negative peer.
            neg_patch_size = \
                patch_size \
                * np.random.uniform(0.5, max(1.5, min(cur_annotation['width'], cur_annotation['height']) / patch_size))
            neg_patch_xmin = min(max(patch_xmin + patch_size * np.random.uniform(-0.4, 0.4), 0),
                                 cur_annotation['width'] - neg_patch_size)
            neg_patch_ymin = min(max(patch_ymin + patch_size * np.random.uniform(-0.4, 0.4), 0),
                                 cur_annotation['height'] - neg_patch_size)
            neg_patch_xmax = neg_patch_xmin + neg_patch_size
            neg_patch_ymax = neg_patch_ymin + neg_patch_size
        else:
            # For validation, use the target area in the previous frame as the negative peer.
            neg_x_mid = (prev_xmin + prev_xmax) * 0.5
            neg_y_mid = (prev_ymin + prev_ymax) * 0.5
            neg_patch_size = min(max(prev_xmax - prev_xmin,
                                     prev_ymax - prev_ymin) * scale,
                                 min(cur_annotation['width'], cur_annotation['height']))
            neg_patch_xmin = max(0, int(neg_x_mid - neg_patch_size * 0.5))
            neg_patch_ymin = max(0, int(neg_y_mid - neg_patch_size * 0.5))
            neg_patch_xmax = patch_xmin + patch_size
            neg_patch_ymax = patch_ymin + patch_size
        neg_sample = cur_img.crop((neg_patch_xmin, neg_patch_ymin, neg_patch_xmax, neg_patch_ymax))

        # Perform horizontal flipping by probability.
        if self._subset == 'train':
            if np.random.uniform(-1, 1) > 0:
                cur_target = F.hflip(cur_target)
                pos_sample = F.hflip(pos_sample)
                neg_sample = F.hflip(neg_sample)
                bbox_x = -bbox_x
                bbox_y = -bbox_y
                pos_bbox_x = -pos_bbox_x
                pos_bbox_y = -pos_bbox_y

        if self._transform is not None:
            return self._transform(cur_target), \
                   self._transform(pos_sample), \
                   self._transform(neg_sample), \
                   torch.FloatTensor([bbox_x, bbox_y, bbox_width, bbox_height]), \
                   torch.FloatTensor([pos_bbox_x, pos_bbox_y, pos_bbox_width, pos_bbox_height])
        else:
            # noinspection PyArgumentList
            return cur_target, \
                   pos_sample, \
                   neg_sample, \
                   torch.FloatTensor([bbox_x, bbox_y, bbox_width, bbox_height]), \
                   torch.FloatTensor([pos_bbox_x, pos_bbox_y, pos_bbox_width, pos_bbox_height])

    def __len__(self):
        return len(self._frames)
