import numpy as np
import torch
import torch.utils.data as data
from utils.img_loader import default_loader
import torchvision.transforms as transforms
import PIL.ImageOps as ImageOps
from utils.utils import read_annotation, load_synsets


class ImageNetVideoDataset(data.Dataset):
    def __init__(self, data_dir, subset='train', transform=None, loader=default_loader):
        # Load synsets.
        self.synsets, self.wnid2id = load_synsets()

        self._data_dir = data_dir
        self._subset = subset

        list_fn = data_dir + '/ImageSets/VID/' + subset + '.txt'
        with open(list_fn, 'r') as f:
            # Skip first frames of the sequences.
            self._frames = [l.split()[0] for l in f if int(l.split()[0][-6:]) > 0]

        self._transform = transform if transform is not None else transforms.ToTensor()
        self._loader = loader

    def __getitem__(self, index):
        cur_frame = self._frames[index]
        prev_frame = cur_frame[:-6] + str(int(cur_frame[-6:]) - 1).zfill(6)

        # Read annotation of the target in the current frame and the previous frame.
        cur_annotation_fn = self._data_dir + '/Annotations/VID/' + self._subset + '/' + cur_frame + '.xml'
        cur_annotation = read_annotation(cur_annotation_fn)
        if len(cur_annotation['objects']) == 0:
            return self.__getitem__((index + 1) % len(self))
        prev_annotation_fn = self._data_dir + '/Annotations/VID/' + self._subset + '/' + prev_frame + '.xml'
        prev_annotation = read_annotation(prev_annotation_fn)
        if len(prev_annotation['objects']) == 0:
            return self.__getitem__((index + 1) % len(self))

        if len(cur_annotation['objects']) > 1 or len(prev_annotation['objects']) > 1:
            print('Multiple objects labeled in the ImageNet Video dataset! This program need modification!')
            exit(-1)

        # Scale factor of the interest region to the bounding box.
        scale_factor = np.random.uniform(0.7, 1.1) if self._subset == 'train' else 1

        # Crop the patch of the target in the current frame.
        xmin = cur_annotation['objects'][0]['xmin']
        xmax = cur_annotation['objects'][0]['xmax']
        ymin = cur_annotation['objects'][0]['ymin']
        ymax = cur_annotation['objects'][0]['ymax']
        x_mid = (xmin + xmax) * 0.5
        y_mid = (ymin + ymax) * 0.5
        target_patch_size = min(max(xmax - xmin, ymax - ymin) * scale_factor,
                                min(cur_annotation['width'], cur_annotation['height']))
        target_patch_xmin = max(0, int(x_mid - target_patch_size * 0.5))
        target_patch_ymin = max(0, int(y_mid - target_patch_size * 0.5))
        target_patch_xmax = target_patch_xmin + target_patch_size
        target_patch_ymax = target_patch_ymin + target_patch_size
        cur_img = self._loader(self._data_dir + '/Data/VID/' + self._subset + '/' + cur_frame + '.JPEG')
        cur_target = cur_img.crop((target_patch_xmin, target_patch_ymin, target_patch_xmax, target_patch_ymax))

        # Calculate bounding box regression target.
        bbox_x = (xmin + xmax - target_patch_xmin - target_patch_xmax) * 0.5 / target_patch_size
        bbox_y = (ymin + ymax - target_patch_ymin - target_patch_ymax) * 0.5 / target_patch_size
        bbox_width = (xmax - xmin) / target_patch_size - 1
        bbox_height = (ymax - ymin) / target_patch_size - 1

        # Use the target from the previous frame as the positive sample for smoothness training.
        prev_img = self._loader(self._data_dir + '/Data/VID/' + self._subset + '/' + prev_frame + '.JPEG')
        prev_xmin = prev_annotation['objects'][0]['xmin']
        prev_xmax = prev_annotation['objects'][0]['xmax']
        prev_ymin = prev_annotation['objects'][0]['ymin']
        prev_ymax = prev_annotation['objects'][0]['ymax']
        pos_x_mid = (prev_xmin + prev_xmax) * 0.5
        pos_y_mid = (prev_ymin + prev_ymax) * 0.5
        pos_patch_size = min(max(prev_xmax - prev_xmin, prev_ymax - prev_ymin) * scale_factor,
                             min(prev_annotation['width'], prev_annotation['height']))
        pos_patch_xmin = max(0, int(pos_x_mid - target_patch_size * 0.5))
        pos_patch_ymin = max(0, int(pos_y_mid - target_patch_size * 0.5))
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

        # Pick negative sample for smoothness task.
        if self._subset == 'train':
            # For training, pick a neighboring area as the negative peer.
            neg_patch_size = target_patch_size \
                             * np.random.uniform(0.5, min(1.5, min(cur_annotation['width'],
                                                                   cur_annotation['height']) / target_patch_size))
            neg_patch_xmin = min(max(target_patch_xmin + target_patch_size * np.random.uniform(-0.4, 0.4), 0),
                                 cur_annotation['width'] - neg_patch_size)
            neg_patch_ymin = min(max(target_patch_ymin + target_patch_size * np.random.uniform(-0.4, 0.4), 0),
                                 cur_annotation['height'] - neg_patch_size)
            neg_patch_xmax = neg_patch_xmin + neg_patch_size
            neg_patch_ymax = neg_patch_ymin + neg_patch_size
        else:
            # For validation, use the target area in the previous frame as the negative peer.
            neg_patch_xmin = pos_patch_xmin
            neg_patch_ymin = pos_patch_ymin
            neg_patch_xmax = pos_patch_xmax
            neg_patch_ymax = pos_patch_ymax
        neg_sample = cur_img.crop((neg_patch_xmin, neg_patch_ymin, neg_patch_xmax, neg_patch_ymax))

        # Perform horizontal flipping by probability.
        if self._subset == 'train':
            if np.random.randint(0, 2):
                cur_target = ImageOps.mirror(cur_target)
                pos_sample = ImageOps.mirror(pos_sample)
                neg_sample = ImageOps.mirror(neg_sample)
                bbox_x = -bbox_x
                bbox_y = -bbox_y
                pos_bbox_x = -pos_bbox_x
                pos_bbox_y = -pos_bbox_y

        # noinspection PyCallingNonCallable
        return self._transform(cur_target), \
               self._transform(pos_sample), \
               self._transform(neg_sample), \
               torch.tensor(self.wnid2id[cur_annotation['name']]), \
               torch.tensor([bbox_x, bbox_y, bbox_width, bbox_height]), \
               torch.tensor([pos_bbox_x, pos_bbox_y, pos_bbox_width, pos_bbox_height])

    def __len__(self):
        return len(self._frames)
