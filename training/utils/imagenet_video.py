from typing import Callable

import numpy as np
import torch
import torch.utils.data as data
from utils.img_loader import default_loader
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from utils.utils import *


class ImageNetVideoDataset(data.Dataset):
    def __init__(self,
                 data_dir: str,
                 subset: str = 'train',
                 transform=None,
                 loader: Callable[[str], Image.Image] = default_loader):
        # Load synsets.
        self.synsets, self.wnid2id = load_synsets()
        self.num_classes = len(self.synsets)

        # Load WordNet Hierarchy.
        self._synset_parent = {}
        for line in read_web_file('http://www.image-net.org/archive/wordnet.is_a.txt').split('\n'):
            if len(line):
                parent, child = line.split()
                self._synset_parent[child] = parent

        # Cache class labels for synset.
        self._class_label_cache = {}

        self._data_dir = data_dir
        self._subset = subset

        list_fn = data_dir + '/ImageSets/VID/' + subset + '.txt'
        with open(list_fn, 'r') as f:
            # Skip first frames of the sequences.
            self._frames = [l.split()[0] for l in f if int(l.split()[0][-6:]) > 0]

        self._transform = transform if transform is not None else transforms.ToTensor()
        self._loader = loader

    def _resolve_class_label(self, synset: str) -> list:
        class_labels = [0] * self.num_classes
        if synset in self.synsets:  # If the synset is labeled in ImageNet.
            class_labels[self.wnid2id[synset]] = 1
        else:  # The synset is not labeled, but its descendants or ancestors might be.
            # Check ancestor first.
            if synset in self._synset_parent:
                ancestor = self._synset_parent[synset]
                while ancestor not in class_labels:
                    if ancestor in self._synset_parent:
                        ancestor = self._synset_parent[ancestor]
                    else:
                        ancestor = None
                        break
                if ancestor is not None:
                    class_labels[self.wnid2id[ancestor]] = 1
            else:
                # Recursively check descendants.
                # The sample should contain multiple class labels.
                def check_descendants(_synset):
                    structure_api = 'http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid='
                    structure = read_web_file(structure_api + _synset).split()
                    for line in structure:
                        if line.startswith('-'):  # Child synsets in the structure file starts with '-'.
                            child = line[1:]
                            if child in self.synsets:
                                class_labels[self.wnid2id[line[1:]]] = 1
                            else:
                                check_descendants(child)

                check_descendants(synset)
        # Cache the class labels for this synset.
        self._class_label_cache = class_labels
        return class_labels

    def __getitem__(self, index):
        cur_frame = self._frames[index]
        prev_frame = cur_frame[:-6] + str(int(cur_frame[-6:]) - 1).zfill(6)

        # Read annotation of the target in the current frame and the previous frame.
        cur_annotation_fn = self._data_dir + '/Annotations/VID/' + self._subset + '/' + cur_frame + '.xml'
        cur_annotation = read_annotation(cur_annotation_fn)
        prev_annotation_fn = self._data_dir + '/Annotations/VID/' + self._subset + '/' + prev_frame + '.xml'
        prev_annotation = read_annotation(prev_annotation_fn)

        # Randomly pick an object that appears in both the current frame and the previous frame.
        valid_objs = []
        obj_map = {}
        for idx, obj in enumerate(cur_annotation['objects']):
            obj_map[obj['name']] = idx
        for idx, obj in enumerate(prev_annotation['objects']):
            if obj['name'] in obj_map:
                valid_objs.append((obj_map[obj['name']], idx))
        if len(valid_objs) < 1:
            return self[np.random.randint(len(self))]  # Randomly pick another sample.
        cur_obj_idx, prev_obj_idx = valid_objs[np.random.randint(0, len(valid_objs))]

        # Resolve class labels.
        name = cur_annotation['objects'][cur_obj_idx]['name']
        if name in self._class_label_cache:
            class_labels = self._class_label_cache[name]
        else:
            class_labels = self._resolve_class_label(name)

        # Scale factor of the interest region to the bounding box.
        scale_factor = np.random.uniform(0.7, 1.1) if self._subset == 'train' else 1

        # Crop the patch of the target in the current frame.
        xmin = cur_annotation['objects'][cur_obj_idx]['xmin']
        xmax = cur_annotation['objects'][cur_obj_idx]['xmax']
        ymin = cur_annotation['objects'][cur_obj_idx]['ymin']
        ymax = cur_annotation['objects'][cur_obj_idx]['ymax']
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
        prev_xmin = prev_annotation['objects'][prev_obj_idx]['xmin']
        prev_xmax = prev_annotation['objects'][prev_obj_idx]['xmax']
        prev_ymin = prev_annotation['objects'][prev_obj_idx]['ymin']
        prev_ymax = prev_annotation['objects'][prev_obj_idx]['ymax']
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
               torch.tensor(class_labels), \
               torch.tensor([bbox_x, bbox_y, bbox_width, bbox_height]), \
               torch.tensor([pos_bbox_x, pos_bbox_y, pos_bbox_width, pos_bbox_height])

    def __len__(self):
        return len(self._frames)
