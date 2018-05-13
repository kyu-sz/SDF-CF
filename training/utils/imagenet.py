import math
from typing import Callable

import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageOps

from .img_loader import default_loader
from .utils import *


class ImageNetDataset(data.Dataset):
    def __init__(self,
                 data_dir: str,
                 transform=None,
                 loader: Callable[[str], Image.Image] = default_loader):
        # Load synsets.
        self.synsets, self.wnid2id = load_synsets()
        self.num_classes = len(self.synsets)

        self._transform = transform if transform is not None else transforms.ToTensor()
        self._loader = loader
        self._data_dir = data_dir
        print('Loading ImageNet from {}...'.format(self._data_dir))

        self._annotation_dir = data_dir + '/Annotation'
        self._image_dir = data_dir + '/Image'

        # Count number of annotated samples in each synset.
        self.synset_sizes = [0] * self.num_classes
        for idx, synset in enumerate(self.synsets):
            if os.path.isdir(self.annotation_dir(synset)):
                self.synset_sizes[idx] = len(os.listdir(self.annotation_dir(synset)))
            else:
                self.synset_sizes[idx] = 0

        # Count the ending index of samples in each synset in the global indexing.
        self._idx_end = np.cumsum(self.synset_sizes)

    def image_dir(self, wnid: str = ''):
        return osp.join(self._image_dir, wnid)

    def annotation_dir(self, wnid: str = ''):
        return osp.join(self._annotation_dir, wnid)

    def __getitem__(self, index: int):
        # Read the annotation file for the frame.
        cid = int(np.searchsorted(self._idx_end, index + 1))  # Class ID of the indexed sample.
        synset = self.synsets[cid]  # WordNet ID of the class.
        idx_in_class = index - (self._idx_end[cid - 1] if cid >= 1 else 0)  # Index of the sample in the class.
        annotation_fn = sorted(os.listdir(self.annotation_dir(synset)))[idx_in_class]  # Retrieve the annotation file.
        img_annotation = read_annotation(osp.join(self.annotation_dir(synset), annotation_fn))  # Read the annotation.

        # Construct 0-1 class label vector.
        class_labels = [0] * self.num_classes
        class_labels[cid] = 1

        # Find the annotation for the object.
        obj_annotation = None
        for obj in img_annotation['objects']:
            if obj['name'] == synset:
                obj_annotation = obj
                break
        if obj_annotation is None:
            return self[np.random.randint(len(self))]

        # Load the image.
        img_path = osp.join(self.image_dir(img_annotation['folder']), img_annotation['filename'] + '.JPEG')
        try:  # Try loading it locally first.
            img = self._loader(img_path)
        except OSError:
            # The image does not exist locally or is problematic. Try downloading it from web.
            mapping_api = "http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid="
            try:
                mappings = read_web_file(mapping_api + synset).split('\n')
            except IOError:
                print('Mapping API broken for synset {}!'.format(synset))
                return self[np.random.randint(len(self))]  # Randomly pick another sample.
            for line in mappings:
                elements = line.split()
                if len(elements) == 2:
                    name, url = elements
                    if name == img_annotation['filename']:
                        ret = download_img(url,
                                           self.image_dir(img_annotation['folder']),
                                           img_annotation['filename'])
                        if not ret:  # The image is not available on the web.
                            return self[np.random.randint(len(self))]  # Randomly pick another sample.
                        break
            try:
                img = self._loader(img_path)
            except OSError:  # The image is not available on the web.
                return self[np.random.randint(len(self))]  # Randomly pick another sample.

        # Randomly flip the image.
        if np.random.randint(0, 2):
            img = ImageOps.mirror(img)
            obj_annotation['xmin'] = img_annotation['width'] - obj_annotation['xmax']
            obj_annotation['xmax'] = img_annotation['width'] - obj_annotation['xmin']
            obj_annotation['ymin'] = img_annotation['height'] - obj_annotation['ymax']
            obj_annotation['ymax'] = img_annotation['height'] - obj_annotation['ymin']

        # Crop a randomly scaled square patch of the object.
        scale_factor = np.random.uniform(0.7, 1.1)
        x_mid = (obj_annotation['xmin'] + obj_annotation['xmax']) * 0.5  # Middle of x-axis.
        y_mid = (obj_annotation['ymin'] + obj_annotation['ymax']) * 0.5  # Middle of y-axis.
        bbox_width = obj_annotation['xmax'] - obj_annotation['xmin']
        bbox_height = obj_annotation['ymax'] - obj_annotation['ymin']
        shorter_side_len = min(img_annotation['width'], img_annotation['height'])
        target_patch_size = max(min(math.ceil(max(bbox_width, bbox_height) * scale_factor), shorter_side_len), 7)
        target_xmin = min(img_annotation['width'] - target_patch_size,
                          max(0, int(x_mid - target_patch_size * (0.5 + np.random.uniform(-0.1, 0.1)))))
        target_ymin = min(img_annotation['height'] - target_patch_size,
                          max(0, int(y_mid - target_patch_size * (0.5 + np.random.uniform(-0.1, 0.1)))))
        target_xmax = target_xmin + target_patch_size
        target_ymax = target_ymin + target_patch_size
        target = img.crop((target_xmin, target_ymin, target_xmax, target_ymax))

        # Calculate bounding box regression target.
        bbox_x = (obj_annotation['xmin'] + obj_annotation['xmax'] - target_xmin - target_xmax) * 0.5 / target_patch_size
        bbox_y = (obj_annotation['ymin'] + obj_annotation['ymax'] - target_ymin - target_ymax) * 0.5 / target_patch_size
        bbox_width = (obj_annotation['xmax'] - obj_annotation['xmin']) / target_patch_size - 1
        bbox_height = (obj_annotation['ymax'] - obj_annotation['ymin']) / target_patch_size - 1

        # Create a positive sample for smoothness training by rotating the target.
        # TODO: Check implementation of rotation.
        pos_sample = img.rotate(np.random.randint(-15, 16), center=(x_mid, y_mid)) \
            .crop((target_xmin, target_ymin, target_xmax, target_ymax))

        # Create a negative sample for smoothness training by randomly scaling and shifting the bounding box.
        neg_patch_size = int(target_patch_size
                             * np.random.uniform(max(0.5, 7. / target_patch_size),
                                                 min(1.5, float(shorter_side_len) / target_patch_size)))
        neg_patch_xmin = min(max(target_xmin + target_patch_size * np.random.uniform(-0.4, 0.4), 0),
                             img_annotation['width'] - neg_patch_size)
        neg_patch_ymin = min(max(target_ymin + target_patch_size * np.random.uniform(-0.4, 0.4), 0),
                             img_annotation['height'] - neg_patch_size)
        neg_patch_xmax = neg_patch_xmin + neg_patch_size
        neg_patch_ymax = neg_patch_ymin + neg_patch_size
        neg_sample = img.crop((neg_patch_xmin, neg_patch_ymin, neg_patch_xmax, neg_patch_ymax))

        return self._transform(target), \
               self._transform(pos_sample), \
               self._transform(neg_sample), \
               torch.tensor(class_labels), \
               torch.tensor([bbox_x, bbox_y, bbox_width, bbox_height]), \
               torch.tensor([bbox_x, bbox_y, bbox_width, bbox_height])

    def __len__(self):
        return sum(self.synset_sizes)


def update_imagenet_annotations(imagenet_dir: str, synset: str = None) -> None:
    if not os.path.isdir(imagenet_dir):
        if os.path.isfile(imagenet_dir):
            print('{} is a file!'.format(imagenet_dir))
            return
        else:
            os.makedirs(imagenet_dir)

    anno_urls_api = "http://www.image-net.org/api/download/imagenet.bbox.synset?wnid="
    synsets, _ = load_synsets()
    tmp_arch_storage = '/tmp/imagenet_update'
    os.makedirs(tmp_arch_storage, exist_ok=True)

    def download(synset: str) -> None:
        anno_arch_path = os.path.join(tmp_arch_storage, synset + '.tar.gz')
        try:
            download_web_file(anno_urls_api + synset, anno_arch_path)
        except IOError:
            return
        ret = extract_archive(anno_arch_path, imagenet_dir)
        if not ret:
            print('Error when processing archive of {}!'.format(synset))
        os.remove(anno_arch_path)

    if synset is None:
        print('Downloading latest annotations from ImageNet for all synsets...')
        import threading
        import queue
        threads = queue.Queue()
        finished_cnt = 0
        num_workers = 32
        for synset in synsets:
            t = threading.Thread(target=download, name=synset, args=[synset])
            t.start()
            threads.put(t)
            if threads.qsize() >= num_workers:
                threads.get().join()
                finished_cnt += 1
                if finished_cnt % num_workers == 0:
                    print('Processed {}/{}'.format(finished_cnt, len(synsets)))
        while not threads.empty():
            threads.get().join()
            finished_cnt += 1
            print('Processed {}/{}'.format(finished_cnt, len(synsets)))
    else:
        print('Downloading latest annotations from ImageNet for synset {}...'.format(synset))
        download(synset)

    os.rmdir(tmp_arch_storage)


def main() -> None:
    # Download latest annotations from ImageNet.
    import sys
    if len(sys.argv) < 2:
        print('Usage: {} <ImageNet Directory> [<synset>]'.format(sys.argv[0]))
        return
    update_imagenet_annotations(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else None)


if __name__ == "__main__":
    main()
