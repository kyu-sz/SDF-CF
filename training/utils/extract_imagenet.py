"""
This script extracts images and annotations from the downloaded ImageNet dataset, .
"""

import os

imagenet_dir = 'datasets/ImageNet'

annotations = os.listdir(imagenet_dir + '/Annotation')
image_data = os.listdir(imagenet_dir + '/fall11_whole')

with open(os.path.dirname(os.path.realpath(__file__)) + '/imagenet_cls.txt', 'r') as f:
    for line in f:
        wnid, cid, name = line.split()
        if (wnid + '.tar.gz') not in annotations:
            print(wnid, cid, name, 'has no annotation!')
        if (wnid + '.tar') not in image_data:
            print(wnid, cid, name, 'has no images!')
