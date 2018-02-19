import numpy as np
from utils import VOTLoader
import cv2
import torch

if __name__ == '__main__':
    data_dir = '../datasets/vot2017'
    for seq in VOTLoader.VOTLoader(data_dir):
        for frame in seq['images']:
            cv2.imshow(seq['name'], frame)
            cv2.waitKey()
        cv2.destroyWindow(seq['name'])