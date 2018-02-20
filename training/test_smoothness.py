import numpy as np
from utils import VOTLoader
import cv2
import torch
from models.MDNet import MDNet
from torch.autograd import Variable

if __name__ == '__main__':
    mdnet = MDNet('models/mdnet_vot-otb.pth')

    data_dir = '../datasets/vot2017'
    for seq in VOTLoader.VOTLoader(data_dir):
        input = np.zeros((len(seq['images']), 3, 224, 224), dtype=np.float32)
        for ind, frame in enumerate(seq['images']):
            input[ind, :, :, :] = np.transpose(cv2.resize(frame, (224, 224)), (2, 0, 1))
            cv2.imshow("B", input[ind, 0, :, :] / 256)
            cv2.imshow("G", input[ind, 1, :, :] / 256)
            cv2.imshow("R", input[ind, 2, :, :] / 256)
            cv2.imshow("Resized", cv2.resize(frame, (224, 224)))
            cv2.imshow(seq['name'], frame)
            cv2.waitKey(1)

        mdnet_conv3_features = mdnet.forward(Variable(torch.from_numpy(input),
                                                      requires_grad=False))
        pretrained_conv3_features = []

        cv2.destroyWindow(seq['name'])
