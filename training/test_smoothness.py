import numpy as np
from utils import VOTLoader
import cv2
import torch
from models.VGGM import VGGM
from torch.autograd import Variable
from matplotlib import pyplot as plt

if __name__ == '__main__':
    mdnet = VGGM(model_path='models/mdnet_vot-otb.pth')
    vgg16 = VGGM(model_path='models/imagenet-vgg-m-2048.mat')

    data_dir = '../datasets/vot2017'
    mdnet_total_dist = 0
    vgg16_total_dist = 0
    for seq in VOTLoader.VOTLoader(data_dir):
        print('Processing {}'.format(seq['name']))
        mdnet_last_feat = None
        vgg16_last_feat = None
        mdnet_dist = []
        vgg16_dist = []
        for ind, frame in enumerate(seq['images']):
            input_tensor = np.zeros((1, 3, 224, 224), dtype=np.float32)
            input_tensor[0, :] = np.transpose(
                cv2.cvtColor(
                    cv2.resize(frame, (224, 224)),
                    cv2.COLOR_BGR2RGB),
                (2, 0, 1))

            feat = mdnet.forward(Variable(torch.from_numpy(input_tensor),
                                          requires_grad=False))
            feat /= torch.sum(feat)
            if mdnet_last_feat is not None:
                mdnet_dist.append(torch.sum(torch.pow(feat - mdnet_last_feat, 2)).data.cpu().numpy()[0])
            mdnet_last_feat = feat

            feat = vgg16.forward(Variable(torch.from_numpy(input_tensor),
                                          requires_grad=False))
            feat /= torch.sum(feat)
            if vgg16_last_feat is not None:
                vgg16_dist.append(torch.sum(torch.pow(feat - vgg16_last_feat, 2)).data.cpu().numpy()[0])
            vgg16_last_feat = feat
            vgg16_total_dist += sum(vgg16_dist)

            # cv2.imshow(seq['name'], frame)
            # cv2.waitKey(1)

        mdnet_seq_dist = np.average(mdnet_dist)
        mdnet_total_dist += mdnet_seq_dist
        vgg16_seq_dist = np.average(vgg16_dist)
        vgg16_total_dist += vgg16_seq_dist

        print('Average distance of MDNet on {}: {}'.format(seq['name'], mdnet_seq_dist))
        print('Average distance of VGG16 on {}: {}'.format(seq['name'], vgg16_seq_dist))

        # plt.figure(seq['name'] + " - MDNet")
        # plt.plot(mdnet_dist)
        # plt.pause(0.001)

        # cv2.destroyWindow(seq['name'])

    print('Average distance of MDNet: {}'.format(mdnet_total_dist))
    print('Average distance of VGG16: {}'.format(vgg16_total_dist))
