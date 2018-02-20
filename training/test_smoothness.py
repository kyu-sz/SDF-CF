import numpy as np
from utils import VOTLoader
import cv2
import torch
from models.VGGM import VGGM
from torch.autograd import Variable
from matplotlib import pyplot as plt


def pt2tuple(pt):
    return int(pt[0]), int(pt[1])


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
            canvas = frame.copy()
            b = (255, 0, 0)
            g = (0, 255, 0)
            r = (0, 0, 255)

            bbox = seq['bbox'][ind]
            if len(bbox) == 8:
                p0 = bbox[0:2]
                p1 = bbox[2:4]
                p2 = bbox[4:6]
                p3 = bbox[6:8]

                centroid = (p0 + p1 + p2 + p3) / 4
                ext_p0 = 2 * p0 - centroid
                ext_p1 = 2 * p1 - centroid
                ext_p2 = 2 * p2 - centroid
                ext_p3 = 2 * p3 - centroid

                average_dist = np.average((np.linalg.norm(p0 - centroid),
                                           np.linalg.norm(p1 - centroid),
                                           np.linalg.norm(p2 - centroid),
                                           np.linalg.norm(p3 - centroid)))
                offset = average_dist
                lu = np.maximum(0, np.minimum([frame.shape[1], frame.shape[0]],
                                              centroid + np.array((-offset, -offset))))
                ll = np.maximum(0, np.minimum([frame.shape[1], frame.shape[0]],
                                              centroid + np.array((-offset, offset))))
                ru = np.maximum(0, np.minimum([frame.shape[1], frame.shape[0]],
                                              centroid + np.array((offset, -offset))))
                rl = np.maximum(0, np.minimum([frame.shape[1], frame.shape[0]],
                                              centroid + np.array((offset, offset))))

                if lu[1] == ll[1] or lu[0] == ru[0]:
                    continue

                # cv2.line(canvas, pt2tuple(p0), pt2tuple(p1), b)
                # cv2.line(canvas, pt2tuple(p1), pt2tuple(p2), b)
                # cv2.line(canvas, pt2tuple(p2), pt2tuple(p3), b)
                # cv2.line(canvas, pt2tuple(p3), pt2tuple(p0), b)
                # cv2.line(canvas, pt2tuple(ext_p0), pt2tuple(ext_p1), g)
                # cv2.line(canvas, pt2tuple(ext_p1), pt2tuple(ext_p2), g)
                # cv2.line(canvas, pt2tuple(ext_p2), pt2tuple(ext_p3), g)
                # cv2.line(canvas, pt2tuple(ext_p3), pt2tuple(ext_p0), g)
                # cv2.line(canvas, pt2tuple(lu), pt2tuple(ll), r)
                # cv2.line(canvas, pt2tuple(ll), pt2tuple(rl), r)
                # cv2.line(canvas, pt2tuple(rl), pt2tuple(ru), r)
                # cv2.line(canvas, pt2tuple(ru), pt2tuple(lu), r)

                target = frame[int(lu[1]):int(ll[1]), int(lu[0]):int(ru[0]), :]
            else:
                continue
                p0 = bbox[1:-1:-1]
                p1 = bbox[3:1:-1]
                cv2.line(canvas, pt2tuple(p0), pt2tuple(p1), r)
                x_min = int(min(p0[0], p1[0]))
                x_max = int(max(p0[0], p1[0]))
                y_min = int(min(p0[1], p1[1]))
                y_max = int(max(p0[1], p1[1]))
                target = frame[y_min:y_max, x_min:x_max, :]
            #
            # cv2.imshow(seq['name'] + ' origin', canvas)
            # cv2.imshow(seq['name'] + ' target', target)
            # cv2.waitKey(1)

            input_tensor = np.zeros((1, 3, 224, 224), dtype=np.float32)
            input_tensor[0, :] = np.transpose(
                cv2.cvtColor(
                    cv2.resize(target, (224, 224)),
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

        if len(mdnet_dist) > 0:
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
