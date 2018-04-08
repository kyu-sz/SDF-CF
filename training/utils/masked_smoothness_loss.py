import torch
import torch.nn as nn


class MaskedSmoothnessLoss(nn.Module):
    def __init__(self, diff_bound=0.005):
        super(MaskedSmoothnessLoss, self).__init__()
        self._diff_bound = diff_bound

    def forward(self, x, pos, neg, bbox, pos_bbox):
        patch_width = x.shape[-1]
        patch_height = x.shape[-2]

        bbox_width = (patch_width * (1 + bbox[:, 2])).int()
        bbox_height = (patch_height * (1 + bbox[:, 3])).int()
        bbox_xmin = (patch_width * 0.5 - patch_width * bbox[:, 0]).int()
        bbox_ymin = (patch_height * 0.5 - patch_height * bbox[:, 1]).int()
        bbox_xmax = bbox_xmin + bbox_width
        bbox_ymax = bbox_ymin + bbox_height

        pos_bbox_width = (patch_width * (1 + pos_bbox[:, 2])).int()
        pos_bbox_height = (patch_height * (1 + pos_bbox[:, 3])).int()
        pos_bbox_xmin = (patch_width * 0.5 - patch_width * pos_bbox[:, 0]).int()
        pos_bbox_ymin = (patch_height * 0.5 - patch_height * pos_bbox[:, 1]).int()
        pos_bbox_xmax = pos_bbox_xmin + pos_bbox_width
        pos_bbox_ymax = pos_bbox_ymin + pos_bbox_height

        intersect_xmin = torch.max(bbox_xmin, pos_bbox_xmin)
        intersect_xmax = torch.min(bbox_xmax, pos_bbox_xmax)
        intersect_ymin = torch.max(bbox_ymin, pos_bbox_ymin)
        intersect_ymax = torch.min(bbox_ymax, pos_bbox_ymax)
        intersect_width = intersect_xmax - intersect_xmin
        intersect_height = intersect_ymax - intersect_ymin

        pos_loss = torch.autograd.Variable(torch.zeros(1), requires_grad=False).cuda(async=True)
        neg_loss = torch.autograd.Variable(torch.zeros(1), requires_grad=False).cuda(async=True)

        for i in range(x.shape[0]):
            if bbox_width[i] > 0 and bbox_height[i] > 0:
                neg_loss -= torch.mean(torch.clamp(
                    torch.abs(x[i, :, bbox_ymin[i]:bbox_ymax[i], bbox_xmin[i]:bbox_xmax[i]]
                              - neg[i, :, bbox_ymin[i]:bbox_ymax[i], bbox_xmin[i]:bbox_xmax[i]]),
                    min=0, max=self._diff_bound))
                if intersect_width[i] > 0 and intersect_height[i] > 0:
                    pos_loss += torch.mean(
                        torch.abs(x[i, :, intersect_ymin[i]:intersect_ymax[i], intersect_xmin[i]:intersect_xmax[i]]
                                  - pos[i, :, intersect_ymin[i]:intersect_ymax[i], intersect_xmin[i]:intersect_xmax[i]]))
        return pos_loss, neg_loss
