import torch
import torch.nn as nn


class SmoothnessLoss(nn.Module):
    def __init__(self, diff_bound=0.0016):
        super(SmoothnessLoss, self).__init__()
        self._diff_bound = diff_bound

    def forward(self, target, pos_sample, neg_sample, target_bbox, pos_bbox):
        patch_width = target.shape[-1]
        patch_height = target.shape[-2]

        target_w = torch.clamp(torch.ceil(patch_width * (1 + target_bbox[:, 2])).int(),
                               1, patch_width)
        target_h = torch.clamp(torch.ceil(patch_height * (1 + target_bbox[:, 3])).int(),
                               1, patch_height)
        target_xmin = torch.clamp((patch_width * target_bbox[:, 0]).int(),
                                  0, patch_width - 1)
        target_ymin = torch.clamp((patch_height * target_bbox[:, 1]).int(),
                                  0, patch_height - 1)
        target_xmax = torch.clamp(target_xmin + target_w,
                                  0, patch_width - 1)
        target_ymax = torch.clamp(target_ymin + target_h,
                                  0, patch_height - 1)

        pos_w = torch.clamp(torch.ceil(patch_width * (1 + pos_bbox[:, 2])).int(),
                            1, patch_width)
        pos_h = torch.clamp(torch.ceil(patch_height * (1 + pos_bbox[:, 3])).int(),
                            1, patch_height)
        pos_xmin = torch.clamp((patch_width * pos_bbox[:, 0]).int(),
                               0, patch_width - 1)
        pos_ymin = torch.clamp((patch_height * pos_bbox[:, 1]).int(),
                               0, patch_height - 1)
        pos_xmax = torch.clamp(pos_xmin + pos_w,
                               0, patch_width - 1)
        pos_ymax = torch.clamp(pos_ymin + pos_h,
                               0, patch_height - 1)

        intersect_xmin = torch.max(target_xmin, pos_xmin)
        intersect_xmax = torch.min(target_xmax, pos_xmax)
        intersect_ymin = torch.max(target_ymin, pos_ymin)
        intersect_ymax = torch.min(target_ymax, pos_ymax)
        intersect_width = intersect_xmax - intersect_xmin
        intersect_height = intersect_ymax - intersect_ymin

        pos_loss = torch.autograd.Variable(torch.zeros(1), requires_grad=False).cuda(async=True)
        neg_loss = torch.autograd.Variable(torch.zeros(1), requires_grad=False).cuda(async=True)

        weight_sum = torch.autograd.Variable(torch.zeros(1), requires_grad=False).cuda(async=True)
        for i in range(target.shape[0]):
            if intersect_width[i] > 0 and intersect_height[i] > 0:
                # Calculate the weight (relevance between the feature map and the target)
                # by the mean response intensity on the feature map.
                weight = torch.clamp(
                        torch.mean(target[i, :, target_ymin[i]:target_ymax[i], target_xmin[i]:target_xmax[i]])
                        + torch.mean(pos_sample[i, :, pos_ymin[i]:pos_ymax[i], pos_xmin[i]:pos_xmax[i]])
                        - torch.mean(neg_sample[i, ...]) * 2,
                        min=0).detach()
                # Calculate the loss from the positive pair and the negative pair.
                pos_loss += weight * torch.mean(torch.abs(
                        target[i, :, intersect_ymin[i]:intersect_ymax[i], intersect_xmin[i]:intersect_xmax[i]]
                        - pos_sample[i, :, intersect_ymin[i]:intersect_ymax[i], intersect_xmin[i]:intersect_xmax[i]]))
                neg_loss -= weight * torch.mean(torch.clamp(torch.abs(
                        target[i, :, target_ymin[i]:target_ymax[i], target_xmin[i]:target_xmax[i]]
                        - neg_sample[i, :, target_ymin[i]:target_ymax[i], target_xmin[i]:target_xmax[i]]),
                        min=0, max=self._diff_bound))
                # Sum up the weight for normalization.
                weight_sum += weight

        if weight_sum.data[0] == 0:
            return pos_loss, neg_loss
        else:
            return pos_loss / weight_sum, neg_loss / weight_sum
