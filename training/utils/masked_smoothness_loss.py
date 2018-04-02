import torch
import torch.nn as nn


class MaskedSmoothnessLoss(nn.Module):
    def __init__(self, diff_bound=1):
        super(MaskedSmoothnessLoss, self).__init__()
        self._diff_bound = diff_bound

    def forward(self, x, pos, neg, bbox, pos_bbox):
        patch_width = x.shape[-1]
        patch_height = x.shape[-2]

        bbox_width = patch_width * bbox[2]
        bbox_height = patch_height * bbox[3]
        bbox_xmin = patch_width * 0.5 - patch_width * bbox[0]
        bbox_ymin = patch_height * 0.5 - patch_height * bbox[1]
        bbox_xmax = bbox_xmin + bbox_width
        bbox_ymax = bbox_ymin + bbox_height

        pos_bbox_width = patch_width * pos_bbox[2]
        pos_bbox_height = patch_height * pos_bbox[3]
        pos_bbox_xmin = patch_width * 0.5 - patch_width * pos_bbox[0]
        pos_bbox_ymin = patch_height * 0.5 - patch_height * pos_bbox[1]
        pos_bbox_xmax = pos_bbox_xmin + pos_bbox_width
        pos_bbox_ymax = pos_bbox_ymin + pos_bbox_height

        intersect_xmin = max(bbox_xmin, pos_bbox_xmin)
        intersect_xmax = min(bbox_xmax, pos_bbox_xmax)
        intersect_ymin = max(bbox_ymin, pos_bbox_ymin)
        intersect_ymax = min(bbox_ymax, pos_bbox_ymax)

        return torch.mean(torch.abs(x[..., intersect_ymin:intersect_ymax, intersect_xmin:intersect_xmax]
                                    - pos[..., intersect_ymin:intersect_ymax, intersect_xmin:intersect_xmax]))\
               - torch.mean(torch.clamp(torch.abs(x[..., bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]
                                                  - neg[..., bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]),
                                        min=0, max=self._diff_bound))