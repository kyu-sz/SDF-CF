import torch
import torch.nn as nn


class SmoothnessLoss(nn.Module):
    def __init__(self, diff_bound=0.1):
        super(SmoothnessLoss, self).__init__()
        self._error_bound = diff_bound

    def forward(self, x, pos, neg):
        return torch.mean(torch.abs(x - pos))\
               - torch.mean(torch.clamp(torch.abs(x - neg), min=0, max=self._error_bound))
