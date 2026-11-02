# 文件路径: /root/lanyun-fs/AMS-Net-main/mmaction/models/losses/label_smooth_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class LabelSmoothCrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss with Label Smoothing.

    Args:
        label_smooth_eps (float): Label smoothing factor. Default: 0.1.
        loss_weight (float): Factor scalar multiplied on the loss. Default: 1.0.
    """

    def __init__(self, label_smooth_eps=0.1, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        self.label_smooth_eps = label_smooth_eps

    def _forward(self, cls_score, label, **kwargs):
        num_classes = cls_score.size(1)

        # 转成 one-hot
        label_one_hot = torch.zeros_like(cls_score).scatter_(
            1, label.unsqueeze(1), 1)

        # label smoothing
        label_one_hot = label_one_hot * (1 - self.label_smooth_eps) + \
            self.label_smooth_eps / num_classes

        log_prob = F.log_softmax(cls_score, dim=1)

        loss = -(label_one_hot * log_prob).sum(dim=1).mean()

        return loss
