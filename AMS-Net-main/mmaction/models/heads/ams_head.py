import torch.nn as nn
import torch

#from ..registry import HEADS
from ..builder import HEADS
from .tsn_head import TSNHead


@HEADS.register_module()
class AMSHead(TSNHead):
    """Class head for AMS.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: https://arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool3d = None

        self.avg_pool2d = None
        self.new_cls = None

    def _init_new_cls(self):
        self.new_cls = nn.Conv3d(self.in_channels, self.num_classes, 1, 1, 0)
        if next(self.fc_cls.parameters()).is_cuda:
            self.new_cls = self.new_cls.cuda()
        self.new_cls.weight.copy_(self.fc_cls.weight[..., None, None, None])
        self.new_cls.bias.copy_(self.fc_cls.bias)

    def forward(self, x, num_segs=None, fcn_test=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int | None): Number of segments into which a video
                is divided. Default: None.
            fcn_test (bool): Whether to apply full convolution (fcn) testing.
                Default: False.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if fcn_test:
            if self.avg_pool3d:
                x = self.avg_pool3d(x)
            if self.new_cls is None:
                self._init_new_cls()
            cls_score_feat_map = self.new_cls(x)
            return cls_score_feat_map

        if self.avg_pool2d is None:
            kernel_size = (1, x.shape[-2], x.shape[-1])
            self.avg_pool2d = nn.AvgPool3d(kernel_size, stride=1, padding=0)

        if num_segs is None:
            # SlowOnly 连续帧：输入 5D [N, C, T, H, W]，T=8 在时序维。
            # 用 AdaptiveAvgPool3d 对 T,H,W 平均池化后直接分类（多 clip 平均由
            # Recognizer3D._do_test 的 average_clip 处理）。
            x = self.avg_pool3d(x)      # [N, C, 1, 1, 1]
            x = torch.flatten(x, 1)     # [N, C]
            if self.dropout is not None:
                x = self.dropout(x)
            cls_score = self.fc_cls(x)  # [N, num_classes]
            return cls_score
        else:
            # TSN 多片段：输入 [N * num_segs, in_channels, 1, 7, 7]（clip 在 batch 维）
            if self.spatial_type == 'avg':
                x = self.avg_pool2d(x)          # [N * num_segs, in_channels, 1, 1]
            x = torch.flatten(x, 1)             # [N * num_segs, in_channels]
            if self.dropout is not None:
                x = self.dropout(x)
            cls_score = self.fc_cls(x)          # [N * num_segs, num_classes]
            cls_score = cls_score.view((-1, num_segs) + cls_score.size()[1:])
            cls_score = self.consensus(cls_score)
            cls_score = cls_score.squeeze(1)
            return cls_score
