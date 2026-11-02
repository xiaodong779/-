from copy import deepcopy
import torch.nn as nn
from torch.utils import checkpoint as cp

# 【核心修改 1】导入你刚才新建的 DGAM 模块，实现分身
from ..common import DGAM 
from ..registry import BACKBONES
from .resnet import Bottleneck, ResNet


class DGAMBlock(nn.Module): 
    """Dynamic Granularity Aggregation Block (DGAM-Block).
    
    该模块将 DGAM 动态注入到 ResNet 的 Bottleneck 结构中，
    通过 1x1 卷积之后、3x3 卷积之前的黄金位置进行动作粒度提纯。
    """

    def __init__(self, block, num_segments, dgam_cfg=dict()):
        super().__init__()
        self.dgam_cfg = deepcopy(dgam_cfg)
        self.block = block
        self.num_segments = num_segments
        
        # 【核心修改 2】装载全新的 DGAM 零件
        self.dgam = DGAM(
            in_channels=block.conv1.out_channels,
            num_segments=num_segments
        )

        if not isinstance(self.block, Bottleneck):
            raise NotImplementedError('DGAM-Blocks have not been fully '
                                      'implemented except the pattern based '
                                      'on Bottleneck block.')

    def forward(self, x):
        if isinstance(self.block, Bottleneck):

            def _inner_forward(x):
                """Forward wrapper for utilizing checkpoint."""
                identity = x

                # 1. 第一层 1x1 卷积
                out = self.block.conv1(x)
                
                # 【核心修改 3】调用全新的 DGAM 处理特征
                # 此时 alpha=0 初始保护机制生效，确保训练初期 100% 稳定
                out = self.dgam(out)
                
                # 2. 核心 3x3 卷积与最后的 1x1 卷积
                out = self.block.conv2(out)
                out = self.block.conv3(out)

                if self.block.downsample is not None:
                    identity = self.block.downsample(x)

                out = out + identity

                return out

            if self.block.with_cp and x.requires_grad:
                out = cp.checkpoint(_inner_forward, x)
            else:
                out = _inner_forward(x)

            out = self.block.relu(out)

            return out


@BACKBONES.register_module()
class ResNet_DGPN(ResNet):
    """Dynamic Granularity Pyramid Network (DGPN) backbone.
    
    融合了 DGAM 的动态粒度聚合机制，专为配合 AMS-Net 提升动作识别精度设计。
    """

    def __init__(self, depth, num_segments, dgam_cfg=dict(), **kwargs):
        super().__init__(depth, **kwargs)
        assert num_segments >= 3
        self.num_segments = num_segments
        self.dgam_cfg = deepcopy(dgam_cfg)

    def init_weights(self):
        """权重初始化，并触发 DGAM 的装配逻辑"""
        super().init_weights()
        self.make_dgam_modeling()

    def make_dgam_modeling(self):
        """用全新的 DGAMBlock 替换原生 ResNet 的 Bottleneck。"""

        def make_dgam_block(stage, num_segments, dgam_cfg=dict()):
            blocks = list(stage.children())
            for i, block in enumerate(blocks):
                # 【核心修改 4】将所有的 Block 替换为 DGAM 增强版本
                blocks[i] = DGAMBlock(block, num_segments, deepcopy(dgam_cfg))
            return nn.Sequential(*blocks)

        for i in range(self.num_stages):
            layer_name = f'layer{i + 1}'
            res_layer = getattr(self, layer_name)
            setattr(self, layer_name,
                    make_dgam_block(res_layer, self.num_segments, self.dgam_cfg))