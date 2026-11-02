from copy import deepcopy

import torch.nn as nn
from torch.utils import checkpoint as cp

# 【修改 1】导入你刚才新建的 AGPM 零件
from ..common import AGPM 
from ..registry import BACKBONES
from .resnet import Bottleneck, ResNet


class AGPMBlock(nn.Module): # 【修改 2】名字改为 AGPMBlock
    """Action Granularity Pyramid Block (AGPM-Block)."""

    def __init__(self, block, num_segments, agpm_cfg=dict()):
        super().__init__()
        self.agpm_cfg = deepcopy(agpm_cfg)
        self.block = block
        self.num_segments = num_segments
        
        # 【修改 3】真正地把你造的 AGPM 零件装载进来！
        self.agpm = AGPM(
            in_channels=block.conv1.out_channels,
            num_segments=num_segments
            # 注意：原 TAM 有 **self.tam_cfg，咱们暂不需要，所以去掉了
        )

        if not isinstance(self.block, Bottleneck):
            raise NotImplementedError('AGPM-Blocks have not been fully '
                                      'implemented except the pattern based '
                                      'on Bottleneck block.')

    def forward(self, x):
        if isinstance(self.block, Bottleneck):

            def _inner_forward(x):
                """Forward wrapper for utilizing checkpoint."""
                identity = x

                out = self.block.conv1(x)
                
                # 【修改 4】调用你的 AGPM 零件处理特征
                out = self.agpm(out)
                
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
class ResNet_AGPN(ResNet):
    """Action Granularity Pyramid Network (AGPN) backbone."""

    def __init__(self, depth, num_segments, agpm_cfg=dict(), **kwargs):
        super().__init__(depth, **kwargs)
        assert num_segments >= 3
        self.num_segments = num_segments
        self.agpm_cfg = deepcopy(agpm_cfg)

    def init_weights(self):
        super().init_weights()
        self.make_agpm_modeling()

    def make_agpm_modeling(self):
        """Replace ResNet-Block with AGPM-Block."""

        def make_agpm_block(stage, num_segments, agpm_cfg=dict()):
            blocks = list(stage.children())
            for i, block in enumerate(blocks):
                # 【修改 5】把装配线上的 block 全换成咱们刚才写的 AGPMBlock
                blocks[i] = AGPMBlock(block, num_segments, deepcopy(agpm_cfg))
            return nn.Sequential(*blocks)

        for i in range(self.num_stages):
            layer_name = f'layer{i + 1}'
            res_layer = getattr(self, layer_name)
            setattr(self, layer_name,
                    make_agpm_block(res_layer, self.num_segments, self.agpm_cfg))