import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.utils import checkpoint as cp

from ..common.pame import PolarityAwareMotionEnhancer
from ..common.mtcm import AdaptiveMTCM
from ...utils import get_root_logger
from ..builder import BACKBONES


class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False):
        super().__init__()
        assert style in ['pytorch', 'caffe']

        self.conv1 = ConvModule(
            inplanes, planes,
            kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, bias=False,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv2 = ConvModule(
            planes, planes,
            kernel_size=3, stride=1, padding=1,
            dilation=1, bias=False,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.style = style
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        assert not with_cp

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet."""
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False):
        super().__init__()
        assert style in ['pytorch', 'caffe']

        self.inplanes = inplanes
        self.planes = planes

        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.conv1 = ConvModule(
            inplanes, planes,
            kernel_size=1, stride=self.conv1_stride, bias=False,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv2 = ConvModule(
            planes, planes,
            kernel_size=3, stride=self.conv2_stride,
            padding=dilation, dilation=dilation, bias=False,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv3 = ConvModule(
            planes, planes * self.expansion,
            kernel_size=1, bias=False,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x):
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   conv_cfg=None,
                   norm_cfg=None,
                   act_cfg=None,
                   with_cp=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = ConvModule(
            inplanes, planes * block.expansion,
            kernel_size=1, stride=stride, bias=False,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

    layers = []
    layers.append(
        block(inplanes, planes, stride, dilation, downsample,
              style=style, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
              act_cfg=act_cfg, with_cp=with_cp))

    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation,
                  style=style, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                  act_cfg=act_cfg, with_cp=with_cp))

    return nn.Sequential(*layers)


@BACKBONES.register_module()
class ResNetAMS(nn.Module):
    """ResNet backbone."""

    arch_settings = {
        18:  (BasicBlock,  (2, 2, 2, 2)),
        34:  (BasicBlock,  (3, 4, 6, 3)),
        50:  (Bottleneck,  (3, 4, 6, 3)),
        101: (Bottleneck,  (3, 4, 23, 3)),
        152: (Bottleneck,  (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 torchvision_pretrain=True,
                 in_channels=3,
                 num_stages=4,
                 out_indices=(3, ),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False,
                 partial_bn=False,
                 with_cp=False,
                 num_segments=8):
        super().__init__()

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.torchvision_pretrain = torchvision_pretrain
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.partial_bn = partial_bn
        self.with_cp = with_cp
        self.num_segments = num_segments

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        # ── ResNet 各 Stage ──────────────────────────────────────
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                self.block, self.inplanes, planes, num_blocks,
                stride=strides[i], dilation=dilations[i],
                style=self.style, conv_cfg=conv_cfg,
                norm_cfg=norm_cfg, act_cfg=act_cfg, with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2 ** (
            len(self.stage_blocks) - 1)

        # ── PAME：layer1 之后（通道 256）────────────────────────
        self.pame = PolarityAwareMotionEnhancer(
            channels=256, num_segments=self.num_segments)

        # ── AdaptiveMTCM：替换原 CPTM，挂在 layer1/layer3 之后 ──
        # 对应论文位置 (c)：1×1 → 3×3 → MTCM → 1×1
        self.mtcm_res2 = AdaptiveMTCM(
            channels=256,  n_frames=self.num_segments)   # layer1 输出
        self.mtcm_res4 = AdaptiveMTCM(
            channels=1024, n_frames=self.num_segments)   # layer3 输出

    def _make_stem_layer(self):
        self.conv1 = ConvModule(
            self.in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # ── 预训练权重加载（保持原版不变）───────────────────────────
    @staticmethod
    def _load_conv_params(conv, state_dict_tv, module_name_tv,
                          loaded_param_names):
        weight_tv_name = module_name_tv + '.weight'
        if conv.weight.data.shape == state_dict_tv[weight_tv_name].shape:
            conv.weight.data.copy_(state_dict_tv[weight_tv_name])
            loaded_param_names.append(weight_tv_name)
        if getattr(conv, 'bias') is not None:
            bias_tv_name = module_name_tv + '.bias'
            if conv.bias.data.shape == state_dict_tv[bias_tv_name].shape:
                conv.bias.data.copy_(state_dict_tv[bias_tv_name])
                loaded_param_names.append(bias_tv_name)

    @staticmethod
    def _load_bn_params(bn, state_dict_tv, module_name_tv,
                        loaded_param_names):
        for param_name, param in bn.named_parameters():
            param_tv_name = f'{module_name_tv}.{param_name}'
            param_tv = state_dict_tv[param_tv_name]
            if param.data.shape == param_tv.shape:
                param.data.copy_(param_tv)
                loaded_param_names.append(param_tv_name)
        for param_name, param in bn.named_buffers():
            param_tv_name = f'{module_name_tv}.{param_name}'
            if param_tv_name in state_dict_tv:
                param_tv = state_dict_tv[param_tv_name]
                if param.data.shape == param_tv.shape:
                    param.data.copy_(param_tv)
                    loaded_param_names.append(param_tv_name)

    def _load_torchvision_checkpoint(self, logger=None):
        state_dict_torchvision = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_torchvision:
            state_dict_torchvision = state_dict_torchvision['state_dict']
        loaded_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                if 'downsample' in name:
                    original_conv_name = name + '.0'
                    original_bn_name = name + '.1'
                else:
                    original_conv_name = name
                    original_bn_name = name.replace('conv', 'bn')
                self._load_conv_params(module.conv, state_dict_torchvision,
                                       original_conv_name, loaded_param_names)
                self._load_bn_params(module.bn, state_dict_torchvision,
                                     original_bn_name, loaded_param_names)
        remaining_names = set(
            state_dict_torchvision.keys()) - set(loaded_param_names)
        if remaining_names:
            logger.info(f'These parameters in pretrained checkpoint are not '
                        f'loaded: {remaining_names}')

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            if self.torchvision_pretrain:
                self._load_torchvision_checkpoint(logger)
            else:
                load_checkpoint(self, self.pretrained,
                                strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # PAME：layer1 之后
            if i == 0:
                x = self.pame(x)

            # AdaptiveMTCM：替换原 CPTM
            if i == 0:
                x = self.mtcm_res2(x)   # [N*T, 256,  H, W]
            elif i == 2:
                x = self.mtcm_res4(x)   # [N*T, 1024, H, W]

            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv1.bn.eval()
            for m in self.conv1.modules():
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _partial_bn(self):
        logger = get_root_logger()
        logger.info('Freezing BatchNorm2D except the first one.')
        count_bn = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                count_bn += 1
                if count_bn >= 2:
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        if mode and self.partial_bn:
            self._partial_bn()
