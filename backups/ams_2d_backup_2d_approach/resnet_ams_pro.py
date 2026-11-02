# 文件路径: /hy-tmp/AMS-Net-main/mmaction/models/backbones/resnet_ams_pro.py
# Clean STG version:
# - 主分支只保留 ResNet 2D backbone
# - 注销/移除 PAME-Pro 与 MTCM-Pro
# - 冻结 EVL CLIP ViT 分支，不反传
# - 默认只在 layer3 后插入 BWSTGBlock
# - 支持 num_segments=8 / 16，通过配置文件控制

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.utils import checkpoint as cp

from ..common.bw_stg import BWSTGBlock
from ..common.adsta import ADSTABlock
from ..common.evl.frozen_evl_vit_branch import FrozenEVLViTBranch
from ...utils import get_root_logger
from ..builder import BACKBONES


class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False, mtcm=None):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        assert not with_cp

        # mtcm 参数只保留旧接口兼容，clean 版不用
        self.mtcm = None

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

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False, mtcm=None):
        super().__init__()
        assert style in ['pytorch', 'caffe']

        # mtcm 参数只保留旧接口兼容，clean 版不用
        self.mtcm = None

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


def make_res_layer(block, inplanes, planes, blocks, stride=1, dilation=1,
                   style='pytorch', conv_cfg=None, norm_cfg=None,
                   act_cfg=None, with_cp=False, last_mtcm=None):
    """构建 ResNet stage。last_mtcm 只保留旧接口兼容，当前不用。"""
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
              act_cfg=act_cfg, with_cp=with_cp, mtcm=None))

    inplanes = planes * block.expansion

    for _ in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation,
                  style=style, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                  act_cfg=act_cfg, with_cp=with_cp, mtcm=None))

    return nn.Sequential(*layers)


class AuxFeatureAlign(nn.Module):
    """将 Frozen EVL 分支特征对齐到主分支 stage 特征。

    支持输入:
    - [B, C, T, H, W]
    - [B*T, C, H, W]
    - [B*T, N, C]，其中 N 是平方数，例如 14*14

    输出:
    - [B, out_channels, T, H, W]
    """

    def __init__(self, in_channels=768, out_channels=1024):
        super().__init__()

        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, s, target, num_segments):
        # target: [B, C, T, H, W]
        if s is None:
            return None

        _, _, t, h, w = target.shape

        if s.dim() == 5:
            aux = s

        elif s.dim() == 4:
            n, c, hs, ws = s.shape
            assert n % num_segments == 0, \
                f'EVL 4D first dim {n} cannot divide num_segments={num_segments}'
            b = n // num_segments
            aux = s.reshape(b, num_segments, c, hs, ws)
            aux = aux.permute(0, 2, 1, 3, 4).contiguous()

        elif s.dim() == 3:
            n, tokens, c = s.shape
            assert n % num_segments == 0, \
                f'EVL token first dim {n} cannot divide num_segments={num_segments}'
            side = int(math.sqrt(tokens))
            assert side * side == tokens, \
                f'EVL token number {tokens} is not square.'
            b = n // num_segments
            aux = s.transpose(1, 2).contiguous().reshape(n, c, side, side)
            aux = aux.reshape(b, num_segments, c, side, side)
            aux = aux.permute(0, 2, 1, 3, 4).contiguous()

        else:
            raise RuntimeError(f'Unsupported EVL feature dim={s.dim()}, shape={tuple(s.shape)}')

        if aux.shape[2:] != (t, h, w):
            aux = F.interpolate(
                aux,
                size=(t, h, w),
                mode='trilinear',
                align_corners=False
            )

        aux = self.proj(aux)
        aux = self.bn(aux)
        aux = self.act(aux)
        return aux


@BACKBONES.register_module()
class ResNetAMS_Pro(nn.Module):
    """ResNetAMS_Pro clean STG version.

    当前版本用于先跑通双分支框架:
    - 主分支: 纯 ResNet 2D backbone
    - 辅助分支: Frozen EVL CLIP ViT-B/16，不反传
    - STG: 默认只接 layer3
    """

    arch_settings = {
        18:  (BasicBlock,  (2, 2, 2, 2)),
        34:  (BasicBlock,  (3, 4, 6, 3)),
        50:  (Bottleneck,  (3, 4, 6, 3)),
        101: (Bottleneck,  (3, 4, 23, 3)),
        152: (Bottleneck,  (3, 8, 36, 3))
    }

    def __init__(self, depth, pretrained=None, torchvision_pretrain=True,
                 in_channels=3, num_stages=4, out_indices=(3,),
                 strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1),
                 style='pytorch', frozen_stages=-1,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False, partial_bn=False,
                 with_cp=False, num_segments=16,
                 use_frozen_evl=True,
                 use_stg=True,
                 use_stg3=True,
                 use_stg4=False,
                 evl_input_norm='imagenet',
                 evl_cfg=None,
                 stg_cfg=None,
                 use_adsta=False,
                 adsta_stages=(3,),
                 adsta_reduction=4,
                 adsta_sample_mode='density_244',
                 adsta_mamba_expand=2,
                 adsta_mamba_d_conv=3,
                 adsta_dropout=0.0):
        super().__init__()

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for ResNet')

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

        self.use_frozen_evl = use_frozen_evl
        self.use_stg = use_stg
        self.use_stg3 = use_stg3
        self.use_stg4 = use_stg4
        self.evl_input_norm = evl_input_norm

        # ADSTA 主分支时序增强模块开关。
        # adsta_stages 支持:
        #   (3,), (2, 3), (2, 3, 4), (1, 2, 3, 4)
        # 也兼容单个 int 或字符串 '2,3'。
        self.use_adsta = use_adsta
        if isinstance(adsta_stages, int):
            adsta_stages = (adsta_stages,)
        elif isinstance(adsta_stages, str):
            adsta_stages = tuple(
                int(s.strip()) for s in adsta_stages.split(',')
                if s.strip())
        else:
            adsta_stages = tuple(adsta_stages)
        self.adsta_stages = adsta_stages
        self.adsta_reduction = adsta_reduction
        self.adsta_sample_mode = adsta_sample_mode

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        self.inplanes = 64
        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2 ** i

            res_layer = make_res_layer(
                self.block, self.inplanes, planes, num_blocks,
                stride=strides[i], dilation=dilations[i],
                style=self.style, conv_cfg=conv_cfg,
                norm_cfg=norm_cfg, act_cfg=act_cfg, with_cp=with_cp,
                last_mtcm=None)

            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2 ** (
            len(self.stage_blocks) - 1)

        # 用于将 mmaction ImageNet Normalize 后的输入转回 CLIP Normalize
        self.register_buffer(
            'imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            'imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            'clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            'clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False
        )

        # 冻结 EVL CLIP ViT 分支
        if self.use_frozen_evl:
            default_evl_cfg = dict(
                backbone_name='ViT-B/16-lnpre',
                backbone_type='clip',
                backbone_path='/hy-tmp/weights/ViT-B-16.pt',
                backbone_mode='freeze_fp32',
                return_layer=-1,
            )
            if evl_cfg is not None:
                default_evl_cfg.update(evl_cfg)

            self.frozen_evl = FrozenEVLViTBranch(**default_evl_cfg)

            for p in self.frozen_evl.parameters():
                p.requires_grad = False
            self.frozen_evl.eval()
        else:
            self.frozen_evl = None

        # EVL 输出固定为 ViT-B/16 的 768 通道 patch map
        self.aux_align3 = AuxFeatureAlign(in_channels=768, out_channels=1024)
        self.aux_align4 = AuxFeatureAlign(in_channels=768, out_channels=2048)

        if stg_cfg is None:
            stg_cfg = dict(
                reduction=4,
                num_heads=8,
                window_size=7,
                fusion_init=(0.5, 0.5),
                gate_init_bias=-2.0,
                lambda_init=0.01,
                lambda_max=0.1,
                attn_drop=0.0,
                proj_drop=0.0,
                qkv_bias=True,
            )
        self.stg_cfg = stg_cfg

        if self.use_stg and self.use_stg3:
            self.bw_stg3 = BWSTGBlock(channels=1024, **self.stg_cfg)
        else:
            self.bw_stg3 = None

        if self.use_stg and self.use_stg4:
            self.bw_stg4 = BWSTGBlock(channels=2048, **self.stg_cfg)
        else:
            self.bw_stg4 = None

        # ADSTA 模块：对齐 D2ST 作者“每个 stage 后插 Adapter”的形式。
        # 这里按配置文件 adsta_stages 选择实际插入哪些 stage。
        # 对 ResNet-50: layer1/2/3/4 通道分别为 256/512/1024/2048。
        self.adsta_layers = nn.ModuleDict()
        if self.use_adsta:
            for stage_id in self.adsta_stages:
                assert 1 <= stage_id <= self.num_stages,                     f'adsta stage {stage_id} must be in [1, {self.num_stages}]'

                channels = self.block.expansion * 64 * 2 ** (stage_id - 1)

                self.adsta_layers[str(stage_id)] = ADSTABlock(
                    in_channels=channels,
                    num_segments=self.num_segments,
                    reduction=adsta_reduction,
                    stage=f'layer{stage_id}',
                    sample_mode=adsta_sample_mode,
                    mamba_expand=adsta_mamba_expand,
                    mamba_d_conv=adsta_mamba_d_conv,
                    dropout=adsta_dropout)

    def _make_stem_layer(self):
        self.conv1 = ConvModule(
            self.in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _to_5d(self, x):
        """[B*T, C, H, W] -> [B, C, T, H, W]"""
        n, c, h, w = x.shape
        t = self.num_segments
        assert n % t == 0, f'Input batch dimension {n} cannot divide num_segments={t}'
        b = n // t
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def _to_4d(self, x):
        """[B, C, T, H, W] -> [B*T, C, H, W]"""
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)

    def _to_clip_norm(self, x):
        """将主分支输入转成 CLIP normalize。

        x: [B*T, 3, 224, 224]

        默认假设 x 来自 mmaction 的 ImageNet Normalize:
            x = (raw/255 - imagenet_mean) / imagenet_std
        """
        if self.evl_input_norm == 'clip':
            return x

        if self.evl_input_norm == 'imagenet':
            x = x * self.imagenet_std + self.imagenet_mean
            x = x.clamp(0.0, 1.0)
            x = (x - self.clip_mean) / self.clip_std
            return x

        if self.evl_input_norm == 'raw':
            x = x.clamp(0.0, 1.0)
            x = (x - self.clip_mean) / self.clip_std
            return x

        raise ValueError(f'Unsupported evl_input_norm: {self.evl_input_norm}')

    @staticmethod
    def _get_from_dict(d, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    def _forward_frozen_evl(self, x):
        """运行冻结 EVL 分支。

        输入:
            x: [B*T, 3, 224, 224]

        输出:
            s3_raw, s4_raw
            当前 FrozenEVLViTBranch 只返回一个 patch map:
            [B, 768, T, 14, 14]
            所以先同时供给 layer3 / layer4。
        """
        if self.frozen_evl is None:
            return None, None

        n, c, h, w = x.shape
        t = self.num_segments

        assert c == 3
        assert n % t == 0, f'Input batch dimension {n} cannot divide num_segments={t}'
        assert h == 224 and w == 224, \
            f'Frozen EVL ViT expects 224x224 input, got {h}x{w}. ' \
            f'请确认 train/test pipeline 都是 CenterCrop 224，不要使用 ThreeCrop 256。'

        b = n // t

        x_clip = self._to_clip_norm(x)

        # [B*T, 3, H, W] -> [B, 3, T, H, W]
        x_clip = x_clip.reshape(b, t, c, h, w)
        x_clip = x_clip.permute(0, 2, 1, 3, 4).contiguous()

        with torch.no_grad():
            evl_out = self.frozen_evl(x_clip)

        if isinstance(evl_out, dict):
            s3 = self._get_from_dict(
                evl_out, ['stage3', 'layer3', 'res4', 's3', 'feat3', 'c4'])
            s4 = self._get_from_dict(
                evl_out, ['stage4', 'layer4', 'res5', 's4', 'feat4', 'c5'])
            return s3, s4

        if isinstance(evl_out, (list, tuple)):
            if len(evl_out) >= 2:
                return evl_out[-2], evl_out[-1]
            if len(evl_out) == 1:
                return evl_out[0], evl_out[0]
            return None, None

        return evl_out, evl_out

    def _apply_stg(self, stg, main_5d, aux_5d):
        """调用 BWSTGBlock。"""
        if stg is None or aux_5d is None:
            return main_5d

        out = stg(main_5d, aux_5d)

        if isinstance(out, (list, tuple)):
            out = out[0]

        return out

    @staticmethod
    def _load_conv_params(conv, state_dict_tv, module_name_tv,
                          loaded_param_names):
        weight_tv_name = module_name_tv + '.weight'
        if weight_tv_name in state_dict_tv and \
                conv.weight.data.shape == state_dict_tv[weight_tv_name].shape:
            conv.weight.data.copy_(state_dict_tv[weight_tv_name])
            loaded_param_names.append(weight_tv_name)

        if getattr(conv, 'bias') is not None:
            bias_tv_name = module_name_tv + '.bias'
            if bias_tv_name in state_dict_tv and \
                    conv.bias.data.shape == state_dict_tv[bias_tv_name].shape:
                conv.bias.data.copy_(state_dict_tv[bias_tv_name])
                loaded_param_names.append(bias_tv_name)

    @staticmethod
    def _load_bn_params(bn, state_dict_tv, module_name_tv,
                        loaded_param_names):
        for param_name, param in bn.named_parameters():
            param_tv_name = f'{module_name_tv}.{param_name}'
            if param_tv_name not in state_dict_tv:
                continue
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

                self._load_conv_params(
                    module.conv, state_dict_torchvision,
                    original_conv_name, loaded_param_names)

                if hasattr(module, 'bn') and module.bn is not None:
                    self._load_bn_params(
                        module.bn, state_dict_torchvision,
                        original_bn_name, loaded_param_names)

        remaining_names = set(
            state_dict_torchvision.keys()) - set(loaded_param_names)

        if remaining_names and logger is not None:
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
            for name, m in self.named_modules():
                # ADSTA 内部已经自己初始化；尤其 up-proj 需要保持 0 初始化，
                # 这样模块初始近似恒等映射，不能被这里的 kaiming_init 覆盖。
                if 'adsta_layers' in name:
                    continue

                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)

            if hasattr(self, 'adsta_layers'):
                for m in self.adsta_layers.values():
                    m.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        # x: [B*T, 3, 224, 224]
        raw_x = x

        s3_raw, s4_raw = None, None
        if self.use_frozen_evl and self.frozen_evl is not None:
            s3_raw, s4_raw = self._forward_frozen_evl(raw_x)

        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # 每个 ResNet stage 后按配置插入 ADSTA。
            # 顺序: ResNet stage -> ADSTA -> BW-STG。
            # 这样先增强主分支时序细节，再与冻结 EVL 分支做 STG 交互。
            stage_id = i + 1
            if self.use_adsta and str(stage_id) in self.adsta_layers:
                x = self.adsta_layers[str(stage_id)](x)

            # layer3 后接 BW-STG
            if i == 2 and self.use_stg and self.bw_stg3 is not None:
                x_5d = self._to_5d(x)
                s3 = self.aux_align3(s3_raw, x_5d, self.num_segments)
                x_5d = self._apply_stg(self.bw_stg3, x_5d, s3)
                x = self._to_4d(x_5d)

            # layer4 后接 BW-STG，默认关闭
            if i == 3 and self.use_stg and self.bw_stg4 is not None:
                x_5d = self._to_5d(x)
                s4 = self.aux_align4(s4_raw, x_5d, self.num_segments)
                x_5d = self._apply_stg(self.bw_stg4, x_5d, s4)
                x = self._to_4d(x_5d)

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

        # 冻结分支保持 eval
        if self.frozen_evl is not None:
            self.frozen_evl.eval()
            for p in self.frozen_evl.parameters():
                p.requires_grad = False

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

        if mode and self.partial_bn:
            self._partial_bn()
