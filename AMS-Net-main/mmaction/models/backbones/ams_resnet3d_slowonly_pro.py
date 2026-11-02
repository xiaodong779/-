#!/usr/bin/env python
"""AMSResNet3dSlowOnlyPro: 3D ResNet 主干 + 双分支专家架构。

上分支（空间专家）: Frozen CLIP ViT-B/16（冻结不反传）
下分支（时序专家）: AMSResNet3dSlowOnly 3D 主干
    每个 stage(1/2/3/4) 后:
        ADSTABlock（可变形时空注意力 + Mamba 时序门控）
        BWSTGBlock（双向窗口交叉注意力，与 ViT 空间特征交互）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ams_resnet3d_slowonly import AMSResNet3dSlowOnly
from .resnet_ams_pro import AuxFeatureAlign
from ..common.evl.frozen_evl_vit_branch import FrozenEVLViTBranch
from ..common.bw_stg import BWSTGBlock
from ..common.adsta import ADSTABlock
from ..builder import BACKBONES


@BACKBONES.register_module()
class AMSResNet3dSlowOnlyPro(AMSResNet3dSlowOnly):
    """3D SlowOnly ResNet 主干 + 双分支专家（ADSTA + BW-STG @ 每 stage）。

    架构:
        输入 [B, 3, T, 224, 224]
          ├── 上分支: Frozen EVL ViT → [B, 768, T, 14, 14]（冻结）
          └── 下分支: conv1 → maxpool
                → stage1 → ADSTA(red=8) → AuxAlign + BW-STG
                → stage2 → ADSTA(red=4) → AuxAlign + BW-STG
                → stage3 → ADSTA(red=4) → AuxAlign + BW-STG
                → stage4 → ADSTA(red=4) → AuxAlign + BW-STG
                → out_indices → AMSHead

    Args:
        继承 AMSResNet3dSlowOnly 所有参数，另增:

        num_segments (int): 时序帧数，默认 8。
            ADSTABlock 内部 reshape 时需要此值，必须与输入 T 一致。

        use_frozen_evl (bool): 是否启用上分支 Frozen EVL ViT 空间专家。
        evl_cfg (dict): FrozenEVLViTBranch 配置。
        evl_input_norm (str): EVL 输入的归一化来源，默认 'imagenet'。

        use_adsta (bool): 是否启用 ADSTA 模块。
        adsta_stages (tuple): 插入 ADSTA 的 stage 编号，(1,2,3,4) 表示全部。
        adsta_reduction (int): ADSTA 内部降维比，默认 4。s1 强制用 adsta_reduction_s1。
        adsta_reduction_s1 (int): stage1 的 ADSTA 降维比，默认 8（控制显存）。
        adsta_sample_mode (str): 采样模式，默认 'density_244'。
        adsta_mamba_expand (int): Mamba 门控扩展因子，默认 1。
        adsta_mamba_d_conv (int): Mamba 时序卷积核大小，默认 3。
        adsta_dropout (float): ADSTA dropout。

        use_stg (bool): 是否启用 BW-STG 双向引导模块。
        stg_stages (tuple): 插入 BW-STG 的 stage 编号，(1,2,3,4) 表示全部。
        stg_cfg (dict): BWSTGBlock 配置（reduction, num_heads, window_size 等）。
    """

    # 各 stage 输出通道（ResNet-50）
    STAGE_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}

    def __init__(
        self,
        # --- AMSResNet3dSlowOnly 原有参数 ---
        depth=50,
        pretrained='torchvision://resnet50',
        pretrained2d=True,
        lateral=False,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        with_pool2=False,
        out_indices=(3,),
        norm_eval=False,
        zero_init_residual=True,
        # --- 双分支新增参数 ---
        num_segments=8,
        # EVL 空间专家
        use_frozen_evl=True,
        evl_cfg=None,
        evl_input_norm='imagenet',
        # ADSTA
        use_adsta=True,
        adsta_stages=(1, 2, 3, 4),
        adsta_reduction=4,
        adsta_reduction_s1=8,
        adsta_sample_mode='density_244',
        adsta_mamba_expand=1,
        adsta_mamba_d_conv=3,
        adsta_dropout=0.0,
        # BW-STG
        use_stg=True,
        stg_stages=(1, 2, 3, 4),
        stg_cfg=None,
        **kwargs,
    ):
        super().__init__(
            depth=depth,
            pretrained=pretrained,
            pretrained2d=pretrained2d,
            lateral=lateral,
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            inflate=inflate,
            with_pool2=with_pool2,
            out_indices=out_indices,
            norm_eval=norm_eval,
            zero_init_residual=zero_init_residual,
            use_cptm=False,
            **kwargs,
        )

        self.num_segments = num_segments
        self.use_frozen_evl = use_frozen_evl
        self.evl_input_norm = evl_input_norm
        self.use_adsta = use_adsta
        self.adsta_stages = adsta_stages
        self.use_stg = use_stg
        self.stg_stages = stg_stages

        # ---- 上分支：Frozen EVL CLIP ViT（空间专家） ----
        self.frozen_evl = None
        if self.use_frozen_evl:
            if evl_cfg is None:
                evl_cfg = {}
            self.frozen_evl = FrozenEVLViTBranch(**evl_cfg)
            for p in self.frozen_evl.parameters():
                p.requires_grad = False
            self.frozen_evl.eval()

            # 5D 归一化 buffer → [1, 3, 1, 1, 1]
            self.register_buffer(
                'imagenet_mean',
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1),
            )
            self.register_buffer(
                'imagenet_std',
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1),
            )
            self.register_buffer(
                'clip_mean',
                torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1, 1),
            )
            self.register_buffer(
                'clip_std',
                torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1, 1),
            )

        # ---- ADSTA 模块（每个 stage 一个） ----
        self.adsta_layers = nn.ModuleDict()
        if self.use_adsta:
            for s in adsta_stages:
                s_str = str(s)
                in_ch = self.STAGE_CHANNELS[s]
                red = adsta_reduction_s1 if s == 1 else adsta_reduction
                self.adsta_layers[s_str] = ADSTABlock(
                    num_segments=num_segments,
                    in_channels=in_ch,
                    reduction=red,
                    stage=f'layer{s}',  # ★ 关键：每个 stage 用各自的采样设置
                    sample_mode=adsta_sample_mode,
                    mamba_expand=adsta_mamba_expand,
                    mamba_d_conv=adsta_mamba_d_conv,
                    dropout=adsta_dropout,
                )

        # ---- BW-STG 模块 + AuxFeatureAlign（每个 stage 一组） ----
        self.bw_stg_layers = nn.ModuleDict()
        self.aux_aligns = nn.ModuleDict()
        if self.use_stg:
            _stg_cfg = dict(
                reduction=4, num_heads=8, window_size=7,
                fusion_init=(0.5, 0.5), gate_init_bias=-2.0,
                lambda_init=0.01, lambda_max=0.2,
                attn_drop=0.0, proj_drop=0.0, qkv_bias=True,
            )
            if stg_cfg is not None:
                _stg_cfg.update(stg_cfg)

            for s in stg_stages:
                s_str = str(s)
                in_ch = self.STAGE_CHANNELS[s]
                self.aux_aligns[s_str] = AuxFeatureAlign(
                    in_channels=768, out_channels=in_ch,
                )
                self.bw_stg_layers[s_str] = BWSTGBlock(
                    channels=in_ch, **_stg_cfg,
                )

    # ==================== Forward ====================

    def forward(self, x):
        """主 forward: 双分支专家交互（SlowOnly 连续帧版）。

        输入来自 Recognizer3D reshape 后的 [B, C, T, H, W]，
        其中 T=8 是 clip_len=8 采样得到的连续帧窗口（3D 卷积的合法输入）。
        主干 3D 卷积 / EVL / BW-STG 全部在 T=8 上跨帧建模，
        输出保持 [B, C, T, H, W]（不做 batch 折叠）。
        AMSHead 用 num_segs=None → 内部 AdaptiveAvgPool3d 对 T,H,W 平均池化，
        输出 [B, num_classes]。

        Args:
            x: [B, C, T, H, W]，T=8 连续帧。

        Returns:
            根据 out_indices 返回单 Tensor 或 tuple（均保持 5D [B, C, T, H, W]）。
        """
        assert x.dim() == 5, \
            f'SlowOnly 连续帧输入应为 5D [B,C,T,H,W]，实际 {tuple(x.shape)}'
        raw_x = x  # [B, 3, T, H, W]

        # ---- 上分支：Frozen EVL ViT 空间专家（no_grad），T=8 ----
        evl_feat = None
        if self.use_frozen_evl and self.frozen_evl is not None:
            evl_feat = self._forward_frozen_evl(raw_x)
            # evl_feat: [B, 768, T, 14, 14]

        # ---- 下分支：3D 主干 + ADSTA + BW-STG（全部 T=8）----
        x = self.conv1(x)    # [B, 64, T, 56, 56]
        x = self.maxpool(x)  # [B, 64, T, 56, 56]

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            if i == 0 and self.with_pool2:
                x = self.pool2(x)

            stage_id = i + 1
            s_str = str(stage_id)

            # ADSTA（时序增强）
            if self.use_adsta and s_str in self.adsta_layers:
                x = self._adsta_bridge(x, self.adsta_layers[s_str])

            # BW-STG（与 ViT 空间专家双向交互）
            if self.use_stg and s_str in self.bw_stg_layers and evl_feat is not None:
                x = self._stg_interact(x, evl_feat, s_str)

            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]  # [B, C, T, H, W]
        return tuple(outs)

    # ==================== 辅助方法 ====================

    def _forward_frozen_evl(self, x):
        """上分支：Frozen EVL ViT 空间专家。

        Args:
            x: [B, 3, T, H, W] 5D 原始输入。

        Returns:
            [B, 768, T, 14, 14] ViT patch map。
        """
        x_clip = self._to_clip_norm(x)  # ImageNet → CLIP norm
        with torch.no_grad():
            evl_out = self.frozen_evl(x_clip)
        # evl_out 可能是 tuple (多输出) 或单 tensor
        if isinstance(evl_out, tuple):
            evl_out = evl_out[0]
        return evl_out

    def _to_clip_norm(self, x):
        """将 ImageNet 归一化后的输入转为 CLIP 归一化。

        Args:
            x: [B, 3, T, H, W]（值域 ≈ 0~1 的 RGB）。

        Returns:
            CLIP 归一化后的 [B, 3, T, H, W]。
        """
        # imagenet_mean/std 与 clip_mean/std 均为 [1, 3, 1, 1, 1]
        x = x * self.imagenet_std.to(x.device) + self.imagenet_mean.to(x.device)
        x = x.clamp(0.0, 1.0)
        x = (x - self.clip_mean.to(x.device)) / self.clip_std.to(x.device)
        return x

    def _adsta_bridge(self, x_5d, adsta_layer):
        """ADSTA 4D↔5D 桥接。

        ADSTABlock 接口: [B*T, C, H, W] (4D)
        3D 主干特征: [B, C, T, H, W] (5D)

        Args:
            x_5d: [B, C, T, H, W]
            adsta_layer: nn.Module (ADSTABlock 实例)

        Returns:
            [B, C, T, H, W]
        """
        B, C, T, H, W = x_5d.shape
        # SlowOnly 连续帧: 输入 [B, C, T=8, H, W]（连续 8 帧窗口）。
        # ADSTABlock 内部按 num_segments=8 把 [B*8, C, H, W] 重组回 [B, C, 8, H, W]
        # 做时序建模（可变形注意力 + Mamba 扫描），再折叠回 [B*8, C, H, W]。
        # 5D → 4D: 展平时空为 batch 交给 ADSTA。
        x_4d = x_5d.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        if not x_4d.is_contiguous():
            x_4d = x_4d.contiguous()
        x_4d = adsta_layer(x_4d)
        # 4D → 5D: 还原 [B, C, T, H, W]。
        # 手动 copy 强制标准内存布局，避免 cudnn 越界访问。
        x_5d = x_4d.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        x_5d = x_5d.new_empty(x_5d.shape).copy_(x_5d)
        return x_5d

    def _stg_interact(self, x, evl_feat, s_str):
        """BW-STG 双向交互：主干 ←→ ViT 空间特征。

        Args:
            x: [B, C, T, H, W] 当前 stage 后的主干特征。
            evl_feat: [B, 768, T, 14, 14] 冻结 ViT patch map。
            s_str: stage 编号字符串 '1'~'4'。

        Returns:
            [B, C, T, H, W] 交互后特征。
        """
        # AuxFeatureAlign 内部自动做 F.interpolate 对齐时空尺寸
        evl_aligned = self.aux_aligns[s_str](evl_feat, x, self.num_segments)
        # BWSTGBlock: y(主干), v(对齐后 ViT 特征) 均为 [B, C, T, H, W]
        x = self.bw_stg_layers[s_str](x, evl_aligned)
        return x

    # ==================== 生命周期 ====================

    def train(self, mode=True):
        """训练模式：确保 Frozen EVL ViT 始终 eval。"""
        super().train(mode)
        if self.use_frozen_evl and self.frozen_evl is not None:
            self.frozen_evl.eval()
            for p in self.frozen_evl.parameters():
                p.requires_grad = False
        return self

    def init_weights(self, pretrained=None):
        """权重初始化：父类做 2D→3D 膨胀，新增模块保持随机初始化。

        注意：父类 AMSResNet3d._init_weights 在 pretrained is None 时会遍历
        self.modules() 对所有 nn.Conv3d 做 kaiming_init、对所有 BatchNorm 做
        constant_init(1)，这会破坏 ADSTA 的 up 零初始化（用于保证模块初始≈
        恒等）和 BW-STG TemporalConvPre 的 bn2 零初始化（同样保证恒等）。
        因此这里拦截 pretrained is None 的情况：先让父类初始化主干，再重新
        对 ADSTA / BW-STG 做一次零初始化，覆盖父类的错误 kaiming 结果。
        """
        if isinstance(pretrained, str) or isinstance(self.pretrained, str):
            # 走 ImageNet 预训练 inflate 分支：父类只膨胀主干 ConvModule，
            # 不触碰 ADSTA/BW-STG 的 nn.Conv3d，零初始化保持完好，无需处理。
            super().init_weights(pretrained)
            return

        # pretrained is None（真正从零训练）：父类会 kaiming_init 所有 Conv3d，
        # 覆盖 ADSTA/BW-STG 的零初始化，需在父类之后重新零初始化这些模块。
        super().init_weights(pretrained)

        # 重新零初始化 ADSTA：up 投影置 0，保证模块初始≈恒等
        for s_str, layer in self.adsta_layers.items():
            layer.init_weights()

        # 重新零初始化 BW-STG 的 TemporalConvPre bn2，保证初始≈恒等
        for s_str, layer in self.bw_stg_layers.items():
            nn.init.zeros_(layer.temporal_conv.bn2.weight)
            nn.init.zeros_(layer.temporal_conv.bn2.bias)
