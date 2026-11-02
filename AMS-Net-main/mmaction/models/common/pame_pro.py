# 文件路径: /root/lanyun-fs/AMS-Net-main/mmaction/models/common/pame_pro_single.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarityAwareMotionEnhancer_Pro(nn.Module):
    """
    极性感知运动增强器 Pro 版 (PAME-Pro) —— TDN对齐版 (单向差分消融)

    核心改动：
    取消了后向差分（bwd），仅保留前向差分（fwd），用于验证双向差分带来的性能提升。
    Fusion层的输入通道数由 mid_channels * 2 减小为 mid_channels。
    """

    def __init__(self, channels=64, num_segments=16, reduction=4):
        super(PolarityAwareMotionEnhancer_Pro, self).__init__()
        self.num_segments = num_segments
        self.long_stride = 2

        mid_channels = max(channels // reduction, 16)

        # ── 可学习高斯平滑（只用于平滑未来帧，解决空间错位）──
        self.frame_smooth = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        gaussian_kernel = torch.tensor(
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]], dtype=torch.float32
        ) / 16.0
        self.frame_smooth.weight.data = gaussian_kernel.expand(
            channels, 1, 3, 3
        ).clone()

        # ── 多尺度差分特征提取 ──
        self.branch_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        self.branch_pool_conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        self.branch_direct = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )

        # ★ 改动：单向差分，输入通道数改为 mid_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.sigmoid = nn.Sigmoid()

        # alpha 负初始化
        self.raw_alpha = nn.Parameter(torch.full((1, channels, 1, 1), -4.0))

    def _multiscale_diff(self, diff):
        """三分支多尺度差分特征提取"""
        _, _, h, w = diff.size()
        b1 = self.branch_direct(diff)
        b2 = self.branch_conv(diff)
        b3 = F.interpolate(
            self.branch_pool_conv(diff),
            size=(h, w), mode='bilinear', align_corners=False
        )
        return b1 + b2 + b3

    def forward(self, x):
        nt, c, h, w = x.size()
        t = self.num_segments
        n = nt // t

        # ── 第一步：只对未来帧做高斯平滑 ──
        x_smoothed = self.frame_smooth(x).view(n, t, c, h, w)
        x_reshaped = x.view(n, t, c, h, w)

        # ── 第二步：单向差分（仅前向），步长固定为2 ──
        s = self.long_stride

        # 前向：smooth(t+2) - x(t)，最后 s 帧保持零
        fwd = torch.zeros_like(x_reshaped)
        fwd[:, :-s] = x_smoothed[:, s:] - x_reshaped[:, :-s]

        fwd = fwd.view(nt, c, h, w)

        # ★ 改动：移除后向差分相关计算

        # ── 第三步：多尺度差分特征提取 ──
        feat_fwd = self._multiscale_diff(fwd)

        # ── 第四步：单向特征直接送入 fusion 生成 attention mask ──
        motion_feats = self.fusion(feat_fwd)
        motion_mask = self.sigmoid(motion_feats)

        # ── 第五步：attention 乘法融合 ──
        safe_alpha = torch.sigmoid(self.raw_alpha)
        out = x + safe_alpha * (x * motion_mask)

        return out