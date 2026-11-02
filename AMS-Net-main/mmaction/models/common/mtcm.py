# 文件路径: /root/lanyun-fs/AMS-Net-main/mmaction/models/common/mtcm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMTCM(nn.Module):
    """
    自适应多尺度时序上下文模块 (Adaptive Multi-scale Temporal Cue Module)
    """
    def __init__(self, channels, n_frames=8, n_branches=3, reduction=2):
        super().__init__()
        self.n_branches = n_branches
        self.n_frames = n_frames
        mid_c = channels // reduction

        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm3d(mid_c),
            nn.ReLU(inplace=True)
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(mid_c, mid_c,
                          kernel_size=(3, 1, 1),
                          padding=(i + 1, 0, 0),
                          dilation=(i + 1, 1, 1),
                          groups=mid_c, bias=False),
                nn.BatchNorm3d(mid_c),
                nn.ReLU(inplace=True)
            )
            for i in range(n_branches)
        ])

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.weight_fc = nn.Sequential(
            nn.Linear(mid_c, mid_c // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_c // 2, n_branches * mid_c, bias=False)
        )

        self.conv2 = nn.Conv3d(mid_c, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        BT, C, H, W = x.shape
        T, B = self.n_frames, BT // self.n_frames
        identity = x

        x3d = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        feats = self.conv1(x3d)
        branch_feats = [br(feats) for br in self.branches]

        context = sum(branch_feats)
        w = self.gap(context).view(B, -1)
        w = self.weight_fc(w).view(B, self.n_branches, -1)
        w = F.softmax(w, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        fused = sum(branch_feats[i] * w[:, i] for i in range(self.n_branches))
        attention = self.sigmoid(self.conv2(fused))

        attention = attention.permute(0, 2, 1, 3, 4).contiguous().view(BT, C, H, W)

        return self.gate * (attention * identity) + identity