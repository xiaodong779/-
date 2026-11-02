
# 文件路径: /root/lanyun-fs/AMS-Net-main/mmaction/models/common/mtcm_pro.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMTCM_Pro(nn.Module):
    """
    自适应多尺度时序上下文模块 Pro 版
    (Adaptive Multi-scale Temporal Cue Module Pro)

    改动1：三个branch融合前加跨branch交互（Cross-branch Interaction）
           让每个branch在融合前能感知其他branch的信息
           ★ 保险：branch_interaction 加残差连接，防止训练初期不稳定

    改动2：α权重从全局静态（GAP压掉T）改为逐帧动态（保留T维度）
           让每一帧根据自身内容动态决定使用哪个尺度的branch
           ★ 保险：Conv1d 改为 kernel_size=1，完全逐帧独立，避免T较小时边界效应

    改动3：gate从标量改为通道级向量
           每个通道独立控制融合强度
    """

    def __init__(self, channels, n_frames=8, n_branches=3, reduction=2):
        super().__init__()
        self.n_branches = n_branches
        self.n_frames = n_frames
        mid_c = channels // reduction
        self.mid_c = mid_c

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

        # ★ 改动1：跨branch交互模块
        self.branch_interaction = nn.Sequential(
            nn.Conv3d(mid_c * n_branches, mid_c * n_branches,
                      kernel_size=1, bias=False,
                      groups=n_branches),
            nn.BatchNorm3d(mid_c * n_branches),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_c * n_branches, mid_c * n_branches,
                      kernel_size=1, bias=False),
            nn.BatchNorm3d(mid_c * n_branches),
            nn.ReLU(inplace=True)
        )

        # ★ 改动2：逐帧动态权重，kernel_size=1 完全逐帧独立
        self.temporal_weight = nn.Sequential(
            nn.Conv1d(mid_c, mid_c // 2,
                      kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_c // 2, n_branches,
                      kernel_size=1, bias=False)
        )

        self.conv2 = nn.Conv3d(mid_c, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # ★ 改动3：gate从标量改为通道级向量
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        BT, C, H, W = x.shape
        T, B = self.n_frames, BT // self.n_frames
        identity = x

        # BT×C×H×W → B×C×T×H×W
        x3d = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # B×C×T×H×W → B×mid_c×T×H×W
        feats = self.conv1(x3d)

        # ===== Step1：3个branch各自卷积 =====
        # 各自：B×mid_c×T×H×W
        branch_feats = [br(feats) for br in self.branches]

        # ★ 改动1：跨branch交互 + 残差
        # cat：B×3*mid_c×T×H×W
        branch_cat = torch.cat(branch_feats, dim=1)
        branch_cat_out = self.branch_interaction(branch_cat)
        # 残差：B×3*mid_c×T×H×W
        branch_cat = branch_cat + branch_cat_out
        # chunk拆回：各自 B×mid_c×T×H×W
        branch_feats = list(torch.chunk(branch_cat, self.n_branches, dim=1))

        # ★ 改动2：逐帧动态权重
        # sum：B×mid_c×T×H×W
        context = sum(branch_feats)
        # mean压H,W保留T：B×mid_c×T
        context_t = context.mean(dim=[-2, -1])
        # Conv1d：B×n_branches×T
        w = self.temporal_weight(context_t)
        # softmax branch维度：B×n_branches×T
        w = F.softmax(w, dim=1)

        # ★ 维度对齐：
        # w[:,i,:]：        B×T
        # unsqueeze(1)：    B×1×T
        # unsqueeze(-1)：   B×1×T×1
        # unsqueeze(-1)：   B×1×T×1×1
        # 广播到：          B×mid_c×T×H×W ✅
        fused = sum(
            branch_feats[i] * w[:, i, :].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            for i in range(self.n_branches)
        )
        # fused：B×mid_c×T×H×W

        # B×mid_c×T×H×W → B×C×T×H×W → sigmoid
        attention = self.sigmoid(self.conv2(fused))
        # B×C×T×H×W → B×T×C×H×W → BT×C×H×W
        attention = attention.permute(
            0, 2, 1, 3, 4).contiguous().view(BT, C, H, W)

        # ★ 改动3：通道级gate
        # gate：1×C×1×1 广播到 BT×C×H×W ✅
        return self.gate * (attention * identity) + identity

