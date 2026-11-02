import torch
import torch.nn as nn

class MGFM(nn.Module):
    """
    Multi-Granularity Fusion Module (MGFM)
    多粒度融合模块：纯净版（已彻底剥离 TSM 移位操作）。
    负责接收前端的运动特征，进行多尺度空间与局部时序的动态注意力融合。
    """
    def __init__(self, in_channels, num_segments=16):
        super(MGFM, self).__init__()
        self.num_segments = num_segments
        self.c = in_channels // 4

        # 1. GTM (Global Temporal Branch) - 全局宏观特征
        self.gtm_pool = nn.AdaptiveAvgPool2d(1)
        self.gtm_conv1 = nn.Conv1d(self.c, self.c // 2, kernel_size=3, padding=1, bias=False)
        self.gtm_conv2 = nn.Conv1d(self.c // 2, self.c, kernel_size=3, padding=1, bias=False)
        self.gtm_relu = nn.ReLU(inplace=True)
        self.gtm_act = nn.Sigmoid()

        # 2. SMM (Spatial Motion Branch) - 纯空间特征提纯
        self.smm_conv = nn.Sequential(
            nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.c),
            nn.ReLU(inplace=True)
        )
        self.smm_act = nn.Sigmoid()

        # 3. LTM (Local Temporal Branch) - 局部时序微调
        self.ltm_conv = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(3,1,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(self.c),
            nn.ReLU(inplace=True)
        )
        self.ltm_act = nn.Sigmoid()

        # 4. Dynamic Fusion (动态融合中枢)
        mid_channels = max(in_channels // 16, 4) 
        self.branch_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 4, 1)
        )
        self.softmax = nn.Softmax(dim=1)

        # 零初始化残差门控
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        nt, c, h, w = x.size()
        t = self.num_segments
        n = nt // t

        # 特征均分
        x1, x2, x3, x4 = torch.split(x, self.c, dim=1)

        # GTM
        g2 = self.gtm_pool(x2).view(n, t, self.c).transpose(1, 2)
        g2 = self.gtm_conv1(g2)
        g2 = self.gtm_relu(g2)
        g2 = self.gtm_conv2(g2)
        g2 = g2.transpose(1, 2).contiguous().view(nt, self.c, 1, 1)
        out2 = x2 * self.gtm_act(g2)

        # SMM (直接过卷积，没有 temporal_shift)
        g3 = self.smm_conv(x3)
        out3 = x3 * self.smm_act(g3)

        # LTM
        g4 = x4.view(n, t, self.c, h, w).transpose(1, 2)
        g4 = self.ltm_conv(g4)
        g4 = g4.transpose(1, 2).contiguous().view(nt, self.c, h, w)
        out4 = x4 * self.ltm_act(g4)

        # 动态打分融合
        feat = torch.cat([x1, out2, out3, out4], dim=1)
        attn = self.branch_attn(feat)
        attn = self.softmax(attn)

        w1, w2, w3, w4 = attn[:, 0:1], attn[:, 1:2], attn[:, 2:3], attn[:, 3:4]
        fused_feat = torch.cat([w1 * x1, w2 * out2, w3 * out3, w4 * out4], dim=1)

        return x + self.alpha * fused_feat