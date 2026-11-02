import torch
import torch.nn as nn

class DGAM(nn.Module):
    """Dynamic Granularity Aggregation Module (DGAM V3)
    
    采用 SK-Net 风格的动态分支融合，并修复了残差连接时的通道对齐问题。
    """
    def __init__(self, in_channels, num_segments=16):
        super(DGAM, self).__init__()

        self.num_segments = num_segments
        self.c = in_channels // 4

        # ---------------- Global Temporal Branch ----------------
        self.gtm_pool = nn.AdaptiveAvgPool2d(1)
        self.gtm_conv1 = nn.Conv1d(self.c, self.c // 2, kernel_size=3, padding=1, bias=False)
        self.gtm_conv2 = nn.Conv1d(self.c // 2, self.c, kernel_size=3, padding=1, bias=False)
        self.gtm_relu = nn.ReLU(inplace=True)
        self.gtm_act = nn.Sigmoid()

        # ---------------- Spatial Motion Branch ----------------
        self.smm_conv = nn.Sequential(
            nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.c),
            nn.ReLU(inplace=True)
        )
        self.smm_act = nn.Sigmoid()

        # ---------------- Local Temporal Branch ----------------
        self.ltm_conv = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(3,1,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(self.c),
            nn.ReLU(inplace=True)
        )
        self.ltm_act = nn.Sigmoid()

        # ---------------- Dynamic Branch Fusion ----------------
        # 增加 max 保护，防止靠前层通道数过少导致除以16后变成0
        mid_channels = max(in_channels // 16, 4) 
        self.branch_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 4, 1)
        )

        self.softmax = nn.Softmax(dim=1)

        # residual gate
        self.alpha = nn.Parameter(torch.zeros(1))

    # ---------------- Temporal Shift (TSM) ----------------
    def temporal_shift(self, x):
        nt, c, h, w = x.size()
        t = self.num_segments
        n = nt // t
        x = x.view(n, t, c, h, w)
        fold = c // 4

        # 使用 clone() 规避 inplace 报错，非常漂亮的写法
        out = x.clone()
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]

        return out.view(nt, c, h, w)

    def forward(self, x):
        nt, c, h, w = x.size()
        t = self.num_segments
        n = nt // t

        # ---------------- Channel Split ----------------
        x1, x2, x3, x4 = torch.split(x, self.c, dim=1)

        # ---------------- GTM ----------------
        g2 = self.gtm_pool(x2).view(n, t, self.c).transpose(1, 2)
        g2 = self.gtm_conv1(g2)
        g2 = self.gtm_relu(g2)
        g2 = self.gtm_conv2(g2)
        g2 = g2.transpose(1, 2).contiguous().view(nt, self.c, 1, 1)
        out2 = x2 * self.gtm_act(g2)

        # ---------------- SMM ----------------
        x3_shift = self.temporal_shift(x3)
        g3 = self.smm_conv(x3_shift)
        out3 = x3 * self.smm_act(g3)

        # ---------------- LTM ----------------
        g4 = x4.view(n, t, self.c, h, w).transpose(1, 2)
        g4 = self.ltm_conv(g4)
        g4 = g4.transpose(1, 2).contiguous().view(nt, self.c, h, w)
        out4 = x4 * self.ltm_act(g4)

        # ---------------- Dynamic Fusion ----------------
        feat = torch.cat([x1, out2, out3, out4], dim=1)

        attn = self.branch_attn(feat)
        attn = self.softmax(attn)

        w1 = attn[:, 0:1]
        w2 = attn[:, 1:2]
        w3 = attn[:, 2:3]
        w4 = attn[:, 3:4]

        # 💥 核心修正：加权后重新使用 torch.cat 拼接回原始宽度 (in_channels)
        fused_feat = torch.cat([w1 * x1, w2 * out2, w3 * out3, w4 * out4], dim=1)

        # 此时 x 的维度和 fused_feat 完全一致，残差连接成立
        return x + self.alpha * fused_feat