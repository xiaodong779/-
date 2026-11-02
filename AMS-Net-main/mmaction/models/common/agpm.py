import torch
import torch.nn as nn
import torch.nn.init as init

class AGPM(nn.Module):
    """Action Granularity Based Pyramid Module (AGPM)
    
    完美对齐 MMAction2 2D Backbone 的 (N*T, C, H, W) 张量流转
    """
    def __init__(self, in_channels, num_segments=8):
        super(AGPM, self).__init__()
        self.num_segments = num_segments
        # 将通道数均分为 4 份
        self.c = in_channels // 4
        
        # ================= Level 2: GTM =================
        self.gtm_pool = nn.AdaptiveAvgPool2d(1)
        self.gtm_conv = nn.Conv1d(self.c, self.c, kernel_size=3, padding=1)
        self.gtm_act = nn.Sigmoid()
        
        # ================= Level 3: SMM =================
        self.smm_conv = nn.Conv2d(self.c, self.c, kernel_size=3, padding=1)
        self.smm_act = nn.Sigmoid()
        
        # ================= Level 4: LTM =================
        self.ltm_conv = nn.Conv3d(self.c, self.c, kernel_size=(3, 1, 1), padding=(1, 0, 0)) 
        self.ltm_act = nn.Sigmoid()

    def init_weights(self):
        """极其优雅的零初始化，保证训练初期不崩"""
        # 核心卷积使用 Kaiming 正态分布初始化，保证特征不消失
        init.kaiming_normal_(self.gtm_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.smm_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.ltm_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # 将偏置(bias)强行初始化为 0
        if self.gtm_conv.bias is not None:
            init.constant_(self.gtm_conv.bias, 0)
        if self.smm_conv.bias is not None:
            init.constant_(self.smm_conv.bias, 0)
        if self.ltm_conv.bias is not None:
            init.constant_(self.ltm_conv.bias, 0)

    def forward(self, x):
        nt, c, h, w = x.size()
        t = self.num_segments
        n = nt // t 
        
        x1, x2, x3, x4 = torch.split(x, self.c, dim=1)
        
        # ================= Level 1 =================
        out1 = x1  
        
        # ================= Level 2 =================
        g2 = self.gtm_pool(x2)
        g2 = g2.view(n, t, self.c).transpose(1, 2)
        g2 = self.gtm_conv(g2) 
        g2 = g2.transpose(1, 2).contiguous().view(nt, self.c, 1, 1)
        g2_attn = self.gtm_act(g2)
        out2 = x2 + x2 * g2_attn 
        
        # ================= Level 3 =================
        x3_in = x3 + out2
        g3 = x3_in.view(n, t, self.c, h, w).mean(dim=1)
        g3 = self.smm_conv(g3)
        g3 = g3.unsqueeze(1).expand(-1, t, -1, -1, -1).contiguous().view(nt, self.c, h, w)
        g3_attn = self.smm_act(g3)
        out3 = x3_in + x3_in * g3_attn
        
        # ================= Level 4 =================
        x4_in = x4 + out3
        g4 = x4_in.view(n, t, self.c, h, w).transpose(1, 2)
        g4 = self.ltm_conv(g4)
        g4 = g4.transpose(1, 2).contiguous().view(nt, self.c, h, w)
        g4_attn = self.ltm_act(g4)
        out4 = x4_in + x4_in * g4_attn
        
        out = torch.cat([out1, out2, out3, out4], dim=1)
        
        return out