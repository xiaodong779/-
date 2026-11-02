import torch
import torch.nn as nn

class PolarityAwareMotionEnhancer(nn.Module):
    """
    极性感知运动增强器 (PAME - V4 安全受限版)
    保留了短时和长时的前向物理差分，引入源头平滑、安全补零。
    最重要的是：加入了对门控参数 alpha 的安全边界约束，防止特征反噬。
    """
    def __init__(self, channels=64, num_segments=16, reduction=4):
        super(PolarityAwareMotionEnhancer, self).__init__()
        self.num_segments = num_segments
        mid_channels = max(channels // reduction, 16)
        
        # 1. 源头空间平滑 (计算量减半，仅对原始通道 C 操作)
        self.spatial_smooth = nn.Conv2d(
            channels, channels, kernel_size=3, 
            padding=1, groups=channels, bias=False
        )
        nn.init.constant_(self.spatial_smooth.weight, 1.0 / 9.0)
        
        # 2. 运动特征编码器 (接收短时+长时拼接后的 2C 通道)
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(channels * 2, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True), 
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 3. 通道级残差门控的基础参数 (重命名为 raw_alpha 以示区分)
        # 初始化为0，确保训练初期的完美恒等映射
        self.raw_alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        nt, c, h, w = x.size()
        t = self.num_segments
        n = nt // t
        
        # --- 第一步：源头平滑，扼杀相机抖动噪声 ---
        x_smoothed = self.spatial_smooth(x)
        x_reshaped = x_smoothed.view(n, t, c, h, w)
        
        # --- 第二步：安全的单向物理差分 (自动补零) ---
        diff_short = torch.zeros_like(x_reshaped)
        diff_long = torch.zeros_like(x_reshaped)
        
        # 短时前向差分: t+1 减 t (最后一帧自动保持为0，代表越界)
        diff_short[:, :-1] = x_reshaped[:, 1:] - x_reshaped[:, :-1]
        
        # 长时前向差分: t+2 减 t (最后两帧自动保持为0)
        diff_long[:, :-2] = x_reshaped[:, 2:] - x_reshaped[:, :-2]
        
        diff_short = diff_short.view(nt, c, h, w)
        diff_long = diff_long.view(nt, c, h, w)
        
        # 拼接形状变为 [NT, 2C, H, W]
        diff_concat = torch.cat([diff_short, diff_long], dim=1)
        
        # --- 第三步：生成运动高光掩膜 ---
        motion_feats = self.motion_encoder(diff_concat)
        motion_mask = self.sigmoid(motion_feats) 
        
        # --- 第四步：安全的通道级恒等映射融合 ---
        # 【核心修改点】将无界的 raw_alpha 强行限制在 [0, 1] 之间
        # 保证运动信息只能被“正向增强”或“完全忽略”，杜绝负数权重的破坏
        safe_alpha = torch.clamp(self.raw_alpha, min=0.0, max=1.0)
        out = x + safe_alpha * (x * motion_mask)
        
        return out