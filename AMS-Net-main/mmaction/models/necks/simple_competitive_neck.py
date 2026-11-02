import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS

@NECKS.register_module()
class SimpleCompetitiveNeck(nn.Module):
    def __init__(self, in_channels, num_segments=8, gamma=1.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.gamma = gamma

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        mid_channels = max(in_channels // 8, 32)
        self.score_net = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels)
        )
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_list, label=None, **kwargs):
        """
        Args:
            x_list: 来自 ResNet_DGPN2 的列表 [x1, out2, out3, out4]
        """
        if not isinstance(x_list, list):
            return x_list, dict()

        # --- 💥 核心修正 1：维度自适应探测 ---
        first_feat = x_list[0]
        feat_shape = first_feat.shape
        
        if len(feat_shape) == 5:
            # 格式为 [N, C, T, H, W]
            n, c, t, h, w = feat_shape
            nt = n * t
        elif len(feat_shape) == 4:
            # 格式为 [N*T, C, H, W]
            nt, c, h, w = feat_shape
            t = self.num_segments
            n = nt // t
        else:
            raise ValueError(f"Unexpected feature shape: {feat_shape}. Expected 4 or 5 dims.")

        num_branches = len(x_list)
        branch_features = []
        branch_scores = []
        
        for feat in x_list:
            # 统一转为 3D 格式 [N, C, T, H, W] 进行处理
            if len(feat.shape) == 4:
                feat_3d = feat.view(n, t, c, h, w).transpose(1, 2).contiguous()
            else:
                feat_3d = feat # 已经是 5D 了
            
            branch_features.append(feat_3d.unsqueeze(1)) 
            
            s = self.avg_pool(feat_3d).view(n, -1)
            branch_scores.append(self.score_net(s).unsqueeze(1)) 

        # 竞争打分
        logits = torch.cat(branch_scores, dim=1)
        weights = self.softmax(logits * self.gamma).view(n, num_branches, c, 1, 1, 1)

        # 融合
        all_features = torch.cat(branch_features, dim=1)
        fused_out = (all_features * weights).sum(dim=1) 
        
        # --- 💥 核心修正 2：还原回 2D 格式并返回元组 ---
        # Recognizer2D 期待的下一步输入通常是 [N*T, C, H, W]
        fused_out = fused_out.transpose(1, 2).contiguous().view(nt, c, h, w)
        
        # 必须返回 (特征, 损失字典)
        return fused_out, dict()