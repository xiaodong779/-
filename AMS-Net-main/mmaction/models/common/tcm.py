import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class TCM(nn.Module):   # Multi-scale Temporal Dynamics Module
    def __init__(self, num_segments, expansion=1, pos=2):
        super(TCM, self).__init__()
        self.num_segments = num_segments
        self.mtdm = MTDM(num_segments, expansion=expansion, pos=pos)
        self.tam = TAM(num_segments=num_segments, expansion=expansion, pos=pos) 
    
    def forward(self, x):
        out = self.mtdm(x)
        out = self.tam(out)
        return x + out


class TAM(nn.Module):   # Temporal Attention Module
    def __init__(self, num_segments, expansion=1, pos=2):
        super(TAM, self).__init__()
        self.num_segments = num_segments
        self.expansion = expansion
        self.pos = pos
        self.out_channel = 64 * (2**(self.pos-1)) * self.expansion
        self.c1 = 16
        self.c2 = 32
        self.c3 = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, self.c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1, groups=self.c1, bias=False),
            nn.BatchNorm2d(self.c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c1, self.c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.c2, self.c2, kernel_size=3, stride=1, padding=1, groups=self.c2, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c2, self.c3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c3),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.c3, self.c3, kernel_size=3, stride=1, padding=1, groups=self.c3, bias=False),
            nn.BatchNorm2d(self.c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c3, self.out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        k_size = int(math.log(num_segments, 2))
        if (k_size & 1) == 0:  # is odd number
            k_size = k_size + 1

        self.ETA = eca_layer(self.num_segments, k_size=k_size)   # efficient temporal attention

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # temporal efficient channel attention
        x = x.view((-1, self.num_segments) + x.size()[1:])        # N T C H W
        N, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # N C T H W  
        x = x.view(-1, T, H, W)                    # NC,T,H,W
        x = self.ETA(x)
        x = x.view(N, C, T, H, W).permute(0, 2, 1, 3, 4).contiguous() # N,T,C,H,W
        x = x.view(-1, C, H, W)
        return x


class MTDM(nn.Module):   # Multi-scale Temporal Dynamics Module
    def __init__(self, num_segments, expansion=1, pos=2):
        super(MTDM, self).__init__()
        patchs = [15, 15, 7, 3]
        self.patch = patchs[pos-1]
        self.patch_dilation = 1
        self.soft_argmax = nn.Softmax(dim=1)
        self.expansion = expansion
        self.num_segments = num_segments
        
        self.chnl_reduction = nn.Sequential(
            nn.Conv2d(128 * self.expansion, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.matching_layer = Matching_layer(patch=self.patch, patch_dilation=self.patch_dilation)   

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm) 
    
    def apply_binary_kernel(self, match, h, w, region):
        device = match.device
        x_line = torch.arange(w, dtype=torch.float, device=device)
        y_line = torch.arange(h, dtype=torch.float, device=device)
        x_kernel_1 = x_line.view(1, 1, 1, 1, w).expand(1, 1, w, h, w)
        y_kernel_1 = y_line.view(1, 1, 1, h, 1).expand(1, h, 1, h, w)
        x_kernel_2 = x_line.view(1, 1, w, 1, 1).expand(1, 1, w, h, w)
        y_kernel_2 = y_line.view(1, h, 1, 1, 1).expand(1, h, 1, h, w)

        ones = torch.ones(1, device=device)
        zeros = torch.zeros(1, device=device)

        eps = 1e-6
        kx = torch.where(torch.abs(x_kernel_1 - x_kernel_2) <= region, ones, zeros)
        ky = torch.where(torch.abs(y_kernel_1 - y_kernel_2) <= region, ones, zeros)
        kernel = kx * ky + eps
        kernel = kernel.view(1, h * w, h * w)                
        return match * kernel

    def apply_gaussian_kernel(self, corr, h, w, p, sigma=5):
        device = corr.device
        b, c, s = corr.size()

        x = torch.arange(p, dtype=torch.float, device=device)
        y = torch.arange(p, dtype=torch.float, device=device)

        idx = corr.max(dim=1)[1] 
        idx_y = (idx // p).view(b, 1, 1, h, w).float()
        idx_x = (idx % p).view(b, 1, 1, h, w).float()

        x = x.view(1, 1, p, 1, 1).expand(1, 1, p, h, w)
        y = y.view(1, p, 1, 1, 1).expand(1, p, 1, h, w)

        gauss_kernel = torch.exp(-((x - idx_x)**2 + (y - idx_y)**2) / (2 * sigma**2))
        gauss_kernel = gauss_kernel.view(b, p * p, h * w)

        return gauss_kernel * corr

    def match_to_flow_soft(self, match, k, h, w, temperature=1, mode='softmax'):        
        device = match.device
        b, c, s = match.size()     
        
        idx = torch.arange(h * w, dtype=torch.float32, device=device)
        idx_x = idx % w
        idx_x = idx_x.repeat(b, k, 1)
        idx_y = torch.floor(idx / w)   
        idx_y = idx_y.repeat(b, k, 1)

        soft_idx_x = idx_x[:, :1]
        soft_idx_y = idx_y[:, :1]
        displacement = (self.patch - 1) / 2
        
        topk_value, topk_idx = torch.topk(match, k, dim=1)    # (B*T-1, k, H*W)
        topk_value = topk_value.view(-1, k, h, w)
        
        match = self.apply_gaussian_kernel(match, h, w, self.patch, sigma=5)
        match = match * temperature
        match_pre = self.soft_argmax(match)
        smax = match_pre           
        smax = smax.view(b, self.patch, self.patch, h, w)
        
        x_kernel = torch.arange(-displacement * self.patch_dilation, displacement * self.patch_dilation + 1, step=self.patch_dilation, dtype=torch.float, device=device)
        y_kernel = torch.arange(-displacement * self.patch_dilation, displacement * self.patch_dilation + 1, step=self.patch_dilation, dtype=torch.float, device=device)
        x_mult = x_kernel.expand(b, self.patch).view(b, self.patch, 1, 1)
        y_mult = y_kernel.expand(b, self.patch).view(b, self.patch, 1, 1)
            
        smax_x = smax.sum(dim=1, keepdim=False) #(b,w=k,h,w)
        smax_y = smax.sum(dim=2, keepdim=False) #(b,h=k,h,w)
        flow_x = (smax_x * x_mult).sum(dim=1, keepdim=True).view(-1, 1, h * w) # (b,1,h,w)
        flow_y = (smax_y * y_mult).sum(dim=1, keepdim=True).view(-1, 1, h * w) # (b,1,h,w)    

        flow_x = (flow_x / (self.patch_dilation * displacement))
        flow_y = (flow_y / (self.patch_dilation * displacement))
            
        return flow_x, flow_y, topk_value     

    def flow_computation(self, x, pos=0, temperature=100):
        size = x.size()               
        x = x.view((-1, self.num_segments) + size[1:])        # N T C H W
        x = x.permute(0, 2, 1, 3, 4).contiguous() # B C T H W   
                        
        k = 1                
        b, c, t, h, w = x.size()            
        t = t - 1         

        if pos == 0:
            x_pre = x[:, :, 0, :].unsqueeze(dim=2).expand((b, c, t, h, w)).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        else:
            x_pre = x[:, :, :-1].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
            
        x_post = x[:, :, 1:].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)

        match = self.matching_layer(x_pre, x_post)    # (B*T-1*group, H*W, H*W)          
        u, v, confidence = self.match_to_flow_soft(match, k, h, w, temperature)
        flow = torch.cat([u, v], dim=1).view(-1, 2 * k, h, w)  #  (b, 2, h, w)  
    
        return flow, confidence

    def forward(self, x):
        # multi-scale temporal action feature
        x_redu = self.chnl_reduction(x)
        flow_1, match_v1 = self.flow_computation(x_redu, pos=1)
        flow_2, match_v2 = self.flow_computation(x_redu, pos=0)
        
        x1 = torch.cat([flow_1, match_v1], dim=1)
        x2 = torch.cat([flow_2, match_v2], dim=1)

        _, c, h, w = x1.size()
        x1 = x1.view(-1, self.num_segments - 1, c, h, w)
        x2 = x2.view(-1, self.num_segments - 1, c, h, w)

        x1 = torch.cat([x1, x1[:, -1:, :, :, :]], dim=1) ## (b,t,3,h,w)
        x2 = torch.cat([x2, x2[:, -1:, :, :, :]], dim=1) ## (b,t,3,h,w)

        out = torch.cat([x1, x2], dim=2)
        out = out.view(-1, 2 * c, h, w)
        return out


# =====================================================================
# 核心重写区：彻底剥离 C++ 编译噩梦的纯 PyTorch 相关性计算算子
# =====================================================================
class Matching_layer(nn.Module):
    def __init__(self, patch, patch_dilation):
        super(Matching_layer, self).__init__()
        self.relu = nn.ReLU()
        self.patch = patch
        self.patch_dilation = patch_dilation
        
    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h, w = feature1.size()

        displacement = (self.patch - 1) // 2
        d_dilated = displacement * self.patch_dilation
        
        # 使用 F.pad 预先补齐边界，避免循环内越界
        f2_padded = F.pad(feature2, (d_dilated, d_dilated, d_dilated, d_dilated))
        corr_list = []

        # 在局部感知野内执行动态特征平移与逐像素匹配
        for dy in range(-displacement, displacement + 1):
            for dx in range(-displacement, displacement + 1):
                # 计算切片起始坐标
                y_start = dy * self.patch_dilation + d_dilated
                x_start = dx * self.patch_dilation + d_dilated
                
                # 截取平移后的目标特征
                f2_shifted = f2_padded[:, :, y_start:y_start+h, x_start:x_start+w]
                
                # 计算内积（通道维度求和）
                c_val = (feature1 * f2_shifted).sum(dim=1, keepdim=True)
                corr_list.append(c_val)

        # 堆叠并转换为 (B, Patch*Patch, H*W) 的形状以完全对齐原算法
        corr = torch.cat(corr_list, dim=1) 
        corr = corr.view(b, self.patch * self.patch, h * w) 
        corr = self.relu(corr)
        return corr


class eca_layer(nn.Module):
    """Constructs a ECA module."""
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)