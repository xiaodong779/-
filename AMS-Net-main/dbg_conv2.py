"""针对 stage4 精确 shape 的最小复现 + 非连续输入测试。"""
import torch
import torch.nn as nn

torch.manual_seed(0)
conv = nn.Conv3d(2048, 2048, kernel_size=(3, 1, 1), padding=(1, 0, 0),
                 groups=2048, bias=False).cuda()

# 1. 标准连续输入
for shape in [(16, 2048, 1, 7, 7), (2, 2048, 1, 7, 7), (16, 2048, 1, 56, 56)]:
    try:
        x = torch.randn(*shape).cuda()
        y = conv(x)
        torch.cuda.synchronize()
        print(f'shape={shape} contig: OK {tuple(y.shape)}')
    except RuntimeError as e:
        print(f'shape={shape} contig: FAIL {e}')
        torch.cuda.synchronize()

# 2. 模拟 ADSTA bridge 后的输入（view+permute+contiguous+clone）
try:
    B, C, T, H, W = 16, 2048, 1, 7, 7
    x5d = torch.randn(B, C, T, H, W).cuda()
    x4d = x5d.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
    x4d = x4d * 1.0  # 模拟 adsta 内操作
    x5d2 = x4d.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous().clone()
    y = conv(x5d2)
    torch.cuda.synchronize()
    print(f'bridge path: OK {tuple(y.shape)} is_contig={x5d2.is_contiguous()}')
except RuntimeError as e:
    print(f'bridge path: FAIL {e}')
    torch.cuda.synchronize()

# 3. 输入来自另一个 conv 输出（与模型更接近）
try:
    proj = nn.Conv3d(2048, 2048, kernel_size=1, bias=False).cuda()
    bn = nn.BatchNorm3d(2048).cuda()
    x = torch.randn(16, 2048, 1, 7, 7).cuda()
    x = proj(x)
    x = bn(x)
    y = conv(x)
    torch.cuda.synchronize()
    print(f'conv-out input: OK {tuple(y.shape)}')
except RuntimeError as e:
    print(f'conv-out input: FAIL {e}')
    torch.cuda.synchronize()
