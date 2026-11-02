"""最小复现：depthwise Conv3d groups=2048 kernel=(3,1,1) 是否触发 CUDA 非法访问。"""
import torch
import torch.nn as nn

torch.manual_seed(0)

for groups in [256, 512, 1024, 2048]:
    try:
        conv = nn.Conv3d(
            groups, groups,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=groups,
            bias=False,
        ).cuda()
        x = torch.randn(2, groups, 8, 56, 56).cuda()
        y = conv(x)
        torch.cuda.synchronize()
        print(f'groups={groups}: OK  out={tuple(y.shape)}')
    except RuntimeError as e:
        print(f'groups={groups}: FAIL {e}')
        torch.cuda.synchronize()

print('--- 尝试 cudnn.enabled=False ---')
torch.backends.cudnn.enabled = False
for groups in [2048]:
    try:
        conv = nn.Conv3d(
            groups, groups,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=groups,
            bias=False,
        ).cuda()
        x = torch.randn(2, groups, 8, 56, 56).cuda()
        y = conv(x)
        torch.cuda.synchronize()
        print(f'groups={groups} cudnn off: OK  out={tuple(y.shape)}')
    except RuntimeError as e:
        print(f'groups={groups} cudnn off: FAIL {e}')

print('--- 尝试 kernel=(3,1,1) T=1 空间 (即 stage4 场景) ---')
torch.backends.cudnn.enabled = True
for shape in [(2, 2048, 1, 7, 7), (2, 2048, 8, 7, 7)]:
    try:
        conv = nn.Conv3d(
            2048, 2048,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=2048,
            bias=False,
        ).cuda()
        x = torch.randn(*shape).cuda()
        y = conv(x)
        torch.cuda.synchronize()
        print(f'shape={shape}: OK  out={tuple(y.shape)}')
    except RuntimeError as e:
        print(f'shape={shape}: FAIL {e}')
        torch.cuda.synchronize()
