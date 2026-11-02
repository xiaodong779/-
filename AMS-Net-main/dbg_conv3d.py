"""最小复现：dw_tconv 同构 Conv3d 是否崩。"""
import torch

m = torch.nn.Conv3d(256, 256, (3, 1, 1), groups=256, padding=(1, 0, 0)).cuda()
x = torch.randn(16, 256, 1, 56, 56).cuda()
print('input', tuple(x.shape), 'contiguous', x.is_contiguous())
out = m(x)
torch.cuda.synchronize()
print('forward ok', tuple(out.shape))
out.sum().backward()
torch.cuda.synchronize()
print('backward ok')

# 非 contiguous 输入
x2 = torch.randn(16, 256, 2, 56, 56).cuda()[:, :, :1]
print('x2', tuple(x2.shape), 'contiguous', x2.is_contiguous())
out2 = m(x2)
torch.cuda.synchronize()
print('non-contig forward ok', tuple(out2.shape))
