"""测试 cudnn benchmark / deterministic / cublas 开关组合。用法: --benchmark --deterministic --cublas-off --seed=N"""
import sys
import torch
import mmcv
from mmaction.models.builder import build_model

args = sys.argv[1:]
benchmark = '--benchmark' in args
deterministic = '--deterministic' in args
cublas_off = '--cublas-off' in args
seed = 0
for a in args:
    if a.startswith('--seed='):
        seed = int(a.split('=')[1])

torch.backends.cudnn.benchmark = benchmark
torch.backends.cudnn.deterministic = deterministic
if cublas_off:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(seed)
cfg = mmcv.Config.fromfile(
    'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py')
cfg.model.backbone.pretrained = None

model = build_model(cfg.model, train_cfg=None, test_cfg=None).cuda()
model.train()

tag = f'seed={seed} bench={benchmark} det={deterministic}'
x5d = torch.randn(16, 3, 1, 224, 224).cuda()
try:
    out = model.backbone(x5d)
    torch.cuda.synchronize()
    o = out[0] if isinstance(out, tuple) else out
    print(f'[{tag}] OK {tuple(o.shape)} fin={torch.isfinite(o).all().item()}')
except RuntimeError as e:
    torch.cuda.synchronize()
    print(f'[{tag}] FAIL {type(e).__name__}: {e}')
