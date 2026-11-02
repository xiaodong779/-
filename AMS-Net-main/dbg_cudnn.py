"""验证 cuDNN 是否是非法访问根因。--cudnn-off 禁用 cuDNN。"""
import sys
import torch
import mmcv
from mmaction.models.builder import build_model

cudnn_off = '--cudnn-off' in sys.argv
seed = 0
for a in sys.argv[1:]:
    if a.startswith('--seed='):
        seed = int(a.split('=')[1])

torch.backends.cudnn.benchmark = False
if cudnn_off:
    torch.backends.cudnn.enabled = False

torch.manual_seed(seed)
cfg = mmcv.Config.fromfile(
    'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py')
cfg.model.backbone.pretrained = None

model = build_model(cfg.model, train_cfg=None, test_cfg=None).cuda()
model.train()

tag = f'seed={seed} cudnn={"off" if cudnn_off else "on"}'
x5d = torch.randn(16, 3, 1, 224, 224).cuda()
try:
    out = model.backbone(x5d)
    torch.cuda.synchronize()
    o = out[0] if isinstance(out, tuple) else out
    print(f'[{tag}] OK {tuple(o.shape)} fin={torch.isfinite(o).all().item()}')
except RuntimeError as e:
    torch.cuda.synchronize()
    print(f'[{tag}] FAIL {type(e).__name__}: {e}')
