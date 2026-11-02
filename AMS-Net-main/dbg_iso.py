"""模块隔离：--no-adsta 禁用 ADSTA；--no-stg 禁用 BW-STG；--seed N。"""
import sys
import torch
import mmcv
from mmaction.models.builder import build_model

args = [a for a in sys.argv[1:] if a.startswith('--')]
seed = 0
for a in args:
    if a.startswith('--seed='):
        seed = int(a.split('=')[1])

no_adsta = '--no-adsta' in args
no_stg = '--no-stg' in args

torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

cfg = mmcv.Config.fromfile(
    'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py')
cfg.model.backbone.pretrained = None
if no_adsta:
    cfg.model.backbone.use_adsta = False
if no_stg:
    cfg.model.backbone.use_stg = False

model = build_model(cfg.model, train_cfg=None, test_cfg=None).cuda()
model.train()

tag = f'seed={seed} adsta={not no_adsta} stg={not no_stg}'
x5d = torch.randn(16, 3, 1, 224, 224).cuda()
try:
    out = model.backbone(x5d)
    torch.cuda.synchronize()
    o = out[0] if isinstance(out, tuple) else out
    print(f'[{tag}] OK {tuple(o.shape)} fin={torch.isfinite(o).all().item()}')
except RuntimeError as e:
    torch.cuda.synchronize()
    print(f'[{tag}] FAIL {type(e).__name__}: {e}')
