"""逐层诊断 NaN/Inf 来源。"""
import torch
import mmcv
from mmaction.models.builder import build_model

torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

cfg = mmcv.Config.fromfile(
    'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py')
cfg.model.backbone.pretrained = None

print('=== building model ===')
model = build_model(cfg.model, train_cfg=None, test_cfg=None)
model = model.cuda()
model.train()

B = 2
imgs = torch.randn(B, 8, 3, 1, 224, 224).cuda()
labels = torch.randint(0, 48, (B, 1)).cuda()


def check(tag, t):
    if t is None:
        print(f'  [{tag}] None')
        return
    if not torch.is_tensor(t):
        print(f'  [{tag}] non-tensor {type(t)}')
        return
    nan = torch.isnan(t).any().item()
    inf = torch.isinf(t).any().item()
    print(f'  [{tag}] shape={tuple(t.shape)} fin={torch.isfinite(t).all().item()} '
          f'nan={nan} inf={inf} min={t.min().item():.4f} max={t.max().item():.4f} '
          f'mean={t.mean().item():.4f} std={t.std().item():.4f}')


# ===== 手动逐步前向（复用 backbone 内部结构）=====
backbone = model.backbone
# Recognizer3D 将 [B, num_clips, C, L, H, W] reshape 为 [B*num_clips, C, L, H, W]
x5d = imgs.reshape(-1, 3, 1, 224, 224)  # [B*8, 3, 1, 224, 224]
print('=== step: full backbone forward (block by block) ===')
x = x5d
# 让 backbone 自己跑，但用 hooks 检查各 stage
import torch.nn as nn

def make_hook(tag):
    def hook(m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        check(tag, o)
    return hook

h1 = backbone.conv1.register_forward_hook(make_hook('conv1'))
h2 = backbone.maxpool.register_forward_hook(make_hook('maxpool'))
for name in backbone.res_layers:
    layer = getattr(backbone, name)
    layer.register_forward_hook(make_hook(f'res_layer {name}'))
for k, m in backbone.adsta_layers.items():
    m.register_forward_hook(make_hook(f'adsta stage{k}'))
for k, m in backbone.bw_stg_layers.items():
    m.register_forward_hook(make_hook(f'bw_stg stage{k}'))
for k, m in backbone.aux_aligns.items():
    m.register_forward_hook(make_hook(f'aux_align stage{k}'))

# EVL 分支单独检查
print('=== EVL branch ===')
evl_feat = backbone._forward_frozen_evl(x5d)
check('evl_feat', evl_feat)

print('=== main branch forward ===')
try:
    out = backbone(x5d)
    check('backbone out', out)
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')
    torch.cuda.synchronize()

print('=== with grad check: rerun with anomaly detection ===')
try:
    with torch.autograd.detect_anomaly():
        losses = model(imgs, labels, return_loss=True)
        loss = sum(v for k, v in losses.items() if 'loss' in k)
        loss.backward()
        print('  backward OK, loss=', float(loss))
except Exception as e:
    print(f'  FAILED with anomaly: {type(e).__name__}: {e}')
