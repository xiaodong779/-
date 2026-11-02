"""Test: is optimizer construction itself the crash trigger?"""
import torch
import mmcv
from mmaction.models.builder import build_model

CFG = 'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py'


def build():
    torch.cuda.empty_cache()
    cfg = mmcv.Config.fromfile(CFG)
    cfg.model.backbone.pretrained = None
    torch.backends.cudnn.benchmark = False
    model = build_model(cfg.model, train_cfg=None, test_cfg=None).cuda().train()
    B = 2
    imgs = torch.randn(B, 8, 3, 1, 224, 224).cuda()
    labels = torch.randint(0, 48, (B, 1)).cuda()
    return model, imgs, labels


def fwd_bwd(model, imgs, labels):
    losses = model(imgs, labels, return_loss=True)
    loss = sum(v for k, v in losses.items() if 'loss' in k)
    assert torch.isfinite(loss)
    loss.backward()
    return float(loss)


print('=== L: manual step (no optimizer class) ===')
try:
    model, imgs, labels = build()
    lr = 0.004
    loss = fwd_bwd(model, imgs, labels)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-lr)
    print(f'  PASS loss={loss:.4f} mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GB')
except RuntimeError as e:
    print(f'  FAIL: {str(e).split(chr(10))[0]}')

print('=== M: empty SGD param group ===')
try:
    model, imgs, labels = build()
    opt = torch.optim.SGD([], lr=0.004, momentum=0.9)
    loss = fwd_bwd(model, imgs, labels)
    print(f'  PASS loss={loss:.4f} mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GB')
except RuntimeError as e:
    print(f'  FAIL: {str(e).split(chr(10))[0]}')

print('=== N: SGD(model.parameters()) ===')
try:
    model, imgs, labels = build()
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.004, momentum=0.9)
    loss = fwd_bwd(model, imgs, labels)
    print(f'  PASS loss={loss:.4f} mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GB')
except RuntimeError as e:
    print(f'  FAIL: {str(e).split(chr(10))[0]}')
