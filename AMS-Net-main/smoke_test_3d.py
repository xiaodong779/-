"""GPU smoke test for AMSResNet3dSlowOnlyPro (Diving48 3D training entry)."""
import torch
import mmcv
from mmaction.models.builder import build_model

cfg = mmcv.Config.fromfile(
    'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py')
# 冒烟测试不下载 pretrained
cfg.model.backbone.pretrained = None
# 与 config 中 cudnn_benchmark=True 保持一致（train.py 会据此开启）
# benchmark=False 时 cuDNN 默认算法在 ADSTA 非常规 conv shape 上触发 illegal memory access
torch.backends.cudnn.benchmark = True

print('=== building model ===')
model = build_model(cfg.model, train_cfg=None, test_cfg=None)
model = model.cuda()
model.train()

n_total = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'total params: {n_total/1e6:.1f}M, trainable: {n_trainable/1e6:.1f}M')

B = 2
# SlowOnly 连续帧采样：clip_len=8, num_clips=1
# 6D [B, num_clips, C, clip_len, H, W] → Recognizer3D reshape → [B, C, 8, H, W]
imgs = torch.randn(B, 1, 3, 8, 224, 224).cuda()
labels = torch.randint(0, 48, (B, 1)).cuda()

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.004, momentum=0.9)

print('=== forward + backward + step ===')
losses = model(imgs, labels, return_loss=True)
for k, v in losses.items():
    print(f'  {k}: {float(v)}')
loss = sum(v for k, v in losses.items() if 'loss' in k)
assert torch.isfinite(loss), 'loss is NaN/Inf'

# shape 校验：AMSHead(num_segs=None) 应输出 [B, 48]
with torch.no_grad():
    feat = model.backbone(imgs.reshape((-1,) + imgs.shape[2:]))
    print(f'  backbone out: {tuple(feat.shape)}')  # 期望 [B, 2048, 8, 7, 7]
    cls = model.cls_head(feat, num_segs=None)
    print(f'  head out: {tuple(cls.shape)}')       # 期望 [B, 48]
    assert cls.shape == (B, 48), f'head output shape wrong: {tuple(cls.shape)}'

loss.backward()
optimizer.step()

print('=== ViT freeze check ===')
n_vit = n_frozen_vit = 0
leaks = []
for name, p in model.named_parameters():
    if 'evl' in name:
        n_vit += 1
        if p.requires_grad:
            leaks.append(name)
        else:
            n_frozen_vit += 1
print(f'  ViT params: {n_vit}, frozen: {n_frozen_vit}, grad leak: {len(leaks)}')
assert not leaks, f'ViT params require grad: {leaks}'

vit_train = []
for name, m in model.named_modules():
    if 'evl' in name and any(p is not None for p in [getattr(m, 'weight', None)]):
        pass
for m in model.modules():
    mod_name = type(m).__module__
    if 'clip' in mod_name.lower() or 'vit' in mod_name.lower():
        if m.training:
            vit_train.append(type(m).__name__)
print(f'  ViT training-mode modules: {vit_train}')

print('=== grad flow check ===')
missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
print(f'  trainable params without grad: {len(missing)}')
assert not missing, f'no grad: {missing[:5]}'

print(f'  peak mem: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
print('SMOKE TEST PASSED')
