"""显存实测：递增 batch 测 forward+backward 峰值显存，确定最大安全 videos_per_gpu。"""
import torch
import mmcv
from mmaction.models.builder import build_model

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

cfg = mmcv.Config.fromfile(
    'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py')
cfg.model.backbone.pretrained = None

model = build_model(cfg.model, train_cfg=None, test_cfg=None).cuda()
model.train()
opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                      lr=0.004, momentum=0.9)

total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'GPU total: {total_gb:.1f} GB')

for B in [2, 4, 6, 8]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    imgs = torch.randn(B, 8, 3, 1, 224, 224).cuda()
    labels = torch.randint(0, 48, (B, 1)).cuda()
    try:
        losses = model(imgs, labels, return_loss=True)
        loss = sum(v for k, v in losses.items() if 'loss' in k)
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f'B={B:2d}  peak={peak:6.2f} GB  ({peak/total_gb*100:.0f}% of GPU)  loss={float(loss):.4f}')
    except RuntimeError as e:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f'B={B:2d}  OOM at peak={peak:6.2f} GB: {str(e)[:80]}')
        break
    del imgs, labels, losses, loss
    torch.cuda.empty_cache()
