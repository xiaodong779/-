"""Quick reliability check: run FULL config fwd+bwd 4x in one process."""
import torch
import mmcv
from mmaction.models.builder import build_model

CFG = 'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py'

ok = 0
for i in range(4):
    torch.cuda.empty_cache()
    cfg = mmcv.Config.fromfile(CFG)
    cfg.model.backbone.pretrained = None
    torch.backends.cudnn.benchmark = False
    try:
        model = build_model(cfg.model, train_cfg=None, test_cfg=None).cuda().train()
        B = 2
        imgs = torch.randn(B, 8, 3, 1, 224, 224).cuda()
        labels = torch.randint(0, 48, (B, 1)).cuda()
        losses = model(imgs, labels, return_loss=True)
        loss = sum(v for k, v in losses.items() if 'loss' in k)
        assert torch.isfinite(loss)
        loss.backward()
        torch.cuda.synchronize()
        ok += 1
        print(f'iter {i}: PASS loss={float(loss):.4f}')
    except RuntimeError as e:
        print(f'iter {i}: FAIL: {str(e).split(chr(10))[0]}')
        break
print(f'SUMMARY: {ok}/4 passed')
