"""无 hooks 完整 backbone forward，多次运行验证稳定性。"""
import torch
import mmcv
from mmaction.models.builder import build_model

torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

cfg = mmcv.Config.fromfile(
    'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py')
cfg.model.backbone.pretrained = None

model = build_model(cfg.model, train_cfg=None, test_cfg=None).cuda()
model.train()

for trial in range(4):
    torch.manual_seed(trial)
    x5d = torch.randn(16, 3, 1, 224, 224).cuda()
    try:
        out = model.backbone(x5d)
        torch.cuda.synchronize()
        print(f'trial {trial}: OK {tuple(out.shape)} fin={torch.isfinite(out).all().item()}')
    except RuntimeError as e:
        print(f'trial {trial}: FAIL {type(e).__name__}: {e}')
        torch.cuda.synchronize()
