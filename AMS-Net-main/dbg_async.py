"""Re-test stage combos WITHOUT CUDA_LAUNCH_BLOCKING to reveal async corruption."""
import torch
import mmcv
from mmaction.models.builder import build_model

CFG = 'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py'


def run(tag, use_stg, adsta_stages, stg_stages):
    torch.cuda.empty_cache()
    cfg = mmcv.Config.fromfile(CFG)
    cfg.model.backbone.pretrained = None
    torch.backends.cudnn.benchmark = False
    cfg.model.backbone.use_stg = use_stg
    cfg.model.backbone.adsta_stages = adsta_stages
    cfg.model.backbone.stg_stages = stg_stages
    print(f'=== {tag}: use_stg={use_stg} adsta={adsta_stages} stg={stg_stages} ===')
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
        print(f'  PASS loss={float(loss):.4f} mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GB')
    except RuntimeError as e:
        print(f'  FAIL: {str(e).split(chr(10))[0]}')


run('A: ADSTA all, no STG', use_stg=False, adsta_stages=(1, 2, 3, 4), stg_stages=())
run('E: FULL config', use_stg=True, adsta_stages=(1, 2, 3, 4), stg_stages=(1, 2, 3, 4))
run('B: stage1 combo', use_stg=True, adsta_stages=(1,), stg_stages=(1,))
run('D: BW-STG stage1 only', use_stg=True, adsta_stages=(), stg_stages=(1,))
