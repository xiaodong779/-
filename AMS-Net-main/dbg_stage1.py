"""Pin down the exact trigger: empty_cache / optimizer.step differences."""
import torch
import mmcv
from mmaction.models.builder import build_model

CFG = 'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py'


def run(tag, with_empty_cache, with_optimizer):
    if with_empty_cache:
        torch.cuda.empty_cache()
    cfg = mmcv.Config.fromfile(CFG)
    cfg.model.backbone.pretrained = None
    torch.backends.cudnn.benchmark = False
    print(f'=== {tag}: empty_cache={with_empty_cache} optimizer={with_optimizer} ===')
    model = build_model(cfg.model, train_cfg=None, test_cfg=None).cuda().train()
    B = 2
    imgs = torch.randn(B, 8, 3, 1, 224, 224).cuda()
    labels = torch.randint(0, 48, (B, 1)).cuda()
    if with_optimizer:
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.004, momentum=0.9)
    try:
        losses = model(imgs, labels, return_loss=True)
        loss = sum(v for k, v in losses.items() if 'loss' in k)
        assert torch.isfinite(loss)
        loss.backward()
        if with_optimizer:
            optimizer.step()
        mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f'  PASS loss={float(loss):.4f} mem={mem:.2f}GB')
        return True
    except RuntimeError as e:
        print(f'  FAIL: {str(e).split(chr(10))[0]}')
        return False


run('H: +opt +empty_cache', with_empty_cache=True, with_optimizer=True)
run('I: +opt no empty_cache (smoke replica)', with_empty_cache=False, with_optimizer=True)
run('J: no opt no empty_cache', with_empty_cache=False, with_optimizer=False)
run('K: no opt +empty_cache', with_empty_cache=True, with_optimizer=False)
