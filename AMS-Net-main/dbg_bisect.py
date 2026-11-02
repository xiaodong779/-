"""二分定位：完整 forward vs 手动等价执行，找出非法访问触发点。"""
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
bb = model.backbone

print(f'frozen_evl training={bb.frozen_evl.training} '
      f'n_params={sum(1 for _ in bb.frozen_evl.parameters())} '
      f'n_buffers={sum(1 for _ in bb.frozen_evl.buffers())} '
      f'n_children={len(list(bb.frozen_evl.children()))}')


def sync_check(tag):
    try:
        torch.cuda.synchronize()
        print(f'  sync OK [{tag}]')
    except RuntimeError as e:
        print(f'  SYNC FAIL [{tag}]: {e}')
        raise


# A) 完整 forward
torch.manual_seed(1)
x5d = torch.randn(16, 3, 1, 224, 224).cuda()
try:
    out = bb(x5d)
    sync_check('A full forward')
    print(f'A) full forward OK {tuple(out.shape)}')
except RuntimeError as e:
    print(f'A) full forward FAIL: {e}')
    torch.cuda.synchronize()

# B) 手动等价执行（全部用完整 BWSTG forward）
torch.manual_seed(1)
x5d = torch.randn(16, 3, 1, 224, 224).cuda()
try:
    evl_feat = bb._forward_frozen_evl(x5d)
    sync_check('B evl')
    x = bb.conv1(x5d)
    x = bb.maxpool(x)
    for i, layer_name in enumerate(bb.res_layers):
        x = getattr(bb, layer_name)(x)
        if i == 0 and bb.with_pool2:
            x = bb.pool2(x)
        s_str = str(i + 1)
        if bb.use_adsta and s_str in bb.adsta_layers:
            x = bb._adsta_bridge(x, bb.adsta_layers[s_str])
        if bb.use_stg and s_str in bb.bw_stg_layers and evl_feat is not None:
            v = bb.aux_aligns[s_str](evl_feat, x, bb.num_segments)
            x = bb.bw_stg_layers[s_str](x, v)
        sync_check(f'B stage{s_str}')
    print(f'B) manual-equiv OK {tuple(x.shape)}')
except RuntimeError as e:
    print(f'B) manual-equiv FAIL: {e}')
    torch.cuda.synchronize()
