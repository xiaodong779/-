"""单步执行 stage4 路径，精确定位崩溃子模块。"""
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

B = 2
x5d = torch.randn(B * 8, 3, 1, 224, 224).cuda()

backbone = model.backbone


def try_step(tag, fn, *args):
    try:
        r = fn(*args)
        torch.cuda.synchronize()
        if torch.is_tensor(r):
            print(f'OK   {tag}: {tuple(r.shape)} contig={r.is_contiguous()} '
                  f'fin={torch.isfinite(r).all().item()}')
        else:
            print(f'OK   {tag}: {type(r)}')
        return r
    except RuntimeError as e:
        print(f'FAIL {tag}: {type(e).__name__}: {e}')
        torch.cuda.synchronize()
        return None


evl_feat = backbone._forward_frozen_evl(x5d)
print(f'evl_feat {tuple(evl_feat.shape)}')

x = backbone.conv1(x5d)
x = backbone.maxpool(x)
for i, layer_name in enumerate(backbone.res_layers):
    res_layer = getattr(backbone, layer_name)
    x = res_layer(x)
    if i == 0 and backbone.with_pool2:
        x = backbone.pool2(x)
    s_str = str(i + 1)
    if backbone.use_adsta and s_str in backbone.adsta_layers:
        x = backbone._adsta_bridge(x, backbone.adsta_layers[s_str])
    if backbone.use_stg and s_str in backbone.bw_stg_layers and evl_feat is not None:
        evl_aligned = backbone.aux_aligns[s_str](evl_feat, x, backbone.num_segments)
        print(f'pre-BWSTG stage{s_str}: x={tuple(x.shape)} contig={x.is_contiguous()} '
              f'stride={x.stride()} ptr={x.data_ptr()} | '
              f'v={tuple(evl_aligned.shape)} contig={evl_aligned.is_contiguous()} '
              f'stride={evl_aligned.stride()} ptr={evl_aligned.data_ptr()}')
        if s_str == '4':
            blk = backbone.bw_stg_layers[s_str]
            y_t = try_step('temporal_conv', blk.temporal_conv, x)
            y_b = try_step('down_y', blk.down_y, y_t)
            v_b = try_step('down_v', blk.down_v, evl_aligned)
            e_v2y = try_step('v_to_y', blk.v_to_y, y_b, v_b)
            e_y2v = try_step('y_to_v', blk.y_to_v, v_b, y_b)
            w = blk.get_fusion_weight()
            f = try_step('fusion', lambda a, b: w[0] * a + w[1] * b, e_v2y, e_y2v)
            gate_in = try_step('cat', torch.cat, [y_b, f, y_b * f], 1)
            gate = try_step('gate_net', blk.gate_net, gate_in)
            z = try_step('gate*', lambda g, yb: g * yb, gate, y_b)
            r = try_step('spatial_conv', blk.spatial_conv, z)
            r = try_step('up', blk.up, r)
            lam = blk.get_lambda()
            out = try_step('residual', lambda y, rr: y + lam * rr, x, r)
            print(f'stage4 BW-STG done: {None if out is None else tuple(out.shape)}')
        else:
            x = backbone._stg_interact(x, evl_feat, s_str)
            torch.cuda.synchronize()
            print(f'stage{s_str} BW-STG OK: {tuple(x.shape)}')

print('ALL DONE')
