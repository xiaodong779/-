"""在 ADSTA 每个子模块输出后同步，定位第一个非法访问 kernel。"""
import torch
import mmcv
from mmaction.models.builder import build_model

torch.backends.cudnn.benchmark = False
torch.manual_seed(1)

cfg = mmcv.Config.fromfile(
    'configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py')
cfg.model.backbone.pretrained = None

model = build_model(cfg.model, train_cfg=None, test_cfg=None).cuda()
model.train()
bb = model.backbone

# 给每个 ADSTA 子模块注册 hook，同步检查
def make_hook(stage, name):
    def hook(m, inp, out):
        try:
            torch.cuda.synchronize()
            o = out[0] if isinstance(out, tuple) else out
            ok = torch.isfinite(o).all().item() if torch.is_tensor(o) else True
            print(f'  [s{stage} {name}] OK {tuple(o.shape)} fin={ok}', flush=True)
        except RuntimeError as e:
            print(f'  [s{stage} {name}] SYNC FAIL: {e}', flush=True)
            raise
    return hook

for k, ad in bb.adsta_layers.items():
    subs = {
        'down': ad.down, 'pos_embed': ad.pos_embed, 's_ln': ad.s_ln,
        's_attn': ad.s_attn, 'temporal_dwconv': ad.temporal_dwconv,
        'temporal_norm': ad.temporal_norm, 'mamba_gate': ad.mamba_gate,
        'up': ad.up,
    }
    for n, m in subs.items():
        m.register_forward_hook(make_hook(k, n))

# 再给 s_attn 内部注册
for k, ad in bb.adsta_layers.items():
    inner = {
        's_attn.conv_offset': ad.s_attn.conv_offset,
        's_attn.proj_q': ad.s_attn.proj_q,
        's_attn.proj_k': ad.s_attn.proj_k,
        's_attn.proj_v': ad.s_attn.proj_v,
        's_attn.proj_out': ad.s_attn.proj_out,
    }
    for n, m in inner.items():
        m.register_forward_hook(make_hook(k, n))

x5d = torch.randn(16, 3, 1, 224, 224).cuda()
print('start forward', flush=True)
try:
    out = bb(x5d)
    torch.cuda.synchronize()
    o = out[0] if isinstance(out, tuple) else out
    print(f'OK {tuple(o.shape)}')
except RuntimeError as e:
    torch.cuda.synchronize()
    print(f'FAIL: {type(e).__name__}: {e}')
