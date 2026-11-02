pgrep -af "python.*test_text_mstff.py"#!/usr/bin/env python
"""Independent test for TextMSTFF (Stage 3).

Checks:
  1. FP32 forward + backward, output shape [B, D]
  2. Weight gradients non-zero for all three branches (smooth constraint)
  3. Hard-min constraint comparison (gradient suppression demo)
  4. AMP forward + backward
  5. NaN/Inf checks
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from Text_MSTFF import TextMSTFF, _smooth_min


def check_nan_inf(tag, t):
    n_nan = int(torch.isnan(t).sum())
    n_inf = int(torch.isinf(t).sum())
    print(f'  [{tag}] nan={n_nan} inf={n_inf}')
    assert n_nan == 0 and n_inf == 0, f'{tag} has NaN/Inf'


def main():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, L, D = 4, 77, 512
    print(f'device={device}')

    mstff = TextMSTFF(embed_dim=D, smooth_weight_constraint=True).to(device)

    # ---------- FP32 forward/backward ----------
    token_features = torch.randn(B, L, D, device=device)
    eot_feature = torch.randn(B, D, device=device)
    fused, w = mstff(token_features, eot_feature)
    print(f'FP32 fused shape={tuple(fused.shape)} weights={w.detach().cpu().tolist()}')
    assert fused.shape == (B, D)
    assert w.shape == (3,)
    check_nan_inf('fused_fp32', fused)

    loss = fused.square().mean()
    loss.backward()
    grads = {n: p.grad for n, p in mstff.named_parameters() if p.grad is not None}
    print(f'  grad non-empty params: {len(grads)} / {sum(1 for _ in mstff.parameters())}')
    for n, g in grads.items():
        if g.abs().max() == 0:
            print(f'  WARNING: zero grad on {n}')
    assert grads['fusion_logits'].abs().max() > 0, 'fusion_logits grad is zero!'
    print(f'  fusion_logits grad absmean={grads["fusion_logits"].abs().mean().item():.6f}')

    # ---------- Smooth vs hard constraint gradient ----------
    mstff.zero_grad()
    # Force weights near cap by pushing logits large.
    with torch.no_grad():
        mstff.fusion_logits.copy_(torch.tensor([5.0, 5.0, 5.0]))
    fused, w = mstff(token_features, eot_feature)
    print(f'  capped-case weights={w.detach().cpu().tolist()}')
    fused.square().mean().backward()
    g_smooth = mstff.fusion_logits.grad.clone()
    print(f'  smooth grad={g_smooth.detach().cpu().tolist()}')

    mstff_hard = TextMSTFF(embed_dim=D, smooth_weight_constraint=False).to(device)
    with torch.no_grad():
        mstff_hard.fusion_logits.copy_(torch.tensor([5.0, 5.0, 5.0]))
    fused_h, w_h = mstff_hard(token_features, eot_feature)
    fused_h.square().mean().backward()
    g_hard = mstff_hard.fusion_logits.grad.clone()
    print(f'  hard  grad={g_hard.detach().cpu().tolist()}')
    print(f'  smooth grad magnitude > hard grad magnitude: '
          f'{g_smooth.abs().sum() > g_hard.abs().sum()}')

    # ---------- AMP (autocast) ----------
    if device == 'cuda':
        mstff.zero_grad()
        mstff.to(torch.float32)
        with torch.cuda.amp.autocast():
            fused_amp, w_amp = mstff(token_features, eot_feature)
            print(f'AMP fused dtype={fused_amp.dtype} shape={tuple(fused_amp.shape)}')
            fused_amp.square().mean().backward()
        check_nan_inf('fused_amp', fused_amp.float())
    else:
        print('AMP skipped (no CUDA)')

    # ---------- Model-wide dtype (model.half) ----------
    mstff_half = TextMSTFF(embed_dim=D).half().to(device)
    fused_half, w_half = mstff_half(token_features.half(), eot_feature.half())
    print(f'half fused dtype={fused_half.dtype} shape={tuple(fused_half.shape)}')
    assert fused_half.dtype == torch.float16
    check_nan_inf('fused_half', fused_half.float())

    # ---------- _smooth_min sanity ----------
    x = torch.tensor([0.1, 0.9, 1.5])
    cap = torch.tensor(0.6)
    print(f'_smooth_min([0.1,0.9,1.5], 0.6) = {_smooth_min(x, cap).tolist()} '
          f'(expect ~[0.1, 0.6, 0.6])')

    # ---------- shape validation (variable L is OK; bad D / bad eot must raise) ----------
    fused_varL, _ = mstff(token_features[:, :L-1, :], eot_feature)
    assert fused_varL.shape == (B, D), 'variable-L input failed'
    print(f'  variable-L input OK: {tuple(fused_varL.shape)}')
    try:
        mstff(token_features, eot_feature[:, :D-1])
        raise AssertionError('should have raised')
    except ValueError as e:
        print(f'  shape check OK: {str(e)[:60]}...')

    print('\n[PASS] Text_MSTFF independent test OK')


if __name__ == '__main__':
    main()
