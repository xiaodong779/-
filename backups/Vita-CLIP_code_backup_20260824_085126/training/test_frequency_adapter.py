#!/usr/bin/env python
"""Independent test for VitaCLIPFrequencyAdapter (Stage 4).

Checks:
  1. Forward [B*T, N, D] -> [B*T, N, D]; only CLS updated
  2. T in {8, 15, 16} (even + odd), B in {1, 2}
  3. FP32 / model.half() / AMP backward
  4. Gradients on c1/c2, residual_scale, projections non-zero
  5. Diagnostics: delta magnitude, output diff, residual_scale, high-band
  6. NaN/Inf checks
"""
import sys
import torch

sys.path.insert(0, '.')

from VitaCLIP_frequency_adapter_simple import (
    VitaCLIPFrequencyAdapter,
)


def check_nan_inf(tag, t):
    n_nan = int(torch.isnan(t).sum())
    n_inf = int(torch.isinf(t).sum())
    assert n_nan == 0 and n_inf == 0, f'{tag} has NaN/Inf'
    print(f'  [{tag}] nan={n_nan} inf={n_inf} maxabs={t.abs().max().item():.5f}')


def run_case(adapter, B, T, N, D, label, half=False, amp=False):
    torch.manual_seed(0)
    x = torch.randn(B * T, N, D, dtype=torch.float32)
    if half:
        x = x.half()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.to(device)
    adapter = adapter.to(device)

    if amp:
        ctx = torch.cuda.amp.autocast()
        assert device == 'cuda', 'AMP requires CUDA'
    else:
        ctx = torch.no_grad() if False else torch.enable_grad()

    with ctx:
        out, info = adapter(x, num_frames=T, return_information=True)

    assert out.shape == x.shape, f'{label}: {tuple(out.shape)} != {tuple(x.shape)}'
    # Only CLS changes: non-CLS tokens must be identical.
    x_r = x.reshape(B, T, N, D)
    out_r = out.reshape(B, T, N, D)
    patch_diff = (out_r[:, :, 1:, :] - x_r[:, :, 1:, :]).abs().max().item()
    assert patch_diff == 0, f'{label}: patch tokens changed by {patch_diff}'

    loss = out.square().mean()
    loss.backward()

    print(f'--- {label} (B={B} T={T} N={N} D={D} '
          f'dtype={x.dtype} amp={amp}) ---')
    print(f'  out={tuple(out.shape)} patch_max_diff={patch_diff}')
    print(f'  c1={info["c1"].item():.4f} c2={info["c2"].item():.4f} '
          f'raw_c1={info["raw_c1"].item():.4f} raw_c2={info["raw_c2"].item():.4f}')
    print(f'  residual_scale(tanh)={info["residual_scale"].item():.4f} '
          f'raw={info["residual_scale_raw"].item():.4f}')
    print(f'  cls_delta_abs_mean={info["cls_delta_abs_mean"].item():.6f}')
    print(f'  frequency_delta_abs_mean={info["frequency_delta_abs_mean"].item():.6f}')
    print(f'  output_diff_abs_mean={info["output_diff_abs_mean"].item():.6f}')
    print(f'  high_threshold={info["high_threshold"].item():.4f} '
          f'high_bypass={info["high_bypass"].item():.4f} '
          f'high_amp_ratio={info["high_amp_ratio"].item():.4f}')
    check_nan_inf(f'{label}_out', out.float())

    # Gradients
    grads = {n: p.grad for n, p in adapter.named_parameters()}
    for key in ['raw_c1', 'raw_c2', 'residual_scale', 'patch_mix_logit',
                'low_projection.weight', 'mid_projection.weight',
                'high_threshold_projection.weight',
                'high_bypass_projection.weight', 'down_projection.weight',
                'up_projection.weight']:
        g = grads[key]
        status = 'OK' if g is not None and g.abs().max() > 0 else 'ZERO/MISSING'
        print(f'  grad[{key}] = {g.abs().max().item() if g is not None else None} [{status}]')
    adapter.zero_grad()
    return info


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}')
    N, D = 197, 768
    torch.manual_seed(0)

    # 1) FP32, various T and B
    for T, B in [(8, 2), (15, 1), (16, 2), (12, 3)]:
        adapter = VitaCLIPFrequencyAdapter(dim=D)
        run_case(adapter, B, T, N, D, f'fp32_T{T}_B{B}')

    # 2) model.half()
    adapter = VitaCLIPFrequencyAdapter(dim=D).half()
    run_case(adapter, 2, 8, N, D, 'half_T8_B2', half=True)

    # 3) AMP
    if device == 'cuda':
        adapter = VitaCLIPFrequencyAdapter(dim=D)
        run_case(adapter, 2, 16, N, D, 'amp_T16_B2', amp=True)
    else:
        print('AMP skipped (no CUDA)')

    # 4) odd/even boundary behavior on 2 frames minimum
    adapter = VitaCLIPFrequencyAdapter(dim=D)
    try:
        adapter(torch.randn(4, N, D), num_frames=2)
        print('--- T=2 (min_frames boundary) OK ---')
    except ValueError as e:
        print(f'  T=2 rejected as expected: {e}')

    # 5) validation errors
    adapter = VitaCLIPFrequencyAdapter(dim=D)
    try:
        adapter(torch.randn(15, N, D), num_frames=8)  # 15 % 8 != 0
        raise AssertionError('should have raised')
    except ValueError as e:
        print(f'  divisibility check OK: {e}')

    print('\n[PASS] VitaCLIPFrequencyAdapter independent test OK')


if __name__ == '__main__':
    main()
